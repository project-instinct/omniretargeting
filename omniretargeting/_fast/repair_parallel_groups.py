"""Persistent CPU proposals for disjoint groups; commits remain ordered.

CPU workers never own geometry query connections. A scoped exact certificate
permits reuse after unrelated commits; changed dependencies are solved again.
The parent applies the complete nonlinear limits and original geometry checks.
"""

from copy import copy
import multiprocessing
import os
import threading

import numpy as np
from threadpoolctl import threadpool_limits

from ..cpu_budget import available_cpus
from . import repair_proposals as proposals
from .repair_frames import group_frames, halo_frames
from .repair_parallel_scope import certificate, footprint
from .runtime_options import OPTIONS

MAX_WORKERS = 8


class ParallelProposal(proposals.RemoteProposal):
    def __init__(self, pool, slot, dependency, current, scope):
        super().__init__(pool, slot, dependency, current)
        self.scope = scope
        self.buffer, self.paused = [], False

    def receive(self):
        return self.buffer.pop(0) if self.buffer else super().receive()

    def pause(self):
        """Finish the first candidate before an overlapping scheduled solve.

        The existing worker then waits for next/cancel, consuming no CPU. Its
        proposal is retained; changed dependency bytes, rather than potential
        overlap, decide whether it must be solved again when consumed.
        """
        if self.paused:
            return
        header = super().receive()
        self.buffer.append(header)
        assert header[0] == 'header'
        direction, stationary, status, prediction = header[1]
        from .repair_prediction import should_gate
        from .mesh_repair import MIN_GROUP_PROGRESS_M
        if direction and not stationary and not should_gate(prediction, MIN_GROUP_PROGRESS_M):
            self.buffer.append(super().receive())
        self.paused = True

    def candidates(self, record):
        try:
            while not self.done:
                if self.waiting:
                    self.slot[1].send('next')
                    self.waiting = False
                kind, value, diagnostics, deltas = self.receive()
                record.update(diagnostics)
                self._counters(deltas)
                if kind == 'done':
                    self.done = True
                    break
                assert kind == 'candidate'
                self.waiting = True
                alpha, candidate = value
                qi = self.pool.limits.coords.qi
                # Restoration is restricted to exactly this projection too.
                original = self.original
                changed = candidate != original
                allowed = np.zeros_like(changed)
                allowed[:, qi] = self.scope.write
                if np.any(changed & ~allowed):
                    raise RuntimeError('Repair proposal wrote outside its certified footprint')
                rebased = self.current.copy()
                rebased[changed] = candidate[changed]
                guards = proposals._guards(self.pool.limits)
                counts = [getattr(guard, 'rejections', None) for guard in guards]
                try:
                    valid, _ = self.pool.limits.check(rebased)
                finally:
                    for guard, count in zip(guards, counts):
                        if count is not None:
                            guard.rejections = count
                if not valid:
                    raise RuntimeError('Certified repair proposal failed complete parent limits')
                yield alpha, rebased
        finally:
            self.close()


class ParallelGroups:
    def __init__(self, limits, geometry, *, count=None):
        self.limits, self.geometry = limits, geometry
        self.workers, self.pending = [], {}
        self.active = None
        self.gates = None
        self.stats = dict(launched=0, reused=0, stale=0, re_solved=0, local=0, discarded=0,
                          useful_cpu_seconds=0., discarded_cpu_seconds=0., wait_seconds=0.,
                          disabled_reason=None, worker_count=0, max_concurrent=0,
                          re_gated=0,
                          prefetch_gate_skips=0,
                          activation='disjoint footprints; immediate persistent fork',
                          footprint='all bodies at movable frames plus third-difference context')
        if count is None:
            count = min(MAX_WORKERS, max(0, available_cpus()-1))
        if not 0 <= count <= MAX_WORKERS:
            raise ValueError('Repair group workers must be between zero and eight')
        reason = self._unsupported()
        if reason:
            count = 0
            self.stats['disabled_reason'] = reason
        self.capacity = count
        self.stats['available_workers'] = count

    def _unsupported(self):
        if (not hasattr(os, 'fork') or multiprocessing.current_process().daemon
                or threading.active_count() != 1):
            return 'requires a single-threaded direct-fork parent'
        if OPTIONS.witness_memory or OPTIONS.repair_aggregate or not proposals.LOCAL_RESTORATION_ONLY:
            return 'global witness/aggregate/restoration contract requires serial repair'
        if (getattr(self.limits.objective, 'priority_clearance', None) is not None
                or getattr(self.limits.objective, 'original_clearance', None) is not None):
            return 'private clearance provider requires serial repair'
        from .reference import ReferenceShape
        from .contact_guard import ContactGuard
        from .acquisition_guard import AcquisitionNormalGuard
        from .continuity import CorrectionContinuity
        from .guard_signals import CachedSequenceContinuityGuard
        from .._sequence.continuity_guard import SequenceContinuityGuard
        known = (ReferenceShape, ContactGuard, AcquisitionNormalGuard, CorrectionContinuity,
                 SequenceContinuityGuard, CachedSequenceContinuityGuard)
        if any(type(guard) not in known for guard in proposals._guards(self.limits)):
            return 'custom guard has no frame-local dependency contract'
        return None

    def _start_workers(self, needed):
        parallel = getattr(self.geometry, 'parallel', None)
        if (threading.active_count() != 1 or (parallel is not None
                and (parallel.prefetched or parallel.static_sender is not None))):
            self.stats['disabled_reason'] = 'requires idle geometry IPC at fork'
            return
        context = multiprocessing.get_context('fork')
        inherited_geometry = [connection for _, connection in getattr(parallel, 'workers', ())]
        try:
            with threadpool_limits(limits=1):
                while len(self.workers) < min(needed, self.capacity):
                    parent, child = context.Pipe()
                    inherited = [parent, *inherited_geometry, *(c for _, c in self.workers)]
                    process = context.Process(target=proposals._serve, daemon=True,
                                              args=(self.limits, proposals.GeometryRows(self.geometry),
                                                    child, inherited))
                    try:
                        process.start()
                    finally:
                        child.close()
                    self.workers.append((process, parent))
        except BaseException:
            self.close(abort=True)
            raise
        self.stats['worker_count'] = max(self.stats['worker_count'], len(self.workers))

    def _discard(self, job):
        # A changed dependency must not hold commits hostage to an obsolete
        # conic solve/restoration. Only this CPU slot is replaced; others persist.
        process, connection = job.slot
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join()
        connection.close()
        self.workers.remove(job.slot)
        self.stats['stale'] += 1
        self.stats['discarded'] += 1
        self.stats['discarded_cpu_seconds'] += job.cpu_seconds

    def prefetch(self, schedule, current, measured, protected):
        if not self.capacity or self.stats['disabled_reason']:
            return
        head = footprint(self.limits, self.geometry.model, measured)
        for group, job in list(self.pending.items()):
            if group != self.limits.movable_body and not head.disjoint(job.scope):
                job.pause()
        self.lookahead = schedule, current, measured, protected
        model, group = self.geometry.model, self.limits.movable_body
        depth = max((d for key, d in measured['depths'].items()
                     if group is None or any(model.body_weldid[model.geom_bodyid[g]] == group
                                             for g in (key[2:] if key[1] == 'self' else key[-1:]))),
                    default=0.)
        self.gates = proposals.gate_predictions(self.limits, current, self.geometry,
                                                measured, protected, depth)
        if not self._gated(self.gates):
            self._dispatch(*self.lookahead)

    @staticmethod
    def _gated(gates):
        from .repair_fast_gate import should_skip
        return gates[0] is not None or (OPTIONS.repair_fast_gate and should_skip(gates[1]))

    def _dispatch(self, schedule, current, measured, protected):
        head = footprint(self.limits, self.geometry.model, measured)
        # Validate stored work before choosing another disjoint batch. Do not
        # anticipate retry attempts; they remain in the established schedule.
        limits = copy(self.limits)
        limits.anchors = [a.copy() for a in self.limits.anchors]
        limits.target_budgets = [a.copy() for a in self.limits.target_budgets]
        from .mesh_repair import _activation_depth, scope_for_body
        occupied = [head]
        for group in schedule.peek_first(len(schedule.ready)):
            proposals.transition_group(limits, current, group)
            model = self.geometry.model
            depth = max((d for key, d in measured['depths'].items()
                         if group is None or any(model.body_weldid[model.geom_bodyid[g]] == group
                                                 for g in (key[2:] if key[1] == 'self' else key[-1:]))),
                        default=0.)
            from .repair_defects import frames as attempt_frames, has_defect
            limits.frame_mask = attempt_frames(limits, measured, model, group,
                                               padding=halo_frames(limits.objective.framerate))
            scope = footprint(limits, model, measured)
            limb = scope_for_body(limits, group)
            fixed = None
            if limb is not None:
                fixed = np.ones(current.shape[1], bool)
                fixed[limits.coords.qi[sorted(limb.columns)]] = False
            old = self.pending.get(group)
            dependency = None
            if old is not None:
                dependency = certificate(limits, current, measured, protected, fixed, depth, scope)
                if old.dependency != dependency:
                    del self.pending[group]
                    self._discard(old)
                    self.stats['re_solved'] += 1
                    old = None
                else:
                    if not old.paused:
                        occupied.append(scope)
                    continue
            if depth <= _activation_depth() or not scope.write.any():
                continue
            if OPTIONS.repair_defect_windows and not has_defect(measured, model, group):
                continue
            if not all(scope.disjoint(other) for other in occupied):
                continue
            gates = proposals.gate_predictions(limits, current, self.geometry, measured, protected, depth)
            if self._gated(gates):
                self.stats['prefetch_gate_skips'] += 1
                continue
            busy = {id(job.slot) for job in self.pending.values()}
            if self.active is not None and not self.active.done:
                busy.add(id(self.active.slot))
            slot = next((slot for slot in self.workers if id(slot) not in busy), None)
            if slot is None and len(self.workers) < self.capacity:
                self._start_workers(len(self.workers)+1)
                slot = next((slot for slot in self.workers if id(slot) not in busy), None)
            if slot is None:
                break
            if dependency is None:
                dependency = certificate(limits, current, measured, protected, fixed, depth, scope)
            slot[1].send((proposals.mutable_state(limits), current, measured, protected, fixed, depth))
            job = ParallelProposal(self, slot, dependency, current, scope)
            job.original = current
            self.pending[group] = job
            occupied.append(scope)
            self.stats['launched'] += 1
            self.stats['max_concurrent'] = max(self.stats['max_concurrent'],
                                                sum(not job.paused for job in self.pending.values())+1)

    def take(self, current, measured, protected, fixed_q, depth):
        job = self._take(current, measured, protected, fixed_q, depth)
        self.active = job if isinstance(job, ParallelProposal) else None
        return job

    def _local(self, current, measured, protected, fixed_q, depth):
        self.stats['local'] += 1
        gates = getattr(self, 'gates', None)
        if gates is not None:
            self.limits._repair_gate_cache = gates
        try:
            return proposals.LocalProposal(self.limits, self.geometry, current,
                                           measured, protected, fixed_q, depth)
        finally:
            self.limits.__dict__.pop('_repair_gate_cache', None)
            self.gates = None

    def _take(self, current, measured, protected, fixed_q, depth):
        group = self.limits.movable_body
        job = self.pending.pop(group, None)
        if job is not None and self.gates is not None and self._gated(self.gates):
            job.close()
            self.stats['re_gated'] += 1
            return self._local(current, measured, protected, fixed_q, depth)
        if job is not None:
            scope = footprint(self.limits, self.geometry.model, measured)
            dependency = certificate(self.limits, current, measured, protected, fixed_q, depth, scope)
            if dependency == job.dependency:
                self.stats['reused'] += 1
                job.current = current
                job.consumed = True
                job.start()
                from .repair_prediction import should_gate
                from .mesh_repair import MIN_GROUP_PROGRESS_M
                if (should_gate(job.prediction, MIN_GROUP_PROGRESS_M)
                        and not np.array_equal(current, job.original)):
                    # Interval tightening in triage/fast-gate can propagate
                    # farther than a QP's third-difference stencil, including
                    # global infeasibility fallback. Recheck skipped headers
                    # on the actual trajectory after any unrelated commit.
                    job.close()
                    self.stats['re_gated'] += 1
                    return self._local(current, measured, protected, fixed_q, depth)
                self.gates = None
                return job
            self._discard(job)
            self.stats['re_solved'] += 1
        return self._local(current, measured, protected, fixed_q, depth)

    def note_outcome(self, changed):
        pass  # Scoped certificates, rather than whole-pose changes, decide reuse.

    def close(self, *, abort=False):
        # Pending jobs have never been committed; terminating CPU-only work is
        # safe and avoids paying for abandoned restoration during final checks.
        for process, connection in self.workers:
            if process.is_alive():
                process.terminate()
            connection.close()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        self.stats['discarded'] += len(self.pending)
        self.pending.clear()
        self.workers.clear()
