"""Ordered repair proposals with conservative, exact speculation certificates.

Workers own only CPU proposal state. Original geometry queries and commits
stay in the parent. The initial certificate covers the entire pose domain,
including a possible full-domain restoration retry; an accepted change thus
invalidates lookahead unless every dependency is still byte-identical.
"""

from contextlib import closing
from copy import copy
import multiprocessing
import os
import pickle
import sys
import threading
import time
import traceback
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from ..cpu_budget import available_cpus
from .repair_candidates import candidate_steps
from .repair_frames import halo_frames, restrict
from .repair_frames import group_frames as group_frames

MAX_WORKERS = 4
MIN_PARALLEL_FRAMES = 100
# Large programs try ordinary fractions before frame-local restoration.
LAZY_MIN_VARIABLES = 4096
# Retain only the frame-local restoration attempt; on those clips the
# full-domain retry changed no final pose.


def transition_group(limits, current, group):
    """Established anchor policy, shared by real and anticipated transitions."""
    if limits.movable_body is not None and group != limits.movable_body:
        limits.kin.evaluate(current, False)
        for index, (points, bodies, budget) in enumerate(limits.point_groups()):
            acquired = limits.correction_mask(bodies, budget)
            limits.anchors[index][acquired] = points[acquired]
            error = np.linalg.norm(points - limits.targets[index], axis=-1)
            limits.target_budgets[index][acquired] = np.minimum(
                limits.target_budgets[index][acquired], error[acquired] + limits.contact_tolerances[index])
    limits.movable_body = group


def mutable_state(limits):
    return dict(movable_body=limits.movable_body,
                frame_mask=None if limits.frame_mask is None else limits.frame_mask.copy(),
                anchors=[a.copy() for a in limits.anchors],
                target_budgets=[a.copy() for a in limits.target_budgets])


def certificate(limits, current, measured, protected, fixed_q, entry_depth, witness_state=()):
    """Exact bytes, not a hash or a body-name approximation of dependencies."""
    state = mutable_state(limits)
    return (current.dtype.str, current.shape, current.tobytes(),
            pickle.dumps((state, measured, protected, fixed_q, entry_depth, witness_state), protocol=5),
            limits.entry.tobytes(), limits.penetration_yield.tobytes(),
            limits.objective.reference.tobytes(), limits.lower.tobytes(), limits.upper.tobytes(),
            tuple(a.tobytes() for a in limits.tracking_budgets))


def _guards(limits):
    from .proposals import guards_of
    return guards_of(limits.objective)[1]


def _candidate_iterator(limits, geometry, current, measured, protected, fixed_q, depth, step, record):
    from .mesh_repair import _joint_projection, restore_repair, DEEP_RESIDUAL_M

    def restore(candidate):
        projection = _joint_projection(limits, geometry.model, measured)
        mask = limits.frame_mask
        result = restore_repair(limits, current, candidate, geometry, measured, protected,
                                restrict(projection, mask))
        if result[0] is None and mask is not None and not mask.all():
            result[1]["full_domain_retry_skipped_experiment"] = True
        return result

    large = (restrict(_joint_projection(limits, geometry.model, measured),
                      limits.frame_mask).shape[1] >= LAZY_MIN_VARIABLES)
    return candidate_steps(limits, current, step, restore, record,
                           lazy=large,
                           fixed_q=fixed_q,
                           restore_below=.5 if depth <= DEEP_RESIDUAL_M and not large else None,
                           parallel_restore=not large)


class LocalProposal:
    def __init__(self, limits, geometry, current, measured, protected, fixed_q, depth):
        from .mesh_repair import _qp_step
        self.limits, self.current = limits, current
        self.args = limits, geometry, current, measured, protected, fixed_q, depth
        prepared = getattr(limits, '_repair_gate_cache', None)
        self.prediction, self.fast_prediction = (gate_predictions(limits, current, geometry,
                                                                measured, protected, depth)
                                                 if prepared is None else prepared)
        if self.prediction is not None:
            if self.prediction["reason"] == "no_local_dofs":
                self.step, self.status, self.stationary = None, "no_local_dofs", False
                return
            # A zero step is only a header carrier: the existing gate consumes
            # this proposal before candidates/restoration can run.
            self.step = np.zeros((limits.frames, limits.width))
            self.status = "pre_qp_certified_small_progress"
            self.stationary = False
            return
        from . import repair_fast_gate
        if self.fast_prediction is not None:
            if repair_fast_gate.should_skip(self.fast_prediction):
                self.prediction = {**self.fast_prediction, 'reason': 'reduced_predicted_small_progress',
                                   'reduced_gate': self.fast_prediction}
                self.step = np.zeros((limits.frames, limits.width))
                self.status, self.stationary = 'reduced_predicted_small_progress', False
                return
        self.step, self.status = _qp_step(limits, current, geometry, measured, protected)
        self.stationary = self.step is not None and np.max(np.abs(self.step), initial=0.) < 1e-9
        from . import repair_prediction
        self.prediction = None
        if self.step is not None and (repair_prediction.PREDICTIVE_GATING or repair_prediction.RECORD_PREDICTIONS):
            self.prediction = repair_prediction.linear_prediction(
                geometry, measured, self.step, limits.movable_body, depth)
            if self.prediction is not None and self.fast_prediction is not None:
                self.prediction['reduced_gate'] = self.fast_prediction

    @property
    def has_direction(self):
        return self.step is not None

    def candidates(self, record):
        return _candidate_iterator(*self.args, self.step, record)

    def close(self):
        pass


def gate_predictions(limits, current, geometry, measured, protected, depth):
    """The existing two gates, also reused before parallel dispatch."""
    from .repair_triage import certificate
    from . import repair_fast_gate
    prediction = certificate(limits, current, geometry, measured, depth)
    fast = None
    if prediction is None:
        fast = repair_fast_gate.predict(limits, current, geometry, measured, protected, depth)
    return prediction, fast


class GeometryRows:
    """Intentionally no scene, moving query, GPU client or measurement API."""
    def __init__(self, geometry):
        self.model, self.width = geometry.model, geometry.width
        self.objective = SimpleNamespace(frames=geometry.objective.frames)

    def linear_rows(self, measurement):
        from .mesh_repair import OriginalGeometry
        return OriginalGeometry.linear_rows(self, measurement)


def _serve(limits, geometry, connection, inherited):
    for other in inherited:
        other.close()
    service_module = sys.modules.get(__package__ + ".gpu_geometry")
    service = None if service_module is None else service_module._SERVICE
    if service is not None and service.pid != os.getpid():
        service.connection.close()
        service_module._SERVICE = None
    limits.objective.pool, limits.objective.worker_output = None, None
    limits.objective.evaluation_backend = "cpu"
    guards = _guards(limits)
    with threadpool_limits(limits=1):
        while True:
            message = connection.recv()
            if message is None:
                break
            state, current, measured, protected, fixed_q, depth = message
            for name, value in state.items():
                setattr(limits, name, value)
            previous = [getattr(g, "rejections", 0) for g in guards]
            record = {}
            cpu_started = time.process_time()

            def send(kind, value):
                now = [getattr(g, "rejections", 0) for g in guards]
                deltas = [a - b for a, b in zip(now, previous)]
                previous[:] = now
                connection.send((kind, value, record, deltas, time.process_time() - cpu_started))
            try:
                job = LocalProposal(limits, geometry, current, measured, protected, fixed_q, depth)
                send("header", (job.has_direction, job.stationary, job.status, job.prediction))
                if not job.has_direction or job.stationary:
                    continue
                from .repair_prediction import should_gate
                from .mesh_repair import MIN_GROUP_PROGRESS_M
                if should_gate(job.prediction, MIN_GROUP_PROGRESS_M):
                    continue
                with closing(job.candidates(record)) as candidates:
                    for alpha, candidate in candidates:
                        send("candidate", (alpha, candidate))
                        command = connection.recv()
                        if command == "cancel":
                            break
                        if command != "next":
                            raise RuntimeError("Invalid repair proposal command")
                    else:
                        send("done", None)
            except BaseException:
                send("error", traceback.format_exc())
    connection.close()


class RemoteProposal:
    def __init__(self, pool, slot, dependency, current):
        self.pool, self.slot, self.dependency, self.current = pool, slot, dependency, current
        self.header, self.waiting, self.done = None, False, False
        self.cpu_seconds, self.consumed, self.accounted = 0., False, False

    def receive(self):
        tick = time.monotonic()
        packet = self.slot[1].recv()
        self.pool.stats["wait_seconds"] += time.monotonic() - tick
        self.cpu_seconds = packet[-1]
        if packet[0] == "error":
            raise RuntimeError(f"Repair proposal worker failed:\n{packet[1]}")
        return packet[:-1]

    def start(self):
        if self.header is None:
            kind, self.header, _, deltas = self.receive()
            assert kind == "header"
            self._counters(deltas)
            if not self.has_direction or self.stationary:
                self.done = True
            from .repair_prediction import should_gate
            from .mesh_repair import MIN_GROUP_PROGRESS_M
            if should_gate(self.prediction, MIN_GROUP_PROGRESS_M):
                self.done = True
        return self

    def _counters(self, deltas):
        for guard, delta in zip(_guards(self.pool.limits), deltas):
            if delta:
                guard.rejections += delta

    @property
    def has_direction(self):
        return self.header[0]

    @property
    def stationary(self):
        return self.header[1]

    @property
    def status(self):
        return self.header[2]

    @property
    def prediction(self):
        return self.header[3]

    def candidates(self, record):
        try:
            while not self.done:
                if self.waiting:
                    self.slot[1].send("next")
                    self.waiting = False
                kind, value, diagnostics, deltas = self.receive()
                record.update(diagnostics)
                self._counters(deltas)
                if kind == "done":
                    self.done = True
                    break
                assert kind == "candidate"
                self.waiting = True
                alpha, candidate = value
                # Full-domain certificate was checked before consumption.
                # Install only the scalar coordinates the proposal changed.
                changed = candidate != self.current
                rebased = self.current.copy()
                rebased[changed] = candidate[changed]
                guards = _guards(self.pool.limits)
                counts = [getattr(g, "rejections", None) for g in guards]
                try:
                    valid, _ = self.pool.limits.check(rebased)
                finally:
                    for guard, count in zip(guards, counts):
                        if count is not None:
                            guard.rejections = count
                if not valid:
                    raise RuntimeError("Certified repair proposal failed parent nonlinear limits")
                yield alpha, rebased
        finally:
            self.close()

    def close(self):
        try:
            if self.done:
                return
            if self.header is None:
                kind, self.header, _, _ = self.receive()  # Discarded diagnostics stay private.
                assert kind == "header"
                if not self.has_direction or self.stationary:
                    self.done = True
                    return
                from .repair_prediction import should_gate
                from .mesh_repair import MIN_GROUP_PROGRESS_M
                if should_gate(self.prediction, MIN_GROUP_PROGRESS_M):
                    self.done = True
                    return
            if not self.waiting:
                kind, _, _, _ = self.receive()
                if kind == "done":
                    self.done = True
                    return
                assert kind == "candidate"
            self.slot[1].send("cancel")
            self.done = True
        finally:
            if self.done and not self.accounted:
                self.accounted = True
                name = "useful_cpu_seconds" if self.consumed else "discarded_cpu_seconds"
                self.pool.stats[name] += self.cpu_seconds
                self.pool.stats["discarded"] += not self.consumed


class RepairProposals:
    def __init__(self, limits, geometry, *, count=None):
        self.limits, self.geometry = limits, geometry
        self.workers, self.pending = [], {}
        self.automatic, self.stable_attempts = count is None, 0
        self.stats = dict(launched=0, reused=0, stale=0, local=0, discarded=0,
                          useful_cpu_seconds=0., discarded_cpu_seconds=0.,
                          wait_seconds=0., disabled_reason=None)
        self.stats["pre_qp_prefetch_skips"] = 0
        self.stats["aborted_stale"] = 0
        self.stats["reduced_prefetch_skips"] = 0
        self._prefetch_triage = False
        parallel = getattr(geometry, "parallel", None)
        if count is None:
            occupied = len(getattr(parallel, "workers", ()))
            count = (min(MAX_WORKERS, max(0, available_cpus() - occupied - 1))
                     if limits.frames >= MIN_PARALLEL_FRAMES else 0)
        if (not hasattr(os, "fork") or multiprocessing.current_process().daemon
                or threading.active_count() != 1
                or (parallel is not None and (parallel.prefetched or parallel.static_sender is not None))):
            count = 0
            self.stats["disabled_reason"] = "requires idle CPU-only proposal state"
        from .reference import ReferenceShape
        from .contact_guard import ContactGuard
        from .acquisition_guard import AcquisitionNormalGuard
        from .continuity import CorrectionContinuity
        from .guard_signals import CachedSequenceContinuityGuard
        from .._sequence.continuity_guard import SequenceContinuityGuard
        supported = (ReferenceShape, ContactGuard, AcquisitionNormalGuard,
                     CorrectionContinuity, SequenceContinuityGuard, CachedSequenceContinuityGuard)
        if any(type(guard) not in supported for guard in _guards(limits)):
            count = 0
            self.stats["disabled_reason"] = "custom guard has no CPU-only proposal contract"
        self.capacity = count
        self.stats.update(worker_count=0, available_workers=count,
                          activation="two consecutive unchanged attempts" if self.automatic else "explicit")
        if not self.automatic:
            self._start_workers()

    def note_outcome(self, changed):
        self.stable_attempts = 0 if changed else self.stable_attempts + 1
        if changed:
            # Every pending certificate covers the complete old pose bytes.
            # An accepted change proves all are stale. They own CPU-only
            # state, so do not wait for QP/restoration headers to discard them.
            for job in self.pending.values():
                process, connection = job.slot
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
                connection.close()
                self.workers.remove(job.slot)
                self.stats["stale"] += 1
                self.stats["discarded"] += 1
                # Received CPU time is a lower bound for aborted jobs.
                self.stats["discarded_cpu_seconds"] += job.cpu_seconds
                self.stats["aborted_stale"] += 1
            self.pending.clear()

    def _start_workers(self):
        limits, geometry = self.limits, self.geometry
        count = self.capacity - len(self.workers)
        parallel = getattr(geometry, "parallel", None)
        if (threading.active_count() != 1 or (parallel is not None
                and (parallel.prefetched or parallel.static_sender is not None))):
            self.stats["disabled_reason"] = "requires idle CPU-only proposal state"
            return
        self.stats["worker_count"] = len(self.workers) + count
        inherited_geometry = [c for _, c in getattr(parallel, "workers", ())]
        context = multiprocessing.get_context("fork") if count else None
        try:
            with threadpool_limits(limits=1):
                for _ in range(count):
                    parent, child = context.Pipe()
                    inherited = [parent, *inherited_geometry, *(c for _, c in self.workers)]
                    process = context.Process(target=_serve, daemon=True,
                                              args=(limits, GeometryRows(geometry), child, inherited))
                    process.start()
                    child.close()
                    self.workers.append((process, parent))
        except BaseException:
            self.close(abort=True)
            raise

    def prefetch(self, schedule, current, measured, protected):
        if not self.capacity or self.stats["disabled_reason"]:
            return
        # Full-pose certificates cannot survive a successful preceding group.
        # Require observed unchanged attempts before paying for lookahead;
        # this decision depends only on accepted outcomes, never wall time.
        if self.automatic and self.stable_attempts < 2:
            return
        # Outcome counts, never wall time, decide whether to stop lookahead.
        if self.stats["stale"] >= 2 and self.stats["reused"] == 0:
            self.stats["disabled_reason"] = "two stale certificates without useful reuse"
            return
        if not self.workers or len(self.workers) < self.capacity:
            self._start_workers()
        if not self.workers:
            return
        limits = copy(self.limits)
        limits.anchors = [a.copy() for a in limits.anchors]
        limits.target_budgets = [a.copy() for a in limits.target_budgets]
        from .mesh_repair import _activation_depth, scope_for_body
        for group in schedule.peek_first(MAX_WORKERS):
            transition_group(limits, current, group)
            model = self.geometry.model
            depths = [d for key, d in measured["depths"].items()
                      if group is None or any(model.body_weldid[model.geom_bodyid[g]] == group
                                              for g in (key[2:] if key[1] == "self" else key[-1:]))]
            depth = max(depths, default=0.)
            if depth <= _activation_depth() or group in self.pending:
                continue
            busy = {id(job.slot) for job in self.pending.values()}
            slot = next((s for s in self.workers if id(s) not in busy), None)
            if slot is None:
                break
            from .repair_defects import frames as attempt_frames
            limits.frame_mask = attempt_frames(limits, measured, model, group,
                                               padding=halo_frames(limits.objective.framerate))
            if self._prefetch_triage:
                from .repair_triage import certificate as progress_certificate
                if progress_certificate(limits, current, self.geometry, measured, depth) is not None:
                    # The real attempt checks again after its actual group
                    # transition. Avoid sending a full witness audit to a
                    # worker solely to return this inexpensive rejection.
                    self.stats["pre_qp_prefetch_skips"] += 1
                    continue
            from .repair_fast_gate import predict, should_skip
            if should_skip(predict(limits, current, self.geometry, measured, protected, depth)):
                # A dispatch decision only. The actual attempt repeats
                # prediction after its actual anchor/group transition.
                self.stats["reduced_prefetch_skips"] += 1
                continue
            scope = scope_for_body(limits, group)
            fixed_q = None
            if scope is not None:
                fixed_q = np.ones(current.shape[1], dtype=bool)
                fixed_q[limits.coords.qi[sorted(scope.columns)]] = False
            dependency = certificate(limits, current, measured, protected, fixed_q, depth)
            slot[1].send((mutable_state(limits), current, measured, protected, fixed_q, depth))
            self.pending[group] = RemoteProposal(self, slot, dependency, current)
            self.stats["launched"] += 1

    def take(self, current, measured, protected, fixed_q, depth):
        group = self.limits.movable_body
        job = self.pending.pop(group, None)
        if job is not None:
            dependency = certificate(self.limits, current, measured, protected, fixed_q, depth)
            if dependency == job.dependency:
                self.stats["reused"] += 1
                job.consumed = True
                job = job.start()
            else:
                self.stats["stale"] += 1
                job.close()
                job = None
        if job is None:
            self.stats["local"] += 1
            job = LocalProposal(self.limits, self.geometry, current, measured, protected, fixed_q, depth)
        if (job.prediction is not None
                and job.prediction.get("model") == "pre_qp_upper_bound"):
            # Clips with no successful certificate pay no extra lookahead
            # checks. Activation uses evidence, never timings or body names.
            self._prefetch_triage = True
        return job

    def close(self, *, abort=False):
        for job in self.pending.values():
            if not abort:
                job.close()
        self.pending.clear()
        for process, connection in self.workers:
            if abort and process.is_alive():
                process.terminate()
            elif process.is_alive():
                connection.send(None)
            connection.close()
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
        self.workers.clear()
