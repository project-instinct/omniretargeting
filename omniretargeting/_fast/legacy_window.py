"""Opt-in overlapping acquisition using the old objective and fixed guards.

Persistent CPU children solve complete local LM excursions. The parent owns
the trajectory, commits whole windows, and checks every whole-trajectory guard
after each sweep. Neither contacts nor reference budgets are reconstructed.
"""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import multiprocessing
from multiprocessing.connection import wait
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

from ..cpu_budget import available_cpus
from .legacy_window_qp import commit_verdict, grid_depths, guard_failures, solve_local
from .report_values import report_value
from .proposals import guards_of
from .solver import CONVERGED_DECREASE, INITIAL_DAMPING
from .window_driver import fork_status
from .window_schedule import dependency_window, independent_batches

CORE_FRAMES = 64
HALO_FRAMES = 15
SWEEP_ITERATIONS = (6, 4)
# Scheduling budgets, not acceptance thresholds. Timing is diagnostic only.
# Twelve workers on an 8P+8E CPU slowed every window iteration by ~20% with
# no shorter batches (a colour rarely holds more than eight windows).
MAX_WORKERS = 8
MIN_CORE_FRAMES = 32
MAX_EXTRA_SWEEPS = 2
EXTRA_ITERATIONS = 4
# 957-frame TRUMANS: 128 -> 64 window iterations of continuation cut acquisition
# 272 -> 207 s and total 506 -> 396 s; static/self penetration, jitter and
# tracking improved, contact-normal residual +2.0 mm, proximity gap +0.4 mm.
MAX_EXTRA_WINDOW_ITERATIONS = 64
# Shorter clips keep the whole-trajectory optimizer: its cost grows roughly
# with frames squared (iterations and per-iteration work), the window
# schedule's stays near constant. Measured: 168 frames 67 s whole vs 440 s
# windows; 957 frames 834 s whole vs 291 s windows after exact CPU/dispatch
# optimizations (G1, i7-13790F). sqrt(291 / 9.2e-4) is about 563 frames;
# retain 600 conservatively until a paired shorter-clip benchmark supports
# lowering it. Contact density and continuation work vary between clips.
MIN_WINDOW_FRAMES = 600


def windows(count, sweep, *, core=CORE_FRAMES, halo=HALO_FRAMES):
    """Shift interfaces on sweep two; keep short tails in their adjacent core."""
    if count <= 0 or core <= 0 or halo < 0:
        raise ValueError('Invalid legacy window dimensions')
    if count <= core:
        return [range(count)]
    shift = (sweep % 2) * (core // 2)
    edges = sorted({0, count, *range(shift or core, count, core)})
    if len(edges) > 2 and edges[-1] - edges[-2] < core // 2:
        del edges[-2]
    return [range(max(0, a - halo), min(count, b + halo)) for a, b in zip(edges, edges[1:])]


def unsupported_reason(objective, poses=None, *, min_frames=0):
    """Why windows cannot replace this acquisition (the caller keeps the
    whole-trajectory optimizer), or None."""
    if min_frames and objective.frames < min_frames:
        return f'clip shorter than {min_frames} frames: whole-trajectory optimizer is faster'
    for guard in guards_of(objective)[1]:
        if not all(callable(getattr(guard, name, None)) for name in
                   ('qp_bounds', 'accepts', 'window_dependencies')):
            return f'No exact legacy window interface for {type(guard).__name__}'
    if poses is not None:
        failures = _failures(objective, poses)
        if failures:
            return f'legacy_windows requires a guard-feasible global entry: {failures}'
    return None


def _validate(objective):
    reason = unsupported_reason(objective)
    if reason is not None:
        raise ValueError(reason)


_failures = guard_failures


def _private_worker(objective):
    """Never use inherited objective pools, device evaluators or geometry IPC."""
    objective.pool, objective.worker_output = None, None
    objective.evaluation_backend = 'cpu'
    objective.__dict__.pop('_window_views', None)
    objective.__dict__.pop('_gpu_frames', None)
    self_cache = getattr(objective, '_window_self_cache', None)
    if self_cache is not None:
        objective._window_self_cache = self_cache.fork_private_client()
    # A frozen depth provider is normally never queried during acquisition.
    # If present, still detach its inherited query resources explicitly.
    provider = None
    if provider is not None:
        from copy import copy
        private = copy(provider)
        private.geometry = provider.geometry.fork_private_client()
        objective.frozen_original_depth = private


def _serve(objective, connection, inherited):
    for other in inherited:
        other.close()
    with threadpool_limits(limits=1):
        _private_worker(objective)
        while True:
            task = connection.recv()
            if task is None:
                break
            try:
                permits = getattr(objective, '_window_restoration_permits', None)
                budget = getattr(objective, '_window_cpu_budget', None)
                lease = budget.window() if budget is not None else permits
                with lease if lease is not None else nullcontext():
                    result = solve_local(objective, *task)
                connection.send(('ok', result))
            except BaseException:
                connection.send(('error', traceback.format_exc()))
    connection.close()


class WindowWorkers:
    """Persistent direct-fork workers; no Pool helper threads or nested forks."""
    def __init__(self, objective, count):
        self.objective, self.jobs = objective, []
        context = multiprocessing.get_context('fork') if count else None
        try:
            for _ in range(count):
                parent, child = context.Pipe()
                inherited = [parent, *(connection for _, connection in self.jobs)]
                process = context.Process(target=_serve, args=(objective, child, inherited),
                                          daemon=getattr(objective, '_window_restoration_permits', None) is None)
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                self.jobs.append((process, parent))
        except BaseException:
            self.close(abort=True)
            raise

    def solve(self, poses, batch, iterations, convergence):
        """Refill each idle child immediately; results retain dispatch order.

        Every task in a color reads the same snapshot. Completion timing may
        choose an idle worker, but never changes tasks, quotas or commit order.
        """
        started = time.monotonic()
        results = [None] * len(batch)
        if not self.jobs:
            for index, window in enumerate(batch):
                queued = time.monotonic() - started
                value = solve_local(self.objective, poses, window.frames, iterations, convergence)
                value.update(queue_seconds=queued, completion_seconds=time.monotonic() - started)
                results[index] = value
            return results
        active, next_index = {}, 0

        def dispatch(connection):
            nonlocal next_index
            if next_index == len(batch):
                return
            index = next_index
            next_index += 1
            queued = time.monotonic() - started
            connection.send((poses, batch[index].frames, iterations, convergence))
            active[connection] = index, queued

        for _, connection in self.jobs:
            dispatch(connection)
        while active:
            ready = wait(list(active))
            for connection in ready:
                index, queued = active.pop(connection)
                try:
                    status, value = connection.recv()
                except EOFError as error:
                    raise RuntimeError(f'Legacy window child exited for {batch[index].frames}') from error
                if status != 'ok':
                    raise RuntimeError(f'Legacy window child failed for {batch[index].frames}:\n{value}')
                value.update(queue_seconds=queued, completion_seconds=time.monotonic() - started)
                results[index] = value
                dispatch(connection)
        return results

    def close(self, *, abort=False):
        for process, connection in self.jobs:
            if abort and process.is_alive():
                process.terminate()
            elif process.is_alive():
                try:
                    connection.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for process, connection in self.jobs:
            connection.close()
            process.join()
        self.jobs.clear()


@contextmanager
def _workers(objective, count):
    service, evaluation_service, workers = None, None, None
    try:
        from .window_restoration_parallel import WINDOW_PARALLEL_RESTORATIONS
        from .window_cpu_budget import WindowCpuBudget, native_workers
        if count:
            if native_workers() > 1:
                objective._window_cpu_budget = WindowCpuBudget(count)
            if WINDOW_PARALLEL_RESTORATIONS:
                budget = getattr(objective, '_window_cpu_budget', None)
                objective._window_restoration_permits = (budget.permits if budget is not None else
                    multiprocessing.get_context('fork').BoundedSemaphore(count))
        workers = WindowWorkers(objective, count)
        yield workers
    except BaseException:
        if workers is not None:
            workers.close(abort=True)
        raise
    finally:
        if workers is not None:
            workers.close()
        objective.__dict__.pop('_window_restoration_permits', None)
        objective.__dict__.pop('_window_cpu_budget', None)
        if service is not None:
            objective.__dict__.pop('_window_shell_client').close()
            for local, _, _ in getattr(objective, '_window_views', {}).values():
                local.__dict__.pop('_window_shell_client', None)
            service.close()
        if evaluation_service is not None:
            objective.__dict__.pop('_window_eval_client').close()
            evaluation_service.close()
            objective.window_evaluation_service_diagnostics = evaluation_service.diagnostics


def _commit(objective, current, snapshot, window, result):
    """Check exact dependencies and actual decrease; do not add a net SQP ratio."""
    started = time.monotonic()
    frames, footprint = window.frames, window.footprint
    span = slice(frames.start, frames.stop)
    context = slice(footprint.start, footprint.stop)
    if current[context].tobytes() != snapshot[context].tobytes():
        raise RuntimeError('Legacy window dependency changed before commit')
    part = np.asarray(result['poses'])
    if part.shape != current[span].shape or not np.isfinite(part).all():
        raise ValueError('Invalid legacy window pose payload')
    candidate = current.copy()
    candidate[span] = part
    certified = result.get('commit')
    # All mutable objective/guard packs stay frozen throughout this driver.
    # Same-color edits cannot change any certified row. This fallback retains
    # the old parent computation for tests and non-worker callers.
    details = dict(certified if certified is not None else
                   commit_verdict(objective, current, candidate, frames))
    details.update(commit_evaluated_by='child' if certified is not None else 'parent',
                   parent_seconds=time.monotonic() - started)
    return (candidate if details['accepted'] else current), details


@dataclass
class WindowTask:
    core: range
    window: object
    reasons: tuple
    retry_count: int = 0
    retry_of: tuple = ()
    halo: int = HALO_FRAMES
    iteration_quota: int | None = None

    @property
    def frames(self):
        return self.window.frames

    @property
    def footprint(self):
        return self.window.footprint

    @property
    def key(self):
        return self.core.start, self.core.stop


@dataclass
class WindowState:
    task: WindowTask
    record: dict
    stamp: np.ndarray


def _task(core, count, components, reasons, *, halo=HALO_FRAMES, shift=0, retries=()):
    first, stop = core.start - halo + shift, core.stop + halo + shift
    retry_count = 0
    if retries:
        base_halo = halo
        halo, retry_count = 2 * base_halo, 1
        first, stop = core.start - halo, core.stop + halo
        for state in retries:
            # Move both old interfaces outward; the entire rejected window
            # remains movable even when the next core grid is shifted.
            first = min(first, state.task.frames.start - base_halo)
            stop = max(stop, state.task.frames.stop + base_halo)
    frames = range(max(0, first), min(count, stop))
    return WindowTask(core, dependency_window(frames, count, components), tuple(reasons),
                      retry_count, tuple(s.record['iteration'] for s in retries), halo)


def expected_work(frames, previous, quota):
    """Frame-iteration work, weighted by previous local iteration demand.

    The additive unit gives unseen/early-stopped regions nonzero weight.
    Neither this proxy nor any scheduling decision reads measured timings.
    """
    weighted, covered = 0., 0
    for row in previous:
        a, b = row['frames']
        overlap = max(0, min(b, frames.stop) - max(a, frames.start))
        if overlap:
            weighted += overlap * row['local_iterations']
            covered += overlap
    return len(frames) * quota * (1 + (weighted / covered if covered else 0.))


def base_plan(count, sweep, components, previous, quota, *, core=None, halo=None, minimum=None):
    """Split high-work cores deterministically; retain halos/crossing rows."""
    cores = windows(count, sweep, halo=0, **({} if core is None else {'core': core}))
    padding = HALO_FRAMES if halo is None else halo
    minimum = MIN_CORE_FRAMES if minimum is None else minimum
    task_kwargs = {} if halo is None else {'halo': halo}
    rows = [state.record for state in previous]
    estimates = [expected_work(range(max(0, c.start - padding),
                                     min(count, c.stop + padding)), rows, quota) for c in cores]
    target = float(np.median(estimates)) if rows else 0.
    plan = []
    for core, estimate in zip(cores, estimates):
        retries = [state for state in previous
                   if state.record['local_stop_reason'] == 'rejected_iteration'
                   and state.task.retry_count == 0
                   and core.start <= (state.task.core.start + state.task.core.stop - 1) // 2 < core.stop]
        if retries:
            plan.append(_task(core, count, components, ('base_sweep', 'rejected_iteration_retry'),
                              retries=retries, **task_kwargs))
        elif target and estimate > target and len(core) >= 2 * minimum:
            middle = (core.start + core.stop) // 2
            for part in (range(core.start, middle), range(middle, core.stop)):
                plan.append(_task(part, count, components, ('base_sweep', 'predicted_work_split'),
                                  **task_kwargs))
        else:
            plan.append(_task(core, count, components, ('base_sweep',), **task_kwargs))
    return plan


def continuation_plan(states, revisions, depths, activation, components, extra_index,
                      *, self_depths=None, self_unknown=None, halo=None):
    """Select only unfinished/stale, penetrating, or once-retried windows."""
    plan, decisions = [], []
    for state in states:
        task, row = state.task, state.record
        footprint = slice(task.footprint.start, task.footprint.stop)
        reasons = []
        if not row['local_converged'] or not row['accepted']:
            reasons.append('not_converged')
        if not np.array_equal(state.stamp, revisions[footprint]):
            reasons.append('convergence_dependency_changed')
        values = depths[footprint]
        if not np.isfinite(values).all():
            reasons.append('grid_depth_unknown')
        if np.any(values > activation):
            reasons.append('acquisition_grid_penetration')
        if self_depths is not None and np.any(self_depths[footprint] > activation):
            reasons.append('acquisition_self_penetration')
        if self_unknown is not None and np.any(self_unknown[footprint]):
            reasons.append('self_depth_unknown')
        rejected = row['local_stop_reason'] == 'rejected_iteration'
        exhausted = rejected and task.retry_count > 0
        if rejected:
            reasons.append('rejected_iteration_retry' if not exhausted else 'rejected_retry_exhausted')
        depth = float(np.max(values, initial=0.))
        decisions.append(dict(core=task.key, frames=(task.frames.start, task.frames.stop),
                              reasons=reasons, selected=bool(reasons and not exhausted),
                              grid_max_depth_m=depth if np.isfinite(depth) else None,
                              activation_depth_m=activation, retry_count=task.retry_count))
        if not reasons or exhausted:
            continue
        if rejected:
            follow = _task(task.core, len(revisions), components, reasons, retries=(state,),
                           **({} if halo is None else {'halo': halo}))
        else:
            # Shift interfaces while keeping the complete logical core movable.
            shift = ((HALO_FRAMES if halo is None else halo) // 2) * (1 if extra_index % 2 == 0 else -1)
            follow = _task(task.core, len(revisions), components, reasons, shift=shift,
                           **({} if halo is None else {'halo': halo}))
            follow.retry_count = task.retry_count
        plan.append(follow)
    return plan, decisions


def _history_row(task, result, details, *, label, index, sweep, color, count, serial_reason,
                 expected, batch_seconds, dispatch_order, started, setup_seconds):
    trace = result['history']
    stats = {k: v for k, v in (trace[-1] if trace else {}).items()
             if k not in ('phase', 'iteration', 'cost', 'after', 'alpha',
                          'seconds', 'step_backend', 'damping')}
    return report_value(dict(
        stats, phase=label, iteration=index, sweep=sweep, color=color,
        core=task.key, frames=(task.frames.start, task.frames.stop),
        footprint=(task.footprint.start, task.footprint.stop), halo_frames=task.halo,
        continuation_reasons=task.reasons, retry_count=task.retry_count, retry_of=task.retry_of,
        step_backend='legacy_windows', alpha=1. if details['accepted'] else 0.,
        damping=trace[-1]['damping'] if trace else INITIAL_DAMPING,
        local_iterations=len(trace), local_history=trace, local_alphas=[r['alpha'] for r in trace],
        local_converged=result['converged'], local_stop_reason=result['stop_reason'],
        accepted_iterations=sum(r['alpha'] > 0 for r in trace),
        window_seconds=result['seconds'], lm_seconds=result.get('lm_seconds'),
        window_evaluation_seconds=result.get('evaluation_seconds'),
        window_evaluation_calls=result.get('evaluation_calls'),
        window_evaluation_jacobian_calls=result.get('evaluation_jacobian_calls'),
        window_performance=(
            result['history'][-1].get('solver_performance') if result['history'] else None),
        qoco_restoration=result.get('qoco_restoration', {}),
        parallel_restoration=result.get('parallel_restoration', {}),
        **({'iteration_quota': task.iteration_quota} if task.iteration_quota is not None else {}),
        batch_seconds=batch_seconds, expected_work=expected, dispatch_order=dispatch_order,
        queue_seconds=result.get('queue_seconds', 0.), completion_seconds=result.get('completion_seconds'),
        worker_pid=result['worker_pid'], worker_count=count, serial_reason=serial_reason,
        setup_seconds=setup_seconds if index == 0 else 0., seconds=time.monotonic() - started, **details))


def focused(plan, focus):
    """Only windows whose movable frames contain a flagged frame."""
    return plan if focus is None else [task for task in plan
                                       if focus[task.frames.start:task.frames.stop].any()]


def optimize_legacy_windows(objective, poses, *, iterations, callback=None,
                            label='contact_scene', converged_decrease=CONVERGED_DECREASE,
                            focus=None, core_frames=None, halo_frames=None, min_core_frames=None,
                            sweep_iterations=None, max_extra_sweeps=None):
    """Old acquisition plus selective continuation; final stages stay intact.

    ``focus`` (frame mask) re-solves only the base-sweep windows containing a
    flagged frame, e.g. a re-weighting pass after a few added-chatter frames;
    every other frame stays exactly as given and no continuation runs.
    Explicit dimensions/sweep quotas support coarse acquisition; omitted
    parameters retain the original full-rate schedule and continuation caps.
    """
    from .mesh_repair import _activation_depth
    sweeps = SWEEP_ITERATIONS if sweep_iterations is None else tuple(sweep_iterations)
    extra_sweeps = MAX_EXTRA_SWEEPS if max_extra_sweeps is None else max_extra_sweeps
    if not sweeps or any(q <= 0 for q in sweeps) or extra_sweeps < 0:
        raise ValueError('Invalid legacy acquisition schedule')
    if ((core_frames is not None and core_frames <= 0)
            or (halo_frames is not None and halo_frames < 0)
            or (min_core_frames is not None and min_core_frames <= 0)):
        raise ValueError('Invalid legacy window dimensions')
    plan_kwargs = {name: value for name, value in
                   (('core', core_frames), ('halo', halo_frames), ('minimum', min_core_frames))
                   if value is not None}
    _validate(objective)
    started = time.monotonic()
    if iterations <= 0:
        return poses.copy(), []
    objective.stop_workers()
    objective.prepare_window_evaluation()
    initial = _failures(objective, poses)
    if initial:
        raise ValueError(f'legacy_windows requires a guard-feasible global entry: {initial}')
    components = [objective, *guards_of(objective)[1]]
    first_plan = focused(base_plan(len(poses), 0, components, [], min(iterations, sweeps[0]),
                                  **plan_kwargs),
                         focus)
    if not first_plan:
        return poses.copy(), []
    safe, serial_reason = fork_status()
    ceiling = min(MAX_WORKERS, max(1, available_cpus() // 2))
    # Later plans can have more independent jobs than the first plan.
    count = ceiling if safe else 0
    current, history, states = poses.copy(), [], {}
    revisions = np.zeros(len(poses), np.int64)
    remaining, extra_started = int(iterations), None
    extra_remaining = MAX_EXTRA_WINDOW_ITERATIONS
    prioritize_penetration = False
    summary = dict(max_extra_sweeps=extra_sweeps,
                   extra_window_iteration_budget=MAX_EXTRA_WINDOW_ITERATIONS,
                   extra_window_iterations_reserved=0, extra_window_iterations_completed=0,
                   extra_sweeps=0, grid_seconds=0., grid_scope='acquisition static/moving SDF shell samples',
                   worker_ceiling=ceiling, stop_reason='extra_sweep_cap')
    setup_seconds = time.monotonic() - started
    with _workers(objective, count) as workers:
        for sweep in range(len(sweeps) + (extra_sweeps if focus is None else 0)):
            extra = sweep >= len(sweeps)
            quota = min(remaining, EXTRA_ITERATIONS if extra else sweeps[sweep])
            if quota <= 0:
                summary['stop_reason'] = 'iteration_budget'
                break
            previous = sorted(states.values(), key=lambda state: state.task.key)
            rows = [s.record for s in previous]
            if extra:
                if extra_started is None:
                    extra_started = time.monotonic()
                tick = time.monotonic()
                depths, families = grid_depths(objective, current)
                self_values, self_unknown = (None, None)
                summary['grid_seconds'] += time.monotonic() - tick
                plan, decisions = continuation_plan(previous, revisions, depths, _activation_depth(),
                    components, sweep - len(sweeps),
                    self_depths=self_values, self_unknown=self_unknown, halo=halo_frames)
                summary.update(selection=decisions, grid_families=families, selection_after_sweep=sweep - 1)
                if not plan:
                    summary['stop_reason'] = ('rejected_retry_exhausted' if any(
                        'rejected_retry_exhausted' in d['reasons'] for d in decisions) else 'no_eligible_windows')
                    break
                quota = min(quota, extra_remaining // len(plan))
                reserved = quota * len(plan)
                if quota <= 0:
                    summary.update(stop_reason='extra_window_iteration_budget',
                                   unfinished_cores=[task.key for task in plan])
                    break
                extra_remaining -= reserved
                summary['extra_window_iterations_reserved'] += reserved
                summary['extra_sweeps'] += 1
            else:
                plan = first_plan if sweep == 0 else focused(
                    base_plan(len(poses), sweep, components, previous, quota, **plan_kwargs), focus)
                states = {}
            remaining -= quota
            checkpoint, first_record = current.copy(), len(history)
            batches = [sorted(batch, key=lambda w: (-expected_work(w.frames, rows, quota),
                                                   w.frames.start, w.frames.stop, w.key))
                       for batch in independent_batches(plan)]

            def commit(current, snapshot, task, result):
                candidate, details = _commit(objective, current, snapshot, task, result)
                if details['accepted']:
                    revisions[task.frames.start:task.frames.stop] += 1
                return candidate, (details, revisions[task.footprint.start:task.footprint.stop].copy())

            from .window_execution import execute
            current, completed, group_times, sweep_schedule = execute(
                workers, current, batches, quota, converged_decrease, commit, solve_local)
            offset = 0
            for color, batch in enumerate(batches):
                results = completed[offset:offset + len(batch)]
                batch_seconds = group_times[color]['active_span_seconds']
                for dispatch_order, (task, (result, (details, stamp))) in enumerate(zip(batch, results)):
                    record = _history_row(
                        task, result, details, label=label, index=len(history), sweep=sweep, color=color,
                        count=count, serial_reason=serial_reason, expected=expected_work(task.frames, rows, quota),
                        batch_seconds=batch_seconds, dispatch_order=dispatch_order, started=started,
                        setup_seconds=setup_seconds)
                    history.append(record)
                    if extra:
                        summary['extra_window_iterations_completed'] += len(result['history'])
                    states[task.key] = WindowState(task, record, stamp)
                if len(history) > first_record:
                    busy = sum(r['seconds'] for r, _ in results)
                    history[-1]['batch_schedule'] = dict(
                        jobs=len(batch), completed=len(results), workers=count,
                        iteration_quota=quota,
                        **group_times[color], completed_worker_seconds=busy,
                        utilization=busy / max(batch_seconds * max(count, 1), 1e-12))
                offset += len(batch)
            if len(history) > first_record:
                history[-1]['sweep_schedule'] = sweep_schedule
            tick = time.monotonic()
            failures = _failures(objective, current)
            audit_seconds = time.monotonic() - tick
            summary.setdefault('sweep_audits', []).append(dict(
                sweep=sweep, accepted=not failures, failures=failures, seconds=audit_seconds))
            if len(history) > first_record:
                history[-1].update(sweep_guard_failures=failures, sweep_guard_accepted=not failures,
                                   sweep_guard_seconds=audit_seconds, seconds=time.monotonic() - started)
            if failures:
                current = checkpoint
                for record in history[first_record:]:
                    record.update(sweep_rolled_back=True, accepted=False, alpha=0.)
                summary['stop_reason'] = 'sweep_guard_failure'
                break
            if callback is not None and len(history) > first_record:
                callback(current.copy(), dict(history[-1]))
        budget = getattr(objective, '_window_cpu_budget', None)
        if budget is not None:
            summary['native_cpu_budget'] = budget.report()
    if history:
        summary['extra_seconds'] = 0. if extra_started is None else time.monotonic() - extra_started
        history[-1].update(seconds=time.monotonic() - started, continuation=report_value(summary))
    return current, history
