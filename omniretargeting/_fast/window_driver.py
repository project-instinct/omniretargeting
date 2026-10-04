"""Overlapping Gauss--Seidel SQP with private local searches and parent commits."""

import multiprocessing
import os
import threading
import time
import traceback

import numpy as np

from .priority_merit import report_value
from .proposals import guards_of
from .qp_types import PriorityMerit
from .window_assembly import LEVEL_WEIGHTS, prepare_window_model
from .window_local import solve_local_window, exact_trial, assess_window, STEP_TOLERANCE
from .window_schedule import WINDOW_SWEEPS, dependency_window, independent_batches, sweep_windows
from .window_journal import WindowJournal

MAX_WINDOW_WORKERS = 4


def fork_status():
    """Unknown live Python/native threading falls back to serial execution."""
    if not hasattr(os, 'fork'):
        return False, 'fork_unavailable'
    if threading.active_count() != 1:
        return False, 'live_python_threads'
    try:
        from threadpoolctl import threadpool_info
        if any(item.get('num_threads', 1) != 1 for item in threadpool_info()):
            return False, 'native_thread_count_not_one'
    except ImportError:
        return False, 'native_thread_state_unknown'
    return True, None


def _serve(objective, poses, frames, iterations, convergence, sender, receiver):
    receiver.close()
    try:
        sender.send(('ok', solve_local_window(objective, poses, frames, iterations, convergence)))
    except BaseException:
        sender.send(('error', traceback.format_exc()))
    finally:
        sender.close()


def solve_batch(objective, poses, windows, iterations, convergence, parallel):
    """Fork processes directly: no task/result threads or shared ledger writes."""
    if not parallel or len(windows) == 1:
        return [('ok', solve_local_window(objective, poses, w.frames, iterations, convergence)) for w in windows]
    context = multiprocessing.get_context('fork')
    jobs = []
    try:
        for window in windows:
            receiver, sender = context.Pipe(duplex=False)
            process = context.Process(target=_serve, args=(objective, poses, window.frames,
                                      iterations, convergence, sender, receiver))
            process.start()
            sender.close()
            jobs.append((process, receiver))
        results = []
        for process, receiver in jobs:
            try:
                results.append(receiver.recv())
            except EOFError:
                results.append(('error', f'Window child exited without a result (pid {process.pid})'))
            process.join()
        return results
    finally:
        for process, receiver in jobs:
            receiver.close()
            if process.is_alive():
                process.terminate()
            process.join()


def _verify_and_commit(objective, poses, frames, result):
    """A whole child excursion is ONE net step; no intermediate credit imports."""
    owner = os.getpid()
    candidate = poses.copy()
    span = slice(frames.start, frames.stop)
    part = np.asarray(result['poses'])
    if part.shape != candidate[span].shape or not np.isfinite(part).all():
        return None, dict(parent_reason='invalid_window_payload')
    candidate[span] = part
    delta = objective.coords.difference(poses, candidate)[span].ravel()
    if np.max(abs(delta), initial=0.) <= STEP_TOLERANCE:
        return None, dict(parent_reason='no_change_above_tolerance')
    tick = time.monotonic()
    evaluation = objective.evaluate(poses, frames=frames)
    model, _ = prepare_window_model(objective, poses, frames, evaluation)
    coordinator = objective.priority_clearance
    before = PriorityMerit.coerce(coordinator.evaluate_trial(poses, frames=frames))
    after, valid = exact_trial(objective, candidate, frames)
    details = dict(parent_reason=after.details.get('reason'), parent_seconds=time.monotonic() - tick)
    if not valid:
        return None, details
    trial_evaluation = objective.evaluate(candidate, with_jacobian=False, frames=frames)
    value = trial_evaluation[0].cost
    accepted, ratio, merit = assess_window(model, delta, before, after, evaluation[0].cost, value,
                                           LEVEL_WEIGHTS * model.scale)
    details.update(cost=evaluation[0].cost, after=value, ratio=ratio,
                   stats=report_value(trial_evaluation[3]), **merit)
    if not accepted:
        details['parent_reason'] = 'net_tracking_ratio'
        return None, details
    assert os.getpid() == owner
    # commit itself remeasures and atomically installs only its new net report.
    # A stale dependency/transaction is rejected before any ledger mutation.
    try:
        coordinator.commit(candidate, frames=frames, transactions=result['transactions'])
    except ValueError as error:
        details['parent_reason'] = str(error)
        return None, details
    details.update(parent_reason=None, parent_seconds=time.monotonic() - tick,
                   priority_merit=report_value(after.__dict__))
    return candidate, details


def optimize_windowed_priority(objective, poses, iterations, callback, label, converged_decrease):
    """Default-off Stage B entry; same external optimizer signature/history."""
    start = time.monotonic()
    coordinator = getattr(objective, 'priority_clearance', None)
    if coordinator is None:
        raise ValueError('Windowed priority requires its nonlinear coordinator')
    # Stage A may have owned a legacy pool. Close it in its owning parent;
    # nothing in this driver starts a Pool/ThreadPoolExecutor or CUDA client.
    stop = getattr(objective, 'stop_workers', None)
    if stop is not None:
        stop()
    provider = getattr(coordinator, 'provider', None)
    if provider is not None and getattr(provider, 'started', False):
        provider.close()
    prepare_objective = getattr(objective, 'prepare_window_evaluation', None)
    if prepare_objective is not None:
        prepare_objective()
    preparation = None
    geometry = getattr(provider, 'geometry', None)
    if geometry is not None:
        prepare = getattr(geometry, 'prepare_fork_queries', None)
        if prepare is None:
            preparation = 'geometry_fork_preparation_unavailable'
        else:
            prepare()
        # libigl reads/caches its environment on first use. Setting this here
        # would not safely reconfigure an already initialized interpreter.
        if os.environ.get('IGL_NUM_THREADS') != '1':
            preparation = 'igl_threads_not_fixed_before_service_start'
    if iterations <= 0:
        return poses.copy(), []
    if not np.array_equal(poses, coordinator.current):
        raise ValueError('Window optimization must start from the committed trajectory')
    poses = poses.copy()
    journal = WindowJournal()
    journal.checkpoint(poses, coordinator, phase=label, iteration=-1)
    history = []
    setup_seconds = time.monotonic() - start
    components = [objective, *guards_of(objective)[1]]
    if not any(c is coordinator for c in components):
        components.append(coordinator)
    for sweep in range(WINDOW_SWEEPS):
        changed = False
        windows = [dependency_window(w, len(poses), components) for w in sweep_windows(len(poses), sweep)]
        batches = independent_batches(windows)
        for color, batch in enumerate(batches):
            # Capped process batches bound COW/private query memory. Windows
            # of this color have disjoint complete dependency footprints.
            for first in range(0, len(batch), MAX_WINDOW_WORKERS):
                group = batch[first:first + MAX_WINDOW_WORKERS]
                safe, reason = fork_status()
                if preparation is not None:
                    safe, reason = False, preparation
                parallel = safe and len(group) > 1
                batch_start = time.monotonic()
                results = solve_batch(objective, poses, group, iterations, converged_decrease, parallel)
                batch_seconds = time.monotonic() - batch_start
                pending_callbacks = []
                for window, (status, result) in zip(group, results):
                    epoch_before = coordinator.epoch
                    frames = window.frames
                    error = result if status != 'ok' else None
                    local_trace = [] if error else result['trace']
                    local_seconds = 0. if error else result['seconds']
                    candidate, details = None, dict(parent_reason='child_error' if error else 'unchanged_local')
                    if error is None and result['changed']:
                        candidate, details = _verify_and_commit(objective, poses, frames, result)
                    retried = False
                    if error is not None or (result['changed'] and candidate is None):
                        retried = True
                        serial = solve_local_window(objective, poses, frames, iterations, converged_decrease)
                        local_trace = local_trace + serial['trace']
                        local_seconds += serial['seconds']
                        if serial['changed']:
                            candidate, details = _verify_and_commit(objective, poses, frames, serial)
                    accepted = candidate is not None
                    if accepted:
                        poses, changed = candidate, True
                    from .solver import INITIAL_DAMPING
                    origin_cost = details.get('cost', local_trace[0]['cost'] if local_trace else float('nan'))
                    stats = details.pop('stats', local_trace[0].get('stats', {}) if local_trace else {})
                    record = dict(stats, phase=label, iteration=len(history), sweep=sweep, color=color,
                                  frames=(frames.start, frames.stop), footprint=(window.footprint.start, window.footprint.stop),
                                  alpha=1. if accepted else 0., step_backend='windowed_priority',
                                  priority_status='accepted_window' if accepted else 'no_acceptable_window_proposal',
                                  parallel=parallel, serial_reason=reason, serial_retry=retried,
                                  child_error=error, local_iterations=len(local_trace), local_trace=local_trace,
                                  local_wall_seconds=local_seconds, batch_seconds=batch_seconds,
                                  setup_seconds=setup_seconds if not history else 0.,
                                  optimality_certified=False, convergence_certified=False,
                                  seconds=time.monotonic() - start, **details)
                    record.update(cost=origin_cost,
                                  after=details.get('after', origin_cost) if accepted else origin_cost,
                                  damping=local_trace[-1]['damping'] if local_trace else INITIAL_DAMPING)
                    history.append(record)
                    journal.window(record, poses, coordinator, epoch_before)
                    if accepted and callback is not None:
                        pending_callbacks.append((poses.copy(), dict(record)))
                # Every committed state/history item is durable before an
                # external progress consumer can fail. The latest checkpoint
                # contains the whole batch, not an intermediate child claim.
                journal.checkpoint(poses, coordinator, phase=label, sweep=sweep, color=color,
                                   iteration=len(history) - 1)
                journal.emit('batch_complete', phase=label, sweep=sweep, color=color,
                             epoch=coordinator.epoch, windows=len(group),
                             checkpoint=str(journal.checkpoint_path))
                for callback_poses, callback_record in pending_callbacks:
                    try:
                        callback(callback_poses, callback_record)
                    except BaseException:
                        journal.emit('callback_error', phase=label, iteration=callback_record['iteration'],
                                     error=traceback.format_exc(), epoch=coordinator.epoch)
                        raise
        if not changed:
            break
    return poses, history
