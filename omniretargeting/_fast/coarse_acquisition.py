"""Opt-in half-rate acquisition with a certified full-rate warm start."""

import gc
import os
import time

import numpy as np

from .coarse_problem import make_coarse_problem
from .coarse_transfer import failure_frames, transfer_correction
from .guard_signals import prepare
from .legacy_window import MIN_WINDOW_FRAMES, optimize_legacy_windows, unsupported_reason
from .solver import optimize


# Read once per process; the experiment launcher already forwards environment.
COARSE_SWEEPS = tuple(int(value) for value in os.environ.get(
    'OMNIRETARGETING_FAST_COARSE_SWEEPS', '6').split(','))
REFINE_ITERATIONS = int(os.environ.get('OMNIRETARGETING_FAST_COARSE_REFINE_ITERATIONS', '2'))
FORCE_WINDOWS = os.environ.get('OMNIRETARGETING_FAST_COARSE_FORCE_WINDOWS', '0') == '1'
MIN_RETENTION = float(os.environ.get('OMNIRETARGETING_FAST_COARSE_MIN_RETENTION', '.25'))
CORE_FRAMES, HALO_FRAMES, MIN_CORE_FRAMES = 32, 8, 16

if (not COARSE_SWEEPS or any(value <= 0 for value in COARSE_SWEEPS)
        or REFINE_ITERATIONS <= 0 or not 0 < MIN_RETENTION <= 1):
    raise ValueError('Invalid coarse acquisition experiment settings')


def _run_coarse(objective, poses, use_windows):
    """Keep coarse state and exception tracebacks out of the full-rate scope."""
    coarse, phase = None, 'coarse construction'
    try:
        coarse, indices = make_coarse_problem(objective)
        prepare(coarse)
        phase = 'coarse entry'
        entry = poses[indices].copy()
        reason = unsupported_reason(coarse, entry)
        if reason is not None:
            return None, None, [], f'{phase}: {reason}'
        phase = 'coarse solve'
        if use_windows:
            acquired, history = optimize_legacy_windows(
                coarse, entry, iterations=sum(COARSE_SWEEPS), label='contact_scene_coarse',
                core_frames=CORE_FRAMES, halo_frames=HALO_FRAMES, min_core_frames=MIN_CORE_FRAMES,
                sweep_iterations=COARSE_SWEEPS, max_extra_sweeps=0)
        else:
            acquired, history = optimize(
                coarse, entry, iterations=sum(COARSE_SWEEPS), label='contact_scene_coarse')
        # An optimizer/test adapter may return a view into its objective.
        # Retain only independent poses, sample indices and JSON history.
        return acquired.copy(), indices.copy(), history, None
    except Exception as error:
        return None, None, [], f'{phase}: {type(error).__name__}: {error}'
    finally:
        if coarse is not None:
            # Legacy windows join their children; the ordinary optimizer
            # owns its pool. Cyclic guards/caches are collected by the caller
            # after this frame (and its exception traceback) is gone.
            coarse.stop_workers()


def _grid_summary(objective, poses, *, chunk_size=64):
    """Serial acquisition-grid samples only; no pool or original-mesh query."""
    tick = time.monotonic()
    summary = dict(scope='acquisition SDF shell samples; scene/objects only; not a mesh certificate',
                   fields={})
    if objective.scene_sdf is not None or objective.objects:
        values = {}
        for first in range(0, len(poses), chunk_size):
            span = slice(first, min(len(poses), first + chunk_size))
            local = objective.for_frames(span)
            local.kin.evaluate(poses[span], with_jacobian=False, native=True)
            for name, *field in local._clearance_fields():
                clearance = local._deepest_samples(*field)[0]
                depths = np.where(np.isfinite(clearance), np.maximum(-clearance, 0.), 0.)
                peak = depths.max(axis=1, initial=0.)
                by_frame = values.setdefault(name, np.zeros(len(poses)))
                by_frame[span] = np.maximum(by_frame[span], peak)
                row = summary['fields'].setdefault(name, dict(max_m=0., worst=None))
                maximum = float(peak.max(initial=0.))
                if maximum > row['max_m']:
                    frame, column = np.unravel_index(np.argmax(depths), depths.shape)
                    geom = int(objective.shell_geoms[column])
                    body = int(objective.model.geom_bodyid[geom])
                    row.update(max_m=maximum, worst=dict(frame=first + int(frame),
                        geom=objective.model.geom(geom).name, body=objective.model.body(body).name))
        for name, by_frame in values.items():
            summary['fields'][name].update(frames_over_5mm=int((by_frame > .005).sum()),
                frames_over_1cm=int((by_frame > .01).sum()), frames_over_2cm=int((by_frame > .02).sum()))
    summary['seconds'] = time.monotonic() - tick
    return summary


def optimize_coarse_acquisition(objective, poses, *, iterations, callback=None,
                                window_status=None, label='contact_scene'):
    """Keep the original reference/guards authoritative across both solves.

    Routing uses the original clip length. A poor or unsupported warm start
    runs the normal full-rate schedule from its original entry, with a reason.
    ContactGuard is deliberately left to the pipeline after full-rate refine.
    """
    # Exact guard acceleration does not require an evaluation worker pool.
    prepare(objective)
    normal_status = (unsupported_reason(objective, poses, min_frames=MIN_WINDOW_FRAMES)
                     or 'used') if window_status is None else window_status
    use_windows = objective.frames >= MIN_WINDOW_FRAMES or FORCE_WINDOWS
    report = dict(enabled=True, original_frames=objective.frames,
                  sampled_frames=(objective.frames + 1) // 2,
                  coarse_sweeps=list(COARSE_SWEEPS), refine_iterations=REFINE_ITERATIONS,
                  coarse_core_frames=CORE_FRAMES, coarse_halo_frames=HALO_FRAMES,
                  coarse_min_core_frames=MIN_CORE_FRAMES, force_windows=FORCE_WINDOWS,
                  coarse_seconds=0., transfer_seconds=0., refine_seconds=0., fallback_seconds=0.,
                  retained_scale=0., transfer_guard_failures=[], fallback_ran=False,
                  safety_net_triggered=False,
                  after_transfer_grid=None, after_refine_grid=None,
                  safety_minimum_retained_scale=MIN_RETENTION,
                  continuity_bounds='unchanged full-rate guards composed with linear correction interpolation',
                  reference_rebased=False, full_rate_guards_authoritative=True)
    objective.stop_workers()

    def fallback(reason):
        report.update(fallback_ran=True, fallback_reason=reason, safety_net_triggered=True)
        tick = time.monotonic()
        solver = optimize_legacy_windows if normal_status == 'used' else optimize
        motion, history = solver(objective, poses, iterations=iterations,
                                 callback=callback, label=label)
        report['fallback_seconds'] = time.monotonic() - tick
        report['after_fallback_grid'] = _grid_summary(objective, motion)
        report['after_acquisition_grid'] = report['after_fallback_grid']
        return motion, history, report

    # Unsupported experimental objective/guard combinations keep their own
    # original solver. Do not reconstruct missing semantics at half rate.
    reason = unsupported_reason(objective, poses)
    if reason is not None:
        return fallback(reason)
    tick = time.monotonic()
    acquired, indices, coarse_history, reason = _run_coarse(objective, poses, use_windows)
    gc.collect()  # guard -> objective -> guard, cached window views, partial constructions
    report.update(coarse_seconds=time.monotonic() - tick, coarse_history=coarse_history,
                  solver='legacy_windows' if use_windows else 'whole_trajectory')
    if reason is not None:
        return fallback(reason)
    tick = time.monotonic()
    warm, transfer = transfer_correction(objective, objective.reference, acquired, indices,
                                         min_retention=MIN_RETENTION)
    report.update(transfer_seconds=time.monotonic() - tick, transfer=transfer,
                  retained_scale=transfer['retained_scale'],
                  safety_net_triggered=transfer['safety_net_triggered'],
                  transfer_guard_failures=transfer['guard_failures'],
                  first_transfer_guard_failure_counts=(transfer['first_failing_candidate'] or {}).get(
                      'guard_failure_counts', {}))
    del acquired, indices
    report['after_transfer_grid'] = _grid_summary(objective, warm)
    if not transfer['accepted']:
        return fallback(f'transfer: {transfer["reason"]}')
    tick = time.monotonic()
    if use_windows:
        motion, history = optimize_legacy_windows(
            objective, warm, iterations=REFINE_ITERATIONS, callback=callback, label=label,
            sweep_iterations=(REFINE_ITERATIONS,), max_extra_sweeps=0)
    else:
        motion, history = optimize(objective, warm, iterations=REFINE_ITERATIONS,
                                   callback=callback, label=label)
    report['refine_seconds'] = time.monotonic() - tick
    failures = failure_frames(objective, motion, certify=True)
    if failures:
        raise RuntimeError(f'Full-rate coarse acquisition refinement violated guards: {failures}')
    report['after_refine_grid'] = _grid_summary(objective, motion)
    report['after_acquisition_grid'] = report['after_refine_grid']
    return motion, history, report
