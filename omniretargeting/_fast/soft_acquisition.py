"""Default-off acquisition: private soft bulk, one projection, exact polish."""

import time

from .legacy_window import optimize_legacy_windows
from .legacy_window_qp import commit_verdict
from .soft_bulk import run_bulk
from .soft_projection import certify, project_bulk

GUARD_STIFFNESS = 1.
POLISH_QUOTA = 2


def optimize_soft_acquisition(objective, poses, *, iterations, callback=None):
    started = time.monotonic()
    # Private infeasible bulk poses must never reach the production callback.
    bulk, bulk_history, bulk_report = run_bulk(objective, poses, iterations=20,
        threshold=.001, speed_gate=False, guard_stiffness=GUARD_STIFFNESS)
    projected, projection = project_bulk(objective, bulk, poses, measure_costs=True, objective_aware=True)
    tick = time.monotonic()
    backend = objective.evaluation_backend
    objective.evaluation_backend = 'cpu'
    objective.start_workers()
    try:
        costs = objective.evaluate(poses, False)[0].cost, objective.evaluate(projected, False)[0].cost
    finally:
        objective.stop_workers()
        objective.evaluation_backend = backend
    verdict = commit_verdict(objective, poses, projected, range(len(poses)), costs=costs)
    projection.update(commit=verdict, certification_cost_seconds=time.monotonic() - tick)
    if not verdict['accepted']:
        projected = poses.copy()
    if callback is not None:
        callback(projected.copy(), dict(phase='soft_projection', **verdict))
    tick = time.monotonic()
    polished, history = optimize_legacy_windows(objective, projected, iterations=iterations,
        callback=callback, label='soft_polish', sweep_iterations=(POLISH_QUOTA,), max_extra_sweeps=0)
    polish_seconds = time.monotonic() - tick
    failures = certify(objective, polished)
    if failures:
        raise RuntimeError(f'Soft acquisition polish failed original guards: {failures}')
    report = dict(mode='soft_first_hard_once', bulk=bulk_report, projection=projection,
        bulk_seconds=bulk_report['seconds'], projection_seconds=projection['seconds'],
        certification_cost_seconds=projection['certification_cost_seconds'],
        polish_seconds=polish_seconds, polish_quota=POLISH_QUOTA,
        seconds=time.monotonic() - started, guard_failures=failures,
        polish_window_iterations=sum(row.get('local_iterations', 0) for row in history),
        fast_self_collision=False)
    from .runtime_options import OPTIONS
    report.update(bulk_cpu_geometry=OPTIONS.bulk_cpu_geometry,
                  repair_triage=OPTIONS.repair_triage, repair_fast_gate=OPTIONS.repair_fast_gate)
    # The bulk curve is in the report, not feasible-trajectory history.
    return polished, history, report
