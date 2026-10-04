"""Whole-trajectory acquisition; final continuity/layout, acquired contacts.

Private bulk may exceed the internal acquisition normal bounds. The final
transaction certifies both ORIGINAL continuity families, boxes/root domains
and layout. Downstream contact regression bounds are frozen once at this
acquired motion; neither temporal budgets nor the reference are rebased.
"""
from contextlib import contextmanager
import copy
import time

import numpy as np

from .acquisition_guard import AcquisitionNormalGuard
from .kinematics import TrajectoryKinematics
from .legacy_window_qp import commit_verdict
from .soft_bulk import run_bulk
from .soft_projection import certify, project_bulk

BALANCED_LIMIT = 30
CLEARANCE_LIMIT = 20
CLEARANCE_MULTIPLIER = 4.


@contextmanager
def final_guards(objective):
    original = objective.candidate_guards
    objective.candidate_guards = [g for g in original if not isinstance(g, AcquisitionNormalGuard)]
    try:
        yield
    finally:
        objective.candidate_guards = original


def acquired_normal_guard(frozen, poses):
    """Keep original masks/targets/yield, max(frozen bound, acquired gap)."""
    guard = copy.copy(frozen)
    guard.kin = TrajectoryKinematics(frozen.objective.model, frozen.objective.coords.cols,
                                     cache=frozen.objective.kin.cache)
    guard.kin.evaluate(poses, with_jacobian=False)
    guard.groups = []
    for kind, indices, local, target, normal, active, established, bound, yielded in frozen.groups:
        offset = guard._points(kind, indices, local) - target
        along = np.sum(offset * normal, axis=-1)
        gap = np.where(established, np.abs(along), np.linalg.norm(offset, axis=-1))
        acquired_bound = np.maximum(bound, gap)
        acquired_bound.setflags(write=False)
        guard.groups.append((kind, indices, local, target, normal, active, established,
                             acquired_bound, yielded))
    guard._regression_reference = 'acquired_motion_max_frozen_and_acquired_gap'
    if not guard.accepts(poses):
        raise RuntimeError('Acquired regression guard does not accept its frozen entry')
    return guard


def project_final(objective, bulk, entry, *, measure_costs=True):
    with final_guards(objective):
        candidate, report = project_bulk(objective, bulk, entry, measure_costs=measure_costs)
        # Report the actual geometry scope; the shared old projector includes
        # acquisition normals in its generic stage labels, not its constraints.
        for row in report.get('cost_by_guard_family', ()):
            if 'guard_families' in row:
                row['guard_families'] = [f for f in row['guard_families'] if f != 'AcquisitionNormalGuard']
        failures = certify(objective, candidate)
        if failures:
            raise RuntimeError(f'Soft-v4 final certificate failed: {failures}')
    report['final_requirements'] = ['SequenceContinuityGuard', 'CorrectionContinuity',
                                    'ReferenceShape', 'joint_limits', 'root_domains']
    report['protected_normal_projection'] = False
    return candidate, report


def optimize_soft_acquisition_v4(objective, poses, *, iterations, callback=None):
    started = time.monotonic()
    # Switch out of balanced fitting at its 1% progress plateau; finish
    # clearance at a tighter 0.3% plateau (two successive small decreases).
    # The caps are safety limits, not fixed iteration quotas. Every phase uses
    # the same frozen guard/reference objects and 1x numerical contact hinges.
    bulk, _, balanced = run_bulk(objective, poses, iterations=BALANCED_LIMIT, threshold=.01,
        damping_floor=.001)
    bulk, _, clearance = run_bulk(objective, bulk, iterations=CLEARANCE_LIMIT, threshold=.003,
        clearance_multiplier=CLEARANCE_MULTIPLIER, initial_damping=1., damping_floor=1.,
        damping_decay=.8, self_monotone=True)
    return finish_acquisition(objective, poses, bulk, [balanced, clearance], started, callback)


def finish_acquisition(objective, poses, bulk, phases, started, callback=None, *,
                       mode='soft_v4_continuity_layout_only', support_projection=False):
    """Shared certified transaction; private bulk states never reach callbacks."""
    projected, projection = project_final(objective, bulk, poses)
    if support_projection:
        from .soft_support_projection import refine
        support_tick = time.monotonic()
        backend = objective.evaluation_backend
        objective.evaluation_backend = 'cpu'
        objective.start_workers()
        try:
            ev = objective.evaluate(projected)
            metric = np.maximum(np.diagonal(ev[0].hessian, axis1=1, axis2=2), 1.)
            with final_guards(objective):
                source_yield=any(p.get('support_rows',{}).get('source_normal_yield',False) for p in phases)
                world_recovery=any('world_target_recovery' in p for p in phases)
                projected, contacts = refine(objective, projected, bulk, metric, evaluate=objective.evaluate,
                    physical=ev,source_yield=source_yield,world_recovery=world_recovery)
            after = objective.evaluate(projected, False)
        finally:
            objective.stop_workers()
            objective.evaluation_backend = backend
        projection['contact_projection'] = contacts
        projection['support_total_seconds'] = time.monotonic() - support_tick
        projection['seconds'] += projection['support_total_seconds']
        projection.setdefault('cost_by_guard_family', []).append(dict(stage='contact_aware_joint_projection',
            cost=after[0].cost, stats=after[3]))
    tick = time.monotonic()
    backend = objective.evaluation_backend
    objective.evaluation_backend = 'cpu'
    objective.start_workers()
    try:
        costs = objective.evaluate(poses, False)[0].cost, objective.evaluate(projected, False)[0].cost
    finally:
        objective.stop_workers()
        objective.evaluation_backend = backend
    with final_guards(objective):
        verdict = commit_verdict(objective, poses, projected, range(len(poses)), costs=costs)
    if not verdict['accepted']:
        projected = poses.copy()
    projection.update(commit=verdict, certification_cost_seconds=time.monotonic() - tick)
    original = next(g for g in objective.candidate_guards if isinstance(g, AcquisitionNormalGuard))
    frozen_diagnostics = original.diagnostics(projected)
    regression = acquired_normal_guard(original, projected)
    objective.candidate_guards = [regression if g is original else g for g in objective.candidate_guards]
    objective.hinge_guards = [regression if g is original else g for g in objective.hinge_guards]
    objective.soft_acquired_normal_guard = regression
    if callback is not None:
        callback(projected.copy(), dict(phase='soft_v4_projection', **verdict))
    report = dict(mode=mode, bulk_phases=phases,
        bulk_seconds=sum(phase['seconds'] for phase in phases),
        projection_seconds=projection['seconds'], projection=projection,
        certification_cost_seconds=projection['certification_cost_seconds'],
        polish_seconds=0., polish_window_iterations=0,
        contact_guard_semantics=regression._regression_reference,
        normals_vs_original_acquisition_bounds=frozen_diagnostics,
        fixed_reference_and_temporal_budgets='unchanged', seconds=time.monotonic() - started)
    from .runtime_options import OPTIONS
    report.update(bulk_cpu_geometry=OPTIONS.bulk_cpu_geometry)
    return projected, [], report
