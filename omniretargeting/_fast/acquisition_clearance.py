"""Finish original-collider acquisition before attained contacts are frozen."""

import numpy as np

from .local_reach import unreachable_geoms
from .original_clearance import OriginalClearance
from .repair_limits import LOCAL_ANGLE, FREE_DRIFT


def complete_priority_clearance(objective, poses, optimize, callback=None):
    """Finish the same priority problem before acquired contacts are frozen.

    No new contact/reference/credit baseline is created by this pass.
    The coordinator's stalled-search fingerprint suppresses an identical
    failed proposal search. A changed QP or search policy is solved again;
    neither stopping nor deduplication certifies hard-level infeasibility.
    """
    coordinator = objective.priority_clearance
    measured = coordinator.provider.measure(poses)
    before = coordinator.provider.summary(measured)
    record = dict(stage='priority_clearance_completion', before=before, history=[])
    if measured.get('failures') or not measured.get('complete', True):
        record.update(status='unknown_query', after=before)
        return poses, [], record
    if max(measured['depths'].values(), default=0.) <= .001:
        record.update(status='below_repair_activation', after=before)
        return poses, [], record
    from .runtime_options import OPTIONS
    if OPTIONS.windowed_priority:
        # Stage B already completed its configured sweeps. Do not launch a
        # duplicate whole acquisition solve before the exceptional repair.
        record.update(status='residual', after=before, audit_only=True,
                      stop_reason='window_sweeps_finished', convergence_certified=False,
                      hard_infeasibility_certified=False, residuals_unresolved=True)
        return poses, [], record
    candidate, history = optimize(objective, poses, iterations=objective.cfg['iterations'],
                                  callback=callback, label='priority_clearance_completion')
    if not np.array_equal(candidate, coordinator.current):
        raise RuntimeError('Priority solver returned a trajectory it did not commit')
    after = coordinator.provider.measure(candidate)
    record.update(status='completed' if max(after['depths'].values(), default=0.) <= .001 else 'residual',
                  after=coordinator.provider.summary(after), history=history)
    _termination(record, history)
    return candidate, history, record


def repair_priority_clearance(objective, poses):
    """Exceptional fixed-root local repair, retaining the acquisition ledger."""
    import time
    from .continuity import CorrectionContinuity
    from .solver import optimize

    started = time.monotonic()
    coordinator = objective.priority_clearance
    measured = coordinator.provider.measure(poses)
    before = coordinator.provider.summary(measured)
    report = dict(before=before, after=before, history=[], root_fixed=True,
                  local_angle_budget_rad=LOCAL_ANGLE, priority_contract=True)
    if measured.get('failures') or not measured.get('complete', True):
        return poses, dict(report, status='unknown_query', seconds=time.monotonic() - started)
    if max(measured['depths'].values(), default=0.) <= .001:
        return poses, dict(report, status='below_repair_activation', seconds=time.monotonic() - started)
    coords = objective.coords
    lower, upper = objective.lower, objective.upper
    guards = objective.candidate_guards
    old_root = getattr(objective, 'fixed_root', None)
    if coords.root_q is not None:
        objective.fixed_root = poses[:, coords.root_q:coords.root_q + 7].copy()
    # Scalar slide locality is metric; hinges keep the existing angular cap.
    import mujoco
    sliding = objective.model.jnt_type[objective.model.dof_jntid[coords.di]] == mujoco.mjtJoint.mjJNT_SLIDE
    local = np.where(sliding, FREE_DRIFT, LOCAL_ANGLE)
    objective.lower = np.maximum(lower, poses[:, coords.qi] - local)
    objective.upper = np.minimum(upper, poses[:, coords.qi] + local)
    objective.candidate_guards = list(guards) + [CorrectionContinuity(coords, poses)]
    try:
        candidate, history = optimize(objective, poses, iterations=min(12, objective.cfg['iterations']),
                                      label='priority_original_mesh_repair')
        if not np.array_equal(candidate, coordinator.current):
            raise RuntimeError('Priority repair returned an uncommitted trajectory')
    finally:
        objective.lower, objective.upper = lower, upper
        objective.candidate_guards = guards
        if old_root is None:
            objective.__dict__.pop('fixed_root', None)
        else:
            objective.fixed_root = old_root
    after = coordinator.provider.measure(candidate)
    report.update(after=coordinator.provider.summary(after), history=history,
                  status='completed' if max(after['depths'].values(), default=0.) <= .001 else 'residual',
                  seconds=time.monotonic() - started,
                  priority=coordinator.diagnostics(candidate, solver_history=history))
    _termination(report, history)
    return candidate, report


def _termination(report, history):
    """Report search failure separately from penetration/hard feasibility."""
    from .priority_reporting import solver_progress
    report.update(solver_progress(history))
    if history:
        status = history[-1].get('priority_status')
        if status in ('no_certified_accepted_proposal', 'repeated_stalled_problem'):
            report.update(stop_reason=status, hard_infeasibility_certified=False,
                          convergence_certified=False, residuals_unresolved=True)


def complete_original_clearance(objective, poses, provider, conflicts, optimize, callback=None):
    """Add newly proved local-box conflicts to acquisition, never to a new reference.

    Initial reference probes cannot predict every overlap a contact-acquiring
    step may introduce. Check again before freezing contacts. Every extra pass
    adds at least one collider to a monotonically growing set, so a stalled
    solve with an unchanged witness set cannot request more iterations.
    """
    selected = {row['geom_id'] for row in conflicts}
    history = []
    conflicts = list(conflicts)
    while True:
        provider(poses)
        geometry = provider.geometry
        found = unreachable_geoms(objective.model, poses, provider.last_measurement,
                                  angle=LOCAL_ANGLE, slide=FREE_DRIFT,
                                  closed_static=geometry._static_bound_reuse)
        added = {row['geom_id'] for row in found} - selected
        if not added:
            return poses, history, conflicts
        selected.update(added)
        conflicts += [dict(row, detected='before_contact_freeze') for row in found if row['geom_id'] in added]
        objective.stop_workers()
        objective.original_clearance = OriginalClearance(geometry, selected)
        objective.start_workers()
        poses, more = optimize(objective, poses, iterations=objective.cfg['iterations'],
                               callback=callback, label='contact_original_meshes')
        history += more
