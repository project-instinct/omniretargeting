"""Independent opt-in correction pipeline; production initialization is shared."""
import time

import numpy as np

from .contact_guard import ContactGuard
from .continuity import CorrectionContinuity, audit_continuity
from .runtime_options import OPTIONS
from .spline_geometry import SplineGeometry
from .spline_solver import SplineSolve


def local_windows(spline, focus):
    """Closure includes cubic support, third differences and real Gaussian kernel."""
    from ..motion_budgets import CHATTER_SCALE_FRAMES
    halo = int(np.ceil(4 * CHATTER_SCALE_FRAMES)) + 3
    # A changed control affects four knot spans; include those observed frames.
    halo += int(np.ceil(4 * spline.spacing * spline.framerate))
    marked = np.flatnonzero(focus)
    windows = []
    for frame in marked:
        lo, hi = max(0, frame - halo), min(spline.count, frame + halo + 1)
        if windows and lo <= windows[-1][1]:
            windows[-1] = windows[-1][0], max(hi, windows[-1][1])
        else:
            windows.append((lo, hi))
    return windows


def local_controls(spline, start, stop):
    """Exterior C2 matching; clip endpoints have no exterior to constrain."""
    if spline.identity:
        return spline.local_controls(start, stop)
    support = spline.support_times
    left, right = spline.times[start], spline.times[stop - 1]
    inside = (support[:, 0] >= left) & (support[:, 1] <= right)
    if start:
        inside &= support[:, 0] > 0
    if stop < spline.count:
        inside &= support[:, 1] < spline.end_time
    return np.flatnonzero(inside)


def run_spline_pipeline(owner, objective, q, objects, scene, resources, *, started,
                        report, trace, source_pose, initialization_continuity,
                        acquisition_guard, acquisition_continuity, fixed_continuity,
                        callback, extra_tasks, surfaces, proximity, grid_seconds,
                        restore_trace, report_lift):
    # Imports avoid a module cycle and reuse the production budget definitions.
    from .pipeline import verify_budgets, REWEIGHT_FACTOR, FREE_STAGE_ITERATIONS, FREE_STAGE_DECREASE
    from .reweight import reweight_temporal_signals
    fixed, shape, coords = objective.reference, objective.source_shape, objective.coords
    owner.last_fast_reference = fixed.copy()
    stage_times, rounds, refinements = {}, [], []
    geometry_tick = time.monotonic()
    geometry = SplineGeometry(objective, objects, scene, resources)
    state = SplineSolve(objective, geometry, adaptive=OPTIONS.spline_adaptive,
                        deadline=started + len(q))
    # A penetrating reference does not by itself prove a representation
    # deficit. Its witnesses already participate in the all-frame main
    # problem; add geometry knots only after the capacity diagnostic.
    weight_updater = None
    if OPTIONS.spline_shared_loop:
        from .spline_reweight import SharedReweight
        weight_updater = SharedReweight(objective, fixed, shape, rounds)
    stage_times['geometry_setup'] = time.monotonic() - geometry_tick
    objective.start_workers()
    try:
        tick = time.monotonic()
        trace.current_stage = 'contact_acquisition'
        motion = state.optimize(objective.cfg['iterations'], label='spline_acquisition', callback=callback,
                                weight_updater=weight_updater)
        stage_times['contact_acquisition'] = time.monotonic() - tick
        checked = verify_budgets(objective.model, coords, fixed, motion, objective.envelope_bodies, shape)
        tick = time.monotonic()
        budgets, chatter, posture = checked
        if not OPTIONS.spline_shared_loop and (chatter.any() or posture.any()) and not state.expired():
            record = dict(before=budgets, reweighted_signals=reweight_temporal_signals(
                objective, fixed, motion, REWEIGHT_FACTOR))
            objective.envelope_weight[posture] *= REWEIGHT_FACTOR
            # New costs/gradient are evaluated from scratch at the retained
            # coefficients; an old-weight predicted reduction is never reused.
            state.damping = max(state.damping, 1e-3)
            motion = state.optimize(objective.cfg['iterations'], label='spline_reweight', callback=callback)
            record['after'] = verify_budgets(objective.model, coords, fixed, motion,
                                            objective.envelope_bodies, shape)[0]
            rounds.append(record)
        stage_times['reweight'] = time.monotonic() - tick
        acquired = motion.copy()
        contact_guard = ContactGuard(objective, acquired)
        anchored = ContactGuard(objective, acquired, preserve_positions=True)
        continuity = CorrectionContinuity(coords, acquired,
            chatter=not OPTIONS.reference_derivative_budget, enforce_chatter=False)
        objective.candidate_guards = [anchored, continuity, acquisition_guard,
                                      acquisition_continuity, fixed_continuity]
        if coords.root_q is not None:
            objective.fixed_root = acquired[:, coords.root_q:coords.root_q + 7].copy()
        tick = time.monotonic()
        trace.current_stage = 'free_bodies'
        focus = geometry.focus(len(q))
        if (not OPTIONS.spline_shared_loop or focus.any()) and not state.expired():
            objective.prepare_free_stage(motion)
            objective.envelope_weight = np.maximum(objective.envelope_weight, REWEIGHT_FACTOR)
            # Immutable workers are restarted only if the objective actually
            # changes; reference, controls and geometry cache remain shared.
            objective.stop_workers()
            objective.start_workers()
            if OPTIONS.spline_shared_loop:
                for start, stop in local_windows(state.spline, focus):
                    if state.remaining() <= 0:
                        break
                    controls = local_controls(state.spline, start, stop)
                    before = state.poses
                    motion = state.optimize(min(objective.cfg['iterations'], FREE_STAGE_ITERATIONS),
                        label='spline_local_free', callback=callback, controls=controls,
                        converged_decrease=FREE_STAGE_DECREASE, weight_updater=weight_updater)
                    outside = np.r_[np.arange(start), np.arange(stop, len(q))]
                    if not np.array_equal(before[outside], motion[outside]):
                        raise RuntimeError('Local spline refinement changed exterior frames')
                    refinements.append(dict(start=start, stop=stop, controls=controls.tolist(),
                                             exterior_values_velocity_acceleration_matched=True))
            else:
                motion = state.optimize(min(objective.cfg['iterations'], FREE_STAGE_ITERATIONS),
                                        label='spline_free', callback=callback,
                                        converged_decrease=FREE_STAGE_DECREASE)
        stage_times['free_bodies'] = time.monotonic() - tick
    finally:
        objective.stop_workers()
        trace.current_stage = None
    # No automatic whole-clip repair after acceptance. The main problem already
    # contains original geometry; residual defects are preserved in this report.
    audit_tick = time.monotonic()
    motion = state.poses
    objective.evaluate(motion, with_jacobian=False)
    verified = verify_budgets(objective.model, coords, fixed, motion, objective.envelope_bodies, shape)
    final_continuity = audit_continuity(coords, fixed, motion)
    correction_continuity = audit_continuity(coords, acquired, motion)
    report['stages']['contact_scene'] = dict(
        history=state.history, seconds=time.monotonic() - geometry_tick, grid_seconds=grid_seconds,
        acquisition_windows='all_original_frames_cubic_correction', final=objective.stats,
        reweight_rounds=rounds, budgets=verified[0], penetration_stages=[],
        original_mesh_repair=dict(automatic_post_audit_optimization=False,
                                  geometry_in_main_loop=geometry.report()),
        continuity_vs_fixed_reference=final_continuity, continuity_vs_acquired=correction_continuity,
        contacts_vs_acquired=contact_guard.diagnostics(motion),
        protected_normals_vs_fixed_reference=acquisition_guard.diagnostics(motion),
        acquisition_continuity_accepted=acquisition_continuity.accepts(motion),
        reference_lift_max_m=report_lift,
        skipped_tasks=[type(t).__name__ for t in extra_tasks
                       if t is not None and t not in surfaces and t not in proximity],
        spline=dict(state.report(), stage_seconds=stage_times, local_refinements=refinements,
                    shared_loop=OPTIONS.spline_shared_loop))
    report.update(status='budget_exhausted' if state.report()['budget_exhausted'] else 'experimental',
        frames=len(motion), framerate=float(objective.framerate), model_nq=objective.model.nq,
        model_nv=objective.model.nv, scalar_joint_count=len(coords.qi),
        history=state.history, final_residuals=objective.stats,
        contact_offset_choice={k: np.asarray(v).tolist() for k, v in objective.tasks.choices.items()},
        source_shape=shape.diagnostics(motion),
        acceptance_status='unverified_pending_independent_quality_geometry_physics_reset',
        experimental_path=True, default_changed=False,
        source_chatter_continuity_accepted=final_continuity['relaxed_chatter_accepted'],
        initialization_continuity=initialization_continuity)
    stage_times['final_audit'] = time.monotonic() - audit_tick
    owner.last_fast_source_pose = source_pose
    owner.last_fast_spline = dict(knots=state.spline.knot_vector, coefficients=state.a,
                                 fixed_reference=state.spline.reference)
    for name in ('contact_acquisition', 'reweight', 'free_bodies', 'final_audit'):
        trace.add_stage_seconds(name, stage_times[name])
    restore_trace()
    report['seconds'] = time.monotonic() - started
    report['performance'] = trace.report(total_seconds=report['seconds'])
    from .gpu_conic import report as gpu_report
    report['gpu_conic'] = gpu_report()
    return motion, report
