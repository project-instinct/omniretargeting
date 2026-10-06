"""Speed-first sequence backend: source-shaped reference, then contacts/scene.

The objective terms and their sigmas are those of the guarded solver. Contact
acquisition preserves fixed source-shape and temporal bounds, with constrained
fallback directions when a banded step cannot remain feasible. Initialization
and final absolute root/joint derivatives are audited separately; remaining
exceedances are reported. Scene clearance uses precomputed grids of
the original scene/objects; the original triangles are audited separately.
No new user configuration: constants below are internal numerical choices.
"""

import time
from contextlib import ExitStack, nullcontext
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.spatial.transform import Rotation

from .._sequence.orientation_seed import source_orientation_seed
from .._sequence.continuity_guard import SequenceContinuityGuard
from .._sequence.proximity import SurfaceProximityTasks
from ..motion_budgets import CHATTER_RMS_FRACTION, CHATTER_WINDOW, windowed_budget
from ..sequence_retargeting import prepare_sequence, sequence_config
from ..surface_contact import SurfaceContactTasks
from .runtime_options import OPTIONS
from .contact_guard import ContactGuard
from .acquisition_guard import AcquisitionNormalGuard
from .continuity import (
    CorrectionContinuity, audit_continuity, chatter_reports, correction_derivatives, derivatives,
)
from .mesh_repair import repair_original_meshes
from .objective import FastObjective
from .performance_trace import PerformanceTrace, explain_guard
from .reference import ReferenceShape, kinematic_tracking_tasks, register_height
from .sdf import object_sdf, static_scene_sdf
from .solver import optimize, threadpool_limits

# Grid proxies of the original geometry (metres).
SCENE_GRID_SPACING = 0.02
OBJECT_GRID_SPACING = 0.01
SCENE_GRID_MARGIN = 0.8
OBJECT_GRID_MARGIN = 0.15
# Source-pose pass keeps observed contacts loose, as the guarded solver does.
SOURCE_POSE_CONTACT_SIGMA = 0.05
# Budget verification: per-frame round-off slack of the guarded solver, and
# "added chatter": a derivative above the nearby reference level by more than
# a fraction of that joint's own reference RMS. Then the re-weighting schedule.
ROUND_OFF_SLACK = 2e-7
CHATTER_WINDOW_FRAMES = CHATTER_WINDOW
REWEIGHT_ROUNDS = 1
REWEIGHT_FACTOR = 10.0
REWEIGHT_DECREASE = 1e-2
# The free-body stage is a small correction whose rows are a small part of the
# total cost: it stops on a much smaller relative decrease, within a budget.
FREE_STAGE_ITERATIONS = 15
FREE_STAGE_DECREASE = 1e-4
# Contact/scene joint corrections are measured against the fixed reference and
# must not add chatter, so their temporal rows start this much stiffer than the
# source-pose rows (root rows keep following the source path); verification
# then re-weights only the frames still flagged.
CORRECTION_TEMPORAL_WEIGHT = REWEIGHT_FACTOR**2


def _continuity_layers(initialization, final, correction, acquisition_accepted):
    """Keep missing/failed source initialization visible after guarded stages."""
    def layer(audit):
        names = ('windowed_budget_accepted', 'strict_accepted', 'relaxed_chatter_accepted')
        return dict(audited=audit is not None, **{
            name: None if audit is None else bool(audit[name]) for name in names})
    return dict(source_initialization_vs_kinematic_seed=layer(initialization),
                final_vs_fixed_reference=layer(final),
                repair_vs_acquired=layer(correction),
                acquisition_guard_accepted=bool(acquisition_accepted),
                acquisition_pass_is_not_full_acceptance=True,
                missing_initialization_audit_is_not_a_pass=True)


def _velocity_limits(owner, model, coords, framerate):
    """Per optimized scalar joint: URDF velocity / clips per second."""
    path = getattr(owner, "robot_urdf_path", None)
    if path is None or Path(path).suffix.lower() not in {".urdf", ".xml"}:
        return None
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None
    declared = {
        joint.get("name"): float(joint.find("limit").get("velocity"))
        for joint in root.findall(".//joint")
        if joint.get("name") is not None and joint.find("limit") is not None
        and joint.find("limit").get("velocity") is not None
    }
    result = np.full(len(coords.qi), np.inf)
    found = False
    for column, address in enumerate(coords.qi):
        candidates = np.flatnonzero(model.jnt_qposadr == int(address))
        if len(candidates) != 1:
            continue
        limit = declared.get(model.joint(int(candidates[0])).name)
        if limit is not None:
            result[column] = float(limit) / framerate
            found = True
    return result if found else None


def _surface_tasks(extra_tasks):
    return [t for t in extra_tasks if isinstance(t, SurfaceContactTasks)]


def _scene_grid(query, positions):
    if query is None or getattr(query, "mesh", None) is None:
        return None
    points = np.asarray(positions).reshape(-1, 3)
    lower = points.min(0) - SCENE_GRID_MARGIN
    upper = points.max(0) + SCENE_GRID_MARGIN
    lower[2] = min(lower[2], float(query.mesh.bounds[0][2]) - 0.05)
    return static_scene_sdf(
        query.mesh, lower, upper, SCENE_GRID_SPACING,
        enclosed_free_space=query.enclosed_free_space,
    )


def _object_grids(owner):
    grids = []
    for obj in getattr(owner, "_scene_objects", ()) or ():
        mesh, scales = obj["mesh"], obj.get("scales")
        evidence = obj.get("collision_evidence")
        if scales is not None:
            scales = np.asarray(scales, dtype=float).reshape(-1)
            if (scales.shape != (len(obj["translations"]),)
                    or not np.isfinite(scales).all() or np.any(scales <= 0)):
                raise ValueError("Object scales must be finite, positive and frame-aligned")
            # Build in metric units even when the authored mesh uses e.g. cm.
            # Relative frame scales below reproduce every recorded size exactly.
            maximum = float(scales.max())
            if maximum != 1.0:
                mesh = mesh.copy()
                mesh.apply_scale(maximum)
                if evidence is not None:
                    evidence = dict(evidence, points=np.asarray(evidence["points"]) * maximum,
                                    radius=float(evidence["radius"]) * maximum)
            scales = scales / maximum
            if np.all(scales == 1.0):
                scales = None
        grids.append(
            {
                "sdf": object_sdf(mesh, OBJECT_GRID_SPACING, OBJECT_GRID_MARGIN, evidence),
                "translations": np.asarray(obj["translations"], dtype=float),
                "rotations": np.asarray(obj["rotations"], dtype=float),
                "scales": scales,
            }
        )
    return grids


def verify_budgets(model, coords, fixed, motion, bodies=None, shape=None):
    """Vectorized AGENTS budgets of ``motion`` against the fixed reference.

    The only posture rule is the layout of ``shape`` (``pose_relations``):
    flagged frames flip a clear layout component. Joint changes, relation
    angles and world rotations of ``bodies`` (and of all bodies) are reported
    only; posture has no angular bound.
    """
    qi = coords.qi
    chatter, chatter_frames = chatter_reports(coords, fixed, motion)
    result = {}
    for order in (1, 2, 3):
        before = np.abs(np.diff(fixed[:, qi], n=order, axis=0))
        after = np.abs(np.diff(motion[:, qi], n=order, axis=0))
        correction = np.abs(np.diff(motion[:, qi] - fixed[:, qi], n=order, axis=0))
        strict = after > before + ROUND_OFF_SLACK
        local = maximum_filter1d(before, CHATTER_WINDOW_FRAMES, axis=0, mode="nearest")
        joint_rms = np.sqrt(np.mean(before**2, axis=0))
        derivative_screen = (
            (correction if OPTIONS.correction_budget else after)
            > local + CHATTER_RMS_FRACTION * joint_rms + ROUND_OFF_SLACK
        )
        result[f"order_{order}"] = {
            "correction_rms": float(np.sqrt(np.mean(correction**2))),
            "correction_peak": float(correction.max(initial=0)),
            "strict_violations": int(strict.sum()),
            "derivative_screen_violations": int(derivative_screen.sum()),
            "rms_before": float(np.sqrt(np.mean(before**2))),
            "rms_after": float(np.sqrt(np.mean(after**2))),
            "peak_before": float(before.max(initial=0)),
            "peak_after": float(after.max(initial=0)),
        }
    if coords.root_q is not None:
        # Root derivative statistics remain diagnostic; position chatter is
        # the high-frequency report above, not a derivative-size comparison.
        root = slice(coords.root_q, coords.root_q + 3)
        for order in (2, 3):
            before = np.linalg.norm(np.diff(fixed[:, root], n=order, axis=0), axis=1)
            after = np.linalg.norm(np.diff(motion[:, root], n=order, axis=0), axis=1)
            correction = np.linalg.norm(np.diff(motion[:, root] - fixed[:, root], n=order, axis=0), axis=1)
            local = maximum_filter1d(before, CHATTER_WINDOW_FRAMES, mode="nearest")
            rms = np.sqrt(np.mean(before**2))
            derivative_screen = (
                (correction if OPTIONS.correction_budget else after)
                > local + CHATTER_RMS_FRACTION * rms + ROUND_OFF_SLACK
            )
            result[f"root_order_{order}"] = {
                "correction_rms": float(np.sqrt(np.mean(correction**2))),
                "derivative_screen_violations": int(derivative_screen.sum()),
                "rms_before": float(np.sqrt(np.mean(before**2))),
                "rms_after": float(np.sqrt(np.mean(after**2))),
            }
    if OPTIONS.correction_budget and coords.root_q is not None:
        original = derivatives(coords, fixed)
        executed = derivatives(coords, motion)
        corrections = correction_derivatives(coords, fixed, motion)
        for order in (1, 2, 3):
            name = f'root_rotation_order_{order}'
            if name not in original:
                continue
            before, after, delta = original[name], executed[name], corrections[name]
            bad = np.abs(delta) > windowed_budget(before) + ROUND_OFF_SLACK
            result[name] = dict(legacy_derivative_budget_violations=int(bad.sum()),
                rms_before_per_coordinate=np.sqrt(np.mean(before**2, axis=0)).tolist(),
                rms_after_per_coordinate=np.sqrt(np.mean(after**2, axis=0)).tolist(),
                correction_rms_per_coordinate=np.sqrt(np.mean(delta**2, axis=0)).tolist())
    change = np.abs(motion[:, qi] - fixed[:, qi])
    posture_frames = np.zeros(len(motion), bool)
    import mujoco

    a, b = mujoco.MjData(model), mujoco.MjData(model)
    relative = np.zeros((len(motion), model.nbody - 1, 3, 3))
    for f, (first, second) in enumerate(zip(fixed, motion)):
        a.qpos[:], b.qpos[:] = first, second
        mujoco.mj_kinematics(model, a)
        mujoco.mj_kinematics(model, b)
        relative[f] = np.matmul(
            b.xmat[1:].reshape(-1, 3, 3), a.xmat[1:].reshape(-1, 3, 3).transpose(0, 2, 1)
        )
    rotation = Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude().reshape(len(motion), -1)
    guarded = rotation if bodies is None else rotation[:, np.asarray(bodies) - 1]
    if shape is not None:
        shape.kin.evaluate(motion, with_jacobian=False)
        flips = []
        for envelope in shape.envelopes:
            flip = envelope.flips(shape.kin)
            posture_frames |= flip > envelope.SLACK
            flips.append(float(flip.max(initial=-np.inf)))
        relation = shape.envelopes[0].summary(shape.kin)
        result.update(
            relation_whole_angle_max_rad=relation["whole_angle_max_rad"],
            relation_part_angle_max_rad=relation["part_angle_max_rad"],
            relation_part_angle_landmark=relation["part_angle_landmark"],
            layout_flip_max=max(flips),
        )
    result.update(
        budget_signal="source-relative high-frequency content of joints and root translation",
        source_relative_chatter={
            name: {
                "violations": int(row["violations"].sum()),
                "violation_frames": np.flatnonzero(row["violations"].any(axis=1)).tolist(),
                "rms_excess_per_coordinate": row["rms_per_coordinate"],
                "peak_excess_per_coordinate": row["peak_per_coordinate"],
            }
            for name, row in chatter.items()
        },
        max_joint_change_rad=float(change.max(initial=0)),
        max_body_rotation_change_rad=float(guarded.max(initial=0)),
        max_any_body_rotation_change_rad=float(rotation.max(initial=0)),
        layout_flip_frames=int(posture_frames.sum()),
        chatter_frames=int(chatter_frames.sum()),
    )
    return result, chatter_frames, posture_frames


def smooth_correction(coords, entry, candidate, entry_budgets, candidate_budgets):
    """Independent source-chatter and sustained-layout acceptance."""
    reasons = []
    reports, bad = chatter_reports(coords, entry, candidate)
    if bad.any():
        reasons.append("source-relative high-frequency chatter")
    if "layout_flip_max" in candidate_budgets:
        limit = max(entry_budgets.get("layout_flip_max", 0.0), 0.0) + 1e-7
        if candidate_budgets["layout_flip_max"] > limit:
            reasons.append("layout flip")
    return not reasons, reasons


def _improves(budgets, chatter, posture, checked, *, strict):
    """Re-weighting must flag fewer frames; the clearance level may not add
    any flagged frame or added-chatter entry (jitter precedes clearance)."""
    after, after_chatter, after_posture = checked
    if not strict:
        return np.sum(after_chatter | after_posture) <= np.sum(chatter | posture)
    added = ["source_relative_chatter"]
    return (
        np.sum(after_chatter) <= np.sum(chatter)
        and np.sum(after_posture) <= np.sum(posture)
        and all(
            after[key]["violations"] <= budgets[key]["violations"]
            for key in added
        )
    )


def solve_fast_sequence(
    owner, model, initial, positions, names, mapping, roots, tasks, priority,
    config, common, extra_tasks, reference_trajectory=None, callback=None,
):
    """Run one fast solve without nested BLAS teams competing with workers.

    Small dense products and banded factors are faster on one BLAS thread;
    frame workers provide the parallelism. Restore caller settings on exit.
    """
    context = nullcontext() if threadpool_limits is None else threadpool_limits(1, user_api="blas")
    with context, ExitStack() as resources:
        return _solve_fast_sequence(
            owner, model, initial, positions, names, mapping, roots, tasks, priority,
            config, common, extra_tasks, reference_trajectory, callback, resources=resources,
        )


def prepare_fast_source(*args):
    """Internal source-only entry, independent of full-pipeline capture hooks."""
    context = nullcontext() if threadpool_limits is None else threadpool_limits(1,user_api='blas')
    with context, ExitStack() as resources:
        return _solve_fast_sequence(*args,resources=resources,_source_only=True)


def _solve_fast_sequence(
    owner, model, initial, positions, names, mapping, roots, tasks, priority,
    config, common, extra_tasks, reference_trajectory=None, callback=None, *, resources=None, _source_only=False,
):
    from .runtime_options import OPTIONS
    started = time.monotonic()
    report = {"backend": "fast", "stages": {}, "runtime_options": OPTIONS.describe()}
    from .gpu_conic import reset_report, report as gpu_report
    reset_report()
    trace = PerformanceTrace(OPTIONS.performance_trace)
    from .._core.preprocess_cache import SourceGraphCache
    orientation_targets = common["body_orientation_targets"]
    shared = {
        "source_graph_cache": SourceGraphCache(),
        "environment_points": common["environment_points"],
        "root_orientation_targets": common["root_orientation_targets"],
        "body_orientation_targets": orientation_targets,
    }
    framerate = common["framerate"]
    surfaces = _surface_tasks(extra_tasks)
    shape = None
    initialization_continuity = None
    seed_envelope = None
    seed_audits = {}
    if OPTIONS.seed_continuity:
        report['initialization_seed_envelope'] = dict(
            enabled=True, reference_rebased=False, audits=seed_audits,
            status='pending' if reference_trajectory is None else 'unknown_reused_reference_seed_unavailable')

    from .._core.source_pose_prefetch import source_key
    prefetch = None if _source_only else getattr(owner,'_source_pose_prefetch',None)
    source_signature = (source_key(initial,positions,names,mapping,roots,tasks,priority,config,common)
                        if _source_only or prefetch is not None else None)
    callbacks = []
    if _source_only:
        callback = lambda pose, record: callbacks.append((np.array(pose,copy=True),dict(record)))
    ready = None if prefetch is None else prefetch.consume(source_signature,callback)
    # A: fit the source action and its permanent world/body-relative tracking
    # semantics before object contacts or clearance can choose the reference.
    tick = time.monotonic()
    loose = sequence_config(
        dict(config, self_collision_enabled=False,
             contact_sigma_m=SOURCE_POSE_CONTACT_SIGMA,
             contact_normal_sigma_m=SOURCE_POSE_CONTACT_SIGMA,
             contact_tangent_sigma_m=SOURCE_POSE_CONTACT_SIGMA)
    )
    if ready is not None:
        source_pose, shape, coords = ready['pose'],ready['shape'],ready['coords']
        initialization_continuity = ready['continuity']
        shared['source_graph_cache'] = ready['graph']
        registration = ready['registration']
        report['stages']['source_pose'] = ready['stage']
        trace.add_stage_seconds('source_pose',ready['stage']['seconds'])
    elif reference_trajectory is None:
        orientation_seed = source_orientation_seed(model, initial, roots, priority, orientation_targets, config)
        # Preserve the unnormalized seed before prepare_sequence can clip it.
        seed_reference = orientation_seed.copy() if OPTIONS.seed_continuity else orientation_seed
        source_tasks = kinematic_tracking_tasks(tasks, priority)
        pose_config = dict(loose, lock_root_xy=True)
        coords, q, task_set, rotations = prepare_sequence(
            model, orientation_seed, positions, names, mapping, roots, source_tasks, priority,
            pose_config, **shared,
        )
        if OPTIONS.seed_continuity:
            from .initialization_envelope import SeedInitialization
            seed_envelope = SeedInitialization(coords, seed_reference)
            seed_audits['source_prepare'] = seed_envelope.check_normalized(q, stage='source_prepare')
        pose_objective = FastObjective(
            task_set, pose_config, reference=q, temporal_reference=None,
            root_rotations=rotations, framerate=framerate,
        )
        restore_pose_trace = trace.install(pose_objective)
        if seed_envelope is not None:
            seed_envelope.attach_source_objective(pose_objective)
        with trace.stage("source_pose"):
            source_pose, history = optimize(
                pose_objective, q, iterations=loose["iterations"], callback=callback, label="source_pose",
            )
        if seed_envelope is not None:
            seed_audits['fit_vs_seed'] = seed_envelope.check_normalized(source_pose, stage='fit_vs_seed')
        # Finish morphology registration against the same source tracking
        # targets. Object reach must not lift the kinematic reference off its
        # world trajectory. All actual contacts enter stage B, unchanged.
        coords, q, task_set, rotations = prepare_sequence(
            model, source_pose, positions, names, mapping, roots, source_tasks, priority,
            loose, **shared,
        )
        if seed_envelope is not None:
            seed_audits['height_prepare'] = seed_envelope.check_normalized(q, stage='height_prepare')
        registration = FastObjective(
            task_set, loose, reference=q, temporal_reference=None,
            root_rotations=rotations, framerate=framerate,
        )
        restore_registration_trace = trace.install(registration)
        if seed_envelope is None:
            with trace.stage("source_pose"):
                source_pose, height_shift = register_height(registration, q)
        else:
            with trace.stage("source_pose"):
                source_pose, height_shift, height_report = seed_envelope.bounded_height_registration(registration, q)
            report['initialization_seed_envelope']['height_registration'] = height_report
            seed_audits['postheight_vs_seed'] = seed_envelope.check_normalized(source_pose, stage='postheight_vs_seed')
        initialization_continuity = (audit_continuity(coords, seed_reference, source_pose)
                                    if seed_envelope is None else
                                    seed_envelope.audit(source_pose, stage='postheight_vs_seed'))
        shape = ReferenceShape(registration, source_pose)
        report["stages"]["source_pose"] = {
            "history": history, "seconds": time.monotonic() - tick,
            "final": pose_objective.stats,
            "height_registration": {
                "robot_only": True,
                "min_shift_m": float(height_shift.min(initial=0)),
                "max_shift_m": float(height_shift.max(initial=0)),
                "angles_and_xy_unchanged": True,
                "targets": "source_world_and_body_relative_tracking",
                "contact_targets_used": False,
            },
            "source_shape": shape.diagnostics(source_pose),
            "budgets_vs_kinematic_seed": verify_budgets(
                model, coords, seed_reference, source_pose, registration.envelope_bodies
            )[0],
            "continuity_vs_kinematic_seed": initialization_continuity,
        }
        trace.add_stage_seconds(
            "source_pose",
            max(0.0, time.monotonic() - tick - trace.stage_seconds["source_pose"]),
        )
        restore_pose_trace()
        restore_registration_trace()
    else:
        source_pose = np.asarray(reference_trajectory, dtype=float).copy()

    if _source_only:
        from .._core.source_pose_prefetch import compact_source_transport, source_registration_view
        compact_source_transport(shape,registration)
        return dict(key=source_signature, pose=source_pose, shape=shape, coords=coords,
                    continuity=initialization_continuity, graph=shared['source_graph_cache'],
                    stage=report['stages']['source_pose'],callbacks=callbacks,
                    registration=source_registration_view(registration,shape))

    # B: contacts, whole-body surfaces and scene grids from the fixed reference.
    tick = time.monotonic()
    # Self-collision rows are part of acquisition (``terms._self_collision_terms``).
    cfg = sequence_config(dict(config))
    proximity = [t for t in extra_tasks if isinstance(t, SurfaceProximityTasks)]
    coords, q, task_set, rotations = prepare_sequence(
        model, source_pose, positions, names, mapping, roots, tasks, priority, cfg,
        extra_tasks=tuple(surfaces), **shared,
    )
    if seed_envelope is not None:
        seed_audits['acquisition_prepare'] = seed_envelope.check_normalized(q, stage='acquisition_prepare')
    grid_tick = time.monotonic()
    scene_grid = _scene_grid(common.get("scene_query"), positions)
    objects = _object_grids(owner)
    grid_seconds = time.monotonic() - grid_tick
    fixed = q.copy()
    objective = FastObjective(
        task_set, cfg, reference=fixed, temporal_reference=fixed[:, coords.qi],
        root_rotations=rotations, framerate=framerate, scene_sdf=scene_grid,
        objects=objects, surface_tasks=surfaces, proximity_tasks=proximity,
        role_policy=task_set.contact_phase,
    )
    restore_objective_trace = trace.install(objective)
    trace.current_stage = "grid_setup"
    # Supports (penetration.py stage 1): a floating base sitting inside an
    # upward support at several collision points (a seat) cannot leave it by
    # any joint, and acquisition keeps the root within its translation ball
    # around this reference. Lift the reference smoothly before it is frozen;
    # joint angles, root orientation and the horizontal path are unchanged.
    lift = objective.reference_lift()
    if seed_envelope is not None:
        lift, lift_report = seed_envelope.bounded_reference_lift(fixed, lift)
        report['initialization_seed_envelope']['reference_lift'] = lift_report
    if lift.any():
        objective.apply_reference_lift(lift)  # ``fixed`` in place
        q[:, coords.root_q + 2] += lift
        if shape is not None:
            shape = ReferenceShape(registration, fixed)
    if seed_envelope is not None:
        seed_audits['postlift_vs_seed'] = seed_envelope.check_normalized(fixed, stage='postlift_vs_seed')
        # Overall initialization acceptance includes lift, not just source fit.
        initialization_continuity = seed_envelope.audit(fixed, stage='postlift_vs_seed')
        report['initialization_seed_envelope']['status'] = 'checked_against_original_seed'
    if shape is None:
        shape = ReferenceShape(objective, fixed)
    shape.freeze_reference(fixed)
    shape.kin.cache = objective.kin.cache
    objective.source_shape = shape
    from .frozen_depth import FrozenDepth
    from .local_reach import unreachable_geoms
    from .original_clearance import OriginalClearance
    from .repair_limits import LOCAL_ANGLE, FREE_DRIFT

    depth_provider = None
    local_conflicts = []
    if (OPTIONS.frozen_depth or OPTIONS.original_clearance) and not (OPTIONS.priority_clearance or OPTIONS.windowed_priority):
        depth_provider = FrozenDepth(objective, getattr(owner, "_scene_objects", ()), common.get("scene_query"))
        if OPTIONS.frozen_depth:
            objective.frozen_original_depth = depth_provider
        depth_provider(fixed)
        if OPTIONS.original_clearance:
            local_conflicts = unreachable_geoms(model, fixed, depth_provider.last_measurement,
                                                angle=LOCAL_ANGLE, slide=FREE_DRIFT,
                                                closed_static=depth_provider.geometry._static_bound_reuse)
            if local_conflicts:
                objective.original_clearance = OriginalClearance(
                    depth_provider.geometry, [r["geom_id"] for r in local_conflicts])
    coordinator = None
    if OPTIONS.priority_clearance or OPTIONS.windowed_priority:
        from .priority_geometry import acquisition_coordinator
        coordinator = acquisition_coordinator(objective, fixed, getattr(owner, '_scene_objects', ()),
                                              common.get('scene_query'), resources)
    joint = None
    if OPTIONS.joint_witness and len(q) < 600:
        from .joint_witness import JointWitness
        joint = objective.joint_witness = JointWitness(
            objective, fixed, getattr(owner, '_scene_objects', ()), common.get('scene_query'))
    acquisition_guard = AcquisitionNormalGuard(objective, fixed)
    frozen_acquisition_guard = acquisition_guard
    acquisition_continuity = SequenceContinuityGuard(task_set, fixed, rotations, windowed=True,
                                                     correction=OPTIONS.correction_budget,
                                                     chatter=not OPTIONS.reference_derivative_budget,
                                                     enforce_chatter=False,
                                                     velocity_limits=_velocity_limits(
                                                         owner, model, coords, framerate
                                                     ))
    fixed_continuity = CorrectionContinuity(coords, fixed, root_norm=OPTIONS.batch2,
                                            chatter=not OPTIONS.reference_derivative_budget,
                                            enforce_chatter=False)
    objective.candidate_guards = [acquisition_guard, acquisition_continuity,
                                  fixed_continuity]
    if coordinator is not None:
        objective.candidate_guards = [acquisition_continuity, fixed_continuity, coordinator]
    if joint is not None:
        objective.candidate_guards.append(joint)
    # The acquisition guard also enters the objective as stiff one-sided rows,
    # so plain steps rarely need the costly constrained fallback.
    objective.hinge_guards = [acquisition_guard] if coordinator is None else []

    objective.correction_temporal_scale = CORRECTION_TEMPORAL_WEIGHT
    if OPTIONS.spline_correction:
        from .spline_pipeline import run_spline_pipeline
        trace.add_stage_seconds('grid_setup', time.monotonic() - tick)
        return run_spline_pipeline(
            owner, objective, q, getattr(owner, '_scene_objects', ()), common.get('scene_query'),
            resources, started=started, report=report, trace=trace, source_pose=source_pose,
            initialization_continuity=initialization_continuity,
            acquisition_guard=acquisition_guard, acquisition_continuity=acquisition_continuity,
            fixed_continuity=fixed_continuity, callback=callback, extra_tasks=extra_tasks,
            surfaces=surfaces, proximity=proximity, grid_seconds=grid_seconds,
            restore_trace=restore_objective_trace, report_lift=float(lift.max(initial=0.)))
    rounds, penetration = [], []
    # Height registration and the support lift have finished. No collision
    # pass may move this reference to gain a new angular or translation allowance.
    report_lift = float(lift.max(initial=0.0))
    # One worker pool serves every pass: re-weighting only changes weights
    # the parent owns or sends with each evaluation.
    coarse_report = soft_report = None
    soft_v4_used = False
    if OPTIONS.coarse_acquisition or OPTIONS.soft_acquisition or OPTIONS.soft_acquisition_v4 or OPTIONS.soft_acquisition_v5:
        # Install exact cached signals without starting a full-rate pool
        # while the coarse problem and its workers are resident.
        from .guard_signals import prepare
        prepare(objective)
    elif not OPTIONS.windowed_priority:
        objective.start_workers()
    trace.add_stage_seconds("grid_setup", time.monotonic() - tick)
    trace.add_stage_detail("grid_setup", "sdf_grid_seconds", grid_seconds)
    trace.add_stage_detail(
        "grid_setup", "prepare_objective_and_worker_seconds",
        max(0.0, trace.stage_seconds["grid_setup"] - grid_seconds))
    try:
        from .legacy_window import MIN_WINDOW_FRAMES, optimize_legacy_windows, unsupported_reason
        # Parallel time windows for long clips; short clips and unsupported
        # combinations (priority coordinator, experimental proposal paths,
        # infeasible entry) keep the whole-trajectory optimizer and record why.
        acquisition_tick = time.monotonic()
        trace.current_stage = "contact_acquisition"
        window_status = unsupported_reason(objective, q, min_frames=MIN_WINDOW_FRAMES) or "used"
        if OPTIONS.coarse_acquisition:
            from .coarse_acquisition import optimize_coarse_acquisition
            motion, history, coarse_report = optimize_coarse_acquisition(
                objective, q, iterations=cfg["iterations"], callback=callback,
                window_status=window_status)
        elif OPTIONS.soft_acquisition_v5 and window_status == "used":
            from .soft_acquisition_v5 import optimize_soft_acquisition_v5
            motion, history, soft_report = optimize_soft_acquisition_v5(
                objective, q, iterations=cfg["iterations"], callback=callback)
            acquisition_guard = objective.soft_acquired_normal_guard
            soft_v4_used = True
        elif OPTIONS.soft_acquisition_v4 and window_status == "used":
            from .soft_acquisition_v4 import optimize_soft_acquisition_v4
            motion, history, soft_report = optimize_soft_acquisition_v4(
                objective, q, iterations=cfg["iterations"], callback=callback)
            acquisition_guard = objective.soft_acquired_normal_guard
            soft_v4_used = True
        elif OPTIONS.soft_acquisition and window_status == "used":
            from .soft_acquisition import optimize_soft_acquisition
            motion, history, soft_report = optimize_soft_acquisition(
                objective, q, iterations=cfg["iterations"], callback=callback)
        elif window_status == "used":
            motion, history = optimize_legacy_windows(objective, q, iterations=cfg["iterations"],
                                                      callback=callback, label="contact_scene")
        else:
            motion, history = optimize(objective, q, iterations=cfg["iterations"],
                                       callback=callback, label="contact_scene")
        if joint is not None:
            report['joint_witness'] = joint.diagnostics(motion)
            joint.close()
            objective.joint_witness = None
            # Downstream passes retain the original acquisition guard/hinges.
            acquisition_guard = AcquisitionNormalGuard(objective, fixed)
            frozen_acquisition_guard = acquisition_guard
            objective.candidate_guards = [acquisition_guard, acquisition_continuity, fixed_continuity]
            objective.hinge_guards = [acquisition_guard]
        verified = verify_budgets(model, coords, fixed, motion, objective.envelope_bodies, shape)
        trace.add_stage_seconds("contact_acquisition", time.monotonic() - acquisition_tick)
        if window_status == "used":
            trace.absorb_history(history)
        trace.current_stage = None
        for round_index in range(REWEIGHT_ROUNDS if coordinator is None else 0):
            budgets, chatter, posture = verified
            rounds.append(budgets)
            if not chatter.any() and not posture.any():
                break
            reweight_tick = time.monotonic()
            candidate_tick = time.monotonic()
            trace.current_stage = "reweight"
            from .reweight import reweight_temporal_signals

            if OPTIONS.batch2:
                rounds[-1]['reweighted_signals'] = reweight_temporal_signals(
                    objective, fixed, motion, REWEIGHT_FACTOR)
            else:
                for op_index, weight in enumerate(objective.temporal_weight):
                    touched = np.convolve(chatter.astype(float), np.ones(op_index + 2), "valid") > 0
                    weight[touched[:len(weight)]] *= REWEIGHT_FACTOR
            objective.envelope_weight[posture] *= REWEIGHT_FACTOR
            label = f"reweight_{round_index + 1}"
            if window_status == "used":
                # Only the windows around flagged frames move; re-solving the
                # whole long clip for a few frames costs minutes.
                candidate, more = optimize_legacy_windows(
                    objective, motion, iterations=cfg["iterations"], callback=callback,
                    label=label, focus=chatter | posture)
            else:
                # ``loose_reweight``: the re-weighted pass makes >90% of its cost
                # decrease in its first four steps; end its tail at a 1% step.
                candidate, more = optimize(objective, motion, iterations=cfg["iterations"],
                                           callback=callback, label=label,
                                           **({"converged_decrease": REWEIGHT_DECREASE}
                                              if OPTIONS.loose_reweight else {}))
            history += more
            checked = verify_budgets(model, coords, fixed, candidate, objective.envelope_bodies, shape)
            trace.add_stage_seconds("reweight", time.monotonic() - reweight_tick)
            if window_status == "used":
                trace.absorb_history(more)
            trace.current_stage = None
            if not _improves(budgets, chatter, posture, checked, strict=False):
                rounds[-1]["rejected"] = label
                rounds[-1]["rejected_budgets"] = checked[0]
                trace.add_failed(time.monotonic() - candidate_tick, "reweight_rejected")
                continue
            motion, verified = candidate, checked
            trace.add_alpha(1.0)
        # Free-body penetration: every contact held, a small correction of the
        # accepted motion, kept only if it adds no chatter or layout flip
        # (jitter and layout precede clearance). See penetration.py.
        # The acquisition workers still hold the same immutable fields and
        # shells: they measure the deepest samples below, before the free
        # stage changes what the objective evaluates.
        free_tick = time.monotonic()
        trace.current_stage = "free_bodies"
        if coordinator is not None:
            from .acquisition_clearance import complete_priority_clearance
            motion, more, completion = complete_priority_clearance(objective, motion, optimize, callback)
            history += more
            penetration.append(completion)
            verified = verify_budgets(model, coords, fixed, motion, objective.envelope_bodies, shape)
            acquired = motion.copy()
            contact_guard = ContactGuard(objective, acquired)
            continuity_guard = CorrectionContinuity(
                coords, acquired, chatter=not OPTIONS.reference_derivative_budget,
                enforce_chatter=False,
            )
        else:
            from .acquisition_clearance import complete_original_clearance
            if OPTIONS.original_clearance:
                motion, more, local_conflicts = complete_original_clearance(
                    objective, motion, depth_provider, local_conflicts, optimize, callback)
                history += more
            verified = verify_budgets(model, coords, fixed, motion, objective.envelope_bodies, shape)
            acquired = motion.copy()
            contact_guard = ContactGuard(objective, acquired)
            anchored_contacts = ContactGuard(objective, acquired, preserve_positions=True)
            continuity_guard = CorrectionContinuity(
                coords, acquired, chatter=not OPTIONS.reference_derivative_budget,
                enforce_chatter=False,
            )
            objective.candidate_guards = [anchored_contacts, continuity_guard,
                                          acquisition_guard, acquisition_continuity] + (
                                          [fixed_continuity] if OPTIONS.correction_budget or soft_v4_used else [])
            if coords.root_q is not None:
                objective.fixed_root = acquired[:, coords.root_q:coords.root_q + 7].copy()
            record = {"stage": "free_bodies", "before": objective.penetration_summary(motion)}
            if OPTIONS.joint_skip_free and joint is not None:
                record.update(skipped=True, reason='joint_skip_free', iterations=0,
                              accepted=False, after=record['before'])
            else:
                candidate_tick = time.monotonic()
                objective.prepare_free_stage(motion)
                objective.envelope_weight = np.maximum(objective.envelope_weight, REWEIGHT_FACTOR)
                objective.stop_workers()
                objective.start_workers()
                candidate, more = optimize(
                    objective, motion, iterations=min(cfg["iterations"], FREE_STAGE_ITERATIONS),
                    callback=callback, label="free_bodies", converged_decrease=FREE_STAGE_DECREASE,
                )
                history += more
                checked = verify_budgets(model, coords, fixed, candidate, objective.envelope_bodies, shape)
                record["candidate"] = objective.penetration_summary(candidate)
                accepted, reasons = smooth_correction(coords, motion, candidate, verified[0], checked[0])
                for name, guard in (("acquired contact positions and precision", anchored_contacts),
                                    ("acquired per-coordinate temporal bounds", continuity_guard)):
                    guard_tick = time.monotonic()
                    guard_accepted = guard.accepts(candidate)
                    explanation = (None if guard_accepted else
                                   explain_guard(objective, guard, candidate))
                    trace.add_guard(guard, guard_accepted,
                                    time.monotonic() - guard_tick, explanation)
                    if not guard_accepted:
                        accepted = False
                        reasons.append(name)
                record["accepted"] = bool(accepted)
                if accepted:
                    motion, verified = candidate, checked
                    trace.add_alpha(1.0)
                else:
                    record["rejected"] = reasons
                    record["rejected_budgets"] = checked[0]
                    trace.add_failed(time.monotonic() - candidate_tick, "free_stage_rejected")
            penetration.append(record)
        trace.add_stage_seconds("free_bodies", time.monotonic() - free_tick)
        trace.current_stage = None
    finally:
        objective.stop_workers()
        if joint is not None:
            joint.close()
    # The selected repair group may approach its original target; all other
    # groups remain anchored by RepairLimits. Precision still uses acquisition.
    objective.candidate_guards = [contact_guard, continuity_guard,
                                  acquisition_guard, acquisition_continuity] + (
                                      [fixed_continuity] if OPTIONS.correction_budget or soft_v4_used else [])
    if coordinator is not None:
        objective.candidate_guards = [acquisition_continuity, fixed_continuity, continuity_guard, coordinator]
    mesh_repair = None
    grid_coverage = [float(np.isfinite(obj["sdf"].clearance).mean()) for obj in objects]
    # Original-geometry repair is not an open-object fallback. A watertight
    # object, a floor-only scene, or self-collision needs the same bounded
    # correction and audit; grid coverage says nothing about those overlaps.
    if common.get("scene_query") is not None or objects or config.get("self_collision_enabled", True):
        repair_tick = time.monotonic()
        trace.current_stage = "original_mesh_repair"
        motion, mesh_repair = repair_original_meshes(
            objective, motion, getattr(owner, "_scene_objects", ()), common.get("scene_query"),
        )
        verified = verify_budgets(model, coords, fixed, motion, objective.envelope_bodies, shape)
        trace.absorb_repair(mesh_repair)
        trace.add_stage_seconds("original_mesh_repair", time.monotonic() - repair_tick)
        trace.current_stage = None
    # Final statistics; worker chunks give the same values as a serial pass.
    audit_tick = time.monotonic()
    trace.current_stage = "final_audit"
    started_workers = objective.start_workers()
    try:
        objective.evaluate(motion, with_jacobian=False)
    finally:
        if started_workers:
            objective.stop_workers()
    if OPTIONS.windowed_priority and coordinator is not None:
        objective.stats.update(coordinator.collision_diagnostics(motion))
    budgets = verified[0]
    final_continuity = audit_continuity(coords, fixed, motion)
    correction_continuity = audit_continuity(coords, acquired, motion)
    report['continuity_acceptance_layers'] = _continuity_layers(
        initialization_continuity, final_continuity, correction_continuity,
        acquisition_continuity.accepts(motion))
    report["stages"]["contact_scene"] = {
        "history": history, "seconds": time.monotonic() - tick, "grid_seconds": grid_seconds,
        "acquisition_windows": window_status,
        "final": objective.stats, "reweight_rounds": rounds, "budgets": budgets,
        "penetration_stages": penetration,
        "original_mesh_repair": mesh_repair,
        "frozen_original_depth": [] if depth_provider is None else depth_provider.audits,
        "local_repair_reach_conflicts": local_conflicts,
        "continuity_vs_fixed_reference": final_continuity,
        "continuity_vs_acquired": correction_continuity,
        "contacts_vs_acquired": contact_guard.diagnostics(motion),
        "protected_normals_vs_fixed_reference": frozen_acquisition_guard.diagnostics(motion),
        "downstream_contact_regression": acquisition_guard.diagnostics(motion),
        "acquisition_continuity_accepted": acquisition_continuity.accepts(motion),
        "object_grid_finite_fraction": grid_coverage,
        "reference_lift_max_m": report_lift,
        "reference_lift_frames": int((lift > 0).sum()),
        "skipped_tasks": [
            type(t).__name__ for t in extra_tasks
            if t is not None and t not in surfaces and t not in proximity
        ],
        "self_collision": ("bounded original-mesh repair; see original_mesh_repair" if mesh_repair is not None
                           else "not optimized in the fast backend; audited separately"),
    }
    if joint is not None:
        from .postprocess_workload import workload
        report['postprocess_workload'] = workload(report)
    if coarse_report is not None:
        report["stages"]["contact_scene"]["coarse_acquisition"] = coarse_report
    if soft_report is not None:
        if soft_v4_used:
            soft_report['final_required_guards'] = dict(
                sequence_continuity=acquisition_continuity.accepts(motion),
                correction_continuity=fixed_continuity.accepts(motion),
                layout=shape.accepts(motion))
            if not all(soft_report['final_required_guards'].values()):
                raise RuntimeError('Soft-v4 final motion lost a required frozen guard')
        report["stages"]["contact_scene"]["soft_acquisition"] = soft_report
    if coordinator is not None:
        priority_history = history + ([] if mesh_repair is None else mesh_repair.get('history', []))
        priority_report = coordinator.diagnostics(motion, solver_history=priority_history)
        report["stages"]["contact_scene"]["priority_clearance"] = priority_report
        report.update({name: priority_report[name] for name in
                       ('accepted_update_count', 'solver_outcome', 'convergence_certified')})
    total_seconds = time.monotonic() - started
    report.update(
        status="solved", frames=len(motion), framerate=float(framerate),
        model_nq=model.nq, model_nv=model.nv, scalar_joint_count=len(coords.qi),
        history=history, final_residuals=objective.stats,
        contact_offset_choice={k: np.asarray(v).tolist() for k, v in task_set.choices.items()},
        seconds=total_seconds,
    )
    if shape is not None:
        report["source_shape"] = shape.diagnostics(motion)
        # A relaxed chatter screen is not a strict per-step certificate. Do
        # not erase initialization changes by only comparing to the new seed.
        layout_ok = (report["source_shape"]["layout_preserved"]
                     and budgets["layout_flip_frames"] == 0)
        report["strict_continuity_accepted"] = bool(
            final_continuity["strict_accepted"] and initialization_continuity is not None
            and initialization_continuity["strict_accepted"])
        # The orientation seed is not the source action. Keep its audit for
        # diagnosis, but final continuity is measured against the frozen
        # source-shaped reference used by every clearance stage.
        report["layout_and_relaxed_chatter_accepted"] = bool(
            layout_ok and final_continuity["relaxed_chatter_accepted"]
        )
        # The enforced per-joint budgets are windowed (motion_budgets); the
        # stencil-exact comparison above stays an informational report.
        report["windowed_continuity_accepted"] = bool(
            final_continuity["windowed_budget_accepted"] and initialization_continuity is not None
            and initialization_continuity["windowed_budget_accepted"])
        # The enforced production criterion is source-relative chatter. The
        # old derivative budget stays visible as a legacy diagnostic layer.
        report["source_chatter_continuity_accepted"] = report[
            "layout_and_relaxed_chatter_accepted"
        ]
        report["layout_continuity_accepted"] = bool(
            layout_ok and report["source_chatter_continuity_accepted"]
        )
        report["acceptance_status"] = (
            "passed" if report["layout_continuity_accepted"]
            else "unresolved_layout_or_continuity"
        )
    total_seconds = time.monotonic() - started
    report["seconds"] = total_seconds
    trace.add_stage_seconds("final_audit", time.monotonic() - audit_tick)
    trace.current_stage = None
    restore_objective_trace()
    report["performance"] = trace.report(total_seconds=total_seconds)
    report["gpu_conic"] = gpu_report()
    owner.last_fast_source_pose = source_pose
    return motion, report
