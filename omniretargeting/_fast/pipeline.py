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
from contextlib import nullcontext

import numpy as np
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.spatial.transform import Rotation

from .._sequence.orientation_seed import source_orientation_seed
from .._sequence.continuity_guard import SequenceContinuityGuard
from .._sequence.proximity import SurfaceProximityTasks
from ..motion_budgets import (
    CHATTER_RMS_FRACTION, CHATTER_SCALE_FRAMES, CHATTER_WINDOW, windowed_budget,
)
from ..sequence_retargeting import prepare_sequence, sequence_config
from ..surface_contact import SurfaceContactTasks
from .runtime_options import OPTIONS
from .contact_guard import ContactGuard
from .acquisition_guard import AcquisitionNormalGuard
from .continuity import CorrectionContinuity, audit_continuity, correction_derivatives, derivatives
from .mesh_repair import repair_original_meshes
from .objective import FastObjective
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
# The free-body stage is a small correction whose rows are a small part of the
# total cost: it stops on a much smaller relative decrease, within a budget.
FREE_STAGE_ITERATIONS = 15
FREE_STAGE_DECREASE = 1e-4
CORRECTION_TOLERANCE = 0.02
# Contact/scene joint corrections are measured against the fixed reference and
# must not add chatter, so their temporal rows start this much stiffer than the
# source-pose rows (root rows keep following the source path); verification
# then re-weights only the frames still flagged.
CORRECTION_TEMPORAL_WEIGHT = REWEIGHT_FACTOR**2


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
    result, chatter_frames = {}, np.zeros(len(motion), bool)
    for order in (1, 2, 3):
        before = np.abs(np.diff(fixed[:, qi], n=order, axis=0))
        after = np.abs(np.diff(motion[:, qi], n=order, axis=0))
        correction = np.abs(np.diff(motion[:, qi] - fixed[:, qi], n=order, axis=0))
        strict = after > before + ROUND_OFF_SLACK
        local = maximum_filter1d(before, CHATTER_WINDOW_FRAMES, axis=0, mode="nearest")
        joint_rms = np.sqrt(np.mean(before**2, axis=0))
        chatter = (correction if OPTIONS.correction_budget else after) > local + CHATTER_RMS_FRACTION * joint_rms + ROUND_OFF_SLACK
        # A smooth correction legitimately changes velocity; chatter shows in
        # acceleration and jerk, so only those orders trigger re-weighting.
        for start in np.flatnonzero(chatter.any(axis=1)) if order > 1 else ():
            chatter_frames[start : start + order + 1] = True
        result[f"order_{order}"] = {
            "correction_rms": float(np.sqrt(np.mean(correction**2))),
            "correction_peak": float(correction.max(initial=0)),
            "strict_violations": int(strict.sum()),
            "added_chatter_violations": int(chatter.sum()),
            "rms_before": float(np.sqrt(np.mean(before**2))),
            "rms_after": float(np.sqrt(np.mean(after**2))),
            "peak_before": float(before.max(initial=0)),
            "peak_after": float(after.max(initial=0)),
        }
    if coords.root_q is not None:
        # Root position chatter, on the same added-chatter criterion.
        root = slice(coords.root_q, coords.root_q + 3)
        for order in (2, 3):
            before = np.linalg.norm(np.diff(fixed[:, root], n=order, axis=0), axis=1)
            after = np.linalg.norm(np.diff(motion[:, root], n=order, axis=0), axis=1)
            correction = np.linalg.norm(np.diff(motion[:, root] - fixed[:, root], n=order, axis=0), axis=1)
            local = maximum_filter1d(before, CHATTER_WINDOW_FRAMES, mode="nearest")
            rms = np.sqrt(np.mean(before**2))
            chatter = (correction if OPTIONS.correction_budget else after) > local + CHATTER_RMS_FRACTION * rms + ROUND_OFF_SLACK
            for start in np.flatnonzero(chatter):
                chatter_frames[start : start + order + 1] = True
            result[f"root_order_{order}"] = {
                "correction_rms": float(np.sqrt(np.mean(correction**2))),
                "added_chatter_violations": int(chatter.sum()),
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
            for start in np.flatnonzero(bad.any(axis=1)) if order > 1 else ():
                chatter_frames[start:start + order + 1] = True
            result[name] = dict(added_chatter_violations=int(bad.sum()),
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
        budget_signal=("correction relative to immutable reference" if OPTIONS.correction_budget
                       else "absolute executed derivatives"),
        max_joint_change_rad=float(change.max(initial=0)),
        max_body_rotation_change_rad=float(guarded.max(initial=0)),
        max_any_body_rotation_change_rad=float(rotation.max(initial=0)),
        layout_flip_frames=int(posture_frames.sum()),
        chatter_frames=int(chatter_frames.sum()),
    )
    return result, chatter_frames, posture_frames


def smooth_correction(coords, entry, candidate, entry_budgets, candidate_budgets):
    """Independent temporal/layout acceptance, using the selected budget semantics."""
    reasons = []
    if OPTIONS.correction_budget:
        if not CorrectionContinuity(coords, entry, root_norm=True).accepts(candidate):
            reasons.append("correction derivatives exceed reference window budgets")
    else:
        signals = [("joint", entry[:, coords.qi], candidate[:, coords.qi])]
        if coords.root_q is not None:
            root = slice(coords.root_q, coords.root_q + 3)
            signals.append(("root", entry[:, root], candidate[:, root]))
        for name, before, after in signals:
            rough = after - before
            rough = rough - gaussian_filter1d(rough, CHATTER_SCALE_FRAMES, axis=0, mode="nearest")
            for order in (2, 3):
                old, new, jitter = (np.diff(x, n=order, axis=0) for x in (before, after, rough))
                if name == "root":
                    old, new, jitter = (np.linalg.norm(x, axis=1, keepdims=True) for x in (old, new, jitter))
                rms = np.sqrt(np.mean(old**2, axis=0))
                allowed = rms * (1 + CORRECTION_TOLERANCE) + ROUND_OFF_SLACK
                if np.any(np.sqrt(np.mean(new**2, axis=0)) > allowed):
                    reasons.append(f"{name} order {order} RMS")
                if np.any(np.abs(new).max(0) > np.abs(old).max(0) * (1 + CORRECTION_TOLERANCE) + ROUND_OFF_SLACK):
                    reasons.append(f"{name} order {order} peak")
                if np.any(np.sqrt(np.mean(jitter**2, axis=0)) > CORRECTION_TOLERANCE * rms + ROUND_OFF_SLACK):
                    reasons.append(f"{name} order {order} added high-frequency")
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
    added = [f"order_{order}" for order in (2, 3)]
    return (
        np.sum(after_chatter) <= np.sum(chatter)
        and np.sum(after_posture) <= np.sum(posture)
        and all(
            after[key]["added_chatter_violations"] <= budgets[key]["added_chatter_violations"]
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
    with context:
        return _solve_fast_sequence(
            owner, model, initial, positions, names, mapping, roots, tasks, priority,
            config, common, extra_tasks, reference_trajectory, callback,
        )


def _solve_fast_sequence(
    owner, model, initial, positions, names, mapping, roots, tasks, priority,
    config, common, extra_tasks, reference_trajectory=None, callback=None,
):
    from .runtime_options import OPTIONS
    started = time.monotonic()
    report = {"backend": "fast", "stages": {}, "runtime_options": OPTIONS.describe()}
    orientation_targets = common["body_orientation_targets"]
    shared = {
        "environment_points": common["environment_points"],
        "root_orientation_targets": common["root_orientation_targets"],
        "body_orientation_targets": orientation_targets,
    }
    framerate = common["framerate"]
    surfaces = _surface_tasks(extra_tasks)
    shape = None
    initialization_continuity = None

    # A: fit the source action and its permanent world/body-relative tracking
    # semantics before object contacts or clearance can choose the reference.
    tick = time.monotonic()
    loose = sequence_config(
        dict(config, self_collision_enabled=False,
             contact_sigma_m=SOURCE_POSE_CONTACT_SIGMA,
             contact_normal_sigma_m=SOURCE_POSE_CONTACT_SIGMA,
             contact_tangent_sigma_m=SOURCE_POSE_CONTACT_SIGMA)
    )
    if reference_trajectory is None:
        orientation_seed = source_orientation_seed(model, initial, roots, priority, orientation_targets, config)
        source_tasks = kinematic_tracking_tasks(tasks, priority)
        pose_config = dict(loose, lock_root_xy=True)
        coords, q, task_set, rotations = prepare_sequence(
            model, orientation_seed, positions, names, mapping, roots, source_tasks, priority,
            pose_config, **shared,
        )
        pose_objective = FastObjective(
            task_set, pose_config, reference=q, temporal_reference=None,
            root_rotations=rotations, framerate=framerate,
        )
        source_pose, history = optimize(
            pose_objective, q, iterations=loose["iterations"], callback=callback, label="source_pose",
        )
        # Finish morphology registration against the same source tracking
        # targets. Object reach must not lift the kinematic reference off its
        # world trajectory. All actual contacts enter stage B, unchanged.
        coords, q, task_set, rotations = prepare_sequence(
            model, source_pose, positions, names, mapping, roots, source_tasks, priority,
            loose, **shared,
        )
        registration = FastObjective(
            task_set, loose, reference=q, temporal_reference=None,
            root_rotations=rotations, framerate=framerate,
        )
        source_pose, height_shift = register_height(registration, q)
        initialization_continuity = audit_continuity(coords, orientation_seed, source_pose)
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
                model, coords, orientation_seed, source_pose, registration.envelope_bodies
            )[0],
            "continuity_vs_kinematic_seed": initialization_continuity,
        }
    else:
        source_pose = np.asarray(reference_trajectory, dtype=float).copy()

    # B: contacts, whole-body surfaces and scene grids from the fixed reference.
    tick = time.monotonic()
    # Self-collision rows are part of acquisition (``terms._self_collision_terms``).
    cfg = sequence_config(dict(config))
    proximity = [t for t in extra_tasks if isinstance(t, SurfaceProximityTasks)]
    coords, q, task_set, rotations = prepare_sequence(
        model, source_pose, positions, names, mapping, roots, tasks, priority, cfg,
        extra_tasks=tuple(surfaces), **shared,
    )
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
    # Supports (penetration.py stage 1): a floating base sitting inside an
    # upward support at several collision points (a seat) cannot leave it by
    # any joint, and acquisition keeps the root within its translation ball
    # around this reference. Lift the reference smoothly before it is frozen;
    # joint angles, root orientation and the horizontal path are unchanged.
    lift = objective.reference_lift()
    if lift.any():
        objective.apply_reference_lift(lift)  # ``fixed`` in place
        q[:, coords.root_q + 2] += lift
        if shape is not None:
            shape = ReferenceShape(registration, fixed)
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
    if OPTIONS.frozen_depth or OPTIONS.original_clearance:
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
    acquisition_guard = AcquisitionNormalGuard(objective, fixed)
    acquisition_continuity = SequenceContinuityGuard(task_set, fixed, rotations, windowed=True,
                                                     correction=OPTIONS.correction_budget)
    fixed_continuity = CorrectionContinuity(coords, fixed, root_norm=OPTIONS.batch2)
    objective.candidate_guards = [acquisition_guard, acquisition_continuity,
                                  fixed_continuity]
    # The acquisition guard also enters the objective as stiff one-sided rows,
    # so plain steps rarely need the costly constrained fallback.
    objective.hinge_guards = [acquisition_guard]

    objective.correction_temporal_scale = CORRECTION_TEMPORAL_WEIGHT
    rounds, penetration = [], []
    # Height registration and the support lift have finished. No collision
    # pass may move this reference to gain a new angular or translation allowance.
    report_lift = float(lift.max(initial=0.0))
    # One worker pool serves every pass: re-weighting only changes weights
    # the parent owns or sends with each evaluation.
    objective.start_workers()
    try:
        motion, history = optimize(objective, q, iterations=cfg["iterations"],
                                   callback=callback, label="contact_scene")
        verified = verify_budgets(model, coords, fixed, motion, objective.envelope_bodies, shape)
        for round_index in range(REWEIGHT_ROUNDS):
            budgets, chatter, posture = verified
            rounds.append(budgets)
            if not chatter.any() and not posture.any():
                break
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
            candidate, more = optimize(objective, motion, iterations=cfg["iterations"],
                                       callback=callback, label=label)
            history += more
            checked = verify_budgets(model, coords, fixed, candidate, objective.envelope_bodies, shape)
            if not _improves(budgets, chatter, posture, checked, strict=False):
                rounds[-1]["rejected"] = label
                rounds[-1]["rejected_budgets"] = checked[0]
                continue
            motion, verified = candidate, checked
        # Free-body penetration: every contact held, a small correction of the
        # accepted motion, kept only if it adds no chatter or layout flip
        # (jitter and layout precede clearance). See penetration.py.
        # The acquisition workers still hold the same immutable fields and
        # shells: they measure the deepest samples below, before the free
        # stage changes what the objective evaluates.
        from .acquisition_clearance import complete_original_clearance
        if OPTIONS.original_clearance:
            motion, more, local_conflicts = complete_original_clearance(
                objective, motion, depth_provider, local_conflicts, optimize, callback)
            history += more
        verified = verify_budgets(model, coords, fixed, motion, objective.envelope_bodies, shape)
        acquired = motion.copy()
        contact_guard = ContactGuard(objective, acquired)
        anchored_contacts = ContactGuard(objective, acquired, preserve_positions=True)
        continuity_guard = CorrectionContinuity(coords, acquired)
        objective.candidate_guards = [anchored_contacts, continuity_guard,
                                      acquisition_guard, acquisition_continuity] + (
                                      [fixed_continuity] if OPTIONS.correction_budget else [])
        if coords.root_q is not None:
            objective.fixed_root = acquired[:, coords.root_q:coords.root_q + 7].copy()
        record = {"stage": "free_bodies", "before": objective.penetration_summary(motion)}
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
            if not guard.accepts(candidate):
                accepted = False
                reasons.append(name)
        record["accepted"] = bool(accepted)
        if accepted:
            motion, verified = candidate, checked
        else:
            record["rejected"] = reasons
            record["rejected_budgets"] = checked[0]
        penetration.append(record)
    finally:
        objective.stop_workers()
    # The selected repair group may approach its original target; all other
    # groups remain anchored by RepairLimits. Precision still uses acquisition.
    objective.candidate_guards = [contact_guard, continuity_guard,
                                  acquisition_guard, acquisition_continuity] + (
                                      [fixed_continuity] if OPTIONS.correction_budget else [])
    mesh_repair = None
    grid_coverage = [float(np.isfinite(obj["sdf"].clearance).mean()) for obj in objects]
    # Original-geometry repair is not an open-object fallback. A watertight
    # object, a floor-only scene, or self-collision needs the same bounded
    # correction and audit; grid coverage says nothing about those overlaps.
    if common.get("scene_query") is not None or objects or config.get("self_collision_enabled", True):
        motion, mesh_repair = repair_original_meshes(
            objective, motion, getattr(owner, "_scene_objects", ()), common.get("scene_query"),
        )
        verified = verify_budgets(model, coords, fixed, motion, objective.envelope_bodies, shape)
    # Final statistics; worker chunks give the same values as a serial pass.
    started_workers = objective.start_workers()
    try:
        objective.evaluate(motion, with_jacobian=False)
    finally:
        if started_workers:
            objective.stop_workers()
    budgets = verified[0]
    final_continuity = audit_continuity(coords, fixed, motion)
    correction_continuity = audit_continuity(coords, acquired, motion)
    report["stages"]["contact_scene"] = {
        "history": history, "seconds": time.monotonic() - tick, "grid_seconds": grid_seconds,
        "final": objective.stats, "reweight_rounds": rounds, "budgets": budgets,
        "penetration_stages": penetration,
        "original_mesh_repair": mesh_repair,
        "frozen_original_depth": [] if depth_provider is None else depth_provider.audits,
        "local_repair_reach_conflicts": local_conflicts,
        "continuity_vs_fixed_reference": final_continuity,
        "continuity_vs_acquired": correction_continuity,
        "contacts_vs_acquired": contact_guard.diagnostics(motion),
        "protected_normals_vs_fixed_reference": acquisition_guard.diagnostics(motion),
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
    report.update(
        status="solved", frames=len(motion), framerate=float(framerate),
        model_nq=model.nq, model_nv=model.nv, scalar_joint_count=len(coords.qi),
        history=history, final_residuals=objective.stats,
        contact_offset_choice={k: np.asarray(v).tolist() for k, v in task_set.choices.items()},
        seconds=time.monotonic() - started,
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
        report["layout_and_relaxed_chatter_accepted"] = bool(
            layout_ok and final_continuity["relaxed_chatter_accepted"]
            and initialization_continuity is not None
            and initialization_continuity["relaxed_chatter_accepted"])
        # The enforced per-joint budgets are windowed (motion_budgets); the
        # stencil-exact comparison above stays an informational report.
        report["windowed_continuity_accepted"] = bool(
            final_continuity["windowed_budget_accepted"] and initialization_continuity is not None
            and initialization_continuity["windowed_budget_accepted"])
        report["layout_continuity_accepted"] = bool(layout_ok and report["windowed_continuity_accepted"])
        if not report["layout_continuity_accepted"]:
            report["acceptance_status"] = "unresolved_layout_or_continuity"
    owner.last_fast_source_pose = source_pose
    return motion, report
