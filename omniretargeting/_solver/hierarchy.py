"""Solve prioritized objectives and apply the bounded configuration step."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
from . import tangent

from .targets import (
    _base_orientation_step_limit,
    _limit_relative_orientation_step,
)


def solve_hierarchy(
    retargeter,
    *,
    base_cfg,
    base_orientation_residual,
    body_obj,
    body_orientation_residual,
    constraints,
    contact_residual,
    self_contact_residual=None,
    surface_contact_residual=None,
    surface_pose_residual=None,
    support_residual=None,
    follow_surface_residual=None,
    support_endpoint_residual=None,
    prioritize_surface_contact=False,
    dqa,
    end_effector_targets,
    endpoint_contact_constraints,
    num_contacts,
    num_swing_orientations,
    num_swings,
    pose_direction_residual,
    previous_target_base_orientation,
    q,
    q_a_current,
    q_last,
    swing_orientation_residual,
    swing_residual,
    target_base_orientation,
    target_base_position,
    temporal_cfg,
):
    """Solve prioritized objectives and apply the bounded configuration step."""
    hierarchy_enabled = bool(
        retargeter.end_effector_enabled
        and retargeter.end_effector_config.get("hierarchy", True)
        and (
            contact_residual is not None
            or swing_residual is not None
            or swing_orientation_residual is not None
            or base_orientation_residual is not None
            or body_orientation_residual is not None
            or pose_direction_residual is not None
        )
    )
    strict_surface = prioritize_surface_contact and surface_contact_residual is not None
    supported = (support_residual is not None
                 or support_endpoint_residual is not None
                 or follow_surface_residual is not None)
    hierarchy_enabled = hierarchy_enabled or strict_surface or supported
    free_priority = str(retargeter.end_effector_config.get("free_motion_priority", "lexicographic"))
    if free_priority not in {"lexicographic", "balanced"}:
        raise ValueError("free_motion_priority must be 'lexicographic' or 'balanced'.")
    balanced_free = free_priority == "balanced"
    if balanced_free and contact_residual is None and not strict_surface and not supported:
        # With no active contact there is no optimum to protect. Retain
        # the existing weighted solver's recovery behavior instead of
        # introducing a stricter failure path merely because contact
        # support is configured for other frames of the same sequence.
        hierarchy_enabled = False
    if not hierarchy_enabled or balanced_free:
        # Trusted contacts are still protected by a separate first level.
        # Free world targets and source shape can instead share the same
        # weighted level, so morphology mismatch does not force an exact
        # free-foot position at the expense of the whole-body pose.
        endpoint_weight = float(retargeter.end_effector_config.get("weighted_scale", 1.0))
        if not hierarchy_enabled and support_endpoint_residual is not None:
            body_obj += endpoint_weight * cp.sum_squares(support_endpoint_residual)
        if contact_residual is not None:
            body_obj += endpoint_weight * cp.sum_squares(contact_residual)
        if swing_residual is not None:
            body_obj += endpoint_weight * cp.sum_squares(swing_residual)
        if swing_orientation_residual is not None:
            body_obj += endpoint_weight * cp.sum_squares(
                swing_orientation_residual
            )
        if pose_direction_residual is not None:
            body_obj += float(
                retargeter.end_effector_config.get("pose_direction_weight", 10.0)
            ) * cp.sum_squares(pose_direction_residual)

    baseline_body_obj = body_obj
    if self_contact_residual is not None:
        # Anatomical surface correspondence is approximate. Share the existing
        # free-motion objective rather than changing support-foot priorities.
        # A weighted term makes source confidence/transition ramps effective;
        # multiplying an isolated lexicographic optimum by a ramp would not.
        body_obj += cp.sum_squares(self_contact_residual)

    if surface_contact_residual is not None and not strict_surface:
        body_obj = body_obj + cp.sum_squares(surface_contact_residual)
    if surface_pose_residual is not None:
        body_obj = body_obj + cp.sum_squares(surface_pose_residual)

    import sys
    show_solver_debug = hasattr(sys, '_omni_frame_count') and sys._omni_frame_count <= 5

    try:
        active_constraints = list(constraints)

        def solve_stage(stage_obj: cp.Expression, stage_name: str) -> cp.Problem:
            nonlocal active_constraints
            stage_problem = cp.Problem(cp.Minimize(stage_obj), active_constraints)

            def run_solver(candidate: cp.Problem) -> None:
                try:
                    candidate.solve(solver=cp.CLARABEL, verbose=False)
                except cp.error.SolverError:
                    pass
                if not tangent.usable(candidate, retargeter) and not tangent.enabled(retargeter):
                    candidate.solve(
                        solver=cp.SCS,
                        verbose=False,
                        eps=float(retargeter.end_effector_config.get("solver_tolerance", 1e-4)),
                        max_iters=int(
                            retargeter.end_effector_config.get("solver_max_iterations", 20000)
                        ),
                    )

            run_solver(stage_problem)
            if (
                not tangent.usable(stage_problem, retargeter)
                and not hierarchy_enabled
                and endpoint_contact_constraints
            ):
                # The contact-plane halfspaces and orientation-step caps can
                # be jointly infeasible with the trust region for a single
                # linearized step.  Relax them before touching the trust
                # region itself: an unbounded linearized step is worse than
                # one frame of relaxed contact-side constraints.
                if show_solver_debug:
                    print(
                        f"    {stage_name} solve failed: {stage_problem.status}, "
                        "retrying without endpoint contact constraints..."
                    )
                relaxable = {id(c) for c in endpoint_contact_constraints}
                active_constraints = [
                    constraint
                    for constraint in active_constraints
                    if id(constraint) not in relaxable
                ]
                stage_problem = cp.Problem(cp.Minimize(stage_obj), active_constraints)
                run_solver(stage_problem)
            if (
                not tangent.usable(stage_problem, retargeter)
                and not hierarchy_enabled
                and not tangent.enabled(retargeter)
            ):
                if show_solver_debug:
                    print(
                        f"    {stage_name} solve failed: {stage_problem.status}, "
                        "trying without SOC..."
                    )
                active_constraints = [
                    constraint
                    for constraint in active_constraints
                    if not isinstance(constraint, cp.constraints.second_order.SOC)
                ]
                stage_problem = cp.Problem(cp.Minimize(stage_obj), active_constraints)
                run_solver(stage_problem)
            return stage_problem

        def protect_endpoint_baseline():
            """Additional surface objectives cannot purchase error at a foot.

            Compare against the same constrained solve without the new surface
            objective, from the same SQP state. Per-endpoint bounds prevent one
            foot improving from hiding a regression at the other. This keeps
            unreachable source targets attainable at their existing error.
            """
            residuals = getattr(retargeter, "_protected_endpoint_residuals", {})
            if (all(r is None for r in (surface_contact_residual, self_contact_residual,
                                       surface_pose_residual)) or not residuals):
                return True
            baseline_problem = solve_stage(baseline_body_obj, "surface-endpoint-reference")
            if not tangent.usable(baseline_problem, retargeter):
                return False
            tasks = (getattr(retargeter, "surface_pose_tasks", None)
                     or getattr(retargeter, "surface_contact_tasks", None))
            allowance = float(tasks.cfg.get("endpoint_regression", 0.001)) if tasks is not None else 0.001
            for residual in residuals.values():
                bound = float(np.linalg.norm(residual.value)) + allowance
                active_constraints.append(cp.norm(residual, 2) <= bound)
            return True

        if hierarchy_enabled:
            relative_tolerance = max(
                float(retargeter.end_effector_config.get("priority_relative_tolerance", 0.01)),
                0.0,
            )
            absolute_tolerance = max(
                float(retargeter.end_effector_config.get("priority_absolute_tolerance", 1e-4)),
                0.0,
            )
            tie_break = float(retargeter.end_effector_config.get("tie_break_regularization", 1e-8))

            # Observed roles: endpoint support, surface support, then the
            # following contacts. Free limbs and posture come later, so they
            # yield only as much as these contacts actually require.
            from ..contact_phase import PRIORITY_TOLERANCE

            if support_endpoint_residual is not None:
                endpoint_support_level = cp.norm(support_endpoint_residual, 2)
                endpoint_support_problem = solve_stage(
                    endpoint_support_level + tie_break * cp.sum_squares(dqa),
                    "endpoint-support",
                )
                if not tangent.usable(endpoint_support_problem, retargeter):
                    return q, np.inf
                active_constraints.append(
                    endpoint_support_level
                    <= (1.0 + relative_tolerance)
                    * max(float(endpoint_support_level.value), 0.0)
                    + absolute_tolerance
                )

            if support_residual is not None:
                support_level = cp.norm(support_residual, 2)
                support_problem = solve_stage(
                    support_level + tie_break * cp.sum_squares(dqa), "surface-support"
                )
                if not tangent.usable(support_problem, retargeter):
                    return q, np.inf
                active_constraints.append(
                    support_level <= max(float(support_level.value), 0.0)
                    + PRIORITY_TOLERANCE
                )

            if contact_residual is not None:
                contact_level = cp.norm(contact_residual, 2)
                contact_problem = solve_stage(
                    contact_level + tie_break * cp.sum_squares(dqa), "contact"
                )
                if not tangent.usable(contact_problem, retargeter):
                    return q, np.inf
                contact_optimum = max(float(contact_level.value), 0.0)
                contact_bound = (
                    (1.0 + relative_tolerance) * contact_optimum
                    + absolute_tolerance
                )
                active_constraints.append(contact_level <= contact_bound)

            if follow_surface_residual is not None:
                follow_surface_level = cp.norm(follow_surface_residual, 2)
                follow_surface_problem = solve_stage(
                    follow_surface_level + tie_break * cp.sum_squares(dqa),
                    "surface-follow",
                )
                if not tangent.usable(follow_surface_problem, retargeter):
                    return q, np.inf
                active_constraints.append(
                    follow_surface_level
                    <= max(float(follow_surface_level.value), 0.0)
                    + PRIORITY_TOLERANCE
                )

            if strict_surface:
                if not protect_endpoint_baseline():
                    return q, np.inf
                surface_level = cp.norm(surface_contact_residual, 2)
                surface_problem = solve_stage(
                    surface_level + tie_break * cp.sum_squares(dqa), "surface-contact"
                )
                if not tangent.usable(surface_problem, retargeter):
                    return q, np.inf
                active_constraints.append(
                    surface_level <= (1.0 + relative_tolerance) * max(float(surface_level.value), 0.0)
                    + absolute_tolerance
                )

            if balanced_free:
                if not strict_surface and not protect_endpoint_baseline():
                    return q, np.inf
                problem = solve_stage(body_obj, "balanced-free-motion")
            else:
                def preserve_swing_level() -> bool:
                    if swing_residual is None:
                        return True
                    swing_level = cp.norm(swing_residual, 2)
                    swing_problem = solve_stage(
                        swing_level + tie_break * cp.sum_squares(dqa), "swing"
                    )
                    if not tangent.usable(swing_problem, retargeter):
                        return False
                    swing_optimum = max(float(swing_level.value), 0.0)
                    swing_relative_tolerance = max(
                        float(
                            retargeter.end_effector_config.get(
                                "swing_priority_relative_tolerance",
                                relative_tolerance,
                            )
                        ),
                        0.0,
                    )
                    swing_absolute_tolerance = max(
                        float(
                            retargeter.end_effector_config.get(
                                "swing_priority_absolute_tolerance",
                                absolute_tolerance,
                            )
                        ),
                        0.0,
                    )
                    swing_bound = (
                        (1.0 + swing_relative_tolerance) * swing_optimum
                        + swing_absolute_tolerance
                    )
                    active_constraints.append(swing_level <= swing_bound)
                    return True

                def preserve_base_orientation_level() -> bool:
                    if base_orientation_residual is None:
                        return True
                    base_orientation_level = cp.norm(
                        base_orientation_residual, 2
                    )
                    base_orientation_problem = solve_stage(
                        base_orientation_level
                        + tie_break * cp.sum_squares(dqa),
                        "base-orientation",
                    )
                    if not tangent.usable(base_orientation_problem, retargeter):
                        return False
                    base_orientation_optimum = max(
                        float(base_orientation_level.value), 0.0
                    )
                    base_orientation_relative_tolerance = max(
                        float(
                            retargeter.end_effector_config.get(
                                "base_orientation_priority_relative_tolerance",
                                relative_tolerance,
                            )
                        ),
                        0.0,
                    )
                    base_orientation_absolute_tolerance = max(
                        float(
                            retargeter.end_effector_config.get(
                                "base_orientation_priority_absolute_tolerance",
                                0.02,
                            )
                        ),
                        0.0,
                    )
                    active_constraints.append(
                        base_orientation_level
                        <= (1.0 + base_orientation_relative_tolerance)
                        * base_orientation_optimum
                        + base_orientation_absolute_tolerance
                    )
                    return True

                # Scene contacts always remain first. Source/robot profiles
                # may then choose whether free-limb world targets or the
                # calibrated whole-trunk frame comes next. The legacy order is
                # retained by default; anatomical torso presets opt into the
                # latter with a bounded angular allowance.
                base_before_swing = bool(
                    retargeter.end_effector_config.get(
                        "base_orientation_before_swing", False
                    )
                )
                priority_stages = (
                    (preserve_base_orientation_level, preserve_swing_level)
                    if base_before_swing
                    else (preserve_swing_level, preserve_base_orientation_level)
                )
                for preserve_stage in priority_stages:
                    if not preserve_stage():
                        return q, np.inf

                if swing_orientation_residual is not None:
                    swing_orientation_level = cp.norm(
                        swing_orientation_residual, 2
                    )
                    swing_orientation_problem = solve_stage(
                        swing_orientation_level
                        + tie_break * cp.sum_squares(dqa),
                        "swing-orientation",
                    )
                    if not tangent.usable(swing_orientation_problem, retargeter):
                        return q, np.inf
                    swing_orientation_optimum = max(
                        float(swing_orientation_level.value), 0.0
                    )
                    orientation_relative_tolerance = max(
                        float(
                            retargeter.end_effector_config.get(
                                "orientation_priority_relative_tolerance",
                                relative_tolerance,
                            )
                        ),
                        0.0,
                    )
                    orientation_absolute_tolerance = max(
                        float(
                            retargeter.end_effector_config.get(
                                "orientation_priority_absolute_tolerance",
                                absolute_tolerance,
                            )
                        ),
                        0.0,
                    )
                    active_constraints.append(
                        swing_orientation_level
                        <= (1.0 + orientation_relative_tolerance)
                        * swing_orientation_optimum
                        + orientation_absolute_tolerance
                    )

                if pose_direction_residual is not None:
                    pose_level = cp.norm(pose_direction_residual, 2)
                    pose_problem = solve_stage(
                        pose_level + tie_break * cp.sum_squares(dqa),
                        "pose-direction",
                    )
                    if not tangent.usable(pose_problem, retargeter):
                        return q, np.inf
                    pose_optimum = max(float(pose_level.value), 0.0)
                    pose_relative_tolerance = max(
                        float(
                            retargeter.end_effector_config.get(
                                "pose_priority_relative_tolerance", 0.02
                            )
                        ),
                        0.0,
                    )
                    pose_absolute_tolerance = max(
                        float(
                            retargeter.end_effector_config.get(
                                "pose_priority_absolute_tolerance", 0.04
                            )
                        ),
                        0.0,
                    )
                    active_constraints.append(
                        pose_level
                        <= (1.0 + pose_relative_tolerance) * pose_optimum
                        + pose_absolute_tolerance
                    )

                # Bone directions preserve the visible source pose. Calibrated
                # full frames then resolve remaining twist; cross-morphology
                # orientation mismatch must not bend a limb away from that pose.
                if body_orientation_residual is not None:
                    body_orientation_level = cp.norm(
                        body_orientation_residual, 2
                    )
                    body_orientation_problem = solve_stage(
                        body_orientation_level
                        + tie_break * cp.sum_squares(dqa),
                        "body-orientation",
                    )
                    if not tangent.usable(body_orientation_problem, retargeter):
                        return q, np.inf
                    body_orientation_optimum = max(
                        float(body_orientation_level.value), 0.0
                    )
                    body_orientation_relative_tolerance = max(
                        float(
                            retargeter.end_effector_config.get(
                                "body_orientation_priority_relative_tolerance",
                                relative_tolerance,
                            )
                        ),
                        0.0,
                    )
                    body_orientation_absolute_tolerance = max(
                        float(
                            retargeter.end_effector_config.get(
                                "body_orientation_priority_absolute_tolerance",
                                0.1,
                            )
                        ),
                        0.0,
                    )
                    active_constraints.append(
                        body_orientation_level
                        <= (1.0 + body_orientation_relative_tolerance)
                        * body_orientation_optimum
                        + body_orientation_absolute_tolerance
                    )

                if not strict_surface and not protect_endpoint_baseline():
                    return q, np.inf
                problem = solve_stage(body_obj, "whole-body")
                if show_solver_debug:
                    print(
                        f"    hierarchy tasks: contact={num_contacts}, "
                        f"swing={num_swings}, "
                        f"swing_orientation={num_swing_orientations}, "
                        f"body_orientation={len(retargeter._source_orientation_indices)}, "
                        f"pose_bones={len(retargeter.pose_bones)}"
                    )
        else:
            if not protect_endpoint_baseline():
                return q, np.inf
            problem = solve_stage(body_obj, "weighted")

        if (
            not tangent.usable(problem, retargeter)
            and show_solver_debug
        ):
            print(f"    Final solve status: {problem.status}")

        retargeter.last_subproblem = dict(status=problem.status,
                                         constraint_violation=tangent.constraint_violation(problem))
        if tangent.usable(problem, retargeter):
            dqa_opt = np.asarray(dqa.value, dtype=float).reshape(-1)
            # First-order SQP guarantees only hold inside the trust
            # region.  Conic solvers may return an OPTIMAL_INACCURATE
            # point with constraint violations, so enforce the radius and
            # safety-critical base envelope explicitly before applying it.
            if hierarchy_enabled and not tangent.enabled(retargeter):
                protected_base_qa = []
                if (
                    target_base_position is not None
                    and bool(base_cfg.get("enabled", False))
                ):
                    base_target = np.asarray(
                        target_base_position, dtype=float
                    ).reshape(3).copy()
                    base_target[2] += float(base_cfg.get("z_offset", 0.0))
                    proposed_base = q[:3].copy()
                    for qpos_idx in (0, 1, 2):
                        if qpos_idx in retargeter.q_a_indices:
                            qa_idx = int(
                                np.where(retargeter.q_a_indices == qpos_idx)[0][0]
                            )
                            proposed_base[qpos_idx] += dqa_opt[qa_idx]

                    if bool(base_cfg.get("track_xy", True)) and (
                        bool(base_cfg.get("lock_xy", False))
                        or bool(base_cfg.get("hard_constraints", True))
                    ):
                        xy_delta = proposed_base[:2] - base_target[:2]
                        xy_norm = float(np.linalg.norm(xy_delta))
                        if bool(base_cfg.get("lock_xy", False)):
                            proposed_base[:2] = base_target[:2]
                        else:
                            max_xy_error = max(
                                float(base_cfg.get("max_xy_error", 0.08)),
                                0.0,
                            )
                            if xy_norm > max_xy_error:
                                proposed_base[:2] = base_target[:2] + (
                                    max_xy_error / max(xy_norm, 1e-12)
                                ) * xy_delta
                        protected_base_qa.extend(
                            int(np.where(retargeter.q_a_indices == qpos_idx)[0][0])
                            for qpos_idx in (0, 1)
                            if qpos_idx in retargeter.q_a_indices
                        )
                    if bool(base_cfg.get("track_z", True)) and bool(
                        base_cfg.get("hard_constraints", True)
                    ):
                        max_z_error = max(
                            float(base_cfg.get("max_z_error", 0.12)), 0.0
                        )
                        proposed_base[2] = np.clip(
                            proposed_base[2],
                            base_target[2] - max_z_error,
                            base_target[2] + max_z_error,
                        )
                        if 2 in retargeter.q_a_indices:
                            protected_base_qa.append(
                                int(np.where(retargeter.q_a_indices == 2)[0][0])
                            )
                    for qpos_idx in (0, 1, 2):
                        if qpos_idx in retargeter.q_a_indices:
                            qa_idx = int(
                                np.where(retargeter.q_a_indices == qpos_idx)[0][0]
                            )
                            if qa_idx in protected_base_qa:
                                dqa_opt[qa_idx] = (
                                    proposed_base[qpos_idx] - q[qpos_idx]
                                )

                fixed_norm_sq = float(
                    np.sum(np.square(dqa_opt[protected_base_qa]))
                )
                free_indices = np.array(
                    [
                        idx
                        for idx in range(len(dqa_opt))
                        if idx not in protected_base_qa
                    ],
                    dtype=int,
                )
                remaining_radius = np.sqrt(
                    max(retargeter.step_size**2 - fixed_norm_sq, 0.0)
                )
                free_norm = float(np.linalg.norm(dqa_opt[free_indices]))
                if free_norm > remaining_radius:
                    dqa_opt[free_indices] *= remaining_radius / max(
                        free_norm, 1e-12
                    )
            cost = float(problem.value)

            q_opt = q.copy()
            q_opt[retargeter.q_a_indices] = dqa_opt + q_a_current
            if tangent.enabled(retargeter):
                q_opt = tangent.integrate_step(retargeter, q, dqa_opt)
            # OPTIMAL_INACCURATE solutions can violate CVXPY bounds by a
            # visible amount. Enforce the exact task-safe range before a
            # configuration is accepted by the next SQP iteration.
            q_opt[retargeter.q_a_indices] = np.clip(
                q_opt[retargeter.q_a_indices], retargeter.q_a_lb, retargeter.q_a_ub
            )

            # CRITICAL FIX: Normalize quaternion with sign continuity to prevent frame-to-frame jumps
            quat_new = q_opt[3:7]
            quat_new = quat_new / (np.linalg.norm(quat_new) + 1e-12)

            # Ensure quaternion sign continuity with previous frame (if available)
            if q_last is not None:
                quat_prev = q_last[3:7]
                # If dot product is negative, quaternions are in opposite hemispheres
                # Flip sign to ensure continuity
                if np.dot(quat_new, quat_prev) < 0:
                    quat_new = -quat_new

                max_orientation_step_error = temporal_cfg.get(
                    "max_base_orientation_step_error"
                )
                sticking_contact_active = support_endpoint_residual is not None or any(
                    bool(target_data.get("contact", False))
                    and str(target_data.get("contact_mode", ""))
                    == "sticking"
                    for target_data in (end_effector_targets or {}).values()
                )
                if (
                    max_orientation_step_error is not None
                    and target_base_orientation is not None
                    and previous_target_base_orientation is not None
                    # An exact post-solve root projection would move a
                    # world-anchored sticking endpoint. In that regime the
                    # conic constraint remains active inside the optimizer,
                    # where the contact task correctly has first priority.
                    and not sticking_contact_active
                ):
                    allowed_step = _base_orientation_step_limit(
                        target_base_orientation,
                        previous_target_base_orientation,
                        quat_prev,
                        float(max_orientation_step_error),
                        catchup_gain=float(
                            temporal_cfg.get(
                                "base_orientation_catchup_gain", 0.0
                            )
                        ),
                        catchup_deadband=float(
                            temporal_cfg.get(
                                "base_orientation_catchup_deadband", 0.0
                            )
                        ),
                        max_extra_step=temporal_cfg.get(
                            "max_base_orientation_catchup_step"
                        ),
                        max_total_step=temporal_cfg.get(
                            "max_base_orientation_total_step"
                        ),
                    )
                    quat_new = _limit_relative_orientation_step(
                        quat_new,
                        quat_prev,
                        allowed_step,
                    )

            q_opt[3:7] = quat_new

            q_opt = retargeter._guard_self_collision_step(q, q_opt, q_last)

            return q_opt, cost
        else:
            if show_solver_debug:
                print(f"    SOLVER FAILED: {problem.status}")
            return q, np.inf

    except Exception as e:
        if show_solver_debug:
            print(f"    EXCEPTION: {e}")
        return q, np.inf
