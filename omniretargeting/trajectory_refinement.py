"""Sparse whole-trajectory refinement for contact-aware retargeting.

The framewise interaction-mesh solver supplies a natural initial motion.  This
module then optimizes all scalar robot joints together.  End-effector SE(3)
tasks are linearized with MuJoCo Jacobians, while acceleration and jerk are
exact linear residuals over the complete trajectory.  Unlike post-hoc signal
filtering, smoothing therefore happens inside the same least-squares problem
as the hand/foot constraints.

Scene non-penetration belongs to that same problem.  ``lsq_linear`` only
accepts box bounds, so penetration enters as a one-sided residual whose active
set is rebuilt at every Gauss--Newton iteration.  That is a penalty and not a
hard constraint: it drives sample points out of the scene but cannot certify a
penetration-free result, so the achieved depth is always measured separately.
The main pipeline supplies a RefinementGuard: it uses CLARABEL to enforce the
final validator's endpoint/correction budgets and checks real FK and geometry
before retaining each step. The box-only standalone API remains available.
"""

from __future__ import annotations

from typing import Any, Mapping

import mujoco
import numpy as np
import trimesh
from scipy import sparse
from scipy.optimize import lsq_linear

from ._refinement.effectors import build_refinement_effectors
from ._refinement.math import (
    _difference_operator as _difference_operator,
    _inset_finite_joint_bounds as _inset_finite_joint_bounds,
    _joint_coordinate_mask as _joint_coordinate_mask,
    _joint_posture_target_vectors as _joint_posture_target_vectors,
    _joint_temporal_weight_vector as _joint_temporal_weight_vector,
    _local_semantic_rotation as _local_semantic_rotation,
    _lock_contact_offsets as _lock_contact_offsets,
    _minimal_vector_alignment as _minimal_vector_alignment,
    _orthonormal_target_rotation as _orthonormal_target_rotation,
    _penetration_depth_weight as _penetration_depth_weight,
    _scalar_joint_coordinates as _scalar_joint_coordinates,
    _skew_symmetric as _skew_symmetric,
    _transported_arm_segment_targets as _transported_arm_segment_targets,
)
from ._refinement.tasks import linearize_trajectory_tasks
from ._refinement.self_collision import (
    guard_self_collision_update,
    linearize_self_collisions,
)
from .self_collision import SelfCollisionQuery
from ._solver.targets import (
    _apply_configured_joint_limits,
    _apply_joint_limit_safety_fraction,
)
from .contact import (
    SceneContactTask,
)
from .scene_geometry import (
    ScenePenetrationProbe,
    SceneQuery,
    collision_geom_ids,
    geom_surface_offsets as geom_surface_offsets,
    measure_scene_penetration as measure_scene_penetration,
)


def refine_joint_trajectory(
    model: mujoco.MjModel,
    trajectory: np.ndarray,
    tasks: Mapping[str, SceneContactTask],
    end_effector_config: Mapping[str, Any],
    config: Mapping[str, Any],
    scene_mesh: trimesh.Trimesh | None = None,
    framerate: float | None = None,
    source_root_positions: np.ndarray | None = None,
    source_joint_positions: np.ndarray | None = None,
    source_joint_names: list[str] | tuple[str, ...] | None = None,
    scene_query: SceneQuery | None = None,
    body_relation_tasks=None,
    self_contact_tasks=None,
    surface_contact_tasks=None,
    refinement_guard=None,
) -> np.ndarray:
    """Jointly refine scalar joints over time with sparse Gauss--Newton.

    The floating base is deliberately held at the framewise solver result.  It
    is already responsible for world registration and morphology compromise;
    allowing a second optimizer to move it would reintroduce the base drift
    that whole-body smoothing is meant to avoid.
    """
    result = np.asarray(trajectory, dtype=float).copy()
    if surface_contact_tasks is not None and (
        surface_contact_tasks.model is not model or len(surface_contact_tasks.targets) != len(result)
    ):
        raise ValueError("Surface contact must use the same model and original timeline")
    if refinement_guard is not None:
        from .trajectory_validation import QueryBudgetExceeded

        validator = refinement_guard.validator
        if validator.model is not model or not np.array_equal(validator.reference, result):
            raise ValueError("Refinement guard must use the same model and input trajectory")
    self_enabled = bool(config.get("self_collision_enabled", True))
    if len(result) == 0 or (not self_enabled and self_contact_tasks is None and surface_contact_tasks is None and (len(result) < 4 or not tasks)):
        return result
    self_query = SelfCollisionQuery(model, config.get("self_collision")) if self_enabled else None

    qpos_indices, dof_indices, joint_lower, joint_upper = (
        _scalar_joint_coordinates(model)
    )
    joint_lower, joint_upper = _apply_joint_limit_safety_fraction(
        model,
        qpos_indices,
        joint_lower,
        joint_upper,
        float(config.get("joint_limit_safety_fraction", 1.0)),
    )
    joint_lower, joint_upper = _apply_configured_joint_limits(
        model,
        qpos_indices,
        joint_lower,
        joint_upper,
        end_effector_config.get("joint_posture_limits"),
    )
    configured_joint_lower = joint_lower.copy()
    configured_joint_upper = joint_upper.copy()
    num_frames = len(result)
    num_joints = len(qpos_indices)
    if num_joints == 0:
        return result
    joint_limit_cfg = dict(config.get("joint_limit_avoidance", {}) or {})
    joint_limit_avoidance_enabled = bool(
        joint_limit_cfg.get("enabled", False)
    )
    joint_limit_hard_margin = max(
        float(joint_limit_cfg.get("hard_margin", 0.0)), 0.0
    )
    joint_limit_hard_margin_ratio = max(
        float(joint_limit_cfg.get("hard_margin_ratio", 0.0)), 0.0
    )
    joint_limit_soft_margin = max(
        float(joint_limit_cfg.get("soft_margin", np.deg2rad(10.0))), 0.0
    )
    joint_limit_soft_margin_ratio = max(
        float(joint_limit_cfg.get("soft_margin_ratio", 0.0)), 0.0
    )
    joint_limit_sigma = max(
        float(joint_limit_cfg.get("sigma", 0.06)), 1e-8
    )
    joint_limit_sigma_ratio = max(
        float(joint_limit_cfg.get("sigma_ratio", 0.0)), 0.0
    )
    if joint_limit_avoidance_enabled and (
        joint_limit_hard_margin > 0.0
        or joint_limit_hard_margin_ratio > 0.0
    ):
        joint_lower, joint_upper = _inset_finite_joint_bounds(
            joint_lower,
            joint_upper,
            joint_limit_hard_margin,
            joint_limit_hard_margin_ratio,
        )
    if source_root_positions is not None:
        source_root_positions = np.asarray(source_root_positions, dtype=float)
        if source_root_positions.shape != (num_frames, 3):
            raise ValueError(
                "source_root_positions must have shape "
                f"({num_frames}, 3), got {source_root_positions.shape}."
            )
    source_name_to_index: dict[str, int] = {}
    if source_joint_positions is not None:
        source_joint_positions = np.asarray(source_joint_positions, dtype=float)
        if (
            source_joint_positions.ndim != 3
            or source_joint_positions.shape[0] != num_frames
            or source_joint_positions.shape[2] != 3
        ):
            raise ValueError(
                "source_joint_positions must have shape (T, J, 3), got "
                f"{source_joint_positions.shape}."
            )
        if (
            source_joint_names is None
            or len(source_joint_names) != source_joint_positions.shape[1]
        ):
            raise ValueError(
                "source_joint_names must identify every source joint when "
                "source_joint_positions are provided."
            )
        source_name_to_index = {
            str(joint_name): joint_idx
            for joint_idx, joint_name in enumerate(source_joint_names)
        }

    effectors = build_refinement_effectors(
        config=config,
        end_effector_config=end_effector_config,
        model=model,
        qpos_indices=qpos_indices,
        source_joint_positions=source_joint_positions,
        source_name_to_index=source_name_to_index,
        tasks=tasks,
    )

    if not effectors and not self_enabled and self_contact_tasks is None and surface_contact_tasks is None:
        return result

    iterations = max(int(config.get("iterations", 3)), 1)
    contact_position_sigma = max(
        float(config.get("contact_position_sigma", 0.025)), 1e-6
    )
    swing_position_sigma = max(
        float(config.get("swing_position_sigma", 0.06)), 1e-6
    )
    reference_sigma = max(float(config.get("reference_sigma", 0.18)), 1e-6)
    acceleration_sigma = max(
        float(config.get("acceleration_sigma", 0.02)), 1e-8
    )
    jerk_sigma = max(float(config.get("jerk_sigma", 0.01)), 1e-8)
    swing_step_cfg = dict(
        config.get("swing_joint_step_avoidance", {}) or {}
    )
    swing_step_enabled = bool(swing_step_cfg.get("enabled", False))
    swing_step_max = max(
        float(swing_step_cfg.get("max_step", np.deg2rad(8.0))), 1e-8
    )
    swing_step_sigma = max(
        float(swing_step_cfg.get("sigma", np.deg2rad(2.0))), 1e-8
    )
    swing_step_masks: dict[str, np.ndarray] = {}
    for effector_name, joint_tokens in dict(
        swing_step_cfg.get("effectors", {}) or {}
    ).items():
        mask = _joint_coordinate_mask(model, qpos_indices, joint_tokens)
        if not np.any(mask):
            raise ValueError(
                "trajectory_refinement.swing_joint_step_avoidance "
                f"effector '{effector_name}' does not match any scalar joint."
            )
        swing_step_masks[str(effector_name)] = mask
    joint_step_cfg = dict(config.get("joint_step_avoidance", {}) or {})
    joint_step_enabled = bool(joint_step_cfg.get("enabled", False))
    joint_step_groups: list[dict[str, Any]] = []
    for group_name, raw_group in dict(
        joint_step_cfg.get("groups", {}) or {}
    ).items():
        group = dict(raw_group or {})
        mask = _joint_coordinate_mask(
            model,
            qpos_indices,
            group.get("joint_tokens", [str(group_name)]),
        )
        if not np.any(mask):
            raise ValueError(
                "trajectory_refinement.joint_step_avoidance "
                f"group '{group_name}' does not match any scalar joint."
            )
        joint_step_groups.append(
            {
                "name": str(group_name),
                "mask": mask,
                "max_step": max(
                    float(group.get("max_step", np.deg2rad(10.0))), 1e-8
                ),
                "sigma": max(
                    float(group.get("sigma", np.deg2rad(1.0))), 1e-8
                ),
            }
        )
    trust_region = max(float(config.get("trust_region", 0.12)), 1e-6)
    solver_tolerance = max(float(config.get("solver_tolerance", 1e-5)), 1e-10)
    solver_iterations = max(int(config.get("solver_iterations", 40)), 1)
    step_tolerance = float(config.get("step_tolerance", 1e-4))
    use_acceleration = bool(config.get("acceleration_enabled", True))
    use_jerk = bool(config.get("jerk_enabled", True))

    # The derivative residuals are per-frame differences, so their sigmas carry
    # implicit units of rad/frame^n.  Rescaling against the framerate they were
    # tuned at keeps the physical smoothing strength constant when the source
    # motion arrives at a different rate.
    reference_framerate = float(config.get("sigma_reference_framerate", 30.0))
    if framerate and framerate > 0.0 and reference_framerate > 0.0:
        rate_ratio = reference_framerate / float(framerate)
        acceleration_sigma *= rate_ratio**2
        jerk_sigma *= rate_ratio**3
        swing_step_max *= rate_ratio
        swing_step_sigma *= rate_ratio
        for group in joint_step_groups:
            group["max_step"] *= rate_ratio
            group["sigma"] *= rate_ratio

    penetration_cfg = dict(config.get("penetration", {}) or {})
    penetration_enabled = (
        bool(penetration_cfg.get("enabled", True)) and scene_mesh is not None
    )
    penetration_clearance = float(penetration_cfg.get("clearance", 0.005))
    penetration_sigma = max(float(penetration_cfg.get("sigma", 0.004)), 1e-8)
    penetration_anchor_radius = max(
        float(penetration_cfg.get("contact_anchor_exclusion_radius", 0.05)), 0.0
    )
    probe: ScenePenetrationProbe | None = None
    if penetration_enabled:
        geom_ids = collision_geom_ids(
            model,
            penetration_cfg.get("geom_groups"),
            penetration_cfg.get("body_tokens"),
        )
        if len(geom_ids) == 0:
            print(
                "Trajectory refinement: no collision geoms resolved, so scene "
                "non-penetration is inactive."
            )
        else:
            probe = ScenePenetrationProbe(
                model,
                scene_mesh,
                geom_ids,
                max(float(penetration_cfg.get("margin", 0.05)), 1e-6),
                query=scene_query,
            )

    contact_offset_choice = _lock_contact_offsets(model, result, effectors)

    reference_joints = result[:, qpos_indices].copy()
    data = mujoco.MjData(model)
    jacobian_position = np.zeros((3, model.nv), dtype=float)
    jacobian_rotation = np.zeros((3, model.nv), dtype=float)

    # The time-difference blocks are exact and constant, so they are assembled
    # once; only their right-hand side is refreshed per iteration.
    temporal_joint_weights = _joint_temporal_weight_vector(
        model,
        qpos_indices,
        config.get("joint_temporal_weights"),
    )
    temporal_joint_scale = sparse.diags(
        np.sqrt(temporal_joint_weights), format="csr"
    )
    temporal_joint_scale_dense = np.sqrt(temporal_joint_weights)[None, :]
    acceleration_operator = _difference_operator(num_frames, 2)
    jerk_operator = _difference_operator(num_frames, 3)
    reference_block = (
        sparse.identity(num_frames * num_joints, format="csr") / reference_sigma
    )
    posture_targets, posture_sigmas = _joint_posture_target_vectors(
        model,
        qpos_indices,
        config.get("joint_posture_targets"),
    )
    posture_joint_indices = np.flatnonzero(np.isfinite(posture_sigmas))
    posture_block: sparse.csr_matrix | None = None
    if len(posture_joint_indices):
        posture_selector = sparse.coo_matrix(
            (
                1.0 / posture_sigmas[posture_joint_indices],
                (
                    np.arange(len(posture_joint_indices)),
                    posture_joint_indices,
                ),
            ),
            shape=(len(posture_joint_indices), num_joints),
        ).tocsr()
        posture_block = sparse.kron(
            sparse.identity(num_frames, format="csr"),
            posture_selector,
            format="csr",
        )
    acceleration_block = (
        sparse.kron(
            acceleration_operator, temporal_joint_scale, format="csr"
        )
        / acceleration_sigma
    )
    jerk_block = (
        sparse.kron(jerk_operator, temporal_joint_scale, format="csr")
        / jerk_sigma
    )

    for iteration in range(iterations):
        current_joints = result[:, qpos_indices].copy()
        task_block, block_rhs, active_penetration_rows = linearize_trajectory_tasks(
            body_relation_tasks=body_relation_tasks, self_contact_tasks=self_contact_tasks, surface_contact_tasks=surface_contact_tasks,
            contact_offset_choice=contact_offset_choice,
            contact_position_sigma=contact_position_sigma,
            data=data,
            dof_indices=dof_indices,
            effectors=effectors,
            end_effector_config=end_effector_config,
            jacobian_position=jacobian_position,
            jacobian_rotation=jacobian_rotation,
            model=model,
            num_frames=num_frames,
            num_joints=num_joints,
            penetration_anchor_radius=penetration_anchor_radius,
            penetration_cfg=penetration_cfg,
            penetration_clearance=penetration_clearance,
            penetration_sigma=penetration_sigma,
            probe=probe,
            result=result,
            source_joint_positions=source_joint_positions,
            source_root_positions=source_root_positions,
            swing_position_sigma=swing_position_sigma,
        )

        systems = [task_block]
        residuals = [np.asarray(block_rhs, dtype=float)]
        active_self_rows = 0
        if self_query is not None:
            self_block, self_rhs = linearize_self_collisions(self_query, result, dof_indices)
            systems.append(self_block)
            residuals.append(self_rhs)
            active_self_rows = len(self_rhs)

        active_limit_rows = 0
        if joint_limit_avoidance_enabled and (
            joint_limit_soft_margin > 0.0
            or joint_limit_soft_margin_ratio > 0.0
        ):
            finite_width = np.isfinite(configured_joint_lower) & np.isfinite(
                configured_joint_upper
            )
            configured_width = configured_joint_upper - configured_joint_lower
            soft_margin = np.full(num_joints, joint_limit_soft_margin)
            if joint_limit_soft_margin_ratio > 0.0:
                soft_margin[finite_width] = (
                    configured_width[finite_width]
                    * joint_limit_soft_margin_ratio
                )
            soft_margin[finite_width] = np.minimum(
                soft_margin[finite_width],
                np.maximum(configured_width[finite_width], 0.0) * 0.25,
            )
            lower_targets = configured_joint_lower + soft_margin
            upper_targets = configured_joint_upper - soft_margin
            lower_targets = np.maximum(lower_targets, joint_lower)
            upper_targets = np.minimum(upper_targets, joint_upper)
            limit_sigmas = np.full(num_joints, joint_limit_sigma)
            if joint_limit_sigma_ratio > 0.0:
                limit_sigmas[finite_width] = np.maximum(
                    configured_width[finite_width] * joint_limit_sigma_ratio,
                    1e-8,
                )
            lower_active = np.isfinite(joint_lower)[None, :] & (
                current_joints < lower_targets[None, :]
            )
            upper_active = np.isfinite(joint_upper)[None, :] & (
                current_joints > upper_targets[None, :]
            )
            active_frames, active_joints = np.nonzero(
                lower_active | upper_active
            )
            if len(active_frames):
                active_limit_rows = len(active_frames)
                active_columns = active_frames * num_joints + active_joints
                limit_targets = np.where(
                    lower_active[active_frames, active_joints],
                    lower_targets[active_joints],
                    upper_targets[active_joints],
                )
                limit_block = sparse.coo_matrix(
                    (
                        1.0 / limit_sigmas[active_joints],
                        (np.arange(active_limit_rows), active_columns),
                    ),
                    shape=(active_limit_rows, num_frames * num_joints),
                ).tocsr()
                systems.append(limit_block)
                residuals.append(
                    (
                        limit_targets
                        - current_joints[active_frames, active_joints]
                    )
                    / limit_sigmas[active_joints]
                )

        active_swing_step_rows = 0
        if swing_step_enabled and num_frames > 1:
            swing_rows: list[int] = []
            swing_columns: list[int] = []
            swing_values: list[float] = []
            swing_rhs: list[float] = []
            for effector_name, coordinate_mask in swing_step_masks.items():
                if effector_name not in effectors:
                    continue
                contact_mask = np.asarray(
                    effectors[effector_name]["task"].contact_mask,
                    dtype=bool,
                )
                swing_pairs = ~(contact_mask[:-1] | contact_mask[1:])
                joint_ids = np.flatnonzero(coordinate_mask)
                for frame_idx in np.flatnonzero(swing_pairs):
                    current_step = (
                        current_joints[frame_idx + 1, joint_ids]
                        - current_joints[frame_idx, joint_ids]
                    )
                    active = np.abs(current_step) > swing_step_max
                    for joint_idx, step_value in zip(
                        joint_ids[active], current_step[active]
                    ):
                        row = active_swing_step_rows
                        swing_rows.extend([row, row])
                        swing_columns.extend(
                            [
                                frame_idx * num_joints + int(joint_idx),
                                (frame_idx + 1) * num_joints + int(joint_idx),
                            ]
                        )
                        swing_values.extend(
                            [-1.0 / swing_step_sigma, 1.0 / swing_step_sigma]
                        )
                        clipped_step = float(
                            np.clip(step_value, -swing_step_max, swing_step_max)
                        )
                        swing_rhs.append(
                            (clipped_step - float(step_value))
                            / swing_step_sigma
                        )
                        active_swing_step_rows += 1
            if active_swing_step_rows:
                swing_step_block = sparse.coo_matrix(
                    (
                        swing_values,
                        (swing_rows, swing_columns),
                    ),
                    shape=(
                        active_swing_step_rows,
                        num_frames * num_joints,
                    ),
                ).tocsr()
                systems.append(swing_step_block)
                residuals.append(np.asarray(swing_rhs, dtype=float))

        active_joint_step_rows = 0
        if joint_step_enabled and num_frames > 1:
            step_rows: list[int] = []
            step_columns: list[int] = []
            step_values: list[float] = []
            step_rhs: list[float] = []
            for group in joint_step_groups:
                joint_ids = np.flatnonzero(group["mask"])
                max_step = float(group["max_step"])
                sigma = float(group["sigma"])
                current_steps = np.diff(current_joints[:, joint_ids], axis=0)
                active_pairs, active_group_joints = np.nonzero(
                    np.abs(current_steps) > max_step
                )
                for frame_idx, group_joint_idx in zip(
                    active_pairs, active_group_joints
                ):
                    joint_idx = int(joint_ids[group_joint_idx])
                    step_value = float(current_steps[frame_idx, group_joint_idx])
                    row = active_joint_step_rows
                    step_rows.extend([row, row])
                    step_columns.extend(
                        [
                            int(frame_idx) * num_joints + joint_idx,
                            (int(frame_idx) + 1) * num_joints + joint_idx,
                        ]
                    )
                    step_values.extend([-1.0 / sigma, 1.0 / sigma])
                    clipped_step = float(
                        np.clip(step_value, -max_step, max_step)
                    )
                    step_rhs.append((clipped_step - step_value) / sigma)
                    active_joint_step_rows += 1
            if active_joint_step_rows:
                joint_step_block = sparse.coo_matrix(
                    (step_values, (step_rows, step_columns)),
                    shape=(active_joint_step_rows, num_frames * num_joints),
                ).tocsr()
                systems.append(joint_step_block)
                residuals.append(np.asarray(step_rhs, dtype=float))

        # Stay near the natural framewise IK solution.  This also makes the
        # normal equations strictly determined in redundant joint subspaces.
        systems.append(reference_block)
        residuals.append(
            ((reference_joints - current_joints) / reference_sigma).reshape(-1)
        )

        if posture_block is not None:
            systems.append(posture_block)
            residuals.append(
                (
                    (
                        posture_targets[posture_joint_indices][None, :]
                        - current_joints[:, posture_joint_indices]
                    )
                    / posture_sigmas[posture_joint_indices][None, :]
                ).reshape(-1)
            )

        if use_acceleration and acceleration_operator.shape[0]:
            systems.append(acceleration_block)
            residuals.append(
                (
                    -(acceleration_operator @ current_joints)
                    * temporal_joint_scale_dense
                    / acceleration_sigma
                ).reshape(-1)
            )

        if use_jerk and jerk_operator.shape[0]:
            systems.append(jerk_block)
            residuals.append(
                (
                    -(jerk_operator @ current_joints)
                    * temporal_joint_scale_dense
                    / jerk_sigma
                ).reshape(-1)
            )

        system = sparse.vstack(systems, format="csr")
        rhs_array = np.concatenate(residuals)

        # A framewise solution can sit marginally outside a joint limit.  The
        # step bounds must always admit zero, otherwise the joint limit and the
        # trust region cross over and the solver rejects the problem outright.
        lower = np.minimum(
            np.maximum(
                np.tile(joint_lower, num_frames) - current_joints.reshape(-1),
                -trust_region,
            ),
            0.0,
        )
        upper = np.maximum(
            np.minimum(
                np.tile(joint_upper, num_frames) - current_joints.reshape(-1),
                trust_region,
            ),
            0.0,
        )
        if refinement_guard is not None:
            try:
                solution = refinement_guard.solve(system, rhs_array, result, lower, upper)
            except QueryBudgetExceeded as exc:
                refinement_guard.report.update(status="query_budget", reason=str(exc))
                break
            if solution is None:
                break
        else:
            solution = lsq_linear(
                system,
                rhs_array,
                bounds=(lower, upper),
                method="trf",
                lsq_solver="lsmr",
                tol=solver_tolerance,
                lsmr_tol=solver_tolerance,
                max_iter=solver_iterations,
                verbose=0,
            )
        if not np.all(np.isfinite(solution.x)):
            raise RuntimeError("trajectory refinement produced a non-finite step")
        if int(getattr(solution, "status", 1)) == 0:
            print(
                "Trajectory refinement warning: the bounded least-squares solve "
                f"reached solver_iterations={solver_iterations} without "
                "converging, so this step is truncated."
            )
        step = solution.x.reshape(num_frames, num_joints)
        candidate = result.copy()
        candidate[:, qpos_indices] = current_joints + step
        if self_query is not None:
            candidate = guard_self_collision_update(self_query, result, candidate, qpos_indices)
        if refinement_guard is not None:
            try:
                candidate = refinement_guard.accept(result, candidate, system, rhs_array)
            except QueryBudgetExceeded as exc:
                refinement_guard.report.update(status="query_budget", reason=str(exc))
                break
        step = candidate[:, qpos_indices] - current_joints
        result = candidate
        print(
            "Trajectory refinement "
            f"{iteration + 1}/{iterations}: cost={solution.cost:.4f}, "
            f"max_joint_step={np.max(np.abs(step)):.5f}, "
            f"penetration_rows={active_penetration_rows}, "
            f"self_collision_rows={active_self_rows}, "
            f"joint_limit_rows={active_limit_rows}, "
            f"swing_step_rows={active_swing_step_rows}, "
            f"joint_step_rows={active_joint_step_rows}"
        )
        if float(np.max(np.abs(step))) < step_tolerance:
            break

    if refinement_guard is not None:
        refinement_guard.report["accepted_update_count"] = sum(
            row["accepted_alpha"] > 0.0 for row in refinement_guard.report["iterations"]
        )
        refinement_guard.report["changed_frames"] = int(np.count_nonzero(
            np.any(result != refinement_guard.validator.reference, axis=1)
        ))
    return result
