"""Linearize contact, swing, body-relation and scene-penetration tasks."""

from __future__ import annotations

import mujoco
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation

from .._solver.targets import (
    _clamp_target_to_reach,
    _swing_target_z_offset,
    _transform_swing_target,
)
from ..contact import SceneContactTask, _is_surface_only_contact_mode
from .math import (
    _orthonormal_target_rotation,
    _penetration_depth_weight,
    _skew_symmetric,
    _transported_arm_segment_targets,
)


def linearize_trajectory_tasks(
    *,
    body_relation_tasks,
    self_contact_tasks=None,
    surface_contact_tasks=None,
    contact_offset_choice,
    contact_position_sigma,
    data,
    dof_indices,
    effectors,
    end_effector_config,
    jacobian_position,
    jacobian_rotation,
    model,
    num_frames,
    num_joints,
    penetration_anchor_radius,
    penetration_cfg,
    penetration_clearance,
    penetration_sigma,
    probe,
    result,
    source_joint_positions,
    source_root_positions,
    swing_position_sigma,
):
    """Linearize contact, swing, body-relation and scene-penetration tasks."""
    block_rows: list[np.ndarray] = []
    block_columns: list[np.ndarray] = []
    block_values: list[np.ndarray] = []
    block_rhs: list[float] = []
    row_index = 0

    def append_block(
        frame_idx: int,
        jacobian: np.ndarray,
        target_delta: np.ndarray,
        sigma: float,
        weight: float = 1.0,
    ) -> None:
        nonlocal row_index
        scale = np.sqrt(max(float(weight), 1e-8)) / sigma
        block = np.atleast_2d(np.asarray(jacobian, dtype=float)) * scale
        target = np.asarray(target_delta, dtype=float).reshape(-1) * scale
        local_rows, local_columns = np.nonzero(np.abs(block) > 1e-14)
        block_rows.append(local_rows + row_index)
        block_columns.append(frame_idx * num_joints + local_columns)
        block_values.append(block[local_rows, local_columns])
        block_rhs.extend(target.tolist())
        row_index += block.shape[0]

    penetration_frames = (
        probe.probe_trajectory(result)
        if probe is not None
        else [None] * num_frames
    )
    active_penetration_rows = 0

    for frame_idx in range(num_frames):
        data.qpos[:] = result[frame_idx]
        mujoco.mj_forward(model, data)
        source_root_position = (
            None
            if source_root_positions is None
            else source_root_positions[frame_idx]
        )
        if body_relation_tasks is not None:
            for weight, residual, jacobian in body_relation_tasks.rows(result[frame_idx], frame_idx, dof_indices):
                append_block(frame_idx, jacobian, -residual, 1.0, weight)
        if self_contact_tasks is not None:
            for weight, residual, jacobian in self_contact_tasks.rows(result[frame_idx], frame_idx, dof_indices):
                append_block(frame_idx, jacobian, -residual, 1.0, weight)
        if surface_contact_tasks is not None:
            for weight, residual, jacobian in surface_contact_tasks.rows(result[frame_idx], frame_idx, dof_indices):
                append_block(frame_idx, jacobian, -residual, 1.0, weight)
        root_z_offset = _swing_target_z_offset(
            end_effector_config,
            robot_root_z=float(result[frame_idx, 2]),
            source_root_position=source_root_position,
        )
        for name, spec in effectors.items():
            task: SceneContactTask = spec["task"]
            body_id = int(spec["body_id"])
            rotation = data.xmat[body_id].reshape(3, 3).copy()
            in_contact = bool(task.contact_mask[frame_idx])
            if in_contact:
                contact_mode = (
                    str(task.contact_modes[frame_idx])
                    if task.contact_modes is not None
                    else "sticking"
                )
                target_position = task.contact_positions[frame_idx]
                offset = spec["contact_offsets"][
                    contact_offset_choice[name][frame_idx]
                ]
                position_sigma = (
                    contact_position_sigma
                    if spec["contact_position_sigma"] is None
                    else float(spec["contact_position_sigma"])
                )
                confidence = max(
                    float(task.contact_confidence[frame_idx]), 0.25
                )
                if task.transition_weights is not None:
                    confidence *= max(
                        float(task.transition_weights[frame_idx]), 1e-3
                    )
            else:
                contact_mode = "none"
                target_position = _transform_swing_target(
                    task.tracking_positions[frame_idx],
                    source_root_position,
                    root_z_offset,
                    float(spec["swing_morphology_scale"]),
                    bool(spec["preserve_world_target"]),
                )
                reach_body_id = int(spec["reach_reference_body_id"])
                if reach_body_id >= 0:
                    reach_rotation = data.xmat[reach_body_id].reshape(3, 3)
                    reach_origin = (
                        data.xpos[reach_body_id]
                        + reach_rotation @ spec["reach_reference_offset"]
                    )
                    target_position = _clamp_target_to_reach(
                        target_position,
                        reach_origin,
                        spec["max_swing_reach"],
                    )
                offset = spec["tracking_offset"]
                position_sigma = (
                    swing_position_sigma
                    if spec["swing_position_sigma"] is None
                    else float(spec["swing_position_sigma"])
                )
                confidence = 1.0

            point = data.xpos[body_id] + rotation @ offset
            jacobian_position.fill(0.0)
            jacobian_rotation.fill(0.0)
            mujoco.mj_jac(
                model,
                data,
                jacobian_position,
                jacobian_rotation,
                point,
                body_id,
            )
            position_delta = target_position - point
            if in_contact and _is_surface_only_contact_mode(contact_mode):
                # A surface contact has only one positional coordinate, so
                # it can be held more tightly than a full XYZ endpoint
                # without overconstraining the limb.  Keep this as an
                # internal semantic default: callers still tune the usual
                # contact sigma, while noisy-video hands get reliable
                # normal placement without another profile knob.
                surface_position_sigma = min(position_sigma, 0.006)
                surface_normal = np.asarray(
                    task.surface_normals[frame_idx].copy(), dtype=float
                )
                surface_normal /= max(
                    float(np.linalg.norm(surface_normal)), 1e-10
                )
                append_block(
                    frame_idx,
                    surface_normal[None, :]
                    @ jacobian_position[:, dof_indices],
                    np.asarray([surface_normal @ position_delta]),
                    surface_position_sigma,
                    confidence,
                )
                # Contact confidence and transition fades may soften how
                # tightly an uncertain video marker is pulled to the
                # surface, but they must never soften the outside
                # half-space itself.  Add a one-sided high-priority row
                # only while the selected hand/foot marker is deeper than
                # the accepted surface tolerance.  This avoids treating a
                # noisy penetrated source marker as permission for the
                # robot to enter the reconstructed scene.
                normal_tolerance = max(
                    float(
                        end_effector_config.get(
                            "contact_normal_tolerance", 0.005
                        )
                    ),
                    0.0,
                )
                inward_violation = (
                    float(surface_normal @ position_delta)
                    - normal_tolerance
                )
                if inward_violation > 0.0:
                    append_block(
                        frame_idx,
                        surface_normal[None, :]
                        @ jacobian_position[:, dof_indices],
                        np.asarray([inward_violation]),
                        min(surface_position_sigma, 0.002),
                    )
                if (
                    contact_mode == "sliding"
                    and spec["track_sliding_tangent"]
                ):
                    tangent_projection = np.eye(3) - np.outer(
                        surface_normal, surface_normal
                    )
                    append_block(
                        frame_idx,
                        tangent_projection
                        @ jacobian_position[:, dof_indices],
                        tangent_projection @ position_delta,
                        max(
                            float(spec["sliding_tangent_sigma"]),
                            position_sigma,
                        ),
                        confidence,
                    )
            elif in_contact or spec["track_swing"]:
                append_block(
                    frame_idx,
                    jacobian_position[:, dof_indices],
                    position_delta,
                    position_sigma,
                    confidence,
                )

            arm_posture = spec["arm_posture"]
            if arm_posture is not None and source_joint_positions is not None:
                shoulder_body_id = int(arm_posture["shoulder_body_id"])
                elbow_body_id = int(arm_posture["elbow_body_id"])
                shoulder_position = data.xpos[shoulder_body_id].copy()
                elbow_position = data.xpos[elbow_body_id].copy()
                current_upper = elbow_position - shoulder_position
                source_frame = source_joint_positions[frame_idx]
                wrist_body_id = int(arm_posture["wrist_body_id"])
                use_wrist_center = wrist_body_id >= 0
                source_endpoint = (
                    source_frame[int(arm_posture["source_wrist_idx"])]
                    if use_wrist_center
                    else task.tracking_positions[frame_idx]
                )
                shape_endpoint_target = _transform_swing_target(
                    source_endpoint,
                    source_root_position,
                    root_z_offset,
                    float(spec["swing_morphology_scale"]),
                    bool(spec["preserve_world_target"]),
                )
                target_upper, target_forearm = (
                    _transported_arm_segment_targets(
                        source_frame[int(arm_posture["source_shoulder_idx"])],
                        source_frame[int(arm_posture["source_elbow_idx"])],
                        source_endpoint,
                        shoulder_position,
                        shape_endpoint_target,
                        float(arm_posture["upper_arm_length"]),
                        float(arm_posture["forearm_length"]),
                    )
                )
                shoulder_jacobian = np.zeros_like(jacobian_position)
                elbow_jacobian = np.zeros_like(jacobian_position)
                mujoco.mj_jac(
                    model,
                    data,
                    shoulder_jacobian,
                    None,
                    shoulder_position,
                    shoulder_body_id,
                )
                mujoco.mj_jac(
                    model,
                    data,
                    elbow_jacobian,
                    None,
                    elbow_position,
                    elbow_body_id,
                )
                shape_endpoint_position = (
                    data.xpos[wrist_body_id].copy()
                    if use_wrist_center
                    else data.xpos[body_id]
                    + rotation @ spec["tracking_offset"]
                )
                shape_endpoint_body_id = (
                    wrist_body_id if use_wrist_center else body_id
                )
                shape_endpoint_jacobian = np.zeros_like(jacobian_position)
                mujoco.mj_jac(
                    model,
                    data,
                    shape_endpoint_jacobian,
                    None,
                    shape_endpoint_position,
                    shape_endpoint_body_id,
                )
                current_forearm = shape_endpoint_position - elbow_position
                append_block(
                    frame_idx,
                    (elbow_jacobian - shoulder_jacobian)[:, dof_indices],
                    target_upper - current_upper,
                    float(arm_posture["upper_direction_sigma"]),
                )
                append_block(
                    frame_idx,
                    (shape_endpoint_jacobian - elbow_jacobian)[:, dof_indices],
                    target_forearm - current_forearm,
                    float(arm_posture["forearm_direction_sigma"]),
                )

            if spec["track_orientation"]:
                target_rotation = _orthonormal_target_rotation(
                    None
                    if task.forward_directions is None
                    else task.forward_directions[frame_idx],
                    None
                    if task.up_directions is None
                    else task.up_directions[frame_idx],
                )
                if target_rotation is not None:
                    semantic_rotation = (
                        rotation @ spec["orientation_local_rotation"]
                    )
                    orientation_jacobian = jacobian_rotation[
                        :, dof_indices
                    ].copy()
                    orientation_coordinate_mask = spec[
                        "orientation_coordinate_mask"
                    ]
                    if orientation_coordinate_mask is not None:
                        orientation_jacobian[
                            :, ~orientation_coordinate_mask
                        ] = 0.0
                    orientation_weight = (
                        float(spec["contact_orientation_weight"])
                        if in_contact
                        else float(spec["swing_orientation_weight"])
                    )
                    orientation_weight *= confidence
                    if (
                        orientation_weight > 0.0
                        and spec["orientation_mode"] == "axes"
                    ):
                        current_forward = semantic_rotation[:, 0]
                        current_up = semantic_rotation[:, 2]
                        # For a world-frame angular increment dtheta,
                        # dv = dtheta x v = -skew(v) dtheta.  Splitting
                        # the two anatomical axes lets finger direction
                        # remain more important than palm roll, instead
                        # of forcing an unreachable full SO(3) target.
                        append_block(
                            frame_idx,
                            -_skew_symmetric(current_forward)
                            @ orientation_jacobian,
                            target_rotation[:, 0] - current_forward,
                            float(spec["forward_orientation_sigma"]),
                            orientation_weight,
                        )
                        append_block(
                            frame_idx,
                            -_skew_symmetric(current_up)
                            @ orientation_jacobian,
                            target_rotation[:, 2] - current_up,
                            float(spec["up_orientation_sigma"]),
                            orientation_weight,
                        )
                    elif orientation_weight > 0.0:
                        rotation_error = Rotation.from_matrix(
                            target_rotation @ semantic_rotation.T
                        ).as_rotvec()
                        append_block(
                            frame_idx,
                            orientation_jacobian,
                            rotation_error,
                            max(float(spec["orientation_sigma"]), 1e-6),
                            orientation_weight,
                        )

        frame_penetration = penetration_frames[frame_idx]
        if frame_penetration is None:
            continue
        # Only a genuinely sticking effector is fixed to a full XYZ scene
        # anchor.  Transition and sliding contacts constrain the surface
        # normal only, so their local chain must remain available to move
        # colliding geometry tangentially out of the scene.  The sticking
        # exemption is by distance, not by body id: colliding geoms often
        # live on the wrist/ankle parent body rather than the configured
        # effector body.  The honest depth still comes from
        # measure_scene_penetration.
        active_anchor_points = [
            np.asarray(spec["task"].contact_positions[frame_idx], dtype=float)
            for spec in effectors.values()
            if penetration_anchor_radius > 0.0
            and bool(spec["task"].contact_mask[frame_idx])
            and (
                spec["task"].contact_modes is None
                or str(spec["task"].contact_modes[frame_idx]) == "sticking"
            )
        ]
        # One-sided: only points that violate the requested clearance get a
        # row, and the active set is rebuilt from the current pose at every
        # Gauss--Newton iteration.
        for index in np.flatnonzero(
            (frame_penetration["clearance"] < penetration_clearance)
            & ~frame_penetration["uncertain"]
        ):
            if any(
                float(
                    np.linalg.norm(
                        frame_penetration["points"][index] - anchor
                    )
                )
                < penetration_anchor_radius
                for anchor in active_anchor_points
            ):
                continue
            jacobian_position.fill(0.0)
            mujoco.mj_jac(
                model,
                data,
                jacobian_position,
                None,
                frame_penetration["points"][index],
                int(frame_penetration["body_ids"][index]),
            )
            append_block(
                frame_idx,
                frame_penetration["directions"][index]
                @ jacobian_position[:, dof_indices],
                penetration_clearance
                - frame_penetration["clearance"][index],
                penetration_sigma,
                _penetration_depth_weight(
                    frame_penetration["clearance"][index],
                    penetration_cfg,
                ),
            )
            active_penetration_rows += 1

    task_block = sparse.coo_matrix(
        (
            np.concatenate(block_values)
            if block_values
            else np.zeros(0, dtype=float),
            (
                np.concatenate(block_rows)
                if block_rows
                else np.zeros(0, dtype=int),
                np.concatenate(block_columns)
                if block_columns
                else np.zeros(0, dtype=int),
            ),
        ),
        shape=(row_index, num_frames * num_joints),
    ).tocsr()

    return task_block, block_rhs, active_penetration_rows
