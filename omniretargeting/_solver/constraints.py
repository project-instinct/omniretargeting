"""Append contact-chain and source-relative floating-base constraints."""

from __future__ import annotations

import cvxpy as cp
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .targets import (
    _base_orientation_step_limit,
)


def _surface_overlay_coordinates(retargeter, frame):
    """Articulated ancestors of active surfaces, excluding floating roots."""
    model = retargeter.robot_model
    tasks = retargeter.surface_contact_tasks
    coordinates = set()
    for index in np.flatnonzero(tasks.active[frame]):
        geom = model.geom(tasks.bindings[int(index)].robot_geom).id
        body = int(model.geom_bodyid[geom])
        while body > 0:
            start = int(model.body_jntadr[body])
            for joint in range(start, start + int(model.body_jntnum[body])):
                kind = model.jnt_type[joint]
                if kind == mujoco.mjtJoint.mjJNT_FREE:
                    continue
                address = int(model.jnt_qposadr[joint])
                width = 4 if kind == mujoco.mjtJoint.mjJNT_BALL else 1
                coordinates.update(range(address, address + width))
            body = int(model.body_parentid[body])
    return coordinates


def build_step_constraints(
    retargeter,
    *,
    constraints,
    dqa,
    end_effector_targets,
    previous_human_joints,
    previous_previous_human_joints,
    previous_target_base_orientation,
    q,
    q_a_current,
    q_last,
    q_prev,
    target_base_orientation,
    target_base_position,
    target_human_joints,
):
    """Append contact-chain and source-relative floating-base constraints."""
    surface_frame = getattr(retargeter, "_surface_overlay_frame", None)
    if surface_frame is not None:
        allowed = _surface_overlay_coordinates(retargeter, surface_frame)
        frozen = np.flatnonzero(
            [int(index) not in allowed for index in retargeter.q_a_indices]
        )
        if len(frozen):
            constraints.append(dqa[frozen] == 0.0)
    if bool(
        retargeter.end_effector_config.get("local_contact_chain_only", False)
    ) and end_effector_targets:
        allowed_contact_qa: set[int] = set()
        localized_contact_active = False
        for name, target_data in end_effector_targets.items():
            if not bool(target_data.get("contact", False)):
                continue
            spec = retargeter.end_effectors.get(name)
            if spec is None or not len(spec["contact_joint_qa_indices"]):
                continue
            localized_contact_active = True
            allowed_contact_qa.update(
                int(index) for index in spec["contact_joint_qa_indices"]
            )
        if localized_contact_active:
            frozen_qa = np.asarray(
                [
                    index
                    for index in range(len(retargeter.q_a_indices))
                    if index not in allowed_contact_qa
                ],
                dtype=int,
            )
            if len(frozen_qa):
                constraints.append(dqa[frozen_qa] == 0.0)

    temporal_cfg = dict(retargeter.end_effector_config.get("temporal", {}) or {})
    if q_last is not None and bool(temporal_cfg.get("enabled", False)):
        # A contact phase can temporarily pull the floating base away
        # from its anatomical torso target. When that phase changes, a
        # pure task hierarchy may recover the accumulated error in one
        # frame, producing a visibly discontinuous whole-body twist. Cap
        # only the robot's *extra* angular step beyond the deliberate
        # source step, so fast captured rotations remain untouched.
        max_orientation_step_error = temporal_cfg.get(
            "max_base_orientation_step_error"
        )
        if (
            max_orientation_step_error is not None
            and target_base_orientation is not None
            and previous_target_base_orientation is not None
            and retargeter.robot_model.jnt_type[0]
            == mujoco.mjtJoint.mjJNT_FREE
        ):
            allowed_step = _base_orientation_step_limit(
                target_base_orientation,
                previous_target_base_orientation,
                q_last[3:7],
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
            base_body_id = int(retargeter.robot_model.jnt_bodyid[0])
            current_rotation = retargeter.robot_data.xmat[base_body_id].reshape(
                3, 3
            )
            last_rotation = Rotation.from_quat(
                np.asarray(q_last[3:7], dtype=float), scalar_first=True
            ).as_matrix()
            current_step = Rotation.from_matrix(
                current_rotation @ last_rotation.T
            ).as_rotvec()
            base_angular_jacobian = retargeter._body_angular_jacobian(
                base_body_id
            )
            constraints.append(
                cp.norm(
                    current_step + base_angular_jacobian @ dqa,
                    2,
                )
                <= allowed_step
            )

        base_qa = [
            int(np.where(retargeter.q_a_indices == index)[0][0])
            for index in (0, 1, 2)
            if index in retargeter.q_a_indices
        ]
        if len(base_qa) == 3:
            new_base = q_a_current[base_qa] + dqa[base_qa]
            max_base_step = max(
                float(temporal_cfg.get("max_base_step", 0.05)), 1e-6
            )
            source_base_step = np.zeros(3, dtype=float)
            source_root_name = retargeter.end_effector_config.get(
                "_position_anchor_source_joint", "Pelvis"
            )
            source_root_idx = (
                retargeter.valid_joint_names.index(source_root_name)
                if source_root_name in retargeter.valid_joint_names
                else 0
            )
            source_motion_available = (
                target_human_joints is not None
                and previous_human_joints is not None
                and source_root_idx < len(target_human_joints)
                and source_root_idx < len(previous_human_joints)
            )
            if source_motion_available:
                source_base_step = (
                    np.asarray(target_human_joints[source_root_idx], dtype=float)
                    - np.asarray(
                        previous_human_joints[source_root_idx], dtype=float
                    )
                )
            # Bound only the robot's deviation from the deliberate source
            # translation.  Bounding the absolute base step makes a fast
            # captured run or vault accumulate metres of artificial lag.
            base_step_residual = (
                new_base - q_last[:3] - source_base_step
            )
            constraints.extend(
                [
                    base_step_residual <= max_base_step,
                    -base_step_residual <= max_base_step,
                ]
            )
            if q_prev is not None:
                previous_base_step = q_last[:3] - q_prev[:3]
                new_base_step = new_base - q_last[:3]
                previous_source_base_step = np.zeros(3, dtype=float)
                previous_source_motion_available = (
                    previous_human_joints is not None
                    and previous_previous_human_joints is not None
                    and source_root_idx < len(previous_human_joints)
                    and source_root_idx < len(previous_previous_human_joints)
                )
                if previous_source_motion_available:
                    previous_source_base_step = (
                        np.asarray(
                            previous_human_joints[source_root_idx], dtype=float
                        )
                        - np.asarray(
                            previous_previous_human_joints[source_root_idx],
                            dtype=float,
                        )
                    )
                max_base_acceleration_step = max(
                    float(
                        temporal_cfg.get(
                            "max_base_acceleration_step", 0.03
                        )
                    ),
                    1e-6,
                )
                constraints.extend(
                    [
                        (new_base_step - source_base_step)
                        - (previous_base_step - previous_source_base_step)
                        <= max_base_acceleration_step,
                        (previous_base_step - previous_source_base_step)
                        - (new_base_step - source_base_step)
                        <= max_base_acceleration_step,
                    ]
                )

    # The interaction-mesh Laplacian is nearly translation invariant.  An
    # explicit morphology-aware base envelope prevents infeasible hand/foot
    # targets from dragging the floating base away from the source actor.
    base_position_obj = None
    base_cfg = dict(
        retargeter.end_effector_config.get("base_position_constraint", {}) or {}
    )
    if target_base_position is not None and bool(base_cfg.get("enabled", False)):
        base_target = np.asarray(target_base_position, dtype=float).reshape(3).copy()
        base_target[2] += float(base_cfg.get("z_offset", 0.0))
        track_xy = bool(base_cfg.get("track_xy", True))
        track_z = bool(base_cfg.get("track_z", True))
        lock_xy = bool(base_cfg.get("lock_xy", False))
        max_xy_error = max(float(base_cfg.get("max_xy_error", 0.08)), 0.0)
        max_z_error = max(float(base_cfg.get("max_z_error", 0.12)), 0.0)

        base_new_components = []
        for qpos_idx in (0, 1, 2):
            if qpos_idx in retargeter.q_a_indices:
                qa_idx = int(np.where(retargeter.q_a_indices == qpos_idx)[0][0])
                base_new_components.append(q[qpos_idx] + dqa[qa_idx])
            else:
                base_new_components.append(cp.Constant(float(q[qpos_idx])))
        base_new = cp.hstack(base_new_components)
        if track_xy and lock_xy:
            # Floating-base translation is affine in the free-joint
            # coordinates, so this is an exact path constraint rather
            # than a linearized kinematic approximation.
            constraints.append(base_new[:2] == base_target[:2])
        elif bool(base_cfg.get("hard_constraints", True)):
            if track_xy:
                constraints.append(
                    cp.norm(base_new[:2] - base_target[:2], 2)
                    <= max_xy_error
                )
            if track_z:
                constraints.extend(
                    [
                        base_new[2] - base_target[2] <= max_z_error,
                        base_target[2] - base_new[2] <= max_z_error,
                    ]
                )
        xy_scale = max(max_xy_error, 1e-6)
        z_scale = max(max_z_error, 1e-6)
        base_position_terms = []
        if track_xy and not lock_xy:
            base_position_terms.append(
                cp.sum_squares(
                    (base_new[:2] - base_target[:2]) / xy_scale
                )
            )
        if track_z:
            base_position_terms.append(
                cp.square((base_new[2] - base_target[2]) / z_scale)
            )
        if base_position_terms:
            base_position_obj = sum(base_position_terms)

    return temporal_cfg, base_cfg, base_position_obj
