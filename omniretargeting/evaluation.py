"""Quantitative evaluation for a final retargeted robot trajectory.

The optimizer's own least-squares cost mixes metres, radians, temporal
derivatives, and active collision samples.  It is therefore not an acceptance
metric.  This module measures the final kinematic result in physical units and
keeps source tracking, contact quality, temporal quality, and penetration as
separate signals.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import mujoco
import numpy as np

from .contact import ContactMode, SceneContactTask
from .trajectory_refinement import measure_scene_penetration


def distribution(values: np.ndarray | list[float]) -> dict[str, float | int]:
    """Return a stable mean/P95/max summary for finite scalar samples."""
    samples = np.asarray(values, dtype=float).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0, "count": 0}
    return {
        "mean": float(np.mean(samples)),
        "p95": float(np.percentile(samples, 95.0)),
        "max": float(np.max(samples)),
        "count": int(len(samples)),
    }


def _split_statistics(
    values: np.ndarray, contact_mask: np.ndarray
) -> dict[str, dict[str, float | int]]:
    values = np.asarray(values, dtype=float)
    contact_mask = np.asarray(contact_mask, dtype=bool)
    return {
        "all": distribution(values),
        "swing": distribution(values[~contact_mask]),
        "contact": distribution(values[contact_mask]),
    }


def _iter_true_runs(mask: np.ndarray):
    start = None
    for index, value in enumerate(np.r_[np.asarray(mask, dtype=bool), False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            yield start, index
            start = None


def _scalar_joint_qpos_indices(model: mujoco.MjModel) -> np.ndarray:
    indices = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            continue
        qpos_index = int(model.jnt_qposadr[joint_id])
        if qpos_index >= 7:
            indices.append(qpos_index)
    return np.asarray(indices, dtype=int)


def _robot_body_positions(
    model: mujoco.MjModel, trajectory: np.ndarray
) -> np.ndarray:
    data = mujoco.MjData(model)
    positions = np.zeros((len(trajectory), model.nbody, 3), dtype=float)
    for frame_idx, qpos in enumerate(trajectory):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        positions[frame_idx] = data.xpos
    return positions


def _vector_angle(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return unsigned row-wise angles, with degenerate vectors marked NaN."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first_norm = np.linalg.norm(first, axis=1)
    second_norm = np.linalg.norm(second, axis=1)
    valid = (first_norm > 1e-10) & (second_norm > 1e-10)
    result = np.full(len(first), np.nan, dtype=float)
    cosine = np.sum(first[valid] * second[valid], axis=1)
    cosine /= first_norm[valid] * second_norm[valid]
    result[valid] = np.arccos(np.clip(cosine, -1.0, 1.0))
    return result


def _elbow_bend_direction(points: np.ndarray) -> np.ndarray:
    """Return the elbow's direction away from the shoulder--wrist axis."""
    points = np.asarray(points, dtype=float)
    chord = points[:, 2] - points[:, 0]
    chord_norm_sq = np.sum(chord * chord, axis=1, keepdims=True)
    shoulder_to_elbow = points[:, 1] - points[:, 0]
    projection_scale = np.sum(
        shoulder_to_elbow * chord, axis=1, keepdims=True
    ) / np.maximum(chord_norm_sq, 1e-20)
    return shoulder_to_elbow - projection_scale * chord


def _arm_pose_fidelity_metrics(
    model: mujoco.MjModel,
    trajectory: np.ndarray,
    source_joint_positions: np.ndarray,
    source_joint_names: list[str] | tuple[str, ...],
    specs: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare bilateral robot arm geometry with the captured source pose.

    The bend-plane metric deliberately compares the two world-space elbow-bend
    directions after projecting each off its own shoulder--wrist axis.  This
    exposes a visibly different elbow branch even when endpoint IK succeeds.
    """
    source = np.asarray(source_joint_positions, dtype=float)
    if source.ndim != 3 or source.shape != (
        len(trajectory),
        len(source_joint_names),
        3,
    ):
        raise ValueError(
            "source_joint_positions must have shape "
            f"({len(trajectory)}, {len(source_joint_names)}, 3), got "
            f"{source.shape}."
        )
    source_indices = {
        str(name): index for index, name in enumerate(source_joint_names)
    }
    robot_positions = _robot_body_positions(model, trajectory)
    result: dict[str, Any] = {}
    for effector_name, raw_spec in specs.items():
        if "hand" not in str(effector_name).lower():
            continue
        arm = dict((raw_spec or {}).get("arm_posture", {}) or {})
        if not bool(arm.get("enabled", False)):
            continue
        robot_names = [
            str(arm[key])
            for key in ("shoulder_body", "elbow_body", "wrist_body")
        ]
        source_names = [
            str(arm[key])
            for key in ("source_shoulder", "source_elbow", "source_wrist")
        ]
        robot_ids = [
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            for body_name in robot_names
        ]
        if any(body_id < 0 for body_id in robot_ids):
            missing = [
                name for name, body_id in zip(robot_names, robot_ids)
                if body_id < 0
            ]
            raise ValueError(
                f"{effector_name}: arm-pose evaluation body is missing: "
                + ", ".join(missing)
            )
        missing_source = [
            name for name in source_names if name not in source_indices
        ]
        if missing_source:
            raise ValueError(
                f"{effector_name}: arm-pose source joint is missing: "
                + ", ".join(missing_source)
            )

        robot_arm = robot_positions[:, robot_ids]
        source_arm = source[
            :, [source_indices[name] for name in source_names]
        ]
        robot_upper = robot_arm[:, 1] - robot_arm[:, 0]
        robot_forearm = robot_arm[:, 2] - robot_arm[:, 1]
        source_upper = source_arm[:, 1] - source_arm[:, 0]
        source_forearm = source_arm[:, 2] - source_arm[:, 1]
        robot_flexion = _vector_angle(robot_upper, robot_forearm)
        source_flexion = _vector_angle(source_upper, source_forearm)
        arm_name = str(effector_name).replace("_hand", "_arm")
        result[arm_name] = {
            "upper_arm_direction_error_rad": distribution(
                _vector_angle(robot_upper, source_upper)
            ),
            "forearm_direction_error_rad": distribution(
                _vector_angle(robot_forearm, source_forearm)
            ),
            "elbow_bend_plane_error_rad": distribution(
                _vector_angle(
                    _elbow_bend_direction(robot_arm),
                    _elbow_bend_direction(source_arm),
                )
            ),
            "elbow_flexion_difference_rad": distribution(
                np.abs(robot_flexion - source_flexion)
            ),
            "robot_elbow_flexion_rad": distribution(robot_flexion),
            "source_elbow_flexion_rad": distribution(source_flexion),
        }
    return result


def _quaternion_angle(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first = first / np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
    second = second / np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-12)
    cosine = np.abs(np.sum(first * second, axis=1))
    return 2.0 * np.arccos(np.clip(cosine, -1.0, 1.0))


def _effector_series(
    model: mujoco.MjModel,
    trajectory: np.ndarray,
    task: SceneContactTask,
    spec: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return semantic marker and run-stable contact-point trajectories."""
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, str(spec["robot_body"])
    )
    if body_id < 0:
        raise ValueError(
            f"Evaluation effector '{task.name}' refers to missing body "
            f"'{spec['robot_body']}'."
        )
    tracking_offset = np.asarray(
        spec.get("tracking_offset", [0.0, 0.0, 0.0]), dtype=float
    ).reshape(3)
    contact_offsets = np.asarray(
        spec.get("contact_offsets", [tracking_offset]), dtype=float
    )
    if contact_offsets.ndim == 1:
        contact_offsets = contact_offsets[None, :]

    data = mujoco.MjData(model)
    tracking = np.zeros((len(trajectory), 3), dtype=float)
    candidates = np.zeros(
        (len(trajectory), len(contact_offsets), 3), dtype=float
    )
    for frame_idx, qpos in enumerate(trajectory):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        rotation = data.xmat[body_id].reshape(3, 3)
        origin = data.xpos[body_id]
        tracking[frame_idx] = origin + rotation @ tracking_offset
        candidates[frame_idx] = origin[None, :] + contact_offsets @ rotation.T

    selected = tracking.copy()
    contact_mask = task.contact_modes != int(ContactMode.NONE)
    if task.active_contact_offset_ids is not None:
        for frame_idx in np.flatnonzero(contact_mask):
            offset_idx = int(task.active_contact_offset_ids[frame_idx])
            if offset_idx >= len(contact_offsets):
                raise ValueError(
                    f"{task.name}: active contact offset {offset_idx} exceeds "
                    f"the configured {len(contact_offsets)} candidates."
                )
            selected[frame_idx] = candidates[frame_idx, offset_idx]
    else:
        for segment_id in np.unique(task.contact_segment_ids[contact_mask]):
            segment = task.contact_segment_ids == segment_id
            segment_errors = np.linalg.norm(
                candidates[segment]
                - task.contact_positions[segment, None, :],
                axis=2,
            )
            offset_idx = int(np.argmin(np.mean(segment_errors, axis=0)))
            selected[segment] = candidates[segment, offset_idx]
    return tracking, selected


def _effective_target_series(
    model: mujoco.MjModel,
    trajectory: np.ndarray,
    task: SceneContactTask,
) -> np.ndarray:
    """Evaluate the optimizer's active target at the solved robot pose.

    Absolute swing modes use their prepared world preview.  A limb-relative
    mode instead recomputes its shoulder/hip anchor with FK for every solved
    frame, then applies the same relative-to-world bridge used by both solvers.
    """
    if task.effective_swing_positions is None:
        raise RuntimeError(
            f"{task.name}: effective swing targets were not prepared."
        )
    targets = np.asarray(task.effective_swing_positions, dtype=float).copy()
    contact_mask = task.contact_modes != int(ContactMode.NONE)
    targets[contact_mask] = task.contact_positions[contact_mask]
    if task.swing_limb_offsets is None:
        return targets

    anchor_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        str(task.swing_anchor_body),
    )
    if anchor_body_id < 0:
        raise ValueError(
            f"{task.name}: live swing anchor body "
            f"'{task.swing_anchor_body}' is missing from the robot model."
        )
    anchor_offset = (
        np.zeros(3, dtype=float)
        if task.swing_anchor_offset is None
        else np.asarray(task.swing_anchor_offset, dtype=float).reshape(3)
    )
    data = mujoco.MjData(model)
    anchors = np.zeros((len(trajectory), 3), dtype=float)
    for frame_idx, qpos in enumerate(trajectory):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        rotation = data.xmat[anchor_body_id].reshape(3, 3)
        anchors[frame_idx] = (
            data.xpos[anchor_body_id] + rotation @ anchor_offset
        )
    alpha = np.asarray(task.contact_bridge_alpha, dtype=float)
    return (
        (1.0 - alpha[:, None])
        * (anchors + np.asarray(task.swing_limb_offsets, dtype=float))
        + alpha[:, None]
        * np.asarray(task.contact_bridge_positions, dtype=float)
    )


def _foot_sliding_metrics(
    contact_points: np.ndarray,
    surface_normals: np.ndarray,
    contact_mask: np.ndarray,
    framerate: float,
) -> dict[str, dict[str, float | int]]:
    displacements: list[float] = []
    speeds: list[float] = []
    for start, end in _iter_true_runs(contact_mask):
        if end - start < 2:
            continue
        anchor = contact_points[start]
        for frame_idx in range(start, end):
            normal = np.asarray(surface_normals[frame_idx], dtype=float)
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            offset = contact_points[frame_idx] - anchor
            tangent = offset - normal * float(np.dot(offset, normal))
            displacements.append(float(np.linalg.norm(tangent)))
        for frame_idx in range(start + 1, end):
            normal = surface_normals[frame_idx - 1] + surface_normals[frame_idx]
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            step = contact_points[frame_idx] - contact_points[frame_idx - 1]
            tangent_step = step - normal * float(np.dot(step, normal))
            speeds.append(float(np.linalg.norm(tangent_step)) * framerate)
    return {
        "tangential_displacement_m": distribution(displacements),
        "tangential_speed_mps": distribution(speeds),
    }


def _deviation_metrics(
    model: mujoco.MjModel,
    reference: np.ndarray,
    result: np.ndarray,
) -> dict[str, Any]:
    reference = np.asarray(reference, dtype=float)
    result = np.asarray(result, dtype=float)
    if reference.shape != result.shape:
        raise ValueError(
            "reference_trajectory and trajectory must have identical shapes, got "
            f"{reference.shape} and {result.shape}."
        )
    joint_indices = _scalar_joint_qpos_indices(model)
    joint_delta = (
        result[:, joint_indices] - reference[:, joint_indices]
        if len(joint_indices)
        else np.zeros((len(result), 0), dtype=float)
    )
    reference_bodies = _robot_body_positions(model, reference)
    result_bodies = _robot_body_positions(model, result)
    reference_relative = reference_bodies - reference[:, None, :3]
    result_relative = result_bodies - result[:, None, :3]
    body_error = np.linalg.norm(result_relative - reference_relative, axis=2)
    body_delta = result_relative - reference_relative
    return {
        "root_translation_m": distribution(
            np.linalg.norm(result[:, :3] - reference[:, :3], axis=1)
        ),
        "root_rotation_rad": distribution(
            _quaternion_angle(result[:, 3:7], reference[:, 3:7])
        ),
        "joint_l2_rad": distribution(
            np.linalg.norm(joint_delta, axis=1)
            if joint_delta.shape[1]
            else np.zeros(len(result), dtype=float)
        ),
        "joint_absolute_rad": distribution(np.abs(joint_delta)),
        "whole_body_root_relative_m": distribution(body_error),
        "whole_body_horizontal_correction_m": distribution(
            np.linalg.norm(body_delta[..., :2], axis=2)
        ),
        "whole_body_vertical_correction_m": distribution(
            np.abs(body_delta[..., 2])
        ),
    }


def _temporal_metrics(
    model: mujoco.MjModel, trajectory: np.ndarray, framerate: float
) -> dict[str, Any]:
    joint_indices = _scalar_joint_qpos_indices(model)
    joints = np.asarray(trajectory, dtype=float)[:, joint_indices]
    velocity = np.diff(joints, axis=0) * framerate
    acceleration = np.diff(joints, n=2, axis=0) * framerate**2
    return {
        "joint_velocity_rad_s": distribution(np.abs(velocity)),
        "joint_acceleration_rad_s2": distribution(np.abs(acceleration)),
    }


def evaluate_retargeted_motion(
    model: mujoco.MjModel,
    trajectory: np.ndarray,
    tasks: Mapping[str, SceneContactTask] | None,
    end_effector_config: Mapping[str, Any] | None,
    *,
    framerate: float,
    reference_trajectory: np.ndarray | None = None,
    source_root_positions: np.ndarray | None = None,
    source_joint_positions: np.ndarray | None = None,
    source_joint_names: list[str] | tuple[str, ...] | None = None,
    scene_mesh: Any | None = None,
    penetration_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure a final robot trajectory without reusing optimizer loss values."""
    trajectory = np.asarray(trajectory, dtype=float)
    fps = max(float(framerate), 1e-8)
    metrics: dict[str, Any] = {
        "units": {
            "position": "m",
            "linear_velocity": "m/s",
            "angle": "rad",
            "angular_velocity": "rad/s",
        },
        "num_frames": int(len(trajectory)),
        "end_effectors": {},
        "temporal": _temporal_metrics(model, trajectory, fps),
    }

    specs = dict((end_effector_config or {}).get("effectors", {}) or {})
    if source_joint_positions is not None:
        if source_joint_names is None:
            raise ValueError(
                "source_joint_names are required for arm-pose evaluation."
            )
        metrics["arm_pose_fidelity"] = _arm_pose_fidelity_metrics(
            model,
            trajectory,
            source_joint_positions,
            source_joint_names,
            specs,
        )
    for name, task in dict(tasks or {}).items():
        if name not in specs:
            continue
        tracking, contact_points = _effector_series(
            model, trajectory, task, dict(specs[name] or {})
        )
        contact_mask = task.contact_modes != int(ContactMode.NONE)
        source_error = np.linalg.norm(
            tracking - task.tracking_positions, axis=1
        )
        active_error = source_error.copy()
        active_error[contact_mask] = np.linalg.norm(
            contact_points[contact_mask]
            - task.contact_positions[contact_mask],
            axis=1,
        )
        effective_targets = _effective_target_series(
            model, trajectory, task
        )
        effective_points = tracking.copy()
        effective_points[contact_mask] = contact_points[contact_mask]
        effective_error = np.linalg.norm(
            effective_points - effective_targets, axis=1
        )
        effector_metrics: dict[str, Any] = {
            "source_world_error_m": _split_statistics(
                source_error, contact_mask
            ),
            "active_target_error_m": _split_statistics(
                active_error, contact_mask
            ),
            "effective_target_error_m": _split_statistics(
                effective_error, contact_mask
            ),
            "contact_residual_m": distribution(
                active_error[contact_mask]
            ),
            "contact_residual_by_mode_m": {
                mode.name.lower(): distribution(
                    active_error[task.contact_modes == int(mode)]
                )
                for mode in (
                    ContactMode.TOUCH,
                    ContactMode.SLIDE,
                    ContactMode.SUPPORT,
                )
            },
        }
        if "foot" in name.lower():
            effector_metrics["foot_sliding"] = _foot_sliding_metrics(
                contact_points,
                task.surface_normals,
                contact_mask,
                fps,
            )
        metrics["end_effectors"][name] = effector_metrics

    if source_root_positions is not None:
        source_root = np.asarray(source_root_positions, dtype=float)
        if source_root.shape == (len(trajectory), 3):
            base_cfg = dict(
                (end_effector_config or {}).get("base_position_constraint", {})
                or {}
            )
            target = source_root.copy()
            target[:, 2] += float(base_cfg.get("z_offset", 0.0))
            metrics["source_root_target_error_m"] = distribution(
                np.linalg.norm(trajectory[:, :3] - target, axis=1)
            )

    if reference_trajectory is not None:
        metrics["deviation_from_pre_refinement"] = _deviation_metrics(
            model, np.asarray(reference_trajectory, dtype=float), trajectory
        )

    if scene_mesh is not None:
        penetration = measure_scene_penetration(
            model,
            trajectory,
            scene_mesh,
            penetration_config,
        )
        per_frame = np.asarray(penetration["per_frame_depth"], dtype=float)
        positive = per_frame[per_frame > 1e-6]
        penetration["p95_depth"] = (
            float(np.percentile(positive, 95.0)) if len(positive) else 0.0
        )
        penetration["frame_ratio"] = (
            float(penetration["frames_penetrating"]) / max(len(trajectory), 1)
        )
        metrics["penetration"] = penetration

    return metrics


def metrics_to_json(metrics: Mapping[str, Any]) -> str:
    """Serialize metrics, including NumPy arrays, for an NPZ metadata field."""
    def convert(value: Any):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot serialize metric value of type {type(value)!r}")

    return json.dumps(metrics, default=convert, sort_keys=True)
