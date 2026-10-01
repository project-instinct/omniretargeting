"""Target transforms, orientation step bounds and joint limits."""

from __future__ import annotations

from typing import Any, Mapping

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def _source_orientation_step(target_data):
    """World angular step of two immutable semantic frames, or no evidence.

    A fixed local semantic-to-body calibration cancels in R(t) R(t-1)^T.
    This measures intentional source motion without assuming a robot name or
    treating a single normal as a complete orientation.
    """
    from .._refinement.math import _orthonormal_target_rotation

    current = _orthonormal_target_rotation(
        target_data.get("forward_direction"), target_data.get("up_direction")
    )
    previous = _orthonormal_target_rotation(
        target_data.get("previous_forward_direction"),
        target_data.get("previous_up_direction"),
    )
    if current is None or previous is None:
        return None
    return Rotation.from_matrix(current @ previous.T).as_rotvec()


def _swing_target_z_offset(
    end_effector_config: Mapping[str, Any],
    *,
    robot_root_z: float | None = None,
    source_root_position: np.ndarray | None = None,
) -> float:
    """Resolve robot/source root registration for a swing target.

    An explicit value remains available for unusual tools or compatibility.
    Normally the offset is measured from the current robot root and source
    torso, so different bodies and poses do not share one tuned constant.
    """
    cfg = dict(end_effector_config or {})
    explicit = cfg.get("swing_target_z_offset")
    if explicit is not None:
        return float(explicit)
    if robot_root_z is not None and source_root_position is not None:
        source_root = np.asarray(source_root_position, dtype=float).reshape(3)
        return float(robot_root_z) - float(source_root[2])
    return 0.0


def _world_target_regime(spec, target):
    """Observed clearance decides the frame; explicit overrides are prepared upstream."""
    return bool(target.get("target_world", spec.get("preserve_world_target", False)))


def _transform_swing_target(
    tracking_position: np.ndarray,
    source_root_position: np.ndarray | None,
    robot_root_z_offset: float,
    morphology_scale: float,
    preserve_world_target: bool = False,
    *,
    robot_root_position: np.ndarray | None = None,
) -> np.ndarray:
    """Express a source end-effector target around the robot root target.

    Free hands follow robot root translation, including horizontal reach
    adaptation. Feet retain their world targets. The explicit height offset
    remains supported; omitting robot_root_position retains legacy XY behavior.
    """
    target = np.asarray(tracking_position, dtype=float).reshape(3)
    if preserve_world_target or source_root_position is None:
        return target.copy()
    source_root = np.asarray(source_root_position, dtype=float).reshape(3)
    robot_root_target = source_root.copy()
    if robot_root_position is not None:
        robot_root_target[:2] = np.asarray(robot_root_position, dtype=float)[:2]
    robot_root_target[2] += float(robot_root_z_offset)
    return robot_root_target + float(morphology_scale) * (target - source_root)


def _approach_contact_target(target, target_data):
    """Hand a free target over to its contact anchor by source progress.

    Returns the blended target and the share that still moves with the free
    target, which scales that target's own Jacobian.
    """
    weight = float(target_data.get("approach_weight", 0.0))
    if weight <= 0.0:
        return target, 1.0
    reference = np.asarray(target_data["approach_position"], dtype=float).reshape(3)
    return target + weight * (reference - target), 1.0 - weight


def _calibrated_orientation_target(
    source_current: np.ndarray,
    source_reference: np.ndarray,
    robot_reference: np.ndarray,
) -> np.ndarray:
    """Map a source world orientation into a calibrated robot body frame.

    Source skeletons and robot links rarely use the same local axes.  The
    constant right-side offset aligns their reference poses, while the source
    world rotation still supplies the complete time-varying body motion.  This
    is the frame-task convention used by standard whole-body IK retargeters.
    """
    source_current = np.asarray(source_current, dtype=float).reshape(3, 3)
    source_reference = np.asarray(source_reference, dtype=float).reshape(3, 3)
    robot_reference = np.asarray(robot_reference, dtype=float).reshape(3, 3)
    return source_current @ source_reference.T @ robot_reference


def _relative_orientation_step_magnitude(
    current_wxyz: np.ndarray,
    previous_wxyz: np.ndarray,
) -> float:
    """Return the shortest SO(3) step between two scalar-first quaternions."""
    current = Rotation.from_quat(
        np.asarray(current_wxyz, dtype=float).reshape(4), scalar_first=True
    )
    previous = Rotation.from_quat(
        np.asarray(previous_wxyz, dtype=float).reshape(4), scalar_first=True
    )
    return float((previous.inv() * current).magnitude())


def _limit_relative_orientation_step(
    current_wxyz: np.ndarray,
    previous_wxyz: np.ndarray,
    max_step: float,
) -> np.ndarray:
    """Project a scalar-first quaternion onto an exact shortest-arc step.

    The convex subproblem constrains a first-order angular-velocity model. A
    later trust-region projection and quaternion normalization can otherwise
    leave the accepted nonlinear rotation just outside that model's cone.
    Applying the same envelope on SO(3) makes the per-frame guarantee exact;
    subsequent SQP iterations can still recover contact accuracy with the
    remaining joints.
    """
    current = Rotation.from_quat(
        np.asarray(current_wxyz, dtype=float).reshape(4), scalar_first=True
    )
    previous_quaternion = np.asarray(previous_wxyz, dtype=float).reshape(4)
    previous = Rotation.from_quat(previous_quaternion, scalar_first=True)
    delta = previous.inv() * current
    delta_rotvec = delta.as_rotvec()
    magnitude = float(np.linalg.norm(delta_rotvec))
    limit = max(float(max_step), 0.0)
    if magnitude > limit and magnitude > 1e-12:
        current = previous * Rotation.from_rotvec(
            delta_rotvec * (limit / magnitude)
        )
    projected = current.as_quat(scalar_first=True)
    if np.dot(projected, previous_quaternion) < 0.0:
        projected *= -1.0
    return projected


def _base_orientation_step_limit(
    target_wxyz: np.ndarray,
    previous_target_wxyz: np.ndarray,
    previous_robot_wxyz: np.ndarray,
    base_extra_step: float,
    catchup_gain: float = 0.0,
    catchup_deadband: float = 0.0,
    max_extra_step: float | None = None,
    max_total_step: float | None = None,
) -> float:
    """Return a source-aware angular rate limit with bounded error feedback.

    A fixed rate limit prevents snapping but can leave a torso visibly lagging
    after an incompatible contact phase. The proportional term spends a small
    part of the already accumulated tracking error on the next frame, while
    ``max_extra_step`` keeps that recovery from becoming another snap.
    """
    source_step = _relative_orientation_step_magnitude(
        target_wxyz, previous_target_wxyz
    )
    previous_tracking_error = _relative_orientation_step_magnitude(
        previous_target_wxyz, previous_robot_wxyz
    )
    catchup_error = max(
        previous_tracking_error - max(float(catchup_deadband), 0.0), 0.0
    )
    extra_step = max(float(base_extra_step), 0.0) + max(
        float(catchup_gain), 0.0
    ) * catchup_error
    if max_extra_step is not None:
        extra_step = min(extra_step, max(float(max_extra_step), 0.0))
    allowed_step = source_step + extra_step
    if max_total_step is not None:
        # Never clip a deliberate source turn merely because the generic
        # recovery envelope is lower than this particular captured frame.
        allowed_step = max(
            source_step,
            min(allowed_step, max(float(max_total_step), 0.0)),
        )
    return allowed_step


def _clamp_target_to_reach(
    target: np.ndarray,
    reach_origin: np.ndarray,
    max_reach: float | None,
) -> np.ndarray:
    """Project an unreachable swing target onto a robot reach sphere.

    Contact targets are intentionally not passed through this helper: they
    describe real scene interactions and must stay in world coordinates.  For
    free-space motion, chasing an impossible target only drives redundant arm
    joints into their limits and creates an unnatural straight or folded arm.
    """
    result = np.asarray(target, dtype=float).reshape(3).copy()
    if max_reach is None or float(max_reach) <= 0.0:
        return result
    origin = np.asarray(reach_origin, dtype=float).reshape(3)
    delta = result - origin
    distance = float(np.linalg.norm(delta))
    if distance <= float(max_reach) or distance <= 1e-10:
        return result
    return origin + delta * (float(max_reach) / distance)


def _apply_configured_joint_limits(
    model: mujoco.MjModel,
    qpos_indices: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    configured_limits: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect mechanical limits with configured task-safe joint bounds."""
    result_lower = np.asarray(lower, dtype=float).copy()
    result_upper = np.asarray(upper, dtype=float).copy()
    coordinate_by_qpos = {
        int(qpos_idx): coordinate_idx
        for coordinate_idx, qpos_idx in enumerate(qpos_indices)
    }
    for joint_name, raw_bounds in dict(configured_limits or {}).items():
        bounds = np.asarray(raw_bounds, dtype=float).reshape(-1)
        if bounds.shape != (2,):
            raise ValueError(
                f"joint_posture_limits['{joint_name}'] must contain [lower, upper]."
            )
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name)
        )
        if joint_id < 0:
            raise ValueError(
                f"joint_posture_limits refers to missing joint '{joint_name}'."
            )
        qpos_idx = int(model.jnt_qposadr[joint_id])
        coordinate_idx = coordinate_by_qpos.get(qpos_idx)
        if coordinate_idx is None:
            continue
        result_lower[coordinate_idx] = max(
            result_lower[coordinate_idx], float(bounds[0])
        )
        result_upper[coordinate_idx] = min(
            result_upper[coordinate_idx], float(bounds[1])
        )
        if result_lower[coordinate_idx] > result_upper[coordinate_idx]:
            raise ValueError(
                f"joint_posture_limits for '{joint_name}' do not overlap its "
                "mechanical range."
            )
    return result_lower, result_upper


def _apply_configured_joint_regularization(
    model: mujoco.MjModel,
    qpos_indices: np.ndarray,
    weights: np.ndarray,
    configured_weights: Mapping[str, Any] | None,
) -> np.ndarray:
    """Apply exact-name posture weights to optimized scalar joints."""
    result = np.asarray(weights, dtype=float).copy()
    coordinate_by_qpos = {
        int(qpos_idx): coordinate_idx
        for coordinate_idx, qpos_idx in enumerate(qpos_indices)
    }
    for joint_name, raw_weight in dict(configured_weights or {}).items():
        weight = float(raw_weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"joint_regularization_weights['{joint_name}'] must be a "
                "finite non-negative number."
            )
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name)
        )
        if joint_id < 0:
            raise ValueError(
                "joint_regularization_weights refers to missing joint "
                f"'{joint_name}'."
            )
        qpos_idx = int(model.jnt_qposadr[joint_id])
        coordinate_idx = coordinate_by_qpos.get(qpos_idx)
        if coordinate_idx is not None:
            result[coordinate_idx] = weight
    return result


def _apply_joint_limit_safety_fraction(
    model: mujoco.MjModel,
    qpos_indices: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    safety_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Shrink scalar mechanical ranges toward the robot's neutral pose."""
    fraction = float(safety_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("joint_limit_safety_fraction must be in (0, 1].")
    result_lower = np.asarray(lower, dtype=float).copy()
    result_upper = np.asarray(upper, dtype=float).copy()
    coordinate_by_qpos = {
        int(qpos_idx): coordinate_idx
        for coordinate_idx, qpos_idx in enumerate(qpos_indices)
    }
    for joint_id in range(model.njnt):
        if not bool(model.jnt_limited[joint_id]):
            continue
        if model.jnt_type[joint_id] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            continue
        qpos_idx = int(model.jnt_qposadr[joint_id])
        coordinate_idx = coordinate_by_qpos.get(qpos_idx)
        if coordinate_idx is None:
            continue
        neutral = float(
            np.clip(
                model.qpos0[qpos_idx],
                result_lower[coordinate_idx],
                result_upper[coordinate_idx],
            )
        )
        result_lower[coordinate_idx] = neutral + fraction * (
            result_lower[coordinate_idx] - neutral
        )
        result_upper[coordinate_idx] = neutral + fraction * (
            result_upper[coordinate_idx] - neutral
        )
    return result_lower, result_upper
