"""Joint-coordinate, semantic-frame and temporal refinement helpers."""

from __future__ import annotations

from math import comb
from typing import Any, Mapping

import mujoco
import numpy as np
from scipy import sparse

from ..contact_schedule import _iter_true_runs


def _scalar_joint_coordinates(
    model: mujoco.MjModel,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return qpos/dof coordinates and their limits for scalar robot joints."""
    qpos_indices: list[int] = []
    dof_indices: list[int] = []
    lower: list[float] = []
    upper: list[float] = []
    for joint_id in range(model.njnt):
        joint_type = model.jnt_type[joint_id]
        if joint_type not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            continue
        qpos_index = int(model.jnt_qposadr[joint_id])
        qpos_indices.append(qpos_index)
        dof_indices.append(int(model.jnt_dofadr[joint_id]))
        if model.jnt_limited[joint_id]:
            joint_lower, joint_upper = model.jnt_range[joint_id]
        else:
            joint_lower, joint_upper = -np.inf, np.inf
        lower.append(float(joint_lower))
        upper.append(float(joint_upper))
    return (
        np.asarray(qpos_indices, dtype=int),
        np.asarray(dof_indices, dtype=int),
        np.asarray(lower, dtype=float),
        np.asarray(upper, dtype=float),
    )


def _joint_temporal_weight_vector(
    model: mujoco.MjModel,
    qpos_indices: np.ndarray,
    rules: Mapping[str, float] | None,
) -> np.ndarray:
    """Resolve substring-based per-joint smoothing weights.

    The whole-trajectory solver must keep hand position residuals in the same
    system as smoothing.  Weighting redundant shoulder/elbow/wrist coordinates
    here selects a smoother arm pose without applying a post-hoc filter that
    would move the end effector away from its scene interaction point.
    """
    weights = np.ones(len(qpos_indices), dtype=float)
    normalized_rules = {
        str(token).lower(): max(float(weight), 0.0)
        for token, weight in dict(rules or {}).items()
    }
    if not normalized_rules:
        return weights

    names_by_qpos: dict[int, str] = {}
    for joint_id in range(model.njnt):
        joint_name = model.joint(joint_id).name
        if joint_name:
            names_by_qpos[int(model.jnt_qposadr[joint_id])] = joint_name.lower()
    for coordinate_idx, qpos_idx in enumerate(qpos_indices):
        joint_name = names_by_qpos.get(int(qpos_idx), "")
        for token, weight in normalized_rules.items():
            if token in joint_name:
                weights[coordinate_idx] = weight
    return weights


def _joint_posture_target_vectors(
    model: mujoco.MjModel,
    qpos_indices: np.ndarray,
    rules: Mapping[str, Mapping[str, float]] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve safe joint targets and sigmas from joint-name substrings.

    The source skeleton supplies wrist positions but no reliable articulated
    hand orientation.  Without a posture term those redundant wrist joints can
    take large angles solely to shave millimetres from endpoint error.  A soft
    neutral target selects the least-folded IK solution while still allowing a
    wrist to move when the endpoint and arm-shape objectives require it.
    """
    targets = np.zeros(len(qpos_indices), dtype=float)
    sigmas = np.full(len(qpos_indices), np.inf, dtype=float)
    normalized_rules = {
        str(token).lower(): dict(spec or {})
        for token, spec in dict(rules or {}).items()
    }
    if not normalized_rules:
        return targets, sigmas

    names_by_qpos: dict[int, str] = {}
    for joint_id in range(model.njnt):
        joint_name = model.joint(joint_id).name
        if joint_name:
            names_by_qpos[int(model.jnt_qposadr[joint_id])] = joint_name.lower()
    for coordinate_idx, qpos_idx in enumerate(qpos_indices):
        joint_name = names_by_qpos.get(int(qpos_idx), "")
        for token, spec in normalized_rules.items():
            if token not in joint_name:
                continue
            targets[coordinate_idx] = float(spec.get("target", 0.0))
            sigmas[coordinate_idx] = max(float(spec.get("sigma", 0.35)), 1e-6)
    return targets, sigmas


def _joint_coordinate_mask(
    model: mujoco.MjModel,
    qpos_indices: np.ndarray,
    joint_name_tokens: list[str] | tuple[str, ...] | None,
) -> np.ndarray:
    """Select scalar coordinates whose joint names contain a configured token.

    Hand orientation is deliberately assigned to the wrist coordinates only.
    Position and arm-shape tasks may still use the complete kinematic chain,
    but an orientation residual must not rotate the shoulder or fold the elbow
    merely to make the palm frame agree with SMPL-X.
    """
    tokens = [
        str(token).strip().lower()
        for token in (joint_name_tokens or [])
        if str(token).strip()
    ]
    mask = np.zeros(len(qpos_indices), dtype=bool)
    if not tokens:
        return mask

    names_by_qpos: dict[int, str] = {}
    for joint_id in range(model.njnt):
        joint_name = model.joint(joint_id).name
        if joint_name:
            names_by_qpos[int(model.jnt_qposadr[joint_id])] = joint_name.lower()
    for coordinate_idx, qpos_idx in enumerate(qpos_indices):
        joint_name = names_by_qpos.get(int(qpos_idx), "")
        mask[coordinate_idx] = any(token in joint_name for token in tokens)
    return mask


def _inset_finite_joint_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    margin: float,
    margin_ratio: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Move finite solver bounds inside the configured/mechanical limits.

    A bounded least-squares solution is allowed to sit exactly on its box
    boundary.  Using a small interior hard margin means even a saturated
    optimization variable cannot equal the robot's actual limit.  The inset
    can be absolute or proportional to each finite joint range and is capped
    at one quarter of that range so narrow joints always retain a non-empty
    feasible interval.
    """
    result_lower = np.asarray(lower, dtype=float).copy()
    result_upper = np.asarray(upper, dtype=float).copy()
    requested = max(float(margin), 0.0)
    requested_ratio = max(float(margin_ratio), 0.0)
    finite_both = np.isfinite(result_lower) & np.isfinite(result_upper)
    widths = result_upper - result_lower
    inset = np.full_like(result_lower, requested)
    if requested_ratio > 0.0:
        inset[finite_both] = widths[finite_both] * requested_ratio
    inset[finite_both] = np.minimum(
        inset[finite_both],
        np.maximum(widths[finite_both], 0.0) * 0.25,
    )
    result_lower[np.isfinite(result_lower)] += inset[
        np.isfinite(result_lower)
    ]
    result_upper[np.isfinite(result_upper)] -= inset[
        np.isfinite(result_upper)
    ]
    return result_lower, result_upper


def _minimal_vector_alignment(
    source_direction: np.ndarray,
    target_direction: np.ndarray,
) -> np.ndarray:
    """Return the minimum-angle rotation aligning two unit directions."""
    source = np.asarray(source_direction, dtype=float).reshape(3)
    target = np.asarray(target_direction, dtype=float).reshape(3)
    source /= max(float(np.linalg.norm(source)), 1e-10)
    target /= max(float(np.linalg.norm(target)), 1e-10)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine > 1e-8:
        skew = np.array(
            [
                [0.0, -cross[2], cross[1]],
                [cross[2], 0.0, -cross[0]],
                [-cross[1], cross[0], 0.0],
            ]
        )
        return np.eye(3) + skew + skew @ skew * ((1.0 - cosine) / sine**2)
    if cosine > 0.0:
        return np.eye(3)
    basis = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(source, basis))) > 0.8:
        basis = np.array([0.0, 1.0, 0.0])
    axis = np.cross(source, basis)
    axis /= max(float(np.linalg.norm(axis)), 1e-10)
    return 2.0 * np.outer(axis, axis) - np.eye(3)


def _transported_arm_segment_targets(
    source_shoulder: np.ndarray,
    source_elbow: np.ndarray,
    source_hand: np.ndarray,
    robot_shoulder: np.ndarray,
    robot_hand_target: np.ndarray,
    upper_arm_length: float,
    forearm_length: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Transport the captured two-link arm shape toward a robot hand target.

    Endpoint-only IK is underdetermined: it can attain the same hand position
    with a visually unrelated shoulder/elbow branch.  This construction first
    rebuilds both captured segment directions at the robot's link lengths,
    then applies one minimum-angle rotation to point their chord toward the
    requested hand position.  A single rotation preserves the captured elbow
    angle and bend plane exactly.  The endpoint objective remains separate, so
    an unreachable target produces a deliberate position compromise instead
    of changing the arm into a folded configuration.
    """
    source_shoulder = np.asarray(source_shoulder, dtype=float).reshape(3)
    source_elbow = np.asarray(source_elbow, dtype=float).reshape(3)
    source_hand = np.asarray(source_hand, dtype=float).reshape(3)
    robot_shoulder = np.asarray(robot_shoulder, dtype=float).reshape(3)
    robot_hand_target = np.asarray(robot_hand_target, dtype=float).reshape(3)
    upper = max(float(upper_arm_length), 1e-6)
    forearm = max(float(forearm_length), 1e-6)

    source_upper = source_elbow - source_shoulder
    source_forearm = source_hand - source_elbow
    upper_norm = float(np.linalg.norm(source_upper))
    forearm_norm = float(np.linalg.norm(source_forearm))
    if upper_norm < 1e-8 or forearm_norm < 1e-8:
        raise ValueError("source arm segments must have non-zero length")

    scaled_upper = upper * source_upper / upper_norm
    scaled_forearm = forearm * source_forearm / forearm_norm
    source_chord = scaled_upper + scaled_forearm
    chord_norm = float(np.linalg.norm(source_chord))
    robot_axis = robot_hand_target - robot_shoulder
    axis_norm = float(np.linalg.norm(robot_axis))
    if chord_norm < 1e-8 or axis_norm < 1e-8:
        return scaled_upper, scaled_forearm

    transport = _minimal_vector_alignment(source_chord, robot_axis)
    return transport @ scaled_upper, transport @ scaled_forearm


def _orthonormal_target_rotation(
    forward_direction: np.ndarray | None,
    up_direction: np.ndarray | None,
) -> np.ndarray | None:
    if forward_direction is None or up_direction is None:
        return None
    forward = np.asarray(forward_direction, dtype=float).reshape(3).copy()
    up = np.asarray(up_direction, dtype=float).reshape(3).copy()
    if not np.all(np.isfinite(forward)) or not np.all(np.isfinite(up)):
        return None
    up_norm = float(np.linalg.norm(up))
    if up_norm < 1e-10:
        return None
    up /= up_norm
    forward -= up * float(np.dot(up, forward))
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-10:
        return None
    forward /= forward_norm
    lateral = np.cross(up, forward)
    lateral_norm = float(np.linalg.norm(lateral))
    if lateral_norm < 1e-10:
        return None
    lateral /= lateral_norm
    forward = np.cross(lateral, up)
    return np.column_stack([forward, lateral, up])


def _local_semantic_rotation(
    forward_direction: np.ndarray | None,
    up_direction: np.ndarray | None,
) -> np.ndarray:
    """Return the body's local forward/lateral/up semantic hand frame.

    MuJoCo body axes are model-construction details and do not necessarily
    mean palm-forward, palm-lateral and palm-normal.  For the G1 rubber hand,
    local +X runs from wrist to fingers while local -Y is the palm normal.
    Mapping through this fixed frame lets the optimizer compare anatomical
    hand directions instead of accidentally aligning unrelated mesh axes.
    """
    rotation = _orthonormal_target_rotation(forward_direction, up_direction)
    if rotation is None:
        raise ValueError(
            "orientation_local_forward and orientation_local_up must be "
            "finite, non-zero, and non-collinear."
        )
    return rotation


def _skew_symmetric(vector: np.ndarray) -> np.ndarray:
    """Return the cross-product matrix for a three-vector."""
    x, y, z = np.asarray(vector, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _difference_operator(num_frames: int, order: int) -> sparse.csr_matrix:
    """Return the exact ``order``-th finite-difference matrix over time."""
    num_rows = num_frames - order
    if num_rows <= 0:
        return sparse.csr_matrix((0, num_frames))
    diagonals = [
        np.full(num_rows, ((-1.0) ** (order - j)) * comb(order, j))
        for j in range(order + 1)
    ]
    return sparse.diags(
        diagonals,
        list(range(order + 1)),
        shape=(num_rows, num_frames),
        format="csr",
    )


def _penetration_depth_weight(
    clearance: float,
    config: Mapping[str, Any],
) -> float:
    """Increase the influence of deep collision samples without a hard cutoff.

    The ordinary quadratic penalty can reduce many shallow intersections while
    allowing one collision point to become much deeper.  This bounded IRLS
    weight makes samples beyond ``depth_priority_start`` dominate that tradeoff
    while keeping the problem a soft least-squares optimization.
    """
    depth = max(-float(clearance), 0.0)
    priority_start = max(float(config.get("depth_priority_start", 0.05)), 1e-8)
    max_weight = max(float(config.get("max_depth_weight", 9.0)), 1.0)
    return min(max_weight, max(1.0, (depth / priority_start) ** 2))


def _lock_contact_offsets(
    model: mujoco.MjModel,
    trajectory: np.ndarray,
    effectors: Mapping[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Choose one contact offset per contiguous contact run.

    Re-selecting the nearest hand or sole keypoint at every frame makes the
    contact residual discontinuous in time, which fights the jerk term living
    in the same objective.  Committing to the offset that is best on average
    over a no-slip run keeps that residual smooth for the whole run, and keeps
    it fixed across Gauss--Newton iterations so the active set cannot chatter.
    """
    data = mujoco.MjData(model)
    num_frames = len(trajectory)
    distances = {
        name: np.zeros((num_frames, len(spec["contact_offsets"])), dtype=float)
        for name, spec in effectors.items()
    }
    for frame_idx in range(num_frames):
        data.qpos[:] = trajectory[frame_idx]
        mujoco.mj_forward(model, data)
        for name, spec in effectors.items():
            body_id = int(spec["body_id"])
            rotation = data.xmat[body_id].reshape(3, 3)
            candidates = (
                data.xpos[body_id][None, :] + spec["contact_offsets"] @ rotation.T
            )
            target = spec["task"].contact_positions[frame_idx]
            distances[name][frame_idx] = np.linalg.norm(
                candidates - target[None, :], axis=1
            )

    selection: dict[str, np.ndarray] = {}
    for name, spec in effectors.items():
        chosen = np.zeros(num_frames, dtype=int)
        mask = np.asarray(spec["task"].contact_mask, dtype=bool)
        for start, end in _iter_true_runs(mask):
            chosen[start:end] = int(
                np.argmin(np.mean(distances[name][start:end], axis=0))
            )
        # The free-to-contact handover must approach the SAME material point.
        # Carry its identity outside contact too; window callers slice this
        # full-timeline plan together with the task's approach reference.
        from ..contact_schedule import approach_blend

        task = spec["task"]
        _, _, contact_frames = approach_blend(
            task.tracking_positions, task.contact_positions, mask,
            return_contact_frames=True,
        )
        chosen[~mask] = chosen[contact_frames[~mask]]
        selection[name] = chosen
    return selection
