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
"""

from __future__ import annotations

from math import comb
from typing import Any, Mapping

import cvxpy as cp
import mujoco
import numpy as np
import trimesh
from scipy import sparse
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from .contact import ContactMode, SceneContactTask, _iter_true_runs
from .retargeting import (
    _apply_configured_joint_limits,
    _resolve_position_sigma,
    _swivel_direction_and_jacobian,
    _swivel_tangent_delta,
    _transform_swing_target,
    _validate_swing_target_space,
)


# This is an aggregate, dimensionless residual-norm budget after every row has
# been scaled by its physical sigma and confidence.  Keep it small: SLIDE
# tangential flexibility belongs in ``slide_tangential_weight`` (and the
# mode-specific residual/sigma), not in a large lexicographic escape hatch.
DEFAULT_CONTACT_PRIORITY_ABSOLUTE_TOLERANCE = 0.03


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
        if qpos_index < 7:
            continue
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


_AXIS_DIRECTIONS = np.array(
    [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ]
)


def geom_surface_offsets(model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    """Return local-frame surface samples for one collision geom.

    MuJoCo capsules and cylinders are aligned with the geom's local **z** axis,
    so the barrel rings and end caps are placed along z.  The offsets only
    depend on the geom's size, so they are computed once and reused for every
    frame instead of being rebuilt per query.
    """
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=float)

    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return _AXIS_DIRECTIONS * float(size[0])
    if geom_type in (
        mujoco.mjtGeom.mjGEOM_BOX,
        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    ):
        return _AXIS_DIRECTIONS * size[:3]
    if geom_type in (
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
    ):
        radius = float(size[0])
        half_length = float(size[1])
        angles = np.arange(4) * (np.pi / 2.0)
        ring = np.column_stack(
            [radius * np.cos(angles), radius * np.sin(angles), np.zeros(4)]
        )
        cap = (
            half_length + radius
            if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE
            else half_length
        )
        return np.vstack(
            [
                ring + [0.0, 0.0, -half_length],
                ring,
                ring + [0.0, 0.0, half_length],
                [[0.0, 0.0, -cap], [0.0, 0.0, cap]],
            ]
        )
    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id >= 0:
            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            vertices = np.asarray(
                model.mesh_vert[start : start + count], dtype=float
            )
            if len(vertices) > 64:
                # A dense collision mesh does not need every vertex to detect
                # scene penetration.  A deterministic stride bounds the query
                # count without biasing the samples toward one region.
                vertices = vertices[:: int(np.ceil(len(vertices) / 64.0))]
            return vertices
    if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
        return np.zeros((0, 3), dtype=float)
    return np.zeros((1, 3), dtype=float)


def geom_surface_points_in_body_frame(
    model: mujoco.MjModel, geom_id: int
) -> np.ndarray:
    """Transform the shared geom-local surface samples into its body frame.

    ``geom_surface_offsets`` remains the single primitive/mesh sampler.  This
    helper only applies the compiled geom pose, allowing contact-patch and
    legacy foot-support code to consume the same capsule/primitive samples as
    scene-penetration refinement.
    """
    offsets = geom_surface_offsets(model, geom_id)
    geom_quat = np.asarray(model.geom_quat[geom_id], dtype=float)
    geom_rotation = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(geom_rotation, geom_quat)
    geom_position = np.asarray(model.geom_pos[geom_id], dtype=float)
    return offsets @ geom_rotation.reshape(3, 3).T + geom_position


def collision_geom_ids(
    model: mujoco.MjModel,
    geom_groups: Any = None,
    body_tokens: Any = None,
) -> np.ndarray:
    """Return the geoms that should be tested against the scene mesh.

    The default is every geom MuJoCo itself would collide, which for a humanoid
    URDF is the primitive collision shell rather than the visual meshes.
    ``geom_groups`` and ``body_tokens`` narrow the set further.
    """
    groups = None if geom_groups is None else {int(g) for g in geom_groups}
    tokens = (
        None
        if body_tokens is None
        else tuple(str(token).lower() for token in body_tokens)
    )
    selected: list[int] = []
    for geom_id in range(model.ngeom):
        if groups is None:
            if (
                int(model.geom_contype[geom_id]) == 0
                and int(model.geom_conaffinity[geom_id]) == 0
            ):
                continue
        elif int(model.geom_group[geom_id]) not in groups:
            continue
        if tokens is not None:
            body_id = int(model.geom_bodyid[geom_id])
            body_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            )
            if not any(token in body_name.lower() for token in tokens):
                continue
        selected.append(geom_id)
    return np.asarray(selected, dtype=int)


def closest_scene_triangles(
    scene_mesh: trimesh.Trimesh, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closest surface point, distance and triangle id for each query point."""
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return (
            np.zeros((0, 3), dtype=float),
            np.zeros(0, dtype=float),
            np.zeros(0, dtype=int),
        )
    try:
        closest, distances, triangle_ids = trimesh.proximity.closest_point(
            scene_mesh, points
        )
    except (ImportError, ModuleNotFoundError):
        closest, distances, triangle_ids = trimesh.proximity.closest_point_naive(
            scene_mesh, points
        )
    return (
        np.asarray(closest, dtype=float),
        np.asarray(distances, dtype=float),
        np.asarray(triangle_ids, dtype=int),
    )


def resolve_scene_normal_orientation(
    scene_mesh: trimesh.Trimesh, reference_points: np.ndarray
) -> float:
    """Fix the global scene normal orientation from known free-side points.

    Reconstructed scans have consistent winding but no guaranteed global
    outward direction, and they are not watertight, so ``signed_distance``
    cannot supply the side test.  Any point set that mostly lies in free space
    resolves the sign: the median offset direction from the surface must agree
    with the outward normal.
    """
    closest, distances, triangle_ids = closest_scene_triangles(
        scene_mesh, reference_points
    )
    face_normals = np.asarray(scene_mesh.face_normals, dtype=float)
    valid = (
        (distances > 1e-6)
        & (triangle_ids >= 0)
        & (triangle_ids < len(face_normals))
    )
    if not np.any(valid):
        return 1.0
    offsets = np.asarray(reference_points, dtype=float)[valid] - closest[valid]
    offsets = offsets / np.linalg.norm(offsets, axis=1, keepdims=True)
    median_cosine = float(
        np.median(np.sum(offsets * face_normals[triangle_ids[valid]], axis=1))
    )
    return -1.0 if median_cosine < 0.0 else 1.0


def scene_signed_clearance(
    scene_mesh: trimesh.Trimesh,
    points: np.ndarray,
    orientation_sign: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return signed clearance and the direction that increases it.

    Clearance is positive in free space and negative inside the scene.  The
    direction is the gradient of the signed clearance with respect to the
    sample point, so a Gauss--Newton step along it moves the point away from
    the scene no matter which side of the surface it currently is on.
    """
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.zeros(0, dtype=float), np.zeros((0, 3), dtype=float)
    closest, distances, triangle_ids = closest_scene_triangles(scene_mesh, points)

    offsets = points - closest
    norms = np.linalg.norm(offsets, axis=1)

    face_normals = np.asarray(scene_mesh.face_normals, dtype=float)
    normals = np.zeros_like(offsets)
    valid = (triangle_ids >= 0) & (triangle_ids < len(face_normals))
    normals[valid] = face_normals[triangle_ids[valid]]
    normals = normals * (orientation_sign or 1.0)

    side = np.where(np.sum(offsets * normals, axis=1) >= 0.0, 1.0, -1.0)
    signed = side * distances

    # Away from the surface the unsigned-distance gradient is the accurate
    # push direction; exactly on the surface that gradient is undefined and
    # the triangle normal is the only usable direction.
    directions = np.where(
        norms[:, None] > 1e-9,
        offsets / np.maximum(norms, 1e-9)[:, None] * side[:, None],
        normals,
    )
    direction_norms = np.linalg.norm(directions, axis=1, keepdims=True)
    return signed, np.where(
        direction_norms > 1e-9,
        directions / np.maximum(direction_norms, 1e-9),
        np.array([0.0, 0.0, 1.0]),
    )


class ScenePenetrationProbe:
    """Signed scene clearance for the robot's collision geoms.

    Reconstructed scene meshes are rarely watertight, so ``signed_distance``
    cannot be trusted.  TSDF scans do have consistent winding, which makes the
    closest triangle's normal a usable side test once its global orientation is
    fixed.  That global sign is resolved from the trajectory itself: almost
    every robot geom is far from the scene and unambiguously on the free side,
    so the median offset direction reveals which way the normals point.

    Queries are batched over the whole trajectory.  A cheap bounding-sphere
    pass on geom centres removes the geoms that cannot reach the scene before
    the expensive surface query runs, and ``geom_rbound`` bounds every sample
    point, so that prefilter can never discard a penetrating geom.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        scene_mesh: trimesh.Trimesh,
        geom_ids: np.ndarray,
        margin: float,
    ) -> None:
        self.model = model
        self.scene_mesh = scene_mesh
        self.geom_ids = np.asarray(geom_ids, dtype=int)
        self.margin = float(margin)
        self.body_ids = np.asarray(
            [int(model.geom_bodyid[gid]) for gid in self.geom_ids], dtype=int
        )
        self.offsets = [geom_surface_offsets(model, gid) for gid in self.geom_ids]
        self.bounding_radii = np.asarray(
            [float(model.geom_rbound[gid]) for gid in self.geom_ids], dtype=float
        )
        self.orientation_sign = 0.0

    def _closest(self, points: np.ndarray):
        return closest_scene_triangles(self.scene_mesh, points)

    def _resolve_orientation(self, centers: np.ndarray) -> None:
        """Fix the global normal orientation from far-field geom centres."""
        if self.orientation_sign != 0.0:
            return
        self.orientation_sign = resolve_scene_normal_orientation(
            self.scene_mesh, centers
        )

    def _clearance(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return scene_signed_clearance(
            self.scene_mesh, points, self.orientation_sign
        )

    def probe_trajectory(
        self, trajectory: np.ndarray
    ) -> list[dict[str, np.ndarray]]:
        """Return per-frame sample points near the scene with their clearance."""
        trajectory = np.asarray(trajectory, dtype=float)
        num_frames = len(trajectory)
        empty = {
            "points": np.zeros((0, 3), dtype=float),
            "body_ids": np.zeros(0, dtype=int),
            "directions": np.zeros((0, 3), dtype=float),
            "clearance": np.zeros(0, dtype=float),
        }
        if num_frames == 0 or len(self.geom_ids) == 0:
            return [dict(empty) for _ in range(num_frames)]

        data = mujoco.MjData(self.model)
        centers = np.zeros((num_frames, len(self.geom_ids), 3), dtype=float)
        for frame_idx, qpos in enumerate(trajectory):
            data.qpos[:] = qpos
            mujoco.mj_forward(self.model, data)
            centers[frame_idx] = data.geom_xpos[self.geom_ids]

        flat_centers = centers.reshape(-1, 3)
        self._resolve_orientation(flat_centers)
        _, center_distances, _ = self._closest(flat_centers)
        near = center_distances.reshape(num_frames, -1) <= (
            self.bounding_radii + self.margin
        )[None, :]

        frame_points: list[np.ndarray] = []
        frame_owners: list[np.ndarray] = []
        counts = np.zeros(num_frames, dtype=int)
        for frame_idx, qpos in enumerate(trajectory):
            data.qpos[:] = qpos
            mujoco.mj_forward(self.model, data)
            points: list[np.ndarray] = []
            owners: list[np.ndarray] = []
            for local_index in np.flatnonzero(near[frame_idx]):
                offsets = self.offsets[local_index]
                if len(offsets) == 0:
                    continue
                geom_id = int(self.geom_ids[local_index])
                rotation = data.geom_xmat[geom_id].reshape(3, 3)
                points.append(
                    data.geom_xpos[geom_id][None, :] + offsets @ rotation.T
                )
                owners.append(np.full(len(offsets), local_index, dtype=int))
            if points:
                frame_points.append(np.concatenate(points, axis=0))
                frame_owners.append(np.concatenate(owners))
            else:
                frame_points.append(np.zeros((0, 3), dtype=float))
                frame_owners.append(np.zeros(0, dtype=int))
            counts[frame_idx] = len(frame_points[-1])

        all_points = np.concatenate(frame_points, axis=0)
        all_owners = np.concatenate(frame_owners)
        signed, directions = self._clearance(all_points)

        boundaries = np.cumsum(counts)[:-1]
        return [
            {
                "points": chunk_points,
                "body_ids": self.body_ids[chunk_owners],
                "directions": chunk_directions,
                "clearance": chunk_signed,
            }
            for chunk_points, chunk_owners, chunk_directions, chunk_signed in zip(
                np.split(all_points, boundaries),
                np.split(all_owners, boundaries),
                np.split(directions, boundaries),
                np.split(signed, boundaries),
            )
        ]


def measure_scene_penetration(
    model: mujoco.MjModel,
    trajectory: np.ndarray,
    scene_mesh: trimesh.Trimesh,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Report how deep the robot's collision geoms enter the scene mesh.

    Depths are positive when a sample point is inside the scene.  This is the
    independent check for the refinement stage: acceleration and jerk are part
    of the objective and are expected to improve, whereas penetration is only
    ever penalized and therefore has to be verified after the fact.
    """
    cfg = dict(config or {})
    trajectory = np.asarray(trajectory, dtype=float)
    geom_ids = collision_geom_ids(
        model, cfg.get("geom_groups"), cfg.get("body_tokens")
    )
    summary: dict[str, Any] = {
        "max_depth": 0.0,
        "mean_depth": 0.0,
        "frames_penetrating": 0,
        "total_penetrating_samples": 0,
        "penetrating_body_count": 0,
        "num_frames": int(len(trajectory)),
        "worst_frame": -1,
        "worst_body": "",
        "per_frame_depth": np.zeros(len(trajectory), dtype=float),
        "per_frame_penetrating_samples": np.zeros(
            len(trajectory), dtype=np.int32
        ),
    }
    if len(geom_ids) == 0 or len(trajectory) == 0:
        return summary

    probe = ScenePenetrationProbe(
        model, scene_mesh, geom_ids, float(cfg.get("margin", 0.05))
    )
    per_frame_depth = np.zeros(len(trajectory), dtype=float)
    per_frame_penetrating_samples = np.zeros(len(trajectory), dtype=np.int32)
    penetrating_body_ids: set[int] = set()
    worst_body_id = -1
    worst_depth = 0.0
    for frame_idx, frame in enumerate(probe.probe_trajectory(trajectory)):
        if len(frame["clearance"]) == 0:
            continue
        depths = np.maximum(-frame["clearance"], 0.0)
        penetrating_samples = depths > 1e-6
        per_frame_penetrating_samples[frame_idx] = int(
            np.count_nonzero(penetrating_samples)
        )
        penetrating_body_ids.update(
            int(body_id)
            for body_id in frame["body_ids"][penetrating_samples]
        )
        worst = int(np.argmax(depths))
        per_frame_depth[frame_idx] = float(depths[worst])
        if per_frame_depth[frame_idx] > worst_depth:
            worst_depth = per_frame_depth[frame_idx]
            worst_body_id = int(frame["body_ids"][worst])

    penetrating = per_frame_depth > 1e-6
    summary.update(
        max_depth=float(np.max(per_frame_depth)),
        mean_depth=float(np.mean(per_frame_depth[penetrating]))
        if np.any(penetrating)
        else 0.0,
        frames_penetrating=int(np.count_nonzero(penetrating)),
        total_penetrating_samples=int(
            np.sum(per_frame_penetrating_samples)
        ),
        penetrating_body_count=int(len(penetrating_body_ids)),
        worst_frame=int(np.argmax(per_frame_depth)),
        worst_body=(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, worst_body_id) or ""
        )
        if worst_body_id >= 0
        else "",
        per_frame_depth=per_frame_depth,
        per_frame_penetrating_samples=per_frame_penetrating_samples,
    )
    return summary


def _lock_contact_offsets(
    model: mujoco.MjModel,
    trajectory: np.ndarray,
    effectors: Mapping[str, dict[str, Any]],
    preset: Mapping[str, np.ndarray] | None = None,
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
        segment_ids = np.asarray(
            spec["task"].contact_segment_ids, dtype=np.int32
        )
        preset_ids = None if preset is None else preset.get(name)
        if preset_ids is not None:
            preset_ids = np.asarray(preset_ids, dtype=np.int32)
        for segment_id in np.unique(segment_ids[segment_ids >= 0]):
            frames = np.flatnonzero(segment_ids == segment_id)
            valid_preset = (
                np.zeros(0, dtype=np.int32)
                if preset_ids is None
                else preset_ids[frames][preset_ids[frames] >= 0]
            )
            if len(valid_preset):
                counts = np.bincount(valid_preset)
                offset_id = int(np.argmax(counts))
            else:
                offset_id = int(
                    np.argmin(np.mean(distances[name][frames], axis=0))
                )
            chosen[frames] = offset_id
        selection[name] = chosen
    return selection


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
    active_contact_offset_ids: Mapping[str, np.ndarray] | None = None,
    joint_mapping: Mapping[str, str] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> np.ndarray:
    """Jointly refine scalar joints over time with sparse Gauss--Newton.

    The floating base is deliberately held at the framewise solver result.  It
    is already responsible for world registration and morphology compromise;
    allowing a second optimizer to move it would reintroduce the base drift
    that whole-body smoothing is meant to avoid.
    """
    result = np.asarray(trajectory, dtype=float).copy()
    if len(result) < 4 or not tasks:
        return result

    qpos_indices, dof_indices, joint_lower, joint_upper = (
        _scalar_joint_coordinates(model)
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

    refinement_pose_edges: list[dict[str, int]] = []
    pose_direction_cfg = dict(
        end_effector_config.get("pose_direction", {}) or {}
    )
    if source_joint_positions is not None and joint_mapping is not None:
        for raw_edge in pose_direction_cfg.get("edges", []):
            if len(raw_edge) != 2:
                continue
            parent_name, child_name = map(str, raw_edge)
            # Arm posture transport below handles the morphology-sensitive arm
            # chains.  Generic unit-direction rows cover torso and legs.
            if any(
                token in parent_name or token in child_name
                for token in ("Shoulder", "Elbow", "Wrist")
            ):
                continue
            if (
                parent_name not in source_name_to_index
                or child_name not in source_name_to_index
                or parent_name not in joint_mapping
                or child_name not in joint_mapping
            ):
                continue
            parent_body_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                str(joint_mapping[parent_name]),
            )
            child_body_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                str(joint_mapping[child_name]),
            )
            if parent_body_id >= 0 and child_body_id >= 0:
                refinement_pose_edges.append(
                    {
                        "parent_body_id": int(parent_body_id),
                        "child_body_id": int(child_body_id),
                        "source_parent_idx": int(
                            source_name_to_index[parent_name]
                        ),
                        "source_child_idx": int(
                            source_name_to_index[child_name]
                        ),
                    }
                )

    raw_specs = dict(end_effector_config.get("effectors", {}) or {})
    effectors: dict[str, dict[str, Any]] = {}
    for name, task in tasks.items():
        if name not in raw_specs:
            continue
        spec = dict(raw_specs[name] or {})
        body_name = spec.get("robot_body")
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, str(body_name)
        )
        if body_id < 0:
            continue
        tracking_offset = np.asarray(
            spec.get("tracking_offset", [0.0, 0.0, 0.0]), dtype=float
        ).reshape(3)
        contact_offsets = np.asarray(
            spec.get("contact_offsets", [tracking_offset]), dtype=float
        )
        if contact_offsets.ndim == 1:
            contact_offsets = contact_offsets[None, :]
        contact_target_space = str(spec["contact_target_space"]).lower()
        if contact_target_space != "world":
            raise ValueError(
                f"End-effector '{name}' contact_target_space must be 'world'."
            )
        swing_target_space = _validate_swing_target_space(
            name, spec["swing_target_space"]
        )
        swing_anchor_body_id = -1
        if swing_target_space == "limb_morphology_scaled":
            if task.swing_anchor_body is None or task.swing_limb_offsets is None:
                raise ValueError(
                    f"End-effector '{name}' is limb-relative but its live "
                    "swing anchor data was not prepared."
                )
            swing_anchor_body_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                str(task.swing_anchor_body),
            )
            if swing_anchor_body_id < 0:
                raise ValueError(
                    f"End-effector '{name}' limb-relative swing task refers "
                    f"to missing anchor body '{task.swing_anchor_body}'."
                )
        raw_arm_posture = dict(spec.get("arm_posture", {}) or {})
        arm_posture = None
        if bool(raw_arm_posture.get("enabled", False)):
            raw_swivel = dict(raw_arm_posture.get("swivel", {}) or {})
            shoulder_name = str(raw_arm_posture["shoulder_body"])
            elbow_name = str(raw_arm_posture["elbow_body"])
            wrist_name = raw_arm_posture.get("wrist_body")
            shoulder_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, shoulder_name
            )
            elbow_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, elbow_name
            )
            wrist_body_id = (
                -1
                if wrist_name is None
                else mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, str(wrist_name)
                )
            )
            if shoulder_body_id < 0 or elbow_body_id < 0:
                raise ValueError(
                    f"End-effector '{name}' arm_posture refers to a missing "
                    "shoulder or elbow body."
                )
            if wrist_name is not None and wrist_body_id < 0:
                raise ValueError(
                    f"End-effector '{name}' arm_posture refers to missing "
                    f"wrist body '{wrist_name}'."
                )
            source_shoulder_name = str(raw_arm_posture["source_shoulder"])
            source_elbow_name = str(raw_arm_posture["source_elbow"])
            source_wrist_name = raw_arm_posture.get("source_wrist")
            if source_joint_positions is not None and (
                source_shoulder_name not in source_name_to_index
                or source_elbow_name not in source_name_to_index
            ):
                raise ValueError(
                    f"End-effector '{name}' arm_posture source joints are missing."
                )
            if (
                source_joint_positions is not None
                and source_wrist_name is not None
                and str(source_wrist_name) not in source_name_to_index
            ):
                raise ValueError(
                    f"End-effector '{name}' arm_posture source wrist is missing."
                )
            if (wrist_name is None) != (source_wrist_name is None):
                raise ValueError(
                    f"End-effector '{name}' arm_posture must configure both "
                    "wrist_body and source_wrist together."
                )
            arm_posture = {
                "shoulder_body_id": int(shoulder_body_id),
                "elbow_body_id": int(elbow_body_id),
                "wrist_body_id": int(wrist_body_id),
                "source_shoulder_idx": source_name_to_index.get(
                    source_shoulder_name
                ),
                "source_elbow_idx": source_name_to_index.get(source_elbow_name),
                "source_wrist_idx": (
                    None
                    if source_wrist_name is None
                    else source_name_to_index.get(str(source_wrist_name))
                ),
                "upper_arm_length": float(raw_arm_posture["upper_arm_length"]),
                "forearm_length": float(raw_arm_posture["forearm_length"]),
                "upper_direction_sigma": max(
                    float(
                        raw_arm_posture.get(
                            "upper_direction_sigma",
                            raw_arm_posture.get("sigma", 0.08),
                        )
                    ),
                    1e-6,
                ),
                "forearm_direction_sigma": max(
                    float(
                        raw_arm_posture.get(
                            "forearm_direction_sigma",
                            raw_arm_posture.get("sigma", 0.08),
                        )
                    ),
                    1e-6,
                ),
                "swivel_enabled": bool(raw_swivel.get("enabled", False)),
                "swivel_sigma": max(
                    float(raw_swivel.get("sigma", np.deg2rad(20.0))), 1e-6
                ),
                "swivel_weight": max(
                    float(raw_swivel.get("weight", 1.0)), 0.0
                ),
                "swivel_min_sine": max(
                    float(raw_swivel.get("min_sine", 0.05)), 0.0
                ),
                "swivel_full_sine": max(
                    float(raw_swivel.get("full_sine", 0.35)),
                    max(float(raw_swivel.get("min_sine", 0.05)), 0.0)
                    + 1e-8,
                ),
            }
        refine_orientation = bool(
            spec.get("refine_orientation", spec.get("track_orientation", False))
        )
        orientation_local_rotation = _local_semantic_rotation(
            spec.get("orientation_local_forward", [1.0, 0.0, 0.0]),
            spec.get("orientation_local_up", [0.0, 0.0, 1.0]),
        )
        orientation_joint_tokens = tuple(
            str(token)
            for token in spec.get("orientation_joint_tokens", [])
        )
        orientation_coordinate_mask = _joint_coordinate_mask(
            model,
            qpos_indices,
            orientation_joint_tokens,
        )
        if (
            refine_orientation
            and orientation_joint_tokens
            and not np.any(orientation_coordinate_mask)
        ):
            raise ValueError(
                f"End-effector '{name}' orientation_joint_tokens do not "
                "match any scalar robot joint."
            )
        effectors[name] = {
            "task": task,
            "body_id": body_id,
            "tracking_offset": tracking_offset,
            "contact_offsets": contact_offsets,
            "contact_target_space": contact_target_space,
            "swing_target_space": swing_target_space,
            "swing_anchor_body_id": int(swing_anchor_body_id),
            "swing_anchor_offset": (
                np.zeros(3, dtype=float)
                if task.swing_anchor_offset is None
                else np.asarray(task.swing_anchor_offset, dtype=float).reshape(3)
            ),
            "contact_position_sigma": _resolve_position_sigma(
                spec, config, "contact", "refinement", 0.025
            ),
            "swing_position_sigma": _resolve_position_sigma(
                spec, config, "swing", "refinement", 0.06
            ),
            "track_orientation": refine_orientation,
            "orientation_sigma": float(
                spec.get(
                    "orientation_sigma",
                    config.get("orientation_sigma", np.deg2rad(7.5)),
                )
            ),
            "orientation_mode": str(
                spec.get("orientation_mode", "rotation")
            ).lower(),
            "forward_orientation_sigma": max(
                float(
                    spec.get(
                        "forward_orientation_sigma",
                        spec.get(
                            "orientation_sigma",
                            config.get("orientation_sigma", np.deg2rad(7.5)),
                        ),
                    )
                ),
                1e-6,
            ),
            "up_orientation_sigma": max(
                float(
                    spec.get(
                        "up_orientation_sigma",
                        spec.get(
                            "orientation_sigma",
                            config.get("orientation_sigma", np.deg2rad(7.5)),
                        ),
                    )
                ),
                1e-6,
            ),
            "contact_orientation_weight": max(
                float(spec.get("contact_orientation_weight", 1.0)), 0.0
            ),
            "swing_orientation_weight": max(
                float(spec.get("swing_orientation_weight", 1.0)), 0.0
            ),
            "orientation_local_rotation": orientation_local_rotation,
            "orientation_coordinate_mask": (
                orientation_coordinate_mask
                if orientation_joint_tokens
                else None
            ),
            "arm_posture": arm_posture,
        }
    if not effectors:
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
    refinement_hierarchy = bool(
        config.get(
            "hierarchy",
            end_effector_config.get("hierarchy", False),
        )
    )
    contact_hierarchy_enabled = refinement_hierarchy and bool(
        config.get("contact_hierarchy", False)
    )
    swivel_hierarchy_enabled = refinement_hierarchy and bool(
        config.get("swivel_hierarchy", False)
    )
    swing_hierarchy_enabled = refinement_hierarchy
    pose_hierarchy_enabled = refinement_hierarchy

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
            )

    contact_offset_choice = _lock_contact_offsets(
        model, result, effectors, active_contact_offset_ids
    )

    reference_joints = result[:, qpos_indices].copy()
    raw_max_joint_deviation = config.get("max_joint_deviation")
    max_joint_deviation = (
        np.inf
        if raw_max_joint_deviation is None
        else float(raw_max_joint_deviation)
    )
    if max_joint_deviation <= 0.0:
        raise ValueError(
            "trajectory_refinement.max_joint_deviation must be positive."
        )
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
        block_rows: list[np.ndarray] = []
        block_columns: list[np.ndarray] = []
        block_values: list[np.ndarray] = []
        block_rhs: list[float] = []
        row_index = 0
        support_block_rows: list[np.ndarray] = []
        support_block_columns: list[np.ndarray] = []
        support_block_values: list[np.ndarray] = []
        support_block_rhs: list[float] = []
        support_row_index = 0
        contact_block_rows: list[np.ndarray] = []
        contact_block_columns: list[np.ndarray] = []
        contact_block_values: list[np.ndarray] = []
        contact_block_rhs: list[float] = []
        contact_row_index = 0
        swing_block_rows: list[np.ndarray] = []
        swing_block_columns: list[np.ndarray] = []
        swing_block_values: list[np.ndarray] = []
        swing_block_rhs: list[float] = []
        swing_row_index = 0
        pose_block_rows: list[np.ndarray] = []
        pose_block_columns: list[np.ndarray] = []
        pose_block_values: list[np.ndarray] = []
        pose_block_rhs: list[float] = []
        pose_row_index = 0
        swivel_block_rows: list[np.ndarray] = []
        swivel_block_columns: list[np.ndarray] = []
        swivel_block_values: list[np.ndarray] = []
        swivel_block_rhs: list[float] = []
        swivel_row_index = 0

        def append_block(
            frame_idx: int,
            jacobian: np.ndarray,
            target_delta: np.ndarray,
            sigma: float,
            weight: float = 1.0,
            priority: str = "regular",
        ) -> None:
            nonlocal row_index, support_row_index, contact_row_index
            nonlocal swing_row_index, pose_row_index, swivel_row_index
            scale = np.sqrt(max(float(weight), 1e-8)) / sigma
            block = np.atleast_2d(np.asarray(jacobian, dtype=float)) * scale
            target = np.asarray(target_delta, dtype=float).reshape(-1) * scale
            local_rows, local_columns = np.nonzero(np.abs(block) > 1e-14)
            if priority == "support":
                support_block_rows.append(local_rows + support_row_index)
                support_block_columns.append(frame_idx * num_joints + local_columns)
                support_block_values.append(block[local_rows, local_columns])
                support_block_rhs.extend(target.tolist())
                support_row_index += block.shape[0]
            elif priority == "contact" and contact_hierarchy_enabled:
                contact_block_rows.append(local_rows + contact_row_index)
                contact_block_columns.append(frame_idx * num_joints + local_columns)
                contact_block_values.append(block[local_rows, local_columns])
                contact_block_rhs.extend(target.tolist())
                contact_row_index += block.shape[0]
            elif priority == "swing" and swing_hierarchy_enabled:
                swing_block_rows.append(local_rows + swing_row_index)
                swing_block_columns.append(
                    frame_idx * num_joints + local_columns
                )
                swing_block_values.append(block[local_rows, local_columns])
                swing_block_rhs.extend(target.tolist())
                swing_row_index += block.shape[0]
            elif priority == "pose" and pose_hierarchy_enabled:
                pose_block_rows.append(local_rows + pose_row_index)
                pose_block_columns.append(
                    frame_idx * num_joints + local_columns
                )
                pose_block_values.append(block[local_rows, local_columns])
                pose_block_rhs.extend(target.tolist())
                pose_row_index += block.shape[0]
            elif priority == "swivel" and swivel_hierarchy_enabled:
                swivel_block_rows.append(local_rows + swivel_row_index)
                swivel_block_columns.append(
                    frame_idx * num_joints + local_columns
                )
                swivel_block_values.append(block[local_rows, local_columns])
                swivel_block_rhs.extend(target.tolist())
                swivel_row_index += block.shape[0]
            else:
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
            for name, spec in effectors.items():
                task: SceneContactTask = spec["task"]
                body_id = int(spec["body_id"])
                rotation = data.xmat[body_id].reshape(3, 3).copy()
                mode = ContactMode(int(task.contact_modes[frame_idx]))
                in_contact = mode != ContactMode.NONE
                position_sigma_scale = (
                    1.0
                    if task.position_sigma_scale is None
                    else float(task.position_sigma_scale[frame_idx])
                )
                orientation_sigma_scale = (
                    1.0
                    if task.orientation_sigma_scale is None
                    else float(task.orientation_sigma_scale[frame_idx])
                )
                if in_contact:
                    target_position = task.contact_positions[frame_idx]
                    offset = spec["contact_offsets"][
                        contact_offset_choice[name][frame_idx]
                    ]
                    position_sigma = (
                        contact_position_sigma
                        if spec["contact_position_sigma"] is None
                        else float(spec["contact_position_sigma"])
                    )
                    position_sigma *= position_sigma_scale
                    if mode == ContactMode.SUPPORT:
                        confidence = 1.0
                    else:
                        transition_weight = float(
                            task.transition_weights[frame_idx]
                        )
                        confidence = max(
                            float(task.contact_confidence[frame_idx]), 0.25
                        ) * transition_weight
                        if mode == ContactMode.TOUCH:
                            confidence *= float(
                                end_effector_config.get("touch_weight", 0.5)
                            )
                else:
                    if task.effective_swing_positions is None:
                        raise RuntimeError(
                            f"{name}: effective swing targets were not prepared."
                        )
                    target_position = task.effective_swing_positions[frame_idx]
                    offset = spec["tracking_offset"]
                    position_sigma = (
                        swing_position_sigma
                        if spec["swing_position_sigma"] is None
                        else float(spec["swing_position_sigma"])
                    )
                    position_sigma *= position_sigma_scale
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
                position_jacobian = jacobian_position[:, dof_indices].copy()
                if (
                    not in_contact
                    and spec["swing_target_space"]
                    == "limb_morphology_scaled"
                ):
                    anchor_body_id = int(spec["swing_anchor_body_id"])
                    anchor_rotation = data.xmat[anchor_body_id].reshape(3, 3)
                    anchor_point = (
                        data.xpos[anchor_body_id]
                        + anchor_rotation @ spec["swing_anchor_offset"]
                    )
                    anchor_jacobian_position = np.zeros(
                        (3, model.nv), dtype=float
                    )
                    anchor_jacobian_rotation = np.zeros(
                        (3, model.nv), dtype=float
                    )
                    mujoco.mj_jac(
                        model,
                        data,
                        anchor_jacobian_position,
                        anchor_jacobian_rotation,
                        anchor_point,
                        anchor_body_id,
                    )
                    alpha = float(task.contact_bridge_alpha[frame_idx])
                    anchor_weight = 1.0 - alpha
                    position_jacobian -= (
                        anchor_weight
                        * anchor_jacobian_position[:, dof_indices]
                    )
                    relative_rhs = (
                        anchor_weight * task.swing_limb_offsets[frame_idx]
                        + alpha * task.contact_bridge_positions[frame_idx]
                    )
                    position_delta = relative_rhs - (
                        point - anchor_weight * anchor_point
                    )
                else:
                    position_delta = target_position - point
                if mode == ContactMode.SLIDE:
                    normal = np.asarray(
                        task.surface_normals[frame_idx], dtype=float
                    ).reshape(3)
                    normal /= max(float(np.linalg.norm(normal)), 1e-10)
                    append_block(
                        frame_idx,
                        normal @ position_jacobian,
                        normal @ position_delta,
                        position_sigma,
                        confidence,
                        "support",
                    )
                    tangential_weight = max(
                        float(
                            end_effector_config.get(
                                "slide_tangential_weight", 0.1
                            )
                        ),
                        0.0,
                    )
                    if tangential_weight > 0.0:
                        tangent_projection = np.eye(3) - np.outer(normal, normal)
                        append_block(
                            frame_idx,
                            tangent_projection @ position_jacobian,
                            tangent_projection @ position_delta,
                            position_sigma,
                            confidence * tangential_weight,
                            "contact",
                        )
                else:
                    append_block(
                        frame_idx,
                        position_jacobian,
                        position_delta,
                        position_sigma,
                        confidence,
                        (
                            "support"
                            if mode == ContactMode.SUPPORT
                            else "contact"
                            if mode == ContactMode.TOUCH
                            else "swing"
                        ),
                    )

                arm_posture = spec["arm_posture"]
                if arm_posture is not None and source_joint_positions is not None:
                    limb_anchored = (
                        spec["swing_target_space"] == "limb_morphology_scaled"
                    )
                    if task.morphology_scale is None or (
                        not limb_anchored
                        and (
                            source_root_positions is None
                            or task.robot_base_reference_w is None
                        )
                    ):
                        raise RuntimeError(
                            f"{name}: arm posture transport requires source and robot "
                            "root references."
                        )
                    shoulder_body_id = int(arm_posture["shoulder_body_id"])
                    elbow_body_id = int(arm_posture["elbow_body_id"])
                    shoulder_position = data.xpos[shoulder_body_id].copy()
                    elbow_position = data.xpos[elbow_body_id].copy()
                    current_upper = elbow_position - shoulder_position
                    source_frame = source_joint_positions[frame_idx]
                    source_shoulder = source_frame[
                        int(arm_posture["source_shoulder_idx"])
                    ]
                    wrist_body_id = int(arm_posture["wrist_body_id"])
                    use_wrist_center = wrist_body_id >= 0
                    source_endpoint = (
                        source_frame[int(arm_posture["source_wrist_idx"])]
                        if use_wrist_center
                        else task.tracking_positions[frame_idx]
                    )
                    # Refinement already knows where the robot shoulder is, so
                    # the limb-anchored shape target uses the solved shoulder
                    # rather than the base-orientation reference used to build
                    # the framewise swing targets.
                    shape_endpoint_target = _transform_swing_target(
                        source_endpoint,
                        source_shoulder
                        if limb_anchored
                        else source_root_positions[frame_idx],
                        shoulder_position
                        if limb_anchored
                        else task.robot_base_reference_w[frame_idx],
                        float(task.morphology_scale),
                        spec["swing_target_space"],
                    )
                    target_upper, target_forearm = (
                        _transported_arm_segment_targets(
                            source_shoulder,
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
                        1.0,
                        "pose",
                    )
                    append_block(
                        frame_idx,
                        (shape_endpoint_jacobian - elbow_jacobian)[:, dof_indices],
                        target_forearm - current_forearm,
                        float(arm_posture["forearm_direction_sigma"]),
                        1.0,
                        "pose",
                    )
                    if (
                        bool(arm_posture["swivel_enabled"])
                        and use_wrist_center
                        and float(arm_posture["swivel_weight"]) > 0.0
                    ):
                        source_direction, _, source_fade, _, _ = (
                            _swivel_direction_and_jacobian(
                                source_frame[
                                    int(arm_posture["source_shoulder_idx"])
                                ],
                                source_frame[
                                    int(arm_posture["source_elbow_idx"])
                                ],
                                source_endpoint,
                                min_sine=float(
                                    arm_posture["swivel_min_sine"]
                                ),
                                full_sine=float(
                                    arm_posture["swivel_full_sine"]
                                ),
                            )
                        )
                        (
                            robot_direction,
                            swivel_jacobian,
                            robot_fade,
                            _,
                            robot_axis,
                        ) = _swivel_direction_and_jacobian(
                            shoulder_position,
                            elbow_position,
                            shape_endpoint_position,
                            shoulder_jacobian,
                            elbow_jacobian,
                            shape_endpoint_jacobian,
                            min_sine=float(arm_posture["swivel_min_sine"]),
                            full_sine=float(arm_posture["swivel_full_sine"]),
                        )
                        swivel_weight = float(
                            arm_posture["swivel_weight"]
                        ) * min(source_fade, robot_fade)
                        if swivel_weight > 0.0 and swivel_jacobian is not None:
                            append_block(
                                frame_idx,
                                swivel_jacobian[:, dof_indices],
                                _swivel_tangent_delta(
                                    robot_direction,
                                    source_direction,
                                    robot_axis,
                                ),
                                float(arm_posture["swivel_sigma"]),
                                swivel_weight,
                                "swivel",
                            )

                if spec["track_orientation"]:
                    target_rotation = (
                        None
                        if task.orientation_target_rotations is None
                        else task.orientation_target_rotations[frame_idx]
                    )
                    if target_rotation is not None and not np.all(
                        np.isfinite(target_rotation)
                    ):
                        target_rotation = None
                    if target_rotation is None:
                        target_up = (
                            task.surface_normals[frame_idx]
                            if mode in (ContactMode.SUPPORT, ContactMode.SLIDE)
                            else (
                                None
                                if task.up_directions is None
                                else task.up_directions[frame_idx]
                            )
                        )
                        target_rotation = _orthonormal_target_rotation(
                            None
                            if task.forward_directions is None
                            else task.forward_directions[frame_idx],
                            target_up,
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
                        forward_orientation_sigma = max(
                            float(spec["forward_orientation_sigma"])
                            * orientation_sigma_scale,
                            1e-6,
                        )
                        up_orientation_sigma = max(
                            float(spec["up_orientation_sigma"])
                            * orientation_sigma_scale,
                            1e-6,
                        )
                        orientation_sigma = max(
                            float(spec["orientation_sigma"])
                            * orientation_sigma_scale,
                            1e-6,
                        )
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
                                forward_orientation_sigma,
                                orientation_weight,
                            )
                            append_block(
                                frame_idx,
                                -_skew_symmetric(current_up)
                                @ orientation_jacobian,
                                target_rotation[:, 2] - current_up,
                                up_orientation_sigma,
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
                                orientation_sigma,
                                orientation_weight,
                            )

            if refinement_pose_edges and source_joint_positions is not None:
                pose_sigma = max(
                    float(pose_direction_cfg.get("sigma", 0.12)), 1e-6
                )
                source_frame = source_joint_positions[frame_idx]
                for edge in refinement_pose_edges:
                    parent_body_id = int(edge["parent_body_id"])
                    child_body_id = int(edge["child_body_id"])
                    parent_position = data.xpos[parent_body_id].copy()
                    child_position = data.xpos[child_body_id].copy()
                    current_vector = child_position - parent_position
                    current_length = float(np.linalg.norm(current_vector))
                    source_vector = (
                        source_frame[int(edge["source_child_idx"])]
                        - source_frame[int(edge["source_parent_idx"])]
                    )
                    source_length = float(np.linalg.norm(source_vector))
                    if current_length < 1e-8 or source_length < 1e-8:
                        continue
                    current_direction = current_vector / current_length
                    source_direction = source_vector / source_length
                    parent_jacobian = np.zeros_like(jacobian_position)
                    child_jacobian = np.zeros_like(jacobian_position)
                    mujoco.mj_jac(
                        model,
                        data,
                        parent_jacobian,
                        None,
                        parent_position,
                        parent_body_id,
                    )
                    mujoco.mj_jac(
                        model,
                        data,
                        child_jacobian,
                        None,
                        child_position,
                        child_body_id,
                    )
                    vector_jacobian = (
                        child_jacobian - parent_jacobian
                    )[:, dof_indices]
                    direction_jacobian = (
                        (np.eye(3) - np.outer(
                            current_direction, current_direction
                        ))
                        @ vector_jacobian
                        / current_length
                    )
                    append_block(
                        frame_idx,
                        direction_jacobian,
                        source_direction - current_direction,
                        pose_sigma,
                        1.0,
                        "pose",
                    )

            frame_penetration = penetration_frames[frame_idx]
            if frame_penetration is None:
                continue
            # An effector held on its scene anchor cannot leave the surface;
            # a penetration row on near-anchor samples has the same stiffness
            # as the contact row and only drags the anchored point in a
            # tug-of-war.  The exemption is by distance, not by body id: the
            # colliding geoms often live on the wrist/ankle parent body rather
            # than the configured effector body.  The honest depth still comes
            # from measure_scene_penetration.
            active_anchor_points = [
                np.asarray(spec["task"].contact_positions[frame_idx], dtype=float)
                for spec in effectors.values()
                if penetration_anchor_radius > 0.0
                and spec["task"].contact_modes[frame_idx]
                == int(ContactMode.SUPPORT)
            ]
            # One-sided: only points that violate the requested clearance get a
            # row, and the active set is rebuilt from the current pose at every
            # Gauss--Newton iteration.
            for index in np.flatnonzero(
                frame_penetration["clearance"] < penetration_clearance
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
        support_block = sparse.coo_matrix(
            (
                np.concatenate(support_block_values)
                if support_block_values
                else np.zeros(0, dtype=float),
                (
                    np.concatenate(support_block_rows)
                    if support_block_rows
                    else np.zeros(0, dtype=int),
                    np.concatenate(support_block_columns)
                    if support_block_columns
                    else np.zeros(0, dtype=int),
                ),
            ),
            shape=(support_row_index, num_frames * num_joints),
        ).tocsr()
        contact_block = sparse.coo_matrix(
            (
                np.concatenate(contact_block_values)
                if contact_block_values
                else np.zeros(0, dtype=float),
                (
                    np.concatenate(contact_block_rows)
                    if contact_block_rows
                    else np.zeros(0, dtype=int),
                    np.concatenate(contact_block_columns)
                    if contact_block_columns
                    else np.zeros(0, dtype=int),
                ),
            ),
            shape=(contact_row_index, num_frames * num_joints),
        ).tocsr()
        swing_block = sparse.coo_matrix(
            (
                np.concatenate(swing_block_values)
                if swing_block_values
                else np.zeros(0, dtype=float),
                (
                    np.concatenate(swing_block_rows)
                    if swing_block_rows
                    else np.zeros(0, dtype=int),
                    np.concatenate(swing_block_columns)
                    if swing_block_columns
                    else np.zeros(0, dtype=int),
                ),
            ),
            shape=(swing_row_index, num_frames * num_joints),
        ).tocsr()
        pose_block = sparse.coo_matrix(
            (
                np.concatenate(pose_block_values)
                if pose_block_values
                else np.zeros(0, dtype=float),
                (
                    np.concatenate(pose_block_rows)
                    if pose_block_rows
                    else np.zeros(0, dtype=int),
                    np.concatenate(pose_block_columns)
                    if pose_block_columns
                    else np.zeros(0, dtype=int),
                ),
            ),
            shape=(pose_row_index, num_frames * num_joints),
        ).tocsr()
        swivel_block = sparse.coo_matrix(
            (
                np.concatenate(swivel_block_values)
                if swivel_block_values
                else np.zeros(0, dtype=float),
                (
                    np.concatenate(swivel_block_rows)
                    if swivel_block_rows
                    else np.zeros(0, dtype=int),
                    np.concatenate(swivel_block_columns)
                    if swivel_block_columns
                    else np.zeros(0, dtype=int),
                ),
            ),
            shape=(swivel_row_index, num_frames * num_joints),
        ).tocsr()

        systems = [task_block]
        residuals = [np.asarray(block_rhs, dtype=float)]
        posture_systems = [reference_block]
        posture_residuals = [
            ((reference_joints - current_joints) / reference_sigma).reshape(-1)
        ]

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
                in_contact = np.asarray(
                    effectors[effector_name]["task"].contact_modes,
                    dtype=np.uint8,
                ) != int(ContactMode.NONE)
                swing_pairs = ~(in_contact[:-1] | in_contact[1:])
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

        if posture_block is not None:
            posture_systems.append(posture_block)
            posture_residuals.append(
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

        motion_system = sparse.vstack(systems, format="csr")
        motion_rhs = np.concatenate(residuals)
        posture_system = sparse.vstack(posture_systems, format="csr")
        posture_rhs = np.concatenate(posture_residuals)
        system = sparse.vstack([motion_system, posture_system], format="csr")
        rhs_array = np.concatenate([motion_rhs, posture_rhs])

        # A framewise solution can sit marginally outside a joint limit.  The
        # step bounds must always admit zero, otherwise the joint limit and the
        # trust region cross over and the solver rejects the problem outright.
        absolute_lower = np.maximum(
            np.tile(joint_lower, num_frames),
            reference_joints.reshape(-1) - max_joint_deviation,
        )
        absolute_upper = np.minimum(
            np.tile(joint_upper, num_frames),
            reference_joints.reshape(-1) + max_joint_deviation,
        )
        lower = np.minimum(
            np.maximum(
                absolute_lower - current_joints.reshape(-1),
                -trust_region,
            ),
            0.0,
        )
        upper = np.maximum(
            np.minimum(
                absolute_upper - current_joints.reshape(-1),
                trust_region,
            ),
            0.0,
        )
        support_optimum = None
        contact_optimum = None
        contact_bound = None
        contact_absolute_tolerance = None
        contact_relative_tolerance = None
        swivel_optimum = None
        has_support_priority = refinement_hierarchy and support_row_index > 0
        has_contact_priority = (
            contact_hierarchy_enabled and contact_row_index > 0
        )
        has_swing_priority = swing_hierarchy_enabled and swing_row_index > 0
        has_pose_priority = pose_hierarchy_enabled and pose_row_index > 0
        has_swivel_priority = swivel_hierarchy_enabled and swivel_row_index > 0
        if (
            has_support_priority
            or has_contact_priority
            or has_swing_priority
            or has_pose_priority
            or has_swivel_priority
        ):
            delta = cp.Variable(num_frames * num_joints)
            hierarchy_constraints = [
                delta >= lower,
                delta <= upper,
            ]
            support_solution = None
            priority_solution_x = None
            if has_support_priority:
                support_rhs = np.asarray(support_block_rhs, dtype=float)
                support_solution = lsq_linear(
                    support_block,
                    support_rhs,
                    bounds=(lower, upper),
                    method="trf",
                    lsq_solver="lsmr",
                    tol=solver_tolerance,
                    lsmr_tol=solver_tolerance,
                    max_iter=solver_iterations,
                    verbose=0,
                )
                support_optimum = float(
                    np.linalg.norm(
                        support_block @ support_solution.x - support_rhs
                    )
                )
                support_relative_tolerance = max(
                    float(
                        config.get(
                            "support_priority_relative_tolerance", 0.01
                        )
                    ),
                    0.0,
                )
                support_absolute_tolerance = max(
                    float(
                        config.get(
                            "support_priority_absolute_tolerance", 0.05
                        )
                    ),
                    0.0,
                )
                support_bound = (
                    (1.0 + support_relative_tolerance) * support_optimum
                    + support_absolute_tolerance
                )
                hierarchy_constraints.append(
                    cp.norm(support_block @ delta - support_rhs, 2)
                    <= support_bound
                )
                priority_solution_x = np.asarray(
                    support_solution.x, dtype=float
                ).reshape(-1)
            contact_problem = None
            if has_contact_priority:
                contact_rhs = np.asarray(contact_block_rhs, dtype=float)
                contact_problem = cp.Problem(
                    cp.Minimize(
                        cp.sum_squares(contact_block @ delta - contact_rhs)
                    ),
                    hierarchy_constraints,
                )
                try:
                    contact_problem.solve(solver=cp.CLARABEL, verbose=False)
                except cp.error.SolverError:
                    pass
                if contact_problem.status not in (
                    cp.OPTIMAL,
                    cp.OPTIMAL_INACCURATE,
                ):
                    contact_problem.solve(
                        solver=cp.SCS,
                        eps=solver_tolerance,
                        max_iters=max(solver_iterations * 200, 2000),
                        verbose=False,
                    )
                if contact_problem.status in (
                    cp.OPTIMAL,
                    cp.OPTIMAL_INACCURATE,
                ):
                    contact_optimum = float(
                        np.linalg.norm(
                            contact_block @ np.asarray(delta.value).reshape(-1)
                            - contact_rhs
                        )
                    )
                    contact_relative_tolerance = max(
                        float(
                            config.get(
                                "contact_priority_relative_tolerance", 0.01
                            )
                        ),
                        0.0,
                    )
                    contact_absolute_tolerance = max(
                        float(
                            config.get(
                                "contact_priority_absolute_tolerance",
                                DEFAULT_CONTACT_PRIORITY_ABSOLUTE_TOLERANCE,
                            )
                        ),
                        0.0,
                    )
                    contact_bound = (
                        (1.0 + contact_relative_tolerance) * contact_optimum
                        + contact_absolute_tolerance
                    )
                    hierarchy_constraints.append(
                        cp.norm(contact_block @ delta - contact_rhs, 2)
                        <= contact_bound
                    )
                    priority_solution_x = np.asarray(
                        delta.value, dtype=float
                    ).reshape(-1)

            failed_priority_problem = (
                contact_problem
                if contact_problem is not None
                and contact_problem.status
                not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
                else None
            )
            kinematic_relative_tolerance = max(
                float(
                    config.get(
                        "temporal_priority_relative_tolerance", 0.01
                    )
                ),
                0.0,
            )
            kinematic_absolute_tolerance = max(
                float(
                    config.get(
                        "temporal_priority_absolute_tolerance", 0.05
                    )
                ),
                0.0,
            )
            for (
                stage_name,
                stage_enabled,
                stage_block,
                stage_rhs_values,
            ) in (
                (
                    "swing",
                    has_swing_priority,
                    swing_block,
                    swing_block_rhs,
                ),
                (
                    "pose",
                    has_pose_priority,
                    pose_block,
                    pose_block_rhs,
                ),
            ):
                if failed_priority_problem is not None or not stage_enabled:
                    continue
                stage_rhs = np.asarray(stage_rhs_values, dtype=float)
                stage_problem = cp.Problem(
                    cp.Minimize(
                        cp.sum_squares(stage_block @ delta - stage_rhs)
                    ),
                    hierarchy_constraints,
                )
                try:
                    stage_problem.solve(solver=cp.CLARABEL, verbose=False)
                except cp.error.SolverError:
                    pass
                if stage_problem.status not in (
                    cp.OPTIMAL,
                    cp.OPTIMAL_INACCURATE,
                ):
                    stage_problem.solve(
                        solver=cp.SCS,
                        eps=solver_tolerance,
                        max_iters=max(solver_iterations * 200, 2000),
                        verbose=False,
                    )
                if stage_problem.status in (
                    cp.OPTIMAL,
                    cp.OPTIMAL_INACCURATE,
                ):
                    stage_optimum = float(
                        np.linalg.norm(
                            stage_block @ np.asarray(delta.value).reshape(-1)
                            - stage_rhs
                        )
                    )
                    hierarchy_constraints.append(
                        cp.norm(stage_block @ delta - stage_rhs, 2)
                        <= (1.0 + kinematic_relative_tolerance)
                        * stage_optimum
                        + kinematic_absolute_tolerance
                    )
                    priority_solution_x = np.asarray(
                        delta.value, dtype=float
                    ).reshape(-1)
                else:
                    failed_priority_problem = stage_problem

            swivel_problem = None
            if failed_priority_problem is None and has_swivel_priority:
                swivel_rhs = np.asarray(swivel_block_rhs, dtype=float)
                swivel_problem = cp.Problem(
                    cp.Minimize(
                        cp.sum_squares(swivel_block @ delta - swivel_rhs)
                    ),
                    hierarchy_constraints,
                )
                try:
                    swivel_problem.solve(solver=cp.CLARABEL, verbose=False)
                except cp.error.SolverError:
                    pass
                if swivel_problem.status not in (
                    cp.OPTIMAL,
                    cp.OPTIMAL_INACCURATE,
                ):
                    swivel_problem.solve(
                        solver=cp.SCS,
                        eps=solver_tolerance,
                        max_iters=max(solver_iterations * 200, 2000),
                        verbose=False,
                    )
                if swivel_problem.status in (
                    cp.OPTIMAL,
                    cp.OPTIMAL_INACCURATE,
                ):
                    swivel_optimum = float(
                        np.linalg.norm(
                            swivel_block @ np.asarray(delta.value).reshape(-1)
                            - swivel_rhs
                        )
                    )
                    swivel_relative_tolerance = max(
                        float(
                            config.get(
                                "swivel_priority_relative_tolerance", 0.01
                            )
                        ),
                        0.0,
                    )
                    swivel_absolute_tolerance = max(
                        float(
                            config.get(
                                "swivel_priority_absolute_tolerance", 5.0
                            )
                        ),
                        0.0,
                    )
                    hierarchy_constraints.append(
                        cp.norm(swivel_block @ delta - swivel_rhs, 2)
                        <= (1.0 + swivel_relative_tolerance) * swivel_optimum
                        + swivel_absolute_tolerance
                    )
                    priority_solution_x = np.asarray(
                        delta.value, dtype=float
                    ).reshape(-1)
                else:
                    failed_priority_problem = swivel_problem

            if failed_priority_problem is not None:
                hierarchy_problem = failed_priority_problem
            else:
                hierarchy_problem = cp.Problem(
                    cp.Minimize(
                        cp.sum_squares(motion_system @ delta - motion_rhs)
                    ),
                    hierarchy_constraints,
                )
                try:
                    hierarchy_problem.solve(solver=cp.CLARABEL, verbose=False)
                except cp.error.SolverError:
                    pass
                if hierarchy_problem.status not in (
                    cp.OPTIMAL,
                    cp.OPTIMAL_INACCURATE,
                ):
                    hierarchy_problem.solve(
                        solver=cp.SCS,
                        eps=solver_tolerance,
                        max_iters=max(solver_iterations * 200, 2000),
                        verbose=False,
                    )
            if hierarchy_problem.status not in (
                cp.OPTIMAL,
                cp.OPTIMAL_INACCURATE,
            ):
                # Preserve the highest achieved priority rather than silently
                # reverting to a weighted solve that can move the contact.
                solution_x = (
                    np.zeros(num_frames * num_joints, dtype=float)
                    if priority_solution_x is None
                    else priority_solution_x
                )
                solution_cost = 0.5 * float(
                    np.linalg.norm(system @ solution_x - rhs_array) ** 2
                )
                solution_status = 1
            else:
                motion_solution_x = np.asarray(delta.value, dtype=float).reshape(-1)
                motion_optimum = float(
                    np.linalg.norm(
                        motion_system @ motion_solution_x - motion_rhs
                    )
                )
                if bool(config.get("posture_hierarchy", True)):
                    motion_relative_tolerance = max(
                        float(
                            config.get(
                                "temporal_priority_relative_tolerance", 0.01
                            )
                        ),
                        0.0,
                    )
                    motion_absolute_tolerance = max(
                        float(
                            config.get(
                                "temporal_priority_absolute_tolerance", 0.05
                            )
                        ),
                        0.0,
                    )
                    posture_constraints = hierarchy_constraints + [
                        cp.norm(motion_system @ delta - motion_rhs, 2)
                        <= (1.0 + motion_relative_tolerance) * motion_optimum
                        + motion_absolute_tolerance
                    ]
                    posture_problem = cp.Problem(
                        cp.Minimize(
                            cp.sum_squares(posture_system @ delta - posture_rhs)
                        ),
                        posture_constraints,
                    )
                    try:
                        posture_problem.solve(solver=cp.CLARABEL, verbose=False)
                    except cp.error.SolverError:
                        pass
                    if posture_problem.status not in (
                        cp.OPTIMAL,
                        cp.OPTIMAL_INACCURATE,
                    ):
                        posture_problem.solve(
                            solver=cp.SCS,
                            eps=solver_tolerance,
                            max_iters=max(solver_iterations * 200, 2000),
                            verbose=False,
                        )
                    solution_x = (
                        np.asarray(delta.value, dtype=float).reshape(-1)
                        if posture_problem.status
                        in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
                        else motion_solution_x
                    )
                else:
                    solution_x = motion_solution_x
                solution_cost = 0.5 * float(
                    np.linalg.norm(system @ solution_x - rhs_array) ** 2
                )
                solution_status = 1
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
            solution_x = solution.x
            solution_cost = float(solution.cost)
            solution_status = int(getattr(solution, "status", 1))
        if not np.all(np.isfinite(solution_x)):
            raise RuntimeError("trajectory refinement produced a non-finite step")
        if solution_status == 0:
            print(
                "Trajectory refinement warning: the bounded least-squares solve "
                f"reached solver_iterations={solver_iterations} without "
                "converging, so this step is truncated."
            )
        step = solution_x.reshape(num_frames, num_joints)
        if (
            diagnostics is not None
            and has_contact_priority
            and contact_optimum is not None
            and contact_bound is not None
            and contact_relative_tolerance is not None
            and contact_absolute_tolerance is not None
        ):
            final_contact_norm = float(
                np.linalg.norm(contact_block @ solution_x - contact_rhs)
            )
            iteration_diagnostic = {
                "level_optimum": float(contact_optimum),
                "configured_relative_tolerance": float(
                    contact_relative_tolerance
                ),
                "configured_absolute_tolerance": float(
                    contact_absolute_tolerance
                ),
                "configured_bound": float(contact_bound),
                "final_residual_norm": final_contact_norm,
                "slack_consumed": max(
                    final_contact_norm - float(contact_optimum), 0.0
                ),
                "active_residual_rows": int(contact_row_index),
                "solver_status": str(hierarchy_problem.status),
                "max_joint_step": float(np.max(np.abs(step))),
            }
            history = diagnostics.setdefault("contact_priority_iterations", [])
            history.append(iteration_diagnostic.copy())
            iteration_diagnostic["solver_failure_count"] = int(
                sum(
                    entry["solver_status"]
                    not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
                    for entry in history
                )
            )
            iteration_diagnostic["max_joint_step"] = float(
                max(entry["max_joint_step"] for entry in history)
            )
            # Physical error must be measured from the refined trajectory.
            # A closed-form conversion is not valid because a configured level
            # may mix residual rows with different sigmas and mode weights.
            diagnostics["contact_priority"] = iteration_diagnostic
        result[:, qpos_indices] = current_joints + step
        print(
            "Trajectory refinement "
            f"{iteration + 1}/{iterations}: cost={solution_cost:.4f}, "
            f"max_joint_step={np.max(np.abs(step)):.5f}, "
            f"support_rows={support_row_index}, "
            f"contact_rows={contact_row_index}, "
            f"swing_rows={swing_row_index}, "
            f"pose_rows={pose_row_index}, "
            f"swivel_rows={swivel_row_index}, "
            + (
                f"support_optimum={support_optimum:.5f}, "
                if support_optimum is not None
                else ""
            )
            + (
                f"contact_optimum={contact_optimum:.5f}, "
                if contact_optimum is not None
                else ""
            )
            + (
                f"swivel_optimum={swivel_optimum:.5f}, "
                if swivel_optimum is not None
                else ""
            )
            + f"penetration_rows={active_penetration_rows}, "
            f"joint_limit_rows={active_limit_rows}, "
            f"swing_step_rows={active_swing_step_rows}, "
            f"joint_step_rows={active_joint_step_rows}"
        )
        if float(np.max(np.abs(step))) < step_tolerance:
            break

    return result
