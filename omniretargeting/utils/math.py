"""Utility functions for OmniRetargeting."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import trimesh
from scipy.spatial.transform import Rotation
import mujoco


def sample_mujoco_geom_local_points(
    geom_type: int,
    size: np.ndarray,
    ring_samples: int = 12,
) -> np.ndarray:
    """Return representative surface points in a MuJoCo geom's local frame.

    MuJoCo cylinders and capsules are aligned with local Z.  Keeping the
    primitive construction in one place prevents terrain penetration and foot
    stabilization from using different axis conventions.
    """
    geom_type = int(geom_type)
    size = np.asarray(size, dtype=float)

    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        radius = size[0]
        return radius * np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ]
        )

    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        hx, hy, hz = size[:3]
        corners = [
            [sx, sy, sz]
            for sx in (-hx, hx)
            for sy in (-hy, hy)
            for sz in (-hz, hz)
        ]
        corners.extend([[0.0, 0.0, -hz], [0.0, 0.0, hz]])
        return np.asarray(corners, dtype=float)

    if geom_type in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        radius, half_length = size[:2]
        theta = np.linspace(0.0, 2.0 * np.pi, ring_samples, endpoint=False)
        rings = [
            np.column_stack(
                [
                    radius * np.cos(theta),
                    radius * np.sin(theta),
                    np.full_like(theta, z),
                ]
            )
            for z in (-half_length, 0.0, half_length)
        ]
        if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            endpoints = [
                [0.0, 0.0, -half_length - radius],
                [0.0, 0.0, half_length + radius],
            ]
        else:
            endpoints = [
                [0.0, 0.0, -half_length],
                [0.0, 0.0, half_length],
            ]
        return np.vstack([*rings, np.asarray(endpoints, dtype=float)])

    if geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        rx, ry, rz = size[:3]
        return np.array(
            [
                [rx, 0.0, 0.0],
                [-rx, 0.0, 0.0],
                [0.0, ry, 0.0],
                [0.0, -ry, 0.0],
                [0.0, 0.0, rz],
                [0.0, 0.0, -rz],
            ]
        )

    if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
        return np.empty((0, 3), dtype=float)

    return np.zeros((1, 3), dtype=float)


def linear_interpolate(
    array: np.ndarray,
    indices: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    """Linearly interpolate *array* at fractional *indices* along *axis*.

    Parameters
    ----------
    array:
        Source array of any shape.
    indices:
        1-D array of float indices into *axis* (must be in [0, N-1]
        where N = array.shape[axis]).
    axis:
        The axis along which to interpolate.

    Returns
    -------
    np.ndarray
        Array whose shape equals ``array.shape`` with ``array.shape[axis]``
        replaced by ``len(indices)``.
    """
    axis = axis % array.ndim
    n = array.shape[axis]
    indices = np.asarray(indices, dtype=np.float64)

    lo = np.clip(np.floor(indices).astype(int), 0, n - 1)
    hi = np.clip(lo + 1, 0, n - 1)
    frac = indices - lo

    # Reshape frac for broadcasting: length-M along *axis*, size-1 elsewhere.
    shape = [1] * array.ndim
    shape[axis] = len(indices)
    frac = frac.reshape(shape)

    return np.take(array, lo, axis=axis) * (1 - frac) + np.take(array, hi, axis=axis) * frac


def _rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """Axis-angle (..., 3) -> quaternion (..., 4) in wxyz order."""
    original_shape = rotvec.shape[:-1]
    flat = rotvec.reshape(-1, 3)
    angle = np.linalg.norm(flat, axis=-1, keepdims=True)
    safe_angle = np.where(angle > 1e-10, angle, np.ones_like(angle))
    axis_normed = flat / safe_angle
    half = angle / 2
    w = np.cos(half)
    xyz = axis_normed * np.sin(half)
    near_zero = (angle < 1e-10).squeeze(-1)
    w[near_zero] = 1.0
    xyz[near_zero] = 0.0
    quat = np.concatenate([w, xyz], axis=-1)
    return quat.reshape(*original_shape, 4)


def _quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    """Quaternion (..., 4) in wxyz order -> axis-angle (..., 3)."""
    original_shape = quat.shape[:-1]
    flat = quat.reshape(-1, 4)
    flat = np.where(flat[:, :1] < 0, -flat, flat)
    w = np.clip(flat[:, 0:1], -1.0, 1.0)
    xyz = flat[:, 1:4]
    half_angle = np.arccos(w)
    angle = 2 * half_angle
    sin_half = np.sin(half_angle)
    safe_sin = np.where(sin_half > 1e-10, sin_half, np.ones_like(sin_half))
    axis_normed = xyz / safe_sin
    rotvec = axis_normed * angle
    near_zero = (sin_half < 1e-10).squeeze(-1)
    rotvec[near_zero] = 0.0
    return rotvec.reshape(*original_shape, 3)


def slerp_interpolate(
    array: np.ndarray,
    indices: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    """Spherical linear interpolation for wxyz quaternion arrays at fractional *indices*.

    Parameters
    ----------
    array:
        Array of wxyz quaternion data.  The last dimension must be 4
        and *axis* selects the time/sequence dimension.  E.g. shape
        ``(T, 4)`` for a single rotation track or ``(T, J, 4)`` for
        *J* joints.
    indices:
        1-D array of float indices into *axis* (in ``[0, N-1]``).
    axis:
        The axis along which to interpolate (the "time" axis).

    Returns
    -------
    np.ndarray
        wxyz quaternion array with ``array.shape[axis]`` replaced by
        ``len(indices)``, same dtype.
    """
    assert array.shape[-1] == 4, f"Last dimension must be 4 (wxyz quaternion), got {array.shape[-1]}"
    axis = axis % array.ndim
    n = array.shape[axis]
    indices = np.asarray(indices, dtype=np.float64)

    lo = np.clip(np.floor(indices).astype(int), 0, n - 1)
    hi = np.clip(lo + 1, 0, n - 1)
    frac = indices - lo

    q0 = np.take(array, lo, axis=axis)
    q1 = np.take(array, hi, axis=axis)

    # Shortest-path: flip q1 when dot < 0
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)

    shape = [1] * q0.ndim
    shape[axis] = len(indices)
    frac = frac.reshape(shape)

    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    near_parallel = (sin_theta < 1e-10)
    safe_sin_theta = np.where(near_parallel, np.ones_like(sin_theta), sin_theta)

    s0 = np.sin((1 - frac) * theta) / safe_sin_theta
    s1 = np.sin(frac * theta) / safe_sin_theta

    s0 = np.where(near_parallel, 1 - frac, s0)
    s1 = np.where(near_parallel, frac, s1)

    result_quat = s0 * q0 + s1 * q1
    result_quat = result_quat / np.linalg.norm(result_quat, axis=-1, keepdims=True)

    return result_quat


def load_terrain_mesh(mesh_path: Path) -> trimesh.Trimesh:
    """Load terrain mesh from various formats."""
    supported_formats = ['.obj', '.stl', '.ply', '.gltf', '.glb']

    if mesh_path.suffix.lower() not in supported_formats:
        raise ValueError(f"Unsupported mesh format: {mesh_path.suffix}. "
                        f"Supported formats: {supported_formats}")

    try:
        mesh = trimesh.load(str(mesh_path))
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Loaded object is not a valid mesh: {type(mesh)}")
        return mesh
    except Exception as e:
        raise ValueError(f"Failed to load mesh from {mesh_path}: {e}")


def compute_mesh_bounding_box(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the bounding box of a mesh."""
    return mesh.bounds[0], mesh.bounds[1]  # min_point, max_point


def scale_mesh(mesh: trimesh.Trimesh, scale_factor: float) -> trimesh.Trimesh:
    """Scale a mesh by a given factor."""
    scaled_mesh = mesh.copy()
    scaled_mesh.apply_scale(scale_factor)
    return scaled_mesh


def normalize_retargeted_output_path(output_path: str) -> str:
    """
    Normalize output filename to end with "retargeted.npz".

    Examples:
        "file" -> "file_retargeted.npz"
        "file.npz" -> "file_retargeted.npz"
        "my_retargeted.npz" -> "my_retargeted.npz"
    """
    normalized = output_path
    if not normalized.endswith("retargeted.npz"):
        if normalized.endswith(".npz"):
            normalized = normalized[:-4]
        if normalized and not normalized.endswith(("_", "-", ".")):
            normalized = f"{normalized}_"
        normalized = f"{normalized}retargeted.npz"
    return normalized


def transform_mesh(mesh: trimesh.Trimesh,
                  translation: np.ndarray,
                  rotation: Optional[np.ndarray] = None) -> trimesh.Trimesh:
    """Transform a mesh with translation and optional rotation."""
    transformed_mesh = mesh.copy()

    if rotation is not None:
        # Apply rotation first
        rot_matrix = Rotation.from_quat(rotation).as_matrix()
        transformed_mesh.apply_transform(rot_matrix)

    # Apply translation
    transformed_mesh.apply_translation(translation)

    return transformed_mesh


def sample_points_on_mesh(mesh: trimesh.Trimesh, num_points: int) -> np.ndarray:
    """Sample points uniformly on the surface of a mesh."""
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points


def compute_mesh_height_at_point(mesh: trimesh.Trimesh, x: float, y: float) -> float:
    """Compute the height (z) of the mesh at a given (x, y) position."""
    # Create a ray from above the point downward
    ray_origin = np.array([x, y, 100.0])  # High z value
    ray_direction = np.array([0, 0, -1])  # Downward

    try:
        # Find intersections with the mesh using trimesh acceleration if available.
        locations, _, _ = mesh.ray.intersects_location(
            ray_origins=[ray_origin],
            ray_directions=[ray_direction]
        )
        if len(locations) > 0:
            # Return the highest intersection point (closest to the ray origin)
            return float(np.max(locations[:, 2]))
    except Exception:
        # Fall back to a dependency-free triangle walk when rtree/pyembree is unavailable.
        pass

    # Fallback: solve height against every triangle in XY projection.
    # This is slower than the ray query but avoids optional spatial index dependencies.
    triangles = np.asarray(mesh.triangles, dtype=float)
    point_xy = np.array([x, y], dtype=float)
    heights = []
    epsilon = 1e-9

    for tri in triangles:
        a_xy, b_xy, c_xy = tri[:, :2]
        v0 = b_xy - a_xy
        v1 = c_xy - a_xy
        v2 = point_xy - a_xy

        denom = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(denom) < epsilon:
            continue

        inv_denom = 1.0 / denom
        u = (v2[0] * v1[1] - v1[0] * v2[1]) * inv_denom
        v = (v0[0] * v2[1] - v2[0] * v0[1]) * inv_denom
        w = 1.0 - u - v

        if u >= -epsilon and v >= -epsilon and w >= -epsilon:
            heights.append(u * tri[1, 2] + v * tri[2, 2] + w * tri[0, 2])

    if heights:
        return float(max(heights))

    # No intersection found, return a default height.
    return 0.0


def align_terrain_to_coordinates(mesh: trimesh.Trimesh,
                               reference_points: np.ndarray) -> Tuple[trimesh.Trimesh, np.ndarray]:
    """
    Align terrain mesh to match reference coordinate system.

    Args:
        mesh: Input terrain mesh
        reference_points: Reference points defining the coordinate system

    Returns:
        Tuple of (aligned_mesh, transformation_matrix)
    """
    # Simple alignment: translate mesh so that its center matches the origin
    mesh_center = mesh.centroid
    translation = -mesh_center

    aligned_mesh = mesh.copy()
    aligned_mesh.apply_translation(translation)

    # For now, return identity transformation
    # TODO: Implement proper coordinate system alignment
    transformation = np.eye(4)
    transformation[:3, 3] = translation

    return aligned_mesh, transformation


def convert_quaternion_format(quaternions: np.ndarray,
                            input_format: str = 'wxyz',
                            output_format: str = 'xyzw') -> np.ndarray:
    """Convert between quaternion formats."""
    if input_format == output_format:
        return quaternions.copy()

    if input_format == 'wxyz' and output_format == 'xyzw':
        return quaternions[:, [1, 2, 3, 0]]
    elif input_format == 'xyzw' and output_format == 'wxyz':
        return quaternions[:, [3, 0, 1, 2]]
    else:
        raise ValueError(f"Unsupported conversion: {input_format} -> {output_format}")


def transform_points_local_to_world(quat, trans, points_local):
    """Transform points from local frame to world frame."""
    transform_matrix = trimesh.transformations.quaternion_matrix(quat)
    transform_matrix[:3, 3] = trans
    hom_points = np.hstack([points_local, np.ones((points_local.shape[0], 1))])
    transformed_points_hom = (transform_matrix @ hom_points.T).T
    return transformed_points_hom[:, :3]


def get_adjacency_list(tetrahedra, num_vertices):
    """Creates an adjacency list from the tetrahedra."""
    adj = [set() for _ in range(num_vertices)]
    for tet in tetrahedra:
        for i in range(4):
            for j in range(i + 1, 4):
                u, v = tet[i], tet[j]
                adj[u].add(v)
                adj[v].add(u)
    return [list(s) for s in adj]


def calculate_exponential_edge_weights(
    vertices: np.ndarray,
    adj_list: list[list[int]],
    kappa: float = 30.0,
):
    """
    Distance-dependent adjacency weights from TopoRetarget (arXiv:2606.16272), Eq. (5).

    For each edge (i, j): w_tilde_ij = exp(-kappa * ||v_i - v_j||), then
    row-normalized so that sum_j w_ij = 1 for every vertex i.

    Per the paper, these weights are computed once on the *source* configuration
    and reused unchanged for the robot-side Laplacian computation.

    Args:
        vertices (np.ndarray): (N, 3) array of source vertex positions.
        adj_list (list of lists): Adjacency list for the mesh.
        kappa (float): Spatial decay factor (paper uses 30 for meter-scale data).

    Returns:
        list of np.ndarray: edge_weights[i] holds the normalized weights for the
            neighbors of vertex i, aligned with the order of adj_list[i].
    """
    edge_weights = []
    for i in range(len(vertices)):
        neighbors_indices = adj_list[i]
        if len(neighbors_indices) > 0:
            neighbor_positions = vertices[neighbors_indices]
            distances = np.linalg.norm(vertices[i] - neighbor_positions, axis=1)
            weights = np.exp(-kappa * distances)
            weights = weights / np.sum(weights)
            edge_weights.append(weights)
        else:
            edge_weights.append(np.zeros(0))
    return edge_weights


def calculate_laplacian_coordinates(
    vertices: np.ndarray,
    adj_list: list[list[int]],
    epsilon: float = 1e-6,
    uniform_weight: bool = True,
    edge_weights: list[np.ndarray] | None = None,
):
    """
    Calculates the Laplacian coordinates for each vertex in the mesh.

    Args:
        vertices (np.ndarray): (N, 3) array of vertex positions.
        adj_list (list of lists): Adjacency list for the mesh.
        epsilon (float): Small value to prevent division by zero.
        uniform_weight (bool): Whether to use uniform weights.
        edge_weights (list of np.ndarray, optional): Precomputed per-edge weights
            (e.g. from calculate_exponential_edge_weights), aligned with adj_list.
            Overrides uniform_weight when provided.

    Returns:
        np.ndarray: (N, 3) array of Laplacian coordinates.
    """
    laplacian = np.zeros_like(vertices)

    for i in range(len(vertices)):
        neighbors_indices = adj_list[i]
        if len(neighbors_indices) > 0:
            vi = vertices[i]
            neighbor_positions = vertices[neighbors_indices]

            if edge_weights is not None:
                weights = edge_weights[i]
            elif uniform_weight:
                weights = np.ones_like(neighbor_positions[:, 0])
            else:
                distances = np.linalg.norm(vi - neighbor_positions, axis=1)
                weights = 1.0 / (1.5 * distances + epsilon)

            sum_of_weights = np.sum(weights)
            weighted_sum_of_neighbors = np.sum(weights[:, np.newaxis] * neighbor_positions, axis=0)
            center_of_neighbors = weighted_sum_of_neighbors / sum_of_weights
            laplacian[i] = vi - center_of_neighbors

    return laplacian


def calculate_laplacian_matrix(
    vertices: np.ndarray,
    adj_list: list[list[int]],
    epsilon: float = 1e-6,
    uniform_weight: bool = True,
    edge_weights: list[np.ndarray] | None = None,
):
    """
    Calculates the Laplacian matrix for the mesh with optional weight schemes.

    Args:
        vertices (np.ndarray): (N, 3) array of vertex positions.
        adj_list (list of lists): Adjacency list for the mesh.
        epsilon (float): Small value to prevent division by zero.
        uniform_weight (bool): If True, use uniform weights; if False, use distance-based weights.
        edge_weights (list of np.ndarray, optional): Precomputed per-edge weights
            (e.g. from calculate_exponential_edge_weights), aligned with adj_list.
            Overrides uniform_weight when provided.

    Returns:
        np.ndarray: (N, N) Laplacian matrix.
    """
    N = len(vertices)
    laplacian_matrix = np.zeros((N, N))

    for i in range(N):
        neighbors_indices = adj_list[i]
        if len(neighbors_indices) > 0:
            if edge_weights is not None:
                weights = edge_weights[i]
            elif uniform_weight:
                weights = np.ones(len(neighbors_indices)) / len(neighbors_indices)
            else:
                vi = vertices[i]
                neighbor_positions = vertices[neighbors_indices]
                distances = np.linalg.norm(vi - neighbor_positions, axis=1)
                weights = 1.0 / (distances + epsilon)
                sum_weights = np.sum(weights)
                weights = weights / sum_weights

            laplacian_matrix[i, i] = 1.0

            for j, neighbor_idx in enumerate(neighbors_indices):
                laplacian_matrix[i, neighbor_idx] = -weights[j]

    return laplacian_matrix


def estimate_body_height(
    positions: np.ndarray,
    target_names: list[str],
    *,
    head_joint: str = "Head",
    foot_joints: tuple[str, str] = ("L_Foot", "R_Foot"),
    head_top_offset: float = 0.12,
    fallback_height: float = 1.75,
    min_height: float = 1.4,
    max_height: float = 2.2,
) -> float | None:
    """Estimate human height from joint positions by finding the head-to-foot distance.

    Estimates height from the maximum head-to-foot Z-distance across all frames,
    clipped to [*min_height*, *max_height*].  If named joints are not found in
    *target_names*, returns *fallback_height*.  If *positions* is empty or
    ``None``, returns ``None``.

    Args:
        positions: Joint positions array of shape ``(T, J, 3)``.
        target_names: List of joint names corresponding to the J axis.
        head_joint: Name of the head joint in *target_names*.
        foot_joints: Names of the two foot joints in *target_names*.
        head_top_offset: Additional offset to add for the top of the head.
        fallback_height: Height to return if named joints are not found.
        min_height: Minimum valid height for clipping.
        max_height: Maximum valid height for clipping.

    Returns:
        Estimated height in meters, or ``None`` if *positions* is empty or ``None``.
    """
    if positions is None or len(positions) == 0:
        return None

    try:
        head_idx = target_names.index(head_joint)
    except ValueError:
        return fallback_height

    if head_idx >= positions.shape[1]:
        return fallback_height

    foot_indices: list[int] = []
    for fn in foot_joints:
        try:
            idx = target_names.index(fn)
            if idx < positions.shape[1]:
                foot_indices.append(idx)
        except ValueError:
            pass

    if not foot_indices:
        return fallback_height

    try:
        head_positions = positions[:, head_idx, 2]
        feet_positions = np.min(positions[:, foot_indices, 2], axis=1)
        per_frame_height = np.abs(head_positions - feet_positions) + head_top_offset
        estimated_height = float(np.max(per_frame_height))
        return float(np.clip(estimated_height, min_height, max_height))
    except (IndexError, TypeError):
        return fallback_height


def validate_robot_joint_mapping(
    robot_model,
    joint_mapping: dict,
    raise_on_missing: bool = False
) -> list:
    """
    Validate that robot links in joint_mapping exist in the robot model.
    
    This is a shared utility to avoid code duplication between OmniRetargeter
    and GenericInteractionRetargeter.
    
    Args:
        robot_model: MuJoCo model of the robot
        joint_mapping: Dictionary mapping source target names to robot link (body) names
        raise_on_missing: If True, raise ValueError when missing links are found.
                         If False, return list of missing links.
    
    Returns:
        List of missing robot link names (empty if all exist)
    
    Raises:
        ValueError: If raise_on_missing=True and missing links are found
    
    Note:
        joint_mapping maps source target names to robot BODY (link) names,
        not joint names. This function checks for body names in the URDF.
    """
    robot_bodies = set()
    for i in range(robot_model.nbody):
        body_name = mujoco.mj_id2name(robot_model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name:
            robot_bodies.add(body_name)
    
    mapped_bodies = set(
        v["robot_link"] if isinstance(v, dict) else v
        for v in joint_mapping.values()
    )
    missing_bodies = mapped_bodies - robot_bodies
    
    if missing_bodies and raise_on_missing:
        missing_list = sorted(list(missing_bodies))
        available_sample = sorted(list(robot_bodies))[:10]
        raise ValueError(
            f"The following robot links from joint_mapping were not found in URDF: {missing_list}. "
            f"Please check your joint_mapping. Available bodies (first 10): {available_sample}..."
        )
    
    return sorted(list(missing_bodies))


def create_flat_terrain(size: float = 10.0, height: float = 0.0, n_points: int = 4) -> trimesh.Trimesh:
    """
    Create a flat plane terrain mesh with minimal triangulation.
    
    Args:
        size: Side length of the square plane (meters)
        height: Z-coordinate of the plane
        n_points: Number of points per side (minimum 2)
    
    Returns:
        Trimesh object representing the flat terrain
    """
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    
    # Create grid of vertices
    x = np.linspace(-size/2, size/2, n_points)
    y = np.linspace(-size/2, size/2, n_points)
    xx, yy = np.meshgrid(x, y)
    
    vertices = np.stack([
        xx.flatten(),
        yy.flatten(),
        np.full(n_points * n_points, height)
    ], axis=1)
    
    # Create triangular faces
    faces = []
    for i in range(n_points - 1):
        for j in range(n_points - 1):
            # Two triangles per grid cell
            v0 = i * n_points + j
            v1 = i * n_points + (j + 1)
            v2 = (i + 1) * n_points + j
            v3 = (i + 1) * n_points + (j + 1)
            
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def resolve_robot_height(config: dict, model: "mujoco.MjModel", data: "mujoco.MjData") -> float:
    """Get robot height from config dict, falling back to MuJoCo geom detection.

    Checks ``robot_height`` key in *config* first; if absent, calls
    :func:`detect_robot_height` as a fallback.
    """
    height = config.get("robot_height")
    if height is not None:
        return float(height)
    return detect_robot_height(model, data)


def detect_robot_height(model: "mujoco.MjModel", data: "mujoco.MjData") -> float:
    """Detect robot height from MuJoCo geom extents (includes head casing, etc.).

    Shared by core.py, and the ``_visualize.py`` scripts to avoid code duplication.
    Prefer :func:`resolve_robot_height` at call sites that already have a config dict.
    """
    mujoco.mj_resetData(model, data)
    if model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and model.nq >= 7:
        data.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    min_z = float("inf")
    max_z = float("-inf")
    for geom_idx in range(model.ngeom):
        z = float(data.geom_xpos[geom_idx][2])
        geom_size = model.geom_size[geom_idx]
        geom_type = model.geom_type[geom_idx]
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = float(geom_size[0])
            min_z = min(min_z, z - r); max_z = max(max_z, z + r)
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            r = float(geom_size[0]); hh = float(geom_size[1])
            min_z = min(min_z, z - hh - r); max_z = max(max_z, z + hh + r)
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            hs = float(geom_size[2])
            min_z = min(min_z, z - hs); max_z = max(max_z, z + hs)
        else:
            min_z = min(min_z, z); max_z = max(max_z, z)
    height = float(max_z - min_z)
    if height < 0.3 or height > 3.0:
        return 1.6
    return height


import re


def _strip_xml_comments(xml_str: str) -> str:
    """Remove XML comments from a string."""
    return re.sub(r"<!--.*?-->", "", xml_str, flags=re.DOTALL)


def _has_floating_joint(xml_str: str) -> bool:
    """Return True if the URDF XML already contains a floating joint."""
    return 'type="floating"' in _strip_xml_comments(xml_str)


def _find_urdf_root_body(xml_str: str) -> str:
    """Find the root body name in a URDF XML string.

    The root body is the link that is never a ``<child>`` of any joint.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_str)
    links: set[str] = set()
    children: set[str] = set()

    for link in root.findall("link"):
        name = link.get("name")
        if name:
            links.add(name)

    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is not None:
            child_name = child.get("link")
            if child_name:
                children.add(child_name)

    root_candidates = links - children
    root_candidates.discard("world")

    if not root_candidates:
        for link in root.findall("link"):
            name = link.get("name")
            if name and name != "world":
                return name
        raise ValueError("No root body found in URDF — cannot inject floating joint.")

    return sorted(root_candidates)[0]


def _inject_floating_joint(xml_str: str) -> str:
    """Inject a world link and floating joint into a URDF XML string."""
    import xml.etree.ElementTree as ET

    root_body = _find_urdf_root_body(xml_str)
    root = ET.fromstring(xml_str)

    has_world = any(
        link.get("name") == "world" for link in root.findall("link")
    )

    if not has_world:
        root.insert(0, ET.Element("link", name="world"))

    floating_joint = ET.Element("joint", name="floating_base", type="floating")
    ET.SubElement(floating_joint, "parent", link="world")
    ET.SubElement(floating_joint, "child", link=root_body)

    world_idx = None
    for i, elem in enumerate(root):
        if elem.tag == "link" and elem.get("name") == "world":
            world_idx = i
            break

    if world_idx is not None:
        root.insert(world_idx + 1, floating_joint)
    else:
        root.insert(0, floating_joint)

    return ET.tostring(root, encoding="unicode")


def load_robot_urdf_with_floating_base(urdf_path: str | Path) -> "mujoco.MjModel":
    """Load a URDF into a MuJoCo model, auto-injecting a floating joint if missing.

    MuJoCo treats a URDF's base link as fixed to the world unless there is a
    ``<joint type="floating">`` connecting a world reference to the base.
    This function detects when a floating joint is absent and injects one
    automatically, so callers always receive a free-floating model.

    Args:
        urdf_path: Path to the URDF file. Relative mesh paths inside the URDF
                   are resolved relative to the URDF directory.

    Returns:
        A MuJoCo ``MjModel`` that is guaranteed to have a floating base joint
        as the first joint.
    """
    import os

    urdf_path = Path(urdf_path)
    xml_str = urdf_path.read_text(encoding="utf-8")

    if not _has_floating_joint(xml_str):
        xml_str = _inject_floating_joint(xml_str)

    old_cwd = os.getcwd()
    try:
        os.chdir(str(urdf_path.parent))
        return mujoco.MjModel.from_xml_string(xml_str)
    finally:
        os.chdir(old_cwd)
