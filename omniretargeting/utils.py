"""Utility functions for OmniRetargeting."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import trimesh
from scipy.spatial.transform import Rotation


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


def validate_smplx_trajectory(trajectory: np.ndarray) -> bool:
    """Validate legacy SMPL-X trajectory format through the neutral motion validator."""
    from omniretargeting.data_sources.base import validate_motion_positions

    return validate_motion_positions(trajectory)


def extract_smplx_joint_positions(trajectory: np.ndarray,
                                joint_indices: list) -> np.ndarray:
    """Extract specific joint positions from SMPLX trajectory."""
    return trajectory[:, joint_indices, :]


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


def calculate_laplacian_coordinates(vertices, adj_list, epsilon=1e-6, uniform_weight=True):
    """
    Calculates the Laplacian coordinates for each vertex in the mesh.

    Args:
        vertices (np.ndarray): (N, 3) array of vertex positions.
        adj_list (list of lists): Adjacency list for the mesh.
        epsilon (float): Small value to prevent division by zero.
        uniform_weight (bool): Whether to use uniform weights.

    Returns:
        np.ndarray: (N, 3) array of Laplacian coordinates.
    """
    laplacian = np.zeros_like(vertices)

    for i in range(len(vertices)):
        neighbors_indices = adj_list[i]
        if len(neighbors_indices) > 0:
            vi = vertices[i]
            neighbor_positions = vertices[neighbors_indices]
            distances = np.linalg.norm(vi - neighbor_positions, axis=1)

            if uniform_weight:
                weights = np.ones_like(distances)
            else:
                weights = 1.0 / (1.5 * distances + epsilon)

            sum_of_weights = np.sum(weights)
            weighted_sum_of_neighbors = np.sum(weights[:, np.newaxis] * neighbor_positions, axis=0)
            center_of_neighbors = weighted_sum_of_neighbors / sum_of_weights
            laplacian[i] = vi - center_of_neighbors

    return laplacian


def calculate_laplacian_matrix(vertices, adj_list, epsilon=1e-6, uniform_weight=True):
    """
    Calculates the Laplacian matrix for the mesh with optional weight schemes.

    Args:
        vertices (np.ndarray): (N, 3) array of vertex positions.
        adj_list (list of lists): Adjacency list for the mesh.
        epsilon (float): Small value to prevent division by zero.
        uniform_weight (bool): If True, use uniform weights; if False, use distance-based weights.

    Returns:
        np.ndarray: (N, N) Laplacian matrix.
    """
    N = len(vertices)
    laplacian_matrix = np.zeros((N, N))

    for i in range(N):
        neighbors_indices = adj_list[i]
        if len(neighbors_indices) > 0:
            if uniform_weight:
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


def compute_world_joint_orientations(*args, **kwargs):
    from omniretargeting.data_sources.smplx import compute_world_joint_orientations as _impl

    return _impl(*args, **kwargs)


def load_smplx_trajectory(*args, **kwargs):
    from omniretargeting.data_sources.smplx import load_smplx_trajectory as _impl



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
    import mujoco
    
    robot_bodies = set()
    for i in range(robot_model.nbody):
        body_name = mujoco.mj_id2name(robot_model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name:
            robot_bodies.add(body_name)
    
    mapped_bodies = set(joint_mapping.values())
    missing_bodies = mapped_bodies - robot_bodies
    
    if missing_bodies and raise_on_missing:
        missing_list = sorted(list(missing_bodies))
        available_sample = sorted(list(robot_bodies))[:10]
        raise ValueError(
            f"The following robot links from joint_mapping were not found in URDF: {missing_list}. "
            f"Please check your joint_mapping. Available bodies (first 10): {available_sample}..."
        )
    
    return sorted(list(missing_bodies))
