"""Utility functions for OmniRetargeting."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import smplx
import torch
import trimesh
from scipy.spatial.transform import Rotation
import os
import re
import mujoco
import xml.etree.ElementTree as ET


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


def sample_points_on_mesh(
    mesh: trimesh.Trimesh, num_points: int, seed: Optional[int] = None
) -> np.ndarray:
    """Sample points uniformly on the surface of a mesh.

    Passing ``seed`` makes the draw reproducible across processes, which is a
    prerequisite for comparing two retargeting runs against each other.
    """
    points, _ = trimesh.sample.sample_surface(mesh, num_points, seed=seed)
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
    """Validate SMPLX trajectory format."""
    if len(trajectory.shape) != 3:
        return False

    num_frames, num_joints, num_coords = trajectory.shape

    if num_coords != 3:
        return False

    if num_frames == 0 or num_joints == 0:
        return False

    # Check for NaN or infinite values
    if not np.isfinite(trajectory).all():
        return False

    return True


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


def compute_world_joint_orientations(
    global_orient: np.ndarray,
    full_pose: np.ndarray,
    parents: np.ndarray,
    num_body_joints: int = 22,
) -> np.ndarray:
    """
    Compute world-frame joint orientations from SMPLX pose parameters.
    
    Args:
        global_orient: Root orientation in axis-angle format, shape (T, 3)
        full_pose: Full pose parameters in axis-angle format, shape (T, J_total, 3)
        parents: Parent indices for kinematic tree, shape (J_total,)
        num_body_joints: Number of body joints to return (default 22)
    
    Returns:
        Joint orientations as quaternions (wxyz format), shape (T, J, 4)
    """
    num_frames = global_orient.shape[0]
    num_joints = min(full_pose.shape[1], num_body_joints)
    
    # Output: quaternions in wxyz format
    joint_orientations = np.zeros((num_frames, num_joints, 4))
    
    for frame_idx in range(num_frames):
        # Store rotations for this frame
        frame_rotations = []
        
        for joint_idx in range(num_joints):
            if joint_idx == 0:
                # Root joint: use global_orient directly
                rot = Rotation.from_rotvec(global_orient[frame_idx])
            else:
                # Other joints: multiply parent's world rotation by local rotation
                parent_idx = parents[joint_idx]
                if parent_idx >= 0 and parent_idx < len(frame_rotations):
                    parent_rot = frame_rotations[parent_idx]
                    local_rot = Rotation.from_rotvec(full_pose[frame_idx, joint_idx])
                    rot = parent_rot * local_rot
                else:
                    # Fallback: use local rotation as world rotation
                    rot = Rotation.from_rotvec(full_pose[frame_idx, joint_idx])
            
            frame_rotations.append(rot)
            # Store as quaternion in wxyz format (scalar_first=True)
            joint_orientations[frame_idx, joint_idx] = rot.as_quat(scalar_first=True)
    
    return joint_orientations


def _farthest_point_sample_indices(points: np.ndarray, max_samples: int) -> np.ndarray:
    """Select deterministic, spatially distributed points from a template part."""
    points = np.asarray(points, dtype=float)
    if len(points) <= max_samples:
        return np.arange(len(points), dtype=int)
    max_samples = max(int(max_samples), 1)
    center = np.mean(points, axis=0)
    selected = [int(np.argmax(np.linalg.norm(points - center, axis=1)))]
    min_squared_distance = np.sum((points - points[selected[0]]) ** 2, axis=1)
    for _ in range(1, max_samples):
        next_idx = int(np.argmax(min_squared_distance))
        selected.append(next_idx)
        squared_distance = np.sum((points - points[next_idx]) ** 2, axis=1)
        min_squared_distance = np.minimum(min_squared_distance, squared_distance)
    return np.asarray(selected, dtype=int)


def _extract_smplx_end_effector_markers(
    body_model,
    smplx_output,
    max_samples: int = 32,
    world_joint_orientations: Optional[np.ndarray] = None,
    shape_betas: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract coherent hand/foot samples and rigid semantic foot frames."""
    vertices = smplx_output.vertices.detach().cpu().numpy()
    weights = body_model.lbs_weights.detach().cpu().numpy()
    template = body_model.v_template.detach().cpu().numpy()
    full_joints = smplx_output.joints.detach().cpu().numpy()
    shaped_template = template
    if shape_betas is not None:
        beta_values = np.asarray(shape_betas, dtype=float).reshape(-1)
        shapedirs = body_model.shapedirs.detach().cpu().numpy()
        num_shapes = min(len(beta_values), shapedirs.shape[2])
        shaped_template = template + np.einsum(
            "vcn,n->vc",
            shapedirs[:, :, :num_shapes],
            beta_values[:num_shapes],
        )
    joint_regressor = body_model.J_regressor.detach().cpu().numpy()
    template_joints = joint_regressor @ shaped_template

    # SMPL-X LBS joint ids: feet 10/11, left fingers 25:40, right
    # fingers 40:55.  Excluding wrist/ankle weights avoids pulling forearm and
    # shin vertices into the contact surface sets.
    part_joint_ids = {
        "left_hand": list(range(25, 40)),
        "right_hand": list(range(40, 55)),
        "left_foot": [10],
        "right_foot": [11],
    }
    fallback_joint_ids = {
        "left_hand": 20,
        "right_hand": 21,
        "left_foot": 10,
        "right_foot": 11,
    }

    markers: Dict[str, Dict[str, np.ndarray]] = {}
    for name, joint_ids in part_joint_ids.items():
        influence = np.sum(weights[:, joint_ids], axis=1)
        vertex_ids = np.flatnonzero(influence > 0.25)
        if len(vertex_ids) == 0:
            tracking = smplx_output.joints[:, fallback_joint_ids[name], :].detach().cpu().numpy()
            markers[name] = {
                "tracking_positions": tracking,
                "contact_samples": tracking[:, None, :],
            }
            continue

        local_ids = _farthest_point_sample_indices(
            template[vertex_ids],
            max_samples=max_samples,
        )
        sampled_vertex_ids = vertex_ids[local_ids]
        part_vertices = vertices[:, vertex_ids, :]
        markers[name] = {
            "tracking_positions": np.mean(part_vertices, axis=1),
            "contact_samples": vertices[:, sampled_vertex_ids, :],
            "vertex_ids": sampled_vertex_ids,
        }
        if (
            name in ("left_hand", "right_hand")
            and world_joint_orientations is not None
        ):
            # Build a rigid palm frame from the shaped neutral template, then
            # carry it with the SMPL-X wrist rotation.  Position-only hand
            # tracking lets the optimizer rotate the G1 wrist to use the long
            # palm offset as a reach extension, which is accurate at one point
            # but produces visibly twisted hands.  The wrist-to-finger-base
            # direction is the hand's local +x; index-to-pinky supplies a
            # stable in-palm lateral axis and fixes the palm-normal sign.
            if name == "left_hand":
                wrist_idx = 20
                finger_base_ids = [25, 28, 31, 34]
                index_idx, pinky_idx = 25, 31
            else:
                wrist_idx = 21
                finger_base_ids = [40, 43, 46, 49]
                index_idx, pinky_idx = 40, 46

            template_forward = (
                np.mean(template_joints[finger_base_ids], axis=0)
                - template_joints[wrist_idx]
            )
            template_forward /= max(
                float(np.linalg.norm(template_forward)), 1e-10
            )
            template_lateral = (
                template_joints[index_idx] - template_joints[pinky_idx]
            )
            template_lateral -= template_forward * float(
                np.dot(template_forward, template_lateral)
            )
            template_lateral /= max(
                float(np.linalg.norm(template_lateral)), 1e-10
            )
            template_up = np.cross(template_forward, template_lateral)
            template_up /= max(float(np.linalg.norm(template_up)), 1e-10)

            rotations = Rotation.from_quat(
                np.asarray(world_joint_orientations)[:, wrist_idx],
                scalar_first=True,
            ).as_matrix()
            markers[name]["forward_directions"] = np.einsum(
                "tij,j->ti", rotations, template_forward
            )
            markers[name]["up_directions"] = np.einsum(
                "tij,j->ti", rotations, template_up
            )
        if name in ("left_foot", "right_foot"):
            ankle_idx = 7 if name == "left_foot" else 8
            foot_idx = 10 if name == "left_foot" else 11

            # Calibrate one rigid, right-handed foot frame on the shaped
            # neutral template.  Per-frame PCA has no persistent eigenvector
            # identity and can turn skin deformation into apparent foot
            # rotation.  The template's thinnest axis is the sole normal; its
            # sign is fixed toward the ankle.  Ankle-to-foot gives the
            # semantic toe direction, projected into the sole plane because
            # rotation-matrix axes must be orthogonal.
            template_part = shaped_template[vertex_ids]
            centered_template = template_part - np.mean(template_part, axis=0)
            covariance = centered_template.T @ centered_template / max(
                len(template_part), 1
            )
            _, eigenvectors = np.linalg.eigh(covariance)
            template_up = eigenvectors[:, 0]
            toward_ankle = (
                template_joints[ankle_idx] - np.mean(template_part, axis=0)
            )
            if float(np.dot(template_up, toward_ankle)) < 0.0:
                template_up *= -1.0
            template_up /= max(float(np.linalg.norm(template_up)), 1e-10)

            template_forward = (
                template_joints[foot_idx] - template_joints[ankle_idx]
            )
            template_forward -= template_up * float(
                np.dot(template_up, template_forward)
            )
            template_forward /= max(
                float(np.linalg.norm(template_forward)), 1e-10
            )
            template_lateral = np.cross(template_up, template_forward)
            template_lateral /= max(
                float(np.linalg.norm(template_lateral)), 1e-10
            )
            template_forward = np.cross(template_lateral, template_up)

            if world_joint_orientations is not None:
                rotations = Rotation.from_quat(
                    np.asarray(world_joint_orientations)[:, foot_idx],
                    scalar_first=True,
                ).as_matrix()
                markers[name]["forward_directions"] = np.einsum(
                    "tij,j->ti", rotations, template_forward
                )
                markers[name]["up_directions"] = np.einsum(
                    "tij,j->ti", rotations, template_up
                )
            else:
                # This path is retained for direct helper use.  Raw SMPL-X
                # loading supplies joint rotations and therefore uses the
                # rigid-frame path above.
                raw_forward = (
                    full_joints[:, foot_idx] - full_joints[:, ankle_idx]
                )
                raw_forward /= np.maximum(
                    np.linalg.norm(raw_forward, axis=1, keepdims=True), 1e-10
                )
                markers[name]["forward_directions"] = raw_forward
    return markers


def _fallback_end_effector_markers(
    joints: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Build point markers when the source file has no SMPL-X body surface."""
    joints = np.asarray(joints, dtype=float)
    if joints.ndim != 3 or joints.shape[-1] != 3:
        raise ValueError(f"Expected joint trajectory with shape (T, J, 3), got {joints.shape}.")
    indices = {
        "left_hand": 20,
        "right_hand": 21,
        "left_foot": 10 if joints.shape[1] > 10 else 7,
        "right_foot": 11 if joints.shape[1] > 11 else 8,
    }
    markers: Dict[str, Dict[str, np.ndarray]] = {}
    for name, joint_idx in indices.items():
        if joint_idx >= joints.shape[1]:
            continue
        positions = joints[:, joint_idx, :]
        markers[name] = {
            "tracking_positions": positions,
            "contact_samples": positions[:, None, :],
        }
    return markers


def _interpolate_rotvecs(
    start: np.ndarray,
    end: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Interpolate independent axis-angle rotations along their shortest arcs."""
    start = np.asarray(start)
    end = np.asarray(end)
    original_shape = start.shape
    start_rot = Rotation.from_rotvec(start.reshape(-1, 3))
    end_rot = Rotation.from_rotvec(end.reshape(-1, 3))
    delta = start_rot.inv() * end_rot
    interpolated = start_rot * Rotation.from_rotvec(delta.as_rotvec() * fraction)
    return interpolated.as_rotvec().reshape(original_shape)


def _densify_smplx_frame_gaps(
    frame_ids: np.ndarray,
    trans: np.ndarray,
    root_orient: np.ndarray,
    pose_body: np.ndarray,
    pose_hand: Optional[np.ndarray] = None,
    *,
    max_gap: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Restore short missing source frames before SMPL-X forward kinematics.

    Positions use linear interpolation and every axis-angle joint uses shortest-
    arc rotation interpolation.  This keeps a missing video frame from being
    played as a double-speed motion sample while leaving complete timelines
    untouched.  Large/non-numeric/disordered gaps are rejected rather than
    guessing across an edit or a different frame-id convention.
    """
    ids = np.asarray(frame_ids)
    if ids.ndim != 1 or len(ids) != len(trans):
        return trans, root_orient, pose_body, pose_hand, ids
    try:
        numeric_ids = np.asarray([int(str(value)) for value in ids], dtype=np.int64)
    except (TypeError, ValueError):
        return trans, root_orient, pose_body, pose_hand, ids

    gaps = np.diff(numeric_ids)
    if len(gaps) == 0 or np.all(gaps == 1):
        return trans, root_orient, pose_body, pose_hand, ids
    if np.any(gaps < 1) or np.any(gaps > max_gap):
        return trans, root_orient, pose_body, pose_hand, ids

    dense_trans = []
    dense_root = []
    dense_body = []
    dense_hand = [] if pose_hand is not None else None
    dense_ids = []
    id_width = max(len(str(value)) for value in ids)

    for frame_idx, gap in enumerate(gaps):
        dense_trans.append(np.asarray(trans[frame_idx]))
        dense_root.append(np.asarray(root_orient[frame_idx]))
        dense_body.append(np.asarray(pose_body[frame_idx]))
        if dense_hand is not None:
            dense_hand.append(np.asarray(pose_hand[frame_idx]))
        dense_ids.append(str(ids[frame_idx]))

        for offset in range(1, int(gap)):
            fraction = offset / float(gap)
            dense_trans.append(
                (1.0 - fraction) * trans[frame_idx]
                + fraction * trans[frame_idx + 1]
            )
            dense_root.append(
                _interpolate_rotvecs(
                    root_orient[frame_idx], root_orient[frame_idx + 1], fraction
                )
            )
            dense_body.append(
                _interpolate_rotvecs(
                    np.asarray(pose_body[frame_idx]).reshape(-1, 3),
                    np.asarray(pose_body[frame_idx + 1]).reshape(-1, 3),
                    fraction,
                ).reshape(np.asarray(pose_body[frame_idx]).shape)
            )
            if dense_hand is not None:
                dense_hand.append(
                    _interpolate_rotvecs(
                        np.asarray(pose_hand[frame_idx]).reshape(-1, 3),
                        np.asarray(pose_hand[frame_idx + 1]).reshape(-1, 3),
                        fraction,
                    ).reshape(np.asarray(pose_hand[frame_idx]).shape)
                )
            dense_ids.append(str(numeric_ids[frame_idx] + offset).zfill(id_width))

    dense_trans.append(np.asarray(trans[-1]))
    dense_root.append(np.asarray(root_orient[-1]))
    dense_body.append(np.asarray(pose_body[-1]))
    if dense_hand is not None:
        dense_hand.append(np.asarray(pose_hand[-1]))
    dense_ids.append(str(ids[-1]))

    return (
        np.asarray(dense_trans, dtype=np.asarray(trans).dtype),
        np.asarray(dense_root, dtype=np.asarray(root_orient).dtype),
        np.asarray(dense_body, dtype=np.asarray(pose_body).dtype),
        None
        if dense_hand is None
        else np.asarray(dense_hand, dtype=np.asarray(pose_hand).dtype),
        np.asarray(dense_ids, dtype=ids.dtype),
    )


def load_smplx_trajectory(
    smplx_file: Path,
    smplx_model_directory: Optional[str] = None,
    gender: str = "neutral",
    return_meta: bool = False,
    return_end_effectors: bool = False,
    end_effector_samples: int = 32,
) -> tuple:
    """
    Load SMPLX trajectory from file.
    
    Handles:
    - Standard .npy files with pre-computed joint positions
    - Pre-processed .npz files with 'global_joint_positions' key
    - Raw SMPLX-NG .npz files (e.g., stageii.npz) with SMPLX parameters
    
    For raw SMPLX-NG files, requires smplx_model_directory to perform forward kinematics.
    
    Args:
        smplx_file: Path to trajectory file (.npy or .npz)
        smplx_model_directory: Path to SMPLX model files (required for raw SMPLX-NG files and orientation computation)
        gender: Gender for SMPLX model ('neutral', 'male', 'female')
        return_meta: If True, also return root orientation and translation.
        return_end_effectors: If True, return semantic hand/foot centers and
            compact SMPL-X surface samples for scene-contact inference.
        end_effector_samples: Maximum number of coherent surface samples per
            end effector.
    
    Returns:
        Tuple beginning with ``(positions, orientations)``.  ``return_meta``
        appends ``(root_orient, trans)`` and ``return_end_effectors`` appends a
        marker dictionary.
        
        If return_meta=False (default):
            - positions: Joint positions array of shape (T, J, 3)
            - orientations: Joint orientations as quaternions (wxyz), shape (T, J, 4)
                           Returns None if orientations cannot be computed (e.g., .npy files)
        
        If return_meta=True:
            - positions: Joint positions array of shape (T, J, 3)
            - orientations: Joint orientations as quaternions (wxyz), shape (T, J, 4)
                           Returns None if orientations cannot be computed
            - root_orient: Root orientation parameters in axis-angle format, shape (T, 3) or None
            - trans: Translation parameters, shape (T, 3) or None
    """
    def pack_result(joints, orientations, root_orient, trans, end_effectors):
        result = [joints, orientations]
        if return_meta:
            result.extend([root_orient, trans])
        if return_end_effectors:
            result.append(end_effectors)
        return tuple(result)

    if smplx_file.suffix == ".npy":
        joints = np.load(smplx_file, allow_pickle=True)
        # Cannot compute orientations from positions only
        print("Warning: Cannot compute orientations from .npy file (positions only). Returning None for orientations.")
        end_effectors = _fallback_end_effector_markers(joints) if return_end_effectors else None
        return pack_result(joints, None, None, None, end_effectors)

    smplx_data = np.load(smplx_file, allow_pickle=True)

    if isinstance(smplx_data, np.lib.npyio.NpzFile) and "human_joints" in smplx_data:
        joints = smplx_data["human_joints"]
        print("Warning: Cannot compute orientations from OMOMO human_joints. Returning None for orientations.")
        end_effectors = _fallback_end_effector_markers(joints) if return_end_effectors else None
        return pack_result(joints, None, None, None, end_effectors)

    if isinstance(smplx_data, np.lib.npyio.NpzFile) and "global_joint_positions" in smplx_data:
        joints = smplx_data["global_joint_positions"]
        root_orient = smplx_data["root_orient"] if "root_orient" in smplx_data else None
        trans = smplx_data["trans"] if "trans" in smplx_data else None
        
        # Try to compute orientations if we have the necessary data
        orientations = None
        if "full_pose" in smplx_data and smplx_model_directory is not None and root_orient is not None:
            # Load body model to get parent structure
            body_model = smplx.create(
                smplx_model_directory,
                "smplx",
                gender=gender,
                use_pca=False,
            )
            full_pose = smplx_data["full_pose"]
            if isinstance(full_pose, np.ndarray) and len(full_pose.shape) == 2:
                # Reshape from (T, J*3) to (T, J, 3)
                full_pose = full_pose.reshape(full_pose.shape[0], -1, 3)
            
            orientations = compute_world_joint_orientations(
                root_orient,
                full_pose,
                body_model.parents.cpu().numpy(),
                num_body_joints=22,
            )
        else:
            print("Warning: Cannot compute orientations from .npz file (missing full_pose, root_orient, or model directory). Returning None for orientations.")
        
        end_effectors = _fallback_end_effector_markers(joints) if return_end_effectors else None
        return pack_result(joints, orientations, root_orient, trans, end_effectors)

    # Raw SMPLX-NG file - need to run forward kinematics
    body_model = smplx.create(
        smplx_model_directory,
        "smplx",
        gender=str(smplx_data.get("gender", "neutral")),
        use_pca=False,
    )

    # Extract and adjust betas
    betas = torch.tensor(smplx_data["betas"]).float().view(1, -1)
    if betas.shape[1] > 10:
        betas = betas[:, :10]

    pose_body = smplx_data["pose_body"]
    root_orient = smplx_data["root_orient"]
    trans = smplx_data["trans"]
    pose_hand = smplx_data["pose_hand"] if "pose_hand" in smplx_data else None
    if "frame_ids" in smplx_data:
        original_num_frames = len(trans)
        trans, root_orient, pose_body, pose_hand, _ = _densify_smplx_frame_gaps(
            smplx_data["frame_ids"],
            trans,
            root_orient,
            pose_body,
            pose_hand,
        )
        if len(trans) != original_num_frames:
            print(
                "Restored "
                f"{len(trans) - original_num_frames} missing SMPL-X timeline frame(s) "
                "from frame_ids."
            )
    num_frames = pose_body.shape[0]
    if pose_hand is not None and pose_hand.ndim == 2 and pose_hand.shape[1] >= 90:
        left_hand_pose = torch.tensor(pose_hand[:, :45]).float()
        right_hand_pose = torch.tensor(pose_hand[:, 45:90]).float()
    else:
        left_hand_pose = torch.zeros(num_frames, 45).float()
        right_hand_pose = torch.zeros(num_frames, 45).float()
    smplx_output = body_model(
        betas=betas,  # Shape parameters (1, 10)
        global_orient=torch.tensor(root_orient).float(),  # (N, 3)
        body_pose=torch.tensor(pose_body).float(),  # (N, 63)
        transl=torch.tensor(trans).float(),  # (N, 3)
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        jaw_pose=torch.zeros(num_frames, 3).float(),  # Pose parameters per frame
        leye_pose=torch.zeros(num_frames, 3).float(),  # Pose parameters per frame
        reye_pose=torch.zeros(num_frames, 3).float(),  # Pose parameters per frame
        expression=torch.zeros(num_frames, 10).float(),  # Expression parameters expanded to num_frames
        return_full_pose=True,
    )

    # Extract joints and their world rotations before building rigid semantic
    # foot frames.
    full_joints = smplx_output.joints.detach().cpu().numpy()
    full_pose = smplx_output.full_pose.detach().cpu().numpy()
    # Reshape from (T, J*3) to (T, J, 3)
    full_pose = full_pose.reshape(num_frames, -1, 3)
    
    orientations = compute_world_joint_orientations(
        root_orient,
        full_pose,
        body_model.parents.cpu().numpy(),
        num_body_joints=22,
    )
    end_effectors = None
    if return_end_effectors:
        end_effectors = _extract_smplx_end_effector_markers(
            body_model,
            smplx_output,
            max_samples=end_effector_samples,
            world_joint_orientations=orientations,
            shape_betas=betas.detach().cpu().numpy(),
        )

    # Return only body joints (first 22)
    joints = full_joints[:, :22, :]
    
    return pack_result(joints, orientations, root_orient, trans, end_effectors)


# Interpolation, body-height estimation and floating-base URDF helpers used
# by the data_sources adapters. Kept here so multi-source loading works
# alongside the contact-priority solver.
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


# SMPL-X end-effector marker sampling and frame densification. Ported from
# the contact-priority line; consumed by contact inference and by callers that
# need hand/foot surface samples rather than joint positions alone.
