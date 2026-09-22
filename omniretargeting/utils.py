"""Utility functions for OmniRetargeting."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Tuple

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .data_sources.smplx_motion import (
    _densify_smplx_frame_gaps as _densify_smplx_frame_gaps,
    _extract_smplx_end_effector_markers as _extract_smplx_end_effector_markers,
    _fallback_end_effector_markers as _fallback_end_effector_markers,
    _farthest_point_sample_indices as _farthest_point_sample_indices,
    _interpolate_rotvecs as _interpolate_rotvecs,
    compute_world_joint_orientations as compute_world_joint_orientations,
    load_smplx_trajectory as load_smplx_trajectory,
)
from .interaction_graph import (
    calculate_exponential_edge_weights as calculate_exponential_edge_weights,
    calculate_laplacian_coordinates as calculate_laplacian_coordinates,
    calculate_laplacian_matrix as calculate_laplacian_matrix,
    calculate_source_laplacian as calculate_source_laplacian,
    get_adjacency_list as get_adjacency_list,
)
from .motion_interpolation import (
    _quat_to_rotvec as _quat_to_rotvec,
    _rotvec_to_quat as _rotvec_to_quat,
    linear_interpolate as linear_interpolate,
    slerp_interpolate as slerp_interpolate,
)


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


def _has_floating_joint(xml_str: str) -> bool:
    """Recognize actual URDF joints regardless of XML quoting or whitespace."""
    return any(joint.get("type") == "floating" for joint in ET.fromstring(xml_str).findall("joint"))


def _find_urdf_root_body(xml_str: str) -> str:
    """Find the root body name in a URDF XML string.

    The root body is the link that is never a ``<child>`` of any joint.
    """

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
