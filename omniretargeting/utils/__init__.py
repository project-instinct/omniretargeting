"""Utility subpackage for OmniRetargeting.

Only the numerical/mesh helpers in :mod:`omniretargeting.utils.math` are
re-exported here. Batch orchestration helpers live in
:mod:`omniretargeting.utils.batch_processing` and must be imported from
there explicitly.
"""

from omniretargeting.utils.math import (
    align_terrain_to_coordinates,
    calculate_exponential_edge_weights,
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
    compute_mesh_bounding_box,
    compute_mesh_height_at_point,
    convert_quaternion_format,
    create_flat_terrain,
    detect_robot_height,
    estimate_body_height,
    get_adjacency_list,
    linear_interpolate,
    load_robot_urdf_with_floating_base,
    load_terrain_mesh,
    normalize_retargeted_output_path,
    resolve_robot_height,
    sample_mujoco_geom_local_points,
    sample_points_on_mesh,
    scale_mesh,
    slerp_interpolate,
    transform_mesh,
    transform_points_local_to_world,
    validate_robot_joint_mapping,
)
from omniretargeting.utils.math import _has_floating_joint, _inject_floating_joint

__all__ = [
    "align_terrain_to_coordinates",
    "calculate_exponential_edge_weights",
    "calculate_laplacian_coordinates",
    "calculate_laplacian_matrix",
    "compute_mesh_bounding_box",
    "compute_mesh_height_at_point",
    "convert_quaternion_format",
    "create_flat_terrain",
    "detect_robot_height",
    "estimate_body_height",
    "get_adjacency_list",
    "linear_interpolate",
    "load_robot_urdf_with_floating_base",
    "load_terrain_mesh",
    "normalize_retargeted_output_path",
    "resolve_robot_height",
    "sample_mujoco_geom_local_points",
    "sample_points_on_mesh",
    "scale_mesh",
    "slerp_interpolate",
    "transform_mesh",
    "transform_points_local_to_world",
    "validate_robot_joint_mapping",
]
