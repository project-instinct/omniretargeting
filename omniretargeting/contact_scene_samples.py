"""Scene queries of source surface samples for contact inference.

Every coherent source surface sample is queried against the scene mesh. The
closest sample keeps a reliable scene normal, and contact frames can use a
robust nearby surface patch instead of one jumping vertex.
"""

from __future__ import annotations

import numpy as np
import trimesh

from .contact_scene_normals import (
    _DEFAULT_NORMAL_ORIENTATION_DEADZONE,
    enclosed_free_space_normals,
)


def _query_scene_samples(
    scene_mesh: trimesh.Trimesh,
    contact_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Query every coherent source surface sample against the scene mesh."""
    samples = np.asarray(contact_samples, dtype=float)
    if samples.ndim == 2:
        samples = samples[:, None, :]
    if samples.ndim != 3 or samples.shape[-1] != 3:
        raise ValueError(
            "contact_samples must have shape (T, K, 3) or (T, 3), "
            f"got {samples.shape}."
        )

    num_frames, num_samples, _ = samples.shape
    flat_samples = samples.reshape(-1, 3)
    try:
        closest, flat_distances, triangle_ids = trimesh.proximity.closest_point(
            scene_mesh,
            flat_samples,
        )
    except (ImportError, ModuleNotFoundError):
        closest, flat_distances, triangle_ids = trimesh.proximity.closest_point_naive(
            scene_mesh,
            flat_samples,
        )

    return (
        samples,
        np.asarray(closest, dtype=float).reshape(num_frames, num_samples, 3),
        np.asarray(flat_distances, dtype=float).reshape(num_frames, num_samples),
        np.asarray(triangle_ids, dtype=int).reshape(num_frames, num_samples),
    )


def _select_closest_scene_samples(
    scene_mesh: trimesh.Trimesh,
    samples: np.ndarray,
    closest: np.ndarray,
    distances: np.ndarray,
    triangle_ids: np.ndarray,
    *,
    normal_orientation_deadzone: float = _DEFAULT_NORMAL_ORIENTATION_DEADZONE,
    enclosed_free_space: bool = False,
    sample_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select the closest sample while retaining a reliable scene normal."""
    num_frames = len(samples)
    sample_ids = np.argmin(distances, axis=1) if sample_ids is None else sample_ids
    frame_ids = np.arange(num_frames)

    selected_samples = samples[frame_ids, sample_ids]
    surface_points = closest[frame_ids, sample_ids]
    selected_distances = distances[frame_ids, sample_ids]
    selected_triangles = triangle_ids[frame_ids, sample_ids]
    if enclosed_free_space:
        return (selected_samples, surface_points,
                enclosed_free_space_normals(scene_mesh, selected_triangles), selected_distances)

    normals = np.zeros((num_frames, 3), dtype=float)
    valid = (selected_triangles >= 0) & (
        selected_triangles < len(scene_mesh.face_normals)
    )
    normals[valid] = np.asarray(scene_mesh.face_normals, dtype=float)[
        selected_triangles[valid]
    ]
    missing = np.linalg.norm(normals, axis=1) < 1e-10
    fallback_normals = missing.copy()
    normals[missing] = selected_samples[missing] - surface_points[missing]
    # Exact on-surface queries can have both an invalid triangle id and a zero
    # point-to-surface offset.  Fall back to the nearest mesh vertex normal so
    # a valid contact never silently loses its orientation constraint.
    missing = np.linalg.norm(normals, axis=1) < 1e-10
    if np.any(missing) and len(scene_mesh.vertices):
        _, vertex_ids = scene_mesh.kdtree.query(selected_samples[missing])
        normals[missing] = np.asarray(scene_mesh.vertex_normals, dtype=float)[
            np.asarray(vertex_ids, dtype=int)
        ]
    missing = np.linalg.norm(normals, axis=1) < 1e-10
    normals[missing] = np.array([0.0, 0.0, 1.0])
    normal_norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(normal_norms, 1e-10)

    # Away from contact, orienting the normal toward the source point gives a
    # useful outside direction.  Near the surface, retain a reliable triangle
    # normal: exact-on-surface query noise must not override consistent mesh
    # winding.  Fallback normals and inconsistently wound meshes still use the
    # source direction because their original orientation is not trustworthy.
    offsets = selected_samples - surface_points
    offset_norms = np.linalg.norm(offsets, axis=1)
    deadzone = max(float(normal_orientation_deadzone), 0.0)
    orient_toward_source = (
        fallback_normals
        | (offset_norms > deadzone)
        | (not scene_mesh.is_winding_consistent)
    )
    flip = orient_toward_source & (np.sum(normals * offsets, axis=1) < 0.0)
    normals[flip] *= -1.0

    return selected_samples, surface_points, normals, selected_distances


def _closest_scene_samples(
    scene_mesh: trimesh.Trimesh,
    contact_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find the closest source sample and scene point at every frame."""
    query = _query_scene_samples(scene_mesh, contact_samples)
    return _select_closest_scene_samples(scene_mesh, *query)


def _contact_patch_surface_points(
    scene_mesh: trimesh.Trimesh,
    all_surface_points: np.ndarray,
    all_distances: np.ndarray,
    triangle_ids: np.ndarray,
    selected_surface_points: np.ndarray,
    stable_normals: np.ndarray,
    contact_mask: np.ndarray,
    *,
    max_patch_distance: float,
    spatial_radius: float,
    normal_angle: float,
) -> np.ndarray:
    """Use a robust nearby surface patch instead of one jumping vertex."""
    result = np.asarray(selected_surface_points, dtype=float).copy()
    face_normals = np.asarray(scene_mesh.face_normals, dtype=float)
    normal_cosine = np.cos(max(float(normal_angle), 0.0))
    for frame_idx in np.flatnonzero(contact_mask):
        keep = all_distances[frame_idx] <= max_patch_distance
        if spatial_radius > 0.0:
            keep &= (
                np.linalg.norm(
                    all_surface_points[frame_idx]
                    - selected_surface_points[frame_idx],
                    axis=1,
                )
                <= spatial_radius
            )
        frame_triangles = triangle_ids[frame_idx]
        valid = (frame_triangles >= 0) & (frame_triangles < len(face_normals))
        if np.any(valid):
            alignment = np.zeros(len(frame_triangles), dtype=float)
            alignment[valid] = np.abs(
                face_normals[frame_triangles[valid]] @ stable_normals[frame_idx]
            )
            keep &= (~valid) | (alignment >= normal_cosine)
        if np.any(keep):
            result[frame_idx] = np.median(
                all_surface_points[frame_idx, keep], axis=0
            )
    return result
