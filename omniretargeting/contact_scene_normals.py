"""Scene-side orientation and temporal continuity of contact normals."""

import numpy as np
import trimesh

from .contact_schedule import _iter_true_runs

_DEFAULT_NORMAL_ORIENTATION_DEADZONE = 0.005


def enclosed_free_space_normals(mesh, triangle_ids):
    """An observed point crossing a room boundary cannot reverse its free side."""
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError("Enclosed free space requires an oriented watertight boundary")
    # File/cached normals may disagree with triangle winding after transforms.
    normals, valid = trimesh.triangles.normals(mesh.triangles)
    if not valid[triangle_ids].all():
        raise ValueError("Contact triangle has no reliable geometric normal")
    return normals[triangle_ids] * (-1 if mesh.volume > 0 else 1)


def _stabilize_contact_normals(
    normals: np.ndarray,
    tracking_positions: np.ndarray,
    surface_points: np.ndarray,
    distances: np.ndarray,
    contact_mask: np.ndarray,
    *,
    onset_window_frames: int,
    smoothing_frames: int,
    normal_orientation_deadzone: float = _DEFAULT_NORMAL_ORIENTATION_DEADZONE,
) -> np.ndarray:
    """Make each contact run use a continuous, outward scene normal.

    Video registration often lets a hand pass a few centimetres through a
    reconstructed surface.  Orienting every normal toward that penetrating
    source sample then flips the robot target into the scene.  The approach
    side at contact onset is the reliable side; subsequent normals are made
    hemisphere-continuous and optionally median-smoothed without changing the
    underlying surface tangent plane.
    """
    result = np.asarray(normals, dtype=float).copy()
    tracking = np.asarray(tracking_positions, dtype=float)
    points = np.asarray(surface_points, dtype=float)
    distance_values = np.asarray(distances, dtype=float)
    onset_window = max(int(onset_window_frames), 1)
    smoothing_window = max(int(smoothing_frames), 1)
    if smoothing_window % 2 == 0:
        smoothing_window += 1

    for start, end in _iter_true_runs(contact_mask):
        context_start = max(0, start - onset_window)
        context_end = min(end, start + 1)
        context_indices = np.arange(context_start, context_end)
        if len(context_indices):
            finite = np.isfinite(distance_values[context_indices])
            reliable_indices = context_indices[finite]
            reference_idx = (
                int(reliable_indices[np.argmax(distance_values[reliable_indices])])
                if len(reliable_indices)
                else start
            )
        else:
            reference_idx = start
        reference = result[reference_idx].copy()
        reference /= max(float(np.linalg.norm(reference)), 1e-10)

        approach_offsets = (
            tracking[context_indices] - points[context_indices]
            if len(context_indices)
            else tracking[start : start + 1] - points[start : start + 1]
        )
        approach_side = np.median(approach_offsets, axis=0)
        if (
            float(np.linalg.norm(approach_side))
            > max(float(normal_orientation_deadzone), 0.0)
            and float(np.dot(reference, approach_side)) < 0.0
        ):
            reference *= -1.0

        previous = reference
        for frame_idx in range(start, end):
            normal = result[frame_idx]
            normal /= max(float(np.linalg.norm(normal)), 1e-10)
            if float(np.dot(normal, previous)) < 0.0:
                normal *= -1.0
            result[frame_idx] = normal
            previous = normal

        if smoothing_window > 1:
            smoothed = result[start:end].copy()
            half_window = smoothing_window // 2
            for local_idx in range(end - start):
                lo = max(0, local_idx - half_window)
                hi = min(end - start, local_idx + half_window + 1)
                normal = np.median(result[start + lo : start + hi], axis=0)
                normal /= max(float(np.linalg.norm(normal)), 1e-10)
                if float(np.dot(normal, reference)) < 0.0:
                    normal *= -1.0
                smoothed[local_idx] = normal
            result[start:end] = smoothed

    return result
