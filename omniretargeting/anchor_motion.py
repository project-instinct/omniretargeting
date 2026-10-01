"""Material contact anchors that allow the source's observed rigid-body turn.

A sticking point constrains that point, not all six coordinates of its owner.
The functions use body-local geometry and recorded surface poses only.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .contact_schedule import _iter_true_runs


def source_material_rotations(orientation, surface_rotation):
    """Average the same material frames at every timestamp, never active slots."""
    relative = surface_rotation[:, None].transpose(0, 1, 3, 2) @ orientation
    return np.asarray(
        [Rotation.from_matrix(frame).mean().as_matrix() for frame in relative]
    )


def fit_moving_anchor(
    local,
    environment,
    active,
    modes,
    confidence,
    source_rotation,
    rotation,
    translation,
    scale,
    *,
    rotation_budget_rad,
):
    """Bound each body pose by its same-frame source orientation.

    Sticking samples retain one observed material anchor for their entire run.
    Sliding samples use the original time-varying observations. The rigid fit
    reports incompatible morphology instead of stretching the robot patch.
    """
    from .contact_anchors import rigid_placement

    material = np.einsum("tni,tij->tnj", environment - translation[:, None], rotation)
    material /= scale[:, None, None]
    targets = material.copy()
    sticky = active & (modes == "sticking")
    for sample in range(local.shape[0]):
        for start, stop in _iter_true_runs(sticky[:, sample]):
            points = material[start:stop, sample]
            weights = confidence[start:stop, sample]
            if not np.any(weights > 0):
                continue
            center = np.average(points, axis=0, weights=weights)
            # Use an actually observed surface point, including curved patches.
            anchor = points[np.argmin(np.linalg.norm(points - center, axis=1))]
            targets[start:stop, sample] = anchor
    world = environment.copy()
    rotations, origins, fits = [], [], []
    for frame in range(len(environment)):
        selected = active[frame] & (confidence[frame] > 0)
        if not selected.any():
            raise ValueError(
                "A moving anchor requires positive evidence in every frame"
            )
        fitted, offset, fit = rigid_placement(
            local[selected] / scale[frame],
            targets[frame, selected],
            confidence[frame, selected],
            source_rotation[frame],
            rotation_budget_rad=rotation_budget_rad,
        )
        posed = local @ fitted.T + offset * scale[frame]
        world[frame] = posed @ rotation[frame].T + translation[frame]
        residual = (
            np.linalg.norm(
                posed[selected] / scale[frame] - targets[frame, selected], axis=1
            )
            * scale[frame]
        )
        fit["weighted_rms_m"] = float(
            np.sqrt(np.average(residual**2, weights=confidence[frame, selected]))
        )
        fit["maximum_m"] = float(residual.max())
        fits.append(fit)
        rotations.append(fitted)
        origins.append(offset)
    return world, {
        "mode": "source_following_material_contacts",
        "rotation": np.asarray(rotations),
        "translation": np.asarray(origins),
        "fit": {
            "maximum_m": max(f["maximum_m"] for f in fits),
            "weighted_rms_m": float(
                np.sqrt(np.mean([f["weighted_rms_m"] ** 2 for f in fits]))
            ),
            "per_frame": fits,
            "robot_metric_size_preserved": True,
        },
        "sticking_material_targets": targets,
        "source_rotation": source_rotation.copy(),
        "orientation_budget_rad": float(rotation_budget_rad),
    }


def observed_material_targets(
    environment, active, modes, confidence, rotation, translation, scale
):
    """Keep contact goals on recorded surfaces, independently of robot shape.

    One actually observed point is retained per contiguous sticking run. A
    rigid robot fit is deliberately absent: incompatibility belongs to IK's
    residual, not to a replacement surface target. Sliding keeps every sample.
    """
    environment = np.asarray(environment, float)
    material = np.einsum("tni,tij->tnj", environment - translation[:, None], rotation)
    material /= scale[:, None, None]
    result = environment.copy()
    sticky = active & (modes == "sticking")
    for sample in range(environment.shape[1]):
        for start, stop in _iter_true_runs(sticky[:, sample]):
            weights = confidence[start:stop, sample]
            positive = weights > 0
            if not positive.any():
                continue
            observed = material[start:stop, sample][positive]
            center = np.average(observed, axis=0, weights=weights[positive])
            anchor = observed[np.argmin(np.linalg.norm(observed - center, axis=1))]
            result[start:stop, sample] = (
                np.einsum("tij,j->ti", rotation[start:stop], anchor)
                * scale[start:stop, None]
                + translation[start:stop]
            )
    return result
