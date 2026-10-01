"""One shaped, skin-derived sole definition shared by SMPL-X consumers."""

import numpy as np

from omniretargeting.foot_frames import (
    calibrate_sole_rotation,
    calibrations_to_metadata,
    world_sole_rotations,
)
from omniretargeting.skin_frames import material_surface_frames


FOOT_JOINTS = {"left": (7, 10), "right": (8, 11)}


def sole_band_vertex_ids(vertices, mask, normal, band_fraction=0.15):
    """Select fixed rest-pose material identities, never per-frame minima."""
    ids = np.flatnonzero(np.asarray(mask, dtype=bool))
    height = np.asarray(vertices, dtype=float)[ids] @ np.asarray(normal, dtype=float)
    take = max(4, int(round(band_fraction * len(ids))))
    return ids[np.argsort(height)[:take]]


def smplx_sole_metadata(vertices, joints, weights, orientations, posed_vertices=None):
    """Calibrate on the actual shaped subject and follow its unchanged skin.

    The heel belongs mainly to the ankle, the forefoot to the toe. Transporting
    the whole sole with the toe alone invents a rotation during toe articulation.
    A fixed-material rigid fit follows both, reporting the nonrigid residual.
    Without skin observations the calibrated joint frame remains a fallback.
    """
    vertices = np.asarray(vertices)
    rotations, calibrations, ids, points, errors = {}, {}, {}, {}, {}
    for side, (ankle, toe) in FOOT_JOINTS.items():
        mask = (weights[:, ankle] + weights[:, toe]) > 0.5
        calibration = calibrate_sole_rotation(side, vertices, joints, mask, ankle, toe)
        calibrations[side] = calibration
        ids[side] = sole_band_vertex_ids(vertices, mask, calibration.rotation[:, 2])
        if posed_vertices is not None:
            points[side] = np.array(posed_vertices[:, ids[side]], dtype=float, copy=True)
            rotations[side], errors[side] = material_surface_frames(
                vertices[ids[side]], points[side], calibration.rotation,
            )
        else:
            rotations[side] = world_sole_rotations(orientations, calibration)
    result = {
        "link_orientations": rotations,
        "foot_sole_calibration": calibrations_to_metadata(calibrations),
        "foot_sole_vertex_ids": ids,
        "foot_sole_template_vertices": np.array(vertices, copy=True),
        "foot_sole_frame_method": (
            "fixed_material_skin_fit" if posed_vertices is not None else "calibrated_toe_joint"
        ),
    }
    if points:
        result.update(
            foot_sole_points=points,
            foot_sole_centres={side: value.mean(axis=1) for side, value in points.items()},
            foot_sole_fit_rms=errors,
        )
    return result
