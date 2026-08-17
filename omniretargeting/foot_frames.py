"""Anatomical sole frames for source and robot feet.

Foot orientation only means something once both sides agree on what "the sole
frame" is.  The raw SMPL-X foot joint frame is not it: measured on the shaped
template, the joint frame and the anatomical sole frame differ by roughly
twenty degrees, dominated by toe-out splay, and the offset is not symmetric
between feet.  Comparing a robot foot body frame directly against a source
joint frame therefore bakes in a fixed error before the optimizer has done
anything.

This module defines both halves of the contract:

* the source side calibrates ``joint frame -> sole frame`` once on the shaped
  template, using the thinnest principal axis of the foot volume as the sole
  normal and the ankle-to-toe vector projected into the sole plane as forward;
* the robot side takes an explicitly configured ``body frame -> sole frame``
  rotation, defaulting to identity.

The robot transform is configured rather than inferred.  Inferring it from
collision geometry is ill-posed for curved soles, mesh-only feet, and sparse
collision models, and the robot description is the authority on where its own
sole is.

Both frames use columns ``[forward, lateral, up]``: +x points at the toe, +z
points out of the sole away from the ground.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np

__all__ = [
    "SoleFrameCalibration",
    "calibrate_sole_rotation",
    "world_sole_rotations",
    "parse_body_to_sole",
    "orthonormalize",
]


@dataclass(frozen=True)
class SoleFrameCalibration:
    """Fixed ``joint frame -> sole frame`` rotation for one foot."""

    side: str
    ankle_index: int
    toe_index: int
    rotation: np.ndarray  # (3, 3)

    def __post_init__(self) -> None:
        if self.rotation.shape != (3, 3):
            raise ValueError(f"{self.side} sole calibration must be a 3x3 rotation")
        if not np.isclose(np.linalg.det(self.rotation), 1.0, atol=1e-5):
            raise ValueError(f"{self.side} sole calibration is not a proper rotation")


def orthonormalize(rotation: np.ndarray) -> np.ndarray:
    """Nearest proper rotation, so accumulated drift cannot leave SO(3)."""
    u, _, vt = np.linalg.svd(np.asarray(rotation, dtype=float))
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def calibrate_sole_rotation(
    side: str,
    template_vertices: np.ndarray,
    template_joints: np.ndarray,
    foot_vertex_mask: np.ndarray,
    ankle_index: int,
    toe_index: int,
    sole_band_fraction: float = 0.15,
    sole_band_isotropy_limit: float = 0.9,
) -> SoleFrameCalibration:
    """Calibrate one foot's sole frame against its toe joint frame.

    Args:
        template_vertices: (V, 3) shaped template mesh in the rest pose.
        template_joints: (J, 3) shaped template joints in the same pose.
        foot_vertex_mask: (V,) boolean selecting the vertices of this foot.
        ankle_index: source joint index of the ankle.
        toe_index: source joint index of the toe base, which is the joint the
            sole frame is attached to.
        sole_band_fraction: fraction of the foot's vertices, taken lowest
            first, treated as the ground-contact band.
        sole_band_isotropy_limit: reject the foot if its two smallest singular
            values are this close together, since then no thin axis exists to
            start the refinement from.

    The calibration is computed on the rest pose only.  Doing it per frame
    would let a mid-stride foot deformation redefine the reference, so the
    reference would move with the very quantity being measured.
    """
    foot = np.asarray(template_vertices, dtype=float)[np.asarray(foot_vertex_mask, dtype=bool)]
    if foot.shape[0] < 4:
        raise ValueError(f"{side} foot vertex mask selected {foot.shape[0]} vertices; need at least 4")

    ankle = np.asarray(template_joints[ankle_index], dtype=float)

    # The thinnest principal axis of the whole foot volume is NOT the sole
    # normal: the instep and ankle mass tilt it, measurably by about fifteen
    # degrees on SMPL-X.  Use it only as a starting direction, then fit a plane
    # to the vertices that actually touch the ground.
    centred = foot - foot.mean(axis=0)
    _, singular, vh = np.linalg.svd(centred, full_matrices=False)
    if singular[1] <= 0.0 or singular[2] / singular[1] > sole_band_isotropy_limit:
        raise ValueError(
            f"{side} foot volume is too isotropic to identify a sole plane "
            f"(singular value ratio {singular[2] / max(singular[1], 1e-12):.3f})"
        )
    up = vh[-1]
    if up @ (ankle - foot.mean(axis=0)) < 0.0:
        up = -up

    # Refine: refit the normal to the vertices lowest along the current
    # direction.  Selecting a fixed fraction of the vertex count rather than a
    # slice of the height range keeps the sample size stable when the initial
    # direction is off, which is exactly the case being corrected here.  Two
    # passes are enough for the direction to settle.
    band_size = max(4, int(round(sole_band_fraction * foot.shape[0])))
    for _ in range(2):
        height = foot @ up
        band = foot[np.argsort(height)[:band_size]]
        if band.shape[0] < 4:
            break
        band_centred = band - band.mean(axis=0)
        _, band_singular, band_vh = np.linalg.svd(band_centred, full_matrices=False)
        if band_singular[1] <= 0.0:
            break
        refined = band_vh[-1]
        if refined @ up < 0.0:
            refined = -refined
        up = refined

    forward = np.asarray(template_joints[toe_index], dtype=float) - np.asarray(
        template_joints[ankle_index], dtype=float
    )
    forward = forward - (forward @ up) * up
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        raise ValueError(f"{side} ankle-to-toe vector is parallel to the sole normal")
    forward = forward / norm

    lateral = np.cross(up, forward)
    rotation = orthonormalize(np.column_stack([forward, lateral, up]))
    return SoleFrameCalibration(side=side, ankle_index=ankle_index, toe_index=toe_index, rotation=rotation)


def world_sole_rotations(
    joint_orientations_wxyz: np.ndarray,
    calibration: SoleFrameCalibration,
) -> np.ndarray:
    """Transport the calibrated sole frame into the world, per frame.

    Args:
        joint_orientations_wxyz: (T, J, 4) world joint orientations, scalar first.

    Returns:
        (T, 3, 3) world rotations whose columns are ``[forward, lateral, up]``.
    """
    from scipy.spatial.transform import Rotation

    quats = np.asarray(joint_orientations_wxyz, dtype=float)
    if quats.ndim != 3 or quats.shape[-1] != 4:
        raise ValueError("joint orientations must have shape (T, J, 4)")
    if calibration.toe_index >= quats.shape[1]:
        raise ValueError(
            f"toe index {calibration.toe_index} is out of range for {quats.shape[1]} source joints"
        )
    joint_world = Rotation.from_quat(quats[:, calibration.toe_index], scalar_first=True).as_matrix()
    return np.einsum("tij,jk->tik", joint_world, calibration.rotation)


def parse_body_to_sole(value: Optional[Sequence[Sequence[float]]], label: str) -> np.ndarray:
    """Read a configured ``body frame -> sole frame`` rotation.

    ``None`` means the robot's foot body frame already is its sole frame, which
    is true for every humanoid currently in ``robot_models``.
    """
    if value is None:
        return np.eye(3)
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"{label} body_to_sole must be a 3x3 rotation matrix")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
        raise ValueError(f"{label} body_to_sole is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{label} body_to_sole is not a proper rotation (determinant is not +1)")
    return rotation


def calibrations_to_metadata(calibrations: Dict[str, SoleFrameCalibration]) -> dict:
    """Serialise calibrations for diagnostics and provenance."""
    return {
        side: {
            "ankle_index": cal.ankle_index,
            "toe_index": cal.toe_index,
            "rotation": cal.rotation.tolist(),
        }
        for side, cal in calibrations.items()
    }
