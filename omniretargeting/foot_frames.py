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
* the robot side reads its sole off the foot body's collision geoms, with an
  optional configured ``body frame -> sole frame`` rotation as an override.

The robot side is derived rather than hand-measured because hand-measured
offsets do not transfer. Across the humanoids in this repository the sole plane
sits between 3.5 cm and 19 cm below the foot body origin, and as a fraction of
robot height that spans 2.8 to 11 percent, so neither a constant nor a
height-normalised constant works. The collision geoms are the surfaces the
robot actually stands on and they are already declared, so they are the right
authority. Derivation is validated rather than trusted: a foot whose geometry
has no distinguishable sole plane is rejected instead of guessed at, and a
profile can always state the transform explicitly.

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
    "DerivedSole",
    "derive_sole_from_points",
    "sample_body_collision_points",
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


@dataclass(frozen=True)
class DerivedSole:
    """Sole geometry read off a foot body's collision geoms."""

    reference_point: np.ndarray  # body-frame point on the sole plane
    normal: np.ndarray           # body-frame outward sole normal, away from the ground
    plane_offset: float          # signed distance from the body origin to the plane
    extent_forward: float        # heel-to-toe span in the sole plane
    extent_lateral: float        # width in the sole plane
    band_points: int             # how many sampled points landed on the plane


def derive_sole_from_points(
    points: np.ndarray,
    band_tolerance: float = 0.01,
    isotropy_limit: float = 0.5,
) -> DerivedSole:
    """Locate the sole plane of a foot from its collision geometry.

    The alternative is a hand-measured offset per robot, which does not
    transfer: across the humanoids in this repository the sole plane sits
    anywhere from 3.5 cm to 19 cm below the foot body origin, and as a fraction
    of robot height that spans 2.8 to 11 percent, so neither a constant nor a
    height-normalised constant works. The collision geoms, on the other hand,
    are exactly the surfaces the robot stands on, and they are already declared.

    The geoms are sampled in the owning body's frame, so this is independent of
    pose and needs no forward kinematics.

    Args:
        points: (N, 3) sampled surface points of the foot's collision geoms,
            in the foot body frame.
        band_tolerance: how far above the lowest point still counts as sole.
        isotropy_limit: reject the foot when its thinnest extent is more than
            this fraction of its longest, meaning it is too blobby for a sole
            plane to be meaningful.

    Returns:
        The sole reference point, its outward normal, and the sole's extents,
        all in the foot body frame.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if points.shape[0] < 4:
        raise ValueError(f"need at least 4 sampled points to fit a sole plane, got {points.shape[0]}")

    centre = points.mean(axis=0)
    centred = points - centre
    _, _, vh = np.linalg.svd(centred, full_matrices=False)

    # Judge the shape by physical extents rather than by singular values.
    # Singular values weight by how the surface happens to be sampled, and for a
    # blade-shaped foot they put thickness and width within a fifth of each
    # other even though the sole is unambiguous: one humanoid here has a single
    # box foot 28 cm long, 3 cm wide and 2.4 cm thick.
    projected = centred @ vh.T
    extents = projected.max(axis=0) - projected.min(axis=0)
    order = np.argsort(extents)
    if extents[order[-1]] <= 0.0 or extents[order[0]] / extents[order[-1]] > isotropy_limit:
        raise ValueError(
            "foot collision geometry has no distinguishable sole plane "
            f"(thinnest/longest extent ratio "
            f"{extents[order[0]] / max(extents[order[-1]], 1e-12):.3f})"
        )

    # The sole normal is across the foot, and the foot body origin sits on the
    # far side of it: the origin is at the ankle, above the sole. Thickness and
    # width can be close, so the origin offset is what disambiguates them --
    # along the sole normal the ankle is clearly to one side, while across the
    # width it sits near the middle.
    offset = -centre  # body origin expressed relative to the geometry centre
    normal = None
    for axis_idx in order:
        axis = vh[axis_idx]
        if abs(offset @ axis) > 0.25 * extents[axis_idx]:
            normal = axis if offset @ axis > 0.0 else -axis
            break
    if normal is None:
        # No axis separates the ankle from the geometry; fall back to the
        # thinnest one and let the caller's validation catch a bad guess.
        normal = vh[order[0]]
        if offset @ normal < 0.0:
            normal = -normal

    # Refine the normal against the band it selects, then reselect. A first
    # normal that is a few degrees off tilts the band, and a tilted band across
    # a long foot only captures the lowest patch of it -- on one robot here the
    # first pass reported a 7 cm sole for a 19 cm foot. Two passes settle it.
    band = points
    for _ in range(3):
        heights = points @ normal
        band = points[heights <= heights.min() + band_tolerance]
        if band.shape[0] < 3:
            band = points[np.argsort(heights)[: max(3, points.shape[0] // 10)]]
            break
        band_centred = band - band.mean(axis=0)
        _, band_sv, band_vh = np.linalg.svd(band_centred, full_matrices=False)
        if band_sv[1] <= 0.0:
            break
        refined = band_vh[-1]
        if refined @ normal < 0.0:
            refined = -refined
        if np.allclose(refined, normal, atol=1e-6):
            break
        normal = refined

    reference = band.mean(axis=0)
    plane_offset = float(reference @ normal)

    # Extents measured in the plane, along the two directions the sole spans.
    in_plane = band - reference
    in_plane = in_plane - np.outer(in_plane @ normal, normal)
    if in_plane.shape[0] >= 2:
        _, plane_sv, plane_vh = np.linalg.svd(in_plane, full_matrices=False)
        forward = plane_vh[0]
        lateral = plane_vh[1] if plane_vh.shape[0] > 1 else np.cross(normal, forward)
        span = lambda axis: float((in_plane @ axis).max() - (in_plane @ axis).min())
        extent_forward, extent_lateral = span(forward), span(lateral)
    else:
        extent_forward = extent_lateral = 0.0

    return DerivedSole(
        reference_point=reference,
        normal=normal / np.linalg.norm(normal),
        plane_offset=plane_offset,
        extent_forward=extent_forward,
        extent_lateral=extent_lateral,
        band_points=int(band.shape[0]),
    )


def sample_body_collision_points(model, body_id: int, max_points: int = 512) -> np.ndarray:
    """Sample the surface of a body's collision geoms, in the body frame.

    Uses the static geom placement, so it is pose independent and needs no
    forward kinematics.  Primitives are preferred when a body carries both a
    primitive proxy and a mesh, since the primitive is what the model author
    declared for collision; a mesh-only body is subsampled rather than skipped,
    because shelled robots often have no primitive foot at all.
    """
    import mujoco
    from scipy.spatial.transform import Rotation

    geom_ids = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == int(body_id)]
    if not geom_ids:
        return np.zeros((0, 3))

    primitives = [g for g in geom_ids if int(model.geom_type[g]) != mujoco.mjtGeom.mjGEOM_MESH]
    if primitives:
        geom_ids = primitives

    clouds = []
    for g in geom_ids:
        gtype = int(model.geom_type[g])
        size = np.asarray(model.geom_size[g], dtype=float)

        if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = size[0]
            local = np.array([[0, 0, -r], [0, 0, r], [r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0]], dtype=float)
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            hx, hy, hz = size[:3]
            local = np.array([[sx, sy, sz] for sx in (-hx, hx) for sy in (-hy, hy) for sz in (-hz, hz)], dtype=float)
        elif gtype in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            r, half = size[0], size[1]
            theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
            rings = [
                np.column_stack([r * np.cos(theta), r * np.sin(theta), np.full_like(theta, z)])
                for z in (-half, 0.0, half)
            ]
            caps = [[0, 0, -half], [0, 0, half]]
            if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
                caps += [[0, 0, -half - r], [0, 0, half + r]]
            local = np.vstack(rings + [np.asarray(caps, dtype=float)])
        elif gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            rx, ry, rz = size[:3]
            local = np.array([[0, 0, -rz], [0, 0, rz], [rx, 0, 0], [-rx, 0, 0], [0, ry, 0], [0, -ry, 0]], dtype=float)
        elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(model.geom_dataid[g])
            if mesh_id < 0:
                continue
            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            verts = np.asarray(model.mesh_vert[start:start + count], dtype=float).reshape(-1, 3)
            stride = max(1, verts.shape[0] // max_points)
            local = verts[::stride]
        else:
            continue

        quat = np.asarray(model.geom_quat[g], dtype=float)
        rot = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        clouds.append(local @ rot.T + np.asarray(model.geom_pos[g], dtype=float))

    if not clouds:
        return np.zeros((0, 3))
    return np.vstack(clouds)
