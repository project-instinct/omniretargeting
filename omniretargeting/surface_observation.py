"""Observe geometric contact evidence in a surface's material coordinates.

The source trajectory is never filtered or displaced. Distance and facing
provide spatial evidence; relative normal speed distinguishes a near pass
from sustained contact. Close, brief touches remain observable. Motion of an
object is removed before classifying sticking versus sliding.
"""

import igl
import numpy as np

from .contact_schedule import _remove_short_runs


class ContactSurface:
    """Immutable local triangle mesh and its original metric pose sequence."""

    def __init__(
        self,
        mesh,
        frames,
        *,
        translations=None,
        rotations=None,
        scales=None,
        name="scene",
    ):
        self.name = str(name)
        self.vertices = np.array(mesh.vertices, dtype=float, copy=True)
        self.faces = np.array(mesh.faces, dtype=int, copy=True)
        self.normals = np.array(mesh.face_normals, dtype=float, copy=True)
        self.oriented = bool(mesh.is_watertight and mesh.is_winding_consistent)
        if self.oriented and mesh.volume < 0:
            self.normals *= -1
        self.translations = (
            np.zeros((frames, 3))
            if translations is None
            else np.array(translations, float, copy=True)
        )
        self.rotations = (
            np.tile(np.eye(3), (frames, 1, 1))
            if rotations is None
            else np.array(rotations, float, copy=True)
        )
        self.scales = (
            np.ones(frames)
            if scales is None
            else np.array(scales, float, copy=True).reshape(-1)
        )
        if (
            not len(self.faces)
            or self.translations.shape != (frames, 3)
            or self.rotations.shape != (frames, 3, 3)
            or self.scales.shape != (frames,)
            or not all(
                np.isfinite(a).all()
                for a in (self.vertices, self.translations, self.rotations, self.scales)
            )
            or np.any(self.scales <= 0)
            or not np.allclose(
                self.rotations @ self.rotations.transpose(0, 2, 1), np.eye(3), atol=1e-5
            )
            or not np.allclose(np.linalg.det(self.rotations), 1, atol=1e-5)
        ):
            raise ValueError(
                "Contact surface geometry and rigid poses must be finite and share the source timeline"
            )
        self.tree = igl.AABB()
        self.tree.init(self.vertices, self.faces)
        for value in (
            self.vertices,
            self.faces,
            self.normals,
            self.translations,
            self.rotations,
            self.scales,
        ):
            value.setflags(write=False)

    def observe(self, points, framerate):
        local = np.einsum(
            "tni,tij->tnj", points - self.translations[:, None], self.rotations
        )
        local /= self.scales[:, None, None]
        nearest, normals = np.empty_like(points), np.empty_like(points)
        distances = np.empty(points.shape[:2])
        for frame in range(len(points)):
            squared, faces, projected = self.tree.squared_distance(
                self.vertices, self.faces, local[frame]
            )
            distances[frame] = np.sqrt(np.maximum(squared, 0)) * self.scales[frame]
            nearest[frame] = (
                projected @ self.rotations[frame].T * self.scales[frame]
                + self.translations[frame]
            )
            normals[frame] = self.normals[faces] @ self.rotations[frame].T
        velocity_local = np.zeros_like(local)
        if len(points) > 1:
            velocity_local = np.gradient(local, axis=0) * framerate
        velocity = (
            np.einsum("tni,tji->tnj", velocity_local, self.rotations)
            * self.scales[:, None, None]
        )
        if not self.oriented:
            # On an open scan only the observed side is known. Select its
            # free-side direction from the full nearest-point displacement,
            # including at edges where a single triangle normal is ambiguous.
            # This is contact evidence, not a claim of solid containment.
            offset = points - nearest
            length = np.linalg.norm(offset, axis=-1)
            normals = np.where(
                (length > 1e-8)[..., None],
                offset / np.maximum(length[..., None], 1e-8),
                normals,
            )
        return distances, nearest, normals, velocity, local


def observe_contacts(
    points, skin_normals, valid, scene_mesh, config, *, framerate=30.0, scene_objects=()
):
    if not np.isfinite(framerate) or framerate <= 0:
        raise ValueError(
            "Surface contact observation requires a positive source framerate"
        )
    frames = len(points)
    surfaces = []
    if scene_mesh is not None and len(scene_mesh.faces):
        surfaces.append(ContactSurface(scene_mesh, frames))
    for spec in scene_objects or ():
        surfaces.append(
            ContactSurface(
                spec["mesh"],
                frames,
                translations=spec.get("translations"),
                rotations=spec.get("rotations"),
                scales=spec.get("scales"),
                name=spec.get("name", "object"),
            )
        )
    if not surfaces:
        raise ValueError("Surface contact observation needs a scene or object mesh")
    shape = points.shape[:2]
    distance = np.full(shape, np.inf)
    nearest, normals, relative_velocity, material_points = (
        np.zeros_like(points) for _ in range(4)
    )
    surface_ids = np.full(shape, -1, int)
    for index, surface in enumerate(surfaces):
        d, p, n, velocity, local = surface.observe(points, framerate)
        take = d < distance
        distance[take], nearest[take], normals[take] = d[take], p[take], n[take]
        relative_velocity[take], material_points[take], surface_ids[take] = (
            velocity[take],
            local[take],
            index,
        )
    normal_speed = np.abs(np.sum(relative_velocity * normals, axis=-1))
    tangent_velocity = (
        relative_velocity
        - np.sum(relative_velocity * normals, axis=-1)[..., None] * normals
    )
    tangent_speed = np.linalg.norm(tangent_velocity, axis=-1)
    opposition = -np.sum(skin_normals * normals, axis=-1)
    close = distance <= config["touch_distance"]
    facing = opposition >= config["normal_opposition"]
    slow = normal_speed <= config["normal_speed"]
    near = distance < config["distance"]
    active = valid & near & facing & (slow | close)
    for index in range(shape[1]):
        active[:, index] = _remove_short_runs(
            active[:, index], int(config["min_contact_frames"])
        )
    # A fixed spatial ramp and fixed per-region normalization avoid the old
    # discontinuity where toggling one point changed every other point's weight.
    strength = np.clip(
        (config["distance"] - distance)
        / (config["distance"] - config["touch_distance"]),
        0,
        1,
    )
    confidence = strength * strength * (3 - 2 * strength)
    confidence *= np.clip(opposition, 0, 1)
    confidence *= active
    modes = np.full(shape, "none", dtype="U8")
    modes[active] = "touch"
    modes[active & slow & (tangent_speed <= config["sticking_speed"])] = "sticking"
    modes[active & slow & (tangent_speed > config["sticking_speed"])] = "sliding"
    return {
        "environment": nearest,
        "normals": normals,
        "active": active,
        "confidence": confidence,
        "modes": modes,
        "surface_ids": surface_ids,
        "material_points": material_points,
        "normal_speed": normal_speed,
        "tangent_speed": tangent_speed,
        "diagnostics": {
            "method": "distance_facing_and_material_relative_motion",
            "surface_names": [s.name for s in surfaces],
            "near_frame_points": int(np.count_nonzero(valid & near)),
            "rejected_facing": int(np.count_nonzero(valid & near & ~facing)),
            "rejected_normal_motion": int(
                np.count_nonzero(valid & near & facing & ~slow & ~close)
            ),
            "mode_frame_points": {
                name: int(np.count_nonzero(modes == name))
                for name in ("touch", "sticking", "sliding")
            },
            "physical_support_verified": False,
        },
    }
