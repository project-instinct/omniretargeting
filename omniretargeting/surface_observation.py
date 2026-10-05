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
        enclosed_free_space=None,
    ):
        self.name = str(name)
        self.vertices = np.array(mesh.vertices, dtype=float, copy=True)
        self.faces = np.array(mesh.faces, dtype=int, copy=True)
        self.normals = np.array(mesh.face_normals, dtype=float, copy=True)
        self.oriented = bool(mesh.is_watertight and mesh.is_winding_consistent)
        # A negatively wound static scene commonly describes the inside of a
        # room. Its interior is the observed free space; moving object meshes
        # still use their exterior as free space.
        self.known_side = enclosed_free_space is not None or name != "scene"
        self.interior_free = (
            bool(enclosed_free_space)
            if enclosed_free_space is not None
            else bool(self.oriented and self.name == "scene" and mesh.volume < 0)
        )
        if enclosed_free_space and not self.oriented:
            raise ValueError("Enclosed free space needs an oriented watertight mesh")
        if self.oriented and mesh.volume < 0:
            self.normals *= -1
        if self.interior_free:
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
        self._tree = None
        for value in (
            self.vertices,
            self.faces,
            self.normals,
            self.translations,
            self.rotations,
            self.scales,
        ):
            value.setflags(write=False)

    @property
    def tree(self):
        """Pose-only consumers need validation, but no spatial query tree."""
        if self._tree is None:
            tree = igl.AABB()
            tree.init(self.vertices, self.faces)
            self._tree = tree
        return self._tree

    def observe(self, points, framerate, *, return_clearance_side=False):
        local = np.einsum(
            "tni,tij->tnj", points - self.translations[:, None], self.rotations
        )
        local /= self.scales[:, None, None]
        # One query for every frame: points are independent in the local tree.
        squared, faces, projected = self.tree.squared_distance(
            self.vertices, self.faces, local.reshape(-1, 3)
        )
        squared = np.asarray(squared).reshape(points.shape[:2])
        faces = np.asarray(faces).reshape(points.shape[:2])
        projected = np.asarray(projected).reshape(points.shape)
        distances = np.sqrt(np.maximum(squared, 0)) * self.scales[:, None]
        nearest = (
            np.matmul(projected, self.rotations.transpose(0, 2, 1))
            * self.scales[:, None, None]
            + self.translations[:, None]
        )
        normals = np.matmul(self.normals[faces], self.rotations.transpose(0, 2, 1))
        velocity_local = np.zeros_like(local)
        if len(points) > 1:
            velocity_local = np.gradient(local, axis=0) * framerate
        velocity = (
            np.einsum("tni,tji->tnj", velocity_local, self.rotations)
            * self.scales[:, None, None]
        )
        offset = points - nearest
        length = np.linalg.norm(offset, axis=-1)
        geometric_side = np.sum(offset * normals, axis=-1)
        clear_side = geometric_side >= -1e-6
        if not self.oriented:
            # On an open scan only the observed side is known. Select its
            # free-side direction from the full nearest-point displacement,
            # including at edges where a single triangle normal is ambiguous.
            # This is contact evidence, not a claim of solid containment.
            normals = np.where(
                (length > 1e-8)[..., None],
                offset / np.maximum(length[..., None], 1e-8),
                normals,
            )
        elif not self.known_side:
            # Closed room scans enclose free space, while object meshes enclose
            # solids. Face winding/volume cannot distinguish these semantics.
            # For source contact evidence orient the nearest face toward the
            # side occupied by the observed skin, preserving its tangent plane.
            toward_skin = np.sum(offset * normals, axis=-1)
            normals = np.where(
                ((length > 1e-8) & (toward_skin < 0))[..., None],
                -normals,
                normals,
            )
        result = distances, nearest, normals, velocity, local
        return (*result, clear_side) if return_clearance_side else result


def observe_contacts(
    points,
    skin_normals,
    valid,
    scene_mesh,
    config,
    *,
    framerate=30.0,
    scene_objects=(),
    scene_enclosed_free_space=None,
):
    if not np.isfinite(framerate) or framerate <= 0:
        raise ValueError(
            "Surface contact observation requires a positive source framerate"
        )
    frames = len(points)
    surfaces = []
    if scene_mesh is not None and len(scene_mesh.faces):
        surfaces.append(
            ContactSurface(
                scene_mesh, frames, enclosed_free_space=scene_enclosed_free_space
            )
        )
    for spec in scene_objects or ():
        surfaces.append(
            ContactSurface(
                spec["mesh"],
                frames,
                translations=spec.get("translations"),
                rotations=spec.get("rotations"),
                scales=spec.get("scales"),
                name=spec.get("name", "object"),
                enclosed_free_space=False,
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
    material_environment = np.zeros_like(points)
    clear_side = np.ones(shape, dtype=bool)
    for index, surface in enumerate(surfaces):
        d, p, n, velocity, local, side = surface.observe(
            points, framerate, return_clearance_side=True
        )
        clear_side &= side
        take = d < distance
        distance[take], nearest[take], normals[take] = d[take], p[take], n[take]
        relative_velocity[take], material_points[take], surface_ids[take] = (
            velocity[take],
            local[take],
            index,
        )
        projected_local = np.einsum("tni,tij->tnj", p - surface.translations[:, None], surface.rotations) / surface.scales[:, None, None]
        material_environment[take] = projected_local[take]
    normal_speed = np.abs(np.sum(relative_velocity * normals, axis=-1))
    tangent_velocity = (
        relative_velocity
        - np.sum(relative_velocity * normals, axis=-1)[..., None] * normals
    )
    tangent_speed = np.linalg.norm(tangent_velocity, axis=-1)
    normal_known = np.linalg.norm(skin_normals, axis=-1) > 1e-10
    opposition = -np.sum(skin_normals * normals, axis=-1)
    # Marker-only adapters may lack a measured skin normal. Such markers are
    # geometric candidates, not a measured facing certificate. Keep the mask
    # in diagnostics rather than fabricating a triangle or rejecting all points.
    opposition = np.where(normal_known, opposition, 1.0)
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
        "distance": distance,
        "environment": nearest,
        "normals": normals,
        "active": active,
        "confidence": confidence,
        "modes": modes,
        "surface_ids": surface_ids,
        "clear_side": clear_side,
        "material_points": material_points,
        "material_environment": material_environment,
        "surface_poses": tuple((s.rotations, s.translations, s.scales) for s in surfaces),
        "normal_speed": normal_speed,
        "tangent_speed": tangent_speed,
        "relative_velocity": relative_velocity,
        "normal_opposition": opposition,
        "skin_normal_known": normal_known,
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
