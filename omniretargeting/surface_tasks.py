"""Optional engineering surface normals, with stable identities and no scaling.

UMR algorithm reference: relative reference-face normal transport and per-FK
slot caching (UMR 8c4db7db); independent implementation.
This is not learned correspondence. Source adapters must supply real surfaces
and explicitly corresponding reference poses, all in the same metric world frame.
"""

from dataclasses import dataclass

import mujoco
import numpy as np


def _face_frames(triangles):
    edge = triangles[..., 1, :] - triangles[..., 0, :]
    second = triangles[..., 2, :] - triangles[..., 0, :]
    cross = np.cross(edge, second)
    length = np.linalg.norm(edge, axis=-1)
    area = np.linalg.norm(cross, axis=-1)
    valid = (
        (length > 1e-7) & (area > 1e-10) & np.all(np.isfinite(triangles), axis=(-2, -1))
    )
    x = edge / np.maximum(length[..., None], 1e-12)
    z = cross / np.maximum(area[..., None], 1e-12)
    return np.stack([x, np.cross(z, x), z], axis=-1), valid


@dataclass(frozen=True)
class SurfaceBinding:
    part: str
    face_id: int
    barycentric: tuple[float, float, float]
    robot_geom: str | int
    local_point: tuple[float, float, float]
    local_normal: tuple[float, float, float]


class SurfaceKinematics:
    """Own FK state and invalidate points/Jacobians on every changed qpos."""

    def __init__(self, model, bindings):
        self.model, self.bindings = model, tuple(bindings)
        self.data = mujoco.MjData(model)
        self.qpos, self.cache, self._key = None, {}, None
        # Per-binding constants; name lookups and tiny numpy calls dominate
        # when thousands of dense samples are queried every frame.
        self._geoms = [int(model.geom(b.robot_geom).id) for b in self.bindings]
        self._bodies = [int(model.geom_bodyid[g]) for g in self._geoms]
        self._points = [np.asarray(b.local_point, dtype=float) for b in self.bindings]
        self._normals = [np.asarray(b.local_normal, dtype=float) for b in self.bindings]

    def at(self, qpos, index):
        key = np.asarray(qpos, dtype=float).tobytes()
        if key != self._key:
            self._key = key
            self.qpos = np.array(qpos, dtype=float, copy=True)
            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)
            self.cache.clear()
        if index not in self.cache:
            gid = self._geoms[index]
            rotation = self.data.geom_xmat[gid].reshape(3, 3)
            point = self.data.geom_xpos[gid] + rotation @ self._points[index]
            normal = rotation @ self._normals[index]
            normal = normal / np.sqrt(normal @ normal)
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jp, jr, point, self._bodies[index])
            # d(normal) = omega x normal = -[normal]x omega.
            x, y, z = normal
            jn = -np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]]) @ jr
            self.cache[index] = (point, normal, jp, jn)
        return self.cache[index]

    def batch(self, qpos, indices):
        """Points, normals and their Jacobians for many bindings at once.

        One body Jacobian per rigid body; each point then moves with it:
        J_p = J_origin - [p - origin]x J_rot, and J_n = -[n]x J_rot.
        """
        if len(indices):
            self.at(qpos, int(indices[0]))  # forward kinematics at qpos
        data, nv = self.data, self.model.nv
        geoms = np.asarray(self._geoms)[indices]
        bodies = np.asarray(self._bodies)[indices]
        rotation = data.geom_xmat[geoms].reshape(-1, 3, 3)
        local = np.asarray(self._points)[indices]
        points = data.geom_xpos[geoms] + np.einsum("kij,kj->ki", rotation, local)
        normals = np.einsum("kij,kj->ki", rotation, np.asarray(self._normals)[indices])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        jp, jn = np.zeros((len(indices), 3, nv)), np.zeros((len(indices), 3, nv))
        for body in np.unique(bodies):
            linear, angular = np.zeros((3, nv)), np.zeros((3, nv))
            mujoco.mj_jacBody(self.model, data, linear, angular, int(body))
            group = bodies == body
            jp[group] = linear - _skew(points[group] - data.xpos[body]) @ angular
            jn[group] = -_skew(normals[group]) @ angular
        return points, normals, jp, jn


def _skew(vectors):
    x, y, z = np.moveaxis(vectors, -1, 0)
    zero = np.zeros_like(x)
    return np.stack(
        [np.stack([zero, -z, y], -1), np.stack([z, zero, -x], -1), np.stack([-y, x, zero], -1)],
        -2,
    )


class SurfaceTasks:
    """Few normal-only auxiliary tasks; contacts remain existing world tasks.

    Human barycentric points are retained as provenance, not turned into
    high-priority human-world position targets. Each part's total normal weight
    is invariant to its sample count. Invalid source triangles disable only that
    frame/slot and are reported. A single normal leaves twist unconstrained.
    """

    def __init__(
        self,
        model,
        bindings,
        reference_vertices_world,
        faces,
        motion_vertices_world,
        robot_reference_qpos,
        *,
        corresponding_reference_poses=False,
        weight=0.05,
    ):
        if not corresponding_reference_poses:
            raise ValueError(
                "Surface tasks require explicitly corresponding source/robot reference poses"
            )
        if not bindings or len(bindings) > 16:
            raise ValueError(
                "Surface enhancement requires between 1 and 16 reliable bindings"
            )
        if not np.isfinite(weight) or not 0 < weight <= 0.1:
            raise ValueError("Auxiliary surface weight must be in (0, 0.1]")
        self.bindings = tuple(bindings)
        self.kinematics = SurfaceKinematics(model, bindings)
        reference = np.asarray(reference_vertices_world, dtype=float)
        motion = np.asarray(motion_vertices_world, dtype=float)
        faces = np.asarray(faces, dtype=int)
        if (
            reference.ndim != 2
            or reference.shape[1] != 3
            or motion.ndim != 3
            or motion.shape[1:] != reference.shape
            or faces.ndim != 2
            or faces.shape[1] != 3
            or not len(faces)
            or np.min(faces) < 0
            or np.max(faces) >= len(reference)
        ):
            raise ValueError(
                "Reliable, topologically identical reference and motion surfaces are required"
            )
        slots = []
        counts = {}
        for binding in bindings:
            if not 0 <= binding.face_id < len(faces):
                raise ValueError(
                    "Surface binding face_id is outside the source topology"
                )
            bary = np.asarray(binding.barycentric)
            normal = np.asarray(binding.local_normal)
            if (
                bary.shape != (3,)
                or not np.all(np.isfinite(bary))
                or np.any(bary < 0)
                or abs(bary.sum() - 1) > 1e-8
                or normal.shape != (3,)
                or not np.all(np.isfinite(normal))
                or np.linalg.norm(normal) < 1e-8
                or np.shape(binding.local_point) != (3,)
                or not np.all(np.isfinite(binding.local_point))
            ):
                raise ValueError(
                    "Invalid barycentric binding or robot local point/normal"
                )
            model.geom(binding.robot_geom)  # Resolve names now, not during solving.
            counts[binding.part] = counts.get(binding.part, 0) + 1
            if counts[binding.part] > 4:
                raise ValueError("At most four auxiliary surface points per part")
            slots.append(faces[binding.face_id])
        reference_frames, valid = _face_frames(reference[slots])
        if not np.all(valid):
            raise ValueError("Degenerate reference surface triangle")
        frames, valid = _face_frames(motion[:, slots])
        normals = np.array(
            [
                self.kinematics.at(robot_reference_qpos, k)[1]
                for k in range(len(bindings))
            ]
        )
        relative = frames @ np.swapaxes(reference_frames, -1, -2)
        self.targets = np.einsum("tkij,kj->tki", relative, normals)
        self.valid = valid
        self.source_points = np.einsum(
            "tkij,ki->tkj",
            motion[:, slots],
            np.array([b.barycentric for b in bindings]),
        )
        self.weights = np.array([weight / counts[b.part] for b in bindings])
        self.frame_start = 0
        self.diagnostics = {
            "kind": "engineering normal-only auxiliary surface task",
            "coordinate_system": "shared metric world; no normalization/scaling",
            "sample_count": len(bindings),
            "part_counts": counts,
            "total_part_weight": weight,
            "position_weight": 0.0,
            "invalid_frame_slots": np.argwhere(~valid).tolist(),
            "twist_constrained": False,
        }

    def rows(self, pose, frame, dofs, excluded_bodies=()):
        """Position/normal FK shares one cache; only normal residuals are used."""
        for index, binding in enumerate(self.bindings):
            gid = self.kinematics.model.geom(binding.robot_geom).id
            if (
                int(self.kinematics.model.geom_bodyid[gid]) in excluded_bodies
                or not self.valid[frame, index]
            ):
                continue
            _point, normal, _jp, jn = self.kinematics.at(pose, index)
            yield self.weights[index], normal - self.targets[frame, index], jn[:, dofs]

    def score(self, pose, frame, excluded_bodies=()):
        return sum(
            weight * float(residual @ residual)
            for weight, residual, _ in self.rows(
                pose, frame, np.arange(self.kinematics.model.nv), excluded_bodies
            )
        )
