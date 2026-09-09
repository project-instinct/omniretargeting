"""Shared scene queries. Positive clearance is free; negative is penetration.

Closed, consistently wound, non-self-intersecting components use winding-number
containment. Open components only have a *local* oriented distance, backed by
fixed contact/approach evidence. Unknown sides are NaN, never free space.
Robot surface sampling is deterministic and approximate, including at validation
resolution; it is not a collision-free certificate.
"""

from __future__ import annotations

from itertools import product

import igl
import mujoco
import numpy as np
import trimesh
from scipy.spatial import ConvexHull, cKDTree


def collision_geom_ids(model, geom_groups=None, body_tokens=None):
    groups = None if geom_groups is None else set(map(int, geom_groups))
    tokens = None if body_tokens is None else tuple(str(t).lower() for t in body_tokens)
    selected = []
    for gid in range(model.ngeom):
        if groups is None:
            if not (model.geom_contype[gid] or model.geom_conaffinity[gid]):
                continue
        elif int(model.geom_group[gid]) not in groups:
            continue
        name = (
            mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])
            )
            or ""
        )
        if tokens is None or any(t in name.lower() for t in tokens):
            selected.append(gid)
    return np.asarray(selected, dtype=int)


def geom_surface_offsets(model, geom_id, spacing=None):
    """Geom-frame collision shell, with box edges and capsule thickness.

    Meshes use the convex hull of compiled MuJoCo vertices (MuJoCo's mesh
    collision representation). No vertex stride removes extremities. ``spacing``
    bounds the sampling grid spacing; callers enforce query budgets explicitly.
    """
    kind = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=float)

    def count(length, minimum=2):
        return max(minimum, int(np.ceil(length / spacing)) + 1) if spacing else minimum

    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        axes = [np.linspace(-s, s, count(2 * s, 3)) for s in size]
        points = []
        for axis in range(3):
            others = [a for a in range(3) if a != axis]
            for side in (-size[axis], size[axis]):
                for a, b in product(axes[others[0]], axes[others[1]]):
                    p = np.zeros(3)
                    p[axis], p[others[0]], p[others[1]] = side, a, b
                    points.append(p)
        return np.unique(points, axis=0)
    if kind in (
        mujoco.mjtGeom.mjGEOM_SPHERE,
        mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
    ):
        radius = float(
            max(size) if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID else size[0]
        )
        n = count(2 * np.pi * radius, 9) - 1
        n = int(np.ceil(n / 4)) * 4
        theta = np.arange(n) * 2 * np.pi / n
        circle = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(n)])
        if kind in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            scales = (
                size if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID else np.full(3, radius)
            )
            rings = [
                circle * np.sin(phi) + [0, 0, np.cos(phi)]
                for phi in np.linspace(0, np.pi, count(np.pi * radius, 5))
            ]
            return np.vstack(rings) * scales
        half = size[1]
        rings = [
            circle * radius + [0, 0, z]
            for z in np.linspace(-half, half, count(2 * half, 3))
        ]
        if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            for sign in (-1, 1):
                for phi in np.linspace(0, np.pi / 2, count(np.pi * radius / 2, 3))[1:]:
                    rings.append(
                        circle * radius * np.cos(phi)
                        + [0, 0, sign * (half + radius * np.sin(phi))]
                    )
        else:
            for sign in (-1, 1):
                for r in np.linspace(0, radius, count(radius, 2))[:-1]:
                    rings.append(circle * r + [0, 0, sign * half])
        return np.unique(np.vstack(rings), axis=0)
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(model.geom_dataid[geom_id])
        start, n = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        vertices = np.asarray(model.mesh_vert[start : start + n], dtype=float)
        hull = ConvexHull(vertices)
        points = [vertices[hull.vertices]]
        for face in hull.simplices:
            a, b, c = vertices[face]
            divisions = (
                count(
                    max(
                        np.linalg.norm(b - a),
                        np.linalg.norm(c - a),
                        np.linalg.norm(c - b),
                    ),
                    3,
                )
                - 1
            )
            points.append(
                np.asarray(
                    [
                        a + (i * (b - a) + j * (c - a)) / divisions
                        for i in range(divisions + 1)
                        for j in range(divisions + 1 - i)
                    ]
                )
            )
        return np.unique(np.vstack(points), axis=0)
    # Infinite robot planes and heightfields have no finite sample shell.
    raise ValueError(f"Unsupported moving collision geom {geom_id}: type {kind}")


class SceneQuery:
    """Immutable geometry/evidence shared by solving and acceptance.

    ``evidence_points/normals`` describe a known local free side, independent of
    the robot being repaired. Evidence is not propagated across unobserved mesh
    regions. Conflicting evidence, holes and mixed winding remain uncertain.
    """

    def __init__(
        self, mesh, evidence_points=None, evidence_normals=None, evidence_radius=0.15,
        *, full_geometry=True,
    ):
        if mesh is None or not len(mesh.faces) or not len(mesh.vertices):
            raise ValueError("Scene collision query requires nonempty mesh geometry")
        if not np.all(np.isfinite(mesh.vertices)):
            raise ValueError("Scene collision mesh contains nonfinite vertices")
        self.mesh = mesh
        self.full_geometry = bool(full_geometry)
        self.geometry_backends = {}
        self.query_points = 0
        self.query_calls = 0
        self.component_query_points = 0
        self.component_query_calls = 0
        self.components = []
        self._component_query_data = []
        self.evidence_radius = float(evidence_radius)
        points = np.asarray(
            evidence_points if evidence_points is not None else [], dtype=float
        ).reshape(-1, 3)
        normals = np.asarray(
            evidence_normals if evidence_normals is not None else [], dtype=float
        ).reshape(-1, 3)
        valid = np.all(np.isfinite(points), axis=1) & (
            np.linalg.norm(normals, axis=1) > 1e-8
        )
        self.evidence_points = points[valid].copy()
        self.evidence_normals = normals[valid].copy()
        self.evidence_normals /= np.maximum(
            np.linalg.norm(self.evidence_normals, axis=1, keepdims=True), 1e-9
        )
        self.evidence_tree = cKDTree(self.evidence_points) if np.any(valid) else None
        # Preserve all faces, including singleton/open components.
        groups = trimesh.graph.connected_components(
            mesh.face_adjacency, nodes=np.arange(len(mesh.faces)), min_len=1
        )
        for faces in groups:
            part = mesh.submesh([faces], append=True, repair=False)
            closed = bool(
                part.is_watertight
                and part.is_winding_consistent
                and abs(part.volume) > 1e-12
            )
            tree = igl.AABB()
            tree.init(np.asarray(part.vertices), np.asarray(part.faces))
            self.components.append((part, np.asarray(faces, dtype=int), closed, tree))
            vertices, triangles = np.asarray(part.vertices), np.asarray(part.faces)
            # Evidence can orient a nearest surface point only if an evidence
            # ball intersects this component. Check the actual triangles once;
            # the immutable contact evidence is shared across all poses.
            has_evidence = False
            if not closed and self.evidence_tree is not None:
                distance2, _, _ = tree.squared_distance(
                    vertices, triangles, self.evidence_points
                )
                has_evidence = bool(np.any(
                    ~np.isfinite(distance2)
                    | (np.sqrt(np.maximum(distance2, 0)) <= self.evidence_radius + 1e-12)
                ))
            self._component_query_data.append(
                (vertices, triangles, np.asarray(part.bounds), has_evidence)
            )

    @classmethod
    def from_tasks(cls, mesh, tasks, **kwargs):
        points, normals = [], []
        for task in tasks.values():
            mask = np.asarray(task.contact_mask) & (
                np.asarray(task.contact_confidence) >= 0.5
            )
            points.extend(np.asarray(task.surface_points)[mask])
            normals.extend(np.asarray(task.surface_normals)[mask])
        return cls(mesh, points, normals, **kwargs)

    def describe(self):
        return {
            "clearance_convention": "positive outside; negative inside; unknown sides are NaN",
            "closed_components": sum(int(part[2]) for part in self.components),
            "open_or_unreliable_components": sum(
                int(not part[2]) for part in self.components
            ),
            "local_evidence_points": len(self.evidence_points),
            "local_evidence_radius": self.evidence_radius,
            "backend": "libigl cached triangle AABB / closed-component winding number",
            "global_sdf_for_open_mesh": False,
            "full_geometry_enabled": self.full_geometry,
            "unsigned_only_components": sum(
                not component[2] and not data[3]
                for component, data in zip(self.components, self._component_query_data)
            ),
            "component_query_points": self.component_query_points,
            "component_query_calls": self.component_query_calls,
            "geometry_queries": [backend.describe() for backend in self.geometry_backends.values()],
        }

    def query(self, points):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        self.query_points += len(points)
        self.query_calls += 1
        n = len(points)
        best = np.full(n, np.inf)
        direction = np.zeros((n, 3))
        closest = np.zeros((n, 3))
        surface = np.full(n, -1, dtype=int)
        obstacle = np.full(n, -1, dtype=int)
        confidence = np.ones(n)
        uncertain = np.zeros(n, dtype=bool)
        unsigned = np.full(n, np.inf)
        for cid, (mesh, faces, closed, tree) in enumerate(self.components):
            if n == 0:
                break
            vertices, triangles, bounds, has_evidence = self._component_query_data[cid]
            if not closed and not has_evidence:
                # This component remains unknown at EVERY distance. It cannot
                # supply a signed constraint, but still supplies the nearest
                # unsigned surface/witness. Its AABB lower bound can only omit
                # queries that cannot change that nearest surface. Never apply
                # this pruning to a closed solid or an evidence-backed surface:
                # a distant surface there may represent deep penetration.
                uncertain[:] = True
                gap = np.maximum(np.maximum(bounds[0] - points, points - bounds[1]), 0)
                active = np.flatnonzero(
                    np.einsum("ij,ij->i", gap, gap) <= (unsigned + 1e-12) ** 2
                )
                if not len(active):
                    continue
                self.component_query_points += len(active)
                self.component_query_calls += 1
                d2, tid, near = tree.squared_distance(vertices, triangles, points[active])
                dist = np.sqrt(np.maximum(d2, 0))
                take = (dist < unsigned[active]) & ~np.isfinite(best[active])
                selected = active[take]
                closest[selected] = near[take]
                surface[selected] = faces[tid[take]]
                obstacle[selected] = cid
                unsigned[active] = np.minimum(unsigned[active], dist)
                continue
            self.component_query_points += n
            self.component_query_calls += 1
            d2, tid, near = tree.squared_distance(
                vertices, triangles, points
            )
            dist = np.sqrt(np.maximum(d2, 0))
            offset = points - near
            normals = np.asarray(mesh.face_normals)[tid]
            reliable = np.ones(n, dtype=bool)
            if closed:
                # Winding magnitude handles consistently reversed shells without
                # altering the source scene; ambiguous/nonmanifold results fail closed.
                winding = np.zeros(n)
                in_bounds = np.all(
                    (points >= bounds[0]) & (points <= bounds[1]), axis=1
                )
                if np.any(in_bounds):
                    winding[in_bounds] = np.abs(
                        igl.fast_winding_number(
                            vertices,
                            triangles,
                            points[in_bounds],
                        )
                    )
                reliable = (
                    (winding < 0.01) | (np.abs(winding - 1) < 0.01) | (dist < 1e-8)
                )
                sign = np.where(winding > 0.5, -1.0, 1.0)
                normals = normals * (1 if mesh.volume > 0 else -1)
            else:
                reliable[:] = False
                if self.evidence_tree is not None:
                    distances, indices = self.evidence_tree.query(
                        near, k=min(4, len(self.evidence_points))
                    )
                    distances = np.asarray(distances).reshape(n, -1)
                    indices = np.asarray(indices).reshape(n, -1)
                    alignment = np.einsum(
                        "nij,nj->ni", self.evidence_normals[indices], normals
                    )
                    eligible = (distances <= self.evidence_radius) & (
                        np.abs(alignment) >= 0.8
                    )
                    vote = np.sum(np.where(eligible, np.sign(alignment), 0), axis=1)
                    reliable = (np.sum(eligible, axis=1) > 0) & (
                        np.abs(vote) == np.sum(eligible, axis=1)
                    )
                    normals = normals * np.where(vote < 0, -1, 1)[:, None]
                sign = np.where(np.sum(offset * normals, axis=1) >= 0, 1.0, -1.0)
            signed = sign * dist
            # Unknown components cannot be discarded just because their nearest
            # surface is far away. Keep this flag even if another component is known.
            uncertain |= ~reliable
            take = (signed < best) & reliable
            grad = np.where(
                (dist > 1e-9)[:, None],
                sign[:, None] * offset / np.maximum(dist[:, None], 1e-9),
                normals,
            )
            best[take], direction[take], closest[take] = (
                signed[take],
                grad[take],
                near[take],
            )
            surface[take], obstacle[take] = faces[tid[take]], cid
            confidence[take] = 1.0 if closed else 0.5
            unknown_nearest = (~reliable) & (dist < unsigned)
            closest[unknown_nearest & ~np.isfinite(best)] = near[
                unknown_nearest & ~np.isfinite(best)
            ]
            surface[unknown_nearest & ~np.isfinite(best)] = faces[
                tid[unknown_nearest & ~np.isfinite(best)]
            ]
            obstacle[unknown_nearest & ~np.isfinite(best)] = cid
            unsigned = np.minimum(unsigned, dist)
        best[~np.isfinite(best)] = np.nan
        confidence[uncertain] = 0.0
        return {
            "clearance": best,
            "directions": direction,
            "surface_points": closest,
            "surface_ids": surface,
            "obstacle_ids": obstacle,
            "confidence": confidence,
            "uncertain": uncertain,
            "unsigned_distance": unsigned,
        }


class ScenePenetrationProbe:
    def __init__(
        self, model, scene_mesh, geom_ids, margin, *, query=None, spacing=None
    ):
        self.model, self.scene_mesh = model, scene_mesh
        self.geom_ids = np.asarray(geom_ids, dtype=int)
        if not len(self.geom_ids):
            raise ValueError("Robot scene collision query requires active collision geoms")
        self.body_ids = np.asarray(model.geom_bodyid[self.geom_ids], dtype=int)
        self.margin = float(margin)
        self.query = query if query is not None else SceneQuery(scene_mesh)
        self.offsets = [
            geom_surface_offsets(model, int(gid), spacing) for gid in self.geom_ids
        ]
        # Cover a scene wholly enclosing a collision shell, including its centre.
        self.offsets = [
            np.unique(np.vstack([offsets, np.zeros((1, 3))]), axis=0)
            for offsets in self.offsets
        ]
        self.bounding_radii = np.asarray(model.geom_rbound[self.geom_ids])
        self.geometry_backend = None
        if self.query.full_geometry:
            from .geometry_query import RigidGeometryQuery
            key = (id(model), tuple(map(int, self.geom_ids)))
            if key not in self.query.geometry_backends:
                self.query.geometry_backends[key] = RigidGeometryQuery(model, self.geom_ids, self.query)
            self.geometry_backend = self.query.geometry_backends[key]

    @property
    def query_work_bound(self):
        pairs = (len(self.geom_ids) * (len(self.geometry_backend.scene_ids) + len(self.geometry_backend.open_parts))
                 if self.geometry_backend is not None else 0)
        return len(self.geom_ids) + sum(map(len, self.offsets)) + pairs

    def world_points(self, data):
        points, owners = [], []
        for local, gid in enumerate(self.geom_ids):
            points.append(
                data.geom_xpos[gid]
                + self.offsets[local] @ data.geom_xmat[gid].reshape(3, 3).T
            )
            owners.append(np.full(len(self.offsets[local]), local, dtype=int))
        return (
            (np.vstack(points), np.concatenate(owners))
            if points
            else (np.empty((0, 3)), np.empty(0, dtype=int))
        )

    def probe_trajectory(self, trajectory):
        data = mujoco.MjData(self.model)
        frames = []
        for q in trajectory:
            data.qpos[:] = q
            mujoco.mj_forward(self.model, data)
            centers = self.query.query(data.geom_xpos[self.geom_ids])
            # Only *confirmed free* and sufficiently distant centres may be culled.
            near = (
                centers["uncertain"]
                | ~np.isfinite(centers["clearance"])
                | (centers["clearance"] <= self.bounding_radii + self.margin)
            )
            points, owners = [], []
            for local in np.flatnonzero(near):
                gid = self.geom_ids[local]
                points.append(
                    data.geom_xpos[gid]
                    + self.offsets[local] @ data.geom_xmat[gid].reshape(3, 3).T
                )
                owners.append(np.full(len(self.offsets[local]), local, dtype=int))
            points = np.vstack(points) if points else np.empty((0, 3))
            owners = np.concatenate(owners) if owners else np.empty(0, dtype=int)
            frame = self.query.query(points)
            frame.update(
                points=points,
                body_ids=self.body_ids[owners],
                geom_ids=self.geom_ids[owners],
            )
            if self.geometry_backend is not None:
                previous_queries = self.geometry_backend.pair_queries
                rows = self.geometry_backend.query(data, self.margin)
                self.query.query_points += self.geometry_backend.pair_queries - previous_queries
                if rows:
                    for key in frame:
                        frame[key] = np.concatenate([frame[key], np.asarray([row[key] for row in rows])])
            frames.append(frame)
        return frames


def measure_scene_penetration(
    model, trajectory, scene_mesh, config=None, *, query=None
):
    cfg = dict(config or {})
    ids = collision_geom_ids(model, cfg.get("geom_groups"), cfg.get("body_tokens"))
    probe = ScenePenetrationProbe(
        model,
        scene_mesh,
        ids,
        cfg.get("margin", 0.05),
        query=query,
        spacing=cfg.get("sample_spacing"),
    )
    frames = probe.probe_trajectory(trajectory)
    depths = np.zeros((len(frames), model.ngeom))
    unknown = np.zeros(len(frames), dtype=int)
    for f, frame in enumerate(frames):
        unknown[f] = np.count_nonzero(frame["uncertain"])
        for gid in np.unique(frame["geom_ids"]):
            valid = (frame["geom_ids"] == gid) & np.isfinite(frame["clearance"])
            depths[f, gid] = np.max(
                np.maximum(-frame["clearance"][valid], 0), initial=0
            )
    per_frame = np.max(depths, axis=1, initial=0)
    worst = (
        np.unravel_index(np.argmax(depths), depths.shape)
        if depths.size and np.max(depths) > 0
        else (-1, -1)
    )
    return {
        "max_depth": float(np.max(per_frame, initial=0)),
        "p95_depth": float(np.percentile(per_frame, 95)) if len(frames) else 0.0,
        "mean_depth": float(np.mean(per_frame[per_frame > 0]))
        if np.any(per_frame > 0)
        else 0.0,
        "frames_penetrating": int(np.count_nonzero(per_frame > 1e-6)),
        "num_frames": len(frames),
        "worst_frame": int(worst[0]),
        "worst_body": model.body(int(model.geom_bodyid[worst[1]])).name
        if worst[1] >= 0
        else "",
        "per_frame_depth": per_frame,
        "per_geom_depth": depths,
        "uncertain_samples": unknown,
        "geometry_method": "deterministic collision-shell samples + triangle closest point / winding containment",
    }
