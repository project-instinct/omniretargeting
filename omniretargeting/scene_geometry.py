"""Shared scene queries. Positive clearance is free; negative is penetration.

Closed, consistently wound, non-self-intersecting components use winding-number
containment. Open components only have a *local* oriented distance, backed by
fixed contact/approach evidence. Unknown sides are NaN, never free space.
Robot surface sampling is deterministic and approximate, including at validation
resolution; it is not a collision-free certificate.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from itertools import product
from threading import RLock

import igl
import mujoco
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import trimesh
from scipy.spatial import ConvexHull, QhullError, cKDTree


def collision_geom_ids(model, geom_groups=None, body_tokens=None):
    """Select robot geoms; a model's world floor is environment, not robot."""
    groups = None if geom_groups is None else set(map(int, geom_groups))
    tokens = None if body_tokens is None else tuple(str(t).lower() for t in body_tokens)
    selected = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] == 0:
            continue
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


# Content-addressed, process-local shells also survive equivalent model reloads.
# Bound retained arrays without keeping any MjModel alive. No disk cache.
_SHELLS = OrderedDict()
_SHELLS_LOCK = RLock()
_SHELLS_MAX_BYTES = 128 * 1024 * 1024
_SHELLS_MAX_ENTRIES = 256
_SHELLS_BYTES = 0


def _geom_surface_key(model, geom_id, spacing):
    """All sampler inputs, read afresh so in-place model edits invalidate hits.

    Positions, rotations and mesh faces are not read by the local-shell sampler.
    Mesh dataid/vertex address/count select the compiled vertices; include both
    that selection and its float64 contents, in their original order.
    """
    kind = int(model.geom_type[geom_id])
    digest = hashlib.sha256()
    digest.update(np.asarray([kind], dtype=np.int64).tobytes())
    digest.update(np.asarray(model.geom_size[geom_id], dtype=float).tobytes())
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(model.geom_dataid[geom_id])
        start, n = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        digest.update(np.asarray([mid, start, n], dtype=np.int64).tobytes())
        digest.update(np.asarray(model.mesh_vert[start:start + n], dtype=float).tobytes())
    # Preserve parameter dtype/bytes: e.g. longdouble must not round to float64.
    parameter = None
    if spacing is not None:
        value = np.asarray(spacing)
        parameter = (value.dtype.str, value.shape, value.tobytes())
    return geom_id, parameter, digest.digest()


def geom_surface_offsets(model, geom_id, spacing=None):
    """Geom-frame collision shell, with box edges and capsule thickness.

    Meshes use the convex hull of compiled MuJoCo vertices (MuJoCo's mesh
    collision representation). No vertex stride removes extremities. ``spacing``
    bounds the sampling grid spacing; callers enforce query budgets explicitly.
    Identical model geometry and parameters reuse a process-local shell;
    callers receive independent writable copies, never the cached array.
    """
    geom_id = int(geom_id)
    key = _geom_surface_key(model, geom_id, spacing)
    global _SHELLS_BYTES
    with _SHELLS_LOCK:
        cached = _SHELLS.get(key)
        if cached is None:
            cached = _geom_surface_offsets(model, geom_id, spacing)
            cached.setflags(write=False)
            if cached.nbytes <= _SHELLS_MAX_BYTES:
                while _SHELLS and (
                    _SHELLS_BYTES + cached.nbytes > _SHELLS_MAX_BYTES
                    or len(_SHELLS) >= _SHELLS_MAX_ENTRIES
                ):
                    _SHELLS_BYTES -= _SHELLS.popitem(last=False)[1].nbytes
                _SHELLS[key] = cached
                _SHELLS_BYTES += cached.nbytes
        else:
            _SHELLS.move_to_end(key)
        return cached.copy()


def _geom_surface_offsets(model, geom_id, spacing=None):
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
        corners = vertices[hull.simplices]
        divisions = np.array([
            count(max(np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b)), 3) - 1
            for a, b, c in corners
        ])
        # Faces with equal subdivision share one barycentric grid; each point
        # is still a + (i (b - a) + j (c - a)) / divisions.
        for size in np.unique(divisions):
            a, b, c = (corners[divisions == size, k][:, None] for k in range(3))
            i, j = np.array([(i, j) for i in range(size + 1) for j in range(size + 1 - i)]).T
            grid = a + (i[None, :, None] * (b - a) + j[None, :, None] * (c - a)) / size
            points.append(grid.reshape(-1, 3))
        return np.unique(np.vstack(points), axis=0)
    # Infinite robot planes and heightfields have no finite sample shell.
    raise ValueError(f"Unsupported moving collision geom {geom_id}: type {kind}")


def _boundary_loops(mesh):
    """Boundary loops of the triangle chain and a cone cap closing each.

    Edges whose directed uses do not cancel form the chain's boundary;
    non-manifold seams whose orientations cancel are not boundary. Returns
    ``(loops, cap_vertices, cap_faces)``: per connected loop its bounding
    sphere (centre, radius) and length, and fan triangles from each centre
    that cancel the boundary, so that the chain plus caps is closed.
    """
    faces = np.asarray(mesh.faces)
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges, inverse = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True)
    net = np.bincount(inverse.ravel(), weights=np.where(directed[:, 0] < directed[:, 1], 1., -1.),
                      minlength=len(edges)).round().astype(int)
    edges, net = edges[net != 0], net[net != 0]
    vertices = np.asarray(mesh.vertices, dtype=float)
    if not len(edges):
        return [], np.zeros((0, 3)), np.zeros((0, 3), int)
    _, labels = connected_components(
        coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(len(vertices),) * 2),
        directed=False)
    loops, cap_vertices, cap_faces = [], [], []
    for label in np.unique(labels[edges[:, 0]]):
        selected = labels[edges[:, 0]] == label
        ends = vertices[np.unique(edges[selected])]
        center = 0.5 * (ends.min(axis=0) + ends.max(axis=0))
        length = float((np.linalg.norm(vertices[edges[selected, 0]] - vertices[edges[selected, 1]], axis=1)
                        * np.abs(net[selected])).sum())
        loops.append((center, float(np.linalg.norm(ends - center, axis=1).max()), length))
        # A chain edge lo->hi (net > 0) is cancelled by the cap edge hi->lo.
        apex = len(vertices) + len(cap_vertices)
        cap_vertices.append(center)
        for (lo, hi), count in zip(edges[selected], net[selected]):
            triangle = [apex, hi, lo] if count > 0 else [apex, lo, hi]
            cap_faces.extend([triangle] * abs(int(count)))
    return loops, np.asarray(cap_vertices), np.asarray(cap_faces, dtype=int)


# Margin of the convex-hull outside test, in mesh units (qhull rounding).
HULL_TOLERANCE = 1e-9
# A generalized winding number this close to 0 or 1 is decisive (the same
# tolerance as for closed and nearly closed components).
WINDING_TOLERANCE = 0.01
# Winding grid of a bounded object's open component: cells along its longest
# extent, and the depth (in cells) beyond which evidence-inside points are vetted.
WINDING_GRID_CELLS = 64
WINDING_GRID_DEPTH = 2.0


class SceneQuery:
    """Immutable geometry/evidence shared by solving and acceptance.

    ``evidence_points/normals`` describe a known local free side, independent of
    the robot being repaired. Evidence is not propagated across unobserved mesh
    regions. Conflicting evidence, holes and mixed winding remain uncertain.
    """

    def __init__(
        self, mesh, evidence_points=None, evidence_normals=None, evidence_radius=0.15,
        *, full_geometry=True, enclosed_free_space=False, closed_sign_method='winding',
        bounded_object=False,
    ):
        if mesh is None or not len(mesh.faces) or not len(mesh.vertices):
            raise ValueError("Scene collision query requires nonempty mesh geometry")
        if not np.all(np.isfinite(mesh.vertices)):
            raise ValueError("Scene collision mesh contains nonfinite vertices")
        self.mesh = mesh
        self.full_geometry = bool(full_geometry)
        self.enclosed_free_space = bool(enclosed_free_space)
        if closed_sign_method not in ('winding', 'pseudonormal'):
            raise ValueError('closed_sign_method must be winding or pseudonormal')
        self.closed_sign_method = closed_sign_method
        self._closed_normals = []
        if self.enclosed_free_space and not (mesh.is_watertight and mesh.is_winding_consistent):
            raise ValueError('Enclosed free space requires an oriented watertight boundary')
        self.geometry_backends = {}
        self.query_points = 0
        self.query_calls = 0
        self.component_query_points = 0
        self.component_query_calls = 0
        self.components = []
        self._component_query_data = []
        self._component_features = []
        self._completions = []
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
        # A bounded object is the closure of its recorded surface: the mesh
        # plus caps spanning its boundary loops, all inside the convex hull of
        # its vertices, the domain of the winding-number segmentation (Jacobson
        # et al. 2013). Free-side evidence extends an open patch's local side;
        # it cannot put a point inside where no closure can be: outside that
        # hull, or where the patch's generalized winding number is decisively
        # zero (nothing encloses the point, e.g. the free space in a chair's
        # concavity). Opt-in: terrain continues below its lowest vertex, and
        # enclosed free space has its solid outside the room shell. A flat
        # mesh has no hull; an open sheet has |w| near 1/2 beside it.
        self.bounded_object = bool(bounded_object) and not self.enclosed_free_space
        self._winding_grids = {}
        self._hull = None
        if self.bounded_object:
            try:
                self._hull = ConvexHull(np.asarray(mesh.vertices, dtype=float)).equations
            except QhullError:
                self._hull = None
            self._hull_bounds = np.asarray(mesh.vertices, dtype=float).min(0), np.asarray(mesh.vertices, dtype=float).max(0)
        # Preserve all faces, including singleton/open components.
        groups = trimesh.graph.connected_components(
            mesh.face_adjacency, nodes=np.arange(len(mesh.faces)), min_len=1
        )
        if self.enclosed_free_space:
            # An enclosing room and its oppositely oriented obstacle shells
            # describe ONE free volume. Treating each shell as a solid makes
            # every point inside the room appear deeply penetrating. Keep all
            # faces and sum their winding, including the holes/obstacles.
            groups = [np.arange(len(mesh.faces))]
        for faces in groups:
            part = mesh.submesh([faces], append=True, repair=False)
            # Derive directions from original triangles, independent of stale
            # normals supplied by a file or cached across mesh transforms.
            normals, valid = trimesh.triangles.normals(part.triangles)
            if valid.all():
                part.face_normals = normals
            closed = bool(
                part.is_watertight
                and part.is_winding_consistent
                and abs(part.volume) > 1e-12
            )
            tree = igl.AABB()
            tree.init(np.asarray(part.vertices), np.asarray(part.faces))
            self.components.append((part, np.asarray(faces, dtype=int), closed, tree))
            from .closed_surface_normals import ClosedSurfaceNormals
            self._closed_normals.append(
                ClosedSurfaceNormals(part) if closed and closed_sign_method == 'pseudonormal' else None)
            from .open_surface_features import OpenSurfaceFeatures
            self._component_features.append(None if closed else OpenSurfaceFeatures(part))
            # Nearly closed: capping every boundary loop adds at most 1% of the
            # surface (the winding tolerance); a cone cap has area at most
            # length * radius / 2. Fragments and open sheets, whose caps are as
            # large as themselves, keep only evidence-based sides.
            completion = None
            if not closed:
                loops, cap_vertices, cap_faces = _boundary_loops(part)
                if sum(l * r / 2 for _, r, l in loops) <= 0.01 * part.area:
                    completion = (loops, np.vstack([np.asarray(part.vertices, float), cap_vertices]),
                                  np.vstack([np.asarray(part.faces), cap_faces]).astype(np.int64))
            self._completions.append(completion)
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
        # Such a component marks EVERY query point uncertain, even far away.
        # Geometry and evidence are immutable, so centre culling can never
        # discard a geom in this case; its prepass would only repeat queries.
        self.has_unsigned_only_components = any(
            self._unsigned_only(cid) for cid in range(len(self.components)))

    def hull_margin(self, points, beyond=np.inf):
        """Lower bound of each point's distance outside the hull (``bounded_object``).

        Positive only outside the hull, where it bounds the distance to the
        object from below; -inf without a certificate. Points farther than
        ``beyond`` outside the vertices' bounding box keep that box distance
        (also a lower bound); only the rest evaluate every hull plane.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if self._hull is None:
            return np.full(len(points), -np.inf)
        lower, upper = self._hull_bounds
        margin = np.linalg.norm(np.maximum(np.maximum(lower - points, points - upper), 0.), axis=1)
        near = np.flatnonzero(margin <= beyond)
        if len(near):
            margin[near] = (points[near] @ self._hull[:, :3].T + self._hull[:, 3]).max(axis=1)
        return margin

    def _winding_free(self, cid, points, dist, enclosed):
        """Indices of ``enclosed`` points no closure of component ``cid`` encloses.

        Evidence decides near the surface. Farther than WINDING_GRID_DEPTH grid
        cells, where the generalized winding number is harmonic across the
        point's cell, all eight cell corners of a once-computed grid must be
        decisively zero. One exact fast-winding call per query cost 16 ms for a
        32k-face object; the grid costs that once.
        """
        grid = self._winding_grids.get(cid)
        if grid is None:
            vertices, triangles, bounds, _ = self._component_query_data[cid]
            spacing = max(float(np.max(bounds[1] - bounds[0])), 1e-9) / WINDING_GRID_CELLS
            lower = bounds[0] - spacing
            counts = np.floor((bounds[1] + spacing - lower) / spacing).astype(int) + 2
            axes = [lower[axis] + spacing * np.arange(counts[axis]) for axis in range(3)]
            nodes = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
            winding = np.abs(igl.fast_winding_number(vertices, triangles, nodes)).reshape(counts)
            grid = self._winding_grids[cid] = (lower, spacing, winding)
        lower, spacing, winding = grid
        candidates = np.flatnonzero(enclosed & (dist >= WINDING_GRID_DEPTH * spacing))
        if not len(candidates):
            return candidates
        base = np.floor((points[candidates] - lower) / spacing).astype(int)
        inside = np.all((base >= 0) & (base < np.array(winding.shape) - 1), axis=1)
        free = np.zeros(len(candidates), bool)
        cells = base[inside]
        grid_free = np.ones(len(cells), bool)
        for corner in product((0, 1), repeat=3):
            index = cells + corner
            grid_free &= winding[index[:, 0], index[:, 1], index[:, 2]] < WINDING_TOLERANCE
        free[inside] = grid_free
        # Beyond the grid (e.g. a small scan fragment nearest to a far point),
        # evaluate the winding number directly: such points are rare.
        if not inside.all():
            vertices, triangles, _, _ = self._component_query_data[cid]
            outside = candidates[~inside]
            free[~inside] = np.abs(igl.fast_winding_number(vertices, triangles, points[outside])) < WINDING_TOLERANCE
        return candidates[free]

    def outside_hull(self, points):
        """Points strictly outside the object's convex hull (``bounded_object``)."""
        return self.hull_classification(points, HULL_TOLERANCE)[:, 0]

    def hull_classification(self, points, *thresholds):
        """Exact threshold decisions with certified cell reuse (no margin approximation)."""
        from .convex_hull_scope import ConvexHullScope

        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if self._hull is None:
            return np.zeros((len(points), len(thresholds)), bool)
        scope = getattr(self, '_hull_scope', None)
        if scope is None:
            scope = self._hull_scope = ConvexHullScope(self._hull, self._hull_bounds)
        return scope.classify(points, thresholds, self.hull_margin)

    def _unsigned_only(self, cid):
        """Open, without evidence and not nearly closed: unknown everywhere."""
        return (not self.components[cid][2] and not self._component_query_data[cid][3]
                and self._completions[cid] is None)

    def _decisive_winding(self, cid, points, dist):
        """Points on a definite side of every completion of this component.

        A nearly closed component plus a cone cap on each boundary loop is a
        closed chain with an integer winding number off its surface
        (generalized winding numbers, Jacobson et al. 2013). Any other cap
        inside a loop's bounding sphere differs by a closed surface there, so
        outside all spheres the side does not depend on the cap. Decided:
        outside the spheres, off the surface, and within 0.01 of 0 or 1 (the
        closed-component tolerance, with the same exact fallback).
        """
        completion = self._completions[cid]
        decided = np.zeros(len(points), bool)
        if completion is None or not len(points):
            return decided, decided.copy()
        loops, vertices, triangles = completion
        candidate = dist >= 1e-8
        for center, radius, _ in loops:
            candidate &= np.linalg.norm(points - center, axis=1) > radius
        winding = np.zeros(len(points))
        if np.any(candidate):
            winding[candidate] = np.abs(igl.fast_winding_number(vertices, triangles, points[candidate]))
        ambiguous = candidate & (winding >= 0.01) & (np.abs(winding - 1) >= 0.01)
        if np.any(ambiguous):
            winding[ambiguous] = np.abs(igl.winding_number(vertices, triangles, points[ambiguous]))
        decided = candidate & ((winding < 0.01) | (np.abs(winding - 1) < 0.01))
        return decided, winding > 0.5

    def nearly_closed(self, cid):
        return self._completions[cid] is not None

    def winding_side(self, points, cid):
        """``query`` fields of one nearly closed component from its winding
        number alone; points it cannot decide are uncertain (NaN)."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        _, faces, _, tree = self.components[cid]
        vertices, triangles, _, _ = self._component_query_data[cid]
        d2, tid, near = tree.squared_distance(vertices, triangles, points)
        dist = np.sqrt(np.maximum(d2, 0))
        decided, inside = self._decisive_winding(cid, points, dist)
        sign = np.where(inside, -1.0, 1.0)
        offset = points - near
        directions = sign[:, None] * offset / np.maximum(dist[:, None], 1e-9)
        directions[~decided] = 0.0
        return {"clearance": np.where(decided, sign * dist, np.nan), "directions": directions,
                "surface_points": near, "surface_ids": faces[tid],
                "obstacle_ids": np.full(len(points), cid), "confidence": np.where(decided, 0.5, 0.0),
                "uncertain": ~decided, "unsigned_distance": dist}

    @classmethod
    def from_tasks(cls, mesh, tasks, **kwargs):
        points, normals = [], []
        for task in tasks.values():
            mask = np.asarray(task.contact_mask) & (
                np.asarray(task.contact_confidence) >= 0.5
            )
            environment_ids = getattr(task, "environment_ids", None)
            if environment_ids is not None:
                mask &= np.asarray(environment_ids) == 0
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
            "enclosed_free_space": self.enclosed_free_space,
            "closed_sign_method": self.closed_sign_method,
            "unsigned_only_components": sum(
                self._unsigned_only(cid) for cid in range(len(self.components))),
            "nearly_closed_components": sum(
                completion is not None for completion in self._completions),
            "component_query_points": self.component_query_points,
            "component_query_calls": self.component_query_calls,
            "geometry_queries": [backend.describe() for backend in self.geometry_backends.values()],
        }

    def query(self, points, *, component_ids=None, outside_hull=None):
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
        selected_components = None if component_ids is None else set(map(int, component_ids))
        for cid, (mesh, faces, closed, tree) in enumerate(self.components):
            if selected_components is not None and cid not in selected_components:
                continue
            if n == 0:
                break
            vertices, triangles, bounds, has_evidence = self._component_query_data[cid]
            if self._unsigned_only(cid):
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
            if closed and self._closed_normals[cid] is not None:
                normals, reliable = self._closed_normals[cid].at(tid, near)
                normals *= (1 if mesh.volume > 0 else -1)
                if self.enclosed_free_space:
                    normals *= -1
                sign = np.where(np.sum(offset * normals, axis=1) >= 0, 1., -1.)
            elif closed:
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
                # The fast multipole approximation can be inconclusive very
                # close to a thin solid. Resolve just those samples by exact
                # solid-angle winding, rather than loosening the reliability
                # threshold or aborting an otherwise valid contact trajectory.
                ambiguous = ((winding >= 0.01) & (np.abs(winding - 1) >= 0.01)
                             & (dist >= 1e-8))
                if np.any(ambiguous):
                    winding[ambiguous] = np.abs(igl.winding_number(
                        vertices, triangles, points[ambiguous]))
                reliable = (
                    (winding < 0.01) | (np.abs(winding - 1) < 0.01) | (dist < 1e-8)
                )
                sign = np.where(winding > 0.5, -1.0, 1.0)
                normals = normals * (1 if mesh.volume > 0 else -1)
                if self.enclosed_free_space:
                    sign = -sign
                    normals = -normals
            else:
                reliable[:] = False
                if self.evidence_tree is not None:
                    # The bound only prunes: a missing neighbor has infinite
                    # distance and is never eligible; its index is clamped.
                    distances, indices = self.evidence_tree.query(
                        near, k=min(4, len(self.evidence_points)),
                        distance_upper_bound=self.evidence_radius * (1 + 1e-9) + 1e-12,
                    )
                    distances = np.asarray(distances).reshape(n, -1)
                    indices = np.minimum(np.asarray(indices).reshape(n, -1),
                                         len(self.evidence_points) - 1)
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
                reliable = self._component_features[cid].reliable_witnesses(
                    points, near, tid, reliable, self.evidence_tree,
                    self.evidence_points, self.evidence_normals, self.evidence_radius,
                )
                # A nearly closed component (cracks, non-manifold seams) has a
                # decisive generalized winding number away from its defects
                # (Jacobson et al. 2013). Where the evidence votes leave a point
                # unknown (they are unstable deep inside such a solid), it
                # states the side; indecisive points (holes) stay unknown.
                gap = np.flatnonzero(~reliable)
                if len(gap):
                    decided, inside = self._decisive_winding(cid, points[gap], dist[gap])
                    sign[gap[decided]] = np.where(inside[decided], -1.0, 1.0)
                    reliable[gap[decided]] = True
                if outside_hull is None:
                    outside_hull = self.outside_hull(points)
                outside = outside_hull
                sign[outside] = 1.0
                reliable[outside] = True
                if self.bounded_object:
                    sign[self._winding_free(cid, points, dist, reliable & (sign < 0))] = 1.0
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
            if self.query.has_unsigned_only_components:
                # The unchanged sample sets include every geom centre. Keep
                # their ordering and all uncertainty/full-geometry checks.
                near = np.ones(len(self.geom_ids), dtype=bool)
            else:
                centers = self.query.query(data.geom_xpos[self.geom_ids])
                # Only confirmed-free, distant centres permit culling.
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
                # A confirmed free centre farther than the geom's bounding
                # radius plus margin cannot produce a rigid contact either.
                rows = self.geometry_backend.query(
                    data, self.margin, active_geom_ids=self.geom_ids[near]
                )
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
