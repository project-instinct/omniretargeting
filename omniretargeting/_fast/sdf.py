"""Precomputed signed clearance grids for fast trajectory optimization.

Clearance values and exit directions come from the existing ``SceneQuery``
on the original triangles, sampled once on a regular grid and interpolated
trilinearly during optimization. The grid is an optimization proxy only:
final acceptance still audits the original triangles.
"""

import hashlib
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

CACHE_DIR = Path(
    os.environ.get(
        "OMNIRETARGETING_SDF_CACHE",
        Path.home() / ".cache" / "omniretargeting" / "sdf",
    )
)
QUERY_BATCH = 500_000


class GridSDF:
    """Trilinear clearance and gradient on an axis-aligned grid.

    Points outside the grid, or at nodes whose original query was unreliable,
    return NaN clearance: they are unknown, never assumed free.
    """

    def __init__(self, lower, spacing, clearance, directions):
        self.lower = np.asarray(lower, dtype=float)
        self.spacing = float(spacing)
        self.clearance = np.asarray(clearance, dtype=np.float32)
        self.directions = np.asarray(directions, dtype=np.float32)
        self.shape = np.array(self.clearance.shape)

    @classmethod
    def build(cls, make_query, lower, upper, spacing, key=None):
        """Sample ``make_query()(points) -> dict(clearance, directions)``.

        The (costly) query is only constructed when the cache misses.
        """
        lower = np.asarray(lower, dtype=float)
        counts = np.maximum(np.ceil((np.asarray(upper) - lower) / spacing), 1)
        counts = counts.astype(int) + 1
        path = None
        if key is not None:
            digest = hashlib.sha256(
                repr((key, lower.round(6).tolist(), counts.tolist(), spacing)).encode()
            ).hexdigest()[:24]
            path = CACHE_DIR / f"{digest}.npz"
            if path.exists():
                with np.load(path) as saved:
                    return cls(saved["lower"], float(saved["spacing"]),
                               saved["clearance"], saved["directions"])
        query = make_query()
        axes = [lower[i] + spacing * np.arange(counts[i]) for i in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
        clearance = np.empty(len(grid), dtype=np.float32)
        directions = np.empty((len(grid), 3), dtype=np.float32)
        for first in range(0, len(grid), QUERY_BATCH):
            measured = query(grid[first : first + QUERY_BATCH])
            value = np.asarray(measured["clearance"], dtype=float)
            unreliable = np.asarray(measured.get("uncertain", False), dtype=bool)
            value = np.where(unreliable, np.nan, value)
            clearance[first : first + QUERY_BATCH] = value
            directions[first : first + QUERY_BATCH] = measured["directions"]
        clearance = clearance.reshape(*counts)
        directions = directions.reshape(*counts, 3)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, lower=lower, spacing=spacing,
                     clearance=clearance, directions=directions)
        return cls(lower, spacing, clearance, directions)

    def query(self, points, gradient=True):
        """Clearance (...,) and, optionally, unit gradient (..., 3) at points."""
        points = np.asarray(points, dtype=float)
        flat = points.reshape(-1, 3)
        index = ((flat - self.lower) / self.spacing).T
        inside = np.all((index >= 0) & (index <= (self.shape - 1)[:, None]), axis=0)
        value = map_coordinates(self.clearance, index, order=1, mode="nearest", prefilter=False)
        value = np.where(inside, value, np.nan).reshape(points.shape[:-1])
        if not gradient:
            return value
        return value, self.gradient(points)

    def value_gradient(self, points):
        """Exact gradient (..., 3) of the trilinear clearance ``query`` returns.

        The derivative of the interpolated scalar itself, so a Gauss-Newton
        model of a clearance residual predicts its actual change; zero where
        a corner is unknown or outside the grid.
        """
        points = np.asarray(points, dtype=float)
        flat = points.reshape(-1, 3)
        index = (flat - self.lower) / self.spacing
        base = np.clip(np.floor(index).astype(int), 0, self.shape - 2)
        t = np.clip(index - base, 0.0, 1.0)
        gradient = np.zeros_like(flat)
        for corner in range(8):
            offset = np.array([(corner >> axis) & 1 for axis in range(3)])
            value = self.clearance[tuple((base + offset).T)].astype(float)
            weights = np.where(offset, t, 1 - t)
            for axis in range(3):
                others = np.prod(np.delete(weights, axis, axis=1), axis=1)
                gradient[:, axis] += (1 if offset[axis] else -1) * others * value
        gradient /= self.spacing
        inside = np.all((index >= 0) & (index <= self.shape - 1), axis=1)
        gradient[~inside | ~np.isfinite(gradient).all(axis=1)] = 0.0
        return gradient.reshape(points.shape)

    def gradient(self, points):
        """Unit clearance gradient (..., 3) from interpolated exit directions."""
        points = np.asarray(points, dtype=float)
        index = ((points.reshape(-1, 3) - self.lower) / self.spacing).T
        direction = np.stack(
            [
                map_coordinates(self.directions[..., axis], index, order=1,
                                mode="nearest", prefilter=False)
                for axis in range(3)
            ],
            -1,
        )
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        direction = np.divide(direction, norm, out=np.zeros_like(direction), where=norm > 1e-9)
        return direction.reshape(points.shape)


def mesh_key(mesh, *extra):
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(mesh.vertices, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(mesh.faces, dtype=np.int64).tobytes())
    return (digest.hexdigest(), *extra)


def scene_query_for(mesh, *, enclosed_free_space, evidence=None, bounded_object=False):
    """Existing original-triangle query; pseudonormals when the mesh allows.

    ``evidence`` (points, normals, radius in mesh coordinates) signs open
    surfaces near observed free sides, as the moving-object query does.
    """
    from ..scene_geometry import SceneQuery

    extra = {"bounded_object": bounded_object}
    if evidence is not None:
        extra.update(evidence_points=evidence["points"], evidence_normals=evidence["normals"],
                     evidence_radius=evidence["radius"])
    try:
        return SceneQuery(
            mesh,
            enclosed_free_space=enclosed_free_space,
            closed_sign_method="pseudonormal",
            **extra,
        ), "pseudonormal"
    except ValueError as error:
        if "Pseudonormals require nondegenerate triangles" not in str(error):
            raise
        return SceneQuery(mesh, enclosed_free_space=enclosed_free_space, **extra), "winding"


def static_scene_sdf(mesh, region_lower, region_upper, spacing, *, enclosed_free_space):
    key = mesh_key(mesh, "static", bool(enclosed_free_space), "decisive_winding")

    def make_query():
        return scene_query_for(mesh, enclosed_free_space=enclosed_free_space)[0].query

    return GridSDF.build(make_query, region_lower, region_upper, spacing, key=key)


def nearest_component_query(mesh, evidence):
    """Sign each point by its nearest original component only.

    The moving-object repair query does the same: closed components are signed
    everywhere, an open component only within its source free-side evidence,
    and a point whose nearest component is open without evidence stays unknown.
    A remote unobserved fragment therefore no longer makes every point unknown.
    A nearly closed component also marks every point it decides inside.
    """
    from ..surface_observation import ContactSurface

    query = scene_query_for(mesh, enclosed_free_space=False, evidence=evidence, bounded_object=True)[0]
    surface = ContactSurface(mesh, 1, name="object")
    owner = np.empty(len(surface.faces), dtype=int)
    for cid, (_, faces, _, _) in enumerate(query.components):
        owner[faces] = cid
    closed = {cid for cid, part in enumerate(query.components) if part[2]}
    observed = {cid for cid, part in enumerate(query.components)
                if not part[2] and query._component_query_data[cid][3]}
    nearly = [cid for cid, part in enumerate(query.components)
              if not part[2] and query.nearly_closed(cid) and abs(part[0].volume) > 1e-12]

    def run(points):
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        clearance = np.full(len(points), np.nan)
        directions = np.zeros((len(points), 3))
        uncertain = np.ones(len(points), dtype=bool)
        _, triangles, _ = surface.tree.squared_distance(surface.vertices, surface.faces, points)
        nearest = owner[triangles]
        scope = np.zeros(len(points), dtype=bool)
        if query.evidence_tree is not None:
            distance, _ = query.evidence_tree.query(points)
            scope = distance <= query.evidence_radius
        # Outside the object's hull a point is known outside at any distance.
        scope |= query.outside_hull(points)
        for cid in np.unique(nearest):
            selected = nearest == cid
            if cid in observed:
                selected &= scope
            elif cid not in closed:
                continue
            if not selected.any():
                continue
            measured = query.query(points[selected], component_ids=[int(cid)])
            clearance[selected] = measured["clearance"]
            directions[selected] = measured["directions"]
            uncertain[selected] = measured["uncertain"]
        # A nearly closed component encloses its volume whichever surface is
        # nearest: a point it decides inside penetrates it, as in the moving
        # query; the deeper decided answer wins, as in ``SceneQuery.query``.
        for cid in nearly:
            measured = query.winding_side(points, int(cid))
            inside = ~measured["uncertain"] & (measured["clearance"] < 0)
            take = inside & (uncertain | ~(clearance <= measured["clearance"]))
            clearance[take] = measured["clearance"][take]
            directions[take] = measured["directions"][take]
            uncertain[take] = False
        return {"clearance": clearance, "directions": directions, "uncertain": uncertain}

    return run


def object_sdf(mesh, spacing, margin, evidence=None):
    """Grid in the object's own coordinates, around its bounds.

    Open objects are signed only near their source free-side evidence; the
    rest of the grid stays unknown, never assumed free.
    """
    lower, upper = mesh.bounds[0] - margin, mesh.bounds[1] + margin

    def make_query():
        if evidence is not None:
            return nearest_component_query(mesh, evidence)
        return scene_query_for(mesh, enclosed_free_space=False)[0].query

    key = mesh_key(mesh, "object")
    if evidence is not None:
        digest = hashlib.sha256()
        for name in ("points", "normals"):
            digest.update(np.ascontiguousarray(evidence[name], dtype=np.float64).tobytes())
        digest.update(np.float64(evidence["radius"]).tobytes())
        # The query semantics are part of the cache identity.
        key = (*key, "evidence", "nearest_component", "decisive_winding", "bounded_object_winding", digest.hexdigest())
    return GridSDF.build(make_query, lower, upper, spacing, key=key)
