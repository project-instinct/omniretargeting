"""Conservative scene-distance cones, used only to omit far CPU queries.

Every finite grid node is a signed-distance lower bound on original geometry.
The maximum of node_value - distance_to_node is still a lower bound, without
assuming trilinear interpolation is exact. Open/unknown geometry is ineligible.
No grid value is returned as a witness, depth, or Jacobian input.
"""

import time

import numpy as np

from .original_geometry_cpu import winding_library


class ClearanceBounds:
    def __init__(self, lower, spacing, values):
        self.lower = np.ascontiguousarray(lower, dtype=np.float64)
        self.spacing = float(spacing)
        self.values = np.ascontiguousarray(values, dtype=np.float64)
        self.shape = np.ascontiguousarray(values.shape, dtype=np.int64)
        self.library = winding_library()
        for array in (self.lower, self.values, self.shape):
            array.setflags(write=False)

    def query(self, points):
        points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
        result = np.empty(len(points))
        self.library.r4_grid_lower(len(points), points.ctypes.data, self.lower.ctypes.data,
                                  self.spacing, self.shape.ctypes.data, self.values.ctypes.data,
                                  result.ctypes.data)
        return result


def build_scene_bounds(geometry):
    if not geometry._static_bound_reuse or not geometry._static_cache:
        return None
    started = time.perf_counter()
    positions = np.array([cached[2] for cached in geometry._static_cache.values()])
    radii = np.array([geometry._static_radii[key[1]] for key in geometry._static_cache])
    lower = np.min(positions - radii[:, None], axis=0) - .04
    upper = np.max(positions + radii[:, None], axis=0) + .04
    extent = upper - lower
    spacing = max(.02, float((np.prod(extent) / 1_000_000) ** (1 / 3)))
    shape = np.maximum(np.ceil(extent / spacing).astype(int) + 1, 2)
    points = np.stack(np.meshgrid(*[lower[j] + spacing * np.arange(shape[j])
                                   for j in range(3)], indexing="ij"), axis=-1).reshape(-1, 3)
    values = np.empty(len(points))
    use_gpu = geometry._gpu_allowed and getattr(geometry, "_gpu_token", None) is not None
    for start in range(0, len(points), 500_000):
        chunk = points[start:start + 500_000]
        if use_gpu:
            from .gpu_geometry import geometry_service
            value = geometry_service().request("bounds", geometry._gpu_token, chunk)
        else:
            measured = geometry.scene.query(chunk)
            value = np.where(measured["uncertain"], np.nan, measured["clearance"] - 1e-9)
        values[start:start + len(chunk)] = value
    result = ClearanceBounds(lower, spacing, values.reshape(shape))
    geometry.static_bounds_stats = dict(build_seconds=time.perf_counter() - started,
                                       nodes=len(points), spacing=spacing, gpu=use_gpu)
    return result


def far_geom(geometry, geom, summary):
    bounds = getattr(geometry, "_static_coarse_bounds", None)
    if bounds is None:
        return False
    lower = float(bounds.query(geometry.data.geom_xpos[geom])[0]) - geometry._static_radii[geom]
    if lower < .008 + 1e-9 or not np.isfinite(lower):
        return False
    summary.update(clearance=lower, _screen_band=lower, _screen_ids=np.empty(0, dtype=int))
    geometry.static_coarse_culled_geoms = getattr(geometry, "static_coarse_culled_geoms", 0) + 1
    return True


def screen_points(geometry, summary, points, ids):
    bounds = getattr(geometry, "_static_coarse_bounds", None)
    if bounds is None:
        return points, ids
    lower = bounds.query(points)
    keep = lower < .008 + 1e-9
    omitted = ~keep
    if omitted.any():
        summary["_screen_ids"] = np.r_[summary.get("_screen_ids", np.empty(0, int)), ids[omitted]]
        summary["_screen_phi"] = np.r_[summary.get("_screen_phi", np.empty(0)), lower[omitted]]
        summary["_screen_sort"] = True
        geometry.static_coarse_culled_points = getattr(geometry, "static_coarse_culled_points", 0) + int(omitted.sum())
    return points[keep], ids[keep]
