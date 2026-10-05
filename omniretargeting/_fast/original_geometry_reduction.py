"""Exact closed-component minima with conservative unsigned distance cones.

Only moving repair's reduction consumer may use this backend: it needs the
minimum clearance when inside the activation band, and every uncertainty flag.
Every point keeps its original fast-winding evaluation. Ambiguous signs always
receive the full original query. Distance cones merely omit points proved
unable to supply a minimum/witness; no cone value becomes geometry evidence.
"""

import time

import numpy as np

from .original_geometry_bounds import ClearanceBounds


class ClosedComponentReduction:
    MAX_NODES = 300_000
    SLACK = 1e-9

    def __init__(self, scene, band):
        self.scene = scene
        self._query = scene.query
        self.bounds = {}
        self.stats = dict(build_seconds=0., nodes=0, points=0, exact_points=0,
                          winding_seconds=0., screen_seconds=0.)
        started = time.perf_counter()
        for cid, (_, _, closed, tree) in enumerate(scene.components):
            if (not closed or scene._closed_normals[cid] is not None
                    or scene.enclosed_free_space):
                continue
            vertices, faces, bounds, _ = scene._component_query_data[cid]
            lower, upper = bounds[0] - band - .001, bounds[1] + band + .001
            extent = upper - lower
            spacing = max(.01, float((np.prod(extent) / self.MAX_NODES) ** (1 / 3)))
            shape = np.maximum(np.ceil(extent / spacing).astype(int) + 1, 2)
            points = np.stack(np.meshgrid(*[lower[j] + spacing * np.arange(shape[j])
                                           for j in range(3)], indexing="ij"), axis=-1).reshape(-1, 3)
            values = np.empty(len(points))
            for start in range(0, len(points), 100_000):
                chunk = points[start:start + 100_000]
                d2, _, _ = tree.squared_distance(vertices, faces, chunk)
                values[start:start + len(chunk)] = np.sqrt(np.maximum(d2, 0.))
            values = values.reshape(shape)
            self.bounds[cid] = (ClearanceBounds(lower, spacing, values),
                                ClearanceBounds(lower, spacing, -values))
            self.stats['nodes'] += len(points)
        self.stats['build_seconds'] = time.perf_counter() - started

    def query(self, points, groups, component, band):
        scene = self.scene
        if component not in self.bounds or scene.query != self._query:
            return scene.query(points, component_ids=[component])
        points = np.asarray(points, dtype=np.float64)
        self.stats['points'] += len(points)
        vertices, faces, bounds, _ = scene._component_query_data[component]
        inside = np.all((points >= bounds[0]) & (points <= bounds[1]), axis=1)
        winding = np.zeros(len(points))
        started = time.perf_counter()
        if inside.any():
            winding[inside] = scene._r4_winding_cache[id(vertices), id(faces)](points[inside])
        self.stats['winding_seconds'] += time.perf_counter() - started
        started = time.perf_counter()
        magnitude = np.abs(winding)
        # Original SceneQuery may resolve these by exact winding or by its
        # zero-distance rule. Neither their sign nor reliability is screened.
        ambiguous = ((magnitude >= .01) & (np.abs(magnitude - 1.) >= .01))
        ambiguous |= ~np.isfinite(magnitude)
        positive, negative = self.bounds[component]
        unsigned_lower, unsigned_upper = positive.query(points), -negative.query(points)
        lower = np.where(magnitude > .5, -unsigned_upper, unsigned_lower)
        upper = np.where(magnitude > .5, -unsigned_lower, unsigned_upper)
        upper[ambiguous] = np.inf
        keep = ambiguous.copy()
        for begin, end in groups:
            # All possible minimizers survive (including equal minima, in
            # their original order). Above band, depth=0 and there is no row.
            threshold = min(float(upper[begin:end].min(initial=np.inf)), band)
            keep[begin:end] |= lower[begin:end] <= threshold + self.SLACK
        selected = np.flatnonzero(keep)
        self.stats['exact_points'] += len(selected)
        self.stats['screen_seconds'] += time.perf_counter() - started
        measured = scene.query_with_winding(points[selected], component, winding[selected])
        # An omitted reliable positive point cannot change depth or the row.
        # A finite positive sentinel also preserves the consumer's finite
        # reliability check. It never becomes a witness or a reported depth.
        result = {}
        for key, value in measured.items():
            shape = (len(points), *value.shape[1:])
            fill = (np.finfo(np.float64).max if key == 'clearance'
                    else np.inf if key == 'unsigned_distance' else 0)
            output = np.full(shape, fill, dtype=value.dtype)
            output[selected] = value
            result[key] = output
        return result
