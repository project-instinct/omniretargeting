"""Conservative nearest-triangle pruning for capsule surface witnesses."""

import numpy as np


class TriangleCandidates:
    """Keep immutable triangle data and the original R-tree face ordering.

    The distance between axis/triangle boxes is a lower bound on the exact
    segment/triangle distance. A triangle vertex supplies an upper bound.
    Only faces strictly beyond that upper bound can be removed; intersecting
    faces and ties always survive, including competing capsule exit vectors.
    """

    def __init__(self, mesh):
        self.triangles = np.asarray(mesh.triangles)
        self.lower = self.triangles.min(axis=1)
        self.upper = self.triangles.max(axis=1)
        self.bounds = np.asarray(mesh.bounds)
        self.tree = mesh.triangles_tree
        self.candidate_faces = 0
        self.retained_faces = 0

    def query(self, start, end, band):
        lower, upper = np.minimum(start, end), np.maximum(start, end)
        if np.any(upper + band < self.bounds[0]) or np.any(lower - band > self.bounds[1]):
            return np.empty(0, dtype=int)
        ids = np.asarray(sorted(self.tree.intersection(np.r_[lower - band, upper + band])), int)
        self.candidate_faces += len(ids)
        if len(ids) > 1:
            triangles = self.triangles[ids]
            axis = end - start
            length2 = float(axis @ axis)
            vertices = triangles.reshape(-1, 3)
            if length2 > 0:
                fraction = np.clip((vertices - start) @ axis / length2, 0, 1)
                delta = vertices - (start + fraction[:, None] * axis)
            else:
                delta = vertices - start
            upper_distance = np.sqrt(np.min(np.einsum("ij,ij->i", delta, delta)))
            gap = np.maximum(np.maximum(self.lower[ids] - upper, lower - self.upper[ids]), 0)
            # Inflate the upper bound, never the scene. Roundoff cannot remove
            # a tied face or an axis intersection used by the exact exit test.
            scale = max(1.0, float(np.abs(triangles).max()), float(np.abs(start).max()),
                        float(np.abs(end).max()))
            # The narrow phase treats distances <= 1e-10 as axis hits and
            # admits tiny face-boundary roundoff. Keep that whole band too.
            allowance = 1e-10 * scale
            keep = np.einsum("ij,ij->i", gap, gap) <= (upper_distance + allowance) ** 2
            ids = ids[keep]
        self.retained_faces += len(ids)
        return ids
