"""Physical mesh penetration independent of collision detection margins.

The compiled collision mesh is convex, even when its source STL is concave.
An inner Minkowski polytope is expanded with exact vertex support queries.
For overlap, its nearest face and the corresponding support plane bracket the
minimum translation distance. Positive results are conservative separating
plane gaps, not necessarily Euclidean distances between closest points.
"""

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .geometry_query import GeometryQueryError

TOLERANCE = 1e-9
MAX_ITERATIONS = 128


class ConvexMeshDistance:
    def __init__(self, model):
        self.model = model
        self.vertices = {}
        axes = np.array(np.meshgrid(*[[-1., 0., 1.]] * 3)).reshape(3, -1).T
        axes = axes[np.linalg.norm(axes, axis=1) > 0]
        self.seed_axes = axes / np.linalg.norm(axes, axis=1)[:, None]

    def world_vertices(self, geom, data):
        mesh = int(self.model.geom_dataid[geom])
        if mesh not in self.vertices:
            start = self.model.mesh_vertadr[mesh]
            count = self.model.mesh_vertnum[mesh]
            vertices = self.model.mesh_vert[start:start + count].astype(float)
            # Drop only interior vertices; the convex collision surface is exact.
            self.vertices[mesh] = vertices[ConvexHull(vertices).vertices]
        return (self.vertices[mesh] @ data.geom_xmat[geom].reshape(3, 3).T
                + data.geom_xpos[geom])

    def query(self, first, second, data):
        """Return clearance, material witnesses and first-body exit direction.

        No result is returned on an unclosed bound: uncertain queries must not
        silently certify acceptance. Original model options and vertices stay
        untouched, and no mesh decimation or collision exclusions are applied.
        """
        a = self.world_vertices(first, data)
        b = self.world_vertices(second, data)
        origin = (a.mean(axis=0) + b.mean(axis=0)) / 2
        a, b = a - origin, b - origin

        def support(axes):
            return a[np.argmax(a @ axes.T, axis=0)], b[np.argmin(b @ axes.T, axis=0)]

        pa, pb = support(self.seed_axes)
        vertices = pa - pb
        for _ in range(MAX_ITERATIONS):
            try:
                hull = ConvexHull(vertices)
            except QhullError as error:
                raise GeometryQueryError("degenerate mesh Minkowski polytope") from error
            face = np.argmax(hull.equations[:, 3])
            normal = hull.equations[face, :3]
            lower = -float(hull.equations[face, 3])
            sa, sb = support(normal[None])
            point = sa[0] - sb[0]
            upper = float(point @ normal)
            if upper <= TOLERANCE:
                # This supporting plane proves separation (or numerical touch).
                return max(0., -upper), np.array([sa[0], sb[0]]) + origin, -normal
            if lower >= 0 and upper - lower <= TOLERANCE:
                # Qhull triangulates coplanar faces; the plane projection may
                # lie in a sibling triangle, not the first one returned.
                coplanar = ((hull.equations[:, :3] @ normal > 1 - 1e-8)
                            & (np.abs(hull.equations[:, 3] + lower) < 1e-8))
                for ids in hull.simplices[coplanar]:
                    weights = np.linalg.lstsq(
                        np.r_[vertices[ids].T, np.ones((1, 3))],
                        np.r_[lower * normal, 1.], rcond=None)[0]
                    if weights.min() >= -1e-6:
                        break
                else:
                    raise GeometryQueryError("invalid mesh penetration face projection")
                witnesses = np.array([weights @ pa[ids], weights @ pb[ids]]) + origin
                return -lower, witnesses, -normal
            if np.min(np.linalg.norm(vertices - point, axis=1)) < 1e-12:
                raise GeometryQueryError("mesh penetration bound did not close")
            pa, pb = np.r_[pa, sa], np.r_[pb, sb]
            vertices = np.r_[vertices, point[None]]
        raise GeometryQueryError("mesh penetration iteration limit")
