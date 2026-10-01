"""Whether points lie within the source-evidence radius, decided by cells.

``nearest evidence distance <= radius`` is the scope test of open-surface
queries. A cell whose centre is closer than the radius by more than its
half-diagonal lies entirely inside the scope; one farther by more than that
lies entirely outside. Only points of the remaining boundary cells need the
nearest-neighbour query, which answers them exactly as before. Cells are
classified on first use; the margin covers the query's own rounding.
"""

import numpy as np

# Cell edge as a fraction of the evidence radius.
CELLS_PER_RADIUS = 16
# Dense grids larger than this use coarser cells (only affects speed).
MAX_CELLS = 1 << 23
_UNKNOWN, _OUTSIDE, _BOUNDARY, _INSIDE = 0, 1, 2, 3


class EvidenceScope:
    def __init__(self, tree, points, radius):
        self.tree, self.radius = tree, float(radius)
        self.bound = self.radius * (1 + 1e-9) + 1e-12
        points = np.asarray(points, dtype=float)
        spacing = self.radius / CELLS_PER_RADIUS
        while True:
            # One spare cell beyond the radius on every side: a point outside
            # the grid is farther than the radius from every evidence point.
            self.lower = points.min(axis=0) - self.radius - spacing
            self.counts = np.ceil((points.max(axis=0) + self.radius + spacing - self.lower) / spacing).astype(int) + 1
            if np.prod(self.counts) <= MAX_CELLS:
                break
            spacing *= 1.25
        self.spacing = spacing
        self.state = np.zeros(tuple(self.counts), dtype=np.int8)
        self.half = spacing * np.sqrt(3) / 2
        self.margin = 1e-9 * self.radius + 1e-12

    def __call__(self, points):
        """``tree.query(points, distance_upper_bound)[0] <= radius``."""
        points = np.asarray(points, dtype=float)
        result = np.zeros(len(points), bool)
        if not len(points):
            return result
        cells = np.floor((points - self.lower) / self.spacing).astype(np.int64)
        in_grid = np.all((cells >= 0) & (cells < self.counts), axis=1)
        index = np.flatnonzero(in_grid)
        flat = np.ravel_multi_index(cells[index].T, self.counts)
        state = self.state.reshape(-1)
        unknown = np.unique(flat[state[flat] == _UNKNOWN])
        if len(unknown):
            centres = self.lower + self.spacing * (np.stack(np.unravel_index(unknown, self.counts), axis=1) + 0.5)
            distance, _ = self.tree.query(centres)
            state[unknown] = np.where(
                distance + self.half <= self.radius - self.margin, _INSIDE,
                np.where(distance - self.half >= self.radius + self.margin, _OUTSIDE, _BOUNDARY))
        cell_state = state[flat]
        result[index[cell_state == _INSIDE]] = True
        boundary = index[cell_state == _BOUNDARY]
        if len(boundary):
            distance, _ = self.tree.query(points[boundary], distance_upper_bound=self.bound)
            result[boundary] = distance <= self.radius
        return result
