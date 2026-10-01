"""Certified cell bounds for repeated convex-hull threshold classifications.

The maximum of the hull's plane functions is Lipschitz. Its value at a cell
centre, plus/minus the cell radius, bounds every point in that cell. We only
reuse the resulting boolean when this interval excludes the threshold;
boundary points retain the original plane evaluation, without quantization.
"""

import numpy as np


class ConvexHullScope:
    def __init__(self, planes, bounds):
        self.planes = planes
        lower, upper = bounds
        self.spacing = max(float(np.max(upper - lower)), 1e-9) / 64
        self.lower = lower - 2 * self.spacing
        self.counts = np.ceil((upper - self.lower) / self.spacing).astype(int) + 3
        self.centres = np.full(tuple(self.counts), np.nan)
        lipschitz = np.linalg.norm(planes[:, :3], axis=1).max()
        # Cover both centre construction and the original BLAS dot product.
        magnitude = max(1., float(np.max(np.abs(bounds))), float(np.max(np.abs(planes))))
        self.radius = self.spacing * np.sqrt(3) / 2 * lipschitz + 1e-12 * magnitude

    def classify(self, points, thresholds, exact):
        """Return hull_margin > threshold; exact computes the original margins."""
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        thresholds = np.asarray(thresholds, dtype=float)
        result = np.empty((len(points), len(thresholds)), bool)
        cells = np.floor((points - self.lower) / self.spacing).astype(np.int64)
        inside = np.all((cells >= 0) & (cells < self.counts), axis=1)
        ids = np.flatnonzero(inside)
        flat = np.ravel_multi_index(cells[ids].T, self.counts)
        values = self.centres.reshape(-1)
        missing = np.unique(flat[np.isnan(values[flat])])
        if len(missing):
            centres = self.lower + self.spacing * (
                np.stack(np.unravel_index(missing, self.counts), axis=1) + .5)
            for first in range(0, len(missing), 1024):
                part = slice(first, first + 1024)
                values[missing[part]] = (
                    centres[part] @ self.planes[:, :3].T + self.planes[:, 3]).max(axis=1)
        centre = values[flat, None]
        decided = np.zeros(len(points), bool)
        certain = np.all((centre - self.radius > thresholds)
                         | (centre + self.radius < thresholds), axis=1)
        decided[ids[certain]] = True
        result[ids[certain]] = centre[certain] > thresholds
        boundary = np.flatnonzero(~decided)
        if len(boundary):
            result[boundary] = exact(points[boundary])[:, None] > thresholds
        return result
