"""Default-off projected-box reduction of restoration constraints.

Fixing nonselected coordinates at zero preserves the mathematical feasible set
but can remove rows/cones from the canonical program and change its numerical
trajectory. Keep this experiment disabled pending the full per-item review.
"""

import numpy as np

from .repair_projected import ProjectedBounds

PROJECTED_RESTORATION_REDUNDANCY = False


class RestorationBounds(ProjectedBounds):
    def _range(self, matrix):
        if self.selected is None:
            return super()._range(matrix)
        _, low, high = self.box
        selected = self.selected.ravel()
        low, high = np.where(selected, low, 0.), np.where(selected, high, 0.)
        positive, negative = matrix.maximum(0), matrix.minimum(0)
        return positive @ low + negative @ high, positive @ high + negative @ low
