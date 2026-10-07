"""Source-relative joint chatter as exact linear rows of the local repair.

``motion_budgets.chatter_excess`` measures |H (q - r)| against a budget of the
fixed source-shaped reference r, with H = I - Gaussian(sigma) in mode
"nearest". For scalar joints H is linear in q, so the final audit criterion
is a pair of linear rows per (frame, joint), not a reweighting heuristic.
"""

import numpy as np
from scipy import sparse

from ..motion_budgets import (
    CHATTER_EXCESS_TOLERANCE, CHATTER_SCALE_FRAMES, chatter_budget, high_frequency,
)

# Rows aim this far inside the audit tolerance (in radians).
CHATTER_RESERVE = 2e-4
# Half width of the Gaussian kernel (scipy truncates at 4 sigma).
KERNEL_RADIUS = int(4.0 * CHATTER_SCALE_FRAMES + 0.5)


def high_pass_matrix(frames):
    """Sparse H with H @ x == high_frequency(x) for a single signal."""
    if not frames:
        return sparse.csr_matrix((0, 0))
    offsets = np.arange(-KERNEL_RADIUS, KERNEL_RADIUS + 1)
    # Match scipy's order-zero kernel, including normalization and rounding
    # of the radius. Clipping columns sums all nearest-boundary kernel mass
    # onto the endpoint when COO duplicates are converted to CSR.
    weights = np.exp(-0.5 / (CHATTER_SCALE_FRAMES * CHATTER_SCALE_FRAMES) * offsets ** 2)
    weights /= weights.sum()
    row = np.repeat(np.arange(frames), len(offsets))
    col = np.clip((np.arange(frames)[:, None] + offsets).ravel(), 0, frames - 1)
    smooth = sparse.csr_matrix((np.tile(weights, frames), (row, col)), shape=(frames, frames))
    return sparse.eye(frames, format="csr") - smooth


class JointChatter:
    """Fixed reference, its budget and the high-pass operator of one clip."""

    def __init__(self, coords, reference):
        self.width = len(coords.qi)
        self.reference = np.asarray(reference)[:, coords.qi].copy()
        self.frames = len(self.reference)
        self.qi = coords.qi
        self.operator = high_pass_matrix(self.frames)
        self.limit = chatter_budget(self.reference) + CHATTER_EXCESS_TOLERANCE

    def content(self, poses):
        return high_frequency(np.asarray(poses)[:, self.qi] - self.reference)

    def excess(self, poses):
        """Audit excess (positive: a violation), as ``chatter_reports``."""
        return np.abs(self.content(poses)) - self.limit

    def rows(self, poses, frames, columns, *, elastic=False):
        """Linear rows on the flattened joint step for the given frames and joints.

        Returns (matrix, lower, upper) with ``lower <= matrix @ d <= upper``.
        Rows already beyond the reserved limit may keep their current value
        (``elastic=False``) so the zero step stays feasible.
        """
        content = self.content(poses)
        frames = np.asarray(frames, int)
        columns = np.asarray(columns, int)
        block = self.operator[frames]                       # (rows, F)
        limit = self.limit[np.ix_(frames, columns)] - CHATTER_RESERVE
        value = content[np.ix_(frames, columns)]
        if not elastic:
            limit = np.maximum(limit, np.abs(value))
        coo = block.tocoo()
        data, row, col = [], [], []
        for index, column in enumerate(columns):
            data.append(coo.data)
            row.append(coo.row * len(columns) + index)
            col.append(coo.col * self.width + column)
        matrix = sparse.csr_matrix((np.concatenate(data), (np.concatenate(row), np.concatenate(col))),
                                   shape=(len(frames) * len(columns), self.frames * self.width))
        value, limit = value.ravel(), limit.ravel()
        return matrix, -limit - value, limit - value

    def stencil_frames(self, mask):
        """Frames whose high-pass row reads any frame of ``mask``."""
        mask = np.asarray(mask, dtype=bool)
        if not len(mask):
            return np.empty(0, dtype=int)
        padded = np.pad(mask.astype(float), (KERNEL_RADIUS, KERNEL_RADIUS))
        touched = np.convolve(padded, np.ones(2 * KERNEL_RADIUS + 1), "valid") > 0
        return np.flatnonzero(touched)
