"""Exact duplicate temporal halfspaces, preserving all nonlinear row values."""

import numpy as np
from scipy import sparse

from ..pose_relations import _row_groups
from .qp_guard_rows import HardBlock


def merge_temporal(blocks):
    """Equal sparse Jacobian rows become their tightest bound, with no rounding.

    Keep every temporal constraint represented. Equal tangent rows can arise
    from both the reference-correction and world-signal continuity guards.
    Their different immutable budgets/residuals still intersect through min.
    """
    if len(blocks) < 2 or any(b.soc or b.evaluate is None for b in blocks):
        return blocks
    matrix = sparse.vstack([b.A for b in blocks], format='csr')
    if matrix.shape[0] == 0:
        return blocks
    groups = _row_groups(matrix)
    count = int(groups.max()) + 1
    if count == len(groups):
        return blocks
    first = np.full(count, len(groups))
    np.minimum.at(first, groups, np.arange(len(groups)))
    order = np.argsort(first)

    def tightest(values):
        result = np.full(count, np.inf)
        np.minimum.at(result, groups, values)
        return result[order]

    def evaluate(poses):
        return tightest(np.concatenate([b.evaluate(poses) for b in blocks]))

    return [HardBlock(matrix[first[order]], tightest(np.concatenate([b.b for b in blocks])),
                      evaluate=evaluate)]
