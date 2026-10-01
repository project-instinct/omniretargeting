"""Batch local vector-norm bounds without expanding trajectory expressions.

Each row retains its own Euclidean budget. Padding with zeros and stacking
independent cones changes neither the feasible set nor contact priorities.
"""

import cvxpy as cp
import numpy as np
from scipy import sparse


class LocalNormBounds:
    def __init__(self, frames, width):
        self.frames, self.width = frames, width
        self.rows = []

    def add(self, residual, budget, *terms):
        """Bound norm(residual + sum(jacobian @ step[frame])) by budget."""
        residual = np.atleast_1d(np.asarray(residual, float))
        self.rows.append((residual, float(budget), terms))

    def constraints(self, flat):
        if not self.rows:
            return []
        height = max(len(row[0]) for row in self.rows)
        offsets = np.zeros((len(self.rows), height))
        indices, columns, values, budgets = [], [], [], []
        for index, (residual, budget, terms) in enumerate(self.rows):
            offsets[index, : len(residual)] = residual
            budgets.append(budget)
            for frame, jacobian in terms:
                jacobian = np.atleast_2d(jacobian)
                rr, cc = np.nonzero(jacobian)
                indices.extend(index * height + rr)
                columns.extend(frame * self.width + cc)
                values.extend(jacobian[rr, cc])
        matrix = sparse.coo_matrix(
            (values, (indices, columns)),
            shape=(len(self.rows) * height, self.frames * self.width),
        ).tocsr()
        return [
            cp.SOC(
                np.asarray(budgets),
                cp.reshape(
                    matrix @ flat + offsets.ravel(), (height, len(self.rows)), order="F"
                ),
                axis=0,
            )
        ]


def squared_residual_objective(matrix, rhs, flat):
    """Sparse quadratic form of ||matrix @ flat - rhs||², without auxiliaries."""
    hessian = (matrix.T @ matrix).tocsc()
    return (
        cp.quad_form(flat, cp.psd_wrap(hessian))
        - 2 * np.asarray(matrix.T @ rhs).ravel() @ flat
    )
