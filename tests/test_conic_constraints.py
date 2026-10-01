"""Batched sparse cones retain independent budgets and cross-frame coupling."""

import cvxpy as cp
import numpy as np
import pytest
from scipy import sparse

from omniretargeting.conic_constraints import (
    LocalNormBounds,
    squared_residual_objective,
)


@pytest.mark.parametrize("scale", [1e-3, 1.0, 100.0])
def test_batched_cones_match_independent_dense_problem(scale):
    rng = np.random.default_rng(90)
    frames, width = 4, 3
    system = rng.normal(size=(22, frames * width))
    rhs = rng.normal(size=22) * scale
    flat = cp.Variable(frames * width)
    separate, bounds = [], LocalNormBounds(frames, width)
    for f in range(frames):
        height = 1 + f % 3
        jac = rng.normal(size=(height, width))
        residual = rng.normal(size=height) * scale * 0.1
        terms = [(f, jac)]
        expression = residual + jac @ flat[f * width : (f + 1) * width]
        if f:
            terms.append((f - 1, -jac))
            expression -= jac @ flat[(f - 1) * width : f * width]
        bounds.add(residual, scale, *terms)
        separate.append(cp.norm(expression) <= scale)
    direct = cp.Problem(cp.Minimize(cp.sum_squares(system @ flat - rhs)), separate)
    direct.solve(
        solver="CLARABEL",
        direct_solve_method="qdldl",
        tol_gap_abs=1e-12,
        tol_gap_rel=1e-12,
        tol_feas=1e-12,
    )
    expected = flat.value.copy()
    batched = cp.Problem(
        cp.Minimize(squared_residual_objective(sparse.csr_matrix(system), rhs, flat)),
        bounds.constraints(flat),
    )
    batched.solve(
        solver="CLARABEL",
        direct_solve_method="qdldl",
        tol_gap_abs=1e-12,
        tol_gap_rel=1e-12,
        tol_feas=1e-12,
    )
    np.testing.assert_allclose(flat.value, expected, rtol=2e-4, atol=scale * 2e-5)
