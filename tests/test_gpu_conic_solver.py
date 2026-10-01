"""Numerical parity checks for the optional CUDA solver path."""

import os

import cvxpy as cp
import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import lsq_linear

from omniretargeting._sequence.linear import bounded_step
from omniretargeting.conic_solver import solve_conic

pytestmark = pytest.mark.skipif(
    os.environ.get("OMNIRETARGETING_CONIC_BACKEND") != "qoco_cuda_strict",
    reason="run with the strict CUDA solver launcher",
)


def test_cuda_bounded_sequence_step_matches_independent_least_squares():
    rng = np.random.default_rng(907)
    matrix = sparse.random(200, 60, density=0.2, random_state=rng, format="csr")
    rhs = rng.normal(size=200)
    lower, upper = np.full(60, -0.05), np.full(60, 0.08)
    expected = lsq_linear(matrix, rhs, bounds=(lower, upper), tol=1e-10)
    actual = bounded_step(matrix, rhs, lower, upper, {})
    assert actual.backend == "qoco_cuda"
    np.testing.assert_allclose(actual.x, expected.x, atol=2e-3, rtol=2e-3)
    assert np.all(actual.x >= lower - 1e-7)
    assert np.all(actual.x <= upper + 1e-7)


def test_cuda_conic_step_preserves_contact_norm_budget():
    step = cp.Variable(12)
    residual = np.array([0.02, -0.01, 0.005])
    jacobian = np.arange(36, dtype=float).reshape(3, 12) / 500
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(step - 0.03)),
        [cp.norm(residual + jacobian @ step) <= 0.025, cp.abs(step) <= 0.04],
    )
    solve_conic(problem, solver=cp.CLARABEL)
    assert problem.solver_stats.solver_name == "QOCO"
    assert np.linalg.norm(residual + jacobian @ step.value) <= 0.0250001
    assert np.max(np.abs(step.value)) <= 0.0400001
