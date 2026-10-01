"""The optional sparse QP solves the same bounded problem, including scale."""

import numpy as np
import pytest
from scipy import sparse
from scipy.optimize import lsq_linear

from omniretargeting._sequence.linear import bounded_step


@pytest.mark.parametrize("seed", range(4))
def test_sparse_qp_matches_independent_dense_bounded_solution(seed):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(120, 30)) * np.geomspace(0.01, 100, 30)
    rhs = rng.normal(size=120)
    lower, upper = np.full(30, -0.03), np.full(30, 0.08)
    expected = lsq_linear(matrix, rhs, bounds=(lower, upper), tol=1e-12)
    actual = bounded_step(
        sparse.csr_matrix(matrix), rhs, lower, upper, {"linear_solver": "sparse_qp"}
    )
    assert actual.backend == "sparse_qp"
    assert np.all((actual.x >= lower) & (actual.x <= upper))
    np.testing.assert_allclose(
        np.linalg.norm(matrix @ actual.x - rhs),
        np.linalg.norm(matrix @ expected.x - rhs),
        rtol=1e-7,
    )


def test_unfinished_qp_uses_the_existing_rectangular_solver():
    rng = np.random.default_rng(29)
    matrix = sparse.csr_matrix(rng.normal(size=(80, 25)))
    actual = bounded_step(
        matrix,
        rng.normal(size=80),
        np.full(25, -0.01),
        np.full(25, 0.01),
        {"linear_solver": "sparse_qp", "linear_solver_iterations": 1},
    )
    assert actual.backend == "lsmr_fallback"
    assert np.max(abs(actual.x)) <= 0.01
