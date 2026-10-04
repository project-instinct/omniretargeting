"""The compact restoration trust cone has the same feasible set and optimum."""
import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast import repair_restoration as restoration


class Bounds:
    def __init__(self, matrix, lower, upper, cones):
        self.matrix, self.lower, self.upper, self._cones = matrix, lower, upper, cones

    def packed(self):
        return self.matrix, self.lower, self.upper

    def cones(self):
        return self._cones


def _problem(seed=3):
    rng = np.random.default_rng(seed)
    full, rows = 60, np.array([3, 7, 8, 20, 31, 32, 45, 59])
    projection = sparse.csc_matrix((np.ones(len(rows)), (rows, np.arange(len(rows)))), shape=(full, len(rows)))
    matrix = sparse.csr_matrix(rng.normal(size=(12, full)) * (rng.random((12, full)) < .3))
    lower, upper = -np.abs(rng.normal(size=12)) - .05, np.abs(rng.normal(size=12)) + .05
    jac = sparse.csr_matrix(rng.normal(size=(6, full)) * (rng.random((6, full)) < .4))
    cones = [(jac, rng.normal(size=(2, 3)) * .01, np.array([.2, .3]))]
    movement = np.zeros(full)
    movement[rows] = rng.normal(size=len(rows)) * .05
    movement[[0, 10]] = .02  # an orthogonal remainder outside the selected rows
    return Bounds(matrix, lower, upper, cones), projection, movement, 1.


@pytest.mark.parametrize("direct", [True, False])
def test_compact_trust_cone_matches_full_cone(monkeypatch, direct):
    bounds, projection, movement, radius = _problem()

    selected_rows = restoration._selected_rows

    def solve(compact):
        monkeypatch.setattr(restoration, "_selected_rows", selected_rows if compact else (lambda _: None))
        if direct:
            return restoration._solve_restoration(
                restoration._restoration_program(bounds, projection, movement, radius))
        return restoration._solve_cvxpy_restoration(bounds, projection, movement, radius)

    full, status_full = solve(False)
    compact, status_compact = solve(True)
    assert status_full == status_compact
    np.testing.assert_allclose(compact, full, atol=1e-6)
    assert np.linalg.norm(projection @ compact + movement) <= radius + 1e-7


def test_non_selector_projection_keeps_the_full_cone():
    scaled = sparse.csc_matrix(np.array([[2.], [0.]]))
    assert restoration._trust_terms(scaled, np.zeros(2)) is None
    shared = sparse.csc_matrix(np.array([[1., 1.], [0., 0.]]))
    assert restoration._trust_terms(shared, np.zeros(2)) is None
