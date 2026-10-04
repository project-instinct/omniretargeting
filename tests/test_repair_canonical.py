"""Direct assembly reproduces CVXPY's numerical program and solved bytes."""

import cvxpy as cp
import numpy as np
import pytest
from scipy import sparse

from omniretargeting.conic_solver import solve_conic
from omniretargeting._fast.repair_canonical import canonical_data, solve_data
from omniretargeting._fast.repair_restoration import _cvxpy_restoration_problem, _trust_terms
from omniretargeting._fast.repair_limits import RepairLimits
from tests.test_fast_restoration_trust import _problem, Bounds
from tests.test_fast_legacy_window import problem
from omniretargeting._fast.contact_guard import ContactGuard


def assert_canonical_equal(bounds, projection, movement, radius, *, solve=True):
    reference, variable = _cvxpy_restoration_problem(bounds, projection, movement, radius)
    expected, _, _ = reference.get_problem_data(cp.CLARABEL)
    actual = canonical_data(bounds, projection, *_trust_terms(projection, movement), radius)
    for key in ("P", "A"):
        x, y = expected[key], actual[key]
        assert x.shape == y.shape
        np.testing.assert_array_equal(x.indptr, y.indptr)
        np.testing.assert_array_equal(x.indices, y.indices)
        assert x.data.tobytes() == y.data.tobytes()
    for key in ("c", "b"):
        assert actual[key].tobytes() == expected[key].tobytes()
    for key in ("zero", "nonneg", "soc", "psd", "exp", "p3d", "pnd"):
        assert getattr(expected["dims"], key) == getattr(actual["dims"], key)
    if solve:
        try:
            solve_conic(reference, solver=cp.CLARABEL, direct_solve_method="qdldl", cpu_only=True)
            status = reference.status
        except cp.error.SolverError:
            status = cp.SOLVER_ERROR
        result, got_status = solve_data(actual, projection.shape[1])
        assert got_status == status
        if result is not None:
            assert result.tobytes() == np.asarray(variable.value).tobytes()


@pytest.mark.parametrize("seed", range(10))
def test_canonical_matrices_and_solutions_match(seed):
    assert_canonical_equal(*_problem(seed))


def test_canonical_real_repair_with_contact_support_and_layout_guards():
    objective, poses = problem(extra=True)
    objective.candidate_guards.append(ContactGuard(objective, poses, preserve_positions=True))
    limits = RepairLimits(objective, poses)
    selected = np.arange(limits.frames * limits.width).reshape(limits.frames, limits.width)[8:15, :2].ravel()
    projection = sparse.eye(limits.frames * limits.width, format="csc")[:, selected]
    for displacement in (0., .01):
        candidate = poses.copy()
        candidate[10, limits.coords.qi[0]] += displacement
        bounds = limits.constraints(candidate, restore_layout=True, projection=projection)
        assert_canonical_equal(bounds, projection, np.zeros(projection.shape[0]), .02)


def test_empty_linear_rows_and_fixed_invalid_cones():
    projection = sparse.eye(3, format="csc")[:, [1]]
    for residual in (np.array([[0., 0., 0.]]), np.array([[1., 2., 3.]])):
        bounds = Bounds(sparse.csr_matrix((0, 3)), np.empty(0), np.empty(0),
                        [(sparse.csr_matrix((3, 3)), residual, np.array([.5]))])
        assert_canonical_equal(bounds, projection, np.zeros(3), .1)


def test_canonical_offset_matches_cvxpy_signed_zero_elimination():
    bounds, projection, movement, radius = _problem()
    movement[:] = -0.
    bounds._cones = [(jac, np.full_like(residual, -0.), allowed)
                     for jac, residual, allowed in bounds._cones]
    assert_canonical_equal(bounds, projection, movement, radius)


def test_solver_adapter_retries_once_with_the_same_cache(monkeypatch):
    from types import SimpleNamespace
    from omniretargeting._fast import repair_canonical
    calls = []

    def solve(self, data, warm_start, verbose, solver_opts, solver_cache):
        calls.append(dict(solver_opts))
        if len(calls) == 1:
            assert not solver_cache
            solver_cache['CLARABEL'] = 'retained'
            return SimpleNamespace(status='InsufficientProgress', x=[0.])
        assert solver_cache['CLARABEL'] == 'retained'
        return SimpleNamespace(status='AlmostSolved', x=[.0123])
    monkeypatch.setattr(repair_canonical.CLARABEL, 'solve_via_data', solve)
    value, status = solve_data({}, 1)
    assert status == cp.OPTIMAL_INACCURATE and value[0] == .0123
    assert calls == [dict(direct_solve_method='qdldl'),
                     dict(direct_solve_method='qdldl', tol_gap_abs=1e-6,
                          tol_gap_rel=1e-6, tol_feas=1e-6)]
