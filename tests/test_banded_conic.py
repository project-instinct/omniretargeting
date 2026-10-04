"""Numerical contracts for the experimental structured conic backend."""

import clarabel
import numpy as np
import pytest
from types import SimpleNamespace
from scipy import sparse

from omniretargeting import banded_conic as bc


@pytest.mark.parametrize("first", ["InsufficientProgress", "MaxIterations", "AlmostPrimalInfeasible"])
def test_direct_restoration_retries_once_after_numerical_stall(monkeypatch, first):
    from omniretargeting._fast.repair_restoration import _solve_restoration
    from omniretargeting.conic_solver import RETRY_TOLERANCE
    calls = []
    outcomes = iter([first, "Solved"])

    class Solver:
        def __init__(self, *args):
            calls.append(args[-1])

        def solve(self):
            return SimpleNamespace(status=next(outcomes), x=[0.])

    monkeypatch.setattr(clarabel, "DefaultSolver", Solver)
    value, status = _solve_restoration((None,) * 5)
    assert status == "optimal" and value[0] == 0
    assert len(calls) == 2
    assert calls[0].tol_feas == 1e-8
    assert calls[1].tol_feas == calls[1].tol_gap_abs == calls[1].tol_gap_rel == RETRY_TOLERANCE


@pytest.mark.parametrize("verdict", ["Solved", "AlmostSolved", "PrimalInfeasible", "DualInfeasible"])
def test_direct_restoration_does_not_retry_success_or_certified_verdict(monkeypatch, verdict):
    from omniretargeting._fast.repair_restoration import _solve_restoration
    calls = []

    class Solver:
        def __init__(self, *args):
            calls.append(args[-1])

        def solve(self):
            return SimpleNamespace(status=verdict, x=[0.])

    monkeypatch.setattr(clarabel, "DefaultSolver", Solver)
    _solve_restoration((None,) * 5)
    assert len(calls) == 1


def test_soc_step_stays_inside_near_boundary():
    v = np.array([[1.0, 1.0 - 1e-10, 0.0, 0.0]])
    dv = np.array([[-0.1, 0.4, 0.0, 0.0]])
    alpha = bc._soc_step(v, dv)
    np.testing.assert_allclose(alpha, (v[0, 0] - v[0, 1]) / 0.5, rtol=1e-8)
    trial = v + 0.99 * alpha * dv
    assert trial[0, 0] > np.linalg.norm(trial[0, 1:])


def test_wide_epigraph_star_uses_sparse_factorization():
    n = 520
    rows = np.arange(1, n)
    links = sparse.csr_matrix((np.full(n - 1, .001), (rows, np.zeros(n - 1, int))),
                              shape=(n, n))
    matrix = 2 * sparse.eye(n) + links + links.T
    factor = bc._Banded(matrix)
    assert factor.width > 256
    factor.factor(matrix, 0.)
    rhs = np.linspace(-1., 1., n)
    np.testing.assert_allclose(matrix @ factor.solve(rhs), rhs, atol=1e-12)


def test_mixed_cone_optimum_matches_clarabel():
    # A strictly convex, feasible problem with a binding SOC and linear rows.
    P = sparse.eye(3, format="csc")
    q = np.array([-2.0, -1.0, -3.0])
    A = sparse.vstack([sparse.eye(3), -sparse.eye(3),
                       sparse.csr_matrix([[0., 0., 0.], [1., 0., 0.],
                                          [0., 1., 0.], [0., 0., 1.]])], format="csc")
    b = np.r_[np.full(6, 10.), 1., 0., 0., 0.]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-10
    ref = clarabel.DefaultSolver(P, q, A, b,
                                 [clarabel.NonnegativeConeT(6), clarabel.SecondOrderConeT(4)],
                                 settings).solve()
    result = bc.solve(P, q, A, b, 6, [4])
    assert result.status == "Solved"
    np.testing.assert_allclose(result.x, ref.x, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(result.obj_val, ref.obj_val, rtol=1e-8, atol=1e-8)


def test_cached_gram_assembly_and_augmented_ldl_match_dense_systems():
    rng = np.random.default_rng(17)
    cones = bc._Cones(7, [4, 5])
    A = sparse.csr_matrix(rng.normal(size=(cones.m, 8)))
    P = sparse.eye(8, format="csr") * 2
    banded = bc._Banded(sparse.csr_matrix(np.ones((8, 8))))
    assembly = bc._ReducedAssembly(P, A, cones, banded)
    augmented = bc._Augmented(P, A, cones, frame_width=2)
    # Reuse both cached maps and the numeric LDL pattern with changed scalings.
    for perturbation in (0.1, 0.7):
        s, z = cones.identity(), cones.identity()
        s[:cones.l] += rng.uniform(0, 1, cones.l)
        z[:cones.l] += rng.uniform(0, 1, cones.l)
        for _, _, ix in cones.groups:
            s[ix[:, 1:]] = perturbation * rng.normal(size=ix[:, 1:].shape)
            z[ix[:, 1:]] = perturbation * rng.normal(size=ix[:, 1:].shape)
            s[ix[:, 0]] += np.linalg.norm(s[ix[:, 1:]], axis=1)
            z[ix[:, 0]] += np.linalg.norm(z[ix[:, 1:]], axis=1)
        scaling = bc._Scaling(cones, s, z)
        H = np.column_stack([scaling.square(np.eye(cones.m)[:, i]) for i in range(cones.m)])
        expected = P.toarray() + bc.RHO * np.eye(8) + A.T @ np.linalg.solve(H + bc.DELTA*np.eye(cones.m), A.toarray())
        ab = assembly.assemble(scaling)
        rebuilt = np.zeros((8, 8))
        for c in range(8):
            for r in range(c + 1):
                rebuilt[r, c] = rebuilt[c, r] = ab[banded.width - c + r, c]
        np.testing.assert_allclose(rebuilt, expected[np.ix_(banded.perm, banded.perm)], rtol=1e-12, atol=1e-12)
        augmented.factor(scaling)
        rx, rz = rng.normal(size=8), rng.normal(size=cones.m)
        dx, dz = augmented.solve(rx, rz)
        np.testing.assert_allclose((P + bc.RHO*sparse.eye(8)) @ dx + A.T @ dz, rx, atol=1e-12)
        np.testing.assert_allclose(A @ dx - (H + bc.DELTA*np.eye(cones.m)) @ dz, rz, atol=1e-12)


def test_direct_restoration_matches_normalized_cvxpy_problem():
    import cvxpy as cp
    from omniretargeting._fast.repair_restoration import _restoration_program, _solve_restoration
    projection = sparse.csr_matrix([[1., 0.], [0., 1.], [0., 0.]])
    matrix = sparse.csr_matrix([[1., 1., 0.], [0., 0., 1.]])
    lo, hi = np.array([-.1, -1.]), np.array([.1, 1.])
    jac = sparse.csr_matrix([[1., 0., 0.], [0., 0., 0.],
                             [0., 0., 0.], [0., 1., 0.],
                             [0., 0., 0.], [0., 0., 1.]])
    residual = np.array([[.02, 0.], [0., .0002], [0., 0.]])
    allowed = np.array([.01, .0001, .1])
    bounds = SimpleNamespace(packed=lambda: (matrix, lo, hi),
                             cones=lambda: [(jac, residual, allowed)])
    movement, radius = np.array([.004, -.001, .0005]), .02
    program = _restoration_program(bounds, projection, movement, radius)
    assert program[0].shape == (2, 2)  # No norm epigraph variables.
    assert [cone.dim for cone in program[-1]] == [2, 4, 3, 3]
    actual, status = _solve_restoration(program)
    delta = cp.Variable(2)
    keep = np.array([True, True, False])
    scale = np.maximum(allowed[keep], 1e-3)
    value = cp.reshape((jac @ projection)[np.repeat(keep, 2)] @ delta,
                       (2, 2), order="C") + residual[keep]
    problem = cp.Problem(cp.Minimize(cp.sum_squares(delta) / radius**2),
                         [(matrix @ projection)[0] @ delta >= lo[0],
                          (matrix @ projection)[0] @ delta <= hi[0],
                          cp.norm(projection @ delta + movement) <= radius,
                          cp.norm(cp.multiply(1 / scale[:, None], value), axis=1) <= allowed[keep] / scale])
    problem.solve(solver="CLARABEL", direct_solve_method="qdldl",
                  tol_feas=1e-12, tol_gap_abs=1e-12, tol_gap_rel=1e-12)
    assert status == "optimal"
    np.testing.assert_allclose(actual, delta.value, atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(actual @ actual / radius**2, problem.value, atol=1e-9)


def test_direct_restoration_flag_is_default_off_and_recognized(monkeypatch):
    from omniretargeting._fast.runtime_options import RuntimeOptions
    monkeypatch.delenv('OMNIRETARGETING_FAST_DISABLE', raising=False)
    monkeypatch.delenv('OMNIRETARGETING_FAST_ENABLE', raising=False)
    assert not RuntimeOptions.from_environment().direct_repair_restoration
    monkeypatch.setenv('OMNIRETARGETING_FAST_ENABLE', 'direct_repair_restoration')
    assert RuntimeOptions.from_environment().direct_repair_restoration


def test_direct_restoration_retains_original_nonlinear_regression(monkeypatch):
    from dataclasses import replace
    from omniretargeting._fast import repair_restoration
    from tests.test_repair_restoration import (
        test_restoration_fixes_small_nonlinear_overshoot_without_undoing_leg_progress,
    )
    monkeypatch.setattr(repair_restoration, 'OPTIONS',
                        replace(repair_restoration.OPTIONS, direct_repair_restoration=True))
    test_restoration_fixes_small_nonlinear_overshoot_without_undoing_leg_progress()
