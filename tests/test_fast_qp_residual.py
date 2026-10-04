"""Independent reconstruction of the full Clarabel stopping residuals."""

import numpy as np
import pytest
from scipy import sparse
from types import SimpleNamespace

from omniretargeting._fast.qp_residual import clarabel_residual
from omniretargeting._fast.qp_types import ConeProgram, settings, solve_program


def test_solver_residual_matches_clarabel_primal_dual_definitions():
    # A norm cone and nonuniformly scaled affine rows make unscaled and
    # solver-normalized definitions observably different.
    A = sparse.csc_matrix([[1., 1e2], [-1e-3, 0.], [0., 0.], [-1., 0.], [0., -1.]])
    b = np.array([.5, .03, 1., 0., 0.])
    P, c = sparse.diags([.2, 2.]), np.array([-1., .1])
    program = ConeProgram(A, b, 2, (3,), 2)
    result = solve_program(program, P, c)
    residual = clarabel_residual(P, c, A, b, 2, (3,), result.x, result.z, settings(), result.s)
    assert residual['primal'] == pytest.approx(result.r_prim, rel=1e-3, abs=1e-15)
    assert residual['stationarity'] == pytest.approx(result.r_dual, rel=1e-3, abs=1e-15)
    assert residual['gap'] == pytest.approx(abs(result.obj_val - result.obj_val_dual), abs=1e-14)
    assert residual['certified']


def test_solver_normalized_certificate_still_rejects_coordinate_stall():
    A = sparse.csc_matrix([[1., -1.], [-1., 1.]])
    result = clarabel_residual(sparse.eye(2), -np.ones(2), A, np.zeros(2), 2, [],
                               np.zeros(2), np.zeros(2), settings())
    assert not result['certified']
    assert result['stationarity'] > .1


def test_projected_slack_cannot_hide_an_infeasible_constant_cone():
    result = clarabel_residual(sparse.eye(2), np.zeros(2), sparse.csc_matrix((4, 2)),
                               np.array([1., 2., 0., 0.]), 0, [4], np.zeros(2), np.zeros(4), settings())
    assert not result['certified']
    assert result['primal'] > .1


@pytest.mark.parametrize('refinement_succeeds', [True, False])
def test_priority_refinement_rechecks_same_program_at_original_tolerance(monkeypatch, refinement_succeeds):
    from omniretargeting._fast import priority_qp
    program = ConeProgram(sparse.csc_matrix([[-1.]]), np.array([-1.]), 1, (), 1)
    P, c = sparse.eye(1, format='csc'), np.zeros(1)
    good = solve_program(program, P, c)
    bad = SimpleNamespace(x=np.asarray(good.x) + .01, z=good.z)
    calls = []
    def solve(p, hessian, linear, *, precision=False):
        assert p is program and hessian is P and linear is c
        calls.append(precision)
        return good if precision and refinement_succeeds else bad
    monkeypatch.setattr(priority_qp, 'solve_program', solve)
    model = SimpleNamespace(solve_diagnostics=[])
    result = priority_qp._solve_certified(model, program, P, c, 'level_0')
    assert calls == [False, True]
    assert (result is good) == refinement_succeeds
    assert not model.solve_diagnostics[0]['certified']
    assert model.solve_diagnostics[1]['certified'] == refinement_succeeds
    assert settings().tol_feas == 1e-8
    assert settings(precision=True).tol_feas < settings().tol_feas
