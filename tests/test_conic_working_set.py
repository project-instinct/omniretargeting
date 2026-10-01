"""Constraint generation returns the optimum of the complete conic problem."""

import cvxpy as cp
import numpy as np
import pytest

import omniretargeting.conic_solver as conic_solver
from omniretargeting.conic_solver import solve_conic, solve_conic_working_set


def _restoration(seed, rows=400, cones=60, size=40):
    """min |W d|^2 over a box, many halfspaces and norm balls, some violated at
    d = 0 but all satisfied by a small step (a nearly admissible candidate)."""
    rng = np.random.default_rng(seed)
    delta = cp.Variable(size)
    feasible = rng.normal(scale=0.01, size=size)
    weight = rng.uniform(0.5, 2.0, size)
    matrix = rng.normal(size=(rows, size))
    slack = matrix @ feasible + np.abs(rng.normal(scale=0.5, size=rows))
    slack[: rows // 10] = matrix[: rows // 10] @ feasible + rng.uniform(0, 1e-3, rows // 10)
    constraints = [matrix @ delta <= slack, delta <= 1.0, delta >= -1.0]
    for index in range(cones):
        jac = rng.normal(size=(3, size))
        centre = rng.normal(scale=0.05, size=3)
        extra = 1e-4 if index % 4 == 0 else 1.0
        budget = np.linalg.norm(jac @ feasible + centre) + extra
        constraints.append(cp.SOC(cp.Constant(budget), jac @ delta + centre))
    objective = cp.Minimize(cp.sum_squares(cp.multiply(weight, delta)))
    return delta, objective, constraints


@pytest.mark.parametrize("seed", range(6))
def test_working_set_equals_complete_solve(seed):
    delta, objective, constraints = _restoration(seed)
    complete = cp.Problem(objective, constraints)
    solve_conic(complete, solver=cp.CLARABEL, direct_solve_method="qdldl")
    expected = np.array(delta.value)
    reduced = cp.Problem(objective, constraints)
    solve_conic_working_set(reduced, solver=cp.CLARABEL, direct_solve_method="qdldl")
    assert reduced.status == complete.status == cp.OPTIMAL
    # Both are interior-point optima to CLARABEL's default 1e-8 absolute gap.
    np.testing.assert_allclose(delta.value, expected, atol=5e-6)
    assert reduced.value == pytest.approx(complete.value, rel=0, abs=2e-8)
    # Every constraint of the complete problem holds at the returned step.
    for constraint in constraints:
        assert constraint.violation().max(initial=0) <= 1e-7


def test_infeasible_problem_falls_back_to_the_complete_solve():
    delta = cp.Variable(4)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(delta)), [delta >= 1, cp.sum(delta) <= 1])
    solve_conic_working_set(problem, solver=cp.CLARABEL, direct_solve_method="qdldl")
    assert problem.status == cp.INFEASIBLE


def test_large_working_sets_solve_the_complete_problem(monkeypatch):
    calls = []
    monkeypatch.setattr(conic_solver, "WORKING_SET_FRACTION", 0.0)
    original = conic_solver.solve_conic
    monkeypatch.setattr(conic_solver, "solve_conic", lambda p, **o: calls.append(1) or original(p, **o))
    delta, objective, constraints = _restoration(0)
    problem = cp.Problem(objective, constraints)
    solve_conic_working_set(problem, solver=cp.CLARABEL, direct_solve_method="qdldl")
    assert calls and problem.status == cp.OPTIMAL
