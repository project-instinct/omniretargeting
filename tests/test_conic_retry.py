"""One relaxed CLARABEL retry after a numerical stall, never after a verdict."""

import cvxpy as cp
import numpy as np
import pytest

from omniretargeting.conic_solver import RETRY_TOLERANCE, solve_conic


class ScriptedProblem:
    """Records solve options; each call yields the next scripted outcome."""

    def __init__(self, *outcomes):
        self.outcomes, self.calls, self.status = list(outcomes), [], None

    def solve(self, **options):
        self.calls.append(options)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        self.status = outcome
        return 1.0


@pytest.fixture(autouse=True)
def cpu_backend(monkeypatch):
    monkeypatch.delenv("OMNIRETARGETING_CONIC_BACKEND", raising=False)


@pytest.mark.parametrize("status", [cp.OPTIMAL, cp.OPTIMAL_INACCURATE, cp.INFEASIBLE])
def test_converged_or_certified_outcomes_are_not_retried(status):
    problem = ScriptedProblem(status)
    solve_conic(problem, solver=cp.CLARABEL)
    assert len(problem.calls) == 1 and problem.status == status


@pytest.mark.parametrize(
    "first", [cp.error.SolverError("NumericalError"), cp.USER_LIMIT, cp.INFEASIBLE_INACCURATE]
)
def test_numerical_stall_is_retried_once_with_relaxed_accuracy(first):
    problem = ScriptedProblem(first, cp.OPTIMAL)
    solve_conic(problem, solver=cp.CLARABEL, direct_solve_method="qdldl")
    assert len(problem.calls) == 2 and problem.status == cp.OPTIMAL
    retry = problem.calls[1]
    assert retry["direct_solve_method"] == "qdldl"
    assert retry["tol_feas"] == retry["tol_gap_abs"] == retry["tol_gap_rel"]
    assert retry["tol_feas"] == RETRY_TOLERANCE


def test_explicit_iteration_budget_is_respected():
    problem = ScriptedProblem(cp.USER_LIMIT)
    solve_conic(problem, solver="CLARABEL", max_iter=80)
    assert len(problem.calls) == 1 and problem.status == cp.USER_LIMIT


def test_failed_retry_still_reports_the_solver_error():
    problem = ScriptedProblem(cp.error.SolverError("first"), cp.error.SolverError("again"))
    with pytest.raises(cp.error.SolverError):
        solve_conic(problem, solver=cp.CLARABEL)
    assert len(problem.calls) == 2


def test_real_problem_solution_is_unchanged_when_the_first_solve_converges():
    x = cp.Variable(3)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - 1)), [cp.norm(x) <= 1])
    solve_conic(problem, solver=cp.CLARABEL)
    assert problem.status == cp.OPTIMAL
    np.testing.assert_allclose(x.value, np.full(3, 1 / np.sqrt(3)), atol=1e-7)
