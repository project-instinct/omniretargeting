"""Diagnostic-only performance tracing for the fast solver."""

import numpy as np
import pytest

from omniretargeting._fast.performance_trace import (
    PerformanceTrace, STAGES, explain_guard,
)
from omniretargeting._fast.solver import optimize
from tests.test_fast_backend import _objective


def test_trace_reports_exclusive_stages_and_requested_solver_counters():
    trace = PerformanceTrace()
    trace.add_stage_seconds("source_pose", 2.0)
    with trace.stage("contact_acquisition"):
        trace.add_stage_seconds("contact_acquisition", 3.0)
        trace.add_operation("banded_solves", 0.25)
        trace.add_operation("objective_evaluations", 0.5)
        trace.add_operation("objective_jacobian_evaluations", 0.5)
        trace.add_alpha(1.0)
        trace.add_failed(0.75, "objective_no_decrease")
    report = trace.report(total_seconds=5.5)

    assert [row["stage"] for row in report["stage_timings"]] == list(STAGES)
    assert report["slowest_stage"] == "contact_acquisition"
    assert report["total_stage_seconds"] == pytest.approx(5.0, abs=1e-3)
    assert report["unattributed_seconds"] == pytest.approx(0.5, abs=1e-3)
    details = report["slowest_stage_details"]
    assert details["operations"]["banded_solves"]["calls"] == 1
    assert details["operations"]["objective_evaluations"]["calls"] == 1
    assert details["operations"]["objective_jacobian_evaluations"]["calls"] == 1
    assert details["accepted_step_distribution"]["1"] == 1
    assert details["failed_candidates"]["reasons"]["objective_no_decrease"] == 1


def test_traced_optimize_does_not_change_motion_and_cleans_instance_patches():
    plain_objective, poses, _ = _objective(temporal=True)
    traced_objective, traced_poses, _ = _objective(temporal=True)
    plain, _ = optimize(plain_objective, poses, iterations=2)

    trace = PerformanceTrace()
    restore = trace.install(traced_objective)
    trace.current_stage = "source_pose"
    try:
        traced, history = optimize(
            traced_objective, traced_poses, iterations=2, label="source_pose")
    finally:
        trace.current_stage = None
        restore()
    report = trace.report()

    np.testing.assert_array_equal(plain, traced)
    assert "evaluate" not in traced_objective.__dict__
    assert "performance_trace" not in traced_objective.__dict__
    operations = report["operations"]["source_pose"]
    assert operations["banded_solves"]["calls"] > 0
    assert operations["objective_evaluations"]["calls"] > 0
    assert operations["objective_jacobian_evaluations"]["calls"] > 0
    assert report["accepted_step_count"] > 0
    assert history[-1]["solver_performance"]["phase"] == "source_pose"


def test_guard_explanation_reports_limit_actual_and_excess():
    class Guard:
        def diagnostics(self, _poses):
            return {
                "accepted": False,
                "worst_constraint": {
                    "group": "left_foot_surface",
                    "frame": 17,
                    "excess_m": 0.003,
                },
            }

    row = explain_guard(objective=None, guard=Guard(), poses=np.zeros((20, 1)))
    assert row["joint_or_contact"] == "left_foot_surface"
    assert row["frames"] == [17, 18]
    assert row["limit"] == 0.0
    assert row["actual"] == row["excess"] == 0.003


def test_nested_proposal_phase_seconds_are_mutually_exclusive(monkeypatch):
    from omniretargeting._fast import performance_trace

    clock = iter([0., 1., 3., 4., 7., 10.])
    monkeypatch.setattr(performance_trace.time, "monotonic", lambda: next(clock))
    trace = PerformanceTrace(stage="contact_acquisition")
    with trace.operation("restoration"):
        with trace.operation("constraint_row_construction"):
            pass
        with trace.operation("numerical_solve"):
            pass
    operations = trace.report()["operations"]["contact_acquisition"]
    assert operations["restoration"] == {"calls": 1, "seconds": 5.}
    assert operations["constraint_row_construction"]["seconds"] == 2.
    assert operations["numerical_solve"]["seconds"] == 3.
    assert sum(row["seconds"] for row in operations.values()) == 10.


def test_report_explicitly_marks_jacobians_as_included_and_merges_iterations():
    trace = PerformanceTrace(stage="contact_acquisition", frames=(12, 20))
    trace.begin_phase("acquire")
    trace.begin_iteration("acquire", 0)
    trace.add_evaluation(.25, with_jacobian=True)
    trace.add_evaluation(.5, with_jacobian=False)
    trace.record_call("proposal", "banded")
    trace.banded_alpha(.125)
    trace.finish_iteration({"alpha": .125}, "banded")
    phase = trace.finish_phase("iteration_cap")
    parent = PerformanceTrace()
    parent.absorb(phase)
    report = parent.report()
    semantics = report["counter_semantics"]["objective_jacobian_evaluations"]
    assert semantics["included_in"] == "objective_evaluations"
    assert semantics["additive"] is False
    assert "never sum" in semantics["description"]
    operations = report["operations"]["contact_acquisition"]
    assert operations["objective_evaluations"] == {"calls": 2, "seconds": .75}
    assert operations["objective_jacobian_evaluations"] == {"calls": 1, "seconds": .25}
    assert report["iterations"] == phase["iterations"]
    assert report["iterations"][0]["frames"] == [12, 20]


def test_real_constraint_proposal_records_construction_canonicalization_setup_and_solve():
    from omniretargeting._fast import shape_step, solver
    from omniretargeting._fast.reference import ReferenceShape

    objective, poses, _ = _objective(temporal=True)
    objective.source_shape = ReferenceShape(objective, poses)
    objective.source_shape.freeze_reference(poses)
    acc, residual, temporal, _, _ = objective.evaluate(poses)
    gradient = acc.gradient.ravel() + temporal.T @ residual
    system = solver.banded_system(objective.frames, objective.width, acc.hessian, temporal)
    diagonal = np.maximum(np.diagonal(acc.hessian, axis1=1, axis2=2).ravel(), 1e-9)
    args = objective, poses, system, gradient, diagonal, 1e-3
    plain = shape_step.bounded_shape_step(*args)
    trace = PerformanceTrace(stage="contact_acquisition")
    original = shape_step.bounded_shape_step
    with trace.instrument():
        traced = shape_step.bounded_shape_step(*args)
    assert shape_step.bounded_shape_step is original
    np.testing.assert_array_equal(plain, traced)
    operations = trace.report()["operations"]["contact_acquisition"]
    for name in ("constraint_row_construction", "canonicalization", "solver_setup", "numerical_solve"):
        assert operations[name]["calls"] > 0
        assert operations[name]["seconds"] > 0


def test_iteration_backend_does_not_inherit_a_rejected_conic_attempt(monkeypatch):
    from omniretargeting._fast import proposals, solver

    class Shape:
        def accepts(self, _poses):
            return True

        def project(self, poses):
            return poses

    monkeypatch.setattr(proposals, "SPECULATE", False)
    pass
    original = proposals.Attempts.banded

    def banded(self, attempt):
        if attempt == 0:
            return np.zeros(self.gradient.size), None, 0.
        return original(self, attempt)

    def conic(objective, poses, *_args, **_kwargs):
        # The real proposal records the solve it actually performs.
        trace = getattr(objective, "performance_trace", None)
        if trace is not None:
            trace.record_call("proposal", "shape_conic")
        return np.zeros(objective.frames * objective.width), poses.copy(), 1.

    monkeypatch.setattr(proposals.Attempts, "banded", banded)
    monkeypatch.setattr(solver, "_conic_proposal", conic)
    plain_objective, poses, _ = _objective()
    plain_objective.source_shape = Shape()
    plain, plain_history = optimize(plain_objective, poses, iterations=2)
    objective, poses, _ = _objective()
    objective.source_shape = Shape()
    trace = PerformanceTrace(stage="contact_acquisition")
    restore = trace.install(objective)
    try:
        traced, history = optimize(objective, poses, iterations=2)
    finally:
        restore()
    np.testing.assert_array_equal(traced, plain)
    assert [row.get("step_backend") for row in history] == [row.get("step_backend") for row in plain_history]
    assert len(trace.report()["iterations"]) == len(history)
    for row in history:
        diagnostic = row["iteration_performance"]
        assert diagnostic["banded_feasible_alpha"] == [0., 1.]
        assert diagnostic["accepted_step_backend"] == "banded"
        assert diagnostic["fallback_reason"] == ["banded_infeasible"]
        assert diagnostic["proposal_calls"] == {"banded": 2, "shape_conic": 1}
        assert diagnostic["proposal_call_count"] == 3
        assert diagnostic["restore_call_count"] == 0
        assert row["step_backend"] == "shape_conic"  # Existing production history stays intact.




def test_rejected_iteration_reports_multiple_rows_and_short_circuited_guard(monkeypatch):
    from omniretargeting._fast import solver

    class Guard:
        def __init__(self, name, excess):
            self.name, self.excess, self.calls = name, excess, 0

        def accepts(self, _poses):
            self.calls += 1
            return False

        def diagnostics(self, _poses):
            return {"groups": [
                {"name": self.name + "_v", "max_excess": self.excess, "violating_frames": [1]},
                {"name": self.name + "_a", "max_excess": self.excess * 2, "violating_frames": [2]},
            ]}

    objective, poses, _ = _objective()
    guards = objective.candidate_guards = [Guard("hip", .01), Guard("knee", .02)]
    monkeypatch.setattr(solver, "MAX_REJECTIONS", 2)
    trace = PerformanceTrace(stage="contact_acquisition")
    restore = trace.install(objective)
    try:
        solved, history = optimize(objective, poses, iterations=3)
    finally:
        restore()
    np.testing.assert_array_equal(solved, poses)
    assert len(history) == 1
    diagnostic = history[0]["iteration_performance"]
    assert diagnostic["accepted_step_backend"] is None
    assert diagnostic["banded_feasible_alpha"] == [0., 0.]
    assert diagnostic["proposal_calls"] == {"banded": 2}
    assert len(diagnostic["binding_rows"]) == 4
    assert {row["joint_or_contact"] for row in diagnostic["binding_rows"]} == {"hip_v", "hip_a", "knee_v", "knee_a"}
    assert guards[0].calls > 0
    assert guards[1].calls == 0  # Diagnostics never re-run acceptance predicates.
    assert all("accepts" not in guard.__dict__ for guard in guards)
    assert trace.report()["stop_reasons"] == {"rejected_iteration": 1}


def test_instrumentation_restores_all_hooks_after_exception():
    import clarabel
    import cvxpy as cp
    from omniretargeting._fast import shape_step

    original = clarabel.DefaultSolver, clarabel.DefaultSolver.solve, cp.Problem.get_problem_data, shape_step.restore_shape_step
    with pytest.raises(RuntimeError, match="diagnostic test"):
        with PerformanceTrace().instrument():
            raise RuntimeError("diagnostic test")
    assert original == (clarabel.DefaultSolver, clarabel.DefaultSolver.solve, cp.Problem.get_problem_data, shape_step.restore_shape_step)




def test_numerical_retry_is_counted_without_changing_retry_options(monkeypatch):
    import clarabel
    import cvxpy as cp
    from omniretargeting.conic_solver import solve_conic

    original = clarabel.DefaultSolver.solve
    actual_calls = []

    def solve(self):
        actual_calls.append(True)
        if len(actual_calls) == 1:
            raise cp.error.SolverError("forced numerical stall")
        return original(self)

    monkeypatch.setattr(clarabel.DefaultSolver, "solve", solve)
    variable = cp.Variable(1)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(variable)), [variable >= 1.])
    trace = PerformanceTrace(stage="contact_acquisition")
    with trace.instrument():
        solve_conic(problem, solver=cp.CLARABEL)
    assert len(actual_calls) == 2
    assert problem.status == cp.OPTIMAL
    assert variable.value[0] == pytest.approx(1., abs=1e-6)
    operations = trace.report()["operations"]["contact_acquisition"]
    assert operations["numerical_solve"]["calls"] == 2
    assert operations["canonicalization"]["calls"] == 2


def test_helpers_imported_during_trace_do_not_retain_trace_wrappers(monkeypatch):
    import sys
    from types import ModuleType
    from omniretargeting._fast import shape_step

    original = shape_step.restore_shape_step
    module = ModuleType("omniretargeting._fast.trace_test_import")
    with PerformanceTrace().instrument():
        module.restore_shape_step = shape_step.restore_shape_step
        assert module.restore_shape_step is not original
        monkeypatch.setitem(sys.modules, module.__name__, module)
    assert module.restore_shape_step is original
