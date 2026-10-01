"""Equivalent bounded least-squares backends for sparse trajectory steps."""

import numpy as np
from scipy import sparse
from scipy.optimize import OptimizeResult, lsq_linear

from ..conic_solver import gpu_enabled, gpu_strict, solve_conic


def bounded_step(matrix, rhs, lower, upper, config):
    if gpu_enabled():
        import cvxpy as cp

        step = cp.Variable(matrix.shape[1])
        problem = cp.Problem(
            cp.Minimize(cp.sum_squares(matrix @ step - rhs)),
            [step >= lower, step <= upper],
        )
        solve_conic(problem, solver=cp.CLARABEL)
        if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} and step.value is not None:
            value = np.asarray(step.value)
            if np.isfinite(value).all() and np.all(value >= lower - 1e-7) and np.all(value <= upper + 1e-7):
                return OptimizeResult(
                    x=np.clip(value, lower, upper), status=1, success=True,
                    backend="qoco_cuda" if problem.solver_stats.solver_name == "QOCO" else "clarabel_gpu_fallback",
                    message=problem.status,
                )
        if gpu_strict():
            raise RuntimeError(f"GPU bounded step failed: {problem.status}")
    backend = config.get("linear_solver", "lsmr")
    if backend not in ("lsmr", "sparse_qp"):
        raise ValueError("linear_solver must be lsmr or sparse_qp")
    if backend == "sparse_qp":
        import osqp

        # Diagonal column equilibration changes neither objective nor bounds.
        hessian = (matrix.T @ matrix).tocsc()
        diagonal = hessian.diagonal()
        scale = np.ones(len(diagonal))
        nonzero = diagonal > 0
        scale[nonzero] = 1 / np.sqrt(diagonal[nonzero])
        scaling = sparse.diags(scale, format="csc")
        tolerance = float(config.get("linear_solver_tolerance", 1e-6))
        iterations = int(config.get("linear_solver_iterations", 10000))
        if not np.isfinite(tolerance) or tolerance <= 0 or iterations < 1:
            raise ValueError(
                "Linear solver tolerance and iteration limit must be positive"
            )
        solver = osqp.OSQP()
        solver.setup(
            P=sparse.triu(scaling @ hessian @ scaling, format="csc"),
            q=-scale * np.asarray(matrix.T @ rhs).ravel(),
            A=sparse.eye(len(scale), format="csc"),
            l=lower / scale,
            u=upper / scale,
            verbose=False,
            eps_abs=tolerance,
            eps_rel=tolerance,
            max_iter=iterations,
            polishing=True,
        )
        solved = solver.solve(raise_error=False)
        if solved.info.status_val == 1 and np.isfinite(solved.x).all():
            step = np.clip(solved.x * scale, lower, upper)
            return OptimizeResult(
                x=step,
                status=1,
                success=True,
                backend="sparse_qp",
                message=solved.info.status,
            )
        # Never accept an infeasible or unfinished QP as a valid linear solve.
        # The established rectangular least-squares backend is the fallback.
    result = lsq_linear(
        matrix, rhs, bounds=(lower, upper), lsmr_tol=1e-4, tol=1e-4, max_iter=50
    )
    result.backend = "lsmr" if backend == "lsmr" else "lsmr_fallback"
    return result
