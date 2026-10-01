"""Minimum-change restoration of nonlinear trajectory constraints."""

import cvxpy as cp
import numpy as np
from scipy import sparse
from ..conic_solver import solve_conic, solve_conic_working_set

# Numerical retries, not an additional physical correction allowance.
RESTORATION_ITERATIONS = 4


def restore_candidate(
    guard, origin, candidate, lower, upper, priority_bounds, frames=None, *, working_set=False
):
    """``frames`` restores a window step: frames outside it stay unchanged.

    ``working_set`` solves each restoration by exact constraint generation
    (``solve_conic_working_set``): the same optimum, found on the few rows a
    small correction can reach."""
    solve = solve_conic_working_set if working_set else solve_conic
    coords = guard.tasks.coords
    count, width = lower.shape
    window = range(len(candidate)) if frames is None else frames
    rows = slice(window.start, window.stop)
    scales = np.maximum(np.maximum(abs(lower), abs(upper)), np.finfo(float).eps)
    weight = sparse.diags(1 / scales.ravel() ** 2, format="csc")

    def admissible(poses):
        errors = guard.priority_errors(poses, frames) if priority_bounds else None
        return guard.accepts(poses, frames) and all(
            errors[frame - window.start, level] <= bound + 1e-8
            for (level, frame), bound in priority_bounds.items()
        )

    for _ in range(RESTORATION_ITERATIONS):
        if admissible(candidate):
            return candidate
        # Keep the original step box as well as the immutable cumulative bounds.
        displacement = coords.difference(origin[rows], candidate[rows])
        delta = cp.Variable(count * width)
        constraints = guard.restoration_constraints(
            candidate,
            delta,
            lower - displacement,
            upper - displacement,
            priority_bounds,
            frames,
        )
        # A guard may defer constraints it separates at the solved step
        # (``separate`` returns the violated ones, empty when none remain).
        separate = getattr(guard, "separate", None)
        for rounds in range(64):
            problem = cp.Problem(
                cp.Minimize(cp.quad_form(delta, cp.psd_wrap(weight))), constraints
            )
            try:
                solve(problem, solver=cp.CLARABEL, direct_solve_method="qdldl")
            except cp.error.SolverError:
                return None
            if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                return None
            if delta.value is None or not np.all(np.isfinite(delta.value)):
                return None
            more = [] if separate is None else separate(delta, rounds)
            if not more:
                break
            constraints = constraints + more
        else:
            return None
        step = np.zeros((len(candidate), width))
        step[rows] = delta.value.reshape(count, width)
        candidate = coords.integrate(candidate, step, 1)
    return candidate if admissible(candidate) else None
