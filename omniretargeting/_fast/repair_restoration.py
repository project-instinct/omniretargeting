"""Restore curved IK constraints without changing a reference or allowance."""

import cvxpy as cp
import numpy as np

from .._sequence.restoration import RESTORATION_ITERATIONS
from ..conic_solver import solve_conic


def restore_repair(
    limits, origin, candidate, geometry, measurement, baseline, projection
):
    """Return a nearby admissible proposal; original geometry is checked later.

    Only the caller's projected scalar coordinates can move. Collision rows
    protect the old verified witnesses; nonlinear original-mesh validation is
    still required before committing the result.
    """
    if not projection.shape[1]:
        return None, {"status": "no_local_dofs"}
    original_proposal = np.array(candidate, copy=True)
    radius = 0.5 * np.linalg.norm(
        candidate[:, limits.coords.qi] - origin[:, limits.coords.qi]
    )
    if radius < 1e-10:
        return None, {"status": "stationary"}
    collision, phi = geometry.linear_rows(measurement)
    budgets = np.array(
        [baseline["depths"].get(key, 0.0) for key, _ in measurement["rows"]]
    )
    for iteration in range(RESTORATION_ITERATIONS):
        if limits.check(candidate)[0]:
            return candidate, {"status": "restored", "iterations": iteration}
        # Only mesh-repair restoration uses signed layout margins.
        bounds = limits.constraints(candidate, restore_layout=True)
        offset = (candidate[:, limits.coords.qi] - origin[:, limits.coords.qi]).ravel()
        bounds.add(
            collision, -np.maximum(budgets, -phi) - phi - collision @ offset, np.inf
        )
        matrix, lo, hi = bounds.packed()
        matrix = matrix @ projection
        matrix.eliminate_zeros()
        variable = np.asarray(matrix.getnnz(axis=1)).ravel() > 0
        lower = np.isfinite(lo) & (variable | (lo > 0))
        upper = np.isfinite(hi) & (variable | (hi < 0))
        delta = cp.Variable(projection.shape[1])
        constraints = [
            matrix[lower] @ delta >= lo[lower],
            matrix[upper] @ delta <= hi[upper],
        ]
        movement = (
            candidate[:, limits.coords.qi] - original_proposal[:, limits.coords.qi]
        ).ravel()
        constraints.append(cp.norm(projection @ delta + movement) <= radius)
        for jac, residual, allowed in bounds.cones():
            jac = (jac @ projection).tocsr()
            jac.eliminate_zeros()
            size = residual.shape[1]
            varying = np.asarray(jac.getnnz(axis=1)).reshape(-1, size).sum(axis=1) > 0
            keep = varying | (np.linalg.norm(residual, axis=1) > allowed + 1e-12)
            if not keep.any():
                continue
            jac, residual, allowed = (
                jac[np.repeat(keep, size)],
                residual[keep],
                allowed[keep],
            )
            scale = np.maximum(allowed, 1e-3)
            value = cp.reshape(jac @ delta, residual.shape, order="C") + residual
            constraints.append(
                cp.norm(cp.multiply(1 / scale[:, None], value), axis=1)
                <= allowed / scale
            )
        problem = cp.Problem(
            cp.Minimize(cp.sum_squares(delta) / radius**2), constraints
        )
        try:
            solve_conic(problem, solver="CLARABEL", direct_solve_method="qdldl")
        except cp.error.SolverError:
            return None, {"status": "solver_error", "iterations": iteration}
        if (
            problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
            or delta.value is None
        ):
            return None, {"status": str(problem.status), "iterations": iteration}
        step = np.asarray(projection @ delta.value).reshape(limits.frames, limits.width)
        if not np.isfinite(step).all():
            return None, {"status": "nonfinite", "iterations": iteration}
        candidate = candidate.copy()
        candidate[:, limits.coords.qi] += step
    valid, check = limits.check(candidate)
    return (candidate if valid else None), {
        "status": "restored" if valid else "unresolved",
        "iterations": RESTORATION_ITERATIONS,
        "check": check,
    }
