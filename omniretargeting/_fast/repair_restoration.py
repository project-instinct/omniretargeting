"""Restore curved IK constraints without changing a reference or allowance."""

import re

import cvxpy as cp
import numpy as np
from scipy import sparse
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import CLARABEL

from .._sequence.restoration import RESTORATION_ITERATIONS
from ..conic_solver import RETRY_TOLERANCE, solve_conic
from .runtime_options import OPTIONS
from .window_qp import WINDOW_FRAMES, solve_standard


def _selected_rows(projection):
    """Row of each column of a pure coordinate selector, or None.

    For a selector ``P`` (one unit entry per column, distinct rows) ``P.T P = I``,
    so ``||P d + m||^2 = ||d + P.T m||^2 + ||m - P P.T m||^2``: the trust cone can
    use the selected coordinates plus one constant component, same feasible set.
    """
    columns = sparse.csc_matrix(projection)
    if (np.diff(columns.indptr) != 1).any() or not np.all(columns.data == 1.):
        return None
    rows = columns.indices
    return rows if len(np.unique(rows)) == len(rows) else None


def _trust_terms(projection, movement):
    """(selected movement, constant orthogonal norm) of a selector, or None."""
    rows = _selected_rows(projection)
    if rows is None:
        return None
    outside = np.ones(len(movement), bool)
    outside[rows] = False
    return movement[rows], float(np.linalg.norm(movement[outside]))


def _restoration_program(bounds, projection, movement, radius):
    """Clarabel form with only delta variables and the unchanged norm budgets."""
    import clarabel

    variables = projection.shape[1]
    matrix, lo, hi = bounds.packed()
    matrix = (matrix @ projection).tocsr()
    matrix.eliminate_zeros()
    varying = np.asarray(matrix.getnnz(axis=1)).ravel() > 0
    lower = np.isfinite(lo) & (varying | (lo > 0))
    upper = np.isfinite(hi) & (varying | (hi < 0))
    linear = sparse.vstack([-matrix[lower], matrix[upper]], format="csc")
    trust = _trust_terms(projection, movement)
    if trust is None:
        targets = [np.r_[-lo[lower], hi[upper]], np.r_[radius, movement]]
        blocks = [linear, sparse.vstack([sparse.csr_matrix((1, variables)), -projection],
                                        format="csc")]
        trust_size = projection.shape[0] + 1
    else:
        selected, orthogonal = trust
        targets = [np.r_[-lo[lower], hi[upper]], np.r_[radius, selected, orthogonal]]
        blocks = [linear, sparse.vstack([sparse.csr_matrix((1, variables)),
                                         -sparse.eye(variables, format="csc"),
                                         sparse.csr_matrix((1, variables))], format="csc")]
        trust_size = variables + 2
    cones = ([clarabel.NonnegativeConeT(linear.shape[0])] if linear.shape[0] else [])
    cones.append(clarabel.SecondOrderConeT(trust_size))
    for jac, residual, allowed in bounds.cones():
        jac = (jac @ projection).tocsr()
        jac.eliminate_zeros()
        dimension = residual.shape[1]
        varying = np.asarray(jac.getnnz(axis=1)).reshape(-1, dimension).sum(axis=1) > 0
        keep = varying | (np.linalg.norm(residual, axis=1) > allowed + 1e-12)
        if not keep.any():
            continue
        jac, residual, allowed = jac[np.repeat(keep, dimension)], residual[keep], allowed[keep]
        count = len(allowed)
        scale = np.maximum(allowed, 1e-3)
        jac = (sparse.diags(np.repeat(1 / scale, dimension)) @ jac).tocoo()
        rows = jac.row + jac.row // dimension + 1
        blocks.append(sparse.csc_matrix((-jac.data, (rows, jac.col)),
                                        shape=(count * (dimension + 1), variables)))
        targets.append(np.c_[allowed / scale, residual / scale[:, None]].ravel())
        cones.extend(clarabel.SecondOrderConeT(dimension + 1) for _ in range(count))
    quadratic = sparse.eye(variables, format="csc") * (2 / radius**2)
    return (quadratic, np.zeros(variables), sparse.vstack(blocks, format="csc"),
            np.concatenate(targets), cones)


def _solve_restoration(program):
    """Match solve_conic's CPU tolerances, single retry and CVXPY status names."""
    import clarabel

    for retry in (False, True):
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        settings.direct_solve_method = "qdldl"
        if retry:
            settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = RETRY_TOLERANCE
        try:
            solution = clarabel.DefaultSolver(*program, settings).solve()
        except (RuntimeError, cp.error.SolverError):
            if retry:
                return None, cp.SOLVER_ERROR
            continue
        status = CLARABEL.STATUS_MAP[str(solution.status)]
        if status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return np.asarray(solution.x), status
        if status in (cp.INFEASIBLE, cp.UNBOUNDED) or retry:
            return None, status
    return None, cp.SOLVER_ERROR


def _solve_windowed(program, frames):
    """Temporal block-coordinate solve of the direct program (``window_qp``), or None.

    Variables are frame-major; the trajectory-wide trust cone enters every
    window whole, with the other windows' fixed part on its right-hand side.
    """
    import clarabel

    quadratic, linear, matrix, offset, cones = program
    variables = len(linear)
    if variables % frames:
        return None
    sizes = [int(re.match(r"\w+\((\d+)\)", str(cone)).group(1)) for cone in cones]
    kinds = [str(cone).split("(")[0] for cone in cones]
    nonneg = sizes[0] if kinds and kinds[0].startswith("Nonnegative") else 0
    soc = sizes[1:] if nonneg else sizes

    def settings():
        value = clarabel.DefaultSettings()
        value.verbose = False
        value.direct_solve_method = "qdldl"
        return value

    return solve_standard(quadratic, linear, matrix, offset, nonneg, soc,
                          np.arange(variables) // (variables // frames), frames, settings=settings)


def _cvxpy_restoration_problem(bounds, projection, movement, radius):
    """Reference expression formulation and fallback for general projections."""
    matrix, lo, hi = bounds.packed()
    matrix = (matrix @ projection).tocsr()
    matrix.eliminate_zeros()
    varying = np.asarray(matrix.getnnz(axis=1)).ravel() > 0
    lower = np.isfinite(lo) & (varying | (lo > 0))
    upper = np.isfinite(hi) & (varying | (hi < 0))
    delta = cp.Variable(projection.shape[1])
    trust = _trust_terms(projection, movement)
    if trust is None:
        constraints = [cp.norm(projection @ delta + movement) <= radius]
    else:
        selected, orthogonal = trust
        constraints = [cp.norm(cp.hstack([delta + selected, np.array([orthogonal])])) <= radius]
    if lower.any():
        constraints.insert(0, matrix[lower] @ delta >= lo[lower])
    if upper.any():
        constraints.insert(int(lower.any()), matrix[upper] @ delta <= hi[upper])
    for jac, residual, allowed in bounds.cones():
        jac = (jac @ projection).tocsr()
        jac.eliminate_zeros()
        dimension = residual.shape[1]
        varying = np.asarray(jac.getnnz(axis=1)).reshape(-1, dimension).sum(axis=1) > 0
        keep = varying | (np.linalg.norm(residual, axis=1) > allowed + 1e-12)
        if not keep.any():
            continue
        jac, residual, allowed = jac[np.repeat(keep, dimension)], residual[keep], allowed[keep]
        scale = np.maximum(allowed, 1e-3)
        value = cp.reshape(jac @ delta, residual.shape, order="C") + residual
        constraints.append(cp.norm(cp.multiply(1 / scale[:, None], value), axis=1) <= allowed / scale)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(delta) / radius**2), constraints)
    return problem, delta


def _solve_cvxpy_restoration(bounds, projection, movement, radius):
    """The established CVXPY numerical program, assembled without its graph."""
    trust = _trust_terms(projection, movement)
    if trust is not None:
        from .repair_canonical import canonical_data, solve_data
        data = canonical_data(bounds, projection, *trust, radius)
        return solve_data(data, projection.shape[1])
    problem, delta = _cvxpy_restoration_problem(bounds, projection, movement, radius)
    try:
        solve_conic(problem, solver="CLARABEL", direct_solve_method="qdldl", cpu_only=True)
    except cp.error.SolverError:
        return None, cp.SOLVER_ERROR
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or delta.value is None:
        return None, problem.status
    return np.asarray(delta.value), problem.status


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
        bounds = limits.constraints(candidate, restore_layout=True, projection=projection)
        offset = (candidate[:, limits.coords.qi] - origin[:, limits.coords.qi]).ravel()
        bounds.add(
            collision, -np.maximum(budgets, -phi) - phi - collision @ offset, np.inf
        )
        movement = (
            candidate[:, limits.coords.qi] - original_proposal[:, limits.coords.qi]
        ).ravel()
        if OPTIONS.direct_repair_restoration:
            program = _restoration_program(bounds, projection, movement, radius)
            delta, status = None, None
            if OPTIONS.window_qp and limits.frames >= 2 * WINDOW_FRAMES:
                delta = _solve_windowed(program, limits.frames)
                status = cp.OPTIMAL
            if delta is None:
                delta, status = _solve_restoration(program)
        else:
            delta, status = _solve_cvxpy_restoration(bounds, projection, movement, radius)
        if delta is None:
            return None, {"status": str(status), "iterations": iteration}
        step = np.asarray(projection @ delta).reshape(limits.frames, limits.width)
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
