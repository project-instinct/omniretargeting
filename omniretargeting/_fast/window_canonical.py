"""Assemble the established window numerical program without CVXPY expressions.

Unlike window_conic, retain every norm epigraph variable and nonnegative row.
The objective arithmetic, sparse ordering and cone packing match CVXPY; no
constraint reduction, alternate scaling or changed solver path is intended.
"""

from types import SimpleNamespace

import cvxpy as cp
import numpy as np
from scipy import sparse
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import CLARABEL

from ..conic_solver import solve_conic, solve_conic_working_set
from .proposals import guards_of


def canonical_data(objective, poses, lower, upper, quadratic, linear):
    variables = lower.size
    eye = sparse.eye(variables, format='csr')
    rows, targets, vectors = [-eye, eye], [-lower.ravel(), upper.ravel()], []
    auxiliary = 0
    # CVXPY groups the nonnegative rows before the SOCs, preserving their
    # original constraint order within each family.
    for guard in guards_of(objective)[1]:
        intervals, cones = guard.qp_bounds(poses)
        for matrix, lo, hi in intervals:
            lo, hi = np.asarray(lo), np.asarray(hi)
            finite_lo, finite_hi = np.isfinite(lo), np.isfinite(hi)
            if finite_lo.any():
                rows.append(-matrix[finite_lo])
                targets.append(-lo[finite_lo])
            if finite_hi.any():
                rows.append(matrix[finite_hi])
                targets.append(hi[finite_hi])
        for matrix, residual, allowed in cones:
            count = len(allowed)
            if not count:
                continue
            scale = 1 / np.maximum(allowed, 1e-3)
            dimension = residual.shape[1]
            jac = (sparse.diags(np.repeat(scale, dimension)) @ matrix).tocoo()
            vectors.append((jac, residual * scale[:, None], auxiliary))
            # Placeholder: these are columns in the norm epigraph block.
            rows.append((auxiliary, count))
            targets.append(allowed * scale)
            auxiliary += count
    size = variables + auxiliary
    blocks = []
    for row in rows:
        if isinstance(row, tuple):
            start, count = row
            block = sparse.csc_matrix((np.ones(count),
                                       (np.arange(count), variables + start + np.arange(count))),
                                      shape=(count, size))
        else:
            block = sparse.hstack([row, sparse.csc_matrix((row.shape[0], auxiliary))], format='csc')
        blocks.append(block)
    nonnegative, sizes = sum(len(value) for value in targets), []
    for jac, residual, start in vectors:
        count, dimension = residual.shape
        leading = np.arange(count) * (dimension + 1)
        row = jac.row + jac.row // dimension + 1
        blocks.append(sparse.csc_matrix((np.r_[-jac.data, -np.ones(count)],
                                        (np.r_[row, leading],
                                         np.r_[jac.col, variables + start + np.arange(count)])),
                                       shape=(count * (dimension + 1), size)))
        targets.append(np.c_[np.zeros(count), residual].ravel())
        sizes.extend([dimension + 1] * count)
    quadratic = sparse.block_diag([quadratic, sparse.csc_matrix((auxiliary, auxiliary))], format='csc')
    offset, cost = np.concatenate(targets), np.r_[linear, np.zeros(auxiliary)]
    # Canonical constant stuffing drops signed zeros into a zero-filled array.
    offset[offset == 0.] = 0.
    cost[cost == 0.] = 0.
    return dict(P=quadratic, c=cost,
                A=sparse.vstack(blocks, format='csc'), b=offset,
                dims=SimpleNamespace(zero=0, nonneg=nonnegative, soc=sizes,
                                     psd=[], exp=0, p3d=[], pnd=[]))


class CanonicalProblem:
    """CPU adapter for the unchanged full/retry/working-set solve sequence.

    Only primal pose coordinates are consumed by window proposals. Preserve
    the Clarabel adapter's cache for a numerical retry, while avoiding the
    expression-tree and dual-variable reconstruction of an equivalent Problem.
    """
    def __init__(self, data, variables):
        self.data, self.variables = data, variables
        self.adapter, self.cache = CLARABEL(), {}
        self.status, self.value, self.step = None, None, None
        self.size_metrics = SimpleNamespace(num_scalar_variables=len(data['c']))

    def get_problem_data(self, solver, **kwargs):
        assert solver == cp.CLARABEL
        return self.data, None, None

    def unpack_results(self, result, chain, inverse):
        status = CLARABEL.STATUS_MAP[str(result.status)]
        if status in cp.settings.ERROR:
            raise cp.error.SolverError(f'Clarabel status: {status}')
        self.status = status
        self.step = (np.asarray(result.x[:self.variables], dtype=float)
                     if status in cp.settings.SOLUTION_PRESENT else None)
        self.value = result.obj_val

    def solve(self, *, solver, verbose=False, warm_start=True, **options):
        assert solver == cp.CLARABEL
        result = self.adapter.solve_via_data(self.data, warm_start=warm_start,
                                            verbose=verbose, solver_opts=options,
                                            solver_cache=self.cache)
        self.unpack_results(result, None, None)
        return self.value


def solve_data(data, variables, *, working_set=False):
    problem = CanonicalProblem(data, variables)
    solve = solve_conic_working_set if working_set else solve_conic
    try:
        solve(problem, solver=cp.CLARABEL, direct_solve_method='qdldl')
    except cp.error.SolverError:
        return None
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return None
    step = problem.step
    return step if step is not None and np.isfinite(step).all() else None


def restore_step(objective, origin, step):
    """The legacy four-step restoration with an identical canonical program."""
    from .._sequence.restoration import RESTORATION_ITERATIONS
    from .shape_step import WORKING_SET_RESTORATION, _ShapeRestoration, _pin_root
    coords = objective.coords
    guard = _ShapeRestoration(objective)
    candidate = objective.project(coords.integrate(origin, step.reshape(len(origin), -1), 1.))
    lower, upper = coords.bounds(origin, objective.cfg['angular_step_rad'],
                                 objective.cfg['translation_step_m'])
    scales = np.maximum(np.maximum(abs(lower), abs(upper)), np.finfo(float).eps)
    # Preserve CVXPY's quad_form canonicalization order: divide, then multiply.
    weight = sparse.diags(1 / scales.ravel() ** 2, format='csc') * 2
    for _ in range(RESTORATION_ITERATIONS):
        if guard.accepts(candidate):
            return objective.project(candidate)
        displacement = coords.difference(origin, candidate)
        lo, hi = lower - displacement, upper - displacement
        lo[:, coords.scalar_cols] = np.maximum(lo[:, coords.scalar_cols],
                                               objective.lower - candidate[:, coords.qi])
        hi[:, coords.scalar_cols] = np.minimum(hi[:, coords.scalar_cols],
                                               objective.upper - candidate[:, coords.qi])
        _pin_root(objective, candidate, lo, hi)
        data = canonical_data(objective, candidate, lo, hi, weight, np.zeros(lower.size))
        from .window_restore_reduction import EXACT_WINDOW_RESTORATION, reduce_epigraphs
        if EXACT_WINDOW_RESTORATION:
            data = reduce_epigraphs(data, lower.size)
        from .window_qoco import WINDOW_QOCO_RESTORATIONS, solve_restoration
        delta = solve_restoration(data, lower.size) if WINDOW_QOCO_RESTORATIONS else None
        if delta is None:
            delta = solve_data(data, lower.size, working_set=WORKING_SET_RESTORATION)
        if delta is None:
            return None
        candidate = coords.integrate(candidate, delta.reshape(lower.shape), 1)
    return objective.project(candidate) if guard.accepts(candidate) else None
