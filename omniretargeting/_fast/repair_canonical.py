"""Assemble CVXPY's existing repair SOCP without expression canonicalization.

Keep its auxiliary norm variables, cone order, scaling and CPU solver adapter.
This is different from the experimental reduced direct-restoration program:
it reproduces the established numerical program, including retry solver reuse.
"""

from types import SimpleNamespace

import cvxpy as cp
import numpy as np
from scipy import sparse
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import CLARABEL

from ..conic_solver import RETRY_TOLERANCE

# Numerical-path experiment. Canonical data/allowances remain unchanged;
# solver arithmetic can differ, so retain QDLDL until full per-item approval.
FAER_MIN_VARIABLES = 2048


def canonical_data(bounds, projection, selected, orthogonal, radius):
    variables = projection.shape[1]
    matrix, lo, hi = bounds.packed()
    matrix = (matrix @ projection).tocsr()
    matrix.eliminate_zeros()
    varying = np.asarray(matrix.getnnz(axis=1)).ravel() > 0
    lower = np.isfinite(lo) & (varying | (lo > 0))
    upper = np.isfinite(hi) & (varying | (hi < 0))
    linear = sparse.vstack([-matrix[lower], matrix[upper]], format="csc")
    cones, scalar_bounds = [], [np.array([radius])]
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
        inverse = 1 / scale
        # Match cp.multiply's constant arithmetic, including its reciprocal
        # multiplication (division of the residual can round differently).
        jac = (sparse.diags(np.repeat(inverse, dimension)) @ jac).tocoo()
        cones.append((jac, residual * inverse[:, None]))
        scalar_bounds.append(allowed / scale)
    auxiliaries = sum(len(b) for b in scalar_bounds)
    size = variables + auxiliaries
    blocks = [sparse.hstack([linear, sparse.csc_matrix((linear.shape[0], auxiliaries))]),
              sparse.hstack([sparse.csc_matrix((auxiliaries, variables)), sparse.eye(auxiliaries)])]
    targets = [np.r_[-lo[lower], hi[upper]], np.concatenate(scalar_bounds)]
    # The scalar norm bound is a separate nonnegative row. Its auxiliary
    # variable supplies the leading coordinate of the SOC, just as CVXPY does.
    trust = sparse.csc_matrix((np.r_[-1., -np.ones(variables)],
                               (np.arange(variables + 1), np.r_[variables, np.arange(variables)])),
                              shape=(variables + 2, size))
    blocks.append(trust)
    targets.append(np.r_[0., selected, orthogonal])
    sizes, auxiliary = [variables + 2], variables + 1
    for jac, residual in cones:
        count, dimension = residual.shape
        row = jac.row + jac.row // dimension + 1
        leading = np.arange(count) * (dimension + 1)
        blocks.append(sparse.csc_matrix((np.r_[-jac.data, -np.ones(count)],
                                        (np.r_[row, leading],
                                         np.r_[jac.col, auxiliary + np.arange(count)])),
                                       shape=(count * (dimension + 1), size)))
        targets.append(np.c_[np.zeros(count), residual].ravel())
        sizes.extend([dimension + 1] * count)
        auxiliary += count
    quadratic = sparse.csc_matrix((np.full(variables, 2. * (1. / radius**2)),
                                   (np.arange(variables), np.arange(variables))), shape=(size, size))
    offset = np.concatenate(targets)
    # CVXPY's sparse affine offset canonicalization discards signed zeros.
    offset[offset == 0.] = 0.
    return dict(P=quadratic, c=np.zeros(size), A=sparse.vstack(blocks, format="csc"),
                b=offset, dims=SimpleNamespace(zero=0, nonneg=linear.shape[0] + auxiliaries,
                                                               soc=sizes, psd=[], exp=0, p3d=[], pnd=[]))


def solve_data(data, variables):
    adapter, cache = CLARABEL(), {}
    options = dict(direct_solve_method="qdldl")
    for retry in (False, True):
        if retry:
            options.update(tol_gap_abs=RETRY_TOLERANCE, tol_gap_rel=RETRY_TOLERANCE,
                           tol_feas=RETRY_TOLERANCE)
        try:
            result = adapter.solve_via_data(data, warm_start=True, verbose=False,
                                           solver_opts=options, solver_cache=cache)
        except cp.error.SolverError:
            if retry:
                return None, cp.SOLVER_ERROR
            continue
        status = CLARABEL.STATUS_MAP[str(result.status)]
        if status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return np.asarray(result.x[:variables]), status
        if status in (cp.INFEASIBLE, cp.UNBOUNDED) or retry:
            return None, status
    return None, cp.SOLVER_ERROR
