"""Exact affine restriction of a conic model, retaining crossing and constant rows."""

from copy import copy
from dataclasses import replace

import numpy as np
from scipy import sparse

from .qp_assembly import quadratic
from .qp_types import QPSolution, solve_program


def _prune(program):
    A = program.A.tocsr()
    touched = np.diff(A.indptr) != 0
    linear = np.flatnonzero(touched[:program.nonneg] | (program.b[:program.nonneg] < 0))
    rows, sizes, start = [linear], [], program.nonneg
    for size in program.soc:
        values = program.b[start:start + size]
        if touched[start:start + size].any() or np.linalg.norm(values[1:]) > values[0]:
            rows.append(start + np.arange(size))
            sizes.append(size)
        start += size
    rows = np.concatenate(rows).astype(int)
    mapping = np.full(len(program.b), -1, int)
    mapping[rows] = np.arange(len(rows))
    soft = copy(program.soft)
    soft.heads = [(mapping[heads], keys) for heads, keys in soft.heads]
    # Every soft head touches its explicit slack, even if its step row is zero.
    assert all(np.all(heads >= 0) for heads, _ in soft.heads)
    return replace(program, A=A[rows].tocsc(), b=program.b[rows], nonneg=len(linear),
                   soc=tuple(sizes), soft=soft)


def restrict_model(model, P, c, transform, fixed, *, lower=None, upper=None, program=None):
    """Substitute delta=fixed+transform@y, leaving every soft slack free."""
    transform = sparse.csc_matrix(transform)
    fixed = np.asarray(fixed)
    soft = copy(model.soft)
    soft.size = transform.shape[1]
    soft.rows = [(A @ transform, bound - A @ fixed, keys) for A, bound, keys in soft.rows]
    soft.cones = [(A @ transform, residual + (A @ fixed).reshape(-1, 3), bound, keys)
                  for A, residual, bound, keys in soft.cones]
    lift = sparse.block_diag((transform, sparse.eye(len(soft.keys))), format='csc')
    offset = np.r_[fixed, np.zeros(len(soft.keys))]
    full = model.program if program is None else program
    restricted = replace(full, A=(full.A @ lift).tocsc(), b=full.b - full.A @ offset,
                          step_size=transform.shape[1], soft=soft, hard_blocks=())
    if lower is not None:
        size = transform.shape[1]
        rows = sparse.hstack((sparse.vstack((sparse.eye(size), -sparse.eye(size))),
                              sparse.csr_matrix((2 * size, len(soft.keys)))), format='csc')
        at = restricted.nonneg
        A = sparse.vstack((restricted.A[:at], rows, restricted.A[at:]), format='csc')
        b = np.r_[restricted.b[:at], upper, -np.asarray(lower), restricted.b[at:]]
        soft.heads = [(heads + (heads >= at) * 2 * size, keys) for heads, keys in soft.heads]
        restricted = replace(restricted, A=A, b=b, nonneg=at + 2 * size)
    restricted.A.eliminate_zeros()
    restricted = _prune(restricted)
    submodel = replace(model, hessian=(transform.T @ model.hessian @ transform).tocsc(),
                       gradient=np.asarray(transform.T @ (model.gradient + model.hessian @ fixed)),
                       program=restricted, soft=restricted.soft, priority_cache=None, solve_diagnostics=[],
                       step_lower=(np.full((1, transform.shape[1]), -np.inf) if lower is None
                                   else np.asarray(lower).reshape(1, -1)),
                       step_upper=(np.full((1, transform.shape[1]), np.inf) if upper is None
                                   else np.asarray(upper).reshape(1, -1)))
    return submodel, (lift.T @ P @ lift).tocsc(), np.asarray(lift.T @ (c + P @ offset))


def solve_restricted(model, damping, transform, fixed, *, priority=False, lower=None, upper=None,
                     program=None):
    P, c = quadratic(model, damping)
    submodel, P, c = restrict_model(model, P, c, transform, fixed, lower=lower, upper=upper, program=program)
    from .runtime_options import OPTIONS
    if priority and OPTIONS.priority_zero_yield_descent:
        from .descent_step import solve_descent
        result = solve_descent(submodel, P, c)
    elif submodel.soft.keys:
        from .priority_qp import solve_priority
        result = solve_priority(submodel, P, c, strict=priority)
    else:
        solution = solve_program(submodel.program, P, c)
        result = None if solution is None else QPSolution(np.asarray(solution.x))
    if result is None:
        return None
    reduced = result.step.copy()
    result.step = np.asarray(fixed + transform @ reduced)
    if result.certificate:
        reductions = model.predicted_reductions(result.step)
        result.certificate['predicted_reductions'] = dict(
            levels=model.soft.level_values(reductions).tolist(), keys=reductions)
        result.certificate['domain'] = 'affine_restriction'
    return result, reduced
