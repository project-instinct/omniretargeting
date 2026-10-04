"""Experimental reduced cones; not dispatched by the production solver.

Eliminating norm epigraphs changes the numerical path; the 957-frame trial
failed the per-item quality comparison. window_canonical retains those
variables and reproduces the established numerical program instead.
The nonlinear guards and model/actual-decrease tests still own acceptance.
No row is removed from bounded steps; restoration alone uses the established
constraint-generation constants and complete-problem fallback.
"""

from functools import lru_cache

import clarabel
import cvxpy as cp
import numpy as np
from scipy import sparse
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import CLARABEL

from ..conic_solver import (RETRY_TOLERANCE, WORKING_SET_FRACTION, WORKING_SET_MARGIN,
                           WORKING_SET_ROUNDS, WORKING_SET_SLACK)
from .._sequence.restoration import RESTORATION_ITERATIONS
from .proposals import guards_of
from .qp_guard_rows import norms


@lru_cache(maxsize=32)
def _soc(size):
    # Clarabel reads these descriptors; each solver owns its numerical cones.
    return clarabel.SecondOrderConeT(size)


def _cones(nonnegative, sizes):
    return ([clarabel.NonnegativeConeT(nonnegative)] if nonnegative else []) + [
        _soc(int(size)) for size in sizes]


def guard_program(objective, poses, lower, upper):
    """Copy WindowGuard.constraints algebra, including every crossing row."""
    count = lower.size
    eye = sparse.eye(count, format='csr')
    linear, targets = [-eye, eye], [-lower.ravel(), upper.ravel()]
    cones = []
    for guard in guards_of(objective)[1]:
        intervals, vectors = guard.qp_bounds(poses)
        for matrix, lo, hi in intervals:
            lo, hi = np.asarray(lo), np.asarray(hi)
            finite_lo, finite_hi = np.isfinite(lo), np.isfinite(hi)
            if finite_lo.any():
                linear.append(-matrix[finite_lo])
                targets.append(-lo[finite_lo])
            if finite_hi.any():
                linear.append(matrix[finite_hi])
                targets.append(hi[finite_hi])
        for matrix, residual, allowed in vectors:
            if len(allowed):
                scale = 1 / np.maximum(allowed, 1e-3)
                cones.append(norms(sparse.diags(np.repeat(scale, residual.shape[1])) @ matrix,
                                   residual * scale[:, None], allowed * scale))
    nonnegative = sum(len(v) for v in targets)
    matrix = sparse.vstack([*linear, *(block.A for block in cones)], format='csc')
    offset = np.concatenate([*targets, *(block.b for block in cones)])
    sizes = np.array([size for block in cones for size in block.soc], dtype=int)
    return matrix, offset, nonnegative, sizes


def _settings(retry=False):
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.direct_solve_method = 'qdldl'
    if retry:
        settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = RETRY_TOLERANCE
    return settings


def solve_program(quadratic, linear, constraints, *, working_set=False):
    """Full bounded solve, or exact restoration constraint generation."""
    matrix, offset, nonnegative, sizes = constraints
    quadratic = sparse.triu(quadratic, format='csc')
    if working_set:
        step = _working_set(quadratic, linear, matrix, offset, nonnegative, sizes)
        if step is not None:
            return step
    for retry in (False, True):
        try:
            result = clarabel.DefaultSolver(quadratic, linear, matrix, offset,
                                            _cones(nonnegative, sizes), _settings(retry)).solve()
        except (RuntimeError, cp.error.SolverError):
            if retry:
                return None
            continue
        status = CLARABEL.STATUS_MAP[str(result.status)]
        if status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            step = np.asarray(result.x)
            return step if np.isfinite(step).all() else None
        if status in (cp.INFEASIBLE, cp.UNBOUNDED):
            return None
    return None


def _working_set(quadratic, linear, matrix, offset, nonnegative, sizes):
    """Restoration only: retain complete cones and certify every omitted row."""
    starts = nonnegative + np.r_[0, np.cumsum(sizes)[:-1]].astype(int)
    groups = [(np.flatnonzero(sizes == size), int(size)) for size in np.unique(sizes)]
    indices = [(which, starts[which, None] + np.arange(size)) for which, size in groups]

    def margins(slack):
        values = np.empty(len(sizes))
        for which, rows in indices:
            block = slack[rows]
            values[which] = block[:, 0] - np.linalg.norm(block[:, 1:], axis=1)
        return slack[:nonnegative], values

    lin, con = margins(offset)
    keep_linear, keep_cones = lin <= WORKING_SET_SLACK, con <= WORKING_SET_SLACK
    matrix = matrix.tocsr()
    for _ in range(WORKING_SET_ROUNDS):
        selected = np.flatnonzero(keep_cones)
        rows = np.concatenate([np.flatnonzero(keep_linear)] + [
            starts[index] + np.arange(sizes[index]) for index in selected]).astype(int)
        if len(rows) > WORKING_SET_FRACTION * matrix.shape[0]:
            return None
        result = clarabel.DefaultSolver(
            quadratic, linear, matrix[rows].tocsc(), offset[rows],
            _cones(int(keep_linear.sum()), sizes[selected]), _settings()).solve()
        if str(result.status) != 'Solved':
            return None
        step = np.asarray(result.x)
        lin, con = margins(offset - matrix @ step)
        add_linear = ~keep_linear & (lin < WORKING_SET_MARGIN)
        add_cones = ~keep_cones & (con < WORKING_SET_MARGIN)
        if not add_linear.any() and not add_cones.any():
            return step if np.isfinite(step).all() else None
        keep_linear |= add_linear
        keep_cones |= add_cones
    return None


def step_bounds(objective, poses, lower=None, upper=None):
    from .shape_step import _pin_root
    coords = objective.coords
    if lower is None:
        lower, upper = coords.bounds(poses, objective.cfg['angular_step_rad'],
                                     objective.cfg['translation_step_m'])
    _pin_root(objective, poses, lower, upper)
    lower[:, coords.scalar_cols] = np.maximum(lower[:, coords.scalar_cols],
                                             objective.lower - poses[:, coords.qi])
    upper[:, coords.scalar_cols] = np.minimum(upper[:, coords.scalar_cols],
                                             objective.upper - poses[:, coords.qi])
    return lower, upper


def bounded_step(objective, poses, system, gradient, diagonal, damping):
    count = len(gradient)
    bands = [system[k, :count - k] for k in range(len(system))]
    lower = sparse.diags(bands, -np.arange(len(bands)), shape=(count, count), format='csc')
    matrix = lower + lower.T - sparse.diags(system[0])
    matrix += sparse.diags(damping * diagonal + 1e-9)
    scale = max(float(np.linalg.norm(gradient, ord=np.inf)), 1.)
    constraints = guard_program(objective, poses, *step_bounds(objective, poses))
    return solve_program(matrix * (2 / scale), 2 * gradient / scale, constraints)


def restore_step(objective, origin, step):
    coords = objective.coords
    candidate = objective.project(coords.integrate(origin, step.reshape(len(origin), -1), 1.))
    lower, upper = coords.bounds(origin, objective.cfg['angular_step_rad'],
                                 objective.cfg['translation_step_m'])
    scales = np.maximum(np.maximum(abs(lower), abs(upper)), np.finfo(float).eps)
    weight = sparse.diags(2 / scales.ravel()**2, format='csc')
    guards = guards_of(objective)[1]
    for _ in range(RESTORATION_ITERATIONS):
        if all(guard.accepts(candidate) for guard in guards):
            return objective.project(candidate)
        displacement = coords.difference(origin, candidate)
        lo, hi = step_bounds(objective, candidate, lower - displacement, upper - displacement)
        constraints = guard_program(objective, candidate, lo, hi)
        delta = solve_program(weight, np.zeros(lower.size), constraints, working_set=True)
        if delta is None:
            return None
        candidate = coords.integrate(candidate, delta.reshape(lower.shape), 1.)
    return objective.project(candidate) if all(guard.accepts(candidate) for guard in guards) else None
