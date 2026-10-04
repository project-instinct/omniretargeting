"""Coupled guard/box working sets using a banded augmented Schur solve.

Each working set factors H + rho A' A once. The multiplier Schur operator
reuses that factor; no dense Schur matrix or sparse normal product is built.
Penalty elimination conditions the equality solve, rather than relaxing its
constraints. Original nonlinear guards still certify every executed step.
"""

import time

import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded
from scipy.sparse.linalg import LinearOperator, cg

from .barrier_assembly import scatter

PENALTY = 1000.
DUAL_REGULARIZATION = 1e-10
MAX_WORKING_SETS = 8
MAX_SCHUR_ITERATIONS = 80


class ActiveSetFailure(RuntimeError):
    pass


def normalized_rows(matrix, bounds, diagonal):
    norm = np.sqrt(np.asarray(matrix.power(2) @ (1 / diagonal)).ravel())
    valid = norm > 1e-15
    if np.any((~valid) & (bounds < -1e-12)):
        raise ActiveSetFailure('binding constraint has no free tangent column')
    scale = np.divide(1., norm, out=np.zeros_like(norm), where=valid)
    return matrix.multiply(scale[:, None]).tocsr(), bounds * scale, valid


def equality_step(system, gradient, matrix, bounds, multiplier, diagnostics,
                  *, regularization=DUAL_REGULARIZATION):
    """Equality-constrained QP through a matrix-free augmented Schur operator.

    A tiny dual regularization handles dependent working-set rows. Its
    residual is checked; wrong-sign multipliers are dropped before a final
    feasibility/stationarity certificate. It is never a guard tolerance.
    """
    ab = system.copy(order='F')
    if matrix.shape[0]:
        scatter(ab, matrix, PENALTY)
    scale = 1 / np.sqrt(ab[0])
    for offset in range(len(ab)):
        length = ab.shape[1] - offset
        ab[offset, :length] *= scale[offset:] * scale[:length]
    diagnostics['factorizations'] += 1
    factor = cholesky_banded(ab, lower=True, overwrite_ab=True, check_finite=False)

    def inverse(rhs):
        diagnostics['band_solves'] += 1
        return scale * cho_solve_banded((factor, True), rhs * scale, check_finite=False)

    direction = inverse(-gradient + PENALTY * (matrix.T @ bounds))
    if not matrix.shape[0]:
        return direction, np.empty(0)

    def product(value):
        return PENALTY * (matrix @ inverse(matrix.T @ value)) + regularization * value

    operator = LinearOperator((len(bounds), len(bounds)), matvec=product, dtype=float)
    rhs = PENALTY * (matrix @ direction - bounds)
    count = [0]

    def iteration(_):
        count[0] += 1

    multiplier, info = cg(operator, rhs, x0=multiplier, rtol=1e-10, atol=1e-10,
                          maxiter=MAX_SCHUR_ITERATIONS, callback=iteration)
    diagnostics['schur_iterations'] += count[0]
    diagnostics['schur_incomplete'] += bool(info)
    direction -= inverse(matrix.T @ multiplier)
    return direction, multiplier


def solve_active(system, gradient, matrix, bounds, warm, diagnostics,
                 *, limit=MAX_WORKING_SETS):
    """Add primal violations and drop negative inequality multipliers in batches."""
    matrix, bounds, valid = normalized_rows(matrix, bounds, system[0])
    active = np.zeros(len(bounds), bool)
    multipliers = np.zeros(len(bounds))
    if warm.get('active') is not None and len(warm['active']) == len(bounds):
        active[:] = warm['active'] & valid
        multipliers[:] = warm['multipliers']
        diagnostics['warm_rows'] = int(active.sum())
    seen = set()
    selective=warm.get('selective',False)
    regularization=1e-4 if selective else DUAL_REGULARIZATION
    for _ in range(limit):
        key = active.tobytes()
        if key in seen:
            raise ActiveSetFailure('cycling constraint working set')
        seen.add(key)
        diagnostics['working_sets'] += 1
        tick = time.monotonic()
        direction, dual = equality_step(system, gradient, matrix[active], bounds[active],
                                        multipliers[active], diagnostics,regularization=regularization)
        excess = matrix @ direction - bounds
        diagnostics.setdefault('working_trace', []).append(dict(rows=int(active.sum()),
            excess=float(excess.max(initial=0.)), min_dual=float(dual.min(initial=0.)),
            seconds=time.monotonic() - tick, schur_iterations=diagnostics['schur_iterations']))
        if not np.isfinite(direction).all() or not np.isfinite(dual).all():
            raise ActiveSetFailure('nonfinite equality-constrained direction')
        multipliers[active] = dual
        # A negative multiplier means that an inequality should be released.
        drop = active & (multipliers < -1e-8)
        if drop.any():
            if selective and drop.sum()>64:
                chosen=np.flatnonzero(drop)[np.argsort(multipliers[drop])[:64]]
                drop[:]=False;drop[chosen]=True
            active[drop] = False
            multipliers[:] = 0.
            diagnostics['dropped_rows'] += int(drop.sum())
            continue
        add = (~active) & valid & (excess > 1e-9)
        if add.any():
            if selective and add.sum()>64:
                chosen=np.flatnonzero(add)[np.argsort(excess[add])[-64:]]
                add[:]=False;add[chosen]=True
            active[add] = True
            diagnostics['added_rows'] += int(add.sum())
            continue
        # Numerical accuracy of the QP, separate from exact physical guards.
        diagnostics['max_normalized_excess'] = float(excess.max(initial=0.))
        if np.any(excess[valid] > 1e-8):
            if selective and regularization>DUAL_REGULARIZATION:
                regularization=DUAL_REGULARIZATION
                seen.discard(key)
                continue
            raise ActiveSetFailure('equality Schur residual did not converge')
        warm.update(active=active.copy(), multipliers=multipliers.copy())
        diagnostics['active_rows'] = int(active.sum())
        return direction
    raise ActiveSetFailure('constraint working-set budget exhausted')
