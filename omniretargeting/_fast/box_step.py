"""Bounded banded directions with primal/dual active-variable updates.

Only scalar position boxes enter this subproblem. All original nonlinear,
contact and temporal guards still check the resulting trajectory. Exhausting
the numerical work budget requests the existing conic fallback, never a
clipped or partly solved direction.
"""
import numpy as np
from scipy.linalg import LinAlgError


MAX_FACTORIZATIONS = 32


class BoxStepFailure(RuntimeError):
    """No certified box direction within the fixed factorization budget."""


def banded_product(matrix, vector, *, absolute=False):
    """Symmetric lower-band matrix-vector product without a dense expansion."""
    values = np.abs(matrix) if absolute else matrix
    result = values[0] * vector
    for offset in range(1, len(values)):
        row = values[offset, :-offset]
        result[offset:] += row * vector[:-offset]
        result[:-offset] += row * vector[offset:]
    return result


def direction_bounds(objective, poses, held):
    coords, width = objective.coords, objective.width
    lower = np.full(len(poses) * width, -np.inf)
    upper = np.full_like(lower, np.inf)
    indices = np.arange(len(poses))[:, None] * width + coords.scalar_cols
    values = poses[:, coords.qi]
    lower[indices] = objective.lower - values
    upper[indices] = objective.upper - values
    lower[held] = upper[held] = 0.
    return lower, upper


def solve_box_banded(matrix, rhs, lower, upper, solve, hold, *, limit=MAX_FACTORIZATIONS):
    """Solve the box QP, fixing violated directions and releasing wrong multipliers.

    The active set changes with the coupled direction, not the uncoupled
    negative gradient. Equality elimination includes nonzero distances to a
    bound, so other variables compensate before any trajectory is integrated.
    Repeated active sets or the work cap fall back to the constrained solver.
    """
    if np.any(lower > upper):
        raise BoxStepFailure('inconsistent position bounds')
    active = np.zeros(len(rhs), dtype=np.int8)
    fixed = lower == upper
    active[fixed] = 2
    seen = set()
    for _ in range(limit):
        key = active.tobytes()
        if key in seen:
            raise BoxStepFailure('cycling box working set')
        seen.add(key)
        indices = np.flatnonzero(active)
        anchor = np.zeros_like(rhs)
        anchor[(active == -1) | fixed] = lower[(active == -1) | fixed]
        anchor[active == 1] = upper[active == 1]
        system = matrix.copy(order='F')
        adjusted = rhs - banded_product(matrix, anchor)
        hold(system, indices)
        adjusted[indices] = anchor[indices]
        try:
            step = solve(system, adjusted)
        except LinAlgError as error:
            raise BoxStepFailure('box factorization failed') from error
        if not np.isfinite(step).all():
            raise BoxStepFailure('nonfinite box direction')
        free = active == 0
        below, above = free & (step < lower), free & (step > upper)
        if below.any() or above.any():
            active[below], active[above] = -1, 1
            continue
        residual = banded_product(matrix, step) - rhs
        # Algebraic stationarity roundoff only; no physical bound is widened.
        tolerance = 64 * np.finfo(float).eps * (
            banded_product(matrix, np.abs(step), absolute=True) + np.abs(rhs) + 1.)
        release = ((active == -1) & (residual < -tolerance)
                   | (active == 1) & (residual > tolerance))
        if release.any():
            active[release] = 0
            continue
        if np.any(step < lower) or np.any(step > upper):
            raise BoxStepFailure('box equality roundoff')
        return step
    raise BoxStepFailure('box factorization budget exhausted')
