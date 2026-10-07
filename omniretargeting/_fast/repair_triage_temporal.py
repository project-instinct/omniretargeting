"""Outward interval propagation of the repair's existing temporal rows."""

import numpy as np

from .repair_limits import NUMERICAL_SLACK


def tighten(limits, values, low, high, sweeps=4):
    """Intersect boxes with necessary per-joint stencil intervals.

    Each update isolates one row coefficient and bounds all other terms over
    the previous box. Finite sweeps need not converge: every intermediate box
    still contains every feasible point. Unsupported row structure falls back.
    This does not assemble contact/layout rows, cones or a solver program.
    """
    low, high = low.ravel().copy(), high.ravel().copy()
    signal = values
    programs = []
    for operator, budget in getattr(limits, 'differences', ()):
        op = operator.tocsr()
        lengths = np.diff(op.indptr)
        if not len(lengths) or not np.all(lengths == lengths[0]) or lengths[0] > 4:
            continue
        count = int(lengths[0])
        columns = op.indices.reshape(-1, count)
        coeff = op.data.reshape(-1, count)
        if not np.isfinite(coeff).all() or np.any(coeff == 0):
            continue
        current = op @ signal.ravel()
        slack = np.minimum(NUMERICAL_SLACK,
                           np.maximum(NUMERICAL_SLACK / 4, np.abs(current) - budget))
        lower = -budget - slack - current - 1e-10
        upper = budget + slack - current + 1e-10
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            continue
        programs.append((columns, coeff, lower, upper))
    for _ in range(sweeps):
        for columns, coeff, lower, upper in programs:
            bottom = np.where(coeff > 0., coeff * low[columns], coeff * high[columns])
            top = np.where(coeff > 0., coeff * high[columns], coeff * low[columns])
            for k in range(columns.shape[1]):
                # Sum the other (at most three) terms directly; subtracting a
                # term from a full sum would amplify cancellation roundoff.
                others = np.arange(columns.shape[1]) != k
                lo = lower - top[:, others].sum(axis=1)
                hi = upper - bottom[:, others].sum(axis=1)
                c = coeff[:, k]
                a, b = np.minimum(lo / c, hi / c), np.maximum(lo / c, hi / c)
                a -= 1e-10 * (1. + np.abs(a))
                b += 1e-10 * (1. + np.abs(b))
                np.maximum.at(low, columns[:, k], a)
                np.minimum.at(high, columns[:, k], b)
        if np.any(low > high):
            return None  # Infeasibility/roundoff is not an existing-gate proof.
    return low.reshape(values.shape), high.reshape(values.shape)
