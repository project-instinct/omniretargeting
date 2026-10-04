"""Conservative box lower bounds that can skip impossible zero-level proofs."""

import math

import numpy as np


def box_level_lower_bounds(model):
    """Lower bounds only, never infeasibility or optimality certificates.

    Scalar witnesses give min_box(A delta - upper). Share their maximum per
    epigraph, charge each key once, and omit cones (a safe zero lower bound).
    Reserve floating-point dot/sum error before claiming a positive bound.
    Unknown/unbounded domains fail open to zero and retain the usual search.
    """
    lower, upper = np.asarray(model.step_lower).ravel(), np.asarray(model.step_upper).ravel()
    soft = model.soft
    result = np.zeros(4)
    if (lower.shape != (soft.size,) or upper.shape != lower.shape
            or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or np.any(lower > upper)):
        return result
    values = np.zeros(len(soft.keys))
    eps, tiny = np.finfo(float).eps, np.nextafter(0., 1.)
    for matrix, bound, indices in soft.rows:
        positive, negative = matrix.maximum(0), matrix.minimum(0)
        with np.errstate(over='ignore', invalid='ignore'):
            raw = positive @ lower + negative @ upper - bound
            magnitude = abs(matrix) @ np.maximum(abs(lower), abs(upper)) + abs(bound)
            operations = 2 * np.diff(matrix.indptr) + 4
            gamma = operations * eps / (1. - operations * eps)
            reserve = gamma / (1. - gamma) * magnitude + operations * tiny
            safe = np.nextafter(raw - reserve, -np.inf)
        safe[~np.isfinite(safe) | (gamma >= 1.)] = -np.inf
        np.maximum.at(values, indices, safe)
    # Round products and each accurately summed positive level downwards.
    with np.errstate(over='ignore', invalid='ignore'):
        products = np.maximum(0., np.nextafter(values * np.asarray(soft.weights), -np.inf))
    if not np.isfinite(products).all():
        return result
    for level in range(4):
        try:
            value = math.fsum(products[np.asarray(soft.levels) == level])
        except OverflowError:
            return np.zeros(4)
        result[level] = max(0., np.nextafter(value, -np.inf))
    return result
