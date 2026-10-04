"""Small frame-local dual certificates from protected collision witnesses."""

import numpy as np
from scipy.optimize import linprog


def maximum(gradient, low, high, witnesses, depths):
    """Upper bound using any nonnegative LP dual, with box-bounded residual.

    Drop every conic/contact/layout and cross-frame constraint. The retained
    inequalities are the existing collision no-regression rows on one frame;
    use current key depths, which only enlarge the protected-depth polytope.
    HiGHS supplies multipliers, never a trusted primal/optimality certificate.
    Clipping multipliers nonnegative and bounding the remaining gradient over
    the box yields a valid upper bound even for an inaccurate LP solution.
    """
    selected = np.flatnonzero((low != 0.) | (high != 0.))
    if not len(selected):
        return 0.
    jac = np.array([row['jacobian'] for _, row in witnesses])[:, selected]
    phi = np.array([row['clearance'] for _, row in witnesses])
    if not np.isfinite(jac).all() or not np.isfinite(phi).all():
        return None
    rhs = np.maximum([depths.get(key, 0.) for key, _ in witnesses], -phi) + phi
    if not np.isfinite(rhs).all():
        return None
    # Outward arithmetic allowance; the original rows have no added slack.
    rhs += 1e-10 * (1. + np.abs(phi))
    varying = np.any(jac != 0., axis=1)
    matrix, rhs = -jac[varying], rhs[varying]
    if not len(rhs):
        return None
    c = gradient[selected]
    try:
        solution = linprog(-c, A_ub=matrix, b_ub=rhs, bounds=list(zip(low[selected], high[selected])),
                           method='highs', options={'presolve': True, 'time_limit': .05})
    except (ValueError, RuntimeError):
        return None
    if not solution.success:
        return None
    dual = np.maximum(-solution.ineqlin.marginals, 0.)
    if not np.isfinite(dual).all():
        return None
    residual = c - dual @ matrix
    support = np.maximum(residual * low[selected], residual * high[selected]).sum()
    upper = float(dual @ rhs + support)
    # Pad dot products and cancellation by their absolute term magnitudes.
    magnitude = float(np.abs(dual) @ (np.abs(rhs) + np.abs(matrix) @
                      np.maximum(np.abs(low[selected]), np.abs(high[selected]))))
    return upper + 1e-10 * (1. + magnitude + abs(support))
