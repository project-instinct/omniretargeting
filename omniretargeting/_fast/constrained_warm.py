"""Dual projections to identify a feasible inequality working set cheaply.

The screened row set shares one banded factor. A partial QP direction can
propose a physical step; only original exact guards and cost certify it.
Rejected proposals can initialize the equality-constrained polishing step.
"""

import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded

from .barrier_assembly import scatter

MAX_PROJECTIONS = 400
RHO = 10.


def working_set(system, gradient, matrix, bounds, warm, diagnostics):
    rho = RHO
    ab = system.copy(order='F')
    scatter(ab, matrix, rho)
    scale = 1 / np.sqrt(ab[0])
    for offset in range(len(ab)):
        length = ab.shape[1] - offset
        ab[offset, :length] *= scale[offset:] * scale[:length]
    diagnostics['factorizations'] += 1
    factor = cholesky_banded(ab, lower=True, overwrite_ab=True, check_finite=False)
    matrix_t = matrix.T.tocsr()
    dual = np.maximum(warm.get('multipliers', np.zeros(len(bounds))), 0.).copy()
    starting=warm.get('direction', np.zeros(len(gradient)))
    if starting is None:
        # Start inactive faces at a useful physical proposal. Initializing
        # every one-sided slack at zero introduces artificial springs over
        # the entire row set, especially with a larger numerical penalty.
        physical=system.copy(order='F');physical_scale=1/np.sqrt(physical[0])
        for offset in range(len(physical)):
            length=physical.shape[1]-offset
            physical[offset,:length]*=physical_scale[offset:]*physical_scale[:length]
        diagnostics['factorizations']+=1
        physical_factor=cholesky_banded(physical,lower=True,check_finite=False)
        starting=physical_scale*cho_solve_banded((physical_factor,True),-gradient*physical_scale,check_finite=False)
        diagnostics['band_solves']+=1
        if 'lower' in warm:starting=np.clip(starting,warm['lower'],warm['upper'])
    slack = np.minimum(matrix@starting, bounds)
    direction = np.zeros(len(gradient))
    for number in range(MAX_PROJECTIONS):
        rhs = -gradient + matrix_t @ (rho * slack - dual)
        direction = scale * cho_solve_banded((factor, True), rhs * scale, check_finite=False)
        diagnostics['band_solves'] += 1
        value = matrix @ direction
        over = 1.6 * value - .6 * slack
        previous = slack
        slack = np.minimum(over + dual / rho, bounds)
        dual += rho * (over - slack)
        if number % 25 == 24:
            residual = float(np.max(np.abs(value - slack), initial=0.))
            if residual < 1e-7:
                break
    diagnostics['projection_iterations'] = diagnostics.get('projection_iterations', 0) + number + 1
    diagnostics['projection_excess'] = float(np.max(value - bounds, initial=0.))
    diagnostics['projection_dual_residual'] = float(np.max(np.abs(matrix_t @ (rho * (slack-previous))), initial=0.))
    return direction, dual
