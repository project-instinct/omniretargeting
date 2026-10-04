"""Current physical GN objective for the private coupled guard projection."""

import time

import numpy as np
from scipy import sparse

from .barrier_assembly import physical_system

PROXIMITY = .1


def curvature(accumulator, temporal):
    """Convert the directly assembled lower band once; never form sparse J'J."""
    band = physical_system(accumulator, temporal)
    size = band.shape[1]
    lower = sparse.diags([band[k, :size-k] for k in range(len(band))],
        -np.arange(len(band)), shape=(size, size), format='csr')
    return lower + lower.T - sparse.diags(band[0], format='csr')


def majorizer(hessian):
    """Diagonal upper bound of GN curvature in equilibrated coordinates."""
    diagonal = np.maximum(hessian.diagonal(), 1.)
    scale = sparse.diags(1/np.sqrt(diagonal), format='csr')
    normalized = scale@hessian@scale
    return diagonal*np.asarray(abs(normalized).sum(axis=1)).ravel()


def model(window, poses, target, columns, report):
    """C/2 + proximity to bulk, with all original rows and crossing stencils.

    The diagonal proximity also conditions unconstrained physical null modes.
    It changes only this numerical objective, never any physical guard budget.
    """
    started = time.monotonic()
    acc, residual, temporal, stats, _ = window.evaluate(poses, True)
    hessian = curvature(acc, temporal)
    gradient = acc.gradient.ravel() + temporal.T @ residual
    diagonal = np.maximum(hessian.diagonal(), 1.)
    if report.get('physical_majorizer'):
        # An upper bound of the same full GN model keeps the true gradient,
        # but avoids dense coordinate coupling in the constrained factor.
        hessian = sparse.diags(majorizer(hessian), format='csr')
    if report.get('physical_first_order'):
        # Exact first-order C/2 model plus a diagonal physical-unit trust
        # metric. This avoids coordinate coupling in the constrained factor.
        hessian = sparse.diags(diagonal, format='csr')
    desired = window.coords.difference(poses, target).ravel()
    hessian += sparse.diags(PROXIMITY * diagonal, format='csr')
    gradient -= PROXIMITY * diagonal * desired
    report['physical_model_seconds'] = report.get('physical_model_seconds', 0.) + time.monotonic()-started
    report['physical_model_evaluations'] = report.get('physical_model_evaluations', 0) + 1
    return hessian[columns][:, columns], gradient[columns], acc.cost, stats


def value(window, poses, report):
    started = time.monotonic()
    acc, _, _, stats, _ = window.evaluate(poses, False)
    report['physical_value_seconds'] = report.get('physical_value_seconds', 0.) + time.monotonic()-started
    report['physical_value_evaluations'] = report.get('physical_value_evaluations', 0) + 1
    return acc.cost, stats
