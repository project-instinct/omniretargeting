"""Banded minimum-change SQP of a linearly feasible acquisition step."""

import time
import numpy as np

from .barrier_solver import _domains, _equalities
from .constrained_ip import QuadraticDomain, solve
from .constrained_step import step_boxes
from .proposals import guards_of
from .solver import _hold

MAX_ROUNDS = 4


def restore(objective, rows, origin, direction, alpha, diagnostics):
    """Keep the original step box and every exact guard during restoration.

    The objective is the legacy trust-coordinate minimum change. Continuity
    couples the correction through complete stencils, including fixed window
    neighbours. Physical descent is checked by the caller, after restoration.
    """
    tick = time.monotonic()
    coords, width = objective.coords, objective.width
    lower, upper = step_boxes(objective, origin)
    scales = np.maximum(np.maximum(np.abs(lower), np.abs(upper)), 1e-12)
    metric = np.zeros((min(4 * width, len(lower)), len(lower)), order='F')
    metric[0] = 1 / scales**2
    held = np.unique(np.r_[_equalities(objective, origin), np.flatnonzero(lower == upper)])
    _hold(metric, held)
    candidate = objective.project(coords.integrate(origin, direction.reshape(len(origin), -1), alpha))
    warm = {}
    guards = guards_of(objective)[1]
    diagnostics['banded_restorations'] = diagnostics.get('banded_restorations', 0) + 1
    for index in range(MAX_ROUNDS + 1):
        values = rows.evaluate(candidate, False)
        if (_domains(objective, candidate) and all(np.isfinite(f.slack).all() and
                np.all(f.slack >= 0) for f in values) and all(g.accepts(candidate) for g in guards)):
            actual = coords.difference(origin, candidate).ravel()
            if np.all(actual >= lower - 1e-12) and np.all(actual <= upper + 1e-12):
                diagnostics['restoration_seconds'] = diagnostics.get('restoration_seconds', 0.) + time.monotonic() - tick
                return candidate
            return None
        if index == MAX_ROUNDS:
            break
        displacement = coords.difference(origin, candidate).ravel()
        families = rows.evaluate(candidate, True)
        domain = QuadraticDomain(families, lower - displacement, upper - displacement, held)
        correction = solve(metric, np.zeros(len(lower)), domain, warm, diagnostics, restoration=True)
        diagnostics['restoration_rounds'] = diagnostics.get('restoration_rounds', 0) + 1
        if diagnostics.get('ip_excess', 0.) > 1e-8:
            # An incomplete convex correction is not an infeasibility
            # certificate. Try a shorter starting fraction instead of
            # spending more SQP rounds on the stalled numerical solve.
            diagnostics['restoration_stalls'] = diagnostics.get('restoration_stalls', 0) + 1
            diagnostics['restoration_seconds'] = diagnostics.get('restoration_seconds', 0.) + time.monotonic() - tick
            return None
        candidate = objective.project(coords.integrate(candidate, correction.reshape(len(origin), -1), 1.))
    diagnostics['restoration_seconds'] = diagnostics.get('restoration_seconds', 0.) + time.monotonic() - tick
    return None
