"""Default-off Newton corrections of proposal-side contact curvature defects.

Only contact guards may fail at entry. The metric is the existing damped
trajectory model, so correction spreads through its temporal coupling. Every
other guard, the original step box and projection still decide feasibility.
This changes the numerical proposal path and requires per-item validation.
"""

import numpy as np
from scipy import sparse
from scipy.linalg import cholesky_banded, cho_solve_banded, LinAlgError

CONTACT_CURVATURE = False
# Effort limits; these do not bound posture or relax any acceptance test.
MAX_ROWS = 96
NEWTON_ITERATIONS = 3


def _contact(guard):
    from .acquisition_guard import AcquisitionNormalGuard
    from .contact_guard import ContactGuard
    return isinstance(getattr(guard, 'guard', guard), (AcquisitionNormalGuard, ContactGuard))


def _violated_rows(guards, poses):
    rows, targets = [], []
    for guard in guards:
        linear, cones = guard.qp_bounds(poses)
        for matrix, lower, upper in linear:
            lo, hi = np.asarray(lower), np.asarray(upper)
            for selected, target in ((lo > 0., lo), (hi < 0., hi)):
                if selected.any():
                    rows.append(matrix[selected])
                    targets.append(target[selected])
        for matrix, residual, allowed in cones:
            length = np.linalg.norm(residual, axis=1)
            selected = length > allowed
            if not selected.any():
                continue
            dimensions = residual.shape[1]
            chosen = matrix[np.repeat(selected, dimensions)]
            direction = residual[selected] / length[selected, None]
            count = len(direction)
            reduce = sparse.csr_matrix((direction.ravel(),
                (np.arange(count).repeat(dimensions), np.arange(count * dimensions))),
                shape=(count, count * dimensions))
            rows.append(reduce @ chosen)
            targets.append(allowed[selected] - length[selected])
    if not rows:
        return None
    matrix, target = sparse.vstack(rows, format='csr'), np.concatenate(targets)
    if len(target) > MAX_ROWS or not np.isfinite(target).all() or not np.isfinite(matrix.data).all():
        return None
    # A zero row cannot correct its violation. Keep this a failed proposal.
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    if np.any(norms == 0.):
        return None
    return sparse.diags(1 / norms) @ matrix, target / norms


def restore_curvature(objective, origin, step, guards, metric):
    """Return a fully feasible correction or None for established restoration."""
    if metric is None or not all(callable(getattr(g, 'qp_bounds', None)) for g in guards):
        return None
    contact = [guard for guard in guards if _contact(guard)]
    if not contact:
        return None
    coords = objective.coords
    candidate = objective.project(coords.integrate(origin, np.asarray(step).reshape(
        objective.frames, objective.width), 1.))
    other = [guard for guard in guards if not _contact(guard)]
    if not all(guard.accepts(candidate) for guard in other):
        return None
    lower, upper = coords.bounds(origin, objective.cfg['angular_step_rad'],
                                 objective.cfg['translation_step_m'])
    system, diagonal, damping = metric
    ab = system.copy(order='F')
    ab[0] += damping * diagonal + 1e-9
    fixed = getattr(objective, 'fixed_root', None)
    if fixed is not None:
        from .solver import _hold
        columns = [coords.inverse[d] for d in range(coords.root_d, coords.root_d + 6)
                   if d in coords.inverse]
        if columns:
            _hold(ab, (np.arange(objective.frames)[:, None] * objective.width + columns).ravel())
    try:
        factor = cholesky_banded(ab, lower=True, check_finite=False)
        for _ in range(NEWTON_ITERATIONS):
            prepared = _violated_rows(contact, candidate)
            if prepared is None:
                return None
            rows, target = prepared
            dense = rows.toarray()
            if fixed is not None and columns:
                dense[:, (np.arange(objective.frames)[:, None] * objective.width + columns).ravel()] = 0.
            inverse_rows = cho_solve_banded((factor, True), dense.T, check_finite=False)
            multipliers = np.linalg.lstsq(dense @ inverse_rows, target * 1.001, rcond=None)[0]
            delta = inverse_rows @ multipliers
            if not np.isfinite(delta).all():
                return None
            candidate = objective.project(coords.integrate(candidate, delta.reshape(
                objective.frames, objective.width), 1.))
            actual = coords.difference(origin, candidate)
            if np.any(actual < lower) or np.any(actual > upper):
                return None
            if all(guard.accepts(candidate) for guard in guards):
                return candidate
            # Newton is only for contact curvature; do not repair another
            # priority's violation with an unvalidated local proposal.
            if not all(guard.accepts(candidate) for guard in other):
                return None
    except (LinAlgError, ValueError):
        return None
    return None
