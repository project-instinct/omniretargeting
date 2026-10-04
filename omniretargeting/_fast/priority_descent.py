"""Independent safe-descent certificate; never a strict QP optimality claim."""

import hashlib

import numpy as np

KIND = 'zero_additional_contact_yield_descent'
VERSION = 1


def pose_digest(poses):
    value = np.asarray(poses, dtype=np.float64)
    digest = hashlib.sha256(repr(value.shape).encode())
    digest.update(value.tobytes(order='C'))
    return digest.hexdigest()


def certify(coordinator, poses, predicted_slacks=None):
    from .priority_clearance import CONTACT_SLACK, DEPTH_PROGRESS
    from .runtime_options import OPTIONS
    if not getattr(OPTIONS, 'priority_zero_yield_descent', False):
        return None
    report = coordinator._evaluate_trial(poses)
    if not report['accepted'] or not report['reliable'] or report['yields']:
        return None
    if (not report['strict_penetration_decrease']
            or report['actual_reductions'][0] <= DEPTH_PROGRESS
            or report['predicted_reductions'][0] <= DEPTH_PROGRESS):
        return None
    objective = coordinator.objective
    step = objective.coords.difference(coordinator.current, poses).ravel()
    model = coordinator.model
    if not np.array_equal(getattr(model, 'origin', None), coordinator.current):
        return None
    # Primal feasibility is checked at the EXECUTED step, including all hard
    # rows/cones and per-channel no-yield bounds. No optimality is required.
    from .descent_step import original_primal_certificate
    primal = original_primal_certificate(model, step)
    if not primal.get('primal_feasible', False):
        return None
    own_slacks = model.predicted_slacks(step)
    if predicted_slacks is not None and (predicted_slacks.keys() != own_slacks.keys()
                                        or any(predicted_slacks[k] != v for k, v in own_slacks.items())):
        return None
    from .proposals import guards_of
    for guard in guards_of(objective)[1]:
        if guard is not coordinator and not guard.accepts(poses):
            return None
    # Do not rely on a shared role/body epigraph to protect individual points
    # or channels. A clean point and an own-body penetrating point are equal.
    candidate_blocks = coordinator._trial[4]
    identities = [key for block in candidate_blocks for key in block.identities]
    if (len(identities) != len(set(identities))
            or set(identities) != coordinator._current_contacts.keys()):
        return None
    maximum, count = 0., 0
    for block in candidate_blocks:
        for identity, value in zip(block.identities, block.values):
            previous = coordinator._current_contacts.get(identity)
            if previous is None or not np.isfinite(value):
                return None
            regression = float(value - previous)
            if regression > CONTACT_SLACK:
                return None
            maximum = max(maximum, regression)
            count += 1
    # Actual non-regression is checked above. Check the original linearized
    # collision keys/bodies as well, not just a sum of predicted depths.
    from .mesh_repair import _regressed, REGRESSION, PENETRATING_REGRESSION_M
    values = np.maximum(-coordinator.phi - coordinator.matrix @ step, 0.)
    for (key, _), value in zip(coordinator.current_measurement['rows'], values):
        if _regressed(value, coordinator.best_keys.get(key, 0.)):
            return None
    predicted, _ = coordinator._prediction(step)
    if np.any(predicted > coordinator.best_depth + REGRESSION
              + (coordinator.best_depth > 0) * PENETRATING_REGRESSION_M):
        return None
    return dict(kind=KIND, version=VERSION, epoch=coordinator.epoch,
                pose_sha256=pose_digest(poses), descent_certified=True,
                optimality_certified=False, convergence_certified=False,
                hard_infeasibility_certified=False, zero_additional_yield=True,
                yield_ledger_unchanged=True, contact_metrics_checked=count,
                max_contact_regression_m=maximum, contact_tolerance_m=CONTACT_SLACK,
                actual_penetration_reduction_m=float(report['actual_reductions'][0]),
                predicted_penetration_reduction_m=float(report['predicted_reductions'][0]),
                original_primal_certificate=primal)
