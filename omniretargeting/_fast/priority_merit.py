"""Pure trial evaluation and lexicographic acceptance; ledger writes stay in solver."""

import numpy as np

from .qp_types import PriorityMerit


def evaluate_priority(objective, poses):
    coordinator = getattr(objective, 'priority_clearance', None)
    if coordinator is None:
        return None
    # An evaluation never commits, including when this runs in a fork.
    return PriorityMerit.coerce(coordinator.evaluate_trial(poses))


def check_trial(objective, poses, guards):
    record = evaluate_priority(objective, poses)
    hard = all(guard.accepts(poses) for guard in guards)
    return hard and (record is None or record.valid)


def compare_levels(before, after):
    if not after.valid:
        return -1, None
    for level, (old, new, tolerance) in enumerate(zip(
            before.levels, after.levels, np.maximum(before.tolerances, after.tolerances))):
        if old - new > tolerance:
            return 1, level
        if new - old > tolerance:
            return -1, level
    return 0, None


def assess(model, step, before, after, cost, trial_cost, certificate):
    """Accepted, damping ratio, and explicit actual/predicted L1 merit decreases."""
    coefficients = np.asarray(certificate['coefficients'])
    predicted_slacks = model.predicted_slacks(step)
    origin_slacks = model.predicted_slacks(np.zeros(model.gradient.size))
    predicted = model.tracking_decrease(step) + model.soft.penalty(origin_slacks, coefficients)
    predicted -= model.soft.penalty(predicted_slacks, coefficients)
    actual = cost - trial_cost + model.soft.penalty(before.slacks, coefficients)
    actual -= model.soft.penalty(after.slacks, coefficients)
    order, level = compare_levels(before, after)
    ratio = actual / predicted if predicted > 0 else -1.
    accepted = bool(order == 0 and np.isfinite(trial_cost) and actual > 0 and ratio > 0)
    if order > 0 and np.isfinite(trial_cost):
        # A tracking increase cannot veto verified higher-level progress.
        predicted_level = model.soft.level_values(origin_slacks)[level]
        predicted_level -= model.soft.level_values(predicted_slacks)[level]
        actual_level = before.levels[level] - after.levels[level]
        ratio = actual_level / predicted_level if predicted_level > 0 else 1.
        accepted = True
    return accepted, float(ratio), dict(actual_merit_decrease=float(actual),
                                      predicted_merit_decrease=float(predicted),
                                      improved_level=level if order > 0 else None)


def report_value(value):
    """JSON history form; proposal objects retain tuple-keyed dictionaries."""
    if isinstance(value, dict):
        return {str(k) if isinstance(k, tuple) else k: report_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [report_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
