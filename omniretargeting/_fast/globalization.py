"""Choose nonlinear merit progress while retaining uniform backtracking fallback."""

import numpy as np

from .priority_merit import compare_levels, evaluate_priority
from .proposals import backtrack


def globalize(objective, model, solution, damping, guards, *, soc=False, frame_lengths=False):
    candidate, alpha = backtrack(objective, model.origin, solution.step, guards)
    selected = (solution, candidate, alpha)
    if not soc and not frame_lengths:
        # The caller owns the objective/merit acceptance test. No alternatives
        # need comparison, so do not evaluate the whole trajectory twice.
        return selected
    priority = getattr(objective, 'priority_clearance', None) is not None
    best_cost = np.inf if candidate is None else objective.evaluate(candidate, with_jacobian=False)[0].cost
    best_merit = None if candidate is None else evaluate_priority(objective, candidate)
    sources = []
    if soc:
        from .second_order_step import corrected_candidates
        sources.append(corrected_candidates(objective, model, solution, damping, guards, priority=priority))
    if frame_lengths:
        from .frame_step import frame_candidates
        sources.append(frame_candidates(objective, model, solution, damping, guards, alpha, priority=priority))
    for source in sources:
        for proposal, trial, fraction in source:
            # Projection can change a tangent step: assess the executed point.
            actual = model.coords.difference(model.origin, trial).ravel()
            if (np.any(actual < model.step_lower.ravel() - 1e-9)
                    or np.any(actual > model.step_upper.ravel() + 1e-9)):
                continue
            merit = evaluate_priority(objective, trial)
            cost = objective.evaluate(trial, with_jacobian=False)[0].cost
            better = np.isfinite(cost) and cost < best_cost
            if priority:
                order = 1 if best_merit is None else compare_levels(best_merit, merit)[0]
                better = merit.valid and (order > 0 or (order == 0 and better))
            if better:
                best_cost, best_merit = cost, merit
                # The correction/fraction QP supplies a new direction. Return
                # it at unit fraction; the effective fraction is in metadata.
                selected = proposal, trial, 1.
    return selected
