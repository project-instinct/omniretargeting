"""Optimize frame fractions under the complete trajectory model and hard rows."""

import numpy as np
from scipy import sparse

from .qp_reduction import solve_restricted
from .second_order_step import failing_frames

MAX_ROUNDS = 5


def frame_candidates(objective, model, solution, damping, guards, uniform, *, priority=False):
    from .priority_merit import check_trial
    frames, width = model.step_lower.shape
    # This substitution retains derivative-of-alpha terms automatically.
    transform = sparse.csc_matrix((solution.step, (np.arange(frames * width),
                                    np.repeat(np.arange(frames), width))), shape=(frames * width, frames))
    caps = np.ones(frames)
    for iteration in range(MAX_ROUNDS):
        result = solve_restricted(model, damping, transform, np.zeros(frames * width),
                                  priority=priority, lower=np.zeros(frames), upper=caps)
        if result is None:
            return
        proposed, fractions = result
        candidate = objective.project(model.coords.integrate(
            model.origin, proposed.step.reshape(frames, width), 1.))
        if check_trial(objective, candidate, guards):
            proposed.certificate['frame_fractions'] = dict(min=float(fractions.min()),
                                                          mean=float(fractions.mean()), rounds=iteration + 1)
            yield proposed, candidate, float(np.min(fractions))
            return
        bad = failing_frames(objective, candidate, guards)
        if not bad.any():
            bad[:] = True  # a priority veto supplies no hard-guard frame mask
        reduced = caps.copy()
        reduced[bad] = np.maximum(uniform, .5 * caps[bad])
        if np.array_equal(reduced, caps):
            # Keep the known uniform vector in the cap box; never grind on a
            # nonlinear-invalid combination at that cap. Caller keeps fallback.
            return
        caps = reduced
