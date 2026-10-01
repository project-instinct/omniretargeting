"""An overshooting constrained step is shortened, not dropped with the pass."""

import numpy as np

import omniretargeting._fast.shape_step as shape_step
import omniretargeting._fast.solver as solver
from tests.test_fast_backend import _objective


class StepBound:
    """Nonlinear guard: no coordinate may change by more than ``limit``."""

    def __init__(self, origin, limit):
        self.origin, self.limit, self.rejections = origin, limit, 0

    def accepts(self, poses):
        ok = bool(np.abs(poses - self.origin).max() <= self.limit)
        self.rejections += not ok
        return ok


def _linearization(objective, poses):
    acc, temporal_r, temporal_j, _, _ = objective.evaluate(poses)
    gradient = acc.gradient.ravel() + temporal_j.T @ temporal_r
    return acc, temporal_j, gradient


def _overshooting_step(objective, gradient):
    """A descent direction far too long for the model: the full step raises the cost."""
    return -gradient / np.abs(gradient).max() * 0.2


def test_short_fraction_of_an_overshooting_step_is_taken():
    objective, poses, _ = _objective()
    acc, temporal_j, gradient = _linearization(objective, poses)
    step = _overshooting_step(objective, gradient)
    full = objective.project(objective.coords.integrate(poses, step.reshape(len(poses), -1), 1.))
    assert objective.evaluate(full, with_jacobian=False)[0].cost > acc.cost
    objective.candidate_guards = [StepBound(poses, 0.5)]
    shortened = solver._shorten_conic(objective, poses, step, 1., acc, temporal_j, gradient, acc.cost)
    assert shortened is not None
    candidate, scaled, trial, ratio = shortened
    assert scaled < 1 and ratio > 0
    assert trial[0].cost < acc.cost
    assert objective.candidate_guards[0].accepts(candidate)


def test_guard_rejected_fractions_are_restored_or_the_attempt_fails(monkeypatch):
    objective, poses, _ = _objective()
    acc, temporal_j, gradient = _linearization(objective, poses)
    step = _overshooting_step(objective, gradient)
    # No fraction satisfies the guard; restoration is the only way to a step.
    objective.candidate_guards = [StepBound(poses, 0.0)]
    calls = []

    def restore(objective_, origin, scaled_step):
        calls.append(scaled_step)
        return None

    monkeypatch.setattr(shape_step, "restore_shape_step", restore)
    assert solver._shorten_conic(objective, poses, step, 1., acc, temporal_j, gradient, acc.cost) is None
    assert 0 < len(calls) <= solver.MAX_CONIC_RESTORATIONS
