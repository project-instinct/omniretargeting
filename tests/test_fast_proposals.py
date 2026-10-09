"""Speculative damping attempts reproduce the sequential Levenberg-Marquardt loop."""

import numpy as np
import pytest

import omniretargeting._fast.proposals as proposals
from omniretargeting._fast.continuity import CorrectionContinuity
from omniretargeting._fast.reference import ReferenceShape
from omniretargeting._fast.solver import optimize
from tests.test_fast_backend import _objective


def _solve(monkeypatch, speculate):
    monkeypatch.setattr(proposals, "SPECULATE", speculate)
    objective, seed, _ = _objective(temporal=True)
    shape = ReferenceShape(objective, seed)
    shape.freeze_reference(seed)
    objective.source_shape = shape
    continuity = CorrectionContinuity(objective.coords, seed)
    objective.candidate_guards = [continuity]
    # An incompatible target: many proposals leave the envelopes and are rejected.
    for effector in objective.effectors:
        effector["contact"] = effector["contact"] + [0.8, 0.0, 0.3]
    original_banded = proposals.Attempts.banded
    original_conic = proposals.Attempts.conic
    # A failed linear solve on the first attempt forces the retry path under
    # both schedules. Later attempts use the actual solver and guard checks.
    monkeypatch.setattr(proposals.Attempts, "banded",
                        lambda self, k: None if k == 0 else original_banded(self, k))
    monkeypatch.setattr(proposals.Attempts, "conic", original_conic)
    speculated = []
    original = proposals.Attempts.speculate
    monkeypatch.setattr(proposals.Attempts, "speculate",
                        lambda self, *a: speculated.append(a) or original(self, *a))
    solved, history = optimize(objective, seed.copy(), iterations=40, converged_decrease=0.0)
    for row in history:
        row.pop("seconds")
    return solved, history, (shape.rejections, continuity.rejections), speculated


@pytest.mark.parametrize("diverse", [False])
def test_speculative_attempts_equal_the_sequential_loop(monkeypatch, diverse):
    with monkeypatch.context() as patch:
        sequential = _solve(patch, False)
    with monkeypatch.context() as patch:
        speculative = _solve(patch, True)
    assert speculative[3], "no iteration rejected its first attempt"
    np.testing.assert_array_equal(speculative[0], sequential[0])
    assert speculative[1] == sequential[1]
    assert speculative[2] == sequential[2]
    assert any(row.get("step_backend") == "shape_conic" for row in sequential[1])
