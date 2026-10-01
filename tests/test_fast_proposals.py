"""Speculative damping attempts reproduce the sequential Levenberg-Marquardt loop."""

import numpy as np
import pytest
from dataclasses import replace
from omniretargeting._fast import runtime_options

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
    speculated = []
    original = proposals.Attempts.speculate
    monkeypatch.setattr(proposals.Attempts, "speculate",
                        lambda self, *a: speculated.append(a) or original(self, *a))
    solved, history = optimize(objective, seed.copy(), iterations=40, converged_decrease=0.0)
    for row in history:
        row.pop("seconds")
    return solved, history, (shape.rejections, continuity.rejections), speculated


@pytest.mark.parametrize("diverse", [False, True])
def test_speculative_attempts_equal_the_sequential_loop(monkeypatch, diverse):
    monkeypatch.setattr(runtime_options, "OPTIONS", replace(runtime_options.OPTIONS, conic_trust_damping=diverse))
    sequential = _solve(monkeypatch, False)
    speculative = _solve(monkeypatch, True)
    assert speculative[3], "no iteration rejected its first attempt"
    np.testing.assert_array_equal(speculative[0], sequential[0])
    assert speculative[1] == sequential[1]
    assert speculative[2] == sequential[2]
    assert any(row.get("step_backend") == "shape_conic" for row in sequential[1])


def test_three_conic_retries_cover_distinct_damping_without_extra_solves(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(runtime_options, "OPTIONS", replace(runtime_options.OPTIONS, conic_trust_damping=True))
    objective = SimpleNamespace(candidate_guards=[])
    dampings = [1e-8,2e-8,8e-8,64e-8]
    def conic(objective, poses, system, gradient, diagonal, damping, guards):
        return (np.array([damping]), None, 0.)
    def make():
        value = proposals.Attempts(objective, np.zeros((1,1)), None, None, None, None, None, None,
                                   dampings, None, conic)
        value._banded = lambda attempt: ((np.zeros(1), None, 0.), [])
        return value
    serial = make()
    expected = [serial.conic(i,i+1)[0][0] for i in range(3)]
    assert expected == [1e-8,1e-3,1.]
    speculative = make()
    actual = [speculative.conic(0,1)[0][0]]
    monkeypatch.setattr(proposals,"SPECULATE",True)
    speculative.speculate(1,1,3,True)
    assert ('conic',3) not in speculative.cache
    actual += [speculative.conic(i,i+1)[0][0] for i in range(1,3)]
    assert actual == expected
    assert [speculative.conic_damping(i) for i in range(3)] == actual
