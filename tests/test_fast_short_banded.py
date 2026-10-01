"""A feasible short direction needs measured descent, never a looser guard."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast import runtime_options, solver


class ShortGuard:
    rejections = 0

    def accepts(self, poses):
        good = bool(np.all(poses <= .04))
        self.rejections += not good
        return good

    def constraints(self, *args):
        return []


class Quadratic:
    frames = width = 1
    lower, upper = np.array([-10.]), np.array([10.])

    def __init__(self, bad=None):
        self.bad = bad
        self.coords = SimpleNamespace(qi=np.array([0]), scalar_cols=np.array([0]),
            root_q=None, root_d=None, inverse={},
            integrate=lambda q, d, alpha: q + alpha * d, difference=lambda a, b: b - a)
        self.source_shape = ShortGuard()
        self.candidate_guards = []
        self.evaluated = []

    def root_boundaries(self, poses):
        return []

    def project(self, poses):
        return poses

    def evaluate(self, poses, **kwargs):
        x = float(poses[0, 0])
        self.evaluated.append(x)
        cost = (x - 1.) ** 2
        if x > .03 and self.bad == 'increase':
            cost = 2.
        if x > .03 and self.bad == 'nan':
            cost = np.nan
        acc = SimpleNamespace(cost=cost, gradient=poses - 1., hessian=np.ones((1, 1, 1)))
        return acc, np.zeros(0), sparse.csr_matrix((0, 1)), {}, None


@pytest.mark.parametrize('enabled', [False, True])
def test_feasible_small_fraction_before_conic(monkeypatch, enabled):
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS, accept_short_banded=enabled))
    objective = Quadratic()
    calls = []

    def conic(*args):
        calls.append(True)
        return np.array([.02]), np.array([[.02]]), 1.

    monkeypatch.setattr(solver, '_conic_proposal', conic)
    result, history = solver._optimize(objective, np.zeros((1, 1)), 1, None, 'probe', .003)
    assert len(calls) == (0 if enabled else 1)
    assert history[0]['step_backend'] == ('banded' if enabled else 'shape_conic')
    assert history[0]['alpha'] == (.03125 if enabled else 1.)
    assert history[0]['after'] < history[0]['cost']
    assert objective.source_shape.accepts(result)
    assert objective.evaluated.count(float(result[0, 0])) == 1


@pytest.mark.parametrize('bad', ['increase', 'nan', 'nonpositive_prediction'])
def test_invalid_descent_keeps_conic_fallback(monkeypatch, bad):
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS, accept_short_banded=True))
    objective = Quadratic(bad)
    original = solver._model_decrease
    if bad == 'nonpositive_prediction':
        monkeypatch.setattr(solver, '_model_decrease',
            lambda acc, jt, delta, grad: -1. if delta[0] > .03 else original(acc, jt, delta, grad))
    calls = []

    def conic(*args):
        calls.append(True)
        return np.array([.02]), np.array([[.02]]), 1.

    monkeypatch.setattr(solver, '_conic_proposal', conic)
    result, history = solver._optimize(objective, np.zeros((1, 1)), 1, None, 'probe', .003)
    assert len(calls) == 1
    assert history[0]['step_backend'] == 'shape_conic'
    assert history[0]['after'] < history[0]['cost']
    assert result[0, 0] == .02
