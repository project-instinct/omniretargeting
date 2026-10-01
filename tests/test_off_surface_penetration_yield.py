"""Penetration elsewhere on a body cannot make an absent contact recede."""
from dataclasses import replace
import numpy as np
import pytest
from omniretargeting._fast import runtime_options
from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
from omniretargeting._fast.contact_guard import ContactGuard
from tests.test_surface_presence import _setup, _moves


@pytest.mark.parametrize('guard_type', [AcquisitionNormalGuard, ContactGuard])
def test_separated_point_only_approaches_even_if_its_body_penetrates(monkeypatch, guard_type):
    objective, poses = _setup(-.1)
    body = objective.model.body_weldid[objective.effectors[0]['body']]
    depths = np.zeros((len(poses), objective.model.nbody))
    depths[:, body] = .02
    objective.frozen_original_depth = lambda q: depths
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS, off_surface_approach_only=False))
    approach, recede = _moves(poses)
    assert guard_type(objective, poses).accepts(recede)  # observed regression
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS, off_surface_approach_only=True))
    guarded = guard_type(objective, poses)
    assert guarded.accepts(approach)
    assert not guarded.accepts(recede)
    linear, norms = guarded.repair_bounds(poses)
    step = np.zeros((len(poses), objective.coords.width))
    step[:, objective.coords.inverse[objective.coords.root_d]] = .01
    flat = step.ravel()
    violations = [np.any(a @ flat < lo - 1e-7) or np.any(a @ flat > hi + 1e-7)
                  for a, lo, hi in linear]
    violations += [np.any(np.linalg.norm(r + (a @ flat).reshape(r.shape), axis=1) > b + 1e-7)
                   for a, r, b in norms]
    assert any(violations)
    # Reaching the surface still permits the frozen own-body exit allowance.
    objective, poses = _setup(0.)
    objective.frozen_original_depth = lambda q: depths
    lifted = poses.copy(); lifted[:, 2] += .01
    assert guard_type(objective, poses).accepts(lifted)
