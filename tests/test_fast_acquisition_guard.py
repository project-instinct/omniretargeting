"""Object/tangent improvement must not detach protected contacts."""

from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pytest

from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
from tests.test_fast_contact_guard import _acquired


@pytest.mark.parametrize('surfaces', [False, True])
def test_normal_regression_is_independent_of_tangent_improvement(surfaces):
    objective, poses = _acquired(surfaces=surfaces)
    objective.tasks.contact_phase = None
    if surfaces:
        for eff in objective.effectors:
            eff['active'][:] = False
        for surf in objective.surfaces:
            surf['sticking'][:] = True
            surf['normals'][:] = [0., 0., 1.]
            surf['targets'] -= [.1, 0., 0.]
    else:
        objective.effectors[0]['normal'][:] = [0., 0., 1.]
        objective.effectors[0]['contact'] -= [.1, 0., 0.]
    guard = AcquisitionNormalGuard(objective, poses)
    assert guard.accepts(poses)
    candidate = poses.copy()
    candidate[:, 0] -= .01
    candidate[:, 2] += .005
    assert not guard.accepts(candidate)
    flat = cp.Variable(objective.frames * objective.width)
    constraints = guard.constraints(poses, flat)
    flat.value = objective.coords.difference(poses, candidate).ravel()
    assert max(np.max(c.violation()) for c in constraints) > .003
    # Tangential progress without detachment remains available.
    candidate[:, 2] = poses[:, 2]
    assert guard.accepts(candidate)
    for eff in objective.effectors:
        eff['contact'] += 10
    assert guard.accepts(poses)  # immutable targets and normal budgets


@pytest.mark.parametrize('role, protected', [('support', True), ('hold', True),
                                           ('control', False), ('follow', False)])
def test_mechanical_role_applies_to_any_region(role, protected):
    objective, poses = _acquired()
    objective.effectors[0]['name'] = 'arbitrary_region'
    objective.effectors[0]['normal'][:] = [0., 0., 1.]
    objective.tasks.contact_phase = SimpleNamespace(role=lambda name, frame: role)
    guard = AcquisitionNormalGuard(objective, poses)
    candidate = poses.copy()
    candidate[:, 2] += .02
    assert guard.accepts(candidate) is not protected
