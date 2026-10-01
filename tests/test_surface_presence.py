"""A contact is protected on the actual surface, not on a far tangent plane."""

import numpy as np
import pytest

from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
from omniretargeting._fast.contact_guard import ContactGuard
from tests.test_fast_contact_guard import _acquired


def _ground(objective, height):
    """A horizontal surface ``height`` metres below the acquired foot point."""
    level = objective.effectors[0]["contact"][0, 2] - height
    objective._clearance_fields = lambda: [("scene_depth_m", None, None, None, None)]
    objective._inside = lambda field, frames, points: points[..., 2] - level
    # No geom penetrates this ground: nothing may yield for clearance.
    objective._deepest_samples = lambda *args: (
        np.full((objective.frames, len(objective.shell_geoms)), np.inf),)


def _setup(height):
    objective, poses = _acquired()
    objective.tasks.contact_phase = None
    effector = objective.effectors[0]
    effector["normal"][:] = [0., 0., 1.]
    effector["contact"] -= [.1, 0., 0.]  # 10 cm tangentially from its target
    _ground(objective, height)
    return objective, poses


def _moves(poses):
    approach_and_lift = poses.copy()
    approach_and_lift[:, 0] -= .01
    approach_and_lift[:, 2] += .005
    recede = poses.copy()
    recede[:, 0] += .01
    return approach_and_lift, recede


@pytest.mark.parametrize("guard_type", [AcquisitionNormalGuard, ContactGuard])
def test_point_on_the_surface_may_not_lift_while_approaching(guard_type):
    objective, poses = _setup(0.)
    guard = guard_type(objective, poses)
    approach_and_lift, recede = _moves(poses)
    assert guard.accepts(poses)
    assert not guard.accepts(approach_and_lift)  # sliding foot stays on the floor


@pytest.mark.parametrize("guard_type", [AcquisitionNormalGuard, ContactGuard])
def test_point_off_every_surface_may_approach_but_not_recede(guard_type):
    objective, poses = _setup(-.1)  # buried 10 cm, not on the surface
    guard = guard_type(objective, poses)
    approach_and_lift, recede = _moves(poses)
    assert guard.accepts(approach_and_lift)
    assert not guard.accepts(recede)


def test_unknown_geometry_keeps_the_existing_protection():
    objective, poses = _setup(0.)
    objective._inside = lambda field, frames, points: np.full(points.shape[:-1], np.nan)
    approach_and_lift, _ = _moves(poses)
    assert not AcquisitionNormalGuard(objective, poses).accepts(approach_and_lift)



@pytest.mark.parametrize("guard_type", [AcquisitionNormalGuard, ContactGuard])
def test_contact_may_rise_by_its_own_body_penetration_depth(guard_type):
    objective, poses = _setup(0.)
    body = objective.model.body_weldid[objective.effectors[0]["body"]]
    geoms = np.asarray(objective.shell_geoms)
    sunk = np.full((objective.frames, len(geoms)), np.inf)
    sunk[:, objective.model.body_weldid[objective.model.geom_bodyid[geoms]] == body] = -.01
    objective._deepest_samples = lambda *args: (sunk,)
    guard = guard_type(objective, poses)
    lifted = poses.copy()
    lifted[:, 2] += .008  # within the 1 cm penetration of the foot geometry
    assert guard.accepts(lifted)
    lifted[:, 2] += .01   # beyond it: separation is protected again
    assert not guard.accepts(lifted)


def test_guard_hinge_rows_vanish_inside_and_push_back_outside():
    objective, poses = _setup(0.)
    guard = AcquisitionNormalGuard(objective, poses)
    frames = slice(0, len(poses))
    weight = np.ones(len(poses))
    objective.kin.evaluate(poses)
    rows, _ = guard.hinge_rows(objective.kin, frames, weight, 1e-3, True)
    assert np.allclose(rows, 0.)
    lifted = poses.copy()
    lifted[:, 2] += .01
    objective.kin.evaluate(lifted)
    rows, jac = guard.hinge_rows(objective.kin, frames, weight, 1e-3, True)
    assert rows.max() > 0
    step = np.zeros((len(poses), objective.coords.width))
    step[:, objective.coords.inverse[objective.coords.root_d + 2]] = 1e-6
    objective.kin.evaluate(objective.coords.integrate(lifted, step, 1.0), with_jacobian=False)
    after, _ = guard.hinge_rows(objective.kin, frames, weight, 1e-3, False)
    np.testing.assert_allclose(after - rows, np.einsum("fsw,fw->fs", jac, step), atol=1e-6)
