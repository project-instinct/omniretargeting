"""Clearance cannot buy a lower cost by spending acquired contact accuracy."""

import numpy as np
import pytest

from omniretargeting._fast.contact_guard import ContactGuard
from omniretargeting._fast.pipeline import smooth_correction, verify_budgets
from omniretargeting._fast.reference import ReferenceShape
from tests.test_fast_backend import _objective


def _acquired(*, surfaces=False, moving=False):
    objective, original, _ = _objective(extra=surfaces)
    poses = np.repeat(original[:1], len(original), axis=0)
    if moving:
        poses[:, 0] += np.linspace(0., .2, len(poses))
    objective.tasks.human = np.repeat(objective.tasks.human[:1], len(poses), axis=0)
    objective.kin.evaluate(poses, with_jacobian=False)
    for effector in objective.effectors:
        effector["active"][:] = effector["name"] == "foot"
        effector["managed"][:] = False
        effector["surface_only"][:] = False
        effector["contact"] = objective.kin.points(
            np.full((len(poses), 1), effector["body"]), effector["offset"][:, None],
        )[:, 0].copy()
    for surface in objective.surfaces:
        surface["targets"] = objective.kin.geom_points(surface["geoms"], surface["local"]).copy()
        surface["active"] = np.ones_like(surface["active"], bool)
    return objective, poses


def test_contact_guard_rejects_constant_lift_that_posture_and_smoothness_allow():
    objective, poses = _acquired()
    shape = ReferenceShape(objective, poses)
    guard = ContactGuard(objective, poses)
    candidate = poses.copy()
    candidate[:, objective.coords.root_q + 2] += .01
    before = verify_budgets(objective.model, objective.coords, poses, poses, objective.envelope_bodies)[0]
    after = verify_budgets(objective.model, objective.coords, poses, candidate, objective.envelope_bodies)[0]
    assert shape.accepts(candidate)
    assert smooth_correction(objective.coords, poses, candidate, before, after)[0]
    assert guard.accepts(poses)
    report = guard.diagnostics(candidate)
    assert not report["accepted"]
    assert report["max_regression_m"] == pytest.approx(.01)
    assert report["violating_frames"] == list(range(len(poses)))


def test_free_hand_is_not_promoted_to_contact():
    objective, poses = _acquired()
    guard = ContactGuard(objective, poses)
    candidate = poses.copy()
    joint = objective.model.joint("shoulder")
    candidate[:, joint.qposadr[0]] += .3
    assert guard.accepts(candidate)
    assert guard.diagnostics(candidate)["protected_endpoint_frame_points"] == len(poses)


def test_moving_targets_and_budgets_survive_objective_mutation():
    objective, poses = _acquired(moving=True)
    guard = ContactGuard(objective, poses)
    assert guard.accepts(poses)
    for effector in objective.effectors:
        effector["contact"] += 2.
        effector["active"][:] = False
        effector["offset"] += 1.
    objective.cfg["endpoint_regression_m"] = .5
    assert guard.accepts(poses)
    candidate = poses.copy()
    candidate[:, objective.coords.root_q + 2] += .002
    assert not guard.accepts(candidate)
    assert not guard.accepts(candidate)  # a rejected proposal never rebases limits


def test_whole_body_hold_is_protected_without_endpoint_or_role_weights():
    objective, poses = _acquired(surfaces=True)
    for effector in objective.effectors:
        effector["active"][:] = False
        effector["managed"][:] = True
    # Missing mechanics/priority rows do not release observed material holds.
    for surface in objective.surfaces:
        surface["priority"][:] = 0.
        surface["task"].modes = np.full(surface["active"].shape, "sticking")
    guard = ContactGuard(objective, poses)
    assert guard.accepts(poses)
    candidate = poses.copy()
    candidate[:, objective.coords.root_q + 2] += .01
    report = guard.diagnostics(candidate)
    assert not report["accepted"]
    assert report["protected_endpoint_frame_points"] == 0
    assert report["protected_surface_frame_points"] > 0


def test_authored_sliding_follows_each_moving_target_with_fixed_surface_budget():
    objective, poses = _acquired(surfaces=True, moving=True)
    for effector in objective.effectors:
        effector["active"][:] = False
    for surface in objective.surfaces:
        surface["task"].modes = np.full(surface["active"].shape, "sliding")
        surface["task"].cfg["regression"] = .0004
        surface["normals"] = np.broadcast_to([0., 0., 1.], surface["targets"].shape).copy()
    guard = ContactGuard(objective, poses)
    assert guard.accepts(poses)  # the material path moves 20 cm across the clip
    for surface in objective.surfaces:
        surface["targets"] += 1.
        surface["active"][:] = False
        surface["task"].cfg["regression"] = 1.
    candidate = poses.copy()
    candidate[:, 0] += .0003
    assert guard.accepts(candidate)
    candidate[:, 0] += .0002
    report = guard.diagnostics(candidate)
    assert not report["accepted"]  # normal alone would miss tangential drift
    assert report["groups"][0]["tolerance_m"] == .0004


def test_sliding_normal_regression_cannot_hide_in_tangent_improvement():
    objective, poses = _acquired()
    effector = objective.effectors[0]
    effector["surface_only"][:] = True
    effector["normal"][:] = [0., 0., 1.]
    effector["contact"] -= [0.1, 0., 0.]
    guard = ContactGuard(objective, poses)
    candidate = poses.copy()
    candidate[:, 0] -= .01
    candidate[:, 2] += .005
    # Total error improves from 10 cm to about 9 cm, but contact separates.
    assert np.hypot(.09, .005) < .1
    assert not guard.accepts(candidate)


def test_contact_schedule_and_stricter_existing_tolerance_are_preserved():
    objective, poses = _acquired()
    objective.cfg["endpoint_regression_m"] = .0002
    objective.effectors[0]["active"][1:] = False
    guard = ContactGuard(objective, poses)
    candidate = poses.copy()
    candidate[1:, 2] += .02
    assert guard.accepts(candidate)
    candidate[0, 2] += .0003
    report = guard.diagnostics(candidate)
    assert not report["accepted"]
    assert report["violating_frames"] == [0]
    assert report["groups"][0]["tolerance_m"] == .0002


def test_nonfinite_candidate_is_never_accepted():
    objective, poses = _acquired()
    guard = ContactGuard(objective, poses)
    candidate = poses.copy()
    candidate[0, 2] = np.nan
    assert not guard.accepts(candidate)


@pytest.mark.parametrize("surface", [False, True])
def test_position_guard_blocks_residual_sphere_motion_but_global_guard_allows_improvement(surface):
    objective, poses = _acquired(surfaces=surface, moving=True)
    if surface:
        for effector in objective.effectors:
            effector["active"][:] = False
        for patch in objective.surfaces:
            patch["targets"] -= [0.1, 0., 0.]
            patch["task"].modes = np.full(patch["active"].shape, "sticking")
            patch["normals"][:] = [0., 0., 1.]
    else:
        objective.effectors[0]["contact"] -= [0.1, 0., 0.]
    precision = ContactGuard(objective, poses)
    positions = ContactGuard(objective, poses, preserve_positions=True)
    assert positions.accepts(poses)  # each timestamp keeps its moving baseline
    rotated = poses.copy()
    angle = .1
    rotated[:, 0] += .1 * (np.cos(angle) - 1.)
    rotated[:, 2] += .1 * np.sin(angle)
    assert not precision.accepts(rotated)  # unchanged norm cannot hide separation
    assert not positions.accepts(rotated)
    tangent = rotated.copy()
    tangent[:, 2] = poses[:, 2]
    tangent[:, 1] += .1 * np.sin(angle)
    assert precision.accepts(tangent)  # unchanged residual within the tangent plane
    assert not positions.accepts(tangent)
    report = positions.diagnostics(rotated)
    assert report["preserve_positions"]
    assert report["max_position_drift_m"] > .009
    improved = poses.copy()
    improved[:, 0] -= .01
    assert precision.accepts(improved)  # grouped repair may improve this contact
    assert not positions.accepts(improved)  # free-body repair must keep its point
    within = poses.copy()
    within[:, 0] -= .0005
    assert positions.accepts(within)
