"""A point can remain planted while its body turns; sliding is not sticking."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting.contact_anchors import build_rigid_anchors
from tests.test_contact_anchors import fixture


@pytest.mark.parametrize("moving_surface", [False, True])
def test_pivot_preserves_source_turn_and_fixed_material_contact(moving_surface):
    model, bindings, motion, obs = fixture()
    angle = np.linspace(0, 0.8, 5)
    turn = Rotation.from_euler("z", angle[:, None]).as_matrix()
    surface_rotation = (
        Rotation.from_euler("x", np.linspace(0, 0.4, 5)[:, None]).as_matrix()
        if moving_surface
        else np.tile(np.eye(3), (5, 1, 1))
    )
    translation = (
        np.column_stack([np.linspace(0, 0.5, 5), np.zeros((5, 2))])
        if moving_surface
        else np.zeros((5, 3))
    )
    motion.frames = (surface_rotation @ turn)[:, None] @ motion.frames
    obs["active"][:] = True
    obs["confidence"][:] = 1
    obs["modes"][:, 0] = "sticking"
    obs["modes"][:, 1] = "sliding"
    material = np.zeros((5, 2, 3))
    material[:, 0] = [-0.1, 0, 0]
    material[:, 1] = material[:, 0] + np.einsum("tij,j->ti", turn, [0.3, 0, 0])
    obs["environment"] = (
        np.einsum("tni,tji->tnj", material, surface_rotation) + translation[:, None]
    )
    original = obs["environment"].copy()
    targets, report = build_rigid_anchors(
        model,
        bindings,
        [0, 1],
        obs,
        motion,
        scene_present=not moving_surface,
        scene_objects=[{"rotations": surface_rotation, "translations": translation}]
        if moving_surface
        else (),
        rotation_budget_rad=0.2,
    )
    # Contact location, authored sliding, scale and timing are all retained.
    np.testing.assert_allclose(targets, original, atol=1e-12)
    np.testing.assert_array_equal(obs["environment"], original)
    np.testing.assert_allclose(
        np.linalg.norm(targets[:, 1] - targets[:, 0], axis=1), 0.3
    )
    assert report["moving_contact_intervals"] == 1
    placement = report["placements"][0]
    assert not placement["locked"]
    np.testing.assert_allclose(placement["rotation"], turn, atol=1e-12)


def test_single_sticking_point_does_not_freeze_unobserved_orientation():
    model, bindings, motion, obs = fixture()
    obs["active"][:, 0] = True
    obs["active"][:, 1] = False
    obs["modes"][:] = "none"
    obs["modes"][:, 0] = "sticking"
    obs["confidence"][:] = obs["active"]
    turn = Rotation.from_euler("y", np.linspace(0, 1.0, 5)[:, None]).as_matrix()
    motion.frames = turn[:, None] @ motion.frames
    targets, report = build_rigid_anchors(
        model,
        bindings,
        [0, 1],
        obs,
        motion,
        scene_present=True,
        rotation_budget_rad=0.2,
    )
    np.testing.assert_allclose(targets[:, 0], obs["environment"][:, 0], atol=1e-12)
    np.testing.assert_allclose(report["placements"][0]["rotation"], turn, atol=1e-12)
    assert report["placements"][0]["rejected_locked_source_error_rad"] > 0.2


def test_dynamic_morphology_mismatch_is_reported_without_changing_size_or_budget():
    model, bindings, motion, obs = fixture()
    turn = Rotation.from_euler("z", np.linspace(0, 0.9, 5)[:, None]).as_matrix()
    motion.frames = turn[:, None] @ motion.frames
    obs["active"][:] = True
    obs["modes"][:] = "sticking"
    obs["confidence"][:] = 1
    targets, report = build_rigid_anchors(
        model,
        bindings,
        [0, 1],
        obs,
        motion,
        scene_present=True,
        rotation_budget_rad=0.2,
    )
    np.testing.assert_allclose(targets, obs["environment"], atol=1e-12)
    assert report["targets_on_observed_surface"]
    assert report["rigid_fit_target_residual_max_m"] > 0.01
    p = report["placements"][0]
    assert p["fit"]["maximum_m"] > 0.01
    assert p["orientation_budget_rad"] == 0.2
    angles = Rotation.from_matrix(p["rotation"] @ turn.transpose(0, 2, 1)).magnitude()
    assert angles.max() <= 0.2 + 1e-12


@pytest.mark.parametrize("surface_axis", ["x", "z"])
def test_incompatible_tilt_cannot_move_targets_off_observed_plane(surface_axis):
    model, bindings, motion, obs = fixture()
    obs["active"][:] = True
    obs["confidence"][:] = 1
    obs["modes"][:] = "sticking"
    source_turn = Rotation.from_euler("y", np.linspace(0, 0.9, 5)[:, None]).as_matrix()
    surface = Rotation.from_euler(
        "y", np.pi / 2 if surface_axis == "x" else 0
    ).as_matrix()
    obs["environment"] = obs["environment"] @ surface.T
    motion.frames = (surface @ source_turn)[:, None] @ motion.frames
    original = obs["environment"].copy()
    targets, report = build_rigid_anchors(
        model,
        bindings,
        [0, 1],
        obs,
        motion,
        scene_present=True,
        rotation_budget_rad=0.2,
    )
    np.testing.assert_allclose(targets, original, atol=1e-12)
    normal = surface[:, 2]
    np.testing.assert_allclose(targets @ normal, 0, atol=1e-12)
    assert report["rigid_fit_target_residual_max_m"] > 0.02
    assert report["targets_on_observed_surface"]


def test_curved_material_target_uses_observed_point_and_retains_sliding():
    from omniretargeting.anchor_motion import observed_material_targets

    turns = Rotation.from_euler("xyz", np.arange(15).reshape(5, 3) * 0.05).as_matrix()
    translation = np.arange(15).reshape(5, 3) * 0.1
    scales = np.linspace(0.7, 1.2, 5)
    angles = np.linspace(-0.4, 0.4, 5)
    material = np.zeros((5, 2, 3))
    material[:, :, 0] = np.cos(angles)[:, None]
    material[:, :, 1] = np.sin(angles)[:, None]
    world = (
        np.einsum("tnj,tij->tni", material, turns) * scales[:, None, None]
        + translation[:, None]
    )
    modes = np.tile(["sticking", "sliding"], (5, 1))
    original = world.copy()
    result = observed_material_targets(
        world, np.ones((5, 2), bool), modes, np.ones((5, 2)), turns, translation, scales
    )
    local = (
        np.einsum("tni,tij->tnj", result - translation[:, None], turns)
        / scales[:, None, None]
    )
    np.testing.assert_allclose(local[:, 0], np.tile(material[2, 0], (5, 1)), atol=1e-12)
    np.testing.assert_array_equal(result[:, 1], original[:, 1])
    np.testing.assert_array_equal(world, original)
    # An averaged point would be inside the cylinder, not on its surface.
    np.testing.assert_allclose(np.linalg.norm(local[:, 0, :2], axis=1), 1, atol=1e-12)
