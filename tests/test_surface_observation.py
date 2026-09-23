"""Regression tests for metric, material-relative surface observations."""

import numpy as np
import trimesh

from omniretargeting.surface_observation import observe_contacts


def _config(**overrides):
    config = {
        "touch_distance": 0.008,
        "distance": 0.03,
        "normal_opposition": 0.2,
        "normal_speed": 0.2,
        "sticking_speed": 0.04,
        "min_contact_frames": 1,
    }
    config.update(overrides)
    return config


def _floor():
    floor = trimesh.creation.box(extents=[2.0, 2.0, 0.1])
    floor.apply_translation([0.0, 0.0, -0.05])
    return floor


def test_fast_near_pass_is_rejected_but_brief_close_touch_is_retained():
    # The first point crosses the 2 cm near band at several m/s.  The second
    # point has the same brief pass, but reaches the 5 mm close band, which is
    # retained even though it is too fast to be classified as sticking.
    points = np.zeros((4, 2, 3))
    points[:, 0, 2] = [0.20, 0.02, 0.03, 0.20]
    points[:, 1, 2] = [0.20, 0.005, 0.006, 0.20]
    skin_normals = np.tile([0.0, 0.0, -1.0], (4, 2, 1))
    valid = np.ones((4, 2), dtype=bool)
    points_before, normals_before, valid_before = (
        points.copy(),
        skin_normals.copy(),
        valid.copy(),
    )

    observed = observe_contacts(
        points, skin_normals, valid, _floor(), _config(), framerate=30.0
    )

    assert not observed["active"][1, 0]
    assert observed["active"][1, 1]
    assert observed["modes"][1, 0] == "none"
    assert observed["modes"][1, 1] == "touch"
    np.testing.assert_array_equal(points, points_before)
    np.testing.assert_array_equal(skin_normals, normals_before)
    np.testing.assert_array_equal(valid, valid_before)


def test_backside_surface_sample_is_rejected():
    points = np.tile([0.0, 0.0, 0.005], (4, 1, 1))
    # The floor normal points upward, so this source normal is on the back
    # side instead of opposing the scene surface normal.
    skin_normals = np.tile([0.0, 0.0, 1.0], (4, 1, 1))
    observed = observe_contacts(
        points,
        skin_normals,
        np.ones((4, 1), dtype=bool),
        _floor(),
        _config(),
        framerate=30.0,
    )
    assert not observed["active"].any()
    assert np.all(observed["confidence"] == 0.0)
    assert observed["diagnostics"]["rejected_facing"] == 4


def test_skin_contact_on_inside_of_closed_room_uses_observed_side():
    # A closed room shell has the same topology as a solid box but the human
    # occupies its interior. Its mesh winding cannot define the contact side.
    room = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    points = np.tile([0.98, 0.0, 0.0], (4, 1, 1))
    observed = observe_contacts(
        points, np.tile([1.0, 0.0, 0.0], (4, 1, 1)),
        np.ones((4, 1), dtype=bool), room, _config(), framerate=30.0,
    )

    assert observed["active"].all()
    np.testing.assert_allclose(
        observed["normals"][:, 0], np.tile([-1.0, 0.0, 0.0], (4, 1))
    )
    np.testing.assert_allclose(observed["distance"], 0.02)


def test_tangential_motion_is_sliding_and_material_point_is_not_pinned():
    points = np.zeros((4, 1, 3))
    points[:, 0, 0] = [0.0, 0.02, 0.04, 0.06]
    points[:, 0, 2] = 0.005
    observed = observe_contacts(
        points,
        np.tile([0.0, 0.0, -1.0], (4, 1, 1)),
        np.ones((4, 1), dtype=bool),
        _floor(),
        _config(),
        framerate=30.0,
    )
    assert observed["active"].all()
    assert np.all(observed["modes"] == "sliding")
    np.testing.assert_allclose(observed["normal_speed"], 0.0, atol=1e-10)
    np.testing.assert_allclose(observed["tangent_speed"], 0.6, atol=1e-10)
    np.testing.assert_allclose(observed["material_points"][:, 0, 0], points[:, 0, 0])
    assert np.ptp(observed["material_points"][:, 0, 0]) > 0.05


def test_translating_and_rotating_object_uses_material_relative_sticking():
    object_mesh = trimesh.creation.box(extents=[0.4, 0.4, 0.1])
    frames = 5
    translations = np.column_stack(
        [np.linspace(0.0, 0.2, frames), np.zeros(frames), np.zeros(frames)]
    )
    rotations = []
    for angle in np.linspace(0.0, np.pi / 2.0, frames):
        rotations.append(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    rotations = np.asarray(rotations)
    material_point = np.array([0.1, 0.0, 0.05])
    points = np.einsum("i,tji->tj", material_point, rotations) + translations
    points = points[:, None, :]
    points_before = points.copy()

    observed = observe_contacts(
        points,
        np.tile([0.0, 0.0, -1.0], (frames, 1, 1)),
        np.ones((frames, 1), dtype=bool),
        None,
        _config(),
        framerate=30.0,
        scene_objects=[
            {
                "name": "moving_panel",
                "mesh": object_mesh,
                "translations": translations,
                "rotations": rotations,
            }
        ],
    )

    assert observed["active"].all()
    assert np.all(observed["modes"] == "sticking")
    np.testing.assert_allclose(observed["normal_speed"], 0.0, atol=1e-9)
    np.testing.assert_allclose(observed["tangent_speed"], 0.0, atol=1e-9)
    np.testing.assert_allclose(
        observed["material_points"][:, 0],
        np.broadcast_to(material_point, (frames, 3)),
        atol=1e-8,
    )
    np.testing.assert_array_equal(points, points_before)


def test_object_scale_stays_in_world_metric_distance():
    object_mesh = trimesh.creation.box(extents=[0.4, 0.4, 0.1])
    frames = 3
    points = np.tile([0.0, 0.0, 0.11], (frames, 1, 1))
    observed = observe_contacts(
        points,
        np.tile([0.0, 0.0, -1.0], (frames, 1, 1)),
        np.ones((frames, 1), dtype=bool),
        None,
        _config(),
        scene_objects=[
            {
                "name": "scaled_panel",
                "mesh": object_mesh,
                "translations": np.zeros((frames, 3)),
                "rotations": np.tile(np.eye(3), (frames, 1, 1)),
                "scales": np.full(frames, 2.0),
            }
        ],
    )

    # The local top is z=.05 and scale=2, so the actual world clearance is
    # 1.1 cm.  A normalized/local-distance implementation would report 0.5 cm
    # and saturate the confidence at one.
    np.testing.assert_allclose(
        observed["environment"][:, 0],
        np.broadcast_to([0.0, 0.0, 0.1], (frames, 3)),
    )
    expected_strength = (0.03 - 0.01) / (0.03 - 0.008)
    expected_confidence = expected_strength**2 * (3.0 - 2.0 * expected_strength)
    np.testing.assert_allclose(observed["confidence"][:, 0], expected_confidence, atol=1e-6)
    assert np.all(observed["active"])


def test_releasing_one_sample_does_not_renormalize_other_objective():
    points = np.zeros((6, 2, 3))
    points[:, 0, 2] = 0.005
    points[:, 1, 2] = [0.005, 0.020, 0.020, 0.029, 0.031, 0.031]
    observed = observe_contacts(
        points,
        np.tile([0.0, 0.0, -1.0], (6, 2, 1)),
        np.ones((6, 2), dtype=bool),
        _floor(),
        _config(),
        framerate=30.0,
    )

    # The permanently touching sample retains its exact objective strength
    # while the neighboring sample fades with its own distance and releases.
    np.testing.assert_allclose(observed["confidence"][:, 0], 1.0, atol=1e-12)
    assert observed["confidence"][2, 1] > observed["confidence"][3, 1] > 0.0
    assert observed["confidence"][5, 1] == 0.0
    assert not observed["active"][5, 1]
