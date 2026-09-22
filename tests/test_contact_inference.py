"""Scene contact timing, normals, anchors and export regression tests."""

import numpy as np
import trimesh

from omniretargeting.contact import (
    infer_scene_contact_tasks,
    stack_contact_tasks,
)


def test_scene_contact_inference_filters_and_anchors_contact_runs():
    scene = trimesh.creation.box(extents=[4.0, 4.0, 0.1])
    scene.apply_translation([0.0, 0.0, -0.05])

    tracking = np.array(
        [
            [0.0, 0.0, 0.30],
            [0.0, 0.0, 0.20],
            [0.0, 0.0, 0.01],
            [0.0, 0.0, 0.01],
            [0.0, 0.0, 0.01],
            [0.0, 0.0, 0.01],
            [0.0, 0.0, 0.01],
            [0.0, 0.0, 0.01],
            [0.0, 0.0, 0.20],
            [0.0, 0.0, 0.30],
        ]
    )
    markers = {
        "left_foot": {
            "tracking_positions": tracking,
            "contact_samples": tracking[:, None, :],
        }
    }

    task = infer_scene_contact_tasks(
        markers,
        scene,
        framerate=30.0,
        config={
            "distance_threshold": 0.03,
            "relative_speed_threshold": 0.1,
            "min_contact_frames": 3,
            "max_gap_frames": 0,
            "anchor_contacts": True,
        },
    )["left_foot"]

    assert np.array_equal(
        task.contact_mask,
        np.array([False, False, False, True, True, True, True, True, False, False]),
    )
    anchored = task.contact_positions[task.contact_mask]
    assert np.all(task.contact_modes[task.contact_mask] == "sticking")
    np.testing.assert_allclose(
        anchored, np.repeat(anchored[0:1], len(anchored), axis=0), atol=1e-10
    )
    np.testing.assert_allclose(anchored[:, 2], 0.0, atol=1e-10)


def test_source_contact_hint_preserves_timing_despite_reconstruction_error():
    scene = trimesh.creation.box(extents=[2.0, 2.0, 0.2])
    scene.apply_translation([0.0, 0.0, -0.1])
    positions = np.asarray(
        [
            [0.0, 0.0, 0.20],
            [0.2, 0.0, 0.18],
            [0.4, 0.0, 0.16],
            [0.6, 0.0, 0.14],
        ]
    )
    source_mask = np.asarray([False, True, True, False])

    task = infer_scene_contact_tasks(
        {
            "right_foot": {
                "tracking_positions": positions,
                "contact_samples": positions[:, None, :],
                "source_contact_mask": source_mask,
            }
        },
        scene,
        framerate=30.0,
        config={
            "distance_threshold": 0.01,
            "relative_speed_threshold": 0.01,
            "min_contact_frames": 3,
            "max_gap_frames": 0,
            "anchor_contacts": False,
        },
    )["right_foot"]

    np.testing.assert_array_equal(task.contact_mask, source_mask)
    np.testing.assert_allclose(
        task.contact_positions[source_mask, 2], 0.0, atol=1e-10
    )
    np.testing.assert_allclose(
        task.contact_confidence[source_mask], 1.0, atol=1e-10
    )


def test_penetrating_video_contact_keeps_normal_on_approach_side():
    """Source penetration must never flip a robot target into the scene."""
    scene = trimesh.creation.box(extents=[2.0, 2.0, 0.2])
    scene.apply_translation([0.0, 0.0, -0.1])
    heights = np.asarray([0.06, 0.02, -0.005, -0.02, -0.005, 0.02, 0.06])
    positions = np.column_stack(
        [np.zeros(len(heights)), np.zeros(len(heights)), heights]
    )

    task = infer_scene_contact_tasks(
        {
            "left_hand": {
                "tracking_positions": positions,
                "contact_samples": positions[:, None, :],
            }
        },
        scene,
        framerate=30.0,
        config={
            "distance_threshold": 0.03,
            "relative_speed_threshold": 10.0,
            "min_contact_frames": 1,
            "max_gap_frames": 0,
            "surface_clearance": 0.005,
            "effectors": {
                "left_hand": {"require_normal_alignment": False}
            },
        },
    )["left_hand"]

    expected = np.asarray([False, True, True, True, True, True, False])
    np.testing.assert_array_equal(task.contact_mask, expected)
    assert np.all(task.surface_normals[expected, 2] > 0.99)
    np.testing.assert_allclose(
        task.contact_positions[expected, 2], 0.005, atol=1e-10
    )


def test_on_surface_contact_noise_preserves_consistent_mesh_normals():
    scene = trimesh.Trimesh(
        vertices=np.asarray(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ]
        ),
        faces=np.asarray([[0, 1, 2], [0, 2, 3]]),
        process=False,
    )
    heights = np.asarray([-0.00010, 0.00008, -0.00009, 0.00006])
    positions = np.column_stack(
        [np.zeros(len(heights)), np.zeros(len(heights)), heights]
    )

    task = infer_scene_contact_tasks(
        {
            "left_foot": {
                "tracking_positions": positions,
                "contact_samples": positions[:, None, :],
                "source_contact_mask": np.ones(len(positions), dtype=bool),
            }
        },
        scene,
        framerate=30.0,
        config={
            "min_contact_frames": 1,
            "surface_clearance": 0.005,
            "normal_orientation_deadzone": 0.005,
        },
    )["left_foot"]

    assert np.all(task.contact_modes == "sticking")
    assert np.all(task.surface_normals[:, 2] > 0.0)
    assert np.all(task.contact_positions[:, 2] >= task.surface_points[:, 2])


def test_large_surface_offset_still_orients_normal_toward_source():
    scene = trimesh.Trimesh(
        vertices=np.asarray(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ]
        ),
        faces=np.asarray([[0, 1, 2], [0, 2, 3]]),
        process=False,
    )
    positions = np.asarray([[0.0, 0.0, -0.02]])

    task = infer_scene_contact_tasks(
        {
            "left_hand": {
                "tracking_positions": positions,
                "contact_samples": positions[:, None, :],
                "source_contact_mask": np.ones(len(positions), dtype=bool),
            }
        },
        scene,
        framerate=30.0,
        config={
            "anchor_contacts": False,
            "stabilize_contact_normals": False,
            "surface_clearance": 0.005,
            "normal_orientation_deadzone": 0.005,
        },
    )["left_hand"]

    assert task.surface_normals[0, 2] < 0.0
    assert task.contact_positions[0, 2] < task.surface_points[0, 2]


def test_contact_modes_separate_normal_transition_from_surface_slide():
    scene = trimesh.creation.box(extents=[2.0, 2.0, 0.2])
    scene.apply_translation([0.0, 0.0, -0.1])
    positions = np.asarray(
        [
            [0.00, 0.0, 0.060],
            [0.00, 0.0, 0.020],
            [0.00, 0.0, 0.005],
            [0.02, 0.0, 0.005],
            [0.04, 0.0, 0.005],
            [0.06, 0.0, 0.050],
        ]
    )

    task = infer_scene_contact_tasks(
        {
            "left_hand": {
                "tracking_positions": positions,
                "contact_samples": positions[:, None, :],
            }
        },
        scene,
        framerate=30.0,
        config={
            "distance_threshold": 0.03,
            "relative_speed_threshold": 10.0,
            "min_contact_frames": 1,
            "max_gap_frames": 0,
            "sticking_tangential_speed_threshold": 0.1,
            "sticking_normal_speed_threshold": 0.1,
            "sliding_normal_speed_threshold": 0.1,
            "min_sliding_frames": 2,
            "effectors": {
                "left_hand": {"require_normal_alignment": False}
            },
        },
    )["left_hand"]

    assert task.contact_modes.tolist() == [
        "none",
        "transition",
        "transition",
        "sliding",
        "sliding",
        "none",
    ]


def test_contact_target_uses_nearby_surface_patch_median():
    scene = trimesh.creation.box(extents=[2.0, 2.0, 0.2])
    scene.apply_translation([0.0, 0.0, -0.1])
    samples = np.asarray(
        [[[-0.02, 0.0, 0.001], [0.0, 0.0, 0.002], [0.02, 0.0, 0.003]]]
    )

    task = infer_scene_contact_tasks(
        {
            "left_hand": {
                "tracking_positions": np.asarray([[0.0, 0.0, 0.005]]),
                "contact_samples": samples,
            }
        },
        scene,
        framerate=30.0,
        config={
            "distance_threshold": 0.01,
            "relative_speed_threshold": 1.0,
            "min_contact_frames": 1,
            "contact_patch_distance": 0.01,
            "contact_patch_spatial_radius": 0.1,
            "effectors": {
                "left_hand": {"require_normal_alignment": False}
            },
        },
    )["left_hand"]

    assert task.contact_mask[0]
    np.testing.assert_allclose(task.surface_points[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(task.contact_positions[0], [0.0, 0.0, 0.0])


def test_contact_inference_classifies_sliding_hand_without_moving_foot_policy():
    scene = trimesh.creation.box(extents=[4.0, 4.0, 0.1])
    scene.apply_translation([0.0, 0.0, -0.05])
    tracking = np.column_stack(
        [np.linspace(0.0, 0.3, 6), np.zeros(6), np.full(6, 0.01)]
    )
    tasks = infer_scene_contact_tasks(
        {
            name: {
                "tracking_positions": tracking,
                "contact_samples": tracking[:, None, :],
            }
            for name in ("left_hand", "left_foot")
        },
        scene,
        framerate=30.0,
        config={
            "distance_threshold": 0.03,
            "relative_speed_threshold": 10.0,
            "min_contact_frames": 1,
            "anchor_contacts": True,
            "effectors": {
                "left_hand": {
                    "sticking_tangential_speed_threshold": 0.1,
                    "sticking_normal_speed_threshold": 0.1,
                },
                "left_foot": {
                    "sticking_tangential_speed_threshold": 10.0,
                    "sticking_normal_speed_threshold": 10.0,
                },
            },
        },
    )

    hand_targets = tasks["left_hand"].contact_positions
    foot_targets = tasks["left_foot"].contact_positions
    assert np.all(tasks["left_hand"].contact_modes == "sliding")
    assert np.all(tasks["left_foot"].contact_modes == "sticking")
    assert np.ptp(hand_targets[:, 0]) > 0.25
    np.testing.assert_allclose(
        foot_targets,
        np.repeat(foot_targets[0:1], len(foot_targets), axis=0),
        atol=1e-10,
    )


def test_contact_tasks_stack_for_downstream_policy_export():
    scene = trimesh.creation.box(extents=[2.0, 2.0, 0.1])
    scene.apply_translation([0.0, 0.0, -0.05])
    positions = np.zeros((4, 3), dtype=float)
    positions[:, 2] = 0.01
    tasks = infer_scene_contact_tasks(
        {
            name: {
                "tracking_positions": positions,
                "contact_samples": positions[:, None, :],
            }
            for name in ("left_hand", "right_foot")
        },
        scene,
        framerate=30.0,
        config={"min_contact_frames": 1},
    )

    payload = stack_contact_tasks(tasks)
    assert payload["contact_mask"].shape == (4, 2)
    assert payload["contact_modes"].shape == (4, 2)
    assert payload["contact_targets_w"].shape == (4, 2, 3)
    assert payload["end_effector_reference_w"].shape == (4, 2, 3)


def test_foot_reference_axes_form_valid_contact_frames():
    scene = trimesh.creation.box(extents=[2.0, 2.0, 0.1])
    scene.apply_translation([0.0, 0.0, -0.05])
    positions = np.zeros((5, 3), dtype=float)
    positions[:, 2] = 0.01
    task = infer_scene_contact_tasks(
        {
            "left_foot": {
                "tracking_positions": positions,
                "contact_samples": positions[:, None, :],
                "forward_directions": np.tile([1.0, 0.0, -0.3], (5, 1)),
                "up_directions": np.tile([0.0, 0.0, 1.0], (5, 1)),
            }
        },
        scene,
        framerate=30.0,
        config={
            "distance_threshold": 0.03,
            "relative_speed_threshold": 0.1,
            "min_contact_frames": 1,
        },
    )["left_foot"]

    np.testing.assert_allclose(
        np.sum(task.forward_directions * task.up_directions, axis=1),
        0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.linalg.norm(task.forward_directions, axis=1), 1.0, atol=1e-10
    )
    np.testing.assert_allclose(
        np.linalg.norm(task.up_directions, axis=1), 1.0, atol=1e-10
    )


def test_hand_orientation_does_not_force_sole_normal_contact_filter():
    scene = trimesh.creation.box(extents=[2.0, 2.0, 0.1])
    scene.apply_translation([0.0, 0.0, -0.05])
    positions = np.tile([0.0, 0.0, 0.01], (5, 1))
    task = infer_scene_contact_tasks(
        {
            "left_hand": {
                "tracking_positions": positions,
                "contact_samples": positions[:, None, :],
                "forward_directions": np.tile([1.0, 0.0, 0.0], (5, 1)),
                "up_directions": np.tile([0.0, 0.0, -1.0], (5, 1)),
            }
        },
        scene,
        framerate=30.0,
        config={
            "distance_threshold": 0.03,
            "relative_speed_threshold": 0.1,
            "min_contact_frames": 1,
            "effectors": {
                "left_hand": {"require_normal_alignment": False}
            },
        },
    )["left_hand"]

    assert np.all(task.contact_mask)

