"""Moving-object contact targets and static-scene evidence boundaries."""

from dataclasses import replace

import mujoco
import numpy as np
import trimesh

from omniretargeting.contact import (
    build_world_tracking_tasks,
    infer_scene_contact_tasks,
    stack_contact_tasks,
)
from omniretargeting.object_contact import merge_object_contacts
from omniretargeting.scene_geometry import SceneQuery
from omniretargeting.trajectory_validation import (
    TrajectoryValidator,
    local_repair_config,
)
from tests._contact_fixtures import _task
from tests.test_local_collision_repair import setup_motion


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _object_case(frames=5, *, local_points=None, released=False):
    mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.1])
    angles = np.linspace(0.0, np.pi / 2.0, frames)
    rotations = np.asarray([_rot_z(angle) for angle in angles])
    scales = np.linspace(1.0, 1.2, frames)
    translations = np.column_stack([
        np.linspace(0.0, 0.08, frames),
        np.zeros(frames),
        np.zeros(frames),
    ])
    if released:
        translations[3:, 0] = 0.5
    if local_points is None:
        local_points = np.tile([0.06, 0.0, 0.05], (frames, 1))
    local_points = np.asarray(local_points, dtype=float)
    points = (
        np.einsum("ti,tji->tj", local_points, rotations) * scales[:, None]
        + translations
    )
    markers = {
        "hand": {
            "tracking_positions": points.copy(),
            "contact_samples": points[:, None, :].copy(),
            "up_directions": np.tile([0.0, 0.0, 1.0], (frames, 1)),
        }
    }
    objects = [{
        "name": "moving_box",
        "mesh": mesh,
        "translations": translations.copy(),
        "rotations": rotations.copy(),
        "scales": scales.copy(),
    }]
    return points, markers, objects


def _merge(points, markers, objects, *, framerate=30.0, original=None, **config):
    if original is None:
        original = build_world_tracking_tasks(
            {name: {"tracking_positions": value["tracking_positions"]}
             for name, value in markers.items()}
        )
    return merge_object_contacts(
        original,
        markers,
        objects,
        framerate=framerate,
        config={"min_contact_frames": 1, **config},
    )


def test_fast_moving_rotating_object_uses_zero_material_speed_for_sticking():
    points, markers, objects = _object_case()
    original_arrays = {
        "tracking_positions": markers["hand"]["tracking_positions"].copy(),
        "contact_samples": markers["hand"]["contact_samples"].copy(),
        "translations": objects[0]["translations"].copy(),
        "rotations": objects[0]["rotations"].copy(),
        "scales": objects[0]["scales"].copy(),
    }
    tasks = _merge(points, markers, objects)
    task = tasks["hand"]

    world_speed = np.linalg.norm(np.diff(points, axis=0), axis=1) * 30.0
    assert np.max(world_speed) > 0.35
    assert task.contact_mask.all()
    assert np.all(task.contact_modes == "sticking")
    np.testing.assert_allclose(task.relative_speeds, 0.0, atol=1e-9)
    np.testing.assert_allclose(task.contact_positions, points, atol=1e-8)
    np.testing.assert_array_equal(
        markers["hand"]["tracking_positions"], original_arrays["tracking_positions"]
    )
    np.testing.assert_array_equal(
        markers["hand"]["contact_samples"], original_arrays["contact_samples"]
    )
    for key in ("translations", "rotations", "scales"):
        np.testing.assert_array_equal(objects[0][key], original_arrays[key])


def test_object_release_returns_to_original_static_free_task():
    points, markers, objects = _object_case(released=True)
    release_points = np.tile([0.0, 0.0, 0.4], (2, 1))
    markers["hand"]["tracking_positions"][3:] = release_points
    markers["hand"]["contact_samples"][3:] = release_points[:, None, :]
    tasks = _merge(points, markers, objects, distance_threshold=0.06)
    task = tasks["hand"]

    assert task.contact_mask[:3].all()
    assert not task.contact_mask[3:].any()
    np.testing.assert_array_equal(task.environment_ids, [1, 1, 1, 0, 0])
    np.testing.assert_allclose(task.contact_positions[3:], release_points)
    np.testing.assert_allclose(task.tracking_positions[3:], release_points)


def test_relative_tangential_motion_is_sliding_and_has_no_static_anchor():
    local_points = np.column_stack([
        np.linspace(-0.06, 0.06, 5),
        np.zeros(5),
        np.full(5, 0.05),
    ])
    points, markers, objects = _object_case(local_points=local_points)
    tasks = _merge(
        points,
        markers,
        objects,
        sticking_tangential_speed_threshold=0.12,
    )
    task = tasks["hand"]

    assert task.contact_mask.all()
    assert np.all(task.contact_modes == "sliding")
    assert np.all(task.relative_speeds[1:-1] > 0.7)
    assert np.all(task.environment_ids == 1)
    # Sliding follows the currently observed material point; only sticking
    # intervals are eligible for the onset anchor.
    assert np.ptp(task.contact_positions[:, 0]) > 0.1
    static_query = SceneQuery.from_tasks(trimesh.creation.box(), tasks)
    assert static_query.evidence_points.shape == (0, 3)


def test_alternating_nearest_skin_sample_is_anchored_in_object_material_space():
    """Nearest-sample tie noise must not chatter a sticking target."""
    frames = 6
    mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.1])
    # Each candidate is almost stationary in material space.  A 1 mm height
    # perturbation alternates the nearest candidate, whose surface points are
    # 3 cm apart in x.
    candidate_a = np.column_stack([
        np.full(frames, 0.03),
        np.zeros(frames),
        np.array([0.054, 0.055, 0.054, 0.055, 0.054, 0.055]),
    ])
    candidate_b = np.column_stack([
        np.full(frames, -0.03),
        np.zeros(frames),
        np.array([0.055, 0.054, 0.055, 0.054, 0.055, 0.054]),
    ])
    samples = np.stack([candidate_a, candidate_b], axis=1)
    tracking = np.tile([0.0, 0.0, 0.055], (frames, 1))
    markers = {
        "hand": {
            "tracking_positions": tracking.copy(),
            "contact_samples": samples.copy(),
            "up_directions": np.tile([0.0, 0.0, 1.0], (frames, 1)),
        }
    }
    objects = [{
        "name": "moving_box",
        "mesh": mesh,
        "translations": np.zeros((frames, 3)),
        "rotations": np.tile(np.eye(3), (frames, 1, 1)),
        "scales": np.ones(frames),
    }]
    original_samples = samples.copy()
    original_tracking = tracking.copy()

    unanchored = _merge(
        tracking, markers, objects,
        anchor_contacts=False,
        contact_sample_switch_weight=0.,
        require_normal_alignment=False,
        distance_threshold=0.02,
    )["hand"]
    anchored = _merge(
        tracking, markers, objects,
        require_normal_alignment=False,
        distance_threshold=0.02,
    )["hand"]

    assert anchored.contact_mask.all()
    assert np.all(anchored.contact_modes == "sticking")
    assert np.ptp(unanchored.contact_positions[:, 0]) > 0.05
    np.testing.assert_allclose(
        anchored.contact_positions,
        np.tile(anchored.contact_positions[0], (frames, 1)),
    )
    np.testing.assert_allclose(
        anchored.surface_points,
        np.tile(anchored.surface_points[0], (frames, 1)),
    )
    np.testing.assert_allclose(
        _merge(
            tracking, markers, objects,
            anchor_contacts=False,
            require_normal_alignment=False,
            distance_threshold=0.02,
        )["hand"].contact_positions,
        anchored.contact_positions,
    )
    np.testing.assert_array_equal(markers["hand"]["contact_samples"], original_samples)
    np.testing.assert_array_equal(markers["hand"]["tracking_positions"], original_tracking)


def test_nearest_active_object_replaces_static_then_releases_and_exports_environment_ids():
    frames = 2
    static_scene = trimesh.creation.box(extents=[2.0, 2.0, 0.02])
    static_scene.apply_translation([0.0, 0.0, -0.01])
    object_mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.1])
    object_translations = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    object_spec = {
        "name": "moving_box",
        "mesh": object_mesh,
        "translations": object_translations,
        "rotations": np.tile(np.eye(3), (frames, 1, 1)),
        "scales": np.ones(frames),
    }
    points = np.array([[0.0, 0.0, 0.05], [0.0, 0.0, 0.0]])
    markers = {
        "hand": {
            "tracking_positions": points.copy(),
            "contact_samples": points[:, None, :].copy(),
            "up_directions": np.tile([0.0, 0.0, 1.0], (frames, 1)),
            "source_contact_mask": np.ones(frames, dtype=bool),
        }
    }
    static_tasks = infer_scene_contact_tasks(
        markers,
        static_scene,
        framerate=30.0,
        config={"min_contact_frames": 1, "distance_threshold": 0.06},
    )
    merged = _merge(
        points,
        markers,
        [object_spec],
        original=static_tasks,
        distance_threshold=0.06,
        require_normal_alignment=False,
    )
    task = merged["hand"]
    exported = stack_contact_tasks(merged)

    assert task.contact_mask.all()
    np.testing.assert_array_equal(task.environment_ids, [1, 0])
    np.testing.assert_allclose(task.contact_positions[0], [0.0, 0.0, 0.05])
    np.testing.assert_allclose(task.contact_positions[1], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(exported["contact_environment_ids"][:, 0], [1, 0])


def _sliding_reference_validator(frames=5, environment_ids=None):
    model, reference, _scene = setup_motion(frames)
    data = mujoco.MjData(model)
    foot_id = model.body("foot").id
    targets = []
    for pose in reference:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        targets.append(data.xpos[foot_id].copy())
    targets = np.asarray(targets)
    task = _task(
        targets,
        np.ones(frames, dtype=bool),
        ["sticking"] * frames,
    )
    if environment_ids is not None:
        task = replace(task, environment_ids=np.asarray(environment_ids, dtype=int))
    priority = {
        "effectors": {
            "left_foot": {
                "robot_body": "foot",
                "tracking_offset": [0.0, 0.0, 0.0],
                "contact_offsets": [[0.0, 0.0, 0.0]],
                "track_orientation": False,
            }
        }
    }
    far_scene = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    far_scene.apply_translation([10.0, 10.0, 10.0])
    validator = TrajectoryValidator(
        model,
        reference,
        SceneQuery(far_scene),
        {"left_foot": task},
        priority,
        local_repair_config({}),
        30.0,
    )
    return validator, reference, targets


def test_sticking_following_fast_moving_object_has_zero_target_relative_slip():
    validator, reference, targets = _sliding_reference_validator()
    world_speed = np.linalg.norm(np.diff(targets, axis=0), axis=1) * 30.0
    assert np.max(world_speed) > 0.35

    fidelity = validator.fidelity(reference)

    assert fidelity["ok"]
    assert fidelity["metrics"]["left_foot"]["support_slip"] == 0.0
    assert fidelity["metrics"]["left_foot"]["support_slip_definition"].startswith(
        "world contact-target residual velocity"
    )
    assert not any(reason.startswith("support_slip:") for reason in fidelity["reasons"])


def test_sticking_environment_change_splits_phase_without_slip_edge():
    validator, reference, _targets = _sliding_reference_validator(
        frames=3,
        environment_ids=[1, 2, 2],
    )

    assert validator.phase(0) != validator.phase(1)
    assert validator.phase(1) == validator.phase(2)
    fidelity = validator.fidelity(reference)
    assert fidelity["ok"]
    assert fidelity["metrics"]["left_foot"]["support_slip"] == 0.0
    assert not any(reason.startswith("support_slip:") for reason in fidelity["reasons"])
