"""Behavioral checks independent of G1, OMOMO names and recording frequency."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from omniretargeting.contact import build_world_tracking_tasks
from omniretargeting.sequence_retargeting import retarget_sequence
from omniretargeting._sequence.coordinates import Coordinates, temporal_operators


def slider_model(joints):
    chain = "".join(
        f'<body name="link_{i}" pos="0 0 .15"><joint type="slide" axis="1 0 0" range="-.8 .8"/><geom size=".03" mass="1"/>'
        for i in range(joints)
    )
    return mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>" + chain + "</body>" * joints + "</worldbody></mujoco>"
    )


@pytest.mark.parametrize("joints,fps", [(1, 15), (2, 30), (4, 60)])
def test_unseen_models_remove_alternating_ik_error_without_changing_targets(
    joints, fps
):
    model = slider_model(joints)
    times = np.linspace(0, 1, fps + 1)
    intended = 0.12 * np.sin(2 * np.pi * times)
    q = np.zeros((len(times), model.nq))
    q[:, 0] = intended
    data = mujoco.MjData(model)
    source = []
    for pose in q:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        source.append(data.xpos[1:].copy())
    source = np.asarray(source)
    names = [f"marker_{i}" for i in range(joints)]
    mapping = dict(zip(names, [f"link_{i}" for i in range(joints)]))
    targets = build_world_tracking_tasks(
        {"tool": {"tracking_positions": source[:, -1]}}
    )
    priority = {
        "effectors": {
            "tool": {"robot_body": f"link_{joints - 1}", "preserve_world_target": True}
        }
    }
    seed = q.copy()
    seed[:, 0] += 0.06 * (-1.0) ** np.arange(len(times))
    original = source.copy()
    seed_copy = seed.copy()
    solved, report = retarget_sequence(
        model,
        seed,
        source,
        names,
        mapping,
        source[:, 0],
        targets,
        priority,
        framerate=fps,
        config={"self_collision_enabled": False, "iterations": 20},
    )
    assert np.array_equal(source, original) and np.array_equal(seed, seed_copy)
    actual = solved.sum(axis=1)
    assert np.max(abs(actual - intended)) < 0.025
    assert np.sqrt(np.mean(np.diff(solved, n=2, axis=0) ** 2)) < 0.2 * np.sqrt(
        np.mean(np.diff(seed, n=2, axis=0) ** 2)
    )
    assert report["scalar_joint_count"] == joints
    assert np.max(abs(solved)) < 0.8


@pytest.mark.parametrize("fps", [15, 30, 60, 120])
def test_temporal_derivatives_use_seconds(fps):
    time = np.arange(fps + 1) / fps
    operators = temporal_operators(len(time), fps, [1.0, 1.0, 1.0])
    assert np.allclose(operators[0] @ time, 1.0)
    assert np.allclose(operators[1] @ time**2, 2.0)
    assert np.allclose(operators[2] @ time**3, 6.0, atol=1e-7)


def test_quaternion_signs_and_non_g1_coordinate_layout():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body><freejoint/><geom size=".1"/><body><joint type="hinge"/><geom size=".1"/></body></body></worldbody></mujoco>'
    )
    coordinates = Coordinates(model, lock_root_xy=True)
    poses = np.tile(model.qpos0, (4, 1))
    poses[1::2, 3:7] *= -1
    normalized = coordinates.normalize(poses)
    assert np.allclose(normalized, model.qpos0)
    assert coordinates.selector.shape == (1, 5)
    step = np.ones((4, 5)) * 0.01
    moved = coordinates.integrate(normalized, step, 1.0)
    assert np.array_equal(moved[:, :2], normalized[:, :2])
    assert np.allclose(np.linalg.norm(moved[:, 3:7], axis=1), 1.0)
    locked = Coordinates(model, lock_root_xy=True, lock_root_z=True, lock_root_orientation=True)
    fully_locked = locked.integrate(normalized, np.full((4, 1), 0.01), 1.0)
    assert np.array_equal(fully_locked[:, :7], normalized[:, :7])


@pytest.mark.parametrize("frames", [1, 2, 3])
def test_short_sequences_are_supported(frames):
    model = slider_model(1)
    positions = np.zeros((frames, 1, 3))
    positions[:, 0, 2] = 0.15
    solved, report = retarget_sequence(
        model,
        np.zeros((frames, 1)),
        positions,
        ["marker"],
        {"marker": "link_0"},
        positions[:, 0],
        {},
        {},
        framerate=24,
        config={"self_collision_enabled": False, "iterations": 2},
    )
    assert solved.shape == (frames, 1) and np.all(np.isfinite(solved))
    assert report["frames"] == frames


@pytest.mark.parametrize("fps", [30, 60])
def test_fast_prescribed_contact_motion_is_not_delayed_or_made_root_relative(fps):
    model = slider_model(1)
    times = np.arange(fps + 1) / fps
    target = np.zeros((len(times), 3))
    target[:, 2] = 0.15
    target[:, 0] = 0.18 * np.exp(-(((times - 0.5) / 0.07) ** 2))
    task = build_world_tracking_tasks({"tool": {"tracking_positions": target}})["tool"]
    task = replace(
        task,
        contact_mask=np.ones(len(times), dtype=bool),
        contact_modes=np.full(len(times), "sticking"),
    )
    priority = {
        "effectors": {
            "tool": {
                "robot_body": "link_0",
                "preserve_world_target": False,
                "swing_morphology_scale": 0.5,
            }
        }
    }
    seed = target[:, :1].copy() + 0.03 * (-1.0) ** np.arange(len(times))[:, None]
    unchanged = task.contact_positions.copy()
    result, report = retarget_sequence(
        model,
        seed,
        target[:, None],
        ["source"],
        {"source": "link_0"},
        np.tile([4.0, 3.0, 2.0], (len(times), 1)),
        {"tool": task},
        priority,
        framerate=fps,
        config={"self_collision_enabled": False, "iterations": 20},
    )
    assert np.array_equal(task.contact_positions, unchanged)
    assert np.argmax(result[:, 0]) == np.argmax(target[:, 0])
    assert np.max(abs(result[:, 0] - target[:, 0])) < 0.006
    assert report["final_residuals"]["sticking_max_m"] < 0.006


def test_final_validator_keeps_the_sequence_material_marker():
    import trimesh
    from omniretargeting.scene_geometry import SceneQuery
    from omniretargeting.trajectory_validation import (
        TrajectoryValidator,
        local_repair_config,
    )

    model = slider_model(1)
    positions = np.tile([0.0, 0.0, 0.15], (3, 1))
    task = build_world_tracking_tasks({"tool": {"tracking_positions": positions}})[
        "tool"
    ]
    task = replace(task, contact_mask=np.ones(3, dtype=bool))
    priority = {
        "effectors": {
            "tool": {
                "robot_body": "link_0",
                "contact_offsets": [[0, 0, 0], [0.1, 0, 0]],
            }
        }
    }
    scene = trimesh.creation.box(extents=[2, 2, 0.1])
    scene.apply_translation([0, 0, -0.05])
    choices = {"tool": np.ones(3, dtype=int)}
    validator = TrajectoryValidator(
        model,
        np.zeros((3, 1)),
        SceneQuery(scene),
        {"tool": task},
        priority,
        local_repair_config({"self_collision_enabled": False}),
        30,
        contact_offset_choice=choices,
    )
    assert np.array_equal(validator.offset_choice["tool"], [1, 1, 1])
    choices["tool"][:] = 0
    assert np.array_equal(validator.offset_choice["tool"], [1, 1, 1])


def test_core_sequence_initializes_from_model_without_a_saved_motion():
    from types import SimpleNamespace
    from omniretargeting._core.sequence import solve_core_sequence

    model = slider_model(1)
    positions = np.zeros((16, 1, 3))
    positions[:, 0, 0] = np.linspace(0.1, 0.3, len(positions))
    positions[:, 0, 2] = 0.15
    task = build_world_tracking_tasks(
        {"tool": {"tracking_positions": positions[:, 0]}}
    )["tool"]
    task = replace(
        task,
        contact_mask=np.ones(len(positions), dtype=bool),
        contact_modes=np.full(len(positions), "sticking"),
    )
    owner = SimpleNamespace(
        robot_model=model,
        smplx_joint_names=["observed"],
        valid_joint_mapping={"observed": "link_0"},
        _source_framerate=24,
        _resolve_source_root_index=lambda: 0,
        retargeting_config={
            "self_collision_enabled": False,
            "sequence_solver": {"iterations": 15},
            "end_effector_priority": {
                "effectors": {
                    "tool": {"robot_body": "link_0", "preserve_world_target": True}
                }
            },
        },
    )
    retargeter = SimpleNamespace(
        robot_reference_qpos=None, terrain_points=np.empty((0, 3)), scene_query=None
    )
    result = solve_core_sequence(
        owner, retargeter, positions, {"tool": task}, None, None
    )
    assert np.max(abs(result[:, 0] - positions[:, 0, 0])) < 0.005
    assert owner.last_sequence_report["initialization"] is not None


def test_cold_foot_stays_on_the_source_side_of_a_finite_terrain_slab():
    import trimesh
    from omniretargeting.scene_geometry import SceneQuery

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="sole"><joint type="slide" axis="0 0 1" range="-1 1"/><geom size=".02" mass="1"/></body></worldbody></mujoco>'
    )
    positions = np.tile([0.0, 0.0, 0.03], (4, 1))
    tasks = build_world_tracking_tasks({"support": {"tracking_positions": positions}})
    priority = {
        "effectors": {
            "support": {
                "robot_body": "sole",
                "preserve_world_target": True,
                "track_orientation": True,
                "contact_offsets": [[0, 0, -0.02]],
            }
        }
    }
    scene = trimesh.creation.box(extents=[2, 2, 0.1])
    scene.apply_translation([0, 0, -0.05])
    original = positions.copy()
    result, _ = retarget_sequence(
        model,
        np.full((4, 1), -0.2),
        positions[:, None],
        ["observed"],
        {"observed": "sole"},
        positions,
        tasks,
        priority,
        framerate=24,
        scene_query=SceneQuery(scene),
        config={"self_collision_enabled": False, "source_side_support": True},
    )
    assert np.min(result[:, 0] - 0.02) >= -1e-5
    assert np.max(abs(result[:, 0] - 0.03)) < 0.005
    assert np.array_equal(positions, original)


def test_sequence_contacts_keep_world_anchors_when_transition_weights_fade():
    model = slider_model(1)
    positions = np.tile([0.0, 0.0, 0.15], (4, 1))
    task = build_world_tracking_tasks(
        {"tool": {"tracking_positions": positions + [1.0, 0.0, 0.0]}}
    )["tool"]
    task = replace(
        task,
        contact_positions=positions.copy(),
        contact_mask=np.ones(4, dtype=bool),
        contact_modes=np.full(4, "sticking"),
        transition_weights=np.full(4, 0.2),
    )
    result, _ = retarget_sequence(
        model,
        np.full((4, 1), 0.4),
        positions[:, None],
        ["observed"],
        {"observed": "link_0"},
        positions,
        {"tool": task},
        {"effectors": {"tool": {"robot_body": "link_0"}}},
        framerate=30,
        config={"self_collision_enabled": False},
    )
    assert np.max(abs(result[:, 0])) < 0.005
    assert np.array_equal(task.transition_weights, np.full(4, 0.2))
    assert np.array_equal(task.contact_positions, positions)
