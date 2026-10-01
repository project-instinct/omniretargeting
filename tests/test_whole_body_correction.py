"""Independent real-MuJoCo tests for the constrained whole-body correction stage."""

from __future__ import annotations

import copy
from dataclasses import replace

import mujoco
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from omniretargeting.contact import SceneContactTask
from omniretargeting.scene_geometry import SceneQuery
from omniretargeting.surface_contact import (
    SurfaceContactTasks,
    transported_contact_normals,
)
from omniretargeting.surface_tasks import SurfaceBinding
from omniretargeting.trajectory_validation import TrajectoryValidator, local_repair_config
from omniretargeting.whole_body_correction import correct_whole_body_trajectory

WHOLE_BODY_XML = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="root">
      <freejoint name="floating_base"/>
      <!-- A real collision geom keeps the probe path active; it is far from the test scene. -->
      <geom name="root_surface" type="sphere" size=".06" contype="1" conaffinity="1"/>
      <body name="waist" pos="0 0 .65">
        <joint name="waist_pitch" type="hinge" axis="0 1 0" range="-.6 .6"/>
        <geom name="pelvis_surface" type="sphere" pos=".16 0 0" size=".08" contype="0" conaffinity="0"/>
        <body name="shoulder" pos="0 .22 .22">
          <joint name="shoulder_pitch" type="hinge" axis="0 1 0" range="-1.4 1.4"/>
          <geom type="sphere" size=".035" contype="0" conaffinity="0"/>
          <body name="forearm" pos=".24 0 0">
            <joint name="elbow" type="hinge" axis="0 1 0" range="-2.6 .1"/>
            <geom type="capsule" fromto="0 0 0 .22 0 0" size=".03" contype="0" conaffinity="0"/>
            <body name="hand" pos=".22 0 0">
              <geom name="hand_surface" type="sphere" size=".045" contype="0" conaffinity="0"/>
            </body>
          </body>
        </body>
      </body>
      <body name="foot" pos="0 -.22 .08">
        <geom name="foot_surface" type="sphere" size=".045" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _model_and_baseline(frames=1):
    model = mujoco.MjModel.from_xml_string(WHOLE_BODY_XML)
    trajectory = np.tile(model.qpos0, (frames, 1))
    return model, trajectory


def _point(model, qpos, geom_name):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return data.geom_xpos[model.geom(geom_name).id].copy()


def _normal(model, qpos, geom_name):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return data.geom_xmat[model.geom(geom_name).id].reshape(3, 3)[:, 2].copy()


def _task(name, targets, *, sticking=True, modes=None):
    targets = np.asarray(targets, dtype=float)
    frames = len(targets)
    return SceneContactTask(
        name=name,
        tracking_positions=targets.copy(),
        contact_positions=targets.copy(),
        surface_points=targets.copy(),
        surface_normals=np.tile([0.0, 0.0, 1.0], (frames, 1)),
        distances=np.zeros(frames),
        relative_speeds=np.zeros(frames),
        contact_confidence=np.ones(frames),
        contact_mask=np.full(frames, sticking, dtype=bool),
        contact_modes=np.full(frames, "sticking" if sticking else "none")
        if modes is None
        else np.asarray(modes),
        transition_weights=np.ones(frames),
    )


def _surface_task(model, baseline, pelvis_targets, *, frames=1, modes=None):
    bindings = [
        SurfaceBinding(
            "pelvis",
            0,
            (1.0, 0.0, 0.0),
            "pelvis_surface",
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    ]
    targets = np.asarray(pelvis_targets, dtype=float).reshape(frames, 1, 3)
    return SurfaceContactTasks(
        model,
        bindings,
        np.zeros_like(targets),
        targets,
        np.ones((frames, 1), dtype=bool),
        np.tile([0.0, 0.0, 1.0], (frames, 1, 1)),
        {"enabled": True, "weight": 1.0, "sigma": 0.01},
        modes=np.full((frames, 1), "sticking") if modes is None else np.asarray(modes).reshape(frames, 1),
        pose_normal_targets=np.tile([0.0, 0.0, 1.0], (frames, 1, 1)),
        pose_normal_valid=np.ones((frames, 1), dtype=bool),
    )


def _priority():
    return {
        "effectors": {
            "left_hand": {
                "robot_body": "hand",
                "tracking_offset": [0.0, 0.0, 0.0],
                "contact_offsets": [[0.0, 0.0, 0.0]],
            },
            "left_foot": {
                "robot_body": "foot",
                "tracking_offset": [0.0, 0.0, 0.0],
                "contact_offsets": [[0.0, 0.0, 0.0]],
            },
        }
    }


def _config(**overrides):
    config = {
        "enabled": True,
        "iterations": 8,
        "max_joint_correction": 0.35,
        "trust_region": 0.08,
        "contact_regression": 0.001,
        "world_position_budget": 0.002,
        "free_position_budget": 0.02,
        "correction_velocity": 1.0,
        "correction_acceleration": 12.0,
        "correction_jerk": 240.0,
        "surface_weight": 1.0,
        "reference_sigma": 0.18,
        "normal_weight": 0.05,
        "moving_collision_enabled": False,
        "penetration_tolerance": 0.0005,
    }
    config.update(overrides)
    return config


@pytest.fixture
def empty_scene():
    # A real closed mesh keeps SceneQuery on its normal geometry path while
    # remaining far from this local kinematic fixture.
    return SceneQuery(trimesh.creation.box(extents=[.1, .1, .1], transform=np.array([
        [1, 0, 0, 10], [0, 1, 0, 10], [0, 0, 1, 10], [0, 0, 0, 1]
    ])))


def test_surface_correction_preserves_hand_and_root_while_waist_reduces_pelvis_error(empty_scene):
    model, baseline = _model_and_baseline()
    desired = baseline[0].copy()
    desired[model.joint("waist_pitch").qposadr] = 0.18
    pelvis_target = _point(model, desired, "pelvis_surface")
    hand_target = _point(model, baseline[0], "hand_surface")
    foot_target = _point(model, baseline[0], "foot_surface")
    surface = _surface_task(model, baseline, pelvis_target)
    tasks = {
        "left_hand": _task("left_hand", [hand_target]),
        "left_foot": _task("left_foot", [foot_target]),
    }
    before = baseline.copy()
    result, report = correct_whole_body_trajectory(
        model,
        baseline,
        tasks,
        _priority(),
        _config(),
        scene_query=empty_scene,
        surface_contact_tasks=surface,
        framerate=30.0,
    )
    assert report["status"] in {"accepted_updates", "no_feasible_update"}
    np.testing.assert_array_equal(baseline, before)
    np.testing.assert_allclose(result[:, :7], before[:, :7], atol=1e-12)
    hand_error_before = np.linalg.norm(_point(model, before[0], "hand_surface") - hand_target)
    hand_error_after = np.linalg.norm(_point(model, result[0], "hand_surface") - hand_target)
    foot_error = np.linalg.norm(_point(model, result[0], "foot_surface") - foot_target)
    pelvis_error_before = np.linalg.norm(_point(model, before[0], "pelvis_surface") - pelvis_target)
    pelvis_error_after = np.linalg.norm(_point(model, result[0], "pelvis_surface") - pelvis_target)
    assert hand_error_after <= hand_error_before + 0.001
    assert foot_error <= 0.002
    assert pelvis_error_after < pelvis_error_before
    assert report["root_invariant"] is True


def test_sliding_tangent_and_normal_targets_remain_protected(empty_scene):
    model, baseline = _model_and_baseline(frames=4)
    hand = _point(model, baseline[0], "hand_surface")
    hand_targets = np.tile(hand, (4, 1))
    hand_targets[:, 0] += np.linspace(0.0, 0.06, 4)
    pelvis = _point(model, baseline[0], "pelvis_surface")
    pelvis_targets = np.tile(pelvis, (4, 1))
    modes = np.full(4, "sliding")
    surface = _surface_task(model, baseline, pelvis_targets, frames=4, modes=modes)
    tasks = {"left_hand": _task("left_hand", hand_targets, modes=modes)}
    result, _report = correct_whole_body_trajectory(
        model,
        baseline,
        tasks,
        _priority(),
        _config(),
        scene_query=empty_scene,
        surface_contact_tasks=surface,
        framerate=30.0,
    )
    baseline_hand_errors = np.linalg.norm(
        np.array([_point(model, baseline[frame], "hand_surface") for frame in range(4)])
        - hand_targets,
        axis=1,
    )
    for frame in range(4):
        assert np.linalg.norm(_point(model, result[frame], "hand_surface") - hand_targets[frame]) <= baseline_hand_errors[frame] + 0.001
        assert np.linalg.norm(_point(model, result[frame], "pelvis_surface") - pelvis_targets[frame]) <= 0.001
        assert np.dot(_normal(model, result[frame], "pelvis_surface"), [0.0, 0.0, 1.0]) >= 1.0 - 1e-3
    assert np.max(np.linalg.norm(result[:, :7] - baseline[:, :7], axis=1)) <= 1e-12
    assert np.max(np.abs(np.diff(result - baseline, axis=0))) <= 1.0 / 30.0 + 1e-6


def test_disabled_stage_is_exact_noop_and_unreachable_root_surface_is_reported(empty_scene):
    model, baseline = _model_and_baseline()
    root_target = _point(model, baseline[0], "root_surface") + [0.2, 0.0, 0.0]
    surface = SurfaceContactTasks(
        model,
        [SurfaceBinding("root", 0, (1.0, 0.0, 0.0), "root_surface", (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))],
        np.zeros((1, 1, 3)),
        np.asarray(root_target).reshape(1, 1, 3),
        np.ones((1, 1), dtype=bool),
        np.tile([0.0, 0.0, 1.0], (1, 1, 1)),
        {"enabled": True, "weight": 1.0, "sigma": 0.01},
    )
    disabled, disabled_report = correct_whole_body_trajectory(
        model, baseline, {}, _priority(), {"enabled": False}, scene_query=empty_scene,
        surface_contact_tasks=surface,
    )
    np.testing.assert_array_equal(disabled, baseline)
    assert disabled_report["status"] == "disabled"
    result, report = correct_whole_body_trajectory(
        model, baseline, {}, _priority(), _config(), scene_query=empty_scene,
        surface_contact_tasks=surface,
    )
    np.testing.assert_allclose(result, baseline, atol=1e-12)
    assert report["status"] in {"no_feasible_update", "no_active_tasks"}
    assert report["root_invariant"] is True


def test_source_tasks_and_marker_identity_are_immutable(empty_scene):
    model, baseline = _model_and_baseline(frames=3)
    target = np.tile(_point(model, baseline[0], "hand_surface"), (3, 1))
    task = _task("left_hand", target, modes=np.full(3, "sticking"))
    original = copy.deepcopy(task.__dict__)
    _result, _report = correct_whole_body_trajectory(
        model, baseline, {"left_hand": task}, _priority(), _config(),
        scene_query=empty_scene, framerate=30.0,
    )
    for key, value in original.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(task, key), value)
        else:
            assert getattr(task, key) == value


def test_windowed_correction_keeps_fixed_reference_and_composed_budgets(
    empty_scene, monkeypatch,
):
    import omniretargeting.whole_body_correction as correction

    monkeypatch.setattr(correction, "WHOLE_BODY_WINDOW_THRESHOLD", 4)
    monkeypatch.setattr(correction, "WHOLE_BODY_WINDOW_FRAMES", 4)
    model, baseline = _model_and_baseline(frames=10)
    desired = baseline[0].copy()
    desired[model.joint("waist_pitch").qposadr] = 0.1
    target = np.tile(_point(model, desired, "pelvis_surface"), (10, 1))
    surface = _surface_task(model, baseline, target, frames=10)
    result, report = correct_whole_body_trajectory(
        model, baseline, {}, _priority(), _config(iterations=1),
        scene_query=empty_scene, surface_contact_tasks=surface,
    )
    assert report["window_frames"] == 4
    assert len({tuple(item["window"]) for item in report["iterations"]}) == 3
    assert report["accepted_update_count"] == 3
    assert report["final_reasons"] == []
    assert report["root_invariant"]
    np.testing.assert_array_equal(result[:, :7], baseline[:, :7])
    assert report["final_fidelity"]["ok"]


def test_validator_window_keeps_authored_approach_reference(empty_scene):
    model, baseline = _model_and_baseline(frames=5)
    targets = np.tile(_point(model, baseline[0], "hand_surface"), (5, 1))
    task = replace(
        _task("left_hand", targets),
        approach_reference=(np.linspace(0, 1, 5), targets.copy()),
    )
    priority = _priority()
    validator = TrajectoryValidator(
        model, baseline, empty_scene, {"left_hand": task}, priority,
        local_repair_config({"enabled": True}, priority=priority), 30.0,
    )
    with validator.segment(2, 3) as view:
        payload = view.tasks["left_hand"].frame(0)
        assert payload["approach_weight"] == 0.5
        np.testing.assert_array_equal(payload["approach_position"], targets[2])


def test_windowed_correction_checks_cumulative_guard_before_acceptance(
    empty_scene, monkeypatch,
):
    import omniretargeting.whole_body_correction as correction

    monkeypatch.setattr(correction, "WHOLE_BODY_WINDOW_THRESHOLD", 4)
    monkeypatch.setattr(correction, "WHOLE_BODY_WINDOW_FRAMES", 4)
    model, baseline = _model_and_baseline(frames=10)
    desired = baseline[0].copy()
    desired[model.joint("waist_pitch").qposadr] = 0.1
    target = np.tile(_point(model, desired, "pelvis_surface"), (10, 1))
    surface = _surface_task(model, baseline, target, frames=10)
    result, report = correct_whole_body_trajectory(
        model, baseline, {}, _priority(), _config(iterations=1),
        scene_query=empty_scene, surface_contact_tasks=surface,
        candidate_acceptance=lambda candidate: np.array_equal(candidate, baseline),
    )
    np.testing.assert_array_equal(result, baseline)
    assert report["accepted_update_count"] == 0
    assert report["final_reasons"] == []
    assert any(
        "fixed_sequence_budget" in rejection["reasons"]
        for event in report["iterations"]
        for rejection in event.get("rejections", [])
    )


def test_surface_pose_normal_transport_uses_source_face_motion_and_robot_reference_normal():
    model, baseline = _model_and_baseline(frames=2)
    reference_vertices = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    reference_faces = np.asarray([[0, 1, 2]], dtype=int)
    turn = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
    source_triangles = np.asarray(
        [reference_vertices, reference_vertices @ turn.T]
    )[:, None]
    source = {
        "full_body": {
            "reference_vertices": reference_vertices.copy(),
            "faces": reference_faces.copy(),
            "face_ids": np.asarray([0], dtype=int),
            "triangles": source_triangles.copy(),
        }
    }
    binding = SurfaceBinding(
        "pelvis",
        0,
        (1.0, 0.0, 0.0),
        "pelvis_surface",
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    )
    pose_normals, valid = transported_contact_normals(
        model, source, [binding], [0], baseline[0]
    )
    assert valid.tolist() == [[True], [True]]
    np.testing.assert_allclose(pose_normals[0, 0], [1.0, 0.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(pose_normals[1, 0], [0.0, 1.0, 0.0], atol=1e-10)
    assert not np.allclose(pose_normals[1, 0], [0.0, 0.0, 1.0])
    source_before = copy.deepcopy(source)
    task = SurfaceContactTasks(
        model,
        [binding],
        np.zeros((2, 1, 3)),
        np.zeros((2, 1, 3)),
        np.ones((2, 1), dtype=bool),
        np.tile([0.0, 0.0, 1.0], (2, 1, 1)),
        {"enabled": True},
        pose_normal_targets=pose_normals,
        pose_normal_valid=valid,
    )
    rows = list(task.normal_rows(baseline[0], 0, np.arange(model.nv)))
    assert len(rows) == 1
    assert np.linalg.norm(rows[0][1]) < 1e-10
    np.testing.assert_array_equal(
        source["full_body"]["reference_vertices"],
        source_before["full_body"]["reference_vertices"],
    )
    np.testing.assert_array_equal(
        source["full_body"]["triangles"], source_before["full_body"]["triangles"]
    )
