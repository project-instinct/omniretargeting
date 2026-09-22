"""Smoke coverage for the OMOMO adapter's source and object contracts."""

import copy
from types import SimpleNamespace

import joblib
import mujoco
import numpy as np
import trimesh

from omniretargeting import OmniRetargeter, core, whole_body_correction
from omniretargeting._core.motion import FramewiseMotionMixin
from omniretargeting.data_sources.omomo import OmomoDataSource
from omniretargeting.data_sources.registry import (
    create_data_source,
    get_source_extensions,
)
from tests._contact_fixtures import _task


def _write_sequence(tmp_path):
    data_root = tmp_path / "omomo"
    object_dir = data_root / "data" / "captured_objects"
    object_dir.mkdir(parents=True)
    mesh = trimesh.creation.box(extents=[0.2, 0.3, 0.4])
    mesh_path = object_dir / "cube_cleaned_simplified.obj"
    mesh.export(mesh_path)

    frames = 3
    sequence = {
        "seq_name": "seq_cube_000",
        "gender": "neutral",
        "trans": np.array(
            [[0.1, 0.2, 0.3], [0.2, 0.2, 0.3], [0.3, 0.2, 0.3]],
            dtype=np.float32,
        ),
        "root_orient": np.zeros((frames, 3), dtype=np.float32),
        "pose_body": np.zeros((frames, 63), dtype=np.float32),
        "pose_hand": np.zeros((frames, 90), dtype=np.float32),
        "betas": np.zeros(10, dtype=np.float32),
        "obj_scale": np.array([1.0, 2.0, 1.5], dtype=np.float32),
        "obj_rot": np.tile(np.eye(3, dtype=np.float32), (frames, 1, 1)),
        "obj_trans": np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
            dtype=np.float32,
        ),
        "obj_com_pos": np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
            dtype=np.float32,
        ),
    }
    sequence_path = tmp_path / "sample.joblib"
    joblib.dump([sequence], sequence_path)
    return sequence_path, data_root, sequence, mesh


def test_omomo_registry_accepts_both_dataset_extensions():
    extensions = get_source_extensions("omomo")
    assert ".p" in extensions
    assert ".joblib" in extensions


def test_omomo_factory_merges_nested_adapter_options_with_runtime_precedence(
    monkeypatch, tmp_path
):
    captured = {}

    class FakeOmomoDataSource:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        "omniretargeting.data_sources.omomo.OmomoDataSource",
        FakeOmomoDataSource,
    )
    source_config = {
        "sequence_index": 3,
        "data_root": "source-root",
        "n_object_samples": 4,
        "model_directory": "source-model",
        "use_smplx_base_pose": True,
        "framerate": 24.0,
        "adapter_options": {
            "sequence_index": 2,
            "data_root": "adapter-root",
            "n_object_samples": 8,
            "model_directory": "adapter-model",
            "use_smplx_base_pose": False,
            "framerate": 48.0,
        },
    }
    runtime_options = {
        "sequence_index": 7,
        "n_object_samples": 12,
        "model_directory": "runtime-model",
        "framerate": 60.0,
    }
    source = create_data_source(
        "omomo", tmp_path / "sequence.joblib", source_config, runtime_options
    )

    assert isinstance(source, FakeOmomoDataSource)
    assert captured["sequence_index"] == 7
    assert captured["data_root"] == "adapter-root"
    assert captured["n_object_samples"] == 12
    assert captured["model_directory"] == "runtime-model"
    assert captured["use_smplx_base_pose"] is False
    assert captured["framerate"] == 60.0


def test_omomo_reuses_smplx_raw_path_and_preserves_object_metric_data(
    monkeypatch, tmp_path
):
    sequence_path, data_root, sequence, mesh = _write_sequence(tmp_path)
    positions = np.zeros((3, 22, 3), dtype=np.float32)
    positions[:, 0] = sequence["trans"] + [0.0, 0.0, 0.5]
    orientations = np.zeros((3, 22, 4), dtype=np.float32)
    orientations[..., 0] = 1.0
    root_orient = sequence["root_orient"].copy()
    captured = {}
    surface_metadata = {"kind": "mock_surface", "frames": 3}
    sole_metadata = {"foot_sole_centres": {"left": np.zeros((3, 3))}}
    model_directory = tmp_path / "mock_smplx"
    model_directory.mkdir()

    def fake_raw_loader(adapter, raw, framerate):
        captured["trans"] = np.array(raw["trans"], copy=True)
        captured["pose_body_shape"] = np.asarray(raw["pose_body"]).shape
        captured["framerate"] = framerate
        return (
            positions.copy(),
            orientations.copy(),
            root_orient.copy(),
            sequence["trans"].copy(),
            framerate,
            {"body_surface_samples": copy.deepcopy(surface_metadata)},
        )

    monkeypatch.setattr(
        "omniretargeting.data_sources.smplx.SmplxDataSource._load_raw_npz",
        fake_raw_loader,
    )
    monkeypatch.setattr(
        "omniretargeting.data_sources.smplx.SmplxDataSource._foot_sole_metadata",
        lambda _adapter, _orientations: copy.deepcopy(sole_metadata),
    )

    source = create_data_source(
        "omomo",
        sequence_path,
        {
            "sequence_index": 0,
            "data_root": str(data_root),
            "model_directory": str(model_directory),
            "framerate": 24.0,
        },
    )
    assert isinstance(source, OmomoDataSource)
    motion = source.load()

    # OMOMO's model translation reaches the shared SMPL-X adapter unchanged;
    # no second trans2joint offset is added before the raw FK call.
    np.testing.assert_array_equal(captured["trans"], sequence["trans"])
    assert captured["pose_body_shape"] == (3, 63)
    assert captured["framerate"] == 24.0
    np.testing.assert_array_equal(motion.positions[:, 0], positions[:, 0])
    assert motion.metadata["body_surface_samples"] == surface_metadata
    np.testing.assert_array_equal(
        motion.metadata["foot_sole_centres"]["left"],
        sole_metadata["foot_sole_centres"]["left"],
    )
    assert motion.metadata["human_translation_convention"].startswith(
        "SMPL-X model translation"
    )

    object_points = np.asarray(motion.object_points)
    local_vertices = np.asarray(mesh.vertices)
    expected = np.asarray(
        [
            local_vertices * scale + translation
            for scale, translation in zip(
                sequence["obj_scale"], sequence["obj_trans"]
            )
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(object_points, expected, atol=1e-6)
    np.testing.assert_array_equal(
        motion.metadata["object_scales"], sequence["obj_scale"]
    )
    np.testing.assert_array_equal(
        motion.metadata["object_translations"], sequence["obj_trans"]
    )


def test_object_contact_merging_keeps_foot_baseline_before_hand_overlay(monkeypatch):
    frames = 3
    model = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody><body name="root"><freejoint/>
        <geom type="sphere" size=".03" mass="1"/></body></worldbody></mujoco>"""
    )
    terrain = trimesh.creation.box(extents=[2.0, 2.0, 0.1])
    terrain.apply_translation([0.0, 0.0, -1.0])
    object_mesh = trimesh.creation.box(extents=[0.2, 0.2, 0.1])
    object_translations = np.column_stack([
        np.zeros(frames), np.zeros(frames), np.zeros(frames)
    ])
    hand_points = np.tile([0.0, 0.0, 0.05], (frames, 1))
    markers = {
        "left_hand": {
            "tracking_positions": hand_points.copy(),
            "contact_samples": hand_points[:, None, :].copy(),
            "up_directions": np.tile([0.0, 0.0, 1.0], (frames, 1)),
        },
        "left_foot": {"tracking_positions": np.tile([0.0, 0.0, 0.5], (frames, 1))},
        "right_foot": {"tracking_positions": np.tile([0.2, 0.0, 0.5], (frames, 1))},
    }
    scene_objects = [{
        "name": "moving_box",
        "mesh": object_mesh,
        "translations": object_translations,
        "rotations": np.tile(np.eye(3), (frames, 1, 1)),
        "scales": np.ones(frames),
    }]
    processed = np.zeros((frames, 1, 3), dtype=float)
    calls = []

    retargeter = OmniRetargeter.__new__(OmniRetargeter)
    retargeter.robot_model = model
    retargeter.robot_data = mujoco.MjData(model)
    retargeter.terrain_mesh = terrain
    retargeter.retargeting_config = {
        "self_collision_enabled": False,
        "end_effector_priority": {
            "enabled": True,
            "world_tracking_enabled": True,
            "preserve_noncontact_baseline": True,
            "contact_detection": {"min_contact_frames": 1},
        },
    }
    retargeter.robot_height = 1.0
    retargeter.valid_joint_mapping = {}
    retargeter.smplx_joint_names = ["Pelvis"]
    retargeter._process_smplx_trajectory = lambda trajectory, scale: processed.copy()
    retargeter._visualize_trajectory = lambda *_args, **_kwargs: None
    retargeter._clamp_final_joint_safety_limits = lambda trajectory: trajectory
    retargeter._apply_trajectory_smoothing = lambda trajectory, **_kwargs: trajectory
    retargeter._validation_foot_references = lambda trajectory, tasks, priority: (
        tasks, priority, []
    )
    retargeter._report_final_motion_quality = lambda *_args, **_kwargs: None

    def fake_perform(_trajectory, _terrain, **kwargs):
        calls.append(kwargs)
        return np.full((frames, model.nq), len(calls), dtype=float)

    retargeter._perform_retargeting = fake_perform
    whole_calls = []

    def fake_whole_body(model_arg, candidate, tasks, priority, config, **kwargs):
        whole_calls.append({
            "model": model_arg,
            "candidate": np.array(candidate, copy=True),
            "tasks": tasks,
            "priority": priority,
            "config": config,
            "kwargs": kwargs,
        })
        return candidate, {
            "status": "disabled",
            "moving_collision": {"status": "disabled"},
            "root_invariant": True,
            "accepted_update_count": 0,
        }

    monkeypatch.setattr(
        whole_body_correction,
        "correct_whole_body_trajectory",
        fake_whole_body,
    )
    monkeypatch.setattr(
        core,
        "select_safe_postprocessing",
        lambda _model, baseline, candidate, *_args, **_kwargs: (
            candidate, {}, {"status": "mocked"}
        ),
    )
    monkeypatch.setattr(
        core,
        "repair_local_collisions",
        lambda _model, candidate, *_args, **_kwargs: (
            candidate,
            {"status": "mocked", "events": [], "unresolved": [], "query_points": 0},
        ),
    )

    _scale, result = retargeter.retarget_motion(
        processed,
        framerate=30.0,
        visualize_trajectory=False,
        end_effector_markers=markers,
        scene_objects=scene_objects,
    )

    assert len(calls) == 2
    assert set(calls[0]["end_effector_tasks"]) == {"left_foot", "right_foot"}
    assert calls[1]["overlay_contact_names"] == {"left_hand"}
    assert calls[0]["apply_surface_contacts"] is False
    assert "apply_surface_contacts" not in calls[1]
    assert np.array_equal(calls[1]["reference_trajectory"], np.full((frames, model.nq), 1.0))
    overlay = calls[1]["end_effector_tasks"]
    assert set(overlay) == {"left_hand", "left_foot", "right_foot"}
    assert overlay["left_hand"].contact_mask.all()
    assert np.all(overlay["left_hand"].environment_ids == 1)
    assert len(whole_calls) == 1
    whole = whole_calls[0]
    np.testing.assert_array_equal(
        whole["candidate"], np.full((frames, model.nq), 2.0)
    )
    assert set(whole["tasks"]) == {"left_hand", "left_foot", "right_foot"}
    assert whole["tasks"]["left_hand"].contact_mask.all()
    assert np.all(whole["tasks"]["left_hand"].environment_ids == 1)
    assert whole["kwargs"]["scene_objects"][0] is scene_objects[0]
    np.testing.assert_array_equal(result, np.full((frames, model.nq), 2.0))


def test_perform_retargeting_defers_surface_tasks_until_overlay(monkeypatch):
    """The feet baseline must not attach globally enabled surface rows."""
    frames = 2
    model = mujoco.MjModel.from_xml_string(
        """<mujoco><worldbody><body name="root"><freejoint/>
        <geom type="sphere" size=".03" mass="1"/></body></worldbody></mujoco>"""
    )
    driver = FramewiseMotionMixin()
    driver.robot_model = model
    driver.robot_data = mujoco.MjData(model)
    driver.robot_height = 1.0
    driver.retargeting_config = {
        "self_collision_enabled": False,
        "surface_contact": {"enabled": True},
        "end_effector_priority": {"enabled": True},
    }
    driver.valid_joint_mapping = {"Pelvis": "root"}
    driver.valid_joint_names = ["Pelvis"]
    driver.mapped_joint_indices = [0]
    driver.smplx_joint_indices = {"Pelvis": 0}
    driver.smplx_joint_names = ["Pelvis"]
    driver.base_orientation_config = {"root_position": "Pelvis"}
    driver._resolve_source_root_index = lambda: 0
    driver._estimate_base_orientation_from_joints = lambda *_args: None
    driver._source_framerate = 30.0
    driver._surface_contacts_built = False
    driver.last_surface_contact_tasks = None
    driver.last_surface_contact_diagnostics = {"status": "disabled"}
    driver.last_motion_solve_reports = []

    observed_surface_tasks = []
    solver_surface_tasks = []

    class FakeRetargeter:
        def __init__(self, *_args, **_kwargs):
            self.robot_reference_qpos = model.qpos0.copy()
            self.surface_contact_tasks = None
            self.last_frame_solve = None

        def retarget_frame(self, _human_joints, q_init, **_kwargs):
            solver_surface_tasks.append(self.surface_contact_tasks)
            return np.asarray(q_init, dtype=float).copy()

    def fake_surface_builder(*_args, **_kwargs):
        task = SimpleNamespace(targets=np.zeros((frames, 1, 3)))
        observed_surface_tasks.append(task)
        return task, {"status": "built", "contact_regions": [],
                      "contact_samples": 0, "active_frame_points": 0}

    monkeypatch.setattr(
        "omniretargeting.retargeting.GenericInteractionRetargeter",
        FakeRetargeter,
    )
    monkeypatch.setattr(
        "omniretargeting.surface_contact.build_surface_contact_tasks",
        fake_surface_builder,
    )
    processed = np.zeros((frames, 1, 3))
    baseline_task = _task(
        np.zeros((frames, 3)), np.zeros(frames, dtype=bool), ["none"] * frames
    )
    overlay_task = _task(
        np.zeros((frames, 3)), np.ones(frames, dtype=bool), ["sticking"] * frames
    )

    driver._perform_retargeting(
        processed, trimesh.creation.box(),
        end_effector_tasks={"left_foot": baseline_task},
        apply_surface_contacts=False,
    )
    assert observed_surface_tasks == []
    assert solver_surface_tasks == [None] * frames
    assert driver.last_surface_contact_tasks is None
    assert driver.last_motion_solve_reports[-1]["surface_contact_enabled"] is False

    driver._perform_retargeting(
        processed, trimesh.creation.box(),
        end_effector_tasks={"left_hand": overlay_task},
        reference_trajectory=np.tile(model.qpos0, (frames, 1)),
        overlay_contact_names={"left_hand"},
    )
    assert len(observed_surface_tasks) == 1
    assert solver_surface_tasks[-frames:] == [observed_surface_tasks[0]] * frames
    assert driver.last_surface_contact_tasks is observed_surface_tasks[0]
    assert driver.last_motion_solve_reports[-1]["surface_contact_enabled"] is True
