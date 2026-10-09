"""Tests for the OMOMO data-source adapter."""

import json
import numpy as np
import joblib
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from omniretargeting.data_sources import omomo
from omniretargeting.data_sources.registry import (
    create_data_source,
    get_source_extensions,
)
from omniretargeting.robot_config import load_robot_config

def test_home_relative_paths_are_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    sequence_file = tmp_path / "Datasets" / "OMOMO" / "data" / "sequence.p"
    data_root = tmp_path / "Datasets" / "OMOMO"
    model_directory = tmp_path / "Datasets" / "smplx"
    sequence_file.parent.mkdir(parents=True)
    sequence_file.touch()
    object_mesh_path = (
        data_root
        / "data"
        / "captured_objects"
        / "floorlamp_cleaned_simplified.obj"
    )
    object_mesh_path.parent.mkdir(parents=True)
    object_mesh_path.touch()
    model_directory.mkdir(parents=True)

    monkeypatch.setattr(
        omomo.joblib,
        "load",
        lambda path: [{"seq_name": "sub17_floorlamp_023"}],
    )
    monkeypatch.setattr(
        omomo.trimesh,
        "load",
        lambda path, force: type("Mesh", (), {"vertices": np.zeros((1, 3))})(),
    )

    source = omomo.OmomoDataSource(
        sequence_file="~/Datasets/OMOMO/data/sequence.p",
        sequence_index=0,
        data_root="~/Datasets/OMOMO",
        model_directory="~/Datasets/smplx",
    )

    assert source.sequence_file == sequence_file
    assert source.data_root == data_root
    assert source.model_directory == str(model_directory)
    assert source._resolve_model_directory() == str(model_directory)


@pytest.fixture
def omomo_sequence(tmp_path):
    root = tmp_path / "data"
    meshes = root / "captured_objects"
    meshes.mkdir(parents=True)
    trimesh.creation.box().export(meshes / "box_cleaned_simplified.obj")
    offsets = np.zeros((24, 3))
    offsets[1] = [1, 0, 0]
    sequence = {
        "seq_name": "sub01_box_000",
        "rest_offsets": offsets,
        "root_orient": np.tile([0, 0, np.pi / 2], (4, 1)),
        "pose_body": np.zeros((4, 63)),
        "trans": np.tile([1, 2, 3], (4, 1)),
        "obj_scale": np.array([1, 1.2, 1.1, 1]),
        "obj_rot": np.tile(np.eye(3), (4, 1, 1)),
        "obj_trans": np.tile([4, 5, 6], (4, 1))[:, :, None],
        "obj_com_pos": np.tile([4, 5, 6], (4, 1)),
    }
    file = root / "sequence.p"
    joblib.dump({0: sequence}, file)
    options = {
        "data_root": str(tmp_path),
        "body_position_mode": "rest_offsets",
        "n_object_samples": 8,
    }
    return file, sequence, options


def test_rest_offset_fk_and_explicit_rigid_scale_migration(omomo_sequence):
    file, sequence, options = omomo_sequence
    legacy = create_data_source("omomo", file, options).load()
    assert legacy.scene is None and legacy.entity_trajectories is None
    np.testing.assert_allclose(legacy.positions[:, 0], sequence["trans"])
    np.testing.assert_allclose(legacy.positions[:, 1], np.tile([1, 3, 3], (4, 1)))
    rigid = create_data_source(
        "omomo", file, options, {"object_scale_mode": "first_frame", "framerate": 60}
    ).load()
    assert rigid.framerate == 60
    np.testing.assert_array_equal(rigid.metadata["object_scales"], np.ones(4))
    np.testing.assert_array_equal(
        rigid.metadata["recorded_object_scales"], sequence["obj_scale"]
    )
    for frame in rigid.iter_frames():
        points = frame.scene.pose("box", frame.entity_poses).to_world(
            frame.scene.entities["box"].local_samples
        )
        np.testing.assert_allclose(points, frame.object_points, atol=1e-6)
    assert ".p" in get_source_extensions("omomo")


def test_loaded_omomo_profile_adapter_options_and_runtime_precedence(
    omomo_sequence, tmp_path
):
    file, sequence, options = omomo_sequence
    profile_path = tmp_path / "robot.json"
    profile_path.write_text(
        json.dumps(
            {
                "source": [
                    {
                        "name": "omomo",
                        "type": "omomo",
                        "target_mapping": {"Pelvis": "pelvis"},
                        "data_root": "/unused/source-field",
                        "sequence_index": 99,
                        "framerate": 1,
                        "adapter_options": {
                            **options,
                            "sequence_index": 0,
                            "framerate": 15,
                            "object_scale_mode": "first_frame",
                        },
                    }
                ],
            }
        )
    )
    profile = load_robot_config(profile_path)
    source = create_data_source("omomo", file, profile["selected_source"])
    assert source.data_root == tmp_path
    assert source.sequence_index == 0 and source.framerate == 15
    assert source.body_position_mode == "rest_offsets"
    assert source.object_scale_mode == "first_frame"
    assert source.load().scene is not None
    overridden = create_data_source(
        "omomo",
        file,
        profile["selected_source"],
        {"framerate": 60, "object_scale_mode": "per_frame"},
    )
    assert overridden.framerate == 60 and overridden.sequence_index == 0
    assert overridden.body_position_mode == "rest_offsets"
    assert overridden.object_scale_mode == "per_frame"
    assert overridden.load().scene is None


@pytest.mark.parametrize("scale_mode", ["first_frame", "per_frame"])
def test_resampled_rotating_object_samples_follow_rigid_pose(
    omomo_sequence, scale_mode
):
    file, sequence, options = omomo_sequence
    sequence["obj_rot"] = Rotation.from_euler(
        "z", np.array([0, 90, 180, 270])[:, None], degrees=True
    ).as_matrix()
    joblib.dump({0: sequence}, file)
    motion = create_data_source(
        "omomo", file, options, {"object_scale_mode": scale_mode, "framerate": 1}
    ).load()
    resampled = motion.resample(2)
    chord_midpoint = (motion.object_points[0] + motion.object_points[1]) / 2
    if scale_mode == "first_frame":
        for frame in resampled.iter_frames():
            expected = frame.entity_poses["box"].to_world(
                frame.scene.entities["box"].local_samples
            )
            np.testing.assert_allclose(frame.object_points, expected, atol=1e-7)
        assert np.max(np.abs(resampled.object_points[1] - chord_midpoint)) > 0.1
    else:
        assert resampled.scene is None
        np.testing.assert_allclose(resampled.object_points[1], chord_midpoint)
