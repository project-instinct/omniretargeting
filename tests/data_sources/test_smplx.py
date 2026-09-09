"""Tests for the SMPL-X data-source adapter."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting.data_sources.smplx import (
    DEFAULT_SMPLX_TARGET_NAMES,
    SmplxDataSource,
    compute_world_joint_orientations,
    create_smplx_data_source,
    validate_smplx_trajectory,
)


def test_validate_smplx_trajectory_valid():
    trajectory = np.random.randn(100, 22, 3)
    assert validate_smplx_trajectory(trajectory) is True


def test_validate_smplx_trajectory_invalid_shape():
    trajectory = np.random.randn(100, 22)
    assert validate_smplx_trajectory(trajectory) is False


def test_validate_smplx_trajectory_nan_values():
    trajectory = np.random.randn(100, 22, 3)
    trajectory[10, 5, 2] = np.nan
    assert validate_smplx_trajectory(trajectory) is False


def test_validate_smplx_trajectory_inf_values():
    trajectory = np.random.randn(100, 22, 3)
    trajectory[10, 5, 2] = np.inf
    assert validate_smplx_trajectory(trajectory) is False


def test_load_npy_generates_default_names_for_extra_joints(tmp_path):
    trajectory = np.random.randn(4, 24, 3)
    motion_file = tmp_path / "motion.npy"
    np.save(motion_file, trajectory)

    motion = SmplxDataSource(motion_file=motion_file).load()

    assert motion.positions.shape == trajectory.shape
    assert len(motion.target_names) == trajectory.shape[1]
    assert motion.target_names[:22] == DEFAULT_SMPLX_TARGET_NAMES
    assert motion.target_names[22:] == ["SMPLX_Joint_22", "SMPLX_Joint_23"]
    assert set(motion.metadata["end_effector_markers"]) == {
        "left_hand",
        "right_hand",
        "left_foot",
        "right_foot",
    }


def test_compute_world_joint_orientations():
    num_frames = 10
    num_joints = 22
    global_orient = np.random.randn(num_frames, 3) * 0.1
    full_pose = np.random.randn(num_frames, num_joints, 3) * 0.1
    parents = np.arange(-1, num_joints - 1)

    orientations = compute_world_joint_orientations(
        global_orient,
        full_pose,
        parents,
        num_body_joints=num_joints,
    )

    assert orientations.shape == (num_frames, num_joints, 4)
    norms = np.linalg.norm(orientations, axis=2)
    assert np.allclose(norms, 1.0, atol=1e-6)
    assert np.isfinite(orientations).all()

    for frame_idx in range(num_frames):
        root_quat = orientations[frame_idx, 0]
        expected_quat = Rotation.from_rotvec(global_orient[frame_idx]).as_quat(scalar_first=True)
        assert np.allclose(root_quat, expected_quat, atol=1e-6) or np.allclose(
            root_quat,
            -expected_quat,
            atol=1e-6,
        )


def _raw_timeline(frame_ids):
    count = len(frame_ids)
    return {
        "frame_ids": np.asarray(frame_ids),
        "trans": np.arange(count * 3, dtype=np.float32).reshape(count, 3),
        "root_orient": np.zeros((count, 3), dtype=np.float32),
        "pose_body": np.zeros((count, 63), dtype=np.float32),
        "pose_hand": np.zeros((count, 90), dtype=np.float64),
    }


def test_provenance_ids_preserve_published_rows_and_original_id_mapping(tmp_path):
    arrays = _raw_timeline(["0000", "0001", "0002", "0004", "0005"])
    motion_path = tmp_path / "published.npz"
    np.savez(motion_path, **arrays, mocap_frame_rate=25.0)
    source = create_smplx_data_source(
        motion_path, {"adapter_options": {"frame_id_semantics": "provenance"}}
    )
    with np.load(motion_path) as motion:
        fps = source._detect_framerate(motion)
        trans, root, body, hand, metadata = source._prepare_raw_timeline(motion, fps)

    assert fps == 25.0
    for actual, key in zip((trans, root, body, hand), ("trans", "root_orient", "pose_body", "pose_hand")):
        np.testing.assert_array_equal(actual, arrays[key], strict=True)
    assert metadata["frame_id_semantics"] == "provenance"
    np.testing.assert_array_equal(metadata["original_frame_ids"], arrays["frame_ids"], strict=True)
    assert "timestamps" not in metadata
    assert len(trans) == 5


def test_default_timeline_still_restores_short_missing_frames(tmp_path):
    arrays = _raw_timeline(["0070", "0072"])
    source = create_smplx_data_source(tmp_path / "source.npz")
    trans, root, body, hand, metadata = source._prepare_raw_timeline(arrays, 30.0)
    assert len(trans) == len(root) == len(body) == len(hand) == 3
    np.testing.assert_array_equal(trans[[0, 2]], arrays["trans"])
    np.testing.assert_allclose(trans[1], (arrays["trans"][0] + arrays["trans"][1]) / 2)
    np.testing.assert_array_equal(metadata["original_frame_ids"], arrays["frame_ids"])
    assert metadata["frame_id_semantics"] == "timeline"


@pytest.mark.parametrize("fps", [None, 0.0, -1.0, np.nan, np.inf])
def test_provenance_ids_require_published_framerate(tmp_path, fps):
    source = SmplxDataSource(tmp_path / "source.npz", frame_id_semantics="provenance")
    with pytest.raises(ValueError, match="finite, positive published framerate"):
        source._prepare_raw_timeline(_raw_timeline(["0000", "0002"]), fps)


def test_provenance_ids_reject_sparse_person_timeline(tmp_path):
    arrays = _raw_timeline(["0000", "0002"])
    arrays["person_timeline_sparse"] = np.bool_(True)
    source = SmplxDataSource(tmp_path / "source.npz", frame_id_semantics="provenance")
    with pytest.raises(ValueError, match="sparse person timeline"):
        source._prepare_raw_timeline(arrays, 30.0)


def test_provenance_ids_reject_misaligned_mapping(tmp_path):
    arrays = _raw_timeline(["0000", "0001"])
    arrays["frame_ids"] = np.asarray(["0000"])
    source = SmplxDataSource(tmp_path / "source.npz", frame_id_semantics="provenance")
    with pytest.raises(ValueError, match="one ID per published motion row"):
        source._prepare_raw_timeline(arrays, 30.0)


def test_invalid_frame_id_semantics_is_explicit_error(tmp_path):
    with pytest.raises(ValueError, match="frame_id_semantics"):
        create_smplx_data_source(
            tmp_path / "source.npz", {"adapter_options": {"frame_id_semantics": "guessed"}}
        )
