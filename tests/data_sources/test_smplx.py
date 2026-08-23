"""Tests for the SMPL-X data-source adapter."""

import numpy as np
from scipy.spatial.transform import Rotation

from omniretargeting.data_sources.smplx import (
    DEFAULT_SMPLX_TARGET_NAMES,
    SmplxDataSource,
    compute_world_joint_orientations,
    smplx_joint_quaternions_to_body_quaternions,
    smplx_root_rotvecs_to_body_quaternions,
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


def test_smplx_root_rotation_is_converted_at_adapter_boundary():
    root_rotvecs = np.zeros((2, 3), dtype=float)
    quaternions = smplx_root_rotvecs_to_body_quaternions(root_rotvecs)

    assert quaternions.shape == (2, 4)
    np.testing.assert_allclose(np.linalg.norm(quaternions, axis=1), 1.0)
    # An identity SMPL-X global rotation still needs the canonical T-pose to
    # body-frame correction, so passing the raw identity downstream is wrong.
    assert not np.allclose(quaternions[0], [1.0, 0.0, 0.0, 0.0])


def test_smplx_joint_frame_uses_the_same_body_axis_conversion_as_root():
    joint_quaternions = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    )
    converted = smplx_joint_quaternions_to_body_quaternions(
        joint_quaternions
    )
    expected = smplx_root_rotvecs_to_body_quaternions(np.zeros((2, 3)))

    np.testing.assert_allclose(converted, expected)
