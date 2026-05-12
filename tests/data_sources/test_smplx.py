"""Tests for the SMPL-X data-source adapter."""

import numpy as np
from scipy.spatial.transform import Rotation

from omniretargeting.data_sources.smplx import compute_world_joint_orientations, validate_smplx_trajectory


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
