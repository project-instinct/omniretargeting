"""Basic tests for omniretargeting package."""

import numpy as np
import pytest
from unittest.mock import Mock, patch
from scipy.spatial.transform import Rotation

from omniretargeting.main import direct_source_config
from omniretargeting.utils import validate_smplx_trajectory, compute_world_joint_orientations


class TestUtils:
    """Test utility functions."""

    def test_validate_smplx_trajectory_valid(self):
        """Test validation of valid SMPLX trajectory."""
        trajectory = np.random.randn(100, 22, 3)
        assert validate_smplx_trajectory(trajectory) is True

    def test_validate_smplx_trajectory_invalid_shape(self):
        """Test validation of invalid trajectory shape."""
        trajectory = np.random.randn(100, 22)  # Missing coordinate dimension
        assert validate_smplx_trajectory(trajectory) is False

    def test_validate_smplx_trajectory_nan_values(self):
        """Test validation with NaN values."""
        trajectory = np.random.randn(100, 22, 3)
        trajectory[10, 5, 2] = np.nan
        assert validate_smplx_trajectory(trajectory) is False

    def test_validate_smplx_trajectory_inf_values(self):
        """Test validation with infinite values."""
        trajectory = np.random.randn(100, 22, 3)
        trajectory[10, 5, 2] = np.inf
        assert validate_smplx_trajectory(trajectory) is False

    @pytest.mark.parametrize(
        ("filename", "source_type"),
        [("vault.pkl", "parc_ms"), ("walk.pickle", "parc_ms"), ("amass.npz", "smplx")],
    )
    def test_direct_source_config_infers_common_formats(
        self, tmp_path, filename, source_type
    ):
        motion_path = tmp_path / filename
        config = direct_source_config(motion_path)
        assert config == {"type": source_type, "motion": str(motion_path.resolve())}

    def test_direct_source_config_accepts_explicit_ambiguous_adapter(self, tmp_path):
        motion_path = tmp_path / "walk.bvh"
        config = direct_source_config(motion_path, "lafan1")
        assert config == {"type": "lafan1", "motion": str(motion_path.resolve())}

    def test_direct_source_config_rejects_ambiguous_format(self, tmp_path):
        with pytest.raises(ValueError, match="--source-type"):
            direct_source_config(tmp_path / "walk.bvh")

    def test_compute_world_joint_orientations(self):
        """Test computation of world-frame joint orientations."""
        num_frames = 10
        num_joints = 22
        
        # Create synthetic SMPLX pose data
        # Root orientation (axis-angle)
        global_orient = np.random.randn(num_frames, 3) * 0.1
        
        # Full pose (axis-angle for all joints)
        full_pose = np.random.randn(num_frames, num_joints, 3) * 0.1
        
        # Simple parent structure (linear chain for testing)
        parents = np.arange(-1, num_joints - 1)
        
        # Compute orientations
        orientations = compute_world_joint_orientations(
            global_orient, full_pose, parents, num_body_joints=num_joints
        )
        
        # Verify output shape
        assert orientations.shape == (num_frames, num_joints, 4), \
            f"Expected shape ({num_frames}, {num_joints}, 4), got {orientations.shape}"
        
        # Verify quaternions are normalized
        norms = np.linalg.norm(orientations, axis=2)
        assert np.allclose(norms, 1.0, atol=1e-6), \
            "Quaternions should be normalized"
        
        # Verify no NaN or inf values
        assert np.isfinite(orientations).all(), \
            "Orientations contain NaN or inf values"
        
        # Verify root orientation matches global_orient
        for t in range(num_frames):
            root_quat = orientations[t, 0]
            expected_rot = Rotation.from_rotvec(global_orient[t])
            expected_quat = expected_rot.as_quat(scalar_first=True)
            
            # Quaternions q and -q represent the same rotation
            assert np.allclose(root_quat, expected_quat, atol=1e-6) or \
                   np.allclose(root_quat, -expected_quat, atol=1e-6), \
                f"Root orientation mismatch at frame {t}"


class TestOmniRetargeter:
    """Test OmniRetargeter class (mocked for testing without real files)."""

    @patch('omniretargeting.core.yourdfpy')
    @patch('omniretargeting.core.mujoco')
    @patch('omniretargeting.core.trimesh')
    def test_initialization(self, mock_trimesh, mock_mujoco, mock_yourdfpy):
        """Test OmniRetargeter initialization with mocked dependencies."""
        from omniretargeting import OmniRetargeter

        # Setup mocks
        mock_urdf = Mock()
        mock_yourdfpy.URDF.load.return_value = mock_urdf

        mock_model = Mock()
        mock_model.nbody = 5
        mock_model.ngeom = 0
        mock_model.njnt = 3
        mock_model.nq = 29  # Total qpos dimension (7 floating base + 22 joints)
        mock_model.nv = 29  # Total qvel dimension
        mock_data = Mock()
        mock_mujoco.MjModel.from_xml_path.return_value = mock_model
        mock_mujoco.MjData.return_value = mock_data
        mock_mujoco.mj_resetData = Mock()
        mock_mujoco.mj_forward = Mock()
        body_names = ["world", "torso_link", "left_hip_yaw_link", "left_hip_link", "right_hip_link"]
        mock_mujoco.mjtObj.mjOBJ_BODY = 1
        mock_mujoco.mj_id2name.side_effect = lambda model, obj_type, i: body_names[i]

        mock_mesh = Mock()
        mock_trimesh.load.return_value = mock_mesh

        # Test initialization - use correct SMPLX joint names
        joint_mapping = {"Pelvis": "torso_link", "L_Hip": "left_hip_yaw_link"}
        retargeter = OmniRetargeter(
            robot_urdf_path="dummy.urdf",
            terrain_mesh_path="dummy.obj",
            joint_mapping=joint_mapping,
            robot_height=1.6
        )

        assert retargeter.joint_mapping == joint_mapping
        assert retargeter.robot_height == 1.6

    def test_joint_mapping_validation(self):
        """Test joint mapping validation."""
        from omniretargeting import OmniRetargeter

        # Create a minimal retargeter instance for testing
        with patch('omniretargeting.core.yourdfpy'), \
             patch('omniretargeting.core.mujoco') as mock_mujoco, \
             patch('omniretargeting.core.trimesh'):

            # Setup mock robot model
            mock_model = Mock()
            mock_model.nbody = 5
            mock_model.ngeom = 0
            mock_model.njnt = 3
            mock_model.nq = 29  # Total qpos dimension
            mock_model.nv = 29  # Total qvel dimension
            mock_model.joint.side_effect = lambda i: Mock(name=f"joint_{i}")
            mock_mujoco.MjModel.from_xml_path.return_value = mock_model
            mock_mujoco.MjData.return_value = Mock()
            mock_mujoco.mj_resetData = Mock()
            mock_mujoco.mj_forward = Mock()
            body_names = ["world", "torso_link", "left_hip_yaw_link", "left_hip_link", "right_hip_link"]
            mock_mujoco.mjtObj.mjOBJ_BODY = 1
            mock_mujoco.mj_id2name.side_effect = lambda model, obj_type, i: body_names[i]

            # Use correct SMPLX joint names
            joint_mapping = {"Pelvis": "torso_link", "L_Hip": "left_hip_yaw_link"}
            retargeter = OmniRetargeter(
                robot_urdf_path="dummy.urdf",
                terrain_mesh_path="dummy.obj",
                joint_mapping=joint_mapping,
                robot_height=1.6
            )

            # Mock the get_joint_names method
            retargeter.get_joint_names = Mock(return_value=["torso_link", "left_hip_yaw_link"])

            missing_joints = retargeter.validate_joint_mapping()
            assert len(missing_joints) == 0  # All joints should be found


class TestPackageImport:
    """Test package import functionality."""

    def test_import_package(self):
        """Test that package can be imported."""
        import omniretargeting
        assert hasattr(omniretargeting, '__version__')
        assert hasattr(omniretargeting, 'OmniRetargeter')

    def test_version_consistency(self):
        """Test version consistency across files."""
        import omniretargeting
        from omniretargeting.__version__ import __version__

        assert omniretargeting.__version__ == __version__ == "0.1.0"
