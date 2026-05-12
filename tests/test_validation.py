"""Tests for joint mapping validation utility."""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from pathlib import Path


class TestValidateRobotJointMapping:
    """Test suite for validate_robot_joint_mapping utility function."""
    
    def test_all_links_exist(self):
        """Test validation passes when all mapped links exist in robot."""
        from omniretargeting.utils import validate_robot_joint_mapping
        
        mock_model = Mock()
        mock_model.nbody = 3
        
        def mock_id2name(model, obj_type, idx):
            body_names = ["world", "pelvis", "left_foot"]
            return body_names[idx] if idx < len(body_names) else None
        
        joint_mapping = {
            "Pelvis": "pelvis",
            "L_Foot": "left_foot"
        }
        
        with patch('mujoco.mj_id2name', side_effect=mock_id2name):
            missing = validate_robot_joint_mapping(mock_model, joint_mapping, raise_on_missing=False)
            assert missing == []
    
    def test_missing_links_return_list(self):
        """Test validation returns list of missing links when raise_on_missing=False."""
        from omniretargeting.utils import validate_robot_joint_mapping
        
        mock_model = Mock()
        mock_model.nbody = 2
        
        def mock_id2name(model, obj_type, idx):
            body_names = ["world", "pelvis"]
            return body_names[idx] if idx < len(body_names) else None
        
        joint_mapping = {
            "Pelvis": "pelvis",
            "L_Foot": "left_foot",
            "R_Foot": "right_foot"
        }
        
        with patch('mujoco.mj_id2name', side_effect=mock_id2name):
            missing = validate_robot_joint_mapping(mock_model, joint_mapping, raise_on_missing=False)
            assert sorted(missing) == ["left_foot", "right_foot"]
    
    def test_missing_links_raise_error(self):
        """Test validation raises ValueError when raise_on_missing=True."""
        from omniretargeting.utils import validate_robot_joint_mapping
        
        mock_model = Mock()
        mock_model.nbody = 2
        
        def mock_id2name(model, obj_type, idx):
            body_names = ["world", "pelvis"]
            return body_names[idx] if idx < len(body_names) else None
        
        joint_mapping = {
            "Pelvis": "pelvis",
            "L_Foot": "nonexistent_link"
        }
        
        with patch('mujoco.mj_id2name', side_effect=mock_id2name):
            with pytest.raises(ValueError) as exc_info:
                validate_robot_joint_mapping(mock_model, joint_mapping, raise_on_missing=True)
            
            assert "nonexistent_link" in str(exc_info.value)
            assert "not found in URDF" in str(exc_info.value)
    
    def test_empty_mapping(self):
        """Test validation with empty joint mapping."""
        from omniretargeting.utils import validate_robot_joint_mapping
        
        mock_model = Mock()
        mock_model.nbody = 2
        
        def mock_id2name(model, obj_type, idx):
            return "body_" + str(idx)
        
        joint_mapping = {}
        
        with patch('mujoco.mj_id2name', side_effect=mock_id2name):
            missing = validate_robot_joint_mapping(mock_model, joint_mapping, raise_on_missing=False)
            assert missing == []
    
    def test_duplicate_robot_links(self):
        """Test validation when multiple source targets map to same robot link."""
        from omniretargeting.utils import validate_robot_joint_mapping
        
        mock_model = Mock()
        mock_model.nbody = 2
        
        def mock_id2name(model, obj_type, idx):
            body_names = ["world", "pelvis"]
            return body_names[idx] if idx < len(body_names) else None
        
        joint_mapping = {
            "Pelvis": "pelvis",
            "Pelvis_Alt": "pelvis"
        }
        
        with patch('mujoco.mj_id2name', side_effect=mock_id2name):
            missing = validate_robot_joint_mapping(mock_model, joint_mapping, raise_on_missing=False)
            assert missing == []
    
    def test_none_body_names_ignored(self):
        """Test that None body names are properly ignored."""
        from omniretargeting.utils import validate_robot_joint_mapping
        
        mock_model = Mock()
        mock_model.nbody = 4
        
        def mock_id2name(model, obj_type, idx):
            body_names = ["world", None, "pelvis", None]
            return body_names[idx] if idx < len(body_names) else None
        
        joint_mapping = {
            "Pelvis": "pelvis"
        }
        
        with patch('mujoco.mj_id2name', side_effect=mock_id2name):
            missing = validate_robot_joint_mapping(mock_model, joint_mapping, raise_on_missing=False)
            assert missing == []


class TestValidationIntegration:
    """Integration tests with real robot models."""
    
    def test_g1_robot_validation(self):
        """Test validation with real Unitree G1 robot model."""
        import mujoco
        from omniretargeting.robot_config import load_robot_config
        from omniretargeting.utils import validate_robot_joint_mapping
        
        repo_root = Path(__file__).resolve().parents[1]
        g1_config_path = repo_root / "robot_models" / "unitree_g1" / "unitree_g1.json"
        
        if not g1_config_path.exists():
            pytest.skip(f"G1 config not found at {g1_config_path}")
        
        config = load_robot_config(g1_config_path)
        urdf_path = config.get("urdf_path")
        joint_mapping = config.get("joint_mapping")
        
        if not urdf_path or not Path(urdf_path).exists():
            pytest.skip(f"G1 URDF not found at {urdf_path}")
        
        robot_model = mujoco.MjModel.from_xml_path(urdf_path)
        
        # Should pass without errors
        missing = validate_robot_joint_mapping(robot_model, joint_mapping, raise_on_missing=False)
        assert missing == [], f"G1 robot has missing links: {missing}"
    
    def test_h1_robot_validation(self):
        """Test validation with real Unitree H1 robot model."""
        import mujoco
        from omniretargeting.robot_config import load_robot_config
        from omniretargeting.utils import validate_robot_joint_mapping
        
        repo_root = Path(__file__).resolve().parents[1]
        h1_config_path = repo_root / "robot_models" / "unitree_h1" / "unitree_h1.json"
        
        if not h1_config_path.exists():
            pytest.skip(f"H1 config not found at {h1_config_path}")
        
        config = load_robot_config(h1_config_path)
        urdf_path = config.get("urdf_path")
        joint_mapping = config.get("joint_mapping")
        
        if not urdf_path or not Path(urdf_path).exists():
            pytest.skip(f"H1 URDF not found at {urdf_path}")
        
        robot_model = mujoco.MjModel.from_xml_path(urdf_path)
        
        # Should pass without errors
        missing = validate_robot_joint_mapping(robot_model, joint_mapping, raise_on_missing=False)
        assert missing == [], f"H1 robot has missing links: {missing}"
