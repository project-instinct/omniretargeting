"""Integration test for OMOMO dataset with object interaction.

OMOMO provides SINGLE OBJECT per sequence only.
For multi-object testing, use a different dataset.

Test case structure:
- Direct .json config files in tests/resources/omomo/
- Or directories with config.json inside
"""

import json
import numpy as np
import pytest
import trimesh
from pathlib import Path

from omniretargeting import OmniRetargeter
from omniretargeting.data_sources.omomo import OmomoDataSource
from omniretargeting.data_sources.smplx import DEFAULT_SMPLX_TARGET_NAMES
from omniretargeting.utils import create_flat_terrain


# Test case paths - can be .json files or directories
OMOMO_TEST_CASES = [
    "tests/resources/omomo/case_01_clothesstand.json",
    "tests/resources/omomo/case_02_floor_lamp.json",
    "tests/resources/omomo/case_03_large_box.json",
]


class TestCaseLoader:
    """Helper to load test case configurations."""
    
    @staticmethod
    def load_config(test_case_path: str) -> dict:
        """Load test case configuration from .json file or directory."""
        path = Path(test_case_path)
        
        # If it's a .json file, load it directly
        if path.suffix == ".json":
            if not path.exists():
                raise FileNotFoundError(f"Config not found: {path}")
            with open(path) as f:
                return json.load(f)
        
        # Otherwise, treat as directory with config.json inside
        config_path = path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path) as f:
            return json.load(f)
    
    @staticmethod
    def create_terrain(terrain_config: dict) -> trimesh.Trimesh:
        """Create terrain mesh from config."""
        if terrain_config["type"] == "flat":
            return create_flat_terrain(
                size=terrain_config.get("size", 10.0),
                height=terrain_config.get("height", 0.0),
                n_points=terrain_config.get("n_points", 4)
            )
        elif terrain_config["type"] == "mesh":
            mesh_path = terrain_config.get("mesh_path")
            if not mesh_path:
                raise ValueError("mesh_path required for terrain type 'mesh'")
            return trimesh.load(mesh_path)
        else:
            raise ValueError(f"Unknown terrain type: {terrain_config['type']}")
    
    @staticmethod
    def load_omomo_data(config: dict) -> tuple:
        """Load OMOMO data source and motion data."""
        source = OmomoDataSource(
            sequence_file=config["omomo_sequence_file"],
            sequence_index=config["omomo_sequence_index"],
            data_root=config["omomo_data_root"],
            n_object_samples=config.get("n_object_samples", 100),
        )
        motion_data = source.load()
        return source, motion_data


class TestOmomoSingleObject:
    """Test cases with OMOMO single object sequences."""
    
    @pytest.mark.parametrize("test_case_path", OMOMO_TEST_CASES)
    def test_omomo_retargeting(self, test_case_path):
        """Test retargeting with OMOMO single object."""
        if not test_case_path:
            pytest.skip("No test cases configured")
        
        # Load test case
        config = TestCaseLoader.load_config(test_case_path)
        print(f"\n=== Test Case: {config['name']} ===")
        print(f"Description: {config['description']}")
        
        # Load OMOMO data
        source, motion_data = TestCaseLoader.load_omomo_data(config)
        
        # Create terrain
        terrain = TestCaseLoader.create_terrain(config["terrain"])
        
        # Validate data structure
        assert motion_data.positions is not None
        assert motion_data.object_points is not None
        T = motion_data.positions.shape[0]
        N_obj = motion_data.object_points.shape[1]
        
        print(f"Sequence: {motion_data.metadata['seq_name']}")
        print(f"Object: {motion_data.metadata['object_name']}")
        print(f"Frames: {T}")
        print(f"Object samples per frame: {N_obj}")
        print(f"Terrain vertices: {len(terrain.vertices)}")
        
        # Validate object points
        assert motion_data.object_points.shape == (T, N_obj, 3)
        assert np.isfinite(motion_data.object_points).all()
        
        # Check frames can be iterated
        frames = list(motion_data.iter_frames())
        assert len(frames) == T
        assert frames[0].object_points is not None
        assert frames[0].object_points.shape == (N_obj, 3)
        
        print(f"✓ Test case passed: {config['name']}")


class TestOmomoCoordinateConventions:
    """Regression checks for OMOMO coordinate and object-transform semantics."""

    def test_reconstructed_pelvis_uses_expected_joint_names(self):
        omomo_root = "/localhdd/Datasets/OMOMO"
        sequence_file = f"{omomo_root}/data/test_diffusion_manip_seq_joints24.p"

        if not Path(sequence_file).exists():
            pytest.skip("OMOMO dataset not available")

        source = OmomoDataSource(
            sequence_file=sequence_file,
            sequence_index=318,
            data_root=omomo_root,
            n_object_samples=20,
        )
        motion_data = source.load()

        assert motion_data.target_names == DEFAULT_SMPLX_TARGET_NAMES
        assert motion_data.positions.shape[1] == len(DEFAULT_SMPLX_TARGET_NAMES)
        assert motion_data.root_translations.shape == (motion_data.positions.shape[0], 3)
        assert np.isfinite(motion_data.positions).all()
        assert np.isfinite(motion_data.root_translations).all()
        np.testing.assert_allclose(motion_data.root_translations, motion_data.positions[:, 0, :], atol=5e-3)

    def test_object_points_follow_recorded_object_com(self):
        omomo_root = "/localhdd/Datasets/OMOMO"
        sequence_file = f"{omomo_root}/data/test_diffusion_manip_seq_joints24.p"

        if not Path(sequence_file).exists():
            pytest.skip("OMOMO dataset not available")

        source = OmomoDataSource(
            sequence_file=sequence_file,
            sequence_index=318,
            data_root=omomo_root,
            n_object_samples=20,
        )
        motion_data = source.load()

        object_centroids = motion_data.object_points.mean(axis=1)
        object_com = motion_data.metadata["object_centroid_world"]

        np.testing.assert_allclose(object_centroids, object_com, atol=3e-2)


class TestOmomoDataValidation:
    """Validation tests for OMOMO data structure."""
    
    def test_omomo_data_structure(self):
        """Test basic OMOMO data loading without retargeting."""
        omomo_root = "/localhdd/Datasets/OMOMO"
        sequence_file = f"{omomo_root}/data/train_diffusion_manip_seq_joints24.p"
        
        if not Path(sequence_file).exists():
            pytest.skip("OMOMO dataset not available")
        
        source = OmomoDataSource(
            sequence_file=sequence_file,
            sequence_index=0,
            data_root=omomo_root,
            n_object_samples=50,
        )
        
        motion_data = source.load()
        
        # Validate structure
        assert motion_data.positions is not None
        assert motion_data.object_points is not None
        assert motion_data.root_translations is not None
        assert motion_data.root_orientations is not None
        
        T = motion_data.positions.shape[0]
        assert motion_data.object_points.shape[0] == T
        assert np.isfinite(motion_data.object_points).all()
        
        print(f"\n✓ OMOMO data structure validated")
        print(f"  Sequence: {motion_data.metadata['seq_name']}")
        print(f"  Object: {motion_data.metadata['object_name']}")
        print(f"  Frames: {T}")
    
    def test_omomo_all_objects_loadable(self):
        """Test that all 15 OMOMO objects can be loaded."""
        omomo_root = "/localhdd/Datasets/OMOMO"
        sequence_file = f"{omomo_root}/data/train_diffusion_manip_seq_joints24.p"
        
        if not Path(sequence_file).exists():
            pytest.skip("OMOMO dataset not available")
        
        import joblib
        data = joblib.load(sequence_file)
        
        # Find one sequence per object
        object_sequences = {}
        for idx, seq in data.items():
            seq_name = seq["seq_name"]
            parts = seq_name.split("_")
            obj_name = "_".join(parts[1:-1])
            if obj_name not in object_sequences:
                object_sequences[obj_name] = idx
        
        print(f"\n✓ Found {len(object_sequences)} unique objects")
        
        # Try loading each object
        failed_objects = []
        for obj_name, seq_idx in sorted(object_sequences.items()):
            try:
                source = OmomoDataSource(
                    sequence_file=sequence_file,
                    sequence_index=seq_idx,
                    data_root=omomo_root,
                    n_object_samples=20,
                )
                motion_data = source.load()
                print(f"  ✓ {obj_name:20s} - {motion_data.object_points.shape[1]} samples")
            except Exception as e:
                failed_objects.append((obj_name, str(e)))
                print(f"  ✗ {obj_name:20s} - FAILED: {e}")
        
        if failed_objects:
            pytest.fail(f"Failed to load {len(failed_objects)} objects: {failed_objects}")


class TestTerrainGeneration:
    """Test terrain generation utilities."""
    
    def test_flat_terrain_generation(self):
        """Test flat terrain mesh generation."""
        terrain = create_flat_terrain(size=10.0, height=0.0, n_points=4)
        
        assert isinstance(terrain, trimesh.Trimesh)
        assert len(terrain.vertices) == 16
        assert len(terrain.faces) == 18
        assert np.allclose(terrain.vertices[:, 2], 0.0)
        
        print(f"\n✓ Flat terrain generated: {len(terrain.vertices)} vertices")
    
    def test_terrain_export(self, tmp_path):
        """Test terrain can be exported to file."""
        terrain = create_flat_terrain(size=5.0, height=0.0, n_points=3)
        
        output_path = tmp_path / "test_terrain.obj"
        terrain.export(output_path)
        
        assert output_path.exists()
        
        # Reload and verify
        reloaded = trimesh.load(output_path)
        assert len(reloaded.vertices) == len(terrain.vertices)
        
        print(f"\n✓ Terrain exported and reloaded: {output_path}")
