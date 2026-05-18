"""Unit tests for object interaction support."""

import numpy as np
import pytest

from omniretargeting import MotionData, MotionFrame, validate_object_points


class TestObjectPoints:
    """Test object_points field in MotionFrame and MotionData."""

    def test_motion_frame_with_object_points(self):
        """MotionFrame with object_points should work."""
        positions = np.random.randn(5, 3)
        object_points = np.random.randn(10, 3)
        frame = MotionFrame(positions=positions, object_points=object_points)
        assert frame.object_points.shape == (10, 3)

    def test_motion_frame_without_object_points(self):
        """MotionFrame without object_points should work."""
        positions = np.random.randn(5, 3)
        frame = MotionFrame(positions=positions)
        assert frame.object_points is None

    def test_motion_frame_invalid_object_points_shape(self):
        """Invalid object_points shape should raise ValueError."""
        positions = np.random.randn(5, 3)
        object_points = np.random.randn(10)  # Wrong shape
        with pytest.raises(ValueError, match="object_points must have finite shape"):
            MotionFrame(positions=positions, object_points=object_points)

    def test_motion_frame_object_points_with_nan(self):
        """Object points with NaN should raise ValueError."""
        positions = np.random.randn(5, 3)
        object_points = np.array([[1.0, 2.0, np.nan]])
        with pytest.raises(ValueError, match="object_points must have finite shape"):
            MotionFrame(positions=positions, object_points=object_points)

    def test_motion_data_with_object_points(self):
        """MotionData with object_points should validate frame count."""
        positions = np.random.randn(10, 5, 3)
        object_points = np.random.randn(10, 20, 3)
        motion_data = MotionData(positions=positions, object_points=object_points)
        assert motion_data.object_points.shape == (10, 20, 3)

    def test_motion_data_without_object_points(self):
        """MotionData without object_points should work."""
        positions = np.random.randn(10, 5, 3)
        motion_data = MotionData(positions=positions)
        assert motion_data.object_points is None

    def test_motion_data_object_points_wrong_frame_count(self):
        """Object points with wrong frame count should raise ValueError."""
        positions = np.random.randn(10, 5, 3)
        object_points = np.random.randn(5, 20, 3)  # Only 5 frames
        with pytest.raises(ValueError, match="object_points has 5 frames but positions has 10 frames"):
            MotionData(positions=positions, object_points=object_points)

    def test_motion_data_object_points_wrong_shape(self):
        """Object points with wrong shape should raise ValueError."""
        positions = np.random.randn(10, 5, 3)
        object_points = np.random.randn(10, 20)  # Missing coordinate dimension
        with pytest.raises(ValueError, match="object_points must have shape"):
            MotionData(positions=positions, object_points=object_points)

    def test_motion_data_iter_frames_with_objects(self):
        """iter_frames should yield object_points per frame."""
        positions = np.random.randn(3, 5, 3)
        object_points = np.random.randn(3, 10, 3)
        motion_data = MotionData(positions=positions, object_points=object_points)
        
        frames = list(motion_data.iter_frames())
        assert len(frames) == 3
        for i, frame in enumerate(frames):
            assert frame.object_points is not None
            assert frame.object_points.shape == (10, 3)
            np.testing.assert_array_equal(frame.object_points, object_points[i])

    def test_motion_data_iter_frames_without_objects(self):
        """iter_frames without object_points should yield None."""
        positions = np.random.randn(3, 5, 3)
        motion_data = MotionData(positions=positions)
        
        frames = list(motion_data.iter_frames())
        assert len(frames) == 3
        for frame in frames:
            assert frame.object_points is None

    def test_validate_object_points_valid(self):
        """Valid object points should pass validation."""
        points = np.random.randn(10, 3)
        assert validate_object_points(points)

    def test_validate_object_points_empty(self):
        """Empty object points should pass validation."""
        points = np.empty((0, 3))
        assert validate_object_points(points)

    def test_validate_object_points_invalid_shape(self):
        """Invalid shape should fail validation."""
        points = np.random.randn(10)
        assert not validate_object_points(points)

    def test_validate_object_points_nan(self):
        """NaN values should fail validation."""
        points = np.array([[1.0, 2.0, np.nan]])
        assert not validate_object_points(points)


class TestScalingWithObjects:
    """Test that object_points scale correctly with enable_scene_scaling."""

    def test_object_points_scale_with_scene_scaling(self):
        """When enable_scene_scaling=True, object_points should scale."""
        # This test will be implemented after core.py integration
        # For now, just verify the data structure supports it
        positions = np.random.randn(5, 3, 3)
        object_points = np.ones((5, 10, 3))
        motion_data = MotionData(positions=positions, object_points=object_points)
        
        # Simulate scaling
        scale = 0.5
        scaled_object_points = motion_data.object_points * scale
        assert scaled_object_points.shape == (5, 10, 3)
        np.testing.assert_array_almost_equal(scaled_object_points, 0.5 * np.ones((5, 10, 3)))
