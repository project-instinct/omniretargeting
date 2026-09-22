"""Rotation adapter contracts independent of the rotation library's layout."""

import numpy as np
import pytest

from omniretargeting.motion_interpolation import _quat_to_rotvec, _rotvec_to_quat


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rotation_conversion_preserves_batch_shape_dtype_and_wxyz(dtype):
    rotvec = np.zeros((2, 3, 3), dtype=dtype)
    for axis in range(3):
        rotvec[0, axis, axis] = np.pi / 2
        rotvec[1, axis, axis] = -np.pi / 2
    expected = np.zeros((2, 3, 4), dtype=dtype)
    expected[..., 0] = np.sqrt(0.5)
    for axis in range(3):
        expected[0, axis, axis + 1] = np.sqrt(0.5)
        expected[1, axis, axis + 1] = -np.sqrt(0.5)

    quat = _rotvec_to_quat(rotvec)

    assert quat.shape == expected.shape
    assert quat.dtype == dtype
    np.testing.assert_allclose(quat, expected, atol=1e-7)
    recovered = _quat_to_rotvec(quat)
    assert recovered.dtype == dtype
    np.testing.assert_allclose(recovered, rotvec, atol=1e-7)
    np.testing.assert_allclose(_quat_to_rotvec(-quat), recovered, atol=1e-7)


def test_rotation_conversion_keeps_small_angles_above_explicit_cutoff():
    # arccos(cos(angle / 2)) rounds this angle to zero in float64; the
    # conversion must use the quaternion's vector component as well.
    angle = 1e-8
    quat = np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
    np.testing.assert_allclose(_quat_to_rotvec(quat), [angle, 0.0, 0.0], atol=1e-20)
    np.testing.assert_array_equal(_rotvec_to_quat(np.array([1e-12, 0.0, 0.0])),
                                  [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(_quat_to_rotvec(np.array([1.0, 5e-13, 0.0, 0.0])),
                                  [0.0, 0.0, 0.0])


def test_rotation_conversion_uses_principal_angle_near_half_turn():
    angles = np.array([np.pi - 1e-7, np.pi, np.pi + 1e-7])
    vectors = np.zeros((3, 3))
    vectors[:, 2] = angles
    recovered = _quat_to_rotvec(_rotvec_to_quat(vectors))
    expected = vectors.copy()
    expected[2, 2] -= 2 * np.pi
    np.testing.assert_allclose(recovered, expected, atol=1e-14)


@pytest.mark.parametrize("shape", [(0,), (2, 0)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rotation_conversion_preserves_empty_tracks(shape, dtype):
    quat = _rotvec_to_quat(np.empty((*shape, 3), dtype=dtype))
    assert quat.shape == (*shape, 4)
    assert quat.dtype == dtype
    recovered = _quat_to_rotvec(quat)
    assert recovered.shape == (*shape, 3)
    assert recovered.dtype == dtype


def test_zero_quaternion_is_rejected_as_invalid_rotation():
    with pytest.raises(ValueError, match="zero norm"):
        _quat_to_rotvec(np.zeros(4))
