"""Position and rotation interpolation for source motion adapters."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def linear_interpolate(
    array: np.ndarray,
    indices: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    """Linearly interpolate *array* at fractional *indices* along *axis*.

    Parameters
    ----------
    array:
        Source array of any shape.
    indices:
        1-D array of float indices into *axis* (must be in [0, N-1]
        where N = array.shape[axis]).
    axis:
        The axis along which to interpolate.

    Returns
    -------
    np.ndarray
        Array whose shape equals ``array.shape`` with ``array.shape[axis]``
        replaced by ``len(indices)``.
    """
    axis = axis % array.ndim
    n = array.shape[axis]
    indices = np.asarray(indices, dtype=np.float64)

    lo = np.clip(np.floor(indices).astype(int), 0, n - 1)
    hi = np.clip(lo + 1, 0, n - 1)
    frac = indices - lo

    # Reshape frac for broadcasting: length-M along *axis*, size-1 elsewhere.
    shape = [1] * array.ndim
    shape[axis] = len(indices)
    frac = frac.reshape(shape)

    return np.take(array, lo, axis=axis) * (1 - frac) + np.take(array, hi, axis=axis) * frac


def _rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    """Axis-angle (..., 3) -> quaternion (..., 4) in wxyz order."""
    original_shape = rotvec.shape[:-1]
    flat = rotvec.reshape(-1, 3)
    if not len(flat):
        return np.empty((*original_shape, 4), dtype=np.result_type(rotvec, 1.0))
    quat = Rotation.from_rotvec(flat).as_quat()[:, [3, 0, 1, 2]]
    # Retain the adapter's explicit identity threshold.
    quat[np.linalg.norm(flat, axis=-1) < 1e-10] = [1.0, 0.0, 0.0, 0.0]
    return quat.astype(np.result_type(rotvec, 1.0)).reshape(*original_shape, 4)


def _quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    """Unit wxyz quaternions (..., 4) -> principal axis-angle (..., 3).

    SciPy handles normalization and the near-identity series expansion; a
    zero quaternion is invalid and raises ValueError instead of masking it.
    """
    original_shape = quat.shape[:-1]
    flat = quat.reshape(-1, 4)
    if not len(flat):
        return np.empty((*original_shape, 3), dtype=np.result_type(quat, 1.0))
    rotvec = Rotation.from_quat(flat[:, [1, 2, 3, 0]]).as_rotvec()
    # The former conversion used sin(half_angle) < 1e-10 as its zero cutoff.
    rotvec[np.linalg.norm(rotvec, axis=-1) < 2e-10] = 0.0
    return rotvec.astype(np.result_type(quat, 1.0)).reshape(*original_shape, 3)


def slerp_interpolate(
    array: np.ndarray,
    indices: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    """Spherical linear interpolation for wxyz quaternion arrays at fractional *indices*.

    Parameters
    ----------
    array:
        Array of wxyz quaternion data.  The last dimension must be 4
        and *axis* selects the time/sequence dimension.  E.g. shape
        ``(T, 4)`` for a single rotation track or ``(T, J, 4)`` for
        *J* joints.
    indices:
        1-D array of float indices into *axis* (in ``[0, N-1]``).
    axis:
        The axis along which to interpolate (the "time" axis).

    Returns
    -------
    np.ndarray
        wxyz quaternion array with ``array.shape[axis]`` replaced by
        ``len(indices)``, same dtype.
    """
    assert array.shape[-1] == 4, f"Last dimension must be 4 (wxyz quaternion), got {array.shape[-1]}"
    axis = axis % array.ndim
    n = array.shape[axis]
    indices = np.asarray(indices, dtype=np.float64)

    lo = np.clip(np.floor(indices).astype(int), 0, n - 1)
    hi = np.clip(lo + 1, 0, n - 1)
    frac = indices - lo

    q0 = np.take(array, lo, axis=axis)
    q1 = np.take(array, hi, axis=axis)

    # Shortest-path: flip q1 when dot < 0
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)

    shape = [1] * q0.ndim
    shape[axis] = len(indices)
    frac = frac.reshape(shape)

    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)

    near_parallel = (sin_theta < 1e-10)
    safe_sin_theta = np.where(near_parallel, np.ones_like(sin_theta), sin_theta)

    s0 = np.sin((1 - frac) * theta) / safe_sin_theta
    s1 = np.sin(frac * theta) / safe_sin_theta

    s0 = np.where(near_parallel, 1 - frac, s0)
    s1 = np.where(near_parallel, frac, s1)

    result_quat = s0 * q0 + s1 * q1
    result_quat = result_quat / np.linalg.norm(result_quat, axis=-1, keepdims=True)

    return result_quat
