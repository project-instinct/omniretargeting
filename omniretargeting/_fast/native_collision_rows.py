"""Batch calls to the original MuJoCo point Jacobian and NumPy reduction."""

import ctypes
import numpy as np

from .batched_self_collision import native_library, NativeUnavailable, _pointer
from .native_mujoco import native_mujoco


def collision_jacobians(query, rows):
    """Return the same full-nv rows as ``query.jacobian`` in row order."""
    bodies = np.ascontiguousarray([row['body_ids'] for row in rows], dtype=np.int64)
    points = np.ascontiguousarray([row['points'] for row in rows], dtype=float)
    directions = np.ascontiguousarray([row['direction'] for row in rows], dtype=float)
    result = np.zeros((len(rows), query.model.nv))
    try:
        _, functions = native_mujoco()
        model = ctypes.c_void_p(query.model._address)
        data = ctypes.c_void_p(query.data._address)
    except (NativeUnavailable, AttributeError):
        return np.array([query.jacobian(row, query.data) for row in rows])
    native_library().ws3_collision_jac(
        len(rows), query.model.nv, functions[3], model, data,
        _pointer(bodies), _pointer(points), _pointer(directions), _pointer(result))
    return result
