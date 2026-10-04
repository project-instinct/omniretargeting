"""Batch the original MuJoCo FK/Jacobian calls without Python frame loops."""

import ctypes

import numpy as np

from .batched_self_collision import native_library, NativeUnavailable, _pointer
from .native_mujoco import native_mujoco


def evaluate_kinematics(kin, poses, jacp, jacr):
    model, data = kin.model, kin.data
    try:
        _, functions = native_mujoco()
        addresses = ctypes.c_void_p(model._address), ctypes.c_void_p(data._address)
        library = native_library()
    except (NativeUnavailable, AttributeError):
        return False
    arrays = (np.ascontiguousarray(poses, dtype=float), data.qpos,
              data.xpos, data.xmat, data.geom_xpos, data.geom_xmat,
              kin.xpos, kin.xmat, kin.geom_xpos, kin.geom_xmat)
    library.ws3_fk(len(poses), model.nbody, model.ngeom, model.nq, model.nv,
                   int(jacp is not None), *functions[:3], *addresses,
                   *(_pointer(array) for array in arrays),
                   *(ctypes.c_void_p() if array is None else _pointer(array) for array in (jacp, jacr)))
    return True
