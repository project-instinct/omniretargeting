"""Optional access to the original MuJoCo C entry points."""

import ctypes
from functools import lru_cache
from pathlib import Path

import mujoco

from .batched_self_collision import NativeUnavailable


@lru_cache(maxsize=1)
def native_mujoco():
    try:
        path = next(Path(mujoco.__file__).parent.glob('libmujoco.so.*'))
        library = ctypes.CDLL(str(path))
        names = ('mj_kinematics', 'mj_comPos', 'mj_jacBody', 'mj_jac')
        functions = tuple(ctypes.cast(getattr(library, name), ctypes.c_void_p) for name in names)
    except (OSError, StopIteration, AttributeError) as error:
        raise NativeUnavailable(f'MuJoCo C entry points unavailable: {error}') from error
    return library, functions
