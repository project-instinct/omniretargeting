"""Optional exact CPU window kernels, built and cached outside the repository."""

import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess

import numpy as np

from .batched_self_collision import NativeUnavailable, _pointer
from .sdf import GridSDF
from .terms import NEAR_MARGIN


@lru_cache(maxsize=1)
def library():
    source = Path(__file__).with_suffix('.cpp')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:24]
    cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'omniretargeting/native'
    path = cache / f'window_{digest}.so'
    try:
        cache.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix(f'.{os.getpid()}.so')
            try:
                subprocess.run(['g++', '-std=c++17', '-O3', '-ffp-contract=off', '-shared',
                                '-fPIC', str(source), '-o', str(temporary)],
                               check=True, capture_output=True, timeout=60)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        result = ctypes.CDLL(str(path))
        result.r3_grid.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
                                  ctypes.c_double] + [ctypes.c_void_p] * 3
        result.r3_shell.argtypes = [ctypes.c_int64] * 4 + [ctypes.c_void_p] * 7 + [ctypes.c_double] + [ctypes.c_void_p] * 7
        result.r3_support.argtypes = [ctypes.c_int64] * 2 + [ctypes.c_void_p] * 2 + [ctypes.c_double] + [ctypes.c_void_p] * 2
        return result
    except (OSError, subprocess.SubprocessError, AttributeError) as error:
        raise NativeUnavailable(f'CPU window kernels unavailable: {error}') from error


class NativeGrid:
    def __init__(self, reference):
        self.reference = reference
        self.grid = np.ascontiguousarray(reference.clearance)
        self.shape = np.asarray(reference.shape, dtype=np.int64)
        self.lower, self.spacing = reference.lower, reference.spacing

    def query(self, points, gradient=True):
        points = np.ascontiguousarray(points, dtype=float)
        result = np.empty(points.shape[:-1], np.float32)
        library().r3_grid(result.size, _pointer(self.grid), _pointer(self.lower), self.spacing,
                          _pointer(self.shape), _pointer(points), _pointer(result))
        return (result, self.reference.gradient(points)) if gradient else result

    def gradient(self, points):
        return self.reference.gradient(points)

    def value_gradient(self, points):
        return self.reference.value_gradient(points)


def support(model, geom, directions):
    import mujoco
    from .._sequence.proximity import _hull_vertices, support_points
    if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
        return support_points(model, geom, directions)
    vertices = np.ascontiguousarray(_hull_vertices(model, int(model.geom_dataid[geom])))
    directions = np.asarray(directions, dtype=float)
    normalized = np.ascontiguousarray(directions / np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12))
    selected = np.empty(len(directions), np.int64)
    uncertain = np.empty(len(directions), np.uint8)
    # Both three-term dot products have error far below this absolute bound.
    # Only a separated winner is certified; ties use the original NumPy BLAS.
    margin = 64 * np.finfo(float).eps * (np.abs(vertices).max() + 1.)
    library().r3_support(len(directions), len(vertices), _pointer(normalized),
                        _pointer(vertices), margin, _pointer(selected), _pointer(uncertain))
    result = vertices[selected]
    fallback = uncertain != 0
    if fallback.any():
        result[fallback] = support_points(model, geom, directions[fallback])
    return result


def deepest(obj, sdf, rotation, translation, scales=None):
    if not isinstance(sdf, GridSDF):
        return obj.__class__._deepest_samples(obj, sdf, rotation, translation, scales)
    arrays = [np.ascontiguousarray(value) for value in (
        np.asarray(obj.shell_geoms, np.int64), np.asarray(obj.shell_index, np.int64),
        obj.shell_local, obj.kin.geom_xpos, obj.kin.geom_xmat, sdf.clearance,
        sdf.lower, np.asarray(sdf.shape, np.int64))]
    field = [None if value is None else np.ascontiguousarray(value, dtype=float)
             for value in (rotation, translation, scales)]
    threshold = np.ascontiguousarray(obj.shell_radius + obj.cfg['scene_clearance_m'] + NEAR_MARGIN)
    clearance = np.empty((obj.frames, len(obj.shell_geoms)))
    choices = np.empty(clearance.shape, np.int64)
    library().r3_shell(obj.frames, len(obj.shell_geoms), obj.model.ngeom, obj.shell_index.shape[1],
        *(_pointer(value) for value in arrays[:7]), sdf.spacing,
        _pointer(arrays[7]),
        *(ctypes.c_void_p() if value is None else _pointer(value) for value in field),
        _pointer(threshold), _pointer(clearance), _pointer(choices))
    frame, geom = np.nonzero(choices >= 0)
    geoms = arrays[0][geom]
    local = obj.shell_local[choices[frame, geom]][:, None]
    world = obj.kin.geom_xpos[frame, geoms][:, None] + np.einsum(
        'nij,nsj->nsi', obj.kin.geom_xmat[frame, geoms], local)
    if rotation is None:
        points = world
    else:
        points = np.einsum('nji,nsj->nsi', rotation[frame], world - translation[frame, None])
        if scales is not None:
            points = points / scales[frame, None, None]
    return clearance, frame, geom, world[:, 0], points[:, 0]
