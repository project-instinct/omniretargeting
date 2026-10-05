"""Certified pruning of the window's original float64/float32 query kernels."""

import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import weakref

import numpy as np

from .batched_self_collision import NativeUnavailable, _pointer

BLOCK = 2
_TREES = weakref.WeakKeyDictionary()
_GRIDS = weakref.WeakKeyDictionary()


@lru_cache(maxsize=1)
def library():
    source = Path(__file__).with_suffix('.cpp')
    try:
        digest = hashlib.sha256(source.read_bytes()+source.with_name('bulk_geometry_native.cpp').read_bytes()).hexdigest()[:24]
        cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache'))/'omniretargeting/native'
        cache.mkdir(parents=True, exist_ok=True)
        path = cache/f'window_query_{digest}.so'
        if not path.exists():
            temporary = path.with_suffix(f'.{os.getpid()}.so')
            try:
                subprocess.run(['g++', '-std=c++17', '-O3', '-ffp-contract=off',
                                '-shared', '-fPIC', str(source), '-o', str(temporary)],
                               capture_output=True, check=True, timeout=60)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        result = ctypes.CDLL(str(path))
        result.window_support.argtypes = [ctypes.c_int64]+[ctypes.c_void_p]*5+[ctypes.c_double]+[ctypes.c_void_p]*2
        result.window_shell.argtypes = ([ctypes.c_int64]*4+[ctypes.c_void_p]*10+
            [ctypes.c_double]+[ctypes.c_void_p]*2+[ctypes.c_int64]+[ctypes.c_void_p]*4)
        result.window_support.restype = result.window_shell.restype = None
        return result
    except (OSError, subprocess.SubprocessError, AttributeError) as error:
        raise NativeUnavailable(f'Window pruning kernel unavailable: {error}') from error


def support_tree(model, geom, vertices):
    mesh = int(model.geom_dataid[geom])
    cache = _TREES.setdefault(model, {})
    if mesh in cache:
        return cache[mesh]
    order, boxes, nodes = np.arange(len(vertices)), [], []

    def split(start, stop):
        points = vertices[order[start:stop]]
        lo, hi = points.min(axis=0), points.max(axis=0)
        node = len(nodes)
        boxes.append(np.r_[lo, hi])
        nodes.append([start, stop, -1, -1])
        if stop-start > 16:
            axis = np.argmax(hi-lo)
            order[start:stop] = order[start:stop][np.argsort(points[:, axis], kind='stable')]
            mid = (start+stop)//2
            nodes[node][2:] = split(start, mid), split(mid, stop)
        return node

    split(0, len(vertices))
    value = (np.ascontiguousarray(order, dtype=np.int64),
             np.ascontiguousarray(boxes, dtype=float), np.ascontiguousarray(nodes, dtype=np.int64))
    for array in value:
        array.setflags(write=False)
    cache[mesh] = value
    return value


def grid_bounds(sdf):
    if sdf in _GRIDS:
        return _GRIDS[sdf]
    shape = (np.asarray(sdf.clearance.shape)+BLOCK-1)//BLOCK
    lower = np.full(shape, np.inf)
    magnitude = np.zeros(shape)
    # A block covers cells, so include their extra upper corner node.
    for x in range(BLOCK+1):
        for y in range(BLOCK+1):
            for z in range(BLOCK+1):
                part = sdf.clearance[x::BLOCK, y::BLOCK, z::BLOCK]
                index = tuple(slice(0, size) for size in part.shape)
                finite = np.where(np.isfinite(part), part, np.inf)
                np.minimum(lower[index], finite, out=lower[index])
                np.maximum(magnitude[index], np.where(np.isfinite(part), np.abs(part), 0), out=magnitude[index])
    # Trilinear interpolation is a convex combination. Cover all double
    # products/sums, then round down past float32's final output rounding.
    bound = lower-64*np.finfo(float).eps*(magnitude+1)
    bound = np.nextafter(bound.astype(np.float32), np.float32(-np.inf))
    bound[np.isposinf(lower)] = np.inf  # Every node unknown: no finite sample.
    coarse = np.ascontiguousarray(shape, dtype=np.int64)
    bound.setflags(write=False)
    value = np.ascontiguousarray(bound), coarse
    _GRIDS[sdf] = value
    return value


class WindowQueries:
    def __init__(self, objective):
        self.objective = objective
        self.library = library()
        self.seeds = {}
        self.diagnostics = {}

    def support(self, geom, vertices, directions, choices, margins, slack):
        tree = support_tree(self.objective.model, geom, vertices)
        self.library.window_support(len(directions),
            *(_pointer(a) for a in (directions, vertices, *tree)), slack,
            _pointer(choices), _pointer(margins))

    def shell(self, sdf, geom, frames, arrays, spacing, value, choice):
        bounds, coarse = grid_bounds(sdf)
        key = sdf, geom
        seeds = self.seeds.get(key)
        if seeds is None:
            seeds = self.seeds[key] = np.full(self.objective.frames, -1, np.int64)
        selected = np.ascontiguousarray(seeds[frames])
        counters = np.zeros(2, np.int64)
        self.library.window_shell(len(frames), len(arrays[1]),
            int(self.objective.shell_geoms[geom]), self.objective.model.ngeom,
            *(None if a is None else _pointer(a) for a in arrays), spacing,
            _pointer(bounds), _pointer(coarse), BLOCK, _pointer(selected),
            _pointer(value), _pointer(choice), _pointer(counters))
        seeds[frames] = choice
        for name, count in zip(('shell_pruned', 'shell_interpolated'), counters):
            self.diagnostics[name] = self.diagnostics.get(name, 0)+int(count)


def prepare(objective):
    """Warm libraries and immutable bounds before the window worker fork."""
    import mujoco
    from .._sequence.proximity import _hull_vertices
    from .sdf import GridSDF
    library()
    for prox in objective.proximities:
        for geom in np.unique(prox['geoms']):
            if objective.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                vertices = _hull_vertices(objective.model, int(objective.model.geom_dataid[geom]))
                support_tree(objective.model, geom, vertices)
    for _, sdf, *_ in objective._clearance_fields():
        if type(sdf) is GridSDF:
            grid_bounds(sdf)
