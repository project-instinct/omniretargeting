"""Exact hull threshold decisions without allocating point-by-facet matrices."""

import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess

import numpy as np


@lru_cache(maxsize=1)
def library():
    source = Path(__file__).with_name('hull_threshold_native.cpp')
    cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'omniretargeting/native'
    cache.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(source.read_bytes()).hexdigest()[:24]
    target = cache / f'hull_threshold_{signature}.so'
    if not target.exists():
        temporary = target.with_suffix(f'.{os.getpid()}.so')
        subprocess.run(['g++', '-std=c++17', '-O3', '-ffp-contract=off', '-shared', '-fPIC',
                        str(source), '-o', str(temporary)], check=True, capture_output=True, timeout=60)
        temporary.replace(target)
    native = ctypes.CDLL(str(target))
    native.hull_thresholds.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64,
        ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p]
    native.hull_thresholds.restype = None
    return native


class HullThresholds:
    """Only certified booleans bypass the original hull margin evaluation.

    A partial facet traversal suffices as soon as every threshold is exceeded.
    Inside points visit all facets. Near a threshold, the original NumPy/BLAS
    arithmetic resolves the decision, including its exact floating-point ties.
    """

    def __init__(self, scene):
        self.scene = scene
        self.library = library()

    def __call__(self, points, *thresholds):
        points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
        thresholds = np.ascontiguousarray(thresholds, dtype=np.float64)
        planes = self.scene._hull
        if planes is None:
            return np.zeros((len(points), len(thresholds)), dtype=bool)
        planes = np.ascontiguousarray(planes, dtype=np.float64)
        flags = np.empty((len(points), len(thresholds)), dtype=np.int8)
        self.library.hull_thresholds(len(points), points.ctypes.data, len(planes),
            planes.ctypes.data, len(thresholds), thresholds.ctypes.data, flags.ctypes.data)
        result = flags == 1
        unresolved = np.flatnonzero(np.any(flags < 0, axis=1))
        if len(unresolved):
            # Retain the *whole* original boundary batch: reducing it to one
            # row can switch BLAS GEMM to GEMV and change last-bit cancellation.
            return self.scene.hull_margin(points)[:, None] > thresholds
        return result
