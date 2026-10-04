"""Direct FP64 band assembly and diagonal coordinate equilibration."""

import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess

import numpy as np
from scipy import sparse


@lru_cache(maxsize=1)
def _library():
    source = Path(__file__).with_suffix('.cpp').with_name('barrier_native.cpp')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:24]
    cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'omniretargeting/native'
    path = cache / f'barrier_{digest}.so'
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
        lib = ctypes.CDLL(str(path))
        lib.barrier_scatter.argtypes = [ctypes.c_int64] * 3 + [ctypes.c_void_p] * 5
        lib.barrier_scatter.restype = ctypes.c_int
        return lib
    except (OSError, subprocess.SubprocessError, AttributeError):
        return None  # Exact NumPy fallback; numerical behavior is unchanged.


def scatter(ab, matrix, weights):
    """Add sum_r weights[r] J_r^T J_r, without constructing J^T J."""
    matrix = sparse.csr_matrix(matrix, copy=False)
    if not matrix.has_canonical_format:
        matrix = matrix.copy()
        matrix.sum_duplicates()
        matrix.sort_indices()
    weights = np.broadcast_to(weights, (matrix.shape[0],)).astype(float, copy=False)
    lib = _library()
    if lib is not None:
        arrays = [np.ascontiguousarray(matrix.indptr, dtype=np.int64),
                  np.ascontiguousarray(matrix.indices, dtype=np.int64),
                  np.ascontiguousarray(matrix.data, dtype=float),
                  np.ascontiguousarray(weights), ab]
        if lib.barrier_scatter(matrix.shape[0], *ab.shape,
                               *[ctypes.c_void_p(a.ctypes.data) for a in arrays]):
            raise ValueError('Barrier row exceeds the declared band')
        return
    for row in range(matrix.shape[0]):
        start, stop = matrix.indptr[row:row + 2]
        columns, values = matrix.indices[start:stop], matrix.data[start:stop]
        i, j = np.tril_indices(len(columns))
        offsets = columns[i] - columns[j]
        if np.any(offsets >= len(ab)):
            raise ValueError('Barrier row exceeds the declared band')
        ab[offsets, columns[j]] += weights[row] * values[i] * values[j]


def physical_system(acc, temporal):
    """Physical GN curvature; allocate for every stencil of up to four frames."""
    frames, width = acc.gradient.shape
    size = frames * width
    ab = np.zeros((min(4 * width, size), size), order='F')
    rows, cols = np.tril_indices(width)
    offsets = (np.arange(frames) * width)[:, None]
    ab[rows - cols, offsets + cols] = acc.hessian[:, rows, cols]
    scatter(ab, temporal, 1.)
    return ab


def equilibrated_solve(ab, rhs, solve):
    """Symmetric diagonal scaling, then undo it in the original coordinates."""
    scale = 1 / np.sqrt(np.maximum(ab[0], 1e-30))
    scaled = ab.copy(order='F')
    for offset in range(len(ab)):
        length = ab.shape[1] - offset
        scaled[offset, :length] *= scale[offset:] * scale[:length]
    return scale * solve(scaled, rhs * scale)
