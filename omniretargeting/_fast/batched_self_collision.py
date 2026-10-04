"""Exact batch acceleration of the existing self-collision row query.

The reference mesh routine returns a separating-plane lower bound, not the
Euclidean gap. A tighter physical OBB bound cannot replace that answer: it
can discard pairs the reference retains. Native queries therefore reproduce
the reference's seed supports, Qhull options/facet order, and first separating
plane. Penetration witnesses use NumPy's own LAPACK; indeterminate cases
use the original query.

Across evaluations, an old sphere separation certificate stays valid after
subtracting a bound on each geom's maximum vertex displacement. This only
skips pairs the original sphere test also skips; computed mesh gaps are not
used to certify a different pose's reference-algorithm decision.
"""

from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import ctypes
from functools import lru_cache
import hashlib
import multiprocessing
import os
from pathlib import Path
import subprocess
import time

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from ..cpu_budget import available_cpus
from ..geometry_query import GeometryQueryError


class NativeUnavailable(RuntimeError):
    """The optional native accelerator cannot reproduce this environment."""


def _pointer(value):
    return ctypes.c_void_p(value.ctypes.data)


@lru_cache(maxsize=1)
def native_library():
    """Compile the small optional kernel once, outside the repository.

    Use NumPy's own BLAS to retain its support dot-product reduction order.
    Verify system Qhull against SciPy before enabling the accelerator. Missing
    compiler/library or a different Qhull implementation keeps the CPU path.
    """
    from threadpoolctl import threadpool_info

    directory = Path(__file__).resolve().parent
    source = directory / 'self_collision_native.cpp'
    try:
        signature = hashlib.sha256(source.read_bytes())
        for header in sorted((directory / '_qhull_r').glob('*.h')):
            signature.update(header.read_bytes())
    except OSError as error:
        raise NativeUnavailable(f'Native kernel sources unavailable: {error}') from error
    cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    cache = cache / 'omniretargeting' / 'native'
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise NativeUnavailable(f'Native kernel cache unavailable: {error}') from error
    path = cache / f'self_collision_{signature.hexdigest()[:24]}.so'
    if not path.exists():
        temporary = path.with_suffix(f'.{os.getpid()}.so')
        command = ['g++', '-std=c++17', '-O3', '-ffp-contract=off', '-shared',
                   '-fPIC', str(source), '-o', str(temporary), '-l:libqhull_r.so.8.0']
        try:
            subprocess.run(command, check=True, capture_output=True, timeout=60)
            temporary.replace(path)
        except (OSError, subprocess.SubprocessError) as error:
            temporary.unlink(missing_ok=True)
            raise NativeUnavailable(f'Native Qhull build unavailable: {error}') from error
    try:
        library = ctypes.CDLL(str(path))
        blas_path = next(p['filepath'] for p in threadpool_info()
                         if p['user_api'] == 'blas' and 'numpy.libs' in p['filepath'])
        blas = ctypes.CDLL(blas_path)
        functions = [getattr(blas, f'scipy_cblas_{name}64_') for name in ('dgemm', 'dgemv', 'ddot')]
        functions.append(getattr(blas, 'scipy_dgelsd_64_'))
        library.ws3_blas.argtypes = [ctypes.c_void_p] * 4
        library.ws3_blas(*(ctypes.cast(function, ctypes.c_void_p) for function in functions))
        library.ws3_hull.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        library.ws3_hull.restype = ctypes.c_int
        library.ws3_pairs.argtypes = [ctypes.c_int64] * 3 + [ctypes.c_void_p] * 15
        library.ws3_pairs.restype = None
        library.ws3_point_rows.argtypes = [ctypes.c_int64] * 4 + [ctypes.c_void_p] * 7
        library.ws3_point_rows.restype = None
        library.ws3_point_jac.argtypes = [ctypes.c_int64] * 4 + [ctypes.c_void_p] * 6
        library.ws3_point_jac.restype = None
        library.ws3_collision_jac.argtypes = [ctypes.c_int64] * 2 + [ctypes.c_void_p] * 7
        library.ws3_collision_jac.restype = None
        library.ws3_fk.argtypes = [ctypes.c_int64] * 6 + [ctypes.c_void_p] * 17
        library.ws3_fk.restype = None
        # Keep the dynamically loaded BLAS resident with the kernel.
        library._blas = blas
    except (OSError, AttributeError, StopIteration) as error:
        raise NativeUnavailable(f'Native BLAS/Qhull unavailable: {error}') from error
    rng = np.random.default_rng(710)
    for points in (rng.normal(size=(26, 3)), rng.normal(size=(41, 3))):
        reference = ConvexHull(points).equations
        result = np.empty((2 * len(points), 4))
        count = library.ws3_hull(_pointer(points), len(points), _pointer(result))
        if count != len(reference) or not np.array_equal(result[:count], reference):
            raise NativeUnavailable('System Qhull equations/facet order differ from SciPy')
    return library


class BatchedSelfCollision:
    """One fixed-model query with exact-pose caches and conservative reuse."""

    def __init__(self, query, workers=12):
        self.query, self.model = query, query.model
        self.library = native_library()
        self.workers = max(1, min(workers, available_cpus()))
        self.first, self.second = (np.ascontiguousarray(c, dtype=np.int64)
                                   for c in query.pairs.T)
        self.starts = np.zeros(self.model.ngeom, np.int64)
        self.counts = np.zeros_like(self.starts)
        self.radius = np.zeros(self.model.ngeom)
        vertices = []
        offset = 0
        for geom in query.geom_ids:
            if self.model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mesh = int(self.model.geom_dataid[geom])
            if mesh not in query.mesh_distance.vertices:
                query.mesh_distance.world_vertices(int(geom), query.data)
            local = query.mesh_distance.vertices[mesh]
            self.starts[geom], self.counts[geom] = offset, len(local)
            self.radius[geom] = np.linalg.norm(local, axis=1).max(initial=0)
            vertices.append(local)
            offset += len(local)
        self.vertices = (np.ascontiguousarray(np.concatenate(vertices)) if vertices
                         else np.zeros((0, 3)))
        self.seed = np.ascontiguousarray(query.mesh_distance.seed_axes)
        self.previous = None
        self.previous_threshold = None
        self.cache = OrderedDict()
        self.seed_provider = None
        self.diagnostics = {}

    def _nearby(self, positions, rotations):
        """Original bounding spheres, with displacement-certified reuse.

        For any local vertex v, |(R'-R)v + (t'-t)| is bounded above by
        |t'-t| + ||R'-R||_F max|v|. The same bound also covers geom-origin
        travel. By the reverse triangle inequality, the previous sphere
        lower bound minus the two travel bounds is a lower bound at q'.
        Strict slack keeps round-off at the reference threshold on its path.
        """
        a, b = self.first, self.second
        bound = self.model.geom_rbound
        bounded = (bound[a] > 0) & (bound[b] > 0)
        threshold = self.query.cfg['detection_distance']
        certified = np.zeros((len(positions), len(a)), bool)
        if self.previous is not None and positions.shape == self.previous[0].shape:
            old_position, old_rotation, old_gap, _ = self.previous
            travel = np.linalg.norm(positions - old_position, axis=-1)
            travel += np.linalg.norm(rotations - old_rotation, axis=(-2, -1)) * self.radius
            # Cover rounding in the norms and the two ways the reference
            # sphere predicate can be written. Relative slack also protects
            # large-coordinate/large-radius models near the cutoff.
            slack = 1e-12 + 128 * np.finfo(float).eps * (
                np.abs(old_gap) + travel[:, a] + travel[:, b]
                + bound[a] + bound[b] + abs(threshold) + 1.)
            certified = bounded & (old_gap - travel[:, a] - travel[:, b] > threshold + slack)
        # Calculate the original predicate for uncertified pairs only.
        frames, pairs = np.nonzero(~certified)
        gap = np.full(certified.shape, np.inf)
        delta = positions[frames, a[pairs]] - positions[frames, b[pairs]]
        distance = np.linalg.norm(delta, axis=1)
        gap[frames, pairs] = distance - bound[a[pairs]] - bound[b[pairs]]
        near = np.zeros_like(certified)
        near[frames, pairs] = (~bounded[pairs]
                              | (distance <= bound[a[pairs]] + bound[b[pairs]] + threshold))
        # Retain finite certificates for the next evaluation, rather than
        # claiming certified pairs have infinite separation.
        if certified.any():
            old_gap = self.previous[2]
            safe = old_gap - travel[:, a] - travel[:, b]
            gap[certified] = safe[certified]
        self.diagnostics['sphere_reused_pairs'] = int(certified.sum())
        return np.ascontiguousarray(near, dtype=np.uint8), gap

    def candidates(self, kin):
        """Exact separating results, and pairs needing the original routine."""
        self.diagnostics.update(native_cpu_seconds=0., native_wall_seconds=0., transform_reused_pairs=0)
        position = np.ascontiguousarray(kin.geom_xpos)
        rotation = np.ascontiguousarray(kin.geom_xmat)
        # Kinematics retains immutable arrays for its last four exact poses.
        # Strong references in this cache prevent object-id recycling.
        key = id(kin.geom_xpos), id(kin.geom_xmat), self.query.cfg['detection_distance']
        if key in self.cache:
            self.previous = self.cache[key]
            self.previous_threshold = key[-1]
            self.cache.move_to_end(key)
            self.diagnostics['exact_pose_reused'] = True
            return self.previous[-1]
        threshold = self.query.cfg['detection_distance']
        if self.previous is not None and threshold == self.previous_threshold:
            old_position, old_rotation, _, old_result = self.previous
            if (np.array_equal(position, old_position)
                    and np.array_equal(rotation, old_rotation)):
                self.diagnostics['exact_pose_reused'] = True
                return old_result
        self.diagnostics['exact_pose_reused'] = False
        near, gap = self._nearby(position, rotation)
        shape = near.shape
        clearance = np.full(shape, np.inf)
        points = np.zeros((*shape, 2, 3))
        directions = np.zeros((*shape, 3))
        status = np.zeros(shape, np.uint8)
        if (self.previous is not None and threshold == self.previous_threshold
                and position.shape == self.previous[0].shape):
            old_position, old_rotation, _, old_result = self.previous
            # Exact placements of BOTH geoms certify the complete witness,
            # even when another geom/frame changed. Do not extrapolate gaps
            # or reuse failed/primitive queries. rows() still recomputes each
            # Jacobian at the current qpos, which can differ at these poses.
            fixed = np.all(position.view(np.uint64) == old_position.view(np.uint64), axis=-1)
            fixed &= np.all(rotation.view(np.uint64) == old_rotation.view(np.uint64), axis=(-2, -1))
            reused = (near != 0) & (old_result[3] == 1) & fixed[:, self.first] & fixed[:, self.second]
            for new, old in zip((clearance, points, directions, status), old_result):
                new[reused] = old[reused]
            near[reused] = 0
            self.diagnostics['transform_reused_pairs'] = int(reused.sum())
        seeds = None
        if self.seed_provider is not None and self.seed_provider.pid == os.getpid():
            seeds = self.seed_provider(self, kin)
        work_cpu = []

        def run(start, stop):
            started = time.thread_time()
            arrays = (self.first, self.second, self.starts, self.counts, self.vertices,
                      rotation[start:stop], position[start:stop], near[start:stop], self.seed,
                      clearance[start:stop], points[start:stop], directions[start:stop], status[start:stop])
            self.library.ws3_pairs(stop - start, self.model.ngeom, len(self.first),
                                   *(_pointer(array) for array in arrays),
                                   *(ctypes.c_void_p() if seeds is None else _pointer(array[start:stop])
                                     for array in ((None, None) if seeds is None else seeds)))
            work_cpu.append(time.thread_time() - started)

        # ctypes releases the GIL; no model, MuJoCo data or Qhull state is
        # shared by the native worker calls. BLAS stays at one thread.
        count = min(self.workers, max(1, len(position) // 8))
        if multiprocessing.current_process().daemon:
            count = 1
        edges = np.linspace(0, len(position), count + 1).astype(int)
        started = time.perf_counter()
        if count == 1:
            run(0, len(position))
        else:
            with ThreadPoolExecutor(max_workers=count) as executor:
                list(executor.map(lambda bounds: run(*bounds), zip(edges[:-1], edges[1:])))
        self.diagnostics.update(native_cpu_seconds=sum(work_cpu),
                                native_wall_seconds=time.perf_counter() - started)
        result = clearance, points, directions, status
        self.previous = position.copy(), rotation.copy(), gap, result
        self.previous_threshold = threshold
        self.cache[key] = (kin.geom_xpos, kin.geom_xmat, gap, result)
        self.cache.move_to_end(key)
        while len(self.cache) > 4:
            self.cache.popitem(last=False)
        self.diagnostics.update(native_pairs=int((status == 1).sum()),
                                fallback_pairs=int((status == 2).sum()))
        return result

    def rows(self, poses, kin, with_jacobian, target=None):
        """Original active-pair order, residuals, Jacobians, and unknown frames."""
        clearance, points, directions, status = self.candidates(kin)
        query = self.query
        threshold = query.cfg['detection_distance']
        per_frame, unknown = [], 0
        self.unknown_frames = []
        for frame, pose in enumerate(poses):
            rows = []
            fallback = np.any(status[frame] == 2)
            cutoff = threshold if target is None else min(threshold, target)
            selected = np.flatnonzero((status[frame] == 2)
                                     | ((status[frame] == 1) & (clearance[frame] < cutoff)))
            if not len(selected):
                per_frame.append(rows)
                continue
            # Mixed primitive queries require MuJoCo's contact-normal fallback.
            if fallback or with_jacobian:
                query.data.qpos[:] = pose
                mixed = np.any((status[frame] == 2)
                               & ((self.counts[self.first] == 0) | (self.counts[self.second] == 0)))
                if mixed:
                    mujoco.mj_forward(query.model, query.data)
                else:
                    # Exactly the position/COM stages mj_jac uses. The original
                    # full forward also runs dynamics and collision detection,
                    # which mesh-distance witnesses do not consume.
                    mujoco.mj_kinematics(query.model, query.data)
                    if with_jacobian:
                        mujoco.mj_comPos(query.model, query.data)
            try:
                for pair in selected:
                    a, b = int(self.first[pair]), int(self.second[pair])
                    phi = clearance[frame, pair]
                    witness, direction = points[frame, pair], directions[frame, pair]
                    if status[frame, pair] == 2:
                        if self.counts[a] and self.counts[b]:
                            phi, witness, direction = query.mesh_distance.query(a, b, query.data)
                        else:
                            # Preserve the entire original query's normal and
                            # failure handling for primitive/mixed models.
                            rows = query.query(data=query.data)
                            break
                    if phi >= threshold:
                        continue
                    rows.append(dict(pair=[a, b], pair_id=int(query.pair_ids[pair]),
                                     pair_index=int(pair),
                                     body_ids=[int(self.model.geom_bodyid[a]), int(self.model.geom_bodyid[b])],
                                     clearance=float(phi), points=witness, direction=direction))
            except GeometryQueryError:
                unknown += 1
                self.unknown_frames.append(frame)
                rows = []
            if with_jacobian:
                active = [row for row in rows if target is None or row['clearance'] < target]
                if active:
                    from .native_collision_rows import collision_jacobians
                    for row, jacobian in zip(active, collision_jacobians(query, active)):
                        row['_jacobian'] = jacobian
            per_frame.append(rows)
        return per_frame, unknown
