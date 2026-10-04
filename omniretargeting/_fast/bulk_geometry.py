"""Opt-in CPU geometry for acquisition; original repair/audit queries stay intact.

Immutable normal groups amortize repeated hull supports. A roundoff certificate
retains the reference support vertex; uncertified choices use the original full
BLAS batch, including its shape and first-maximum tie behavior. Fused shell
queries retain every sample, float32 grid rounding and first-minimum selection.
"""
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess

import mujoco
import numpy as np

from .._sequence.proximity import _hull_vertices, support_points
from ..cpu_budget import available_cpus
from .batched_self_collision import NativeUnavailable, _pointer
from .sdf import GridSDF
from .terms import FrameTerms, NEAR_MARGIN


@lru_cache(maxsize=1)
def native_library():
    source = Path(__file__).with_name('bulk_geometry_native.cpp')
    try:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:24]
        directory = Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache'))
        directory = directory/'omniretargeting'/'native'
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise NativeUnavailable(f'CPU geometry source/cache unavailable: {error}') from error
    path = directory/f'bulk_geometry_{digest}.so'
    if not path.exists():
        temporary = path.with_suffix(f'.{os.getpid()}.so')
        try:
            subprocess.run(['g++', '-std=c++17', '-O3', '-ffp-contract=off',
                            '-shared', '-fPIC', str(source), '-o', str(temporary)],
                           capture_output=True, check=True, timeout=60)
            temporary.replace(path)
        except (OSError, subprocess.SubprocessError) as error:
            temporary.unlink(missing_ok=True)
            raise NativeUnavailable(f'CPU geometry build unavailable: {error}') from error
    try:
        library = ctypes.CDLL(str(path))
    except OSError as error:
        raise NativeUnavailable(f'CPU geometry load unavailable: {error}') from error
    library.bulk_support.argtypes = [ctypes.c_int64]*2+[ctypes.c_void_p]*4
    library.bulk_grid.argtypes = ([ctypes.c_int64]+[ctypes.c_void_p]*3+
                                 [ctypes.c_double]+[ctypes.c_void_p]*2)
    library.bulk_shell.argtypes = ([ctypes.c_int64]*4+[ctypes.c_void_p]*10+
                                  [ctypes.c_double]+[ctypes.c_void_p]*2)
    for name in ('bulk_support', 'bulk_grid', 'bulk_shell'):
        getattr(library, name).restype = None
    return library


class BulkGeometry:
    def __init__(self, objective):
        self.objective = objective
        self.library = native_library()
        self.groups, self.grids, self.local = {}, {}, {}
        self.diagnostics = {}
        self.vertices = {}
        # Parent preparation shares immutable hulls with every fork worker.
        for prox in getattr(objective, 'proximities', ()):
            for geom in np.unique(prox['geoms']):
                if objective.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
                    self.vertices[int(geom)] = _hull_vertices(objective.model, int(objective.model.geom_dataid[geom]))

    def group(self, prox, geom):
        key = id(prox), geom
        if key not in self.groups:
            index = np.flatnonzero(prox['geoms'] == geom)
            normal = np.ascontiguousarray(prox['normals'][:, index], dtype=float)
            frames, count = normal.shape[:2]
            keys = np.empty((frames*count, 4), np.uint64)
            keys[:, 0] = np.arange(frames).repeat(count)
            keys[:, 1:] = normal.reshape(-1, 3).view(np.uint64)
            _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
            self.groups[key] = first, inverse
        return self.groups[key]

    def support(self, prox, geom, directions):
        first, inverse = self.group(prox, geom)
        directions = np.asarray(directions, float)
        obj = self.objective
        if obj.model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            return support_points(obj.model, geom, directions[first])[inverse]
        selected, ambiguous = self._mesh_support(geom, directions[first])
        if ambiguous.any():
            # BLAS can use different tail kernels within one batch. Preserve
            # each original row's choice, even among identical directions.
            return support_points(obj.model, geom, directions)
        return selected[inverse]

    def support_unique(self, geom, directions, *, fallback=support_points):
        """Window grouping supplies the reference batch, including its ties."""
        obj = self.objective
        if obj.model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            return support_points(obj.model, geom, directions)
        selected, ambiguous = self._mesh_support(geom, directions)
        return fallback(obj.model, geom, directions) if ambiguous.any() else selected

    def _mesh_support(self, geom, directions):
        obj = self.objective
        if geom not in self.vertices:
            self.vertices[geom] = _hull_vertices(obj.model, int(obj.model.geom_dataid[geom]))
        vertices = self.vertices[geom]
        normalized = np.array(directions, dtype=float, order='C', copy=True)
        normalized /= np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-12)
        choices, margins = np.empty(len(normalized), np.int64), np.empty(len(normalized))
        self.library.bulk_support(len(normalized), len(vertices),
            *(_pointer(a) for a in (normalized, vertices, choices, margins)))
        slack = 64*np.finfo(float).eps*(np.abs(vertices).max()+1)
        # One nonzero component has a single multiplication in every BLAS
        # reduction. Exact ties and zero directions preserve the first index.
        ambiguous = (margins <= slack) & (np.count_nonzero(normalized, axis=1) > 1)
        self.diagnostics['support_queries'] = self.diagnostics.get('support_queries', 0)+len(normalized)
        self.diagnostics['support_fallbacks'] = self.diagnostics.get('support_fallbacks', 0)+int(ambiguous.sum())
        return vertices[choices], ambiguous

    def grid(self, sdf):
        key = id(sdf)
        if key not in self.grids:
            if sdf.clearance.ndim != 3 or np.asarray(sdf.lower).shape != (3,):
                raise ValueError('CPU grid requires a 3D field and three lower bounds')
            self.grids[key] = (np.ascontiguousarray(sdf.clearance),
                              np.ascontiguousarray(sdf.shape, dtype=np.int64),
                              np.ascontiguousarray(sdf.lower, dtype=float))
        return self.grids[key]

    def query(self, sdf, points):
        points = np.ascontiguousarray(points, dtype=float)
        if points.ndim == 0 or points.shape[-1] != 3:
            raise ValueError('CPU grid points must have a final axis of length three')
        value = np.empty(points.shape[:-1], np.float32)
        grid, shape, lower = self.grid(sdf)
        self.library.bulk_grid(value.size, *(_pointer(a) for a in (grid, shape, lower)),
                               sdf.spacing, _pointer(points), _pointer(value))
        return value

    def deepest(self, sdf, rotation, translation, scales=None):
        obj = self.objective
        if type(sdf) is not GridSDF or (scales is not None and np.asarray(scales).dtype != np.dtype(float)):
            # Grid float32 * scale float32 rounds before first-min selection.
            # Retain this less common mixed-precision definition verbatim.
            return FrameTerms._deepest_samples(obj, sdf, rotation, translation, scales)
        kin, geoms = obj.kin, np.asarray(obj.shell_geoms)

        def to_field(points, frames):
            if rotation is None:
                return points
            local = np.einsum('nji,nsj->nsi', rotation[frames], points-translation[frames, None])
            return local if scales is None else local/scales[frames, None, None]

        frame_all = np.arange(obj.frames).repeat(len(geoms))
        centres = kin.geom_xpos[:, geoms].reshape(-1, 1, 3)
        centre_value = self.query(sdf, to_field(centres, frame_all)[:, 0]).reshape(obj.frames, -1)
        if scales is not None:
            centre_value = centre_value*scales[:, None]
        near = np.isfinite(centre_value) & (centre_value < obj.shell_radius+obj.cfg['scene_clearance_m']+NEAR_MARGIN)
        clearance = np.full(near.shape, np.inf)
        frame_index, geom_index = np.nonzero(near)
        world, field = np.zeros((len(frame_index), 3)), np.zeros((len(frame_index), 3))
        grid, shape, lower = self.grid(sdf)
        position = np.ascontiguousarray(kin.geom_xpos)
        orientations = np.ascontiguousarray(kin.geom_xmat)
        field_r = None if rotation is None else np.ascontiguousarray(rotation, dtype=float)
        field_t = None if rotation is None else np.ascontiguousarray(translation, dtype=float)
        field_s = None if scales is None else np.ascontiguousarray(scales, dtype=float)
        for geom in np.unique(geom_index):
            pairs = np.flatnonzero(geom_index == geom)
            samples = obj.shell_index[geom]
            samples = samples[samples >= 0]
            if int(geom) not in self.local:
                self.local[int(geom)] = np.ascontiguousarray(obj.shell_local[samples])
            frames = np.ascontiguousarray(frame_index[pairs])
            value, choice = np.empty(len(pairs)), np.empty(len(pairs), np.int64)
            arrays = (frames, self.local[int(geom)], position, orientations,
                      field_r, field_t, field_s, grid, shape, lower)
            self.library.bulk_shell(len(pairs), len(samples), int(geoms[geom]), obj.model.ngeom,
                *(None if a is None else _pointer(a) for a in arrays), sdf.spacing,
                _pointer(value), _pointer(choice))
            # Selected-point reconstruction keeps the original NumPy path.
            points = kin.geom_xpos[frames, geoms[geom]][:, None]+np.einsum(
                'nij,nsj->nsi', kin.geom_xmat[frames, geoms[geom]], obj.shell_local[samples[choice]][:, None])
            clearance[frames, geom] = value
            world[pairs], field[pairs] = points[:, 0], to_field(points, frames)[:, 0]
            self.diagnostics['shell_queries'] = self.diagnostics.get('shell_queries', 0)+len(pairs)*len(samples)
        return clearance, frame_index, geom_index, world, field


def prepare(objective):
    """Install private hooks on this evaluator only, including copied chunks."""
    from .runtime_options import OPTIONS
    if not OPTIONS.bulk_cpu_geometry:
        objective.__dict__.pop('_bulk_geometry_unavailable', None)
        helper = objective.__dict__.pop('_bulk_geometry', None)
        if helper is not None:
            for name in ('_proximity_support', '_deepest_samples'):
                if getattr(objective.__dict__.get(name), '__self__', None) is helper:
                    objective.__dict__.pop(name, None)
        return
    if getattr(objective, '_bulk_geometry_unavailable', False):
        return
    # Windows/device evaluators already own their geometry providers.
    if getattr(objective, '_proximity_points', None) is not None:
        return
    helper = getattr(objective, '_bulk_geometry', None)
    if helper is None or helper.objective is not objective:
        try:
            helper = BulkGeometry(objective)
        except NativeUnavailable:
            objective._bulk_geometry_unavailable = True
            return
        objective._bulk_geometry = helper
        objective._proximity_support = helper.support
        objective._deepest_samples = helper.deepest


def worker_schedule(workers, chunks_per_worker):
    """More small CPU chunks hide the variable geometry cost of a frame.

    The optional bulk kernel gate measured 12/16/24 workers on the same frozen
    evaluation. Respect the existing per-process CPU allocation for windows.
    """
    from .runtime_options import OPTIONS
    if OPTIONS.bulk_cpu_geometry:
        return min(24, available_cpus()), 4
    return workers, chunks_per_worker
