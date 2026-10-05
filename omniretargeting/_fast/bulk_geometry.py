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
        # Only window evaluators opt into this directional certificate. Whole
        # acquisition keeps its existing batching and original support path.
        self.window_support_reuse = False
        self.support_previous = {}
        self.window_queries = None
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
        slack = 64*np.finfo(float).eps*(np.abs(vertices).max()+1)
        choices, margins = np.empty(len(normalized), np.int64), np.empty(len(normalized))
        if not self.window_support_reuse:
            self.library.bulk_support(len(normalized), len(vertices),
                *(_pointer(a) for a in (normalized, vertices, choices, margins)))
        else:
            self._window_support(geom, vertices, normalized, choices, margins, slack)
        # One nonzero component has a single multiplication in every BLAS
        # reduction. Exact ties and zero directions preserve the first index.
        ambiguous = (margins <= slack) & (np.count_nonzero(normalized, axis=1) > 1)
        self.diagnostics['support_queries'] = self.diagnostics.get('support_queries', 0)+len(normalized)
        self.diagnostics['support_fallbacks'] = self.diagnostics.get('support_fallbacks', 0)+int(ambiguous.sum())
        return vertices[choices], ambiguous

    def _window_support(self, geom, vertices, normalized, choices, margins, slack):
        reused = np.zeros(len(normalized), bool)
        previous = self.support_previous.get(geom)
        if previous is not None and previous[0].shape == normalized.shape:
            old_direction, old_choice, old_margin, radius = previous
            # Each vertex dot product moves by at most |v| |d'-d|. Subtract
            # twice this bound from the old winner/runner-up gap. Additional
            # slack covers both native reductions and reference BLAS rounding.
            displacement = 2*radius*np.linalg.norm(normalized-old_direction, axis=1)
            remaining = old_margin-displacement
            reused = remaining > 4*slack
            choices[reused], margins[reused] = old_choice[reused], remaining[reused]
        pending = np.flatnonzero(~reused)
        if len(pending):
            directions = np.ascontiguousarray(normalized[pending])
            selected, gap = np.empty(len(pending), np.int64), np.empty(len(pending))
            if self.window_queries is None:
                self.library.bulk_support(len(pending), len(vertices),
                    *(_pointer(a) for a in (directions, vertices, selected, gap)))
            else:
                self.window_queries.support(geom, vertices, directions, selected, gap, slack)
            choices[pending], margins[pending] = selected, gap
        radius = previous[3] if previous is not None else np.linalg.norm(vertices, axis=1).max(initial=0.)
        anchor_margin = margins.copy()
        anchor_direction = normalized.copy()
        if reused.any():
            # Keep the original certificate anchor; repeated subtractions
            # would accumulate rounding across a long sequence of tiny trials.
            anchor_direction[reused] = old_direction[reused]
            anchor_margin[reused] = old_margin[reused]
        self.support_previous[geom] = anchor_direction, choices.copy(), anchor_margin, radius
        self.diagnostics['support_reused'] = self.diagnostics.get('support_reused', 0)+int(reused.sum())

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
        if self.window_queries is None or getattr(self.objective, '_window_cpu_budget', None) is None:
            return self._deepest(sdf, rotation, translation, scales)
        from .window_native_parallel import shell_workers
        with shell_workers(self):
            return self._deepest(sdf, rotation, translation, scales)

    def _deepest(self, sdf, rotation, translation, scales=None):
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
            if self.window_queries is None:
                self.library.bulk_shell(len(pairs), len(samples), int(geoms[geom]), obj.model.ngeom,
                    *(None if a is None else _pointer(a) for a in arrays), sdf.spacing,
                    _pointer(value), _pointer(choice))
            else:
                self.window_queries.shell(sdf, int(geom), frames, arrays, sdf.spacing, value, choice)
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
