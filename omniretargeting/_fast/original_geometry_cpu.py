"""Parent-resident serial query structures for forked geometry consumers.

The winding BVH uses the same libigl/SideFX algorithm and float conversions as
the installed binding. All reported distances still use its original AABB.
"""

from copy import copy
import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
from types import FunctionType, MethodType, SimpleNamespace

import igl
import mujoco
import numpy as np


@lru_cache(maxsize=1)
def winding_library():
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "omniretargeting" / "native"
    include = Path(os.environ.get("OMNIRETARGETING_WINDING_INCLUDE", cache / "r4_winding_headers"))
    source = Path(__file__).with_name("original_winding_native.cpp")
    signature = hashlib.sha256(source.read_bytes())
    for name in ("FastWindingNumberForSoups.h", "parallel_for.h"):
        signature.update((include / name).read_bytes())
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / ("r4_winding_" + signature.hexdigest()[:24] + ".so")
    if not target.exists():
        temporary = target.with_suffix(f".{os.getpid()}.so")
        subprocess.run(["g++", "-std=c++17", "-O3", "-ffp-contract=off", "-shared", "-fPIC",
                        "-I" + str(include), str(source), "-o", str(temporary)],
                       check=True, capture_output=True, timeout=60)
        temporary.replace(target)
    lib = ctypes.CDLL(str(target))
    lib.r4_winding_build.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    lib.r4_winding_build.restype = ctypes.c_void_p
    lib.r4_winding_query.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p]
    lib.r4_winding_free.argtypes = [ctypes.c_void_p]
    lib.r4_grid_lower.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    return lib


class CachedWinding:
    def __init__(self, vertices, faces):
        self.vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        self.faces = np.ascontiguousarray(faces, dtype=np.int64)
        self.library = winding_library()
        self.state = self.library.r4_winding_build(len(self.vertices), self.vertices.ctypes.data,
                                                  len(self.faces), self.faces.ctypes.data)
        if not self.state:
            raise RuntimeError("Could not build resident winding BVH")

    def __call__(self, points):
        started = time.perf_counter()
        points = np.ascontiguousarray(points, dtype=np.float64)
        result = np.empty(len(points), dtype=np.float64)
        self.library.r4_winding_query(self.state, len(points), points.ctypes.data, result.ctypes.data)
        self.seconds = getattr(self, "seconds", 0.) + time.perf_counter() - started
        self.points = getattr(self, "points", 0) + len(points)
        return result


class SerialAABB:
    """Stay below libigl's parallel-for threshold, irrespective of its singleton."""
    def __init__(self, tree):
        self.tree = tree
        self.seconds, self.points = 0., 0

    def squared_distance(self, vertices, faces, points):
        started = time.perf_counter()
        self.points += len(points)
        if len(points) < 1000:
            result = self.tree.squared_distance(vertices, faces, points)
        else:
            parts = [self.tree.squared_distance(vertices, faces, points[i:i + 999])
                     for i in range(0, len(points), 999)]
            result = tuple(np.concatenate([part[j] for part in parts]) for j in range(3))
        self.seconds += time.perf_counter() - started
        return result


def serial_scene(scene):
    if getattr(scene, "_r4_serial", False):
        return scene
    result = copy(scene)
    result.components = [(part, faces, closed, SerialAABB(tree)) for part, faces, closed, tree in scene.components]
    cache = {}
    scope = {}
    for vertices, faces, _, _ in scene._component_query_data:
        cache[id(vertices), id(faces)] = CachedWinding(vertices, faces)
    for completion in scene._completions:
        if completion is not None:
            vertices, faces = completion[1:]
            cache[id(vertices), id(faces)] = CachedWinding(vertices, faces)

    def winding(vertices, faces, points):
        backend = cache.get((id(vertices), id(faces)))
        if backend is None:
            raise RuntimeError("Unprepared winding geometry in a forked query")
        component = scope.get("component")
        if component is not None:
            v, f, bounds, _ = scene._component_query_data[component]
            if vertices is v and faces is f:
                original_points = scope["points"]
                inside = np.all((original_points >= bounds[0]) & (original_points <= bounds[1]), axis=1)
                return scope["winding"][inside]
        hints = scope.get("hints")
        if hints is None or len(scene.components) != 1 or not scene.components[0][2]:
            return backend(points)
        bounds = scene._component_query_data[0][2]
        original_points = scope["points"]
        inside = np.all((original_points >= bounds[0]) & (original_points <= bounds[1]), axis=1)
        values = hints[inside]
        known = np.isfinite(values)
        result = np.empty(len(points))
        result[known] = np.logical_xor(values[known] < 0, scene.enclosed_free_space).astype(float)
        if (~known).any():
            result[~known] = backend(points[~known])
        return result

    def exact_winding(vertices, faces, points):
        # The exact winding binding parallelizes query points, not tree grow.
        if len(points) < 1000:
            return igl.winding_number(vertices, faces, points)
        return np.concatenate([igl.winding_number(vertices, faces, points[i:i + 999])
                               for i in range(0, len(points), 999)])

    proxy = SimpleNamespace(fast_winding_number=winding, winding_number=exact_winding)
    for name in ("query", "_decisive_winding", "_winding_free"):
        original = getattr(type(scene), name)
        replacement = FunctionType(original.__code__, {**original.__globals__, "igl": proxy},
                                   original.__name__, original.__defaults__, original.__closure__)
        replacement.__kwdefaults__ = original.__kwdefaults__
        setattr(result, name, MethodType(replacement, result))
    result._r4_serial, result._r4_winding_cache = True, cache
    result._winding_grids = dict(scene._winding_grids)
    def query_with_sign_hints(points, hints):
        scope.update(points=points, hints=hints)
        try:
            return result.query(points)
        finally:
            scope.clear()
    result.query_with_sign_hints = query_with_sign_hints
    def query_with_winding(points, component, values):
        # Values are the original native fast-winding answers, not inferred
        # signs. The original query still resolves ambiguity with exact winding.
        scope.update(points=points, component=component, winding=values)
        try:
            return result.query(points, component_ids=[component])
        finally:
            scope.clear()
    result.query_with_winding = query_with_winding
    return result


def prepare_fork_queries(geometry):
    if getattr(geometry, "_fork_queries_prepared", False):
        return
    if os.getpid() != geometry._query_owner or getattr(geometry, "_fork_cpu_only", False):
        raise RuntimeError("Prepare fork geometry in the parent process")
    if not geometry._reference_observations_frozen:
        raise RuntimeError("Freeze the immutable reference before preparing fork geometry")
    geometry.stop_workers()
    seed_static_masks(geometry)
    # Construct every structure before dropping parent resources. No reference
    # or initial trajectory is measured by this hook.
    warm_scene_queries(geometry)
    if not hasattr(geometry, "_static_coarse_bounds"):
        from .original_geometry_bounds import build_scene_bounds
        geometry._static_coarse_bounds = build_scene_bounds(geometry)
    query = geometry.self_query
    for geom in query.geom_ids:
        if geometry.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            query.mesh_distance.world_vertices(int(geom), query.data)
    if not hasattr(query, "_normal_model"):
        query._normal_model = copy(geometry.model)
        query._normal_model.geom_margin[:] = np.maximum(geometry.model.geom_margin, 1e-5)
        query._normal_model.pair_margin[:] = np.maximum(geometry.model.pair_margin, 1e-5)
        query._normal_data = mujoco.MjData(query._normal_model)
    from .batched_self_collision import BatchedSelfCollision, NativeUnavailable
    from .self_collision_window import SelfCollisionWindow
    try:
        geometry._fork_self_window = SelfCollisionWindow(query, columns=geometry.columns).prepare()
        seed_self_cache(geometry)
    except NativeUnavailable as error:
        geometry.fork_self_unavailable = str(error)
        geometry._fork_self_query = BatchedSelfCollision(query, workers=1)
    gpu_geometry = sys.modules.get(__package__ + ".gpu_geometry")
    service = None if gpu_geometry is None else gpu_geometry._SERVICE
    if service is not None and service.pid == os.getpid():
        service.close()
        gpu_geometry._SERVICE = None
    geometry._gpu_allowed = False
    for name in tuple(vars(geometry)):
        if name.startswith("_gpu_") and name not in ("_gpu_owner", "_gpu_allowed"):
            delattr(geometry, name)
    geometry._fork_queries_prepared = True


def seed_self_cache(geometry):
    """Copy completed witness evidence into pair reuse; no geometry re-query."""
    poses = np.zeros((geometry.frames, geometry.model.nq))
    rows, positions, rotations, frames = [], [], [], []
    for frame, cache in sorted(geometry._window_frame_cache.items()):
        if not cache:
            continue
        key, (_, witnesses, _, failures) = next(reversed(cache.items()))
        if failures:
            continue
        poses[frame] = np.frombuffer(key[1], dtype=np.float64)
        geometry.data.qpos[:] = poses[frame]
        mujoco.mj_forward(geometry.model, geometry.data)
        frames.append(frame)
        rows.extend(witnesses)
        positions.append(geometry.data.geom_xpos.copy())
        rotations.append(geometry.data.geom_xmat.reshape(-1, 3, 3).copy())
    if frames:
        measured = dict(rows=rows, complete=True, failures=[], reliable=True)
        kin = SimpleNamespace(geom_xpos=np.array(positions), geom_xmat=np.array(rotations))
        geometry.fork_self_seed_stats = geometry._fork_self_window.seed_completed(
            poses, measured, kin, frames=frames)


def warm_scene_queries(geometry):
    """Build serial original-distance/sign structures before parent measurement."""
    if geometry.scene is not None:
        geometry.scene = serial_scene(geometry.scene)
    if geometry.moving is not None:
        geometry.moving.objects = [{**obj, "query": serial_scene(obj["query"])}
                                   for obj in geometry.moving.objects]
    queries = ([] if geometry.scene is None else [geometry.scene])
    queries += ([] if geometry.moving is None else [obj["query"] for obj in geometry.moving.objects])
    for scene in queries:
        if scene.bounded_object:
            scene.hull_classification(np.empty((0, 3)), 1e-9)
            for cid, (_, _, closed, _) in enumerate(scene.components):
                if not closed:
                    scene._winding_free(cid, np.empty((0, 3)), np.empty(0), np.empty(0, bool))


def measure_parent(geometry, poses, jacobians, protected_depths):
    """Temporary parent workers/GPU for initial measurement and final audits."""
    after_fork = getattr(geometry, "_fork_queries_prepared", False)
    if after_fork:
        from .gpu_geometry import initialize_static
        geometry._fork_queries_prepared = False
        try:
            geometry._gpu_allowed = initialize_static(geometry)
        except (ImportError, RuntimeError, ValueError, OSError, EOFError) as error:
            geometry._gpu_allowed = False
            geometry.gpu_unavailable = str(error)
    temporary_pool = getattr(geometry, "parallel", None) is None
    if temporary_pool:
        geometry.start_workers(len(poses))
    geometry._parent_full_query = True
    try:
        return geometry.measure(poses, jacobians=jacobians, protected_depths=protected_depths)
    finally:
        geometry._parent_full_query = False
        if temporary_pool:
            geometry.stop_workers()
        if after_fork:
            prepare_fork_queries(geometry)


def seed_static_masks(geometry):
    """Parent-only inflated-band GPU screen, reusable by pure CPU children."""
    if not geometry._gpu_allowed or not geometry._static_cache:
        return
    from .gpu_geometry import geometry_service
    pending = []
    for key, (_, summary, position, rotation) in geometry._static_cache.items():
        if not summary["unknown"] and not summary["row"] and not summary["rows"] and summary["clearance"] >= .02:
            summary["_screen_band"], summary["_screen_ids"] = .02, np.empty(0, int)
            summary["_screen_phi"] = np.empty(0)
        else:
            pending.append((key, summary, position, rotation))
    if not pending:
        return
    starts = np.array([geometry._gpu_starts[key[1]] for key, _, _, _ in pending], dtype=np.int32)
    counts = np.array([len(geometry._gpu_groups[key[1]]) for key, _, _, _ in pending], dtype=np.int32)
    result = geometry_service().candidates(geometry._gpu_token, np.array([u[2] for u in pending]),
        np.array([u[3] for u in pending]), starts, counts, band=.02)
    begin = 0
    for job, (key, summary, _, _) in enumerate(pending):
        end = int(np.searchsorted(result["jobs"], job, side="right"))
        ids = geometry._gpu_groups[key[1]][result["samples"][begin:end]].copy()
        from .gpu_geometry import DISTANCE_BAND
        phi = result["phi"][begin:end].astype(float)
        values = np.where((phi > 0) & ~result["ambiguous"][begin:end], phi - DISTANCE_BAND, np.nan)
        previous_ids, previous_phi = summary.get("_screen_ids"), summary.get("_screen_phi")
        if previous_ids is not None and previous_phi is not None and len(previous_ids) == len(previous_phi):
            lookup = np.searchsorted(ids, previous_ids)
            present = lookup < len(ids)
            present[present] &= ids[lookup[present]] == previous_ids[present]
            values[lookup[present]] = previous_phi[present]
        summary["_screen_ids"], summary["_screen_phi"] = ids, values
        summary["_screen_band"] = .02
        begin = end


def screened_static_samples(geometry, geom, selected, cached, summary):
    """Exact Lipschitz exclusion at fixed material IDs; near values stay CPU."""
    if (cached is None or not geometry._static_bound_reuse
            or "_screen_ids" not in cached[1]):
        return selected
    shift = np.linalg.norm(geometry.data.geom_xpos[geom] - cached[2])
    turn = np.linalg.norm(geometry.data.geom_xmat[geom].reshape(3, 3) - cached[3].reshape(3, 3))
    moved = shift + turn * geometry._static_radii[geom] + 1e-9
    band = cached[1].get("_screen_band", .008) - moved
    if band < .008 + 1e-9:
        return selected
    summary["_screen_band"] = band
    summary["clearance"] = band  # private lower bound; never returned as a witness
    previous_phi = cached[1].get("_screen_phi")
    if previous_phi is not None and len(previous_phi) == len(cached[1]["_screen_ids"]):
        summary["_reuse_sign_ids"] = cached[1]["_screen_ids"]
        summary["_reuse_sign_phi"] = np.where(np.abs(previous_phi) > moved + 1e-5, previous_phi, np.nan)
        lower = previous_phi - moved
        keep = ~np.isfinite(lower) | (lower < .008 + 1e-9)
        summary["_screen_ids"] = cached[1]["_screen_ids"][~keep].copy()
        summary["_screen_phi"] = lower[~keep]
        summary["_screen_sort"] = True
        return np.asarray(cached[1]["_screen_ids"], dtype=int)[keep]
    return np.asarray(cached[1]["_screen_ids"], dtype=int)
