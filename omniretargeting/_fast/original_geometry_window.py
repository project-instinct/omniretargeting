"""Exact frame caches and window-local views of original collision witnesses."""

from collections import OrderedDict
from copy import copy, deepcopy
import os
import sys
from types import SimpleNamespace

import mujoco
import numpy as np

from .parallel_geometry import merge_frames

POSES_PER_FRAME = 3


def validate_frames(geometry, frames):
    if (not isinstance(frames, range) or frames.step != 1 or frames.start < 0
            or frames.stop < frames.start or frames.stop > geometry.frames):
        raise ValueError("Geometry frames must be a contiguous in-range unit-step range")
    return frames


def _pose_key(geometry, pose):
    return geometry.reference_observation_id, np.asarray(pose, dtype=np.float64).tobytes()


def copy_rows(rows):
    """Own mutable witness values without recursively copying immutable keys."""
    return [(key, {name: (value.copy() if isinstance(value, np.ndarray)
                         else value.copy() if isinstance(value, list)
                         and all(type(v) in (int, float, bool, str, bytes, type(None)) for v in value)
                         else deepcopy(value) if isinstance(value, (dict, list, set)) else value)
                   for name, value in row.items()}) for key, row in rows]


def remember_frames(geometry, poses, parts, *, owned=False):
    """Keep completed per-frame results, never early-stop partial diagnostics."""
    from .repair_linearize import remember
    remember(geometry, poses, parts)
    if (not any(part["rejection"] is not None for part in parts)
            and len({item[0] for part in parts for item in part["frames"]}) == geometry.frames):
        geometry._window_reference = np.array(poses, copy=True)
    for part in parts:
        for frame, rows, depths, failures in part["frames"]:
            if frame == part["rejection"]:
                continue  # A moving-query rejection can leave this frame partial.
            cache = geometry._window_frame_cache.setdefault(frame, OrderedDict())
            key = _pose_key(geometry, poses[frame])
            witnesses = []
            for collision, row in rows:
                witness = {name: value for name, value in row.items() if name != "jacobian"}
                if collision[1] == "moving":
                    witness["jacobian"] = np.empty(0, dtype=np.float64)
                witnesses.append((collision, witness))
            cache[key] = (frame, witnesses if owned else copy_rows(witnesses),
                          dict(depths), list(failures))
            cache.move_to_end(key)
            while len(cache) > POSES_PER_FRAME:
                cache.popitem(last=False)


def _annotate(geometry, measured, frames, completed):
    nbody, weld = geometry.model.nbody, geometry.model.body_weldid
    values = np.zeros((len(frames), nbody), dtype=np.float64)
    done = np.array([frame in completed for frame in frames], dtype=bool)
    reliable = np.broadcast_to(done[:, None], values.shape).copy()
    for frame, _, _, failures in completed.values():
        if failures:
            reliable[frame - frames.start] = False
    for key, depth in measured["depths"].items():
        geoms = key[2:] if key[1] == "self" else key[-1:]
        for geom in geoms:
            body = int(weld[geometry.model.geom_bodyid[geom]])
            values[key[0] - frames.start, body] = max(values[key[0] - frames.start, body], depth)
    values, reliable = values[:, weld], reliable[:, weld]
    values[~reliable] = np.nan
    measured.update(queried_frames=tuple(frames), frame_complete=done,
                    reliable=bool(done.all() and reliable.all()), max_depth=values,
                    body_reliable=reliable, evidence_id=geometry.reference_observation_id)
    return measured


def measure_window(geometry, poses, frames, jacobians, protected_depths, activation):
    poses = np.asarray(poses, dtype=np.float64)
    if poses.shape != (geometry.frames, geometry.model.nq) or not np.isfinite(poses).all():
        raise ValueError("Geometry poses must be finite (frames, model.nq)")
    frames = validate_frames(geometry, range(geometry.frames) if frames is None else frames)
    if activation is not None and (not np.isfinite(activation) or not 0 <= activation <= .008):
        raise ValueError("Witness activation_distance must be in [0, 0.008] metres")
    completed, missing = {}, []
    for frame in frames:
        cache = geometry._window_frame_cache.get(frame, {})
        key = _pose_key(geometry, poses[frame])
        if key in cache:
            completed[frame] = cache[key]
            cache.move_to_end(key)
        else:
            missing.append(frame)
    cached_count = len(completed)
    rejection = None
    if protected_depths is not None:
        # Existing complete evidence can reject without preparing any GPU work.
        from .mesh_repair import _regressed
        rejection = next((frame for frame, _, depths, failures in completed.values()
                          if failures or any(_regressed(value, protected_depths.get(key, 0.))
                                             for key, value in depths.items())), None)
    queried, partial = [], set()
    if missing and rejection is None:
        # Probe likely failures once, then amortize preparation/IPC over all
        # remaining frames. Repeated small batches slowed complete candidates.
        batches = [missing]
        if protected_depths is not None and len(missing) > 32:
            wanted = set(missing)
            recent = [f for f in getattr(geometry, "_window_rejections", ()) if f in wanted]
            reference = getattr(geometry, "_window_reference", None)
            order = missing
            if reference is not None:
                change = np.abs(poses - reference).max(axis=1)
                order = sorted(missing, key=lambda f: -change[f])
            probe = set((recent + [f for f in order if f not in recent])[:32])
            batches = [sorted(probe), [f for f in missing if f not in probe]]
            if recent:
                # Repeated lost-observation/penetration failures often reject
                # every fraction at the same frame. Query it before preparing
                # the 31 other probe frames; a passing candidate still visits
                # the remaining probe and the complete missing-frame suffix.
                first = recent[0]
                batches = [[first], sorted(probe - {first}), batches[1]]
        for chunk in batches:
            if not chunk:
                continue
            parts = _query_missing(geometry, poses, chunk, protected_depths)
            queried.extend(chunk)
            completed.update({item[0]: item for part in parts for item in part["frames"]})
            rejected = [part["rejection"] for part in parts if part["rejection"] is not None]
            partial.update(rejected)
            if rejected:
                rejection = min(rejected)
                break
    if rejection is not None:
        recent = getattr(geometry, "_window_rejections", ())
        geometry._window_rejections = [rejection, *(f for f in recent if f != rejection)][:32]
    results = [completed[frame] for frame in frames if frame in completed]
    if rejection is not None:
        results = [item for item in results if item[0] <= rejection]
    measured = merge_frames([dict(frames=results, rejection=rejection)])
    measured["rows"] = copy_rows(measured["rows"])
    _annotate(geometry, measured, frames, {item[0]: item for item in results if item[0] not in partial})
    omitted = np.zeros(len(frames), dtype=np.int64)
    if activation is not None:
        retained = []
        for key, row in measured["rows"]:
            if row["clearance"] < activation:
                retained.append((key, row))
            else:
                omitted[key[0] - frames.start] += 1
        measured["rows"] = retained
    measured.update(activation_distance=activation, omitted_near_misses=omitted,
                    query_scope=dict(requested_frames=list(frames), requested_frame_count=len(frames),
                                     queried_frames=queried, queried_frame_count=len(queried),
                                     cached_frame_count=cached_count))
    if jacobians:
        measured = geometry.linearize(poses, measured)
    geometry.window_stats = dict(requested_frames=len(frames), cached_frames=cached_count,
                                 queried_frames=len(queried))
    totals = getattr(geometry, "window_totals", None)
    if totals is None:
        totals = geometry.window_totals = dict(calls=0, requested_frames=0, cached_frames=0,
                                               queried_frames=0, rejected=0, cached_rejections=0)
    totals["calls"] += 1
    for name, value in geometry.window_stats.items():
        totals[name] += value
    totals["rejected"] += rejection is not None
    totals["cached_rejections"] += rejection is not None and not queried
    return measured


def _query_missing(geometry, poses, missing, protected_depths):
    static = geometry._prepare_gpu_static(poses, frames=missing)
    moving = geometry._prepare_gpu_moving(poses, frames=missing)
    native_self = getattr(geometry, "_fork_self_query", None)
    self_window = getattr(geometry, "_fork_self_window", None)
    if self_window is not None or native_self is not None:
        positions, rotations = [], []
        for frame in missing:
            geometry.data.qpos[:] = poses[frame]
            mujoco.mj_forward(geometry.model, geometry.data)
            positions.append(geometry.data.geom_xpos.copy())
            rotations.append(geometry.data.geom_xmat.reshape(-1, 3, 3).copy())
        kin = SimpleNamespace(geom_xpos=np.array(positions), geom_xmat=np.array(rotations))
        if self_window is not None:
            measured_self = self_window.query_frames(poses, missing, kinematics=kin)
            by_frame = {frame: [] for frame in missing}
            for key, row in measured_self["rows"]:
                by_frame[key[0]].append(row)
            geometry._fork_self_frames = {
                frame: (poses[frame].tobytes(), by_frame[frame],
                        [failure for failure in measured_self["failures"]
                         if failure.startswith(f"unknown self frame {frame}: ")])
                for frame in missing}
        else:
            rows, _ = native_self.rows(poses[missing], kin, False)
            unknown = set(native_self.unknown_frames)
            geometry._fork_self_frames = {frame: (poses[frame].tobytes(), row)
                                          for index, (frame, row) in enumerate(zip(missing, rows))
                                          if index not in unknown}
    parallel = getattr(geometry, "parallel", None)
    if parallel is None:
        parts = [geometry.measure_frames(poses, missing, protected_depths=protected_depths)]
    else:
        parts = parallel.measure(poses, False, protected_depths, static_updates=static,
                                 moving_updates=moving, frames=missing)
        if geometry.moving is not None:
            geometry.moving.query_count = parallel.query_count
    # These private query results never escape. The public window result is
    # still copied below, including lists/dicts and every mutable array.
    remember_frames(geometry, poses, parts, owned=True)
    return parts


def fork_private_client(geometry):
    """Clone with CPU buffers/caches and no inherited pool or CUDA commands."""
    client = copy(geometry)
    client.parallel = None
    client.data = mujoco.MjData(geometry.model)
    client.moving = None if geometry.moving is None else geometry.moving.fresh_state()
    client.self_query = geometry.self_query.fresh_state()
    client._static_cache = deepcopy(geometry._static_cache)
    client._self_reuse = deepcopy(geometry._self_reuse)
    client._window_frame_cache = deepcopy(geometry._window_frame_cache)
    for name in tuple(vars(client)):
        if name.startswith("_gpu_"):
            delattr(client, name)
    client._gpu_owner = client._query_owner = os.getpid()
    client._gpu_allowed = False
    client._fork_cpu_only = True
    if getattr(geometry, "_fork_self_window", None) is not None:
        client._fork_self_window = geometry._fork_self_window.fork_private_client()
        client._fork_self_frames = {}
    if getattr(geometry, "_fork_self_query", None) is not None:
        client._fork_self_query = copy(geometry._fork_self_query)
        client._fork_self_query.query = client.self_query
        client._fork_self_query.cache = OrderedDict()
        client._fork_self_query.previous = None
        client._fork_self_query.seed_provider = None
        client._fork_self_frames = {}
    if os.getpid() != geometry._query_owner:
        parallel = getattr(geometry, "parallel", None)
        if parallel is not None:
            for _, connection in parallel.workers:
                connection.close()
        gpu_geometry = sys.modules.get(__package__ + ".gpu_geometry")
        service = None if gpu_geometry is None else gpu_geometry._SERVICE
        if service is not None and service.pid != os.getpid():
            service.connection.close()
            gpu_geometry._SERVICE = None
    return client
