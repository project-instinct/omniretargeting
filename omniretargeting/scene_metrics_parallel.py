"""Multiprocessing helpers for the original-scene acceptance audit.

The parent owns aggregation. Workers inherit the already-loaded immutable
geometry through fork and return only raw batch/frame results, keeping the
serial query boundaries and numerical path unchanged.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import multiprocessing
import os
import sys
import time


import mujoco
import numpy as np


@dataclass
class _StaticWorkerState:
    model: mujoco.MjModel
    query: object
    poses: np.ndarray
    ids: np.ndarray
    offsets: list[np.ndarray]
    owners: np.ndarray
    open_scan: bool
    data: mujoco.MjData | None = None
    backend: object | None = None


@dataclass
class _MovingWorkerState:
    scene: object
    poses: np.ndarray
    start: int
    batch: int
    query: object | None = None


_STATIC_STATE: _StaticWorkerState | None = None
_MOVING_STATE: _MovingWorkerState | None = None
_THREAD_LIMIT = None
_WORKER_THREADS = 1


def _batch_ranges(frames, batch):
    return [(begin, min(begin + batch, frames)) for begin in range(0, frames, batch)]


def _init_static_worker():
    # Replace the parent's MjData explicitly: fork gives each process a private
    # address space, and this also makes the invariant obvious in direct mode.
    _STATIC_STATE.data = mujoco.MjData(_STATIC_STATE.model)


def _init_moving_worker():
    _MOVING_STATE.query = _MOVING_STATE.scene.fresh_state()


def _normal_static_batch(task):
    begin, end = task[0], task[1]
    state = _STATIC_STATE
    chunk = state.poses[begin:end]
    points = []
    forward_started = time.perf_counter()
    for pose in chunk:
        state.data.qpos[:] = pose
        mujoco.mj_forward(state.model, state.data)
        points.append(
            np.concatenate(
                [
                    state.data.geom_xpos[g] + p @ state.data.geom_xmat[g].reshape(3, 3).T
                    for g, p in zip(state.ids, state.offsets)
                ]
            )
        )
    forward_seconds = time.perf_counter() - forward_started
    query_started = time.perf_counter()
    query_cpu_started = time.process_time()
    all_points = np.concatenate(points)
    measured = state.query.query(all_points)
    query_seconds = time.perf_counter() - query_started
    query_cpu_seconds = time.process_time() - query_cpu_started
    uncertain = int(measured["uncertain"].sum())
    signed = measured["clearance"].reshape(len(chunk), len(state.owners))
    if not np.isfinite(signed).all():
        raise RuntimeError(
            "Original static scene has unresolved sign; cannot certify depths"
        )
    depths = np.zeros((len(chunk), state.model.ngeom))
    for gid in state.ids:
        depths[:, gid] = np.maximum(
            -signed[:, state.owners == gid], 0
        ).max(axis=1)
    deepest = int(np.argmin(measured["clearance"]))
    depth = -float(measured["clearance"][deepest])
    worst_exit = depth * measured["directions"][deepest] if depth > 0 else np.zeros(3)
    return begin, {
        "depths": depths,
        "uncertain_samples": uncertain,
        "worst_depth": depth,
        "worst_exit": worst_exit,
        "worker": os.getpid(),
        "forward_seconds": forward_seconds,
        "query_seconds": query_seconds,
        "query_cpu_seconds": query_cpu_seconds,
    }


def _open_static_batch(task):
    from omniretargeting._fast.original_rigid_fallback import uncovered_components
    from omniretargeting.geometry_query import OPEN_SURFACE_GEOMS, RigidGeometryQuery

    begin, end = task
    state = _STATIC_STATE
    if state.backend is None:
        state.backend = RigidGeometryQuery(state.model, state.ids, state.query)
    backend = state.backend
    unsupported = [
        int(g)
        for g in state.ids
        if state.model.geom_type[g] not in OPEN_SURFACE_GEOMS
    ]
    uncovered = uncovered_components(state.query, backend)
    if unsupported or uncovered:
        return begin, {
            "backend_error": (
                f"open-scan audit cannot certify geoms {unsupported} / "
                f"components {uncovered}"
            )
        }

    depths = np.zeros((end - begin, state.model.ngeom))
    uncertain = 0
    worst_exit, worst_depth = np.zeros(3), 0.0
    for frame, pose in enumerate(state.poses[begin:end]):
        state.data.qpos[:] = pose
        mujoco.mj_forward(state.model, state.data)
        for row in backend.query(state.data, 0.0):
            uncertain += bool(row["uncertain"])
            depth = max(-float(row["clearance"]), 0.0)
            gid = int(row["geom_ids"])
            depths[frame, gid] = max(depths[frame, gid], depth)
            if depth > worst_depth:
                worst_depth, worst_exit = depth, depth * np.asarray(row["directions"])
    return begin, {
        "depths": depths,
        "uncertain_samples": uncertain,
        "worst_depth": worst_depth,
        "worst_exit": worst_exit,
        "backend_failures": list(backend.failures),
    }


def _static_batch(task):
    if _STATIC_STATE.open_scan:
        return _open_static_batch(task)
    return _normal_static_batch(task)


def _run_batch_tasks(worker, tasks, workers):
    global _WORKER_THREADS
    # libigl's AABB/winding queries parallelize internally. Constraining all
    # eight workers to one native thread made each batch about eight times
    # slower; give every worker an equal share of the available cores instead.
    _WORKER_THREADS = max(1, len(os.sched_getaffinity(0)) // workers)
    if workers == 1:
        yield from (worker(task) for task in tasks)
        return
    # Fork inherits prepared.npy-derived geometry copy-on-write. Tasks carry
    # only frame indices; result arrays are the only worker-to-parent payload.
    context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=context, initializer=worker_initializer
    ) as executor:
        yield from executor.map(worker, tasks, chunksize=1)


def worker_initializer():
    global _THREAD_LIMIT
    # The audit is process-parallel; cap native libraries to the selected share.
    from threadpoolctl import threadpool_limits

    _THREAD_LIMIT = threadpool_limits(limits=_WORKER_THREADS)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    if _STATIC_STATE is not None:
        _init_static_worker()
    if _MOVING_STATE is not None:
        _init_moving_worker()


def _aggregate_static(ordered, *, model, poses, open_scan):
    global _STATIC_STATE
    depths = np.zeros((len(poses), model.ngeom))
    uncertain = 0
    worst_exit, worst_depth = np.zeros(3), 0.0
    worker_seconds = {}
    worker_batches = {}
    query_seconds = query_cpu_seconds = forward_seconds = 0.0
    try:
        for begin, result in ordered:
            if result.get("backend_error") is not None:
                raise RuntimeError(result["backend_error"])
            if result.get("backend_failures"):
                raise RuntimeError(
                    "open-scan audit query failures: "
                    f"{result['backend_failures'][:3]}"
                )
            depths[begin : begin + len(result["depths"])] = result["depths"]
            uncertain += int(result["uncertain_samples"])
            if result["worst_depth"] > worst_depth:
                worst_depth = result["worst_depth"]
                worst_exit = result["worst_exit"]
            if open_scan:
                continue
            print(
                f"static {begin + len(result['depths'])}/{len(poses)} "
                f"max={depths.max():.6f} m",
                flush=True,
            )
            worker = result["worker"]
            worker_seconds[worker] = worker_seconds.get(worker, 0.0) + (
                result["forward_seconds"] + result["query_seconds"]
            )
            worker_batches[worker] = worker_batches.get(worker, 0) + 1
            query_seconds += result["query_seconds"]
            query_cpu_seconds += result["query_cpu_seconds"]
            forward_seconds += result["forward_seconds"]
    finally:
        _STATIC_STATE = None
    if not open_scan:
        print(
            "timing static_workers="
            f"batches={sorted(worker_batches.values())} "
            f"parallel_seconds={max(worker_seconds.values(), default=0):.3f} "
            f"sum_seconds={sum(worker_seconds.values()):.3f} "
            f"forward_seconds={forward_seconds:.3f} "
            f"query_seconds={query_seconds:.3f} "
            f"query_cpu_seconds={query_cpu_seconds:.3f}",
            flush=True,
        )
    return depths, uncertain, worst_exit


def run_static_batches(
    *,
    model,
    query,
    poses,
    ids,
    offsets,
    owners,
    batch,
    workers,
    open_scan,
    prepare_moving=None,
):
    global _STATIC_STATE, _WORKER_THREADS
    _WORKER_THREADS = 1
    _STATIC_STATE = _StaticWorkerState(
        model=model,
        query=query,
        poses=poses,
        ids=ids,
        offsets=offsets,
        owners=owners,
        open_scan=open_scan,
    )
    _init_static_worker()
    tasks = _batch_ranges(len(poses), batch)

    def submit(executor, begin, end):
        return executor.submit(_static_batch, (begin, end))

    if workers == 1 or open_scan:
        ordered = [_static_batch((begin, end)) for begin, end in tasks]
    else:
        cores = len(os.sched_getaffinity(0))
        _WORKER_THREADS = max(1, cores // workers)
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("fork"),
            initializer=worker_initializer,
        ) as executor:
            futures = {}
            for begin, end in tasks:
                futures[begin] = submit(executor, begin, end)
            # Workers fork at submission; the parent now builds the moving
            # query while they run. Neither result depends on this order.
            if prepare_moving is not None:
                prepare_moving()
            ordered = [futures[begin].result() for begin in sorted(futures)]

    # Open scans always run serially through one backend, whose final
    # ``describe()`` is the original report (counters and test description).
    open_backend = _STATIC_STATE.backend
    depths, uncertain, worst_exit = _aggregate_static(
        ordered, model=model, poses=poses, open_scan=open_scan
    )
    return depths, uncertain, worst_exit, (
        None if open_backend is None else open_backend.describe())


def _moving_batch(task):
    begin, end = task
    state = _MOVING_STATE
    query = state.query
    query_count_before = query.query_count
    query_started = time.perf_counter()
    depths = np.zeros((end - begin, state.scene.model.ngeom))
    failures = []
    worst_exit, worst_depth = np.zeros(3), 0.0
    for local_frame, pose in enumerate(state.poses[begin:end]):
        frame = begin + local_frame
        result = query.query(pose, frame)
        depths[local_frame] = np.maximum(
            result["forward_depth"], result["reverse_depth"]
        ).max(axis=0)
        failures.extend((frame + state.start, message) for message in result["failures"])
        for row in result["rows"]:
            if -row["clearance"] > worst_depth:
                worst_depth = -row["clearance"]
                worst_exit = worst_depth * row["direction"]
    query_seconds = time.perf_counter() - query_started
    return begin, {
        "depths": depths,
        "failures": failures,
        "worst_depth": worst_depth,
        "worst_exit": worst_exit,
        "query_count": query.query_count - query_count_before,
        "worker": os.getpid(),
        "query_seconds": query_seconds,
    }


def run_moving_batches(*, scene, poses, start, batch, workers):
    global _MOVING_STATE
    _MOVING_STATE = _MovingWorkerState(
        scene=scene, poses=poses, start=start, batch=batch
    )
    _init_moving_worker()
    depths = np.zeros((len(poses), scene.model.ngeom))
    failures = []
    worst_exit, worst_depth = np.zeros(3), 0.0
    query_count = 0
    tasks = _batch_ranges(len(poses), batch)
    worker_seconds = {}
    worker_batches = {}
    query_seconds = 0.0
    try:
        for begin, result in _run_batch_tasks(_moving_batch, tasks, workers):
            depths[begin : begin + len(result["depths"])] = result["depths"]
            failures.extend(result["failures"])
            if result["worst_depth"] > worst_depth:
                worst_depth = result["worst_depth"]
                worst_exit = result["worst_exit"]
            query_count += int(result["query_count"])
            end = begin + len(result["depths"])
            if end % batch == 0:
                print(
                    f"moving {end}/{len(poses)} max={depths.max():.6f} m", flush=True
                )
            worker = result["worker"]
            worker_seconds[worker] = (
                worker_seconds.get(worker, 0.0) + result["query_seconds"]
            )
            worker_batches[worker] = worker_batches.get(worker, 0) + 1
            query_seconds += result["query_seconds"]
    finally:
        _MOVING_STATE = None
    print(
        "timing moving_workers="
        f"batches={sorted(worker_batches.values())} "
        f"parallel_seconds={max(worker_seconds.values(), default=0):.3f} "
        f"sum_seconds={sum(worker_seconds.values()):.3f} "
        f"query_seconds={query_seconds:.3f}",
        flush=True,
    )
    described = scene.describe()
    described["query_count"] = query_count
    return depths, failures, worst_exit, described
