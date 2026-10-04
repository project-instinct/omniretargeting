"""Unchanged-input repair benchmark with process/function timers.

Run ``python -m tests.instrument_gpu_repair --capture DIRECTORY CLI [ARGS]``
to capture a live pipeline objective. ``--replay DIRECTORY --mode cpu|gpu``
repeats one captured repair without the earlier stages. Run sequentially in
``systemd-run --user --scope -p MemoryMax=10G -p MemorySwapMax=0``.
All artifacts live in DIRECTORY; nothing in the production batch is changed.
"""

from __future__ import annotations

import argparse
import collections
import cProfile
import functools
import hashlib
import json
import multiprocessing
import os
import pickle
import runpy
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

_STATS = collections.Counter()
_SCOPE = threading.local()
_CONNECTION_SCOPES = {}
_STATS_LOCK = threading.Lock()
_DIRECTORY = None
_MODE = None
_PROFILE = None
_ORIGINAL_REPAIR = None


def _dump(role):
    target = _DIRECTORY / f"{_MODE}_{role}_{os.getpid()}"
    target.with_suffix(".json").write_text(json.dumps(dict(_STATS), indent=2))
    if _PROFILE is not None:
        _PROFILE.dump_stats(str(target.with_suffix(".pstats")))


def _wrap(owner, name, key, *, scope=None, trace=False):
    original = getattr(owner, name)

    @functools.wraps(original)
    def timed(*args, **kwargs):
        old_scope = getattr(_SCOPE, "value", "other_ipc")
        if scope is not None:
            _SCOPE.value = scope(args) if callable(scope) else scope
        current_scope = getattr(_SCOPE, "value", "other_ipc")
        label = key(current_scope) if callable(key) else key
        previous_gpu = getattr(args[0], "gpu_stats", None) if key == "gpu_prepare" else None
        previous_affinity = None
        if key in ("solve_step", "restore_repair") and "WS2_PROFILE_SOLVER_CORE" in os.environ:
            # Control P/E-core migration for the unrelated solver phases only.
            # Geometry and its CPU/CUDA workers retain their original affinity.
            previous_affinity = os.sched_getaffinity(0)
            os.sched_setaffinity(0, {int(os.environ["WS2_PROFILE_SOLVER_CORE"])})
        start, cpu_start = time.perf_counter(), time.process_time()
        try:
            value = original(*args, **kwargs)
        except Exception:
            if key == "resident_gpu":
                traceback.print_exc()
            raise
        finally:
            elapsed, cpu_elapsed = time.perf_counter() - start, time.process_time() - cpu_start
            if previous_affinity is not None:
                os.sched_setaffinity(0, previous_affinity)
            with _STATS_LOCK:
                _STATS[label + "_seconds"] += elapsed
                _STATS[label + "_cpu_seconds"] += cpu_elapsed
                _STATS[label + "_calls"] += 1
            _SCOPE.value = old_scope
        if key == "gpu_prepare":
            current_gpu = getattr(args[0], "gpu_stats", None)
            args[0]._ws2_profile_gpu_new = current_gpu is not None and current_gpu is not previous_gpu
            if current_gpu is not None and current_gpu is not previous_gpu:
                for field in ("scan_seconds", "center_seconds", "setup_seconds", "sphere_culled_geoms"):
                    _STATS["gpu_" + field] += current_gpu.get(field, 0.)
        if key == "measure" and getattr(args[0], "_ws2_profile_gpu_new", False):
            for field in ("seconds", "points", "candidates"):
                _STATS["gpu_" + field] += args[0].gpu_stats.get(field, 0.)
            args[0]._ws2_profile_gpu_new = False
        if trace:
            tick = time.perf_counter()
            with (_DIRECTORY / f"{_MODE}_measurements.pkl").open("ab") as output:
                pickle.dump((args[1], kwargs, value), output, protocol=5)
            _STATS["trace_seconds"] += time.perf_counter() - tick
            (_DIRECTORY / f"{_MODE}_live.json").write_text(json.dumps(dict(_STATS)))
        return value

    setattr(owner, name, timed)


def _child_wrapper(owner, name, role):
    original = getattr(owner, name)

    @functools.wraps(original)
    def profiled(*args, **kwargs):
        global _STATS, _PROFILE
        if _PROFILE is not None:
            _PROFILE.disable()
        _STATS = collections.Counter()
        _PROFILE = cProfile.Profile()
        _PROFILE.enable()
        try:
            return original(*args, **kwargs)
        finally:
            _PROFILE.disable()
            _dump(role)

    setattr(owner, name, profiled)


def _install(directory, mode, *, server=False):
    global _DIRECTORY, _MODE, _STATS, _PROFILE
    _DIRECTORY, _MODE = Path(directory), mode
    _STATS = collections.Counter()
    import omniretargeting._fast.mesh_repair as mr
    import omniretargeting._fast.parallel_geometry as parallel
    parallel.WORKERS = int(os.environ.get("WS2_PROFILE_WORKERS", "8"))
    import omniretargeting._fast.gpu_geometry as gpu
    import omniretargeting._fast.speculation as speculation
    from omniretargeting._fast.repair_limits import RepairLimits
    from omniretargeting.moving_scene import MovingSceneQuery
    from omniretargeting.scene_geometry import SceneQuery
    from omniretargeting.self_collision import SelfCollisionQuery
    from multiprocessing.connection import Connection, _ForkingPickler

    for owner, name, key in (
        (mr.OriginalGeometry, "__init__", "geometry_init"),
        (mr.OriginalGeometry, "measure_frames", "measure_frames"),
        (mr.OriginalGeometry, "linearize", "linearize"),
        (mr.OriginalGeometry, "_prepare_gpu_static", "gpu_prepare"),
        (mr.OriginalGeometry, "_prepare_gpu_moving", "gpu_moving_prepare"),
        (mr.OriginalGeometry, "_query_static_batch", "static_reduce"),
        (mr.OriginalGeometry, "start_workers", "worker_start"),
        (mr.OriginalGeometry, "stop_workers", "worker_stop"),
        (mr, "repair_original_meshes", "repair_total"),
        (mr, "solve_step", "solve_step"),
        (mr, "restore_repair", "restore_repair"),
        (mr, "_joint_projection", "projection"),
        (RepairLimits, "__init__", "limits_init"),
        (RepairLimits, "constraints", "limits_constraints"),
        (RepairLimits, "check", "limits_check"),
        (SceneQuery, "query", "exact_scene"),
        (MovingSceneQuery, "query", "exact_moving"),
        (SelfCollisionQuery, "query", "exact_self"),
        (gpu, "complete_static", "exact_static_completion"),
        (gpu.ResidentScene, "candidates", "resident_gpu"),
    ):
        _wrap(owner, name, key)
    _wrap(mr.OriginalGeometry, "measure", "measure", trace=True)
    def worker_scope(args):
        for _, connection in args[0].workers:
            _CONNECTION_SCOPES[connection.fileno()] = "worker_ipc"
        return "worker_ipc"

    def cuda_scope(args):
        _CONNECTION_SCOPES[args[0].connection.fileno()] = "cuda_ipc"
        return "cuda_ipc"

    def connection_scope(args):
        return _CONNECTION_SCOPES.get(args[0].fileno(), getattr(_SCOPE, "value", "other_ipc"))

    _wrap(parallel.ParallelMeasure, "measure", "parallel_measure", scope=worker_scope)
    _wrap(parallel.ParallelMeasure, "prefetch_static", "static_prefetch", scope=worker_scope)
    _wrap(gpu.GeometryService, "request", "cuda_request", scope=cuda_scope)
    for name in ("send", "recv"):
        _wrap(Connection, name, lambda scope, name=name: scope + "_" + name, scope=connection_scope)
    for name in ("dumps", "loads"):
        _wrap(_ForkingPickler, name, lambda scope, name=name: scope + "_pickle_" + name)
    _child_wrapper(parallel, "_serve", "worker")
    _child_wrapper(speculation, "_serve", "restoration")
    if not server:
        original_popen = subprocess.Popen

        def server_popen(command, *args, **kwargs):
            if isinstance(command, list) and command[1:3] == ["-m", gpu.__name__]:
                command = [command[0], "-m", "tests.instrument_gpu_repair", "--server-fd", command[-1],
                           "--directory", str(directory), "--mode", mode]
            return original_popen(command, *args, **kwargs)

        gpu.subprocess.Popen = server_popen
    _PROFILE = cProfile.Profile()
    _PROFILE.enable()
    if not server:
        hashes = {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                  for name, module in list(sys.modules.items())
                  if name.startswith("omniretargeting.") and getattr(module, "__file__", "").endswith(".py")}
        (_DIRECTORY / f"{mode}_code_hashes.json").write_text(json.dumps(hashes, indent=2))


def _repair_worker(mode, objective, poses, objects, scene, directory, ready, start):
    os.environ["OMNIRETARGETING_GPU_GEOMETRY"] = "0" if mode == "cpu" else "1"
    import omniretargeting._fast.mesh_repair as mr
    if _ORIGINAL_REPAIR is not None:
        mr.repair_original_meshes = _ORIGINAL_REPAIR
    _install(directory, mode)
    ready.set()
    start.wait()
    try:
        tick = time.perf_counter()
        result, report = mr.repair_original_meshes(objective, poses, objects, scene)
        elapsed = time.perf_counter() - tick
        np.savez(Path(directory) / f"{mode}_final.npz", qpos=result)
        (Path(directory) / f"{mode}_repair.json").write_text(json.dumps(report, indent=2, default=float))
        print(mode, "REPAIR", elapsed, flush=True)
    finally:
        from omniretargeting._fast import gpu_geometry
        if gpu_geometry._SERVICE is not None:
            gpu_geometry._SERVICE.close()
        _PROFILE.disable()
        _dump("parent")


def compare_repairs(directory):
    directory = Path(directory)
    from tests.test_gpu_geometry import assert_identical, assert_repair_equivalent
    cpu = np.load(directory / "cpu_final.npz")["qpos"]
    gpu = np.load(directory / "gpu_final.npz")["qpos"]
    assert_identical(gpu, cpu, "qpos")
    a = json.loads((directory / "cpu_repair.json").read_text())
    b = json.loads((directory / "gpu_repair.json").read_text())
    diagnostics = assert_repair_equivalent(b, a)
    with (directory / "cpu_measurements.pkl").open("rb") as cpu_trace, \
            (directory / "gpu_measurements.pkl").open("rb") as gpu_trace:
        complete, rejected = 0, 0
        while True:
            try:
                before = pickle.load(cpu_trace)
            except EOFError:
                assert not gpu_trace.read(1), "Extra GPU measurement"
                break
            after = pickle.load(gpu_trace)
            assert_identical(after[:2], before[:2], "measurement inputs")
            assert before[2]["complete"] == after[2]["complete"], "accept/reject decision"
            if before[2]["complete"]:
                assert_identical(after[2], before[2], "accepted measurement")
                complete += 1
            else:
                rejected += 1
        print("EQUIVALENT measurements", complete, "complete", rejected, "rejected", flush=True)
    print("EQUIVALENT qpos and repair", json.dumps(diagnostics), flush=True)
    return cpu, a


def summarize(directory):
    directory = Path(directory)
    result = {}
    for mode in ("cpu", "gpu"):
        parents = list(directory.glob(f"{mode}_parent_*.json"))
        assert len(parents) == 1, parents
        parent = json.loads(parents[0].read_text())
        workers = collections.Counter()
        for filename in directory.glob(f"{mode}_worker_*.json"):
            workers.update(json.loads(filename.read_text()))
        servers = collections.Counter()
        for filename in directory.glob(f"{mode}_cuda_*.json"):
            servers.update(json.loads(filename.read_text()))
        accounted = sum(parent.get(key + "_seconds", 0.) for key in (
            "measure", "solve_step", "restore_repair", "linearize", "geometry_init",
            "worker_start", "worker_stop", "limits_init", "trace"))
        parent["other_repair_seconds"] = parent["repair_total_seconds"] - accounted
        result[mode] = dict(parent=parent, workers=workers, cuda=servers)
        print(mode, "parent wall", {k: round(v, 3) for k, v in parent.items()
                                    if k.endswith("_seconds") and not k.endswith("_cpu_seconds")})
        print(mode, "workers summed wall", {k: round(v, 3) for k, v in workers.items()
                                             if k.endswith("_seconds") and not k.endswith("_cpu_seconds")})
        print(mode, "CUDA interpreter wall", {k: round(v, 3) for k, v in servers.items()
                                              if k.endswith("_seconds") and not k.endswith("_cpu_seconds")})
    print("REPAIR SPEEDUP", result["cpu"]["parent"]["repair_total_seconds"]
          / result["gpu"]["parent"]["repair_total_seconds"])
    (directory / "timing_split.json").write_text(json.dumps(result, indent=2))


def replay_measurements(args):
    """Audit every captured repair decision without repeating the conic solves."""
    from omniretargeting._fast import mesh_repair, parallel_geometry
    from omniretargeting._fast import gpu_geometry
    from omniretargeting.scene_geometry import SceneQuery
    from tests.test_gpu_geometry import assert_identical

    with (Path(args.replay) / "inputs.pkl").open("rb") as handle:
        objective, _, objects, mesh, enclosed = pickle.load(handle)
    objective.kin.cache.clear()
    mesh._cache.clear()
    os.environ["OMNIRETARGETING_GPU_GEOMETRY"] = "0" if args.mode == "cpu" else "1"
    parallel_geometry.WORKERS = args.workers
    geometry = mesh_repair.OriginalGeometry(objective, objects,
                                            SceneQuery(mesh, enclosed_free_space=enclosed), manifold=True)
    geometry.start_workers(objective.frames)
    complete, rejected, elapsed = 0, 0, 0.
    started = time.perf_counter()
    try:
        with Path(args.measurements).open("rb") as trace:
            while True:
                try:
                    poses, kwargs, expected = pickle.load(trace)
                except EOFError:
                    break
                tick = time.perf_counter()
                actual = geometry.measure(poses, **kwargs)
                elapsed += time.perf_counter() - tick
                assert actual["complete"] == expected["complete"], "accept/reject decision"
                if expected["complete"]:
                    assert_identical(actual, expected, "accepted measurement")
                    complete += 1
                else:
                    rejected += 1
                print("MEASUREMENT", complete + rejected, "complete", actual["complete"],
                      "elapsed", elapsed, getattr(geometry, "gpu_moving_stats", {}), flush=True)
    finally:
        geometry.stop_workers()
        if gpu_geometry._SERVICE is not None:
            gpu_geometry._SERVICE.close()
    print("EQUIVALENT", complete, "complete", rejected, "rejected",
          "measurement seconds", elapsed, "audit seconds", time.perf_counter() - started,
          "moving audit", getattr(geometry, "gpu_moving_audit", {}), flush=True)


def main():
    global _ORIGINAL_REPAIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture")
    parser.add_argument("--replay")
    parser.add_argument("--server-fd", type=int)
    parser.add_argument("--directory")
    parser.add_argument("--mode")
    parser.add_argument("--summarize")
    parser.add_argument("--compare")
    parser.add_argument("--measurements")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--solver-core", type=int)
    args, command = parser.parse_known_args()
    if args.measurements:
        replay_measurements(args)
    elif args.compare:
        compare_repairs(args.compare)
    elif args.summarize:
        summarize(args.summarize)
    elif args.server_fd is not None:
        from multiprocessing.connection import Connection
        _install(args.directory, args.mode, server=True)
        from omniretargeting._fast.gpu_geometry import _gpu_server
        try:
            _gpu_server(Connection(args.server_fd))
        finally:
            _PROFILE.disable()
            _dump("cuda")
    elif args.replay:
        assert args.mode in ("cpu", "gpu"), "Replay one mode at a time"
        os.environ["WS2_PROFILE_WORKERS"] = str(args.workers)
        if args.solver_core is not None:
            assert args.solver_core in os.sched_getaffinity(0)
            os.environ["WS2_PROFILE_SOLVER_CORE"] = str(args.solver_core)
        from omniretargeting.scene_geometry import SceneQuery
        with (Path(args.replay) / "inputs.pkl").open("rb") as handle:
            objective, poses, objects, mesh, enclosed = pickle.load(handle)
        objective.kin.cache.clear()
        mesh._cache.clear()
        scene = SceneQuery(mesh, enclosed_free_space=enclosed)
        directory = Path(args.directory or args.replay)
        directory.mkdir(parents=True, exist_ok=True)
        assert not (directory / f"{args.mode}_measurements.pkl").exists(), "Use fresh outputs"
        event = multiprocessing.get_context("fork").Event()
        event.set()
        _repair_worker(args.mode, objective, poses, objects, scene, directory, event, event)
    else:
        import omniretargeting._fast.mesh_repair as mr
        _ORIGINAL_REPAIR = mr.repair_original_meshes
        directory = Path(args.capture)
        directory.mkdir(parents=True, exist_ok=True)

        def capture(objective, poses, scene_objects=(), scene_query=None):
            try:
                with (directory / "inputs.pkl").open("wb") as handle:
                    pickle.dump((objective, poses, scene_objects, scene_query.mesh,
                                 scene_query.enclosed_free_space), handle, protocol=5)
                print("CAPTURED repair inputs", flush=True)
            except Exception as error:
                print("CAPTURE ERROR", repr(error), flush=True)
            return _ORIGINAL_REPAIR(objective, poses, scene_objects, scene_query)

        mr.repair_original_meshes = capture
        sys.argv = command
        runpy.run_path(command[0], run_name="__main__")


if __name__ == "__main__":
    main()
