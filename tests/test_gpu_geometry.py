"""GPU geometry equivalence checks and saved-pose measurement benchmark.

Offline harness (no pipeline integration)::

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m \
        tests.test_gpu_geometry RUN_DIRECTORY --parallel --save /tmp/ws2_cpu.pkl
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import igl
import numpy as np
import pytest
import trimesh

from omniretargeting._fast.mesh_repair import OriginalGeometry
from omniretargeting._fast.parallel_geometry import merge_frames
from omniretargeting.scene_geometry import SceneQuery, collision_geom_ids, geom_surface_offsets
from omniretargeting.sequence_retargeting import sequence_config


def assert_identical(actual, expected, path="result"):
    """Compare every field, including array dtype, signed zero and NaN bits."""
    assert type(actual) is type(expected), path
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            assert_identical(actual[key], expected[key], f"{path}[{key!r}]")
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected), path
        for index, (a, b) in enumerate(zip(actual, expected)):
            assert_identical(a, b, f"{path}[{index}]")
    elif isinstance(expected, np.ndarray):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype, path
        assert actual.tobytes() == expected.tobytes(), path
    elif isinstance(expected, (float, np.floating)):
        assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes(), path
    else:
        assert actual == expected, path


class _TimedQuery:
    def __init__(self, query, stats, name):
        self.wrapped, self.stats, self.name = query, stats, name

    def __getattr__(self, name):
        return getattr(self.wrapped, name)

    def query(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return self.wrapped.query(*args, **kwargs)
        finally:
            self.stats[self.name] = self.stats.get(self.name, 0.) + time.perf_counter() - started
            self.stats[self.name + "_calls"] = self.stats.get(self.name + "_calls", 0) + 1


class _ProfileGeometry(OriginalGeometry):
    def measure_frames(self, *args, **kwargs):
        stats = {}
        scene, moving, self_query = self.scene, self.moving, self.self_query
        winding, exact_winding = igl.fast_winding_number, igl.winding_number

        def timed_winding(*args):
            start = time.perf_counter()
            try:
                return winding(*args)
            finally:
                stats["winding"] = stats.get("winding", 0.) + time.perf_counter() - start

        def timed_exact_winding(*args):
            start = time.perf_counter()
            try:
                return exact_winding(*args)
            finally:
                stats["exact_winding"] = stats.get("exact_winding", 0.) + time.perf_counter() - start

        igl.fast_winding_number, igl.winding_number = timed_winding, timed_exact_winding
        if scene is not None:
            self.scene = _TimedQuery(scene, stats, "static")
        if moving is not None:
            self.moving = _TimedQuery(moving, stats, "moving")
        self.self_query = _TimedQuery(self_query, stats, "self")
        started = time.perf_counter()
        try:
            result = super().measure_frames(*args, **kwargs)
            result["profile"] = {**stats, "total": time.perf_counter() - started}
            return result
        finally:
            self.scene, self.moving, self.self_query = scene, moving, self_query
            igl.fast_winding_number, igl.winding_number = winding, exact_winding


class _PrototypeGeometry(_ProfileGeometry):
    def _query_static_batch(self, pending):
        begins = np.r_[0, np.cumsum([len(points) for _, points, _ in pending])]
        groups = list(zip(begins[:-1], begins[1:]))
        if getattr(self, "defer_exact", False):
            points = np.concatenate([p for _, p, _ in pending])
            selected = self.prefilter.candidates(points, groups)
            for (summary, points, ids), (begin, end) in zip(pending, groups):
                local = selected[(selected >= begin) & (selected < end)] - begin
                if len(local):
                    self.pending_exact.append((summary, points[local], ids[local]))
            return
        original = self.scene
        self.scene = SimpleNamespace(query=lambda points: self.prefilter.query_reduction(points, groups))
        try:
            super()._query_static_batch(pending)
        finally:
            self.scene = original

    def flush_exact(self):
        pending, count = [], 0
        for summary, points, ids in self.pending_exact:
            begin = 0
            while begin < len(points):
                take = min(len(points) - begin, self.SCENE_BATCH_POINTS - count)
                pending.append((summary, points[begin:begin + take], ids[begin:begin + take]))
                begin, count = begin + take, count + take
                if count == self.SCENE_BATCH_POINTS:
                    OriginalGeometry._query_static_batch(self, pending)
                    pending, count = [], 0
        if pending:
            OriginalGeometry._query_static_batch(self, pending)
        self.pending_exact.clear()
        self.defer_exact = False


def load_saved_run(directory, limit=None):
    """Rebuild precisely the geometry-only portion of the saved objective."""
    directory = Path(directory)
    prepared = np.load(directory / "prepared.npy", allow_pickle=True).item()
    poses = np.load(directory / "before_mesh_repair.npz")["qpos"]
    if limit is not None:
        poses = poses[:limit]
    model = mujoco.MjModel.from_xml_path(prepared["model_path"])
    config = sequence_config(prepared["captured_solver_config"])
    geoms = collision_geom_ids(model)
    offsets = [geom_surface_offsets(model, int(g), config["collision_spacing_m"]) for g in geoms]
    objective = SimpleNamespace(
        model=model, frames=len(poses),
        coords=SimpleNamespace(qi=np.arange(7, model.nq), di=np.arange(6, model.nv)),
        shell_geoms=geoms,
        shell_sample_geom=np.concatenate([np.full(len(o), g, int) for g, o in zip(geoms, offsets)]),
        shell_local=np.concatenate(offsets),
    )
    scene = SceneQuery.from_tasks(
        trimesh.load_mesh(directory / "static_scene.obj", process=False), prepared["tasks"],
        enclosed_free_space=prepared.get("scene_enclosed_free_space", False))
    objects = np.load(directory / "objects.npy", allow_pickle=True).tolist()
    if limit is not None:
        objects = [{k: v[:limit] if k in ("translations", "rotations", "scales") else v
                    for k, v in spec.items()} for spec in objects]
    return objective, poses, objects, scene


def benchmark_cpu(geometry, poses, parallel=False, jacobians=False):
    started = time.perf_counter()
    if parallel:
        geometry.start_workers(len(poses))
    try:
        if getattr(geometry, "parallel", None) is not None and geometry._gpu_allowed:
            geometry.parallel.prefetch_self(poses)
        gpu_started = time.perf_counter()
        static_updates = geometry._prepare_gpu_static(poses)
        gpu_prepare = time.perf_counter() - gpu_started
        parallel_worker = getattr(geometry, "parallel", None)
        if (parallel_worker is not None and isinstance(static_updates, list)
                and os.environ.get("OMNIRETARGETING_GPU_GEOMETRY_MOVING", "0") != "0"
                and os.environ.get("OMNIRETARGETING_GPU_GEOMETRY_OVERLAP", "0") != "0"):
            parallel_worker.prefetch_static(static_updates)
        moving_started = time.perf_counter()
        moving_updates = geometry._prepare_gpu_moving(poses)
        moving_prepare = time.perf_counter() - moving_started
        if getattr(geometry, "parallel", None) is not None:
            parts = geometry.parallel.measure(poses, jacobians, None, static_updates=static_updates,
                                               moving_updates=moving_updates)
            if geometry.moving is not None:
                geometry.moving.query_count = geometry.parallel.query_count
        else:
            parts = [geometry.measure_frames(poses, range(len(poses)), jacobians=jacobians)]
        measurement = merge_frames(parts)
        profile = {}
        for part in parts:
            for key, value in part["profile"].items():
                profile[key] = profile.get(key, 0) + value
        profile["wall"] = time.perf_counter() - started
        profile["gpu_prepare_wall"] = gpu_prepare
        profile["gpu_moving_prepare_wall"] = moving_prepare
        profile["gpu_stats"] = getattr(geometry, "gpu_stats", {})
        profile["gpu_unavailable"] = getattr(geometry, "gpu_unavailable", None)
        profile["workers"] = len(parts)
        profile["frames"] = len(poses)
        profile["per_frame_ms"] = {k: 1000 * profile.get(k, 0.) / len(poses)
                                   for k in ("static", "moving", "self", "total")}
        return measurement, geometry.summary(measurement), profile
    finally:
        geometry.stop_workers()


def calibrate(objective, poses, scene, samples=1024):
    """Compare actual shell positions over every saved frame with float64 CPU."""
    from omniretargeting._fast.gpu_geometry import WarpScenePrefilter

    selected = np.unique(np.linspace(0, len(objective.shell_local) - 1, samples).astype(int))
    gids = objective.shell_sample_geom[selected]
    data = mujoco.MjData(objective.model)
    points = []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_forward(objective.model, data)
        points.append(data.geom_xpos[gids] + np.einsum(
            "nij,nj->ni", data.geom_xmat[gids].reshape(-1, 3, 3), objective.shell_local[selected]))
    points = np.concatenate(points)
    prefilter = WarpScenePrefilter(scene)
    distance, _, winding = prefilter.raw(points)
    start = time.perf_counter()
    cpu = scene.query(points)
    cpu_seconds = time.perf_counter() - start
    sign = np.where(np.abs(winding) > .5, -1., 1.)
    if scene.enclosed_free_space:
        sign = -sign
    phi = sign * distance
    error = np.abs(distance - cpu["unsigned_distance"])
    report = dict(points=len(points), gpu_seconds=prefilter.stats["gpu_seconds"],
                  cpu_seconds=cpu_seconds, max_unsigned_error=float(error.max()),
                  sign_disagreements=int(np.count_nonzero((phi < 0) != (cpu["clearance"] < 0))),
                  cpu_unknown=int(cpu["uncertain"].sum()), slivers=prefilter.sliver_count,
                  thresholds={})
    for threshold in [-.01, -.001, -.0005, -5.1e-5, -1e-6, 0., .008]:
        safe = np.abs(phi - threshold) > prefilter.eps
        mismatch = safe & ((phi < threshold) != (cpu["clearance"] < threshold))
        report["thresholds"][str(threshold)] = int(mismatch.sum())
    print("CALIBRATION", json.dumps(report), flush=True)


def assert_repair_equivalent(actual, expected, path="repair"):
    """Exact accepted results; flag the existing worker-race diagnostics.

    A rejected prefix stops workers at whichever rejection finishes first.
    Its frame, partial failures/regressions, and cumulative query count are
    documented diagnostics in parallel_geometry, not accepted measurements.
    Keep comparing status, alpha and every other field of those records.
    """
    diagnostics = []
    partial = {"rejected_geometry_prefix_frame", "query_failures",
               "collision_regressions", "regression_examples"}

    def compare(a, b, parts, label):
        assert type(a) is type(b), label
        if isinstance(b, dict):
            ignored = {"seconds"}
            if parts == ("after", "moving_query"):
                ignored.add("query_count")
            if (len(parts) == 2 and parts[0] == "history"
                    and ("rejected_geometry_prefix_frame" in a
                         or "rejected_geometry_prefix_frame" in b)):
                ignored.update(partial)
            for key in sorted(ignored - {"seconds"}):
                if key not in a and key not in b:
                    continue
                try:
                    assert key in a and key in b
                    assert_identical(a[key], b[key], label)
                except AssertionError:
                    diagnostics.append(dict(path=f"{label}[{key!r}]",
                                            cpu=b.get(key), gpu=a.get(key)))
            assert a.keys() - ignored == b.keys() - ignored, label
            for key in b:
                if key not in ignored:
                    compare(a[key], b[key], (*parts, key), f"{label}[{key!r}]")
        elif isinstance(b, (tuple, list)):
            assert len(a) == len(b), label
            for index, (x, y) in enumerate(zip(a, b)):
                compare(x, y, (*parts, index), f"{label}[{index}]")
        else:
            assert_identical(a, b, label)

    compare(actual, expected, (), path)
    return diagnostics


def compare_pipeline_runs(cpu_directory, gpu_directory):
    """Strict saved-run comparison with separately reported race diagnostics."""
    cpu_directory, gpu_directory = Path(cpu_directory), Path(gpu_directory)
    for filename in ("before_mesh_repair.npz", "final_retargeted.npz"):
        cpu = np.load(cpu_directory / filename)["qpos"]
        gpu = np.load(gpu_directory / filename)["qpos"]
        print(filename, "identical", cpu.tobytes() == gpu.tobytes(),
              "max_difference", float(np.max(np.abs(cpu - gpu))), flush=True)
        assert_identical(gpu, cpu, filename)

    cpu = json.loads((cpu_directory / "fast_report.json").read_text())
    gpu = json.loads((gpu_directory / "fast_report.json").read_text())
    for field in ("final_residuals", "source_shape", "strict_continuity_accepted",
                  "layout_and_relaxed_chatter_accepted", "windowed_continuity_accepted",
                  "layout_continuity_accepted", "acceptance_status"):
        assert_identical(gpu[field], cpu[field], field)
    for stage in cpu["stages"]:
        if "original_mesh_repair" not in cpu["stages"][stage]:
            continue
        a = cpu["stages"][stage]["original_mesh_repair"]
        b = gpu["stages"][stage]["original_mesh_repair"]
        print("repair_seconds", stage, "cpu", a["seconds"], "gpu", b["seconds"], flush=True)
        for diagnostic in assert_repair_equivalent(b, a, f"repair[{stage}]"):
            print("RACE_DIAGNOSTIC", json.dumps(diagnostic), flush=True)
    print("IDENTICAL final qpos, quality fields, accepted measurements, decisions and repair history; "
          "timing and flagged partial diagnostics excluded", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--jacobians", action="store_true")
    parser.add_argument("--save")
    parser.add_argument("--compare")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--cpu-sign", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--batch-exact", action="store_true")
    parser.add_argument("--repeat", action="store_true")
    parser.add_argument("--compare-run")
    args = parser.parse_args()
    if args.compare_run:
        compare_pipeline_runs(args.directory, args.compare_run)
        return
    started = time.perf_counter()
    objective, poses, objects, scene = load_saved_run(args.directory, args.limit)
    print("LOAD", time.perf_counter() - started, "samples/frame", len(objective.shell_local),
          "scene", scene.describe(), flush=True)
    if args.calibrate:
        calibrate(objective, poses, scene)
        return
    cls = _PrototypeGeometry if args.gpu else _ProfileGeometry
    geometry = cls(objective, objects, scene, manifold=True)
    print("INIT", time.perf_counter() - started, "moving", geometry.moving.describe(), flush=True)
    prepass = 0.
    if args.gpu:
        from omniretargeting._fast.gpu_geometry import WarpScenePrefilter
        build = time.perf_counter()
        geometry.prefilter = WarpScenePrefilter(scene, cpu_sign=args.cpu_sign)
        print("GPU_BUILD", time.perf_counter() - build, "slivers", geometry.prefilter.sliver_count, flush=True)
        prepass_start = time.perf_counter()
        geometry.defer_exact, geometry.pending_exact = args.batch_exact, []
        for index, _ in enumerate(geometry._static_frames(poses)):
            if index == 0:
                print("GPU_PREPASS", time.perf_counter() - prepass_start, geometry.prefilter.stats, flush=True)
        if args.batch_exact:
            exact_start = time.perf_counter()
            geometry.flush_exact()
            print("GPU_EXACT_BATCH", time.perf_counter() - exact_start, flush=True)
        prepass = time.perf_counter() - prepass_start
    measured, summary, profile = benchmark_cpu(geometry, poses, args.parallel, args.jacobians)
    if args.repeat:
        geometry._static_cache.clear()
        if geometry.moving is not None:
            geometry.moving = geometry.moving.fresh_state()
        measured, summary, profile = benchmark_cpu(geometry, poses, args.parallel, args.jacobians)
        print("WARM_REPEAT", flush=True)
    profile["static_prepass_wall"] = prepass
    profile["combined_wall"] = profile["wall"] + prepass
    print("PROFILE", json.dumps(profile), flush=True)
    print("RESULT", len(measured["rows"]), len(measured["depths"]), len(measured["failures"]),
          json.dumps({k: v for k, v in summary.items() if k != "residuals_over_1cm"}), flush=True)
    if args.compare:
        with open(args.compare, "rb") as handle:
            expected = pickle.load(handle)
        assert_identical(measured, expected["measurement"])
        assert_identical(summary, expected["summary"])
        print("IDENTICAL: every measurement and summary field", flush=True)
        # Force a real zero-protection regression: exact same partial frame,
        # rows and failure fields as the CPU path, independent of fork races.
        cpu_geometry = _ProfileGeometry(objective, objects, scene, manifold=True)
        cpu_geometry._gpu_allowed = False
        cpu_cut = cpu_geometry.measure(poses[:1], protected_depths={})
        gpu_cut = geometry.measure(poses[:1], protected_depths={})
        assert_identical(gpu_cut, cpu_cut, "rejection")
        print("IDENTICAL rejection", gpu_cut.get("rejection_frame"), flush=True)
    if args.save:
        with open(args.save, "wb") as handle:
            pickle.dump(dict(measurement=measured, summary=summary, profile=profile), handle)


@pytest.mark.parametrize("cpu_sign", [False, True])
def test_prefilter_keeps_exact_minima_threshold_rows_and_unknowns(cpu_sign):
    pytest.importorskip("warp")
    from omniretargeting._fast.gpu_geometry import WarpScenePrefilter

    scene = SceneQuery(trimesh.creation.box(extents=[2., 2., 2.]))
    # Include penetration, zero, the witness threshold and its neighbours,
    # plus exact equal minima. Every output witness must still be CPU data.
    x = np.r_[.7, 1., 1.008 - 1e-10, 1.008, 1.008 + 1e-10,
              1.1, 1.1, 1.3, 1.4]
    points = np.column_stack([x, np.zeros(len(x)), np.zeros(len(x))])
    prefilter = WarpScenePrefilter(scene, cpu_sign=cpu_sign, device="cuda:0")
    expected = scene.query(points)
    actual = prefilter.query_reduction(points, [(0, 5), (5, len(points))])
    for begin, end in [(0, 5), (5, len(points))]:
        winner = begin + np.argmin(expected["clearance"][begin:end])
        eligible = np.flatnonzero(expected["clearance"][begin:end] < .008) + begin
        for index in np.r_[winner, eligible]:
            for key in expected:
                np.testing.assert_array_equal(actual[key][index], expected[key][index])
    assert not actual["uncertain"].any()
    assert prefilter.stats["exact_points"] < len(points)


def test_open_scene_prefilter_uses_whole_exact_cpu_result():
    pytest.importorskip("warp")
    from omniretargeting._fast.gpu_geometry import WarpScenePrefilter

    mesh = trimesh.Trimesh(vertices=[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
                           faces=[[0, 1, 2]], process=False)
    scene = SceneQuery(mesh)
    points = np.array([[.2, .2, .1], [.1, .1, -1.], [10., 10., 10.]])
    prefilter = WarpScenePrefilter(scene)
    assert not prefilter.enabled
    assert_identical(prefilter.query_reduction(points, [(0, 3)]), scene.query(points))


def test_deferred_exact_batches_preserve_complete_original_geometry_summary():
    pytest.importorskip("warp")
    from omniretargeting._fast.gpu_geometry import WarpScenePrefilter
    from tests.test_original_geometry_batches import _problem, _scene

    objective, poses = _problem()
    scene = _scene("closed")
    expected_geometry = OriginalGeometry(objective, [], scene, manifold=True)
    actual_geometry = _PrototypeGeometry(objective, [], scene, manifold=True)
    actual_geometry.prefilter = WarpScenePrefilter(scene)
    actual_geometry.defer_exact, actual_geometry.pending_exact = True, []
    actual_geometry.SCENE_BATCH_POINTS = 19
    list(actual_geometry._static_frames(poses))
    actual_geometry.flush_exact()
    actual = actual_geometry.measure(poses, jacobians=True)
    expected = expected_geometry.measure(poses, jacobians=True)
    assert_identical(actual, expected)
    assert_identical(actual_geometry.summary(actual), expected_geometry.summary(expected))


@pytest.mark.parametrize("parallel,stream", [(False, False), (True, False), (True, True)])
def test_integrated_service_has_exact_rows_jacobians_and_debug_coverage(monkeypatch, parallel, stream):
    pytest.importorskip("warp")
    from tests.test_original_geometry_batches import _problem, _scene

    objective, poses = _problem()
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "0")
    expected = OriginalGeometry(objective, [], _scene("closed"), manifold=True).measure(poses, jacobians=True)
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "1")
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY_DEBUG", "1")
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY_STREAM", "1" if stream else "0")
    actual_geometry = OriginalGeometry(objective, [], _scene("closed"), manifold=True)
    if parallel:
        from omniretargeting._fast import parallel_geometry
        monkeypatch.setattr(parallel_geometry, "MIN_FRAMES_PER_WORKER", 2)
        monkeypatch.setattr(parallel_geometry, "WORKERS", 3)
        actual_geometry.start_workers(len(poses))
    try:
        actual = actual_geometry.measure(poses, jacobians=True)
        assert not hasattr(actual_geometry, "gpu_unavailable"), getattr(actual_geometry, "gpu_unavailable", None)
        assert hasattr(actual_geometry, "gpu_stats")
        assert_identical(actual, expected)
        assert_identical(actual_geometry.measure(poses, jacobians=True), expected)
    finally:
        actual_geometry.stop_workers()


def test_stream_service_failure_after_dispatch_keeps_complete_exact_measurement(monkeypatch):
    pytest.importorskip("warp")
    from omniretargeting._fast import gpu_geometry, parallel_geometry
    from tests.test_original_geometry_batches import _problem, _scene

    objective, poses = _problem()
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "0")
    expected = OriginalGeometry(objective, [], _scene("closed"), manifold=True).measure(poses, jacobians=True)
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "1")
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY_STREAM", "1")
    geometry = OriginalGeometry(objective, [], _scene("closed"), manifold=True)
    monkeypatch.setattr(parallel_geometry, "MIN_FRAMES_PER_WORKER", 2)
    monkeypatch.setattr(parallel_geometry, "WORKERS", 3)
    service = gpu_geometry.geometry_service()
    original, calls = service.candidates, []

    def lose_service(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise EOFError("CUDA interpreter disconnected")
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "candidates", lose_service)
    geometry.start_workers(len(poses))
    try:
        assert_identical(geometry.measure(poses, jacobians=True), expected)
        assert len(calls) == 2 and not geometry._gpu_allowed
        assert geometry.gpu_unavailable == "CUDA interpreter disconnected"
        assert_identical(geometry.measure(poses, jacobians=True), expected)
    finally:
        geometry.stop_workers()


@pytest.mark.parametrize("stream", [False, True])
def test_lazy_static_keeps_accepted_rows_and_recovers_from_partial_rejection(monkeypatch, stream):
    pytest.importorskip("warp")
    from omniretargeting._fast import parallel_geometry
    from tests.test_original_geometry_batches import _problem, _scene

    objective, poses = _problem()
    poses = np.tile(poses, (10, 1))
    objective.frames = len(poses)
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "0")
    reference = OriginalGeometry(objective, [], _scene("closed"), manifold=True)
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "1")
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY_STREAM", "1" if stream else "0")
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY_LAZY_STATIC", "1")
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY_OVERLAP", "0")
    geometry = OriginalGeometry(objective, [], _scene("closed"), manifold=True)
    monkeypatch.setattr(parallel_geometry, "WORKERS", 3)
    geometry.start_workers(len(poses))
    try:
        assert_identical(geometry.measure(poses, jacobians=True), reference.measure(poses, jacobians=True))
        changed = poses.copy()
        changed[:, 0] += .001
        complete = reference.measure(changed, jacobians=True)
        assert_identical(geometry.measure(changed, jacobians=True, protected_depths=complete["depths"]), complete)
        previous = {key: cached[0] for key, cached in geometry._static_cache.items()}
        changed[:, 0] += .001
        complete = reference.measure(changed, jacobians=True)
        rejected = geometry.measure(changed, protected_depths={})
        assert not rejected["complete"] and rejected["rejection_frame"] in range(len(poses))
        updated = sum(cached[0] != previous[key] for key, cached in geometry._static_cache.items())
        assert 0 < updated < len(previous), "Rejected candidate queried every static frame"
        assert_identical(geometry.measure(changed, jacobians=True, protected_depths=complete["depths"]), complete)
    finally:
        geometry.stop_workers()


def test_lazy_static_finishes_started_frame_when_peer_rejects_during_self_query(monkeypatch):
    pytest.importorskip("warp")
    from omniretargeting._fast.gpu_geometry import measure_lazy_static
    from tests.test_original_geometry_batches import _problem, _scene

    objective, poses = _problem()
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "0")
    reference = OriginalGeometry(objective, [], _scene("closed"), manifold=True)
    expected = reference.measure_frames(poses, range(len(poses)))
    updates = []
    for key, (signature, _, position, rotation) in reference._static_cache.items():
        ids = np.flatnonzero(objective.shell_sample_geom == key[1]).astype(np.int32)
        summary = dict(row=None, rows=[], clearance=np.inf, unknown=False,
                       origin=position.copy(), _gpu_ids=ids)
        updates.append((key, signature, summary, position, rotation))
    geometry = OriginalGeometry(objective, [], _scene("closed"), manifold=True)
    stop = SimpleNamespace(value=-1)
    original = geometry.self_query.query

    def reject_peer(*args, **kwargs):
        result = original(*args, **kwargs)
        stop.value = 7
        return result

    monkeypatch.setattr(geometry.self_query, "query", reject_peer)
    depths = {key: value for _, _, d, _ in expected["frames"] for key, value in d.items()}
    actual, completed = measure_lazy_static(geometry, poses, list(range(len(poses))),
                                           updates, False, depths, stop)
    assert_identical(actual["frames"], expected["frames"][:1])
    assert completed and not hasattr(geometry, "_gpu_pending_static")


@pytest.mark.parametrize("parallel", [False, True])
def test_moving_gpu_minima_keep_every_exact_row_depth_and_repeat_cache(monkeypatch, parallel):
    pytest.importorskip("warp")
    from scipy.spatial.transform import Rotation
    from omniretargeting._fast import parallel_geometry
    from tests.test_original_geometry_batches import _problem, _scene

    objective, poses = _problem()
    objects = [dict(mesh=trimesh.creation.box(extents=[.55, .6, .5]),
                    translations=np.tile([.1, 0., .6], (len(poses), 1)),
                    rotations=Rotation.from_euler("z", (np.arange(len(poses)) * .04)[:, None]).as_matrix())]
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "0")
    reference = OriginalGeometry(objective, objects, _scene("closed"), manifold=True)
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "1")
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY_MOVING", "1")
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY_OVERLAP", "1")
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY_DEBUG", "1")
    geometry = OriginalGeometry(objective, objects, _scene("closed"), manifold=True)
    if parallel:
        monkeypatch.setattr(parallel_geometry, "MIN_FRAMES_PER_WORKER", 2)
        monkeypatch.setattr(parallel_geometry, "WORKERS", 3)
        geometry.start_workers(len(poses))
    try:
        for candidate in (poses, poses, poses + np.eye(1, poses.shape[1], 0) * .001):
            actual = geometry.measure(candidate, jacobians=True)
            expected = reference.measure(candidate, jacobians=True)
            assert hasattr(geometry, "_gpu_moving_tokens")
            assert not hasattr(geometry, "gpu_moving_unavailable"), getattr(geometry, "gpu_moving_unavailable", None)
            assert_identical(actual, expected)
            assert_identical(geometry.summary(actual), reference.summary(expected))
    finally:
        geometry.stop_workers()


def test_moving_gpu_chunked_minima_preserve_component_and_first_tie_order(monkeypatch):
    pytest.importorskip("warp")
    from tests.test_original_geometry_batches import _problem, _scene

    objective, poses = _problem()
    parts = [trimesh.creation.box(extents=[.55, .6, .5]),
             trimesh.creation.box(extents=[.7, .8, .6])]
    objects = [dict(mesh=trimesh.util.concatenate(parts),
                    translations=np.tile([.1, 0., .6], (len(poses), 1)))]
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "0")
    reference = OriginalGeometry(objective, objects, _scene("closed"), manifold=True)
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "1")
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY_MOVING", "1")
    geometry = OriginalGeometry(objective, objects, _scene("closed"), manifold=True)
    geometry.SCENE_BATCH_POINTS = 2
    assert len(geometry.moving.objects[0]["closed"]) == 2
    assert_identical(geometry.measure(poses, jacobians=True), reference.measure(poses, jacobians=True))


def test_moving_gpu_reuses_unchanged_geoms_after_local_joint_edit(monkeypatch):
    pytest.importorskip("warp")
    from tests.test_original_geometry_batches import _problem, _scene

    objective, poses = _problem()
    objects = [dict(mesh=trimesh.creation.box(extents=[.55, .6, .5]),
                    translations=np.tile([.1, 0., .6], (len(poses), 1)))]
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "0")
    reference = OriginalGeometry(objective, objects, _scene("closed"), manifold=True)
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "1")
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY_MOVING", "1")
    geometry = OriginalGeometry(objective, objects, _scene("closed"), manifold=True)
    geometry.measure(poses)
    changed = poses.copy()
    changed[:, 7] += .001
    assert_identical(geometry.measure(changed, jacobians=True),
                     reference.measure(changed, jacobians=True))
    stats = geometry.gpu_moving_stats
    assert stats["reused_geoms"] == 2 * len(poses)
    assert stats["screened_geoms"] == len(poses)
    # Returning to a previous pose must replace only the edited body's cache.
    assert_identical(geometry.measure(poses), reference.measure(poses))
    assert geometry.gpu_moving_stats["reused_geoms"] == 2 * len(poses)
    geometry.moving = geometry.moving.fresh_state()
    assert_identical(geometry.measure(poses), reference.measure(poses))
    assert geometry.moving.forward_cache is geometry._gpu_moving_cache


def test_gpu_minimum_compaction_keeps_exact_winner_and_every_ambiguous_sample():
    pytest.importorskip("warp")
    from omniretargeting._fast.gpu_geometry import ResidentScene

    scene = SceneQuery(trimesh.creation.box(extents=[2., 2., 2.]))
    points = np.column_stack([np.r_[np.linspace(.2, .9, 110), 1., 1.008, 1.5],
                              np.zeros(113), np.zeros(113)])
    resident = ResidentScene(scene, points, np.arange(len(points), dtype=np.int32))
    arguments = (np.zeros((1, 3)), np.eye(3)[None], np.array([0], np.int32),
                 np.array([len(points)], np.int32))
    full = resident.candidates(*arguments)
    compact = resident.candidates(*arguments, minimum=True)
    exact = scene.query(points)
    winner = int(np.argmin(exact["clearance"]))
    assert compact["candidates"] < full["candidates"] / 2
    assert winner in compact["samples"]
    assert set(full["samples"][full["ambiguous"]]).issubset(compact["samples"])
    queried = scene.query(points[compact["samples"]])
    found = int(np.argmin(queried["clearance"]))
    assert compact["samples"][found] == winner
    assert_identical(queried["clearance"][found], exact["clearance"][winner])
    audited = resident.candidates(*arguments, debug=True, minimum=True)
    assert 100 in audited["samples"][audited["debug"]]


def test_resident_query_with_no_candidates_preserves_empty_cuda_result(monkeypatch):
    pytest.importorskip("warp")
    from omniretargeting._fast.gpu_geometry import geometry_service
    from tests.test_original_geometry_batches import _problem, _scene

    objective, _ = _problem()
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "1")
    geometry = OriginalGeometry(objective, [], _scene("closed"))
    gid = int(objective.shell_geoms[0])
    result = geometry_service().candidates(
        geometry._gpu_token, np.array([[5., 4., 3.]]), np.eye(3)[None],
        np.array([geometry._gpu_starts[gid]], np.int32),
        np.array([len(geometry._gpu_groups[gid])], np.int32))
    assert result["candidates"] == 0 and len(result["jobs"]) == len(result["samples"]) == 0
    assert result["lower_bounds"][0] > .008


@pytest.mark.parametrize("unsupported", ["open", "scale"])
def test_unsupported_moving_minima_keep_original_cpu_unknowns_and_rows(monkeypatch, unsupported):
    pytest.importorskip("warp")
    from tests.test_original_geometry_batches import _problem, _scene

    objective, poses = _problem()
    mesh = trimesh.creation.box(extents=[.55, .6, .5])
    if unsupported == "open":
        mesh.update_faces(np.arange(len(mesh.faces) - 2))
    objects = [dict(mesh=mesh, translations=np.tile([.1, 0., .6], (len(poses), 1)),
                    scales=np.full(len(poses), 1.1 if unsupported == "scale" else 1.))]
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "0")
    reference = OriginalGeometry(objective, objects, _scene("closed"), manifold=True)
    expected = reference.measure(poses, jacobians=True)
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "1")
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY_MOVING", "1")
    geometry = OriginalGeometry(objective, objects, _scene("closed"), manifold=True)
    actual = geometry.measure(poses, jacobians=True)
    assert not hasattr(geometry, "_gpu_moving_tokens")
    assert_identical(actual, expected)
    assert_identical(geometry.summary(actual), reference.summary(expected))


@pytest.mark.parametrize("parallel", [False, True])
def test_moving_minimum_debug_audit_preserves_unknown_flags(monkeypatch, parallel):
    pytest.importorskip("warp")
    from tests.test_original_geometry_batches import _problem, _scene

    objective, poses = _problem()
    objects = [dict(mesh=trimesh.creation.box(extents=[.55, .6, .5]),
                    translations=np.tile([.1, 0., .6], (len(poses), 1)))]
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "1")
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY_MOVING", "1")
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY_OVERLAP", "1")
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY_DEBUG", "1")
    geometry = OriginalGeometry(objective, objects, _scene("closed"))
    query = geometry.moving.objects[0]["query"]
    original = query.query

    def hide_unknown(*args, **kwargs):
        measured = original(*args, **kwargs)
        measured["uncertain"][:] = True
        return measured

    monkeypatch.setattr(query, "query", hide_unknown)
    if parallel:
        from omniretargeting._fast import parallel_geometry
        monkeypatch.setattr(parallel_geometry, "MIN_FRAMES_PER_WORKER", 2)
        monkeypatch.setattr(parallel_geometry, "WORKERS", 3)
        geometry.start_workers(len(poses))
    try:
        with pytest.raises(AssertionError, match="Moving GPU screening"):
            geometry.measure(poses)
    finally:
        geometry.stop_workers()
    assert not geometry._gpu_moving_cache.cached


def test_far_lower_bounds_keep_decisions_when_motion_invalidates_the_bound(monkeypatch):
    pytest.importorskip("warp")
    from tests.test_original_geometry_batches import _problem

    objective, poses = _problem()
    floor = trimesh.creation.box(extents=[4., 4., .1])
    floor.apply_translation([0., 0., -.05])
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "0")
    reference = OriginalGeometry(objective, [], SceneQuery(floor), manifold=True)
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "1")
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY_DEBUG", "1")
    actual = OriginalGeometry(objective, [], SceneQuery(floor), manifold=True)
    assert_identical(actual.measure(poses), reference.measure(poses))
    for cached in actual._static_cache.values():
        assert cached[1]["clearance"] >= .008 and not cached[1]["unknown"]
    changed = poses.copy()
    changed[:, 7] += .031
    changed[:, 2] = .03
    measured = actual.measure(changed)
    assert_identical(measured, reference.measure(changed))
    assert any(key[1] == "static" for key, _ in measured["rows"])


@pytest.mark.parametrize("centers", [False, True])
def test_debug_checks_raise_if_a_screened_sample_is_unknown(monkeypatch, centers):
    pytest.importorskip("warp")
    from tests.test_original_geometry_batches import _problem

    objective, poses = _problem()
    floor = trimesh.creation.box(extents=[4., 4., .1])
    floor.apply_translation([0., 0., -.05])
    scene = SceneQuery(floor)
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "1")
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY_DEBUG", "1")
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY_CENTERS", "1" if centers else "0")
    geometry = OriginalGeometry(objective, [], scene)
    original, calls = scene.query, []

    def hide_uncertainty(points, **kwargs):
        measured = original(points, **kwargs)
        calls.append(1)
        if centers or len(calls) > 1:
            measured["uncertain"][:] = True
        return measured

    monkeypatch.setattr(scene, "query", hide_uncertainty)
    with pytest.raises(AssertionError, match="Sphere bound"):
        geometry.measure(poses)
    assert not geometry._static_cache  # No partially verified results committed.


def test_missing_optional_backend_preserves_cpu_query(monkeypatch):
    pytest.importorskip("warp")
    from omniretargeting._fast import gpu_geometry
    from tests.test_original_geometry_batches import _problem, _scene

    objective, poses = _problem()
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "0")
    reference = OriginalGeometry(objective, [], _scene("closed")).measure(poses)

    def unavailable(_):
        raise ImportError("No Warp")

    monkeypatch.setattr(gpu_geometry, "initialize_static", unavailable)
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "1")
    geometry = OriginalGeometry(objective, [], _scene("closed"))
    assert not geometry._gpu_allowed and geometry.gpu_unavailable == "No Warp"
    assert_identical(geometry.measure(poses), reference)


def test_exact_query_errors_are_not_backend_availability_failures(monkeypatch):
    pytest.importorskip("warp")
    from omniretargeting.geometry_query import GeometryQueryError
    from tests.test_original_geometry_batches import _problem, _scene

    objective, poses = _problem()
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "1")
    scene = _scene("closed")
    geometry = OriginalGeometry(objective, [], scene)

    def fail_exact_query(_):
        raise GeometryQueryError("Exact query failed")

    monkeypatch.setattr(scene, "query", fail_exact_query)
    with pytest.raises(GeometryQueryError, match="Exact query failed"):
        geometry.measure(poses)
    assert geometry._gpu_allowed and not geometry._static_cache


def _repair_report_fixture():
    return dict(seconds=1., before=dict(max_depth=.01),
                after=dict(max_depth=.002, moving_query=dict(query_count=12)),
                history=[dict(status="accepted", alpha=.5, after=.002,
                              rejected_geometry_prefix_frame=4,
                              query_failures=[], collision_regressions=1,
                              regression_examples=[dict(key=[4, "static", 1], increase=.001)])])


def test_repair_comparison_flags_only_documented_partial_diagnostics():
    cpu = _repair_report_fixture()
    gpu = copy.deepcopy(cpu)
    gpu["seconds"] = 2.
    gpu["after"]["moving_query"]["query_count"] += 3
    gpu["history"][0]["rejected_geometry_prefix_frame"] = 6
    gpu["history"][0]["collision_regressions"] = 2
    diagnostics = assert_repair_equivalent(gpu, cpu)
    assert len(diagnostics) == 3
    assert {item["path"].split("[")[-1] for item in diagnostics} == {
        "'query_count']", "'rejected_geometry_prefix_frame']", "'collision_regressions']"}


@pytest.mark.parametrize("field,value", [("status", "rejected"), ("alpha", .25), ("after", .003)])
def test_repair_comparison_rejects_changed_accepted_history(field, value):
    cpu = _repair_report_fixture()
    gpu = copy.deepcopy(cpu)
    gpu["history"][0][field] = value
    with pytest.raises(AssertionError, match=field):
        assert_repair_equivalent(gpu, cpu)


def test_repair_comparison_keeps_complete_measurements_strict():
    cpu = _repair_report_fixture()
    gpu = copy.deepcopy(cpu)
    gpu["after"]["max_depth"] += 1e-12
    with pytest.raises(AssertionError, match="max_depth"):
        assert_repair_equivalent(gpu, cpu)
    cpu["history"][0].pop("rejected_geometry_prefix_frame")
    gpu = copy.deepcopy(cpu)
    gpu["history"][0]["collision_regressions"] += 1
    with pytest.raises(AssertionError, match="collision_regressions"):
        assert_repair_equivalent(gpu, cpu)


if __name__ == "__main__":
    main()
