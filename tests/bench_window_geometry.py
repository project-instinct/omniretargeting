"""Saved-pose microbenchmark; no solver/pipeline or source inputs are changed.

Run from one frozen package copy under a memory-capped user service. Measures
manifold/full-root-column geometry, with CPU/GPU variants run sequentially.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import numpy as np

from omniretargeting._fast.mesh_repair import OriginalGeometry
from omniretargeting._fast.parallel_geometry import ParallelMeasure
from tests.test_gpu_geometry import _TimedQuery, assert_identical, load_saved_run


class TimedGeometry(OriginalGeometry):
    def _query_static_batch(self, pending):
        started = time.perf_counter()
        try:
            return super()._query_static_batch(pending)
        finally:
            self._static_pending_seconds = getattr(self, "_static_pending_seconds", 0.) + time.perf_counter() - started
            self._static_pending_points = getattr(self, "_static_pending_points", 0) + sum(len(p) for _, p, _ in pending)
            self._static_pending_calls = getattr(self, "_static_pending_calls", 0) + 1

    def measure_frames(self, *args, **kwargs):
        stats = {}
        moving, self_query = self.moving, self.self_query
        before_static = getattr(self, "_static_pending_seconds", 0.)
        if moving is not None:
            self.moving = _TimedQuery(moving, stats, "moving")
        self.self_query = _TimedQuery(self_query, stats, "self")
        started = time.perf_counter()
        try:
            result = super().measure_frames(*args, **kwargs)
            stats["static"] = getattr(self, "_static_pending_seconds", 0.)
            stats["static_points"] = getattr(self, "_static_pending_points", 0)
            stats["static_calls"] = getattr(self, "_static_pending_calls", 0)
            stats["worker_total"] = time.perf_counter() - started + before_static
            result["timing"] = stats
            self._static_pending_seconds = 0.
            self._static_pending_points = self._static_pending_calls = 0
            self._last_serial_timing = stats
            return result
        finally:
            self.moving, self.self_query = moving, self_query


def _timers(geometry):
    totals = Counter()
    for name in ("_prepare_gpu_static", "_prepare_gpu_moving", "linearize"):
        original = getattr(geometry, name)

        def timed(*args, _original=original, _name=name, **kwargs):
            started = time.perf_counter()
            try:
                return _original(*args, **kwargs)
            finally:
                totals[_name] += time.perf_counter() - started
        setattr(geometry, name, timed)
    return totals


def _legacy(measured):
    return {name: measured[name] for name in ("rows", "depths", "failures", "complete")}


def benchmark(directory):
    objective, poses, objects, scene = load_saved_run(directory)
    objective.coords.cols = np.arange(objective.model.nv)
    frames = range(min(16, len(poses)), min(48, len(poses)))
    original_parallel = ParallelMeasure.measure
    records, expected, expected_trial = {}, None, None
    for mode in ("cpu", "gpu_default", "gpu_options"):
        os.environ["OMNIRETARGETING_GPU_GEOMETRY"] = "0" if mode == "cpu" else "1"
        for suffix in ("CENTERS", "MOVING", "OVERLAP"):
            os.environ["OMNIRETARGETING_GPU_GEOMETRY_" + suffix] = "1" if mode == "gpu_options" else "0"
        started = time.perf_counter()
        geometry = TimedGeometry(objective, objects, scene, manifold=True,
                                 columns=objective.coords.cols, reference_poses=poses)
        init = time.perf_counter() - started
        geometry.start_workers(len(poses))
        totals = _timers(geometry)
        worker_stats = []

        def parallel_timed(*args, **kwargs):
            started = time.perf_counter()
            result = original_parallel(*args, **kwargs)
            totals["parallel_wall"] += time.perf_counter() - started
            worker_stats.extend(part.get("timing", {}) for part in result)
            return result
        ParallelMeasure.measure = parallel_timed

        def run(label, qpos, selected=None, *, client=None):
            queried = geometry if client is None else client
            worker_stats.clear()
            before = totals.copy()
            started = time.perf_counter()
            measured = queried.measure(qpos, frames=selected, jacobians=True)
            matrix, _ = queried.linear_rows(measured, frames=selected)
            elapsed = time.perf_counter() - started
            delta = {key: totals[key] - before[key] for key in totals}
            delta.update(seconds=elapsed, rows=matrix.shape[0], columns=matrix.shape[1],
                         worker_sum=dict(sum((Counter(item) for item in worker_stats), Counter())),
                         worker_max={key: max(item.get(key, 0.) for item in worker_stats)
                                     for key in ("static", "moving", "self", "worker_total")} if worker_stats else {})
            if client is not None:
                delta["serial_query"] = getattr(client, "_last_serial_timing", {})
            if selected is not None:
                delta["cache"] = queried.window_stats
            delta["gpu_allowed"] = queried._gpu_allowed
            if hasattr(queried, "gpu_stats"):
                delta["gpu_static"] = queried.gpu_stats
            if hasattr(queried, "gpu_moving_stats"):
                delta["gpu_moving"] = queried.gpu_moving_stats
            records[mode][label] = delta
            print(mode, label, json.dumps(delta), flush=True)
            return measured

        records[mode] = dict(init_seconds=init)
        try:
            full = run("full", poses)
            if expected is None:
                expected = _legacy(full)
            else:
                assert_identical(_legacy(full), expected)
            warm = run("window_cached", poses, frames)
            restricted = {**expected, "rows": [item for item in expected["rows"] if item[0][0] in frames],
                          "depths": {k: v for k, v in expected["depths"].items() if k[0] in frames}}
            assert_identical(_legacy(warm), restricted)
            trial = poses.copy()
            trial[list(frames), 7] += .001
            changed = run("window_changed", trial, frames)
            if expected_trial is None:
                expected_trial = _legacy(changed)
            else:
                assert_identical(_legacy(changed), expected_trial)
            run("window_trial_cached", trial, frames)
            one = trial.copy()
            one[frames.start + 1, 7] += .001
            run("window_one_changed", one, frames)
            private = geometry.fork_private_client()
            for name in ("_prepare_gpu_static", "_prepare_gpu_moving", "linearize"):
                delattr(private, name)  # Remove benchmark closures bound to the parent.
            run("private_window_cached", poses, frames, client=private)
            private_trial = poses.copy()
            private_trial[list(frames), 7] += .002
            run("private_window_changed", private_trial, frames, client=private)
            run("private_trial_cached", private_trial, frames, client=private)
        finally:
            ParallelMeasure.measure = original_parallel
            geometry.stop_workers()
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    records = benchmark(args.directory)
    Path(args.output).write_text(json.dumps(records, indent=2) + "\n")
    print("EQUIVALENCE: every legacy row/field/array byte identical", flush=True)
