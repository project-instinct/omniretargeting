"""Capped frozen-copy parent initial/audit timing; no solver or pipeline."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from omniretargeting._fast.mesh_repair import OriginalGeometry
from tests.test_gpu_geometry import assert_identical, load_saved_run


def legacy(measured):
    return {name: measured[name] for name in ("rows", "depths", "failures", "complete")}


def benchmark(directory, cpu_oracle):
    objective, poses, objects, scene = load_saved_run(directory)
    objective.coords.cols = np.arange(objective.model.nv)
    started = time.perf_counter()
    geometry = OriginalGeometry(objective, objects, scene, manifold=True,
                                columns=objective.coords.cols, reference_poses=poses)
    result = dict(frames=len(poses), setup_seconds=time.perf_counter() - started,
                  window_parent=geometry._window_parent_mode, gpu=geometry._gpu_allowed)
    started = time.perf_counter()
    initial = geometry.measure(poses)
    result["initial_seconds"] = time.perf_counter() - started
    result["initial_gpu"] = dict(getattr(geometry, "gpu_stats", {}))
    print("INITIAL", json.dumps(result), flush=True)
    if cpu_oracle:
        started = time.perf_counter()
        oracle = OriginalGeometry(objective, objects, geometry.scene, manifold=True,
                                  columns=objective.coords.cols, reference_poses=poses)
        oracle._window_parent_mode = oracle._gpu_allowed = False
        oracle.start_workers(len(poses))
        try:
            expected = oracle.measure(poses)
            result["cpu_seconds"] = time.perf_counter() - started
            assert_identical(legacy(initial), legacy(expected))
        finally:
            oracle.stop_workers()
        print("CPU_ORACLE", json.dumps({"seconds": result["cpu_seconds"], "bit_identical": True}), flush=True)
    started = time.perf_counter()
    geometry.prepare_fork_queries()
    result["prepare_seconds"] = time.perf_counter() - started
    result["bounds"] = geometry.static_bounds_stats
    result["self_seed"] = getattr(geometry, "fork_self_seed_stats", {})
    print("PREPARE", json.dumps({name: result[name] for name in ("prepare_seconds", "bounds", "self_seed")}), flush=True)
    started = time.perf_counter()
    audit = geometry.measure(poses)
    result["audit_seconds"] = time.perf_counter() - started
    assert_identical(legacy(initial), legacy(audit))
    result["initial_audit_identical"] = True
    result["audit_closed_resources"] = geometry.parallel is None and not geometry._gpu_allowed
    print("AUDIT", json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu-oracle", action="store_true")
    args = parser.parse_args()
    result = benchmark(args.directory, args.cpu_oracle)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
