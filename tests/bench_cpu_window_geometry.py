"""One capped serial microbenchmark for parent-prepared CPU window queries."""

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from tests.bench_window_geometry import TimedGeometry
from tests.test_gpu_geometry import assert_identical, load_saved_run


def run(directory):
    objective, poses, objects, scene = load_saved_run(directory)
    objective.coords.cols = np.arange(objective.model.nv)
    frames = range(16, min(48, len(poses)))
    parent = TimedGeometry(objective, objects, scene, manifold=True,
                           columns=objective.coords.cols, reference_poses=poses)
    tick = time.perf_counter()
    baseline = parent.measure(poses)
    result = dict(initial_seconds=time.perf_counter() - tick)
    tick = time.perf_counter()
    parent.prepare_fork_queries()
    result["prepare_seconds"] = time.perf_counter() - tick
    result["bounds"] = parent.static_bounds_stats
    result["seed"] = getattr(parent, "fork_self_seed_stats", {})
    print("PREPARE", json.dumps(result), flush=True)
    connection, child = multiprocessing.get_context("fork").Pipe()
    process = multiprocessing.get_context("fork").Process(target=run_windows,
        args=(parent, poses, frames, child))
    process.start()
    child.close()
    if not connection.poll(100):
        process.terminate()
        raise TimeoutError("window microbenchmark child timed out")
    ok, payload = connection.recv()
    process.join(timeout=10)
    if not ok:
        raise AssertionError(payload)
    result.update(payload)
    return result


def run_windows(parent, poses, frames, connection):
    try:
        connection.send((True, window_variants(parent, poses, frames)))
    except Exception:
        import traceback
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()


def window_variants(parent, poses, frames):
    result = dict(child_tasks=len(list(Path("/proc/self/task").iterdir())), gpu_allowed=False)
    for label, screen, native, coarse in (("resident", False, False, False),
        ("point_screen", True, False, False), ("point_screen_native_self", True, True, False),
        ("coarse_screen_native_self", True, True, True)):
        client = parent.fork_private_client()
        if not screen:
            client._fork_queries_prepared = False  # Keep native winding, disable all material-mask reuse.
        if not native:
            client._fork_self_query = None
            client._fork_self_window = None
        if not coarse:
            client._static_coarse_bounds = None
        variants = {}
        for name, delta in (("cached", 0.), ("changed32", .001), ("incremental32", .0011),
                            ("one_changed", .0012)):
            candidate = poses.copy()
            if name == "one_changed":
                candidate[list(frames), 7] += .0011
                candidate[frames.start + 1, 7] += delta - .0011
            else:
                candidate[list(frames), 7] += delta
            trees = [component[3] for component in client.scene.components]
            windings = list(client.scene._r4_winding_cache.values())
            before_tree = sum(tree.seconds for tree in trees)
            before_winding = sum(getattr(w, "seconds", 0.) for w in windings)
            before_winding_points = sum(getattr(w, "points", 0) for w in windings)
            tick = time.perf_counter()
            measured = client.measure(candidate, frames=frames, jacobians=True)
            matrix, phi = client.linear_rows(measured, frames=frames)
            seconds = time.perf_counter() - tick
            variants[name] = dict(seconds=seconds, rows=matrix.shape[0], cache=client.window_stats,
                                  scope=measured["query_scope"],
                                  queries={} if not client.window_stats["queried_frames"] else
                                          getattr(client, "_last_serial_timing", {}),
                                  static_native=dict(aabb_seconds=sum(t.seconds for t in trees) - before_tree,
                                      winding_seconds=sum(getattr(w, "seconds", 0.) for w in windings) - before_winding,
                                      winding_points=sum(getattr(w, "points", 0) for w in windings) - before_winding_points),
                                  native_self=getattr(getattr(client, "_fork_self_window", None), "diagnostics",
                                      getattr(getattr(client, "_fork_self_query", None), "diagnostics", {})))
            variants[name]["coarse_culled_points"] = getattr(client, "static_coarse_culled_points", 0)
            print(label, name, json.dumps(variants[name]), flush=True)
            # Save each first exact result, compare every subsequent variant to it.
            bare = {k: measured[k] for k in ("rows", "depths", "failures", "complete")}
            if label == "resident":
                result.setdefault("exact", {})[name] = bare
            else:
                assert_identical(bare, result["exact"][name])
        result[label] = variants
    result.pop("exact")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = run(args.directory)
    Path(args.output).write_text(json.dumps(data, indent=2) + "\n")
    print("EQUIVALENCE: all legacy witness and summary field bytes identical", flush=True)
