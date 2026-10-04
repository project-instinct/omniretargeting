"""Locked external runner helper: capture and time repair, or replay its input.

This script does not acquire the pipeline lock: invoke the whole process with
flock /tmp/speed15x_pipeline.lock. Imports must come from a private snapshot.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import runpy
import sys
import time
from functools import wraps

from repair_input_capture import capture_inputs



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--replay")
    parser.add_argument("--full-assembly", action="store_true")
    parser.add_argument("--proposal-workers", type=int)
    args, command = parser.parse_known_args()
    sys.path.insert(0, str(Path(args.snapshot).resolve()))
    import numpy as np
    from omniretargeting._fast import mesh_repair, repair_restoration, repair_canonical
    from omniretargeting._fast.repair_limits import RepairLimits
    if args.proposal_workers is not None:
        from omniretargeting._fast import repair_proposals
        factory = repair_proposals.RepairProposals
        repair_proposals.RepairProposals = lambda limits, geometry: factory(
            limits, geometry, count=args.proposal_workers)
    output = Path(args.directory)
    output.mkdir(parents=True, exist_ok=True)
    totals = {}

    def time_function(owner, name, label=None):
        original = getattr(owner, name)
        label = label or name

        @wraps(original)
        def timed(*a, **kw):
            started = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                value = totals.setdefault(label, dict(calls=0, seconds=0.))
                value["calls"] += 1
                value["seconds"] += time.perf_counter() - started
        setattr(owner, name, timed)

    if args.full_assembly:
        original = RepairLimits.constraints

        def full(self, poses, **kwargs):
            kwargs.pop("projection", None)
            return original(self, poses, **kwargs)
        RepairLimits.constraints = full
    for owner, name in ((RepairLimits, "constraints"), (RepairLimits, "check"),
                        (mesh_repair, "_qp_step"), (mesh_repair, "restore_repair"),
                        (mesh_repair.OriginalGeometry, "measure"),
                        (mesh_repair.OriginalGeometry, "linearize"),
                        (repair_canonical, "canonical_data"), (repair_canonical, "solve_data"),
                        (repair_restoration, "_solve_cvxpy_restoration")):
        time_function(owner, name)
    original_repair = mesh_repair.repair_original_meshes

    def capture(objective, poses, scene_objects=(), scene_query=None):
        if not args.replay:
            try:
                capture_inputs(output / "inputs.pkl", (objective, poses, scene_objects,
                                                       scene_query.mesh, scene_query.enclosed_free_space))
                print("CAPTURED repair inputs", flush=True)
            except Exception as error:
                print("CAPTURE ERROR", repr(error), flush=True)
        totals.clear()
        import clarabel
        factory = clarabel.DefaultSolver

        def trace_program(p, q, a, b, cones, settings):
            signature = hashlib.sha256()
            for value in (p, a):
                signature.update(str(value.shape).encode())
                for array in (value.indptr, value.indices, value.data):
                    signature.update(array.tobytes())
            for array in (q, b):
                signature.update(array.tobytes())
            signature.update(str(cones).encode())
            signature.update(str(settings).encode())
            with (output / f"programs-{os.getpid()}.jsonl").open("a") as handle:
                handle.write(json.dumps(dict(program=signature.hexdigest(), variables=len(q),
                                            rows=len(b))) + "\n")
            return factory(p, q, a, b, cones, settings)
        clarabel.DefaultSolver = trace_program
        try:
            result, report = original_repair(objective, poses, scene_objects, scene_query)
        finally:
            clarabel.DefaultSolver = factory
        np.savez(output / "final.npz", qpos=result)
        (output / "repair.json").write_text(json.dumps(report, indent=2, default=float))
        (output / "timers.json").write_text(json.dumps(totals, indent=2))
        print("REPAIR_TIMERS", json.dumps(totals), flush=True)
        return result, report

    mesh_repair.repair_original_meshes = capture
    if args.replay:
        import torch
        torch.set_num_threads(4)  # Match run_trumans_cli.py's process setup.
        from omniretargeting.scene_geometry import SceneQuery
        with Path(args.replay).open("rb") as handle:
            objective, poses, objects, mesh, enclosed = pickle.load(handle)
        objective.kin.cache.clear()
        mesh._cache.clear()
        scene = SceneQuery(mesh, enclosed_free_space=enclosed)
        capture(objective, poses, objects, scene)
    else:
        # pipeline may have been imported indirectly before the patch.
        from omniretargeting._fast import pipeline
        pipeline.repair_original_meshes = capture
        sys.argv = command
        runpy.run_path(command[0], run_name="__main__")


if __name__ == "__main__":
    main()
