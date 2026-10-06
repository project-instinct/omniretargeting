"""Audit original geometry in batches using independent winding containment.

Penetration is a sampled lower bound, not a collision-free certificate. Keep
per-frame/body depths and exit-vector components separate from tracking errors.
"""

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import trimesh

from omniretargeting.moving_scene import MovingSceneQuery
from omniretargeting.scene_metrics_parallel import (
    run_moving_batches,
    run_static_batches,
)
from omniretargeting.scene_geometry import (
    SceneQuery,
    collision_geom_ids,
    geom_surface_offsets,
)


def report_depths(depths, model, start):
    frame, geom = np.unravel_index(np.argmax(depths), depths.shape)
    peak = depths.max(axis=1)
    return {
        "max_m": float(peak.max()),
        "p95_m": float(np.quantile(peak, 0.95)),
        "mean_m": float(peak.mean()),
        "worst_frame": int(frame + start),
        "worst_body": model.body(int(model.geom_bodyid[geom])).name,
        "affected_frames": (np.flatnonzero(peak > 1e-6) + start).tolist(),
    }


def audit(args):
    started = time.perf_counter()
    last = started

    def stage(name):
        nonlocal last
        now = time.perf_counter()
        print(f"timing {name}={now - last:.3f}s", flush=True)
        last = now

    inputs = np.load(args.prepared, allow_pickle=True).item()
    stage("load_prepared")
    payload = np.load(args.motion)
    poses = payload["qpos"]
    start = int(payload.get("frame_start", 0))
    model = mujoco.MjModel.from_xml_path(inputs["model_path"])
    stage("load_model")
    ids = collision_geom_ids(model)
    offsets = [
        np.unique(
            np.r_[geom_surface_offsets(model, int(g), args.spacing), np.zeros((1, 3))],
            axis=0,
        )
        for g in ids
    ]
    stage("geom_surface_offsets")
    owners = np.concatenate([np.full(len(p), g, int) for g, p in zip(ids, offsets)])
    scene = trimesh.load_mesh(args.prepared.parent / "static_scene.obj", process=False)
    stage("load_static_mesh")
    query = SceneQuery(
        scene,
        enclosed_free_space=inputs.get("scene_enclosed_free_space", False),
        full_geometry=False,
    )
    stage("build_scene_query")
    open_scan = query.has_unsigned_only_components
    objects = np.load(args.objects, allow_pickle=True).tolist() if args.objects else []
    stage("load_objects")
    moving_holder = []

    def build_moving_query(*, overlapped=args.workers > 1):
        callback_started = time.perf_counter()
        local_objects = [
            {
                **obj,
                "translations": obj["translations"][start : start + len(poses)],
                "rotations": obj["rotations"][start : start + len(poses)],
            }
            for obj in objects
        ]
        moving_holder.append(MovingSceneQuery(model, local_objects, len(poses)))
        if overlapped:
            print(
                "timing build_moving_query_overlapped="
                f"{time.perf_counter() - callback_started:.3f}s",
                flush=True,
            )
        else:
            stage("build_moving_query")

    depths, unknown, worst_exit, backend = run_static_batches(
        model=model,
        query=query,
        poses=poses,
        ids=ids,
        offsets=offsets,
        owners=owners,
        batch=args.batch,
        workers=args.workers,
        open_scan=open_scan,
        prepare_moving=build_moving_query if args.workers > 1 and objects else None,
    )
    stage("static_parallel")
    static = report_depths(depths, model, start)
    static.update(
        uncertain_samples=unknown,
        spacing_m=args.spacing,
        worst_exit_xy_m=float(np.linalg.norm(worst_exit[:2])),
        worst_exit_z_m=float(worst_exit[2]),
        method=("original triangles; exact geom/finite-triangle crossing (open scans, no solid side)"
                if open_scan else "original triangles; winding containment"),
        collision_free_certificate=False,
    )
    if open_scan:
        static.update(open_scan_backend={key: backend[key] for key in (
            "open_surface_test", "pair_queries", "cylinder_surface_faces",
            "ambiguous_surface_intersections", "uncertified_cylinder_witnesses")})
    saved = {"static_depths": depths}
    result = {
        "motion": str(args.motion),
        "start": start,
        "frames": len(poses),
        "static": static,
        "moving_object_count": len(objects),
    }
    if objects:
        if not moving_holder:
            build_moving_query()
        moving = moving_holder[0]
        depths, failures, moving_exit, moving_description = run_moving_batches(
            scene=moving,
            poses=poses,
            start=start,
            batch=args.batch,
            workers=args.workers,
        )
        stage("moving_parallel")
        result["moving"] = {
            **report_depths(depths, model, start),
            "failures": failures,
            "worst_exit_xy_m": float(np.linalg.norm(moving_exit[:2])),
            "worst_exit_z_m": float(moving_exit[2]),
            **moving_description,
        }
        saved["moving_depths"] = depths
    args.output.write_text(json.dumps(result, indent=2))
    np.savez(args.output.with_suffix(".npz"), **saved)
    stage("write_outputs")
    print("SAVED", args.output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--objects", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacing", type=float, default=0.02)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument(
        "--workers", type=int, default=1, help="parallel audit workers (default: 1)"
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    audit(args)
