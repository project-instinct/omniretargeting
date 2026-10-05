"""Audit original geometry in batches using independent winding containment.

Penetration is a sampled lower bound, not a collision-free certificate. Keep
per-frame/body depths and exit-vector components separate from tracking errors.
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from omniretargeting.moving_scene import MovingSceneQuery
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


def open_surface_depths(model, poses, ids, query):
    """Exact geom / finite-triangle crossing depths of a scene of open scans.

    A scene with unsigned-only components has no sampled side anywhere. Its
    collision geoms are tested against the actual triangles (capsule, sphere
    and solid cylinder). An open scan has no solid side: a geom wholly behind
    one is not a crossing. Unsupported geoms or signed components raise.
    """
    from omniretargeting._fast.original_rigid_fallback import uncovered_components
    from omniretargeting.geometry_query import OPEN_SURFACE_GEOMS, RigidGeometryQuery

    backend = RigidGeometryQuery(model, ids, query)
    unsupported = [int(g) for g in ids if model.geom_type[g] not in OPEN_SURFACE_GEOMS]
    uncovered = uncovered_components(query, backend)
    if unsupported or uncovered:
        raise RuntimeError(f"open-scan audit cannot certify geoms {unsupported} / components {uncovered}")
    data = mujoco.MjData(model)
    depths = np.zeros((len(poses), model.ngeom))
    uncertain, worst_exit, worst_depth = 0, np.zeros(3), 0.0
    for frame, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        for row in backend.query(data, 0.0):
            uncertain += bool(row["uncertain"])
            depth = max(-float(row["clearance"]), 0.0)
            gid = int(row["geom_ids"])
            depths[frame, gid] = max(depths[frame, gid], depth)
            if depth > worst_depth:
                worst_depth, worst_exit = depth, depth * np.asarray(row["directions"])
    if backend.failures:
        raise RuntimeError(f"open-scan audit query failures: {backend.failures[:3]}")
    return depths, uncertain, worst_exit, backend.describe()


def audit(args):
    inputs = np.load(args.prepared, allow_pickle=True).item()
    payload = np.load(args.motion)
    poses = payload["qpos"]
    start = int(payload.get("frame_start", 0))
    model = mujoco.MjModel.from_xml_path(inputs["model_path"])
    data = mujoco.MjData(model)
    ids = collision_geom_ids(model)
    offsets = [
        np.unique(
            np.r_[geom_surface_offsets(model, int(g), args.spacing), np.zeros((1, 3))],
            axis=0,
        )
        for g in ids
    ]
    owners = np.concatenate([np.full(len(p), g, int) for g, p in zip(ids, offsets)])
    scene = trimesh.load_mesh(args.prepared.parent / "static_scene.obj", process=False)
    query = SceneQuery(
        scene,
        enclosed_free_space=inputs.get("scene_enclosed_free_space", False),
        full_geometry=False,
    )
    depths = np.zeros((len(poses), model.ngeom))
    unknown = 0
    worst_exit, worst_depth = np.zeros(3), 0.0
    open_scan = query.has_unsigned_only_components
    if open_scan:
        depths, unknown, worst_exit, backend = open_surface_depths(model, poses, ids, query)
    for begin in range(0, len(poses) if not open_scan else 0, args.batch):
        chunk = poses[begin : begin + args.batch]
        points = []
        for pose in chunk:
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            points.append(
                np.concatenate(
                    [
                        data.geom_xpos[g] + p @ data.geom_xmat[g].reshape(3, 3).T
                        for g, p in zip(ids, offsets)
                    ]
                )
            )
        measured = query.query(np.concatenate(points))
        unknown += int(measured["uncertain"].sum())
        signed = measured["clearance"].reshape(len(chunk), len(owners))
        if not np.isfinite(signed).all():
            raise RuntimeError(
                "Original static scene has unresolved sign; cannot certify depths"
            )
        for gid in ids:
            depths[begin : begin + len(chunk), gid] = np.maximum(
                -signed[:, owners == gid], 0
            ).max(axis=1)
        deepest = int(np.argmin(measured["clearance"]))
        depth = -float(measured["clearance"][deepest])
        if depth > worst_depth:
            worst_depth = depth
            worst_exit = depth * measured["directions"][deepest]
        print(
            f"static {begin + len(chunk)}/{len(poses)} max={depths.max():.6f} m",
            flush=True,
        )
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
    }
    objects = np.load(args.objects, allow_pickle=True).tolist() if args.objects else []
    result["moving_object_count"] = len(objects)
    if objects:
        objects = [
            {
                **obj,
                "translations": obj["translations"][start : start + len(poses)],
                "rotations": obj["rotations"][start : start + len(poses)],
            }
            for obj in objects
        ]
        moving = MovingSceneQuery(model, objects, len(poses))
        depths = np.zeros_like(depths)
        failures = []
        moving_exit = np.zeros(3)
        moving_worst = 0.0
        for f, pose in enumerate(poses):
            frame = moving.query(pose, f)
            depths[f] = np.maximum(frame["forward_depth"], frame["reverse_depth"]).max(
                axis=0
            )
            failures.extend((f + start, message) for message in frame["failures"])
            for row in frame["rows"]:
                if -row["clearance"] > moving_worst:
                    moving_worst = -row["clearance"]
                    moving_exit = moving_worst * row["direction"]
            if (f + 1) % args.batch == 0:
                print(
                    f"moving {f + 1}/{len(poses)} max={depths.max():.6f} m", flush=True
                )
        result["moving"] = {
            **report_depths(depths, model, start),
            "failures": failures,
            "worst_exit_xy_m": float(np.linalg.norm(moving_exit[:2])),
            "worst_exit_z_m": float(moving_exit[2]),
            **moving.describe(),
        }
        saved["moving_depths"] = depths
    args.output.write_text(json.dumps(result, indent=2))
    np.savez(args.output.with_suffix(".npz"), **saved)
    print("SAVED", args.output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--objects", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacing", type=float, default=0.02)
    parser.add_argument("--batch", type=int, default=64)
    audit(parser.parse_args())
