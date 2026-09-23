#!/usr/bin/env python3
"""Batch retarget TRUMANS motions that belong to one terrain/scene.

TRUMANS stores per-frame global arrays (seg_name.npy, scene_flag.npy,
scene_list.npy) that pair every motion segment with a scene.  This entry
point mirrors the pairing logic of ``trumans_pairing.py`` but keeps the
conversion and retargeting self-contained so it can run from this repo.

TRUMANS scene meshes and ``human_joints.npy`` use the SMPL-X world frame
(+X left, +Y up, +Z forward).  The OmniRetargeting repo/robot frame is
+X forward, +Y left, +Z up.  Both the selected motion segments and the
scene mesh are therefore converted to the repo frame here, using the same
axis permutation already used by ``omniretargeting.data_sources.smplx_visualize``
(``SMPLX_TO_WORLD_AXES = (2, 0, 1)``).

Examples:
    python truemans_batch.py \
        --dataset_root /home/ziwen/Datasets/TRUMANS/Data_release \
        --terrain_name 3a9a1c7e-f4cd-46f7-8449-466cbfd82c08 \
        --num_motions 3 \
        --robot-config robot_models/kengo/kengo.json \
        --scale-factor 0.95
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

from omniretargeting.utils.batch_processing import (
    _summarize,
    get_activation_prefix,
    run_single_job,
    write_source_config,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_ROBOT_CONFIG = REPO_ROOT / "robot_models" / "unitree_g1" / "unitree_g1.json"
DEFAULT_FRAMERATE = 30.0

# TRUMANS / SMPL-X world frame: +X left, +Y up, +Z forward.
# Repo/robot world frame: +X forward, +Y left, +Z up.
# world_xyz = smplx_xyz[:, SMPLX_TO_WORLD_AXES] -> (z_smplx, x_smplx, y_smplx).
SMPLX_TO_WORLD_AXES = (2, 0, 1)


# --------------------------------------------------------------------------- #
# dataset discovery
# --------------------------------------------------------------------------- #
def resolve_dataset_root(path: str) -> Path:
    """Accept either the Data_release folder or its parent."""
    root = Path(path).expanduser().resolve()
    if (root / "seg_name.npy").is_file():
        return root
    nested = root / "Data_release"
    if (nested / "seg_name.npy").is_file():
        return nested
    raise SystemExit(
        f"error: no seg_name.npy found under {root!r} or {nested!r}. "
        "Point --dataset_root at the Data_release folder (or its parent)."
    )


def referenced_scene_names(dataset: Path) -> list[str]:
    """Return scene names that are actually referenced by motion frames."""
    scene_list = np.load(dataset / "scene_list.npy", allow_pickle=True)
    scene_flag = np.load(dataset / "scene_flag.npy", mmap_mode="r")
    used_flags = set(int(f) for f in np.unique(scene_flag))
    return sorted({str(scene_list[i]) for i in used_flags})


def match_scene(names: list[str], terrain_name: str) -> str:
    """Exact match first, then unique case-insensitive substring match."""
    if terrain_name in names:
        return terrain_name
    q = terrain_name.lower()
    hits = [name for name in names if q in name.lower()]
    if not hits:
        raise SystemExit(
            f"error: no terrain/scene matches {terrain_name!r}. "
            f"Use a full scene UUID from the dataset (e.g. {names[0]!r})."
        )
    if len(hits) > 1:
        preview = "\n".join(f"  {name}" for name in hits)
        raise SystemExit(
            f"error: {terrain_name!r} matches {len(hits)} terrains. "
            f"Use an exact scene UUID.\n{preview}"
        )
    return hits[0]


def scene_flags_for_name(dataset: Path, scene_name: str) -> set[int]:
    scene_list = np.load(dataset / "scene_list.npy", allow_pickle=True)
    return {int(i) for i, name in enumerate(scene_list) if str(name) == scene_name}


def iter_scene_segments(dataset: Path, scene_flags: set[int]):
    """Yield dicts for motion segments whose scene_flag is in *scene_flags*."""
    seg = np.load(dataset / "seg_name.npy", mmap_mode="r", allow_pickle=True)
    scene_flag = np.load(dataset / "scene_flag.npy", mmap_mode="r")
    frame_id = np.load(dataset / "frame_id.npy", mmap_mode="r")

    n_frames = len(seg)
    boundaries = np.concatenate(
        [[0], np.where(seg[1:] != seg[:-1])[0] + 1, [n_frames]]
    )

    for k in range(len(boundaries) - 1):
        start = int(boundaries[k])
        end = int(boundaries[k + 1])
        flag = int(scene_flag[start])
        if scene_flags and flag not in scene_flags:
            continue
        name = str(seg[start])
        yield {
            "name": name,
            "scene_flag": flag,
            "start": start,
            "end": end,
            "n_frames": end - start,
            "frame_id_start": int(frame_id[start]),
            "frame_id_end": int(frame_id[end - 1]),
        }


# --------------------------------------------------------------------------- #
# TRUMANS frame conversion (SMPL-X world -> repo/robot world)
# --------------------------------------------------------------------------- #
def convert_terrain_to_world(mesh_path: Path, out_path: Path) -> None:
    """Convert a TRUMANS scene mesh from SMPL-X frame (+Y up) to repo frame (+Z up)."""
    mesh = trimesh.load(str(mesh_path), force="mesh")
    mesh.vertices = mesh.vertices[:, SMPLX_TO_WORLD_AXES].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_path))


def write_world_motion_npy(
    human_joints, seg: dict, out_path: Path
) -> None:
    """Write one segment's first 22 SMPL-X body joints as a repo-frame .npy.

    ``human_joints.npy`` has shape (T, 24, 3); the first 22 joints follow the
    SMPL-X body-joint order used by the repo's smplx data source.
    """
    native = np.asarray(
        human_joints[seg["start"] : seg["end"], :22, :], dtype=np.float32
    )
    world = native[:, :, SMPLX_TO_WORLD_AXES]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, world)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="truemans_batch.py",
        description="Batch retarget TRUMANS motions for one terrain/scene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_root", required=True,
                        help="TRUMANS dataset root (Data_release folder or its parent)")
    parser.add_argument("--terrain_name", required=True,
                        help="Scene UUID to query motions for (exact match preferred)")
    parser.add_argument("--num_motions", type=int, required=True,
                        help="Maximum number of motions to retarget")
    parser.add_argument("--robot-config", default=str(DEFAULT_ROBOT_CONFIG),
                        help="Robot profile JSON used for retargeting")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: /tmp/truemans_batch/<terrain_name>)")
    parser.add_argument("--framerate", type=float, default=DEFAULT_FRAMERATE,
                        help="Framerate forwarded to omniretargeting.main")
    parser.add_argument("--timeout", type=float, default=3600,
                        help="Per-motion timeout in seconds")
    parser.add_argument("--scale-factor", type=float, default=None,
                        help="Use one fixed scene scale for every motion instead of "
                             "per-motion height-based scaling")
    parser.add_argument("--penetration-resolver",
                        choices=["hard_constraint", "hard_constraint_slack", "xyz_nudge"],
                        default="hard_constraint_slack",
                        help="Contact handling mode forwarded to omniretargeting.main")
    parser.add_argument("--progress", action="store_true",
                        help="Forward --progress to omniretargeting.main")
    parser.add_argument("--dry-run", action="store_true",
                        help="List the selected motions and planned commands, then exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_motions <= 0:
        print("error: --num_motions must be positive")
        sys.exit(2)
    if args.scale_factor is not None and args.scale_factor <= 0:
        print("error: --scale-factor must be a finite positive number")
        sys.exit(2)

    dataset = resolve_dataset_root(args.dataset_root)

    names = referenced_scene_names(dataset)
    scene_name = match_scene(names, args.terrain_name)
    terrain_path = dataset / "Scene_mesh" / f"{scene_name}.obj"
    if not terrain_path.is_file():
        raise SystemExit(f"error: terrain mesh not found: {terrain_path}")

    scene_flags = scene_flags_for_name(dataset, scene_name)
    all_segments = list(iter_scene_segments(dataset, scene_flags))
    all_segments.sort(key=lambda seg: seg["name"])
    selected = all_segments[: args.num_motions]

    print(f"dataset    : {dataset}")
    print(f"terrain    : {scene_name}")
    print(f"terrain obj: {terrain_path}")
    print(f"scene flags: {sorted(scene_flags)}")
    print(f"motions    : {len(all_segments)} total segments")
    print(f"selected   : {len(selected)} motion(s)\n")
    for seg in selected:
        print(f"  - {seg['name']}  ({seg['n_frames']} frames, "
              f"frame_id {seg['frame_id_start']}..{seg['frame_id_end']})")

    if not selected:
        raise SystemExit("error: no motions found for this terrain")

    robot_config = str(Path(args.robot_config).expanduser())
    if not Path(robot_config).is_file():
        raise SystemExit(f"error: robot config not found: {args.robot_config}")

    output_dir = Path(args.output_dir) if args.output_dir else Path("/tmp/truemans_batch") / scene_name
    staging_dir = output_dir / "source_npy"
    motions_dir = output_dir / "motions"
    terrain_dir = output_dir / "terrain"
    terrain_world_path = terrain_dir / f"{scene_name}_world.obj"

    activation_prefix = get_activation_prefix()
    if activation_prefix:
        print(f"environment : detected ({activation_prefix})")
    else:
        print("environment : no conda/virtualenv detected; using bare Python")

    if args.scale_factor is not None:
        scaling_args = ["--scale-factor", str(args.scale_factor)]
        print(f"scaling    : uniform scene scale {args.scale_factor} (no per-motion scaling)")
    else:
        scaling_args = ["--enable-scene-scaling"]
        print("scaling    : per-motion height-based scene scaling")
    print(f"convention : SMPL-X (+Y up) -> repo/robot (+Z up) "
          f"(axes {SMPLX_TO_WORLD_AXES})")
    print(f"penetration: {args.penetration_resolver}")

    if args.dry_run:
        print("\nDry run: no retargeting performed.\n")
        for seg in selected:
            motion_path = staging_dir / f"{seg['name']}_world.npy"
            config_path = output_dir / "configs" / f"{seg['name']}_config.yaml"
            output_path = motions_dir / f"{seg['name']}_retargeted.npz"
            cmd = [
                sys.executable, "-m", "omniretargeting.main",
                "--robot-config", robot_config,
                "--source-config", str(config_path),
                "--output", str(output_path),
                "--framerate", str(args.framerate),
                "--penetration-resolver", args.penetration_resolver,
                *scaling_args,
            ]
            print(f"{seg['name']}:")
            print(f"  motion -> {motion_path}")
            print(f"  terrain-> {terrain_world_path}")
            print(f"  config -> {config_path}")
            print(f"  output -> {output_path}")
            print(f"  cmd    : {' '.join(cmd)}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    motions_dir.mkdir(parents=True, exist_ok=True)
    terrain_dir.mkdir(parents=True, exist_ok=True)

    print(f"output dir : {output_dir}")

    # Convert the terrain once; every motion in this scene shares it.
    if not terrain_world_path.exists():
        print(f"converting terrain mesh -> {terrain_world_path}")
        convert_terrain_to_world(terrain_path, terrain_world_path)
    else:
        print(f"reusing converted terrain mesh: {terrain_world_path}")

    human_joints = np.load(dataset / "human_joints.npy", mmap_mode="r")

    results = []
    for idx, seg in enumerate(selected, start=1):
        motion_path = staging_dir / f"{seg['name']}_world.npy"
        try:
            write_world_motion_npy(human_joints, seg, motion_path)
        except Exception as exc:
            print(f"[{idx}/{len(selected)}] {seg['name']}: conversion failed: {exc}")
            results.append({
                "returncode": -1, "elapsed": 0.0, "log_file": "",
                "motion_file": str(motion_path), "motion_stem": seg["name"],
                "error": f"conversion failed: {exc}",
            })
            continue

        config_path = write_source_config(
            motion_path,
            "smplx",
            output_dir,
            terrain_path=str(terrain_world_path),
        )
        output_path = motions_dir / f"{seg['name']}_retargeted.npz"
        log_file = output_dir / "logs" / f"{seg['name']}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "-m", "omniretargeting.main",
            "--robot-config", robot_config,
            "--source-config", str(config_path),
            "--output", str(output_path),
            "--framerate", str(args.framerate),
            "--penetration-resolver", args.penetration_resolver,
            *scaling_args,
        ]
        if args.progress:
            cmd.append("--progress")

        print(f"[{idx}/{len(selected)}] {seg['name']} "
              f"({seg['n_frames']} frames)")
        result = run_single_job(cmd, activation_prefix, log_file, args.timeout)
        result["motion_file"] = str(motion_path)
        result["motion_stem"] = seg["name"]
        results.append(result)

        status = "TIMED OUT" if result.get("timed_out") else (
            f"OK ({result['elapsed']:.1f}s)" if result["returncode"] == 0
            else f"FAILED (rc={result['returncode']})"
        )
        print(f"  {status}  log={log_file}")

    failed = _summarize(results)
    print(f"Converted source motions kept under {staging_dir}")
    print(f"Converted terrain mesh kept at {terrain_world_path}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
