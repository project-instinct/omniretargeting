#!/usr/bin/env python
"""Compare retargeted runs of the same clip on the metrics that decide acceptance.

Synthetic tests answer "is the solver self-consistent". They do not answer "does
this look right on the real clip", and the two have already disagreed once: a
change that measured sub-millimetre on achievable synthetic contacts made the
real motion jitter 27x worse and pushed the wrist 10 cm into the scene.

So this reads the metrics the pipeline already exports in
``evaluation_metrics_json`` and lays several runs side by side.  It computes
nothing new; ``omniretargeting.evaluation`` remains the single source of truth.

    python scripts/compare_real_clip.py output/a.npz output/b.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

EFFECTORS = ("left_foot", "right_foot", "left_hand", "right_hand")


def _load(path: Path) -> dict[str, Any]:
    payload = np.load(path, allow_pickle=True)
    run: dict[str, Any] = {"path": path, "metrics": None}
    if "evaluation_metrics_json" in payload.files:
        run["metrics"] = json.loads(str(payload["evaluation_metrics_json"]))
    names = [str(n) for n in payload["contact_names"]]
    if "contact_mode" in payload.files:
        contact = payload["contact_mode"] != 0
    else:
        contact = payload["contact_mask"].astype(bool)
    run["contact_frames"] = {n: int(contact[:, i].sum()) for i, n in enumerate(names)}
    joints = payload["qpos"][:, 7:]
    run["jerk_rms"] = float(np.sqrt((np.diff(joints, n=3, axis=0) ** 2).mean()))
    run["accel_max"] = float(np.abs(np.diff(joints, n=2, axis=0)).max())
    run["base_z"] = (float(payload["qpos"][:, 2].min()), float(payload["qpos"][:, 2].max()))
    if "smplx_joints_w" in payload.files:
        pelvis = payload["smplx_joints_w"][:, 0, 2]
        run["pelvis_z"] = (float(pelvis.min()), float(pelvis.max()))
    return run


def _stat(metrics: Any, *keys: str) -> float:
    node = metrics
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return float("nan")
        node = node[key]
    return float(node) if isinstance(node, (int, float)) else float("nan")


def _row(label: str, values: list[float], scale: float = 1.0, width: int = 12) -> str:
    cells = "".join(
        f"{value * scale:>{width}.2f}" if value == value else f"{'n/a':>{width}}"
        for value in values
    )
    return f"  {label:<34s}{cells}"


SMPLX_JOINTS = (
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar",
    "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist",
)

#: Whole-body similarity is the primary acceptance signal, not per-effector
#: contact error.  A run can put both feet exactly on the scene and still be
#: unusable if the torso chain has been dragged 30 cm off the source pose --
#: that is what "the result looks wrong" means in practice, and no end-effector
#: metric reports it.
LIMB_GROUPS = {
    "torso": ("Pelvis", "Spine1", "Spine2", "Spine3", "Neck", "Head"),
    "legs": ("L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle"),
    "arms": ("L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"),
}


def whole_body_similarity(path: Path, robot_config: Path) -> dict[str, np.ndarray] | None:
    """Return per-SMPL-X-joint distance from the robot's mapped link, in metres."""
    try:
        import json

        import mujoco

        from omniretargeting.robot_config import load_robot_config
    except Exception:
        return None
    payload = np.load(path, allow_pickle=True)
    if "smplx_joints_w" not in payload.files:
        return None
    config = load_robot_config(robot_config)
    model = mujoco.MjModel.from_xml_path(config["urdf_path"])
    data = mujoco.MjData(model)
    qpos = payload["qpos"]
    if model.nq != qpos.shape[1]:
        return None
    source = payload["smplx_joints_w"]
    order = {name: index for index, name in enumerate(SMPLX_JOINTS)}
    errors: dict[str, np.ndarray] = {}
    for joint_name, link in config["joint_mapping"].items():
        if joint_name not in order:
            continue
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(link))
        if body_id < 0:
            continue
        distance = np.zeros(len(qpos), dtype=float)
        for frame_idx, frame_qpos in enumerate(qpos):
            data.qpos[:] = frame_qpos
            mujoco.mj_forward(model, data)
            distance[frame_idx] = np.linalg.norm(
                data.xpos[body_id] - source[frame_idx, order[joint_name]]
            )
        errors[joint_name] = distance
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=None,
        help="Robot profile used for whole-body similarity (all runs must share it).",
    )
    args = parser.parse_args()

    runs = [_load(path) for path in args.runs]
    header = "".join(f"{path.stem[:11]:>12}" for path in args.runs)
    print(f"\n{'':36s}{header}")

    if args.robot_config is not None:
        similarity = [whole_body_similarity(p, args.robot_config) for p in args.runs]
        if any(s is not None for s in similarity):
            print("\nWHOLE-BODY SIMILARITY TO THE SOURCE MOTION (cm) -- PRIMARY SIGNAL")
            for group, joints in LIMB_GROUPS.items():
                values = [
                    float(np.mean([s[j].mean() for j in joints if j in s])) * 100
                    if s else float("nan")
                    for s in similarity
                ]
                print(_row(f"{group} mean", values))
            overall = [
                float(np.mean([v.mean() for v in s.values()])) * 100
                if s else float("nan")
                for s in similarity
            ]
            print(_row("ALL JOINTS mean", overall))

    print("\nSOURCE-TRACKING ERROR (cm) -- how far the robot ended up from the human")
    for effector in EFFECTORS:
        print(f"\n {effector}")
        for split in ("swing", "contact"):
            for stat in ("mean", "p95", "max"):
                values = [
                    _stat(r["metrics"], "end_effectors", effector,
                          "source_world_error_m", split, stat)
                    for r in runs
                ]
                if all(v != v for v in values):
                    continue
                print(_row(f"{split:<8s} {stat}", values, 100.0))
        print(_row("contact frames",
                   [float(r["contact_frames"].get(effector, 0)) for r in runs]))

    print("\nSCENE PENETRATION")
    for label, keys, scale in (
        ("max depth (cm)", ("penetration", "max_depth"), 100.0),
        ("p95 depth (cm)", ("penetration", "p95_depth"), 100.0),
        ("frames penetrating", ("penetration", "frames_penetrating"), 1.0),
    ):
        print(_row(label, [_stat(r["metrics"], *keys) for r in runs], scale))

    print("\nSMOOTHNESS (robot joints, rad/frame^k)")
    print(_row("jerk rms", [r["jerk_rms"] for r in runs], 1.0))
    print(_row("accel max", [r["accel_max"] for r in runs], 1.0))

    print("\nROOT")
    print(_row("source root target err mean (cm)",
               [_stat(r["metrics"], "source_root_target_error_m", "mean") for r in runs],
               100.0))
    for run in runs:
        pelvis = run.get("pelvis_z")
        pelvis_text = f"  source pelvis z [{pelvis[0]:.3f}, {pelvis[1]:.3f}]" if pelvis else ""
        print(f"  {run['path'].stem[:30]:<32s} base z "
              f"[{run['base_z'][0]:.3f}, {run['base_z'][1]:.3f}]{pelvis_text}")
    print()


if __name__ == "__main__":
    main()
