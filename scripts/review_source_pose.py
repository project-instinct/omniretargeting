"""Measure mapped limb directions against the immutable source skeleton."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


def evaluate(prepared, motion):
    inputs = np.load(prepared, allow_pickle=True).item()
    saved = np.load(motion)
    poses = saved["qpos"]
    start = int(saved.get("frame_start", 0))
    model = mujoco.MjModel.from_xml_path(inputs["model_path"])
    names = list(inputs["mapping"])
    bodies = [model.body(inputs["mapping"][n]).id for n in names]
    source_ids = [list(inputs["source_names"]).index(n) for n in names]
    source = inputs["positions"][start : start + len(poses), source_ids]
    mapped = {b: i for i, b in enumerate(bodies)}
    edges = []
    for i, body in enumerate(bodies):
        parent = int(model.body_parentid[body])
        while parent and parent not in mapped:
            parent = int(model.body_parentid[parent])
        if parent in mapped:
            edges.append((mapped[parent], i))
    data = mujoco.MjData(model)
    angles = np.full((len(poses), len(edges)), np.nan)
    targets = inputs.get("body_orientation_targets", ())
    orientation_ids = [
        (
            model.body(t["robot_body"]).id,
            model.body(t["parent_body"]).id if t.get("parent_body") else None,
        )
        for t in targets
    ]
    orientation_errors = np.zeros((len(poses), len(targets)))
    for f, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        for k, (a, b) in enumerate(edges):
            desired = source[f, b] - source[f, a]
            actual = data.xpos[bodies[b]] - data.xpos[bodies[a]]
            norms = np.linalg.norm(desired) * np.linalg.norm(actual)
            if norms > 1e-12:
                angles[f, k] = np.arccos(np.clip(actual @ desired / norms, -1, 1))
        for k, (body, parent) in enumerate(orientation_ids):
            actual = data.xmat[body].reshape(3, 3)
            if parent is not None:
                actual = data.xmat[parent].reshape(3, 3).T @ actual
            orientation_errors[f, k] = Rotation.from_matrix(
                targets[k]["rotations"][start + f] @ actual.T
            ).magnitude()
    parts = {}
    for k, (a, b) in enumerate(edges):
        values = angles[:, k]
        if np.isfinite(values).any():
            parts[f"{names[a]} -> {names[b]}"] = {
                "mean_rad": float(np.nanmean(values)),
                "p95_rad": float(np.nanquantile(values, 0.95)),
                "max_rad": float(np.nanmax(values)),
                "worst_frame": int(np.nanargmax(values) + start),
            }
    return {
        "motion": str(motion),
        "metric": "mapped bone directions in world axes",
        "source_modified": False,
        "mean_rad": float(np.nanmean(angles)),
        "p95_rad": float(np.nanquantile(angles, 0.95)),
        "parts": parts,
        "calibrated_body_orientation": {
            target["robot_body"]: {
                "reference_frame": target.get("parent_body", "world"),
                "mean_rad": float(orientation_errors[:, k].mean()),
                "p95_rad": float(np.quantile(orientation_errors[:, k], 0.95)),
                "max_rad": float(orientation_errors[:, k].max()),
                "worst_frame": int(np.argmax(orientation_errors[:, k]) + start),
            }
            for k, target in enumerate(targets)
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.prepared, args.motion)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
