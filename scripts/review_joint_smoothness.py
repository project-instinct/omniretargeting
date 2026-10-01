"""Audit each scalar joint's derivatives against a fixed prior trajectory."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from omniretargeting._sequence.continuity_guard import ACCEPTANCE_SLACK


def audit(model_path, before_path, after_path):
    model = mujoco.MjModel.from_xml_path(model_path)
    saved_before, saved_after = np.load(before_path), np.load(after_path)
    before, after = saved_before["qpos"], saved_after["qpos"]
    fps = float(saved_before["framerate"])
    if (
        before.shape != after.shape
        or before.shape[1] != model.nq
        or fps <= 0
        or fps != float(saved_after["framerate"])
    ):
        raise ValueError(
            "Joint audit requires the same model, frames, and positive framerate"
        )
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("Joint trajectories must be finite")
    joints = np.flatnonzero(
        np.isin(
            model.jnt_type, [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]
        )
    )
    q = model.jnt_qposadr[joints]
    rows = {
        model.joint(int(j)).name: {
            "units": "rad" if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE else "m",
            "orders": {},
        }
        for j in joints
    }
    violations = 0
    for order in (1, 2, 3):
        first = np.diff(before[:, q], n=order, axis=0)
        second = np.diff(after[:, q], n=order, axis=0)
        for column, joint in enumerate(joints):
            name = model.joint(int(joint)).name
            if not len(first):
                rows[name]["orders"][order] = {"count": 0, "violations": 0}
                continue
            excess = np.abs(second[:, column]) - np.abs(first[:, column])
            count = int(np.count_nonzero(excess > ACCEPTANCE_SLACK))
            violations += count
            rows[name]["orders"][order] = {
                "count": len(first),
                "violations": count,
                "max_increase_per_frame_order": float(excess.max()),
                "worst_increase_frame": int(excess.argmax() + order),
                "before_rms_per_second_order": float(
                    np.sqrt(np.mean(first[:, column] ** 2)) * fps**order
                ),
                "after_rms_per_second_order": float(
                    np.sqrt(np.mean(second[:, column] ** 2)) * fps**order
                ),
                "before_peak_per_second_order": float(
                    np.max(np.abs(first[:, column])) * fps**order
                ),
                "after_peak_per_second_order": float(
                    np.max(np.abs(second[:, column])) * fps**order
                ),
            }
    return {
        "before": str(before_path),
        "after": str(after_path),
        "frames": len(before),
        "fps": fps,
        "numerical_slack_per_frame_order": ACCEPTANCE_SLACK,
        "scalar_joint_count": len(joints),
        "violating_joint_time_order_entries": violations,
        "all_individual_budgets_pass": violations == 0,
        "joints": rows,
        "absolute_smoothness_certificate": False,
        "scope": "No additional per-joint derivative spikes relative to fixed input; existing source/solver variation is not certified absent.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.model, args.before, args.after)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "joints"}, indent=2))
