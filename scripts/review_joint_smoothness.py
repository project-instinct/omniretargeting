"""Audit source-relative chatter and mechanical joint limits."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from omniretargeting._sequence.continuity_guard import ACCEPTANCE_SLACK
from omniretargeting.motion_budgets import CHATTER_EXCESS_TOLERANCE, chatter_excess
from omniretargeting.effort_audit import effort_report
from omniretargeting.effort_contacts import contact_evidence, load_fixed_objects
from omniretargeting.effort_dynamics import mechanical_limits


def _joint_limits(model_path, model, limits_urdf=None):
    """Read URDF/MJCF velocity and effort limits by MuJoCo joint name."""
    velocity, effort, _ = mechanical_limits(model_path, model, limits_urdf)
    return velocity, effort


def audit(model_path, before_path, after_path, *, limits_urdf=None, prepared=None, objects=None):
    model = mujoco.MjModel.from_xml_path(str(model_path))
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
    if not np.isfinite(before).all() or not np.isfinite(after).all() or not np.isfinite(fps):
        raise ValueError("Joint trajectories must be finite")
    joints = np.flatnonzero(
        np.isin(
            model.jnt_type, [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]
        )
    )
    q = model.jnt_qposadr[joints]
    names = [model.joint(int(joint)).name for joint in joints]
    if not len(joints):
        raise ValueError("Joint audit requires scalar actuated joint coordinates")
    velocity_limits, effort_limits, provenance = mechanical_limits(model_path, model, limits_urdf)
    saved_hash = saved_after.get("collision_model_sha256")
    if saved_hash is not None and str(saved_hash) != provenance["collision_model_sha256"]:
        raise ValueError("Effort audit model does not match saved collision_model_sha256")
    fixed = load_fixed_objects(prepared, objects, len(after), fps)
    evidence = contact_evidence(saved_after, model, fixed)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    efforts, dynamics = effort_report(model, before, after, fps, joints, effort_limits, evidence)
    rows = {
        name: {
            "units": "rad" if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_HINGE else "m",
            "orders": {},
            "mechanical_limits": {
                "velocity_per_second": velocity_limits.get(name),
                "effort": effort_limits.get(name),
            },
        }
        for name, joint in zip(names, joints)
    }

    chatter = chatter_excess(before[:, q], after[:, q])
    root_chatter = {}
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free):
        root_q = int(model.jnt_qposadr[free[0]])
        root_chatter["translation"] = {
            "violations": int(np.count_nonzero(
                chatter_excess(before[:, root_q:root_q + 3],
                               after[:, root_q:root_q + 3]) > CHATTER_EXCESS_TOLERANCE
            )),
        }
        rotation_before = Rotation.from_quat(
            before[:, root_q + 3:root_q + 7], scalar_first=True
        )
        rotation_after = Rotation.from_quat(
            after[:, root_q + 3:root_q + 7], scalar_first=True
        )
        angular_before = (rotation_before[1:] * rotation_before[:-1].inv()).as_rotvec()
        angular_after = (rotation_after[1:] * rotation_after[:-1].inv()).as_rotvec()
        root_chatter["rotation"] = {
            "violations": int(np.count_nonzero(
                chatter_excess(angular_before, angular_after) > CHATTER_EXCESS_TOLERANCE
            )),
        }
    velocity = np.diff(after[:, q], axis=0) * fps
    velocity_reference = np.diff(before[:, q], axis=0) * fps
    chatter_violations = int(np.count_nonzero(chatter > CHATTER_EXCESS_TOLERANCE))
    velocity_violations = 0
    effort_violations = 0
    for column, name in enumerate(names):
        limit = velocity_limits.get(name)
        speed = np.abs(velocity[:, column])
        speed_reference = np.abs(velocity_reference[:, column])
        over_speed = np.zeros(len(speed), bool)
        if limit is not None:
            over_speed = speed > limit + 1e-6
            velocity_violations += int(over_speed.sum())
        effort_violations += efforts[name]["violations"]

        for order in (1, 2, 3):
            first = np.diff(before[:, q], n=order, axis=0)
            second = np.diff(after[:, q], n=order, axis=0)
            if not len(first):
                rows[name]["orders"][order] = {"count": 0, "violations": 0}
                continue
            derivative_excess = np.abs(second[:, column]) - np.abs(first[:, column])
            rows[name]["orders"][order] = {
                "count": len(first),
                "violations": 0,
                "diagnostic_derivative_increase": int(
                    np.count_nonzero(derivative_excess > ACCEPTANCE_SLACK)
                ),
                "max_increase_per_frame_order": float(chatter[:, column].max()),
                "worst_increase_frame": int(np.argmax(chatter[:, column])),
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
        rows[name]["source_relative_chatter"] = {
            "max_excess": float(chatter[:, column].max(initial=0)),
            "worst_frame": int(np.argmax(chatter[:, column])) if len(chatter) else None,
            "violations": int(np.count_nonzero(chatter[:, column] > CHATTER_EXCESS_TOLERANCE)),
        }
        rows[name]["mechanical_limits"]["velocity"] = {
            "peak_per_second": float(speed.max(initial=0)),
            "reference_peak_per_second": float(speed_reference.max(initial=0)),
            "violations": int(over_speed.sum()),
            "worst_frame": int(np.argmax(speed)) if len(speed) else None,
        }
        rows[name]["mechanical_limits"]["effort"] = efforts[name]

    violations = chatter_violations + velocity_violations + effort_violations
    return {
        "before": str(before_path),
        "after": str(after_path),
        "frames": len(before),
        "fps": fps,
        "numerical_slack_per_frame_order": ACCEPTANCE_SLACK,
        "scalar_joint_count": len(joints),
        "violating_joint_time_order_entries": chatter_violations,
        "velocity_limit_violations": velocity_violations,
        "effort_limit_violations": effort_violations,
        "all_individual_budgets_pass": (
            violations == 0 and not dynamics["unknown_frames"]
            and not dynamics["unaudited_endpoint_frames"]
            and not dynamics["missing_effort_limits"]
            and all(name in velocity_limits for name in names)
        ),
        "mechanical_model": provenance,
        "missing_velocity_limits": [name for name in names if name not in velocity_limits],
        "effort_dynamics": dynamics,
        "joints": rows,
        "root_source_relative_chatter": root_chatter,
        "absolute_smoothness_certificate": False,
        "scope": (
            "Added high-frequency joint chatter relative to the fixed source-fit "
            "trajectory, original URDF velocity/effort limits, and instantaneous "
            "contact-force feasibility with exact floating-base balance. Unknown "
            "contacts/object loads and unaudited endpoints do not pass. This is "
            "not physical execution certification."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--limits-urdf", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--objects", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.model, args.before, args.after, limits_urdf=args.limits_urdf,
                   prepared=args.prepared, objects=args.objects)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in {"joints", "effort_dynamics"}}, indent=2))
