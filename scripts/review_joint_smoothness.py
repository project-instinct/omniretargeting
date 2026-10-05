"""Audit source-relative chatter and mechanical joint limits."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from omniretargeting._sequence.continuity_guard import ACCEPTANCE_SLACK
from omniretargeting.motion_budgets import CHATTER_EXCESS_TOLERANCE, chatter_excess


def _joint_limits(model_path, model):
    """Read URDF/MJCF velocity and effort limits by MuJoCo joint name."""
    velocity, effort = {}, {}
    try:
        root = ET.parse(model_path).getroot()
    except (OSError, ET.ParseError):
        root = None
    if root is not None:
        for joint in root.findall(".//joint"):
            name = joint.get("name")
            limit = joint.find("limit")
            if name is None or limit is None:
                continue
            if limit.get("velocity") is not None:
                velocity[name] = float(limit.get("velocity"))
            if limit.get("effort") is not None:
                effort[name] = float(limit.get("effort"))
    for index in range(model.njnt):
        name = model.joint(index).name
        if model.jnt_actfrclimited[index]:
            lower, upper = model.jnt_actfrcrange[index]
            effort.setdefault(name, max(abs(float(lower)), abs(float(upper))))
    return velocity, effort


def _velocities(model, poses, fps):
    """Finite-difference generalized velocities matching model.nv."""
    result = np.empty((len(poses) - 1, model.nv))
    for frame in range(len(poses) - 1):
        mujoco.mj_differentiatePos(
            model, result[frame], 1.0 / fps, poses[frame], poses[frame + 1]
        )
    return result


def _inverse_forces(model, poses, fps):
    """Contact-free generalized inverse dynamics at centered stencils."""
    if len(poses) < 3:
        return np.empty((0, model.nv)), np.empty(0, int)
    velocity = _velocities(model, poses, fps)
    acceleration = np.diff(velocity, axis=0) * fps
    data = mujoco.MjData(model)
    force = np.empty((len(poses) - 2, model.nv))
    stencil_frames = np.arange(len(poses) - 2)
    for index, frame in enumerate(stencil_frames):
        data.qpos[:] = poses[frame + 2]
        data.qvel[:] = velocity[frame + 1]
        data.qacc[:] = acceleration[frame]
        mujoco.mj_inverse(model, data)
        force[index] = data.qfrc_inverse
    return force, stencil_frames


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
    dofs = model.jnt_dofadr[joints]
    names = [model.joint(int(joint)).name for joint in joints]
    velocity_limits, effort_limits = _joint_limits(model_path, model)
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
    forces, force_frames = _inverse_forces(model, after, fps)
    chatter_violations = int(np.count_nonzero(chatter > CHATTER_EXCESS_TOLERANCE))
    velocity_violations = 0
    effort_violations = 0
    for column, name in enumerate(names):
        limit = velocity_limits.get(name)
        speed = np.abs(velocity[:, column])
        speed_reference = np.abs(velocity_reference[:, column])
        over_speed = np.zeros(len(speed), bool)
        if limit is not None:
            over_speed = speed > limit + ACCEPTANCE_SLACK * fps
            velocity_violations += int(over_speed.sum())
        effort_limit = effort_limits.get(name)
        force = np.abs(forces[:, dofs[column]]) if len(forces) else np.empty(0)
        over_force = np.zeros(len(force), bool)
        if effort_limit is not None:
            over_force = force > effort_limit + ACCEPTANCE_SLACK * fps
            effort_violations += int(over_force.sum())

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
        rows[name]["mechanical_limits"]["effort"] = {
            "peak": float(force.max(initial=0)),
            "violations": int(over_force.sum()),
            "worst_frame": (
                int(force_frames[np.argmax(force)]) if len(force) and over_force.any() else None
            ),
            "contact_force_split": False,
        }

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
        "all_individual_budgets_pass": violations == 0,
        "joints": rows,
        "root_source_relative_chatter": root_chatter,
        "absolute_smoothness_certificate": False,
        "scope": (
            "Added high-frequency joint chatter relative to the fixed source-fit "
            "trajectory, URDF velocity limits, and contact-free inverse-dynamics "
            "effort screening. Effort needs a contact-force split for final "
            "certification; deliberate low-frequency source/solver speed is not chatter."
        ),
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
