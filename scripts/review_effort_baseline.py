"""Reaudit frozen fast outputs without changing motions or rerunning the solver."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from omniretargeting.effort_dynamics import derivatives, mechanical_limits, motor_demand
from scripts.review_joint_smoothness import audit

ROOT = Path(__file__).resolve().parents[1]
CLIPS = ("sit180", "wipe300", "chair150", "fridge957", "omomo_5", "omomo_78",
         "omomo_80", "omomo_485", "omomo_182", "omomo_254")


def _save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legacy_decomposition(model_path, poses, fps, effort):
    """Reproduce the old index/stencil and expose its soft-constraint loads."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    intervals = np.zeros((len(poses) - 1, model.nv))
    for frame in range(len(intervals)):
        mujoco.mj_differentiatePos(model, intervals[frame], 1 / fps, poses[frame], poses[frame + 1])
    acceleration = np.diff(intervals, axis=0) * fps
    frames = []
    for index in range(len(acceleration)):
        data.qpos[:] = poses[index + 2]
        data.qvel[:] = intervals[index + 1]
        data.qacc[:] = acceleration[index]
        mujoco.mj_inverse(model, data)
        smooth = np.empty(model.nv)
        mujoco.mj_mulM(model, data, smooth, acceleration[index])
        smooth += data.qfrc_bias - data.qfrc_passive
        for joint in range(model.njnt):
            name = model.joint(joint).name
            if name not in effort:
                continue
            dof = int(model.jnt_dofadr[joint])
            value = float(data.qfrc_inverse[dof])
            if abs(value) > effort[name] + 1e-6:
                kinds = dict(zip(*[a.tolist() for a in np.unique(data.efc_type[:data.nefc], return_counts=True)]))
                frames.append({"joint": name, "old_report_frame": index,
                               "pose_frame": index + 2, "acceleration_frame": index + 1,
                               "old_inverse_effort_Nm": value,
                               "without_soft_constraints_Nm": float(smooth[dof]),
                               "soft_constraint_contribution_Nm": float(-data.qfrc_constraint[dof]),
                               "limit_Nm": effort[name], "self_contact_count": int(data.ncon),
                               "constraint_types": kinds})
    return frames


def stage_screen(model_path, poses, fps, effort):
    """Raw motor demand is diagnostic only; it has no environment reactions."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    data = mujoco.MjData(model)
    frames, velocity, acceleration = derivatives(model, poses, fps)
    forces = np.array([motor_demand(model, data, poses[frame], velocity[k], acceleration[k])
                       for k, frame in enumerate(frames)])
    rows = {}
    for joint in range(model.njnt):
        name = model.joint(joint).name
        if name not in effort:
            continue
        dof = model.jnt_dofadr[joint]
        value = np.abs(forces[:, dof])
        rows[name] = {"contact_free_peak_Nm": float(value.max()),
                      "worst_frame": int(frames[value.argmax()]),
                      "contact_free_exceedance_frames": frames[value > effort[name] + 1e-6].tolist(),
                      "acceleration_peak_rad_s2": float(np.abs(acceleration[:, dof]).max()),
                      "acceleration_worst_frame": int(frames[np.argmax(np.abs(acceleration[:, dof]))])}
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free):
        dof = model.jnt_dofadr[free[0]]
        linear = np.linalg.norm(acceleration[:, dof:dof + 3], axis=1)
        angular = np.linalg.norm(acceleration[:, dof + 3:dof + 6], axis=1)
        rows["_root"] = {"linear_acceleration_peak_m_s2": float(linear.max()),
                         "linear_acceleration_worst_frame": int(frames[linear.argmax()]),
                         "angular_acceleration_peak_rad_s2": float(angular.max()),
                         "angular_acceleration_worst_frame": int(frames[angular.argmax()])}
    return rows


def run(baseline, output):
    output.mkdir(parents=True, exist_ok=True)
    model_path = ROOT / "robot_models/unitree_g1/g1_29dof_torsobase_mesh.xml"
    limits_path = ROOT / "robot_models/unitree_g1/g1_29dof_popsicle.urdf"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    _, effort, provenance = mechanical_limits(model_path, model, limits_path)
    summary = {"baseline": str(baseline), "baseline_commit": "1d4e2de",
               "model": provenance, "clips": {}, "solver_changed": False}
    for name in CLIPS:
        started = time.perf_counter()
        clip = baseline / "A/normal/repeat_1" / name
        destination = output / name
        destination.mkdir(exist_ok=True)
        motion = clip / "final_retargeted.npz"
        inputs = [motion, clip / "fixed_reference.npz", clip / "stage_0.npz", clip / "before_mesh_repair.npz"]
        hashes = {str(p): _sha(p) for p in inputs}
        old = json.loads((clip / "joint_smoothness.json").read_text())
        result = audit(model_path, clip / "fixed_reference.npz", motion,
                       limits_urdf=limits_path, prepared=clip / "prepared.npy", objects=clip / "objects.npy")
        _save(destination / "joint_smoothness.json", result)
        fps = result["fps"]
        qpos = np.load(motion)["qpos"]
        decomposition = legacy_decomposition(model_path, qpos, fps, effort)
        _save(destination / "legacy_decomposition.json", decomposition)
        stages = {stage: stage_screen(model_path, np.load(path)["qpos"], fps, effort)
                  for stage, path in zip(("final", "fixed_reference", "stage_0", "before_mesh_repair"), inputs)}
        _save(destination / "stage_diagnostics.json", stages)
        assert hashes == {str(p): _sha(p) for p in inputs}, "Read-only reaudit changed its inputs"
        old_joints = {joint: row["mechanical_limits"]["effort"] for joint, row in old["joints"].items()
                      if row["mechanical_limits"]["effort"]["violations"]}
        row = {
            "frames": result["frames"], "fps": fps,
            "old_effort_violations": old["effort_limit_violations"],
            "new_effort_violations": result["effort_limit_violations"],
            "old_velocity_violations": old["velocity_limit_violations"],
            "new_velocity_violations": result["velocity_limit_violations"],
            "old_missing_velocity_limits": sum(x["mechanical_limits"]["velocity_per_second"] is None for x in old["joints"].values()),
            "old_effort_joints": old_joints,
            "new_effort_joints": {joint: x["mechanical_limits"]["effort"] for joint, x in result["joints"].items()
                                  if x["mechanical_limits"]["effort"]["violations"]},
            "frame_status_counts": result["effort_dynamics"]["frame_status_counts"],
            "unknown_frames": result["effort_dynamics"]["unknown_frames"],
            "unaudited_endpoint_frames": result["effort_dynamics"]["unaudited_endpoint_frames"],
            "source_fit_overload_frames": result["effort_dynamics"]["source_fit_unavoidable_overload_frames"],
            "source_derivatives_at_final_pose_overload_frames": result["effort_dynamics"]["source_derivatives_at_final_pose_overload_frames"],
            "overload_frame_attribution": result["effort_dynamics"]["overload_frame_attribution"],
            "corrected_joint_diagnostics": {joint: x["mechanical_limits"]["effort"] for joint, x in result["joints"].items()},
            "old_overloads_still_over_without_soft_constraints": sum(abs(x["without_soft_constraints_Nm"]) > x["limit_Nm"] + 1e-6 for x in decomposition),
            "old_overloads_with_self_contact": sum(x["self_contact_count"] > 0 for x in decomposition),
            "old_overload_decomposition_count": len(decomposition),
            "read_only_motion_sha256": hashes,
            "audit_seconds": time.perf_counter() - started,
            "production_seconds": json.loads((clip / "timing.json").read_text()).get("production_seconds"),
            "stage_root_diagnostics": {stage: data.get("_root") for stage, data in stages.items()},
        }
        summary["clips"][name] = row
        _save(output / "summary.json", summary)
        print(name, row["old_effort_violations"], "->", row["new_effort_violations"],
              row["frame_status_counts"], f"{row['audit_seconds']:.2f}s", flush=True)
    summary["totals"] = {key: sum(row[key] for row in summary["clips"].values())
                         for key in ("old_effort_violations", "new_effort_violations", "new_velocity_violations",
                                     "old_overloads_still_over_without_soft_constraints", "audit_seconds")}
    _save(output / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.baseline.resolve(), args.output.resolve())
