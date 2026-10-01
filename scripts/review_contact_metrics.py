"""Compare robot trajectories against one immutable, prepared contact plan."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from omniretargeting._sequence.proximity import SurfaceProximityTasks, support_points
from omniretargeting.contact import _is_surface_only_contact_mode


def summary(values):
    values = np.asarray(values, float).reshape(-1)
    return {
        "count": len(values),
        "mean": float(values.mean()) if len(values) else None,
        "p95": float(np.quantile(values, 0.95)) if len(values) else None,
        "max": float(values.max()) if len(values) else None,
    }


def evaluate(prepared, motion, choices, *, start=0):
    inputs = np.load(prepared, allow_pickle=True).item()
    saved = np.load(motion)
    poses = saved["qpos"]
    if "frame_start" in saved:
        start = int(saved["frame_start"])
    model = mujoco.MjModel.from_xml_path(inputs["model_path"])
    data = mujoco.MjData(model)
    source_anchor_body = inputs["effectors"].get("source_root_robot_body")
    source_anchor_id = (
        None if source_anchor_body is None else model.body(source_anchor_body).id
    )
    source_anchor_positions = np.zeros((len(poses), 3))
    values = {
        n: {
            "world_position": [],
            "contact_vector": [],
            "contact_normal": [],
            "sticking": [],
        }
        for n in inputs["tasks"]
    }
    surface_values = {}
    surface_normal_values = {}
    surface_points = np.full(
        (len(poses), len(inputs["surface"].get("surface_contact_parts", ())), 3), np.nan
    )
    surface_scene_gaps = {}
    proximity_values = {}
    proximity = (
        SurfaceProximityTasks(model, **inputs["proximity"])
        if inputs.get("proximity") is not None
        else None
    )
    for local, pose in enumerate(poses):
        f = local + start
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        source_anchor_positions[local] = (
            pose[:3] if source_anchor_id is None else data.xpos[source_anchor_id]
        )
        for name, task in inputs["tasks"].items():
            spec = inputs["effectors"]["effectors"][name]
            body = model.body(spec["robot_body"]).id
            rotation = data.xmat[body].reshape(3, 3)
            center = data.xpos[body] + rotation @ spec["tracking_offset"]
            values[name]["world_position"].append(
                np.linalg.norm(center - task.tracking_positions[f])
            )
            if task.contact_mask[f] and not task.managed_frames[f]:
                point = data.xpos[body] + rotation @ np.asarray(
                    spec["contact_offsets"][choices[name][f]]
                )
                delta = point - task.contact_positions[f]
                values[name]["contact_vector"].append(np.linalg.norm(delta))
                normal = task.surface_normals[f]
                values[name]["contact_normal"].append(
                    abs(delta @ normal / np.linalg.norm(normal))
                )
                if not _is_surface_only_contact_mode(task.contact_modes[f]):
                    values[name]["sticking"].append(np.linalg.norm(delta))
        surface = inputs["surface"]
        for k in np.flatnonzero(surface["surface_contact_mask"][f]):
            gid = int(surface["surface_contact_robot_geoms"][k])
            point = (
                data.geom_xpos[gid]
                + data.geom_xmat[gid].reshape(3, 3)
                @ surface["surface_contact_robot_local_points"][k]
            )
            surface_points[local, k] = point
            residual = np.linalg.norm(
                point - surface["surface_contact_targets_w"][f, k]
            )
            part = str(surface["surface_contact_parts"][k])
            surface_values.setdefault(part, []).append(residual)
            normal = surface["surface_contact_normals_w"][f, k]
            normal_length = np.linalg.norm(normal)
            if normal_length > 0:
                source_surface_gap = float(
                    (point - surface["surface_contact_environment_w"][f, k])
                    @ normal
                    / normal_length
                )
                surface_scene_gaps.setdefault(part, []).append(source_surface_gap)
                normal_error = (
                    abs((point - surface["surface_contact_targets_w"][f, k]) @ normal)
                    / normal_length
                )
                surface_normal_values.setdefault(part, []).append(normal_error)
        if proximity is not None:
            for k in np.flatnonzero(proximity.weights[f] > 0):
                gid = proximity.geoms[k]
                normal = proximity.normals[f, k]
                rotation = data.geom_xmat[gid].reshape(3, 3)
                local_point = support_points(model, gid, (-normal @ rotation)[None])[0]
                robot_point = data.geom_xpos[gid] + rotation @ local_point
                source_gap = float(proximity.support_gaps[f, k])
                robot_gap = float((robot_point - proximity.environment[f, k]) @ normal)
                part = str(proximity.parts[k])
                rows = proximity_values.setdefault(
                    part, {"source_gap": [], "robot_gap": [], "error": []}
                )
                rows["source_gap"].append(source_gap)
                rows["robot_gap"].append(robot_gap)
                rows["error"].append(abs(robot_gap - source_gap))
    source = inputs["roots"][start : start + len(poses)]
    surface = inputs["surface"]
    slip = {}
    fps = float(saved.get("framerate", 30.0))
    if len(poses) > 1 and surface_points.shape[1]:
        target = surface["surface_contact_targets_w"][start : start + len(poses)]
        mode = surface["surface_contact_modes"][start : start + len(poses)]
        sid = surface["surface_contact_surface_ids"][start : start + len(poses)]
        sticking = (
            (mode[1:] == "sticking") & (mode[:-1] == "sticking") & (sid[1:] == sid[:-1])
        )
        error_velocity = (
            np.linalg.norm(np.diff(surface_points - target, axis=0), axis=-1) * fps
        )
        for part in np.unique(surface["surface_contact_parts"]):
            group = np.asarray(surface["surface_contact_parts"]) == part
            valid = sticking[:, group] & np.isfinite(error_velocity[:, group])
            slip[str(part)] = summary(error_velocity[:, group][valid])
    delta = poses[:, :3] - source
    anchor_delta = source_anchor_positions - source
    step = np.linalg.norm(np.diff(delta, axis=0), axis=1)
    worst = np.argsort(step)[-10:][::-1]
    desired = Rotation.from_quat(
        inputs["root_orientation_targets"][start : start + len(poses)],
        scalar_first=True,
    )
    actual = Rotation.from_quat(poses[:, 3:7], scalar_first=True)
    error = desired.inv() * actual
    angular_steps = (error[1:] * error[:-1].inv()).as_rotvec()
    actual_angular_steps = (actual[1:] * actual[:-1].inv()).as_rotvec()
    source_angular_steps = (desired[1:] * desired[:-1].inv()).as_rotvec()
    return {
        "motion": str(motion),
        "start": start,
        "frames": len(poses),
        "endpoints_m": {
            n: {metric: summary(v) for metric, v in rows.items()}
            for n, rows in values.items()
        },
        "surface_vector_error_m": {
            name: summary(v) for name, v in surface_values.items()
        },
        "surface_normal_error_m": {
            name: summary(v) for name, v in surface_normal_values.items()
        },
        "surface_sticking_target_relative_slip_m_s": slip,
        "mapped_point_signed_gap_to_observed_plane_m": {
            name: summary(v) for name, v in surface_scene_gaps.items()
        },
        "contact_ownership": {
            name: ("rigid_surface_block" if task.managed_frames.all()
                   else "mixed_surface_endpoint" if task.managed_frames.any() else "endpoint")
            for name, task in inputs["tasks"].items()
        },
        "closest_surface_gap_m": {
            name: {key: summary(vals) for key, vals in rows.items()}
            for name, rows in proximity_values.items()
        },
        "root_xy_offset_m": summary(np.linalg.norm(delta[:, :2], axis=1)),
        "root_z_offset_m": summary(abs(delta[:, 2])),
        "source_root_correspondence": {
            "robot_body": source_anchor_body,
            "floating_base_offset_is_not_anatomical_root_error": source_anchor_id
            is not None,
            "xy_error_m": summary(np.linalg.norm(anchor_delta[:, :2], axis=1)),
            "z_error_m": summary(abs(anchor_delta[:, 2])),
            "correction_step_m": summary(
                np.linalg.norm(np.diff(anchor_delta, axis=0), axis=1)
            ),
        },
        "joint_acc_rms_rad_per_frame2": float(
            np.sqrt(np.mean(np.diff(poses[:, 7:], n=2, axis=0) ** 2))
        ),
        "root_offset_acc_rms_m_per_frame2": float(
            np.sqrt(np.mean(np.diff(delta, n=2, axis=0) ** 2))
        ),
        "root_offset_step_m": summary(step),
        "root_world_step_m": summary(
            np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1)
        ),
        "source_root_world_step_m": summary(
            np.linalg.norm(np.diff(source, axis=0), axis=1)
        ),
        "root_world_acc_rms_m_per_frame2": float(
            np.sqrt(np.mean(np.diff(poses[:, :3], n=2, axis=0) ** 2))
        ),
        "source_root_world_acc_rms_m_per_frame2": float(
            np.sqrt(np.mean(np.diff(source, n=2, axis=0) ** 2))
        ),
        "root_world_orientation_step_rad": summary(
            np.linalg.norm(actual_angular_steps, axis=1)
        ),
        "source_root_orientation_step_rad": summary(
            np.linalg.norm(source_angular_steps, axis=1)
        ),
        "root_orientation_offset_rad": summary(error.magnitude()),
        "root_orientation_offset_step_rad": summary(
            np.linalg.norm(angular_steps, axis=1)
        ),
        "root_orientation_offset_acc_rms_rad_per_frame2": float(
            np.sqrt(np.mean(np.diff(angular_steps, axis=0) ** 2))
        ),
        "worst_root_offset_steps": [
            {"frame": int(i + start + 1), "metres": float(step[i])} for i in worst
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()
    choice_path = args.prepared.parent / "contact_choices.npz"
    if choice_path.exists():
        choices = np.load(choice_path)
    else:
        inputs = np.load(args.prepared, allow_pickle=True).item()
        from omniretargeting._refinement.math import _lock_contact_offsets

        model = mujoco.MjModel.from_xml_path(inputs["model_path"])
        specs = {
            name: dict(
                inputs["effectors"]["effectors"][name],
                body_id=model.body(
                    inputs["effectors"]["effectors"][name]["robot_body"]
                ).id,
                task=task,
            )
            for name, task in inputs["tasks"].items()
        }
        choices = _lock_contact_offsets(model, inputs["initial"], specs)
    report = evaluate(args.prepared, args.motion, choices, start=args.start)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
