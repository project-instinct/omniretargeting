"""Evaluate saved A/B/C clips with independent pose and geometry measurements.

Usage: .venv/bin/python scripts/evaluate_body_relations.py out/body_relation_round
Geometry uses the existing HER audit method: capsule axes at 1 mm spacing vs
original scene triangles, independently of the solver's sampled collision rows.
"""

import argparse
import json
import os
from pathlib import Path
import pickle
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "2"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from omniretargeting.contact import _is_surface_only_contact_mode
from omniretargeting.main import _landmark_torso_rotation
from omniretargeting.scene_geometry import SceneQuery
from omniretargeting.trajectory_validation import TrajectoryValidator


def stats(values):
    values = np.asarray(values)
    if not values.size:
        return None
    return dict(mean=float(np.mean(values)), p95=float(np.percentile(values, 95)), max=float(np.max(values)))


def angles(a, b):
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1, 1)))


def fk(model, q):
    data = mujoco.MjData(model)
    positions, rotations = [], []
    for row in q:
        data.qpos[:] = row
        mujoco.mj_forward(model, data)
        positions.append(data.xpos.copy())
        rotations.append(data.xmat.reshape(-1, 3, 3).copy())
    return np.asarray(positions), np.asarray(rotations)


def independent_geometry(model, scene, q):
    started = time.perf_counter()
    data = mujoco.MjData(model)
    ids = np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    offsets, radii, geom_ids = [], [], []
    for g in ids:
        radius, half = model.geom_size[g, :2]
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            half = 0
        elif model.geom_type[g] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            return dict(status="unsupported_geometry", geom=int(g)), np.empty((0, 0))
        local = np.zeros((max(1, int(np.ceil(2 * half / .001)) + 1), 3))
        local[:, 2] = np.linspace(-half, half, len(local))
        offsets.append(local)
        radii.extend([radius] * len(local))
        geom_ids.extend([g] * len(local))
    radii, geom_ids = np.array(radii), np.array(geom_ids)
    depths = []
    for pose in q:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        points = np.concatenate([o @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
                                 for g, o in zip(ids, offsets)])
        _, distance, _ = trimesh.proximity.closest_point(scene, points)
        per = np.zeros(model.ngeom)
        np.maximum.at(per, geom_ids, np.maximum(radii - distance, 0))
        depths.append(per)
    depths = np.array(depths)
    frame, geom = np.unravel_index(np.argmax(depths), depths.shape)
    return dict(status="measured", surface_intersection_m=stats(depths.max(axis=1)),
                worst_frame=int(frame), worst_body=model.body(int(model.geom_bodyid[geom])).name,
                frames_over_1mm=np.flatnonzero(depths.max(axis=1) > .001).tolist(),
                axis_distance_error_bound_m=.0005, signed_solid_depth=False,
                seconds=time.perf_counter() - started), depths


def evaluate(base, clip, geometry):
    folder = base / clip
    case = json.loads((base / "cases.json").read_text())[clip]
    profile = json.loads((base / "robot_profile.json").read_text())
    model = mujoco.MjModel.from_xml_path(profile["urdf_path"])
    scene = trimesh.load(case["scene"], force="mesh", process=False)
    with (folder / "A/validation_context.pkl").open("rb") as f:
        context = pickle.load(f)
    source = np.load(folder / "A/source_arrays.npz")
    names = source["names"].tolist()
    indices = {name: index for index, name in enumerate(names)}
    human = np.asarray(source["positions"], dtype=float)
    baseline = np.load(folder / "A/final_retargeted.npz")["qpos"]
    validator = TrajectoryValidator(model, baseline, SceneQuery(scene), context["tasks"],
                                    context["priority"], context["config"], context["framerate"])
    with (folder / "C/body_relations.pkl").open("rb") as f:
        relations = pickle.load(f)
    with (folder / "A/calibration.pkl").open("rb") as f:
        calibration = pickle.load(f)
    source_rotations = Rotation.from_quat(source["source_joint_orientations"].reshape(-1, 4), scalar_first=True).as_matrix().reshape(len(human), -1, 3, 3)
    landmarks = dict(origin="Pelvis", left_shoulder="L_Shoulder", right_shoulder="R_Shoulder")
    source_trunk = _landmark_torso_rotation(human, indices, **landmarks)
    fps = float(context["framerate"])
    result, arrays = {}, dict(source=human)
    baseline_positions = fk(model, baseline)[0]
    for variant in ("A", "B", "C"):
        path = folder / variant
        payload = np.load(path / "final_retargeted.npz")
        q = payload["qpos"]
        diagnostics = json.loads((path / "final_retargeted.diagnostics.json").read_text())
        positions, rotations = fk(model, q)
        mapped = np.zeros_like(human)
        for name, body in profile["joint_mapping"].items():
            mapped[:, indices[name]] = positions[:, model.body(body).id]
        arrays[variant + "_robot"] = mapped
        row = dict(frames=len(q), fps=fps, relations=relations.metrics(q),
                   exact_equal_A=bool(np.array_equal(q, baseline)),
                   max_abs_qpos_change=float(np.max(np.abs(q - baseline))),
                   max_joint_change_rad=float(np.max(np.abs(q[:, 7:] - baseline[:, 7:]))),
                   rms_joint_change_rad=float(np.sqrt(np.mean((q[:, 7:] - baseline[:, 7:])**2))),
                   root_xy_change_max_m=float(np.linalg.norm(q[:, :2] - baseline[:, :2], axis=-1).max()),
                   root_z_change_m=stats(np.abs(q[:, 2] - baseline[:, 2])),
                   joint_step_rad=stats(np.abs(np.diff(q[:, 7:], axis=0))),
                   joint_acceleration_rms_rad_s2=float(np.sqrt(np.mean(np.diff(q[:, 7:], n=2, axis=0)**2)) * fps**2),
                   joint_jerk_rms_rad_s3=float(np.sqrt(np.mean(np.diff(q[:, 7:], n=3, axis=0)**2)) * fps**3),
                   seconds=json.loads((path / "run_summary.json").read_text())["seconds"],
                   accepted_postprocess_fraction=diagnostics["postprocessing"]["accepted_fraction"],
                   final_equals_framewise=bool(np.array_equal(q, np.load(path / "framewise_qpos.npy"))),
                   local_changed_frames=np.flatnonzero(np.any(q != np.load(path / "before_local_repair_qpos.npy"), axis=1)).tolist(),
                   self_collision="not_verified")
        for label, target, current in (
            ("chest_world_deg", Rotation.from_quat(source["base_orientations"], scalar_first=True).as_matrix(), rotations[:, model.body(profile["joint_mapping"]["Spine3"]).id]),
            ("trunk_landmarks_deg", source_trunk, _landmark_torso_rotation(mapped, indices, **landmarks)),
        ):
            error = np.degrees(Rotation.from_matrix(target @ np.swapaxes(current, -1, -2)).magnitude())
            arrays[variant + "_" + label] = error
            row[label] = stats(error)
        direction_errors = {}
        for first, second in context["priority"]["pose_direction"]["edges"]:
            if first not in profile["joint_mapping"] or second not in profile["joint_mapping"]:
                continue
            a, b = indices[first], indices[second]
            error = angles(human[:, b] - human[:, a], mapped[:, b] - mapped[:, a])
            direction_errors[first + "->" + second] = stats(error)
            arrays[variant + "_" + first + "->" + second] = error
        row["bone_direction_deg"] = direction_errors
        row["parent_relative_orientation_deg"] = {}
        for name, spec in calibration["tasks"].items():
            if "parent_source" not in spec:
                continue
            parent = spec["parent_source"]
            target = source_rotations[:, indices[name]] @ calibration["source"][name].T @ calibration["robot"][name]
            parent_target = source_rotations[:, indices[parent]] @ calibration["source"][parent].T @ calibration["robot"][parent]
            if model.body_parentid[spec["parent_body_id"]] == 0:
                parent_target = Rotation.from_quat(source["base_orientations"], scalar_first=True).as_matrix()
            relative = np.swapaxes(parent_target, -1, -2) @ target
            current = np.swapaxes(rotations[:, spec["parent_body_id"]], -1, -2) @ rotations[:, spec["body_id"]]
            row["parent_relative_orientation_deg"][name] = stats(np.degrees(Rotation.from_matrix(relative @ np.swapaxes(current, -1, -2)).magnitude()))
        current_effectors = validator.effectors(q)
        row["effectors"] = {}
        for name, spec in validator.specs.items():
            task = spec["task"]
            points = current_effectors[name]["position"]
            world_error = np.linalg.norm(points - task.tracking_positions, axis=-1)
            contact, slip = [], []
            for frame in np.flatnonzero(task.contact_mask):
                error = points[frame] - task.contact_positions[frame]
                if _is_surface_only_contact_mode(task.frame(frame)["contact_mode"]):
                    normal = task.surface_normals[frame]
                    error = np.dot(error, normal / np.linalg.norm(normal))
                contact.append(float(np.linalg.norm(error)))
                if frame and task.frame(frame)["contact_mode"] == task.frame(frame - 1)["contact_mode"] == "sticking":
                    slip.append(np.linalg.norm(points[frame] - points[frame - 1]) * fps)
            row["effectors"][name] = dict(world_m=stats(world_error), contact_frames=len(contact),
                                          contact_residual_m=stats(contact), sticking_slip_m_s=stats(slip))
            arrays[variant + "_" + name + "_world_m"] = world_error
        for index, binding in enumerate(relations.bindings):
            errors = np.array([relations.at(pose, index)[0] - relations.targets[t, index] for t, pose in enumerate(q)])
            arrays[variant + "_" + binding.group + "_relation_m"] = np.linalg.norm(errors, axis=-1)
            row["relations"][binding.group]["residual_second_difference_rms_m"] = float(np.sqrt(np.mean(np.diff(errors, n=2, axis=0)**2)))
            delta = positions[:, model.body(binding.point_body).id] - baseline_positions[:, model.body(binding.point_body).id]
            row["relations"][binding.group]["achieved_wrist_change_xy_m"] = stats(np.linalg.norm(delta[:, :2], axis=-1))
            row["relations"][binding.group]["achieved_wrist_change_abs_z_m"] = stats(np.abs(delta[:, 2]))
        collision = diagnostics["local_repair"]["after"]
        row["production_collision"] = {key: collision[key] for key in ("max_depth", "worst_body", "worst_frame", "interframe_max_depth", "interframe_samples", "uncertain_frames", "subdivision_limit_intervals")}
        if geometry:
            if variant == "B" and row["exact_equal_A"]:
                independent, depths = result["A"]["independent_collision"], arrays["A_depth"]
            else:
                independent, depths = independent_geometry(model, scene, q)
            row["independent_collision"] = independent
            arrays[variant + "_depth"] = depths
        # Every saved source, frame and contact-plan array must match A exactly.
        a_payload = np.load(folder / "A/final_retargeted.npz")
        invariant_keys = [k for k in a_payload.files if k.startswith(("source_", "contact_", "end_effector_"))]
        for key in invariant_keys:
            np.testing.assert_array_equal(payload[key], a_payload[key])
        np.testing.assert_array_equal(q, np.load(path / "after_local_repair_qpos.npy"))
        row["saved_invariants_checked"] = invariant_keys
        result[variant] = row
    changes = {}
    for key in ("chest_world_deg", "trunk_landmarks_deg", "left_foot_world_m", "right_foot_world_m",
                "left_hand_relation_m", "right_hand_relation_m"):
        change = arrays["C_" + key] - arrays["A_" + key]
        worst = int(np.argmax(change))
        changes[key] = dict(mean_change=float(change.mean()), worst_regression_frame=worst,
                            worst_regression=float(change[worst]), regressed_frames=np.flatnonzero(change > 1e-8).tolist())
    result["regressions"] = changes
    result["bindings"] = relations.diagnostics
    (folder / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    np.savez(folder / "evaluation_arrays.npz", **arrays)
    print(clip, {v: dict(relation=[round(x["mean_m"] * 1000, 3) for x in result[v]["relations"].values()],
                         chest=result[v]["chest_world_deg"]["mean"], same=result[v]["exact_equal_A"])
                  for v in ("A", "B", "C")}, flush=True)


def visualize(base, clip):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation
    import imageio_ffmpeg

    path = base / clip
    arrays = np.load(path / "evaluation_arrays.npz")
    summary = json.loads((path / "evaluation.json").read_text())
    frames = np.arange(len(arrays["source"]))
    fig, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True, constrained_layout=True)
    for ax, key, label, factor in zip(axes.flat,
        ["left_hand_relation_m", "right_hand_relation_m", "left_foot_world_m", "right_foot_world_m", "chest_world_deg", "trunk_landmarks_deg"],
        ["Left wrist / chest", "Right wrist / chest", "Left foot world", "Right foot world", "Chest world", "Trunk landmarks"],
        [1000, 1000, 1000, 1000, 1, 1]):
        for variant, color in [("A", "#245ca0"), ("C", "#c46219")]:
            ax.plot(frames, arrays[variant + "_" + key] * factor, color=color, label=variant, lw=1.5)
        ax.set_title(label)
        ax.set_ylabel("mm" if factor == 1000 else "degrees")
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    for ax in axes[-1]:
        ax.set_xlabel("Original frame (zero-based)")
    fig.suptitle(f"{clip}: full clip, A baseline vs C sparse body relations; B is exactly A")
    fig.savefig(path / "comparison.png", dpi=160)
    plt.close(fig)
    source = np.load(path / "A/source_arrays.npz")
    names = source["names"].tolist()
    edges = [("Pelvis", "Spine3"), ("Spine3", "L_Shoulder"), ("Spine3", "R_Shoulder"),
             ("Pelvis", "L_Hip"), ("Pelvis", "R_Hip")]
    edges += [(a, b) for a, b in (key.split("->") for key in summary["A"]["bone_direction_deg"]) if (a, b) not in edges]
    ix = [(names.index(a), names.index(b)) for a, b in edges]
    points = arrays["source"][:, sorted({i for edge in ix for i in edge})]
    low, high = points.min(axis=(0, 1)), points.max(axis=(0, 1))
    center = (low + high) / 2
    radius = max(high - low) * .6
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    artists = []
    for col, (key, label, color) in enumerate([("source", "Source skeleton", "#444444"),
                                               ("A_robot", "A baseline robot skeleton", "#245ca0"),
                                               ("C_robot", "C sparse relations robot skeleton", "#c46219")]):
        ax = fig.add_subplot(1, 3, col + 1, projection="3d")
        ax.set_title(label)
        ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
               zlim=(center[2]-radius, center[2]+radius), xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
        ax.set_box_aspect((1, 1, 1)); ax.view_init(18, -55)
        lines = [ax.plot([], [], [], color=color, marker="o", markersize=2, lw=2)[0] for _ in ix]
        artists.append((arrays[key], lines))
    title = fig.suptitle("")

    def draw(frame):
        title.set_text(f"{clip} / frame {frame}/{len(frames)-1} / shared metric world; skeleton QA only")
        for data, lines in artists:
            for (a, b), line in zip(ix, lines):
                xyz = data[frame, [a, b]]
                line.set_data(xyz[:, 0], xyz[:, 1]); line.set_3d_properties(xyz[:, 2])
        return [title]

    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    movie = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / summary["A"]["fps"])
    movie.save(path / "whole_clip_skeletons.mp4", writer=animation.FFMpegWriter(fps=summary["A"]["fps"], bitrate=2200), dpi=100)
    draw(int(np.argmax(arrays["C_left_hand_relation_m"] - arrays["A_left_hand_relation_m"])))
    fig.savefig(path / "worst_relation_regression.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--clip", choices=["010", "1530", "synthetic"])
    parser.add_argument("--geometry", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    for clip in ([args.clip] if args.clip else ["010", "1530", "synthetic"]):
        evaluate(args.output.resolve(), clip, args.geometry)
        if args.visualize:
            visualize(args.output.resolve(), clip)
