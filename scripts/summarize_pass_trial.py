"""Compare original, matched control, bounded trial, and fresh trial honestly."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from review_contact_metrics import evaluate, summary
from review_motion_displacement import measure
from review_source_pose import evaluate as source_pose
from trumans_review_inputs import restore_surface

from omniretargeting._sequence.continuity_guard import SequenceContinuityGuard
from omniretargeting._sequence.pose_guard import SequencePoseGuard
from omniretargeting._sequence.surface_guard import SurfaceSequenceGuard
from omniretargeting.contact_schedule import approach_blend
from omniretargeting.sequence_retargeting import prepare_sequence


def summarize(root, baseline, bounded_dir="bounded", control_dir="bounded_control"):
    case = root.name
    inputs = np.load(baseline / "prepared.npy", allow_pickle=True).item()
    decisions = np.load(root / "decisions.npy", allow_pickle=True).item()
    prepared = root / "prepared.npy"
    choices = dict(np.load(baseline / "contact_choices.npz"))
    paths = {
        "original": baseline / "final.npz",
        "matched_control": root / control_dir / "trial.npz",
        "bounded_pass": root / bounded_dir / "trial.npz",
        "fresh_pass": root / "final.npz",
    }
    scene_paths = {
        "original": baseline / "final_scene.json",
        "matched_control": root / control_dir / "scene.json",
        "bounded_pass": root / bounded_dir / "scene.json",
        "fresh_pass": root / "final_scene.json",
    }
    for path in [*paths.values(), *scene_paths.values()]:
        if not path.exists():
            raise FileNotFoundError(f"Incomplete comparison: {path}")
    model = mujoco.MjModel.from_xml_path(inputs["model_path"])
    data = mujoco.MjData(model)
    original = np.load(paths["original"])["qpos"]
    trial_inputs = np.load(prepared, allow_pickle=True).item()
    config = json.loads((baseline / "stage_configs.json").read_text())["final"]
    surface = restore_surface(
        model, trial_inputs["surface"], trial_inputs["surface_config"]
    )
    _, _, tasks, rotations = prepare_sequence(
        model,
        original,
        inputs["positions"],
        inputs["source_names"],
        inputs["mapping"],
        inputs["roots"],
        trial_inputs["tasks"],
        inputs["effectors"],
        config,
        environment_points=inputs["environment_points"],
        root_orientation_targets=inputs["root_orientation_targets"],
        body_orientation_targets=inputs["body_orientation_targets"],
        contact_offset_choice=choices,
        extra_tasks=(() if surface is None else (surface,)),
    )
    pose_guard = SequencePoseGuard(tasks, original)
    continuity_guard = SequenceContinuityGuard(tasks, original, rotations)
    contact_guard = SurfaceSequenceGuard(
        tasks, original, config["endpoint_regression_m"]
    )
    results, signals = {}, {}
    fps = float(np.load(paths["original"])["framerate"])
    active = np.zeros(len(original), bool)
    event_mask = active.copy()
    for name, decision in decisions.items():
        task = inputs["tasks"][name]
        _, _, assigned = approach_blend(
            task.tracking_positions,
            task.contact_positions,
            task.contact_mask,
            return_contact_frames=True,
        )
        active |= decision.pass_mask | (
            decision.pass_mask[assigned] & (task.approach[0] > 0)
        )
        event_mask |= decision.pass_mask
    regions = [n for n, d in decisions.items() if d.pass_mask.any()]
    for name, path in paths.items():
        poses = np.load(path)["qpos"]
        metrics = evaluate(prepared, path, choices)
        delta = poses[:, :3] - inputs["roots"]
        signal = {
            "root_step_mm": np.r_[
                0, np.linalg.norm(np.diff(delta, axis=0), axis=1) * 1000
            ],
            "joint_acc": np.r_[
                0, 0, np.linalg.norm(np.diff(poses[:, 7:], n=2, axis=0), axis=1)
            ],
        }
        dropped = {}
        for region in regions:
            spec = inputs["effectors"]["effectors"][region]
            body = model.body(spec["robot_body"]).id
            wanted = inputs["tasks"][region]
            distance, normal_error, world = [], [], []
            for frame, pose in enumerate(poses):
                data.qpos[:] = pose
                mujoco.mj_forward(model, data)
                rotation = data.xmat[body].reshape(3, 3)
                point = data.xpos[body] + rotation @ np.asarray(
                    spec["contact_offsets"][choices[region][frame]]
                )
                residual = point - wanted.contact_positions[frame]
                distance.append(float(np.linalg.norm(residual)))
                normal_error.append(float(residual @ wanted.surface_normals[frame]))
                world.append(data.xpos[body] + rotation @ spec["tracking_offset"])
            mask = decisions[region].pass_mask
            dropped[region] = {
                "vector_m": summary(np.asarray(distance)[mask]),
                "signed_normal_m": summary(np.asarray(normal_error)[mask]),
            }
            signal[f"{region}_gap_mm"] = np.asarray(normal_error) * 1000
            signal[f"{region}_world"] = np.asarray(world)
        displacement = measure(model, original, poses, 0.02)
        guarded_change = pose_guard.accepts(poses)
        metrics.update(
            dropped_candidate_errors=dropped,
            displacement_from_original=displacement,
            within_existing_cumulative_pose_budget=guarded_change,
            within_existing_continuity_budget=continuity_guard.accepts(poses),
            within_retained_contact_budget=contact_guard.accepts(poses),
            source_pose=source_pose(prepared, path),
            scene=json.loads(scene_paths[name].read_text()),
            influenced_root_step_max_m=float(
                signal["root_step_mm"][active].max() / 1000
            ),
            influenced_joint_acc_rms=float(
                np.sqrt(np.mean(np.diff(poses[:, 7:], n=2, axis=0)[active[2:]] ** 2))
            ),
        )
        results[name] = metrics
        signals[name] = signal
    report = {
        "case": case,
        "frames": len(original),
        "fps": fps,
        "changed_frames": np.flatnonzero(event_mask).tolist(),
        "influenced_frames": np.flatnonzero(active).tolist(),
        "results": results,
        "bounded_checks": json.loads((root / bounded_dir / "checks.json").read_text()),
        "control_checks": json.loads((root / control_dir / "checks.json").read_text()),
        "uncertainty_calibrated": False,
        "mechanical_support_verified": False,
        "default_enabled": False,
    }
    (root / "comparison.json").write_text(json.dumps(report, indent=2))
    selected = np.flatnonzero(active)
    time = np.arange(len(original)) / fps
    figures, axes = plt.subplots(
        3, 1, figsize=(10, 8), sharex=True, constrained_layout=True
    )
    labels = {
        "original": "Original",
        "matched_control": "Same extra iterations; original contacts",
        "bounded_pass": "Bounded near-pass trial",
        "fresh_pass": "REJECTED: fresh solve exceeds pose budget",
    }
    for name, signal in signals.items():
        marker_change = np.linalg.norm(
            signal[f"{regions[0]}_world"] - signals["original"][f"{regions[0]}_world"],
            axis=1,
        )
        axes[0].plot(time, marker_change * 1000, label=labels[name])
        axes[1].plot(time, signal["root_step_mm"])
        axes[2].plot(time, signal["joint_acc"])
    for axis in axes:
        axis.axvspan(
            time[event_mask].min(), time[event_mask].max(), color="grey", alpha=0.15
        )
        axis.set_xlim(time[selected[0]], time[selected[-1]])
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Wrist-marker change from original (mm)")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("Root correction step (mm/frame)")
    axes[2].set_ylabel("Joint acceleration norm (rad/frame²)")
    axes[2].set_xlabel("Original recording time (s)")
    figures.suptitle(f"{case}: unchanged source, scene, and timing")
    figures.savefig(root / "comparison.png", dpi=160)
    plt.close(figures)
    print(
        json.dumps(
            {
                n: {
                    k: r[k]
                    for k in (
                        "joint_acc_rms_rad_per_frame2",
                        "influenced_joint_acc_rms",
                        "influenced_root_step_max_m",
                        "within_existing_cumulative_pose_budget",
                    )
                }
                for n, r in results.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--bounded-dir", default="bounded")
    parser.add_argument("--control-dir", default="bounded_control")
    args = parser.parse_args()
    summarize(args.trial, args.baseline, args.bounded_dir, args.control_dir)
