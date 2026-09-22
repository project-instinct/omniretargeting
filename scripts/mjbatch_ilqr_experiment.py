"""Run isolated physics iLQR against saved flat-ground retargeting windows.

Install optional dependencies into a separate overlay, not the production venv:
  uv pip install --python .venv/bin/python --target out/mjbatch_ilqr_runtime \
      --no-deps mjbatch==0.1.1 mujoco==3.13.0
  PYTHONPATH=out/mjbatch_ilqr_runtime OPENBLAS_NUM_THREADS=1 \
      .venv/bin/python scripts/mjbatch_ilqr_experiment.py --output out/ilqr_trial

The output consists of experimental candidates and a report. No candidate is
published into the normal pipeline. An iLQR cost improvement is not acceptance.
"""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mujoco
import numpy as np
import trimesh

from omniretargeting._ilqr.model import build_g1_model
from omniretargeting._ilqr.solver import SolverSettings, solve
from omniretargeting._ilqr.tracking import (
    ReferenceTracker,
    TrackingWeights,
    inverse_seed,
    reference_velocity,
    replay,
)
from omniretargeting.contact import build_world_tracking_tasks
from omniretargeting.data_sources.smplx import SmplxDataSource
from omniretargeting.motion_quality import summary, trajectory_metrics
from omniretargeting.robot_config import load_robot_config
from omniretargeting.scene_geometry import SceneQuery, measure_scene_penetration
from omniretargeting.trajectory_validation import (
    TrajectoryValidator,
    local_repair_config,
)


def hashes(paths):
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def write_json(path, value):
    def default(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)

    path.write_text(json.dumps(value, default=default, indent=2, allow_nan=False))


def temporal(poses, fps):
    result = {}
    for order, name in ((2, "acceleration_rad_s2"), (3, "jerk_rad_s3")):
        values = np.diff(poses[:, 7:], n=order, axis=0) * fps**order
        result[name] = dict(
            summary(abs(values)), rms=float(np.sqrt(np.mean(values**2)))
        )
    return result


def measure(
    model, poses, reference, source, names, mapping, priority, tasks, fps, scene
):
    metrics = trajectory_metrics(
        model, poses, source, names, mapping, priority, tasks, fps
    )
    metrics["temporal"] = temporal(poses, fps)
    displacement = poses[:, :3] - reference[:, :3]
    rotation_error = []
    for ref, q in zip(reference, poses):
        delta = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, delta, 1.0, ref, q)
        rotation_error.append(np.linalg.norm(delta[3:6]))
    metrics["root_change"] = {
        "horizontal_m": summary(np.linalg.norm(displacement[:, :2], axis=1)),
        "vertical_m": summary(abs(displacement[:, 2])),
        "rotation_rad": summary(rotation_error),
    }
    geometry = measure_scene_penetration(model, poses, scene)
    metrics["scene_penetration"] = {
        k: v for k, v in geometry.items() if k not in ("per_geom_depth",)
    }
    metrics["contact_evidence"] = (
        "No semantic contacts in these saved flat-ground windows; residuals/slip are n/a"
    )
    return metrics


def plot_comparison(path, reference, variants, fps):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), constrained_layout=True)
    for name, poses in {"reference": reference, **variants}.items():
        jerk = np.diff(poses[:, 7:], n=3, axis=0) * fps**3
        axes[0].plot(
            (np.arange(len(jerk)) + 1.5) / fps,
            np.sqrt(np.mean(jerk**2, axis=1)),
            label=name,
        )
        shift = np.linalg.norm(poses[:, :3] - reference[:, :3], axis=1)
        axes[1].plot(np.arange(len(poses)) / fps, shift * 100, label=name)
        axes[2].plot(np.arange(len(poses)) / fps, poses[:, 2], label=name)
    for ax, ylabel in zip(
        axes, ["Joint jerk RMS (rad/s³)", "Root displacement (cm)", "Root height (m)"]
    ):
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("Source time within window (s)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(output, records):
    text = [
        "# mjbatch iLQR 首轮实验",
        "",
        "输入为已保存的平地重定向短窗；根部、时序和源文件未改写。候选是完整物理 rollout。",
        "模型保留 29 关节和碰撞体，添加显式力矩电机；这不是训练 PD 控制器的等价复现。",
        "没有语义接触标签，因此接触残差和支撑滑移为 n/a；不外推到场景手部交互。",
        "",
        "| 窗口 / 方案 | jerk RMS rad/s³ | 根水平 / 竖直最大偏移 cm | 场景最大深度 cm | 保真验收 |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for record in records:
        for label, entry in [
            ("reference", record["baseline"]),
            *record["variants"].items(),
        ]:
            m = entry["metrics"]
            root = m["root_change"]
            result = (
                "基线"
                if label == "reference"
                else ", ".join(entry["fidelity"]["reasons"])
                or "通过保真；未做完整几何验收"
            )
            text.append(
                f"| {record['window']} / {label} | {m['temporal']['jerk_rad_s3']['rms']:.1f} | "
                f"{root['horizontal_m']['max'] * 100:.2f} / {root['vertical_m']['max'] * 100:.2f} | "
                f"{m['scene_penetration']['max_depth'] * 100:.2f} | {result} |"
            )
    text += [
        "",
        "完整末端误差、接触样本数、源动作幅度/滞后、关节限位和逐轮成本见各窗口 report.json。",
        "独立 serial MuJoCo 回放检查不使用反馈或逐帧重置。图中 jerk 下降须结合根部和末端误差判断。",
        "实验目录中的 candidate.npz 仅供检查，不作为主流程接纳结果。",
    ]
    (output / "report.md").write_text("\n".join(text) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-directory",
        type=Path,
        default=ROOT / "out/motion_quality_c167_timeline/A",
    )
    parser.add_argument("--windows", nargs="+", default=["raised_hands", "bent_knee"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-directory", type=Path, default=ROOT / "assets/body_models"
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--substeps", type=int, default=17)
    parser.add_argument("--slew-weights", type=float, nargs="+", default=[0.0, 10.0])
    args = parser.parse_args()
    import importlib.metadata

    if mujoco.__version__ != "3.13.0":
        parser.error(
            "Run with the isolated mjbatch/MuJoCo 3.13 overlay described above"
        )
    if args.iterations < 0 or args.threads < 1 or args.substeps < 1:
        parser.error("Invalid iteration, thread or substep count")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest_path = args.input_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    scene_path = args.input_directory / "flat_ground.ply"
    source_path = Path(manifest["source"])
    model_path = Path(manifest["robot_model"])
    protected = [manifest_path, scene_path, source_path, model_path]
    protected += [
        args.input_directory / f"{name}_A{ext}"
        for name in args.windows
        for ext in (".npz", ".json")
    ]
    original_hashes = hashes(protected)
    if original_hashes[str(source_path.resolve())] != manifest["source_file_sha256"]:
        raise ValueError("Source motion differs from the saved baseline manifest")
    write_json(
        args.output / "manifest.json",
        {
            "command": sys.argv,
            "protected_inputs": original_hashes,
            "upstream_reference": "kevinzakka/mjbatch@2f169146b643adda65e195d8ad13d997abdd2a2e",
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("mujoco", "mjbatch", "numpy", "scipy")
            },
            "code": hashes(
                [Path(__file__), *sorted((ROOT / "omniretargeting/_ilqr").glob("*.py"))]
            ),
            "scope": "saved real-motion flat-ground windows; no semantic scene-contact adaptation",
        },
    )
    print("Loading immutable source markers", flush=True)
    motion = SmplxDataSource(
        source_path, model_directory=str(args.model_directory)
    ).load()
    scene = trimesh.load(scene_path)
    profile = load_robot_config(manifest["robot_profile"])
    records = []
    for label in args.windows:
        out = args.output / label
        out.mkdir()
        with np.load(args.input_directory / f"{label}_A.npz") as saved:
            reference = saved["postprocessed"].copy()
            source = saved["source"].copy()
            ids = saved["source_frame_ids"].copy()
            fps = float(saved["fps"])
        # Re-evaluating float32 SMPL-X can differ by sub-micrometre roundoff.
        # The saved source positions remain the metric reference; no rewrite.
        source_roundoff = float(np.max(abs(motion.positions[ids] - source)))
        np.testing.assert_allclose(motion.positions[ids], source, atol=1e-6, rtol=0)
        previous = json.loads((args.input_directory / f"{label}_A.json").read_text())
        config = previous["effective_config"]
        priority = config["end_effector_priority"]
        markers = {
            name: {
                k: v[ids]
                if isinstance(v, np.ndarray)
                and v.ndim
                and len(v) == len(motion.positions)
                else v
                for k, v in marker.items()
            }
            for name, marker in motion.metadata["end_effector_markers"].items()
            if name in priority["effectors"]
        }
        tasks = build_world_tracking_tasks(markers)
        original, model, model_info = build_g1_model(
            model_path, scene, fps, args.substeps
        )
        write_json(out / "model.json", model_info)
        # Addition of actuators must not change metric robot kinematics.
        a, b = mujoco.MjData(original), mujoco.MjData(model)
        for pose in reference:
            a.qpos[:], b.qpos[:] = pose, pose
            mujoco.mj_forward(original, a)
            mujoco.mj_forward(model, b)
            np.testing.assert_allclose(a.xpos, b.xpos, atol=1e-12, rtol=0)
        velocity = reference_velocity(model, reference, 1 / fps)
        controls, seed_report = inverse_seed(model, reference, velocity, 1 / fps)
        write_json(out / "initialization.json", seed_report)
        initial = np.r_[reference[0], velocity[0], controls[0]]
        query = SceneQuery(scene, full_geometry=True)
        validation_config = local_repair_config(
            config["local_collision_repair"],
            priority=priority,
            self_collision_enabled=config["self_collision_enabled"],
            joint_limit_safety_fraction=config["joint_limit_safety_fraction"],
        )
        validator = TrajectoryValidator(
            original, reference, query, tasks, priority, validation_config, fps
        )

        def metrics(
            poses,
            original=original,
            reference=reference,
            source=source,
            priority=priority,
            tasks=tasks,
            fps=fps,
        ):
            return measure(
                original,
                poses,
                reference,
                source,
                motion.target_names,
                profile["joint_mapping"],
                priority,
                tasks,
                fps,
                scene,
            )

        record = {
            "window": label,
            "frames": len(reference),
            "source_frame_ids": ids,
            "fps": fps,
            "source_regeneration_max_roundoff_m": source_roundoff,
            "baseline": {"metrics": metrics(reference)},
            "variants": {},
        }
        np.savez_compressed(
            out / "reference.npz",
            qpos=reference,
            qvel=velocity,
            framerate=fps,
            source_joints_w=source,
            source_joint_names=motion.target_names,
        )
        variants = {}
        for slew in args.slew_weights:
            name = f"slew_{slew:g}"
            folder = out / name
            folder.mkdir()
            weights = TrackingWeights(slew_weight=slew)
            problem = ReferenceTracker(
                model,
                reference,
                velocity,
                priority["effectors"],
                "torso_link",
                args.substeps,
                weights=weights,
                threads=args.threads,
            )

            def progress(entry, label=label, name=name):
                print(
                    json.dumps({"window": label, "variant": name, **entry}), flush=True
                )

            print(f"START {label} {name}", flush=True)
            started = perf_counter()
            states, us, solver_report = solve(
                problem,
                initial,
                controls[:-1],
                SolverSettings(iterations=args.iterations),
                progress,
            )
            poses = states[:, : model.nq]
            replayed = replay(model, initial, us, args.substeps)
            replay_error = float(
                np.max(abs(replayed - states[:, : model.nq + model.nv]))
            )
            np.savez_compressed(
                folder / "candidate.npz",
                qpos=poses,
                qvel=states[:, model.nq : model.nq + model.nv],
                ctrl=us,
                framerate=fps,
                source_joints_w=source,
                source_joint_names=motion.target_names,
            )
            result = {
                "weights": asdict(weights),
                "solver": solver_report,
                "serial_replay_max_error": replay_error,
                "serial_replay_passed": replay_error < 1e-8,
                "fidelity": validator.fidelity(poses),
                "metrics": metrics(poses),
                "control_slew_rms_nm": float(
                    np.sqrt(np.mean(np.diff(us, axis=0) ** 2))
                ),
                "control_saturation_fraction": float(
                    np.mean(np.isclose(abs(us), problem.hi))
                ),
                "seconds": perf_counter() - started,
                "production_accepted": False,
                "acceptance_scope": "Fidelity screening only; no production adoption or full interframe geometry acceptance",
            }
            write_json(folder / "report.json", result)
            variants[name] = poses
            record["variants"][name] = result
            write_json(out / "report.json", record)
            print(
                f"DONE {label} {name}: replay={replay_error:.3g} reasons={result['fidelity']['reasons']}",
                flush=True,
            )
        plot_comparison(out / "comparison.png", reference, variants, fps)
        records.append(record)
        write_summary(args.output, records)
    if hashes(protected) != original_hashes:
        raise RuntimeError("Protected input hashes changed during experiment")
    write_json(
        args.output / "completed.json",
        {"protected_inputs_unchanged": True, "windows": args.windows},
    )


if __name__ == "__main__":
    main()
