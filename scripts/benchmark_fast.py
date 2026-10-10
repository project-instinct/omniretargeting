"""Production-equivalent fast benchmarks; never overwrite a run.

Use --prepare to freeze HEAD and create a manifest. --run-dir selects that run;
--arm A uses the frozen baseline, other arms require an explicit source root.
Normal timing and --trace are separate output trees. Audits never auto-resolve.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TRUMANS = {
    "sit180": ("2023-02-13@22-31-47", 50, 230),
    "wipe300": ("2023-01-16@23-06-03", 120, 420),
    "chair150": ("2023-02-16@17-10-37", 0, 150),
    "fridge957": ("2023-02-19@22-45-43", 0, 957),
    "chair2745": ("2023-02-16@17-10-37", 0, 2745),
}
OMOMO = (5, 78, 80, 485, 182, 254)  # test_diffusion_manip_seq_joints24.p sequence indices
CLIPS = list(TRUMANS) + [f"omomo_{i}" for i in OMOMO]
ARMS = {"A": "baseline", "candidate": "explicit frozen source root"}
THREADS = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "NUMEXPR_NUM_THREADS": "1"}


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + "\n")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def command_output(args):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    return {"args": args, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def source_config(clip):
    if clip in TRUMANS:
        segment, first, stop = TRUMANS[clip]
        return dict(type="trumans", motion="/home/duanxin/Xyk/TRUMANS/Data_release", segment=segment,
                    frame_start=first, frame_stop=stop, model_directory=str(ROOT / "assets/body_models"))
    return dict(type="omomo", motion="/home/duanxin/Datasets/OMOMO/data/test_diffusion_manip_seq_joints24.p",
                sequence_index=int(clip.split("_")[1]), data_root="/home/duanxin/Datasets/OMOMO",
                model_directory=str(ROOT / "assets/body_models"), framerate=30.0)


def prepare(args):
    run = args.run_dir or ROOT / "out" / time.strftime("fast_%H%M%S")
    run = run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    source = run / "baseline_source"
    source.mkdir()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    archive = run / "baseline_source.tar"
    with archive.open("wb") as handle:
        subprocess.run(["git", "archive", head], cwd=ROOT, stdout=handle, check=True)
    with tarfile.open(archive) as bundle:
        bundle.extractall(source)
    if not (source / "assets/body_models").exists():
        (source / "assets").mkdir(exist_ok=True)
        (source / "assets/body_models").symlink_to(ROOT / "assets/body_models", target_is_directory=True)
    # Capture wrapper is part of the benchmark, identical across all arms.
    shutil.copy2(Path(__file__), run / "benchmark_runner.py")
    # The run keeps a self-contained profile: its model path is resolved once here.
    from omniretargeting.robot_config import load_robot_config
    profile = ROOT / "robot_models/unitree_g1" / args.robot_profile
    frozen = dict(json.loads(profile.read_text()), urdf_path=load_robot_config(profile)["urdf_path"])
    save(run / "g1_profile.json", frozen)
    configs = {clip: source_config(clip) for clip in CLIPS}
    save(run / "inputs.json", configs)
    env = {k: v for k, v in os.environ.items() if k.startswith("OMNIRETARGETING_") or k in THREADS}
    manifest = dict(head=head, source_archive_sha256=digest(archive),
        initial_git_status=command_output(["git", "status", "--porcelain"]),
        authorization_initial_state=json.loads((run.parent / "initial_state.json").read_text())
            if (run.parent / "initial_state.json").exists() else None,
        profile_sha256=digest(run / "g1_profile.json"), runner_sha256=digest(run / "benchmark_runner.py"),
        model_path=frozen["urdf_path"],
        hardware={"platform": platform.platform(), "lscpu": command_output(["lscpu"]),
                  "gpu": command_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv"]),
                  "memory": command_output(["free", "-b"])},
        inherited_runtime_env=env, threads=THREADS, torch_threads=4,
        cache_policy="OS page cache retained; no cache drop; fresh subprocess each clip; repeated serial order",
        source_configs=configs, arms=ARMS, physics="unverified: no physical replay/reset acceptance harness configured",
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    model = Path(manifest["model_path"])
    manifest["model_sha256"] = digest(model)
    save(run / "manifest.json", manifest)
    print(run, flush=True)


def child(args):
    import numpy as np
    import torch
    import omniretargeting._fast.pipeline as pipeline
    import omniretargeting.main as cli
    from omniretargeting._fast.runtime_options import OPTIONS
    from omniretargeting.contact import stack_contact_tasks
    from omniretargeting.robot_config import load_robot_config

    for module in (pipeline, cli):
        if not Path(module.__file__).resolve().is_relative_to(args.source_root.resolve()):
            raise RuntimeError(f'Snapshot import escaped source root: {module.__file__}')

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((args.run_dir / "inputs.json").read_text())[args.child]
    save(out / "source.json", config)
    profile = args.run_dir / "g1_profile.json"
    model_path = load_robot_config(profile)["urdf_path"]
    timing, captured = {}, {}
    original, original_repair, create = pipeline.solve_fast_sequence, pipeline.repair_original_meshes, cli.create_data_source
    trace_add = pipeline.PerformanceTrace.add_stage_seconds

    def capture_stage(trace, name, seconds):
        stages = timing.setdefault('stage_seconds', {})
        stages[name] = stages.get(name, 0.) + float(seconds)
        return trace_add(trace, name, seconds)

    def capture_repair(objective, poses, *a, **kw):
        np.savez(out / "before_mesh_repair.npz", qpos=poses)
        np.savez(out / "fixed_reference.npz", qpos=objective.reference,
                 framerate=objective.framerate)
        return original_repair(objective, poses, *a, **kw)

    def adapter(*a, **kw):
        tick = time.perf_counter()
        obj = create(*a, **kw)
        data = obj.load()
        captured["source"] = data
        with (out / "source.pkl").open("wb") as handle:
            pickle.dump(data, handle)
        timing["source_load_seconds"] = time.perf_counter() - tick
        save(out / "effective_input_sha256.json", {"source.pkl": digest(out / "source.pkl"), "source.json": digest(out / "source.json")})
        return obj

    def capture(owner, model, initial, positions, names, mapping, roots, tasks, priority,
                solver_config, common, extra, *a, **kw):
        captured["owner"] = owner
        surface = owner.last_surface_contact_tasks
        prepared = dict(model_path=model_path, initial=initial, positions=positions, source_names=names,
            mapping=mapping, roots=roots, effectors=priority,
            tasks={n: replace(t, contact_phase=None) for n, t in tasks.items()},
            root_orientation_targets=common["root_orientation_targets"],
            body_orientation_targets=common["body_orientation_targets"], environment_points=common["environment_points"],
            surface={} if surface is None else surface.export(), surface_config={} if surface is None else surface.cfg,
            proximity=None if owner.last_surface_proximity_tasks is None else owner.last_surface_proximity_tasks.export(),
            captured_solver_config=dict(solver_config), initialization="model_reference",
            scene_enclosed_free_space=common["scene_query"].enclosed_free_space,
            morphology_calibration=getattr(owner, "last_morphology_calibration", {}))
        np.save(out / "prepared.npy", prepared, allow_pickle=True)
        np.save(out / "objects.npy", np.array(owner._scene_objects, dtype=object), allow_pickle=True)
        common["scene_query"].mesh.export(out / "static_scene.obj")
        tick = time.perf_counter()
        motion, report = original(owner, model, initial, positions, names, mapping, roots,
                                  tasks, priority, solver_config, common, extra, *a, **kw)
        if getattr(owner, "last_fast_reference", None) is not None:
            np.savez(out / "fixed_reference.npz", qpos=owner.last_fast_reference,
                     framerate=common["framerate"], source_joints_w=positions)
        timing["fast_backend_seconds"] = time.perf_counter() - tick
        np.savez(out / "solver_motion.npz", qpos=motion)
        np.savez(out / "stage_0.npz", qpos=owner.last_fast_source_pose,
                 framerate=common["framerate"], source_joints_w=positions)
        np.savez(out / "contact_plan.npz", **stack_contact_tasks(tasks), **prepared["surface"])
        if report.get("contact_offset_choice"):
            np.savez(out / "contact_choices.npz", **report["contact_offset_choice"])
        captured["report"] = report
        save(out / "fast_report.json", report)
        save(out / "final.json", dict(stages=[], main_pipeline=True, effective_config=owner.retargeting_config))
        return motion, report

    torch.set_num_threads(4)
    argv = ["omniretargeting.main", "--source-config", str(out / "source.json"), "--robot-config", str(profile),
            "--solver", "fast", "--output-base-link", "torso_link", "--output", str(out / "final_retargeted.npz")]
    save(out / "invocation.json", dict(argv=argv, source_root=str(args.source_root), runtime_options=OPTIONS.describe(),
        environment={k: v for k, v in os.environ.items() if k.startswith("OMNIRETARGETING_") or k in THREADS}))
    started = time.perf_counter()
    try:
        with patch.object(sys, "argv", argv), patch.object(cli, "create_data_source", adapter), \
             patch.object(pipeline, "solve_fast_sequence", capture), patch.object(pipeline, "repair_original_meshes", capture_repair), \
             patch.object(pipeline.PerformanceTrace, 'add_stage_seconds', capture_stage):
            cli.main()
        (out / "final.npz").symlink_to("final_retargeted.npz")
        z, data = np.load(out / "final.npz"), captured["source"]
        objects = data.metadata.get("scene_objects") or []
        assert np.array_equal(z["source_joints_w"], data.positions)
        assert float(z["terrain_scale"]) == 1.0
        assert len(z["qpos"]) == len(data.positions)
        if objects:
            assert np.array_equal(z["scene_object_pos_w"], np.stack([o["translations"] for o in objects], axis=1))
        save(out / "invariants.json", dict(source_joints=True, terrain_scale=True, object_translations=True, frames=len(data.positions)))
    finally:
        timing["wall_seconds"] = time.perf_counter() - started
        owner = captured.get('owner')
        deadline = getattr(owner, '_production_deadline', None)
        if deadline is not None:
            report = captured.get('report', {})
            timing['production_seconds'] = report.get('deadline', {}).get('production_seconds', deadline.elapsed())
            timing['deadline'] = report.get('deadline', {})
            if report:
                save(out / 'fast_report.json', report)
        save(out / "timing.json", timing)
    print("TIMING", json.dumps(timing), flush=True)
    print("SOURCE_AND_OBJECT_INVARIANTS_PASS", flush=True)


def run_audits(out, source, env, python, *, scene_workers=None, concurrent=True, self_audit_root=None):
    """Audits only read the saved motion. ``concurrent`` starts the independent
    ones together; ``scene_workers`` selects the bit-identical parallel scene
    audit. ``self_audit_root`` adds self-collision/reset geometry, which consumes
    the finished scene audit, from one fixed snapshot shared by every arm. None
    of these change an audit input, threshold or output."""
    motion = out / "final_retargeted.npz"
    if not motion.exists():
        return {"status": "missing_motion", "physics": "unverified", "reset": "unverified"}
    # Acceptance keeps default geometry semantics, independent of solver switches.
    env = {key: value for key, value in env.items() if not key.startswith("OMNIRETARGETING_")}
    prepared = out / "prepared.npy"
    invocation = json.loads((out / "invocation.json").read_text())
    profile = Path(invocation["argv"][invocation["argv"].index("--robot-config") + 1])
    model = json.loads(profile.read_text())["urdf_path"]
    scene_tail = ["--workers", str(scene_workers)] if scene_workers else []
    self_tail = ["--workers", str(scene_workers)] if scene_workers else []
    commands = [
        ("scene_metrics", ["--prepared", str(prepared), "--motion", str(motion), "--objects", str(out / "objects.npy"), "--output", str(out / "scene_metrics.json"), "--batch", "16"] + scene_tail),
        ("source_pose", ["--prepared", str(prepared), "--motion", str(motion), "--output", str(out / "source_pose.json")]),
        ("contact_metrics", ["--prepared", str(prepared), "--motion", str(motion), "--output", str(out / "contact_metrics.json")]),
        ("joint_smoothness", ["--model", model, "--before", str(out / "fixed_reference.npz"), "--after", str(motion), "--prepared", str(prepared), "--objects", str(out / "objects.npy"), "--output", str(out / "joint_smoothness.json")]),
    ]
    jobs = [(name, [str(python), str(source / "scripts" / f"review_{name}.py")] + tail, source, env)
            for name, tail in commands]
    root = Path(self_audit_root or source).resolve()
    dependent = ("training_collision", [
        str(python), str(root / "scripts/review_training_collision.py"), "--prepared", str(prepared),
        "--motion", str(motion), "--objects", str(out / "objects.npy"),
        "--scene-metrics", str(out / "scene_metrics.json"), "--output", str(out / "training_collision.json")] + self_tail,
        root, dict(env, PYTHONPATH=str(root)))

    persistence = ("penetration_persistence", [str(python), str(root / "scripts/review_penetration_persistence.py"),
        "--clip-dir", str(out), "--output", str(out / "penetration_persistence.json")] + self_tail,
        root, dict(env, PYTHONPATH=str(root)))

    def start(job):
        name, cmd, cwd, job_env = job
        with (out / f"audit_{name}.log").open("w") as handle:
            return name, cmd, time.perf_counter(), subprocess.Popen(
                cmd, cwd=cwd, env=job_env, stdout=handle, stderr=subprocess.STDOUT)

    def scene_receipt(receipts):
        # The dependent audit verifies which inputs the scene audit read; the
        # final receipt replaces this provisional one after every audit ends.
        save(out / "receipt.json", dict(provisional=True, acceptance=dict(
            receipts=[r for r in receipts if r["name"] == "scene_metrics"])))

    def drain(running, receipts):
        """Wait for every running audit, recording each one's own completion time."""
        while running:
            for item in [item for item in running if item[3].poll() is not None]:
                name, cmd, tick, process = item
                running.remove(item)
                receipts.append(dict(name=name, args=cmd, returncode=process.returncode,
                                     seconds=time.perf_counter() - tick))
                if name == "scene_metrics" and dependent is not None:
                    scene_receipt(receipts)
                    running.append(start(dependent))
                    running.append(start(persistence))
            time.sleep(.02)

    receipts = []
    if concurrent:
        drain([start(job) for job in jobs], receipts)
    else:
        for job in jobs:
            drain([start(job)], receipts)
    return dict(status="completed_with_failures" if any(r["returncode"] for r in receipts) else "geometry_and_metrics_completed",
                receipts=receipts, concurrent=concurrent, scene_workers=scene_workers,
                physics="unverified: physical replay unavailable", reset="see penetration_persistence.json: invalid/unknown/candidate")


def run(args):
    run_dir = args.run_dir.resolve()
    source = (run_dir / "baseline_source") if args.arm == "A" else args.source_root
    if source is None:
        raise ValueError("Candidate arms require --source-root; use a frozen candidate snapshot")
    source = source.resolve()
    python = ROOT / ".venv/bin/python"
    mode = "trace" if args.trace else "normal"
    clips = args.clips.split(",") if args.clips else CLIPS
    env = dict(os.environ, **THREADS, PYTHONPATH=str(source))
    for key in list(env):
        if key.startswith("OMNIRETARGETING_"):
            del env[key]
    if args.trace:
        env["OMNIRETARGETING_FAST_ENABLE"] = "performance_trace"
    for repeat in range(args.repeat_offset + 1, args.repeat_offset + args.repeats + 1):
        for clip in clips:
            if clip not in CLIPS:
                raise ValueError(f"Unknown clip: {clip}")
            out = run_dir / args.arm / mode / f"repeat_{repeat}" / clip
            if out.exists():
                raise FileExistsError(f"Refusing to overwrite {out}")
            out.parent.mkdir(parents=True, exist_ok=True)
            cmd = [str(python), str(run_dir / "benchmark_runner.py"), "--run-dir", str(run_dir),
                   "--child", clip, "--source-root", str(source), "--output", str(out)]
            # Retain the existing global heavy-job lock; one clip at a time.
            full = ["flock", "/tmp/speed15x_pipeline.lock", "systemd-run", "--user", "--scope", "--quiet",
                    "-p", "MemoryMax=40G", "-p", "MemorySwapMax=0", "/usr/bin/time", "-v"] + cmd
            tick = time.perf_counter()
            print(f"START {args.arm} {mode} repeat={repeat} {clip}", flush=True)
            with (out.parent / f"{clip}.log").open("x") as handle:
                code = subprocess.call(full, cwd=source, env=env, stdout=handle, stderr=subprocess.STDOUT)
            solver_cost = time.perf_counter() - tick
            # The child mkdir can fail before producing a directory.
            out.mkdir(exist_ok=True)
            audit = run_audits(out, source, env, python, scene_workers=args.audit_workers,
                               concurrent=not args.serial_audits, self_audit_root=args.self_audit_root)
            receipt = dict(arm=args.arm, mode=mode, repeat=repeat, clip=clip, args=full, source_root=str(source),
                solver_returncode=code, production_process_wall_seconds=solver_cost,
                end_to_end_seconds=time.perf_counter() - tick, acceptance=audit)
            save(out / "receipt.json", receipt)
            with (run_dir / "progress.jsonl").open("a") as handle:
                handle.write(json.dumps(receipt) + "\n")
            print(f"DONE {clip} exit={code} full_seconds={receipt['end_to_end_seconds']:.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--robot-profile", default="unitree_g1_mesh.json",
                        help="Profile under robot_models/unitree_g1 frozen by --prepare")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--arm", default="A", help="A for baseline; any other name for a candidate")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--clips", default="")
    parser.add_argument("--repeat-offset", type=int, default=0,
                        help="first repeat index minus one; lets callers interleave arms clip by clip")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--audit-workers", type=int, default=None,
                        help="scene_metrics --workers (bit-identical parallel audit)")
    parser.add_argument("--self-audit-root", type=Path, default=None,
                        help="snapshot whose review_training_collision.py audits self collision and reset geometry")
    parser.add_argument("--serial-audits", action="store_true",
                        help="run independent audits serially instead of concurrently")
    parser.add_argument("--child", default="")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.source_root is not None:
        sys.path.insert(0, str(args.source_root.resolve()))
    if args.prepare:
        prepare(args)
    elif args.child:
        child(args)
    elif args.run_dir:
        run(args)
    else:
        parser.error("--prepare or --run-dir required")


if __name__ == "__main__":
    main()
