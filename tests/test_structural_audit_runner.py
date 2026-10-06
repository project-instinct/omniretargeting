"""Fake-subprocess coverage for the supplemental structural audit runner."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pytest

import scripts.audit_structural_spline as runner


def _write_final(path: Path, *, source_file: str, changed_contact=False):
    source_joints = np.full((2, 3, 3), np.nan)
    contact_target = np.zeros((2, 4, 3))
    if changed_contact:
        contact_target[1, 0, 0] = 0.25
    np.savez(
        path,
        framerate=np.asarray(30.0),
        qpos=np.zeros((2, 4)),
        source_file=np.asarray(source_file),
        source_type=np.asarray("smplx"),
        source_joints_w=source_joints,
        smplx_joints_w=np.arange(18.0).reshape(2, 3, 3),
        scene_object_names=np.asarray(["chair"]),
        scene_object_pos_w=np.ones((2, 1, 3)),
        scene_object_quat_w=np.asarray([[1.0, 0, 0, 0], [0, 1, 0, 0]]),
        terrain_scale=np.asarray(1.0),
        contact_mask=np.ones((2, 4), dtype=bool),
        contact_targets_w=contact_target,
        contact_modes=np.asarray([["none"], ["sliding"]]),
        surface_contact_mask=np.ones((2, 8), dtype=bool),
        self_contact_targets_local=np.ones((2, 5, 3)),
        surface_pose_targets=np.ones((2, 16, 3)),
        end_effector_reference_w=np.ones((2, 4, 3)),
        end_effector_world_tracking=np.ones(4, dtype=bool),
    )


def _write_prepared(path: Path, *, changed_root=False):
    roots = np.zeros((2, 3))
    if changed_root:
        roots[1, 2] = 0.1
    np.save(path, {
        "roots": roots,
        "positions": np.ones((2, 3, 3)),
        "root_orientation_targets": np.tile([1.0, 0, 0, 0], (2, 1)),
        "body_orientation_targets": [{
            "robot_body": "pelvis", "sigma": 0.1,
            "rotations": np.tile([1.0, 0, 0, 0], (2, 1)),
        }],
    })


def _clip(root: Path, arm: str, repeat: int, clip: str, *, final=True, solver_code=0):
    output = root / arm / "normal" / f"repeat_{repeat}" / clip
    output.mkdir(parents=True)
    (output / "receipt.json").write_text(json.dumps({
        "arm": arm, "mode": "normal", "repeat": repeat, "clip": clip,
        "solver_returncode": solver_code,
        "production_process_wall_seconds": 1.0,
        "end_to_end_seconds": 10.0 + len(arm) + repeat,
    }))
    if not final:
        return output
    _write_final(
        output / "final_retargeted.npz",
        source_file=f"/source/{arm}_{clip}.pkl",
        changed_contact=arm != "A",
    )
    _write_prepared(output / "prepared.npy", changed_root=arm == "B")
    fixed = np.arange(8, dtype=float).reshape(2, 4)
    np.savez(output / "fixed_reference.npz", qpos=fixed)
    (output / "objects.npy").write_bytes(b"fake objects")
    (output / "scene_metrics.json").write_text("{}")
    (output / "static_scene.obj").write_bytes(b"static scene bytes")
    profile = output / "profile.json"
    profile.write_text(json.dumps({"urdf_path": "/fake/g1.xml"}))
    (output / "invocation.json").write_text(json.dumps({
        "argv": ["retarget", "--robot-config", str(profile)]
    }))
    return output


def _paths(root: Path):
    baseline = root / "measure" / "baseline_source" / "scripts"
    candidate = root / "measure" / "candidate_v3" / "scripts"
    audit = root / "audit_v1" / "scripts"
    for directory, names in (
        (baseline, ("review_joint_smoothness.py",)),
        (candidate, ("review_joint_smoothness.py",)),
        (audit, ("review_training_collision.py", "review_spline_quality.py")),
    ):
        directory.mkdir(parents=True)
        for name in names:
            (directory / name).write_text("# fake\n")
    return baseline.parent, candidate.parent, audit.parent


def _args(root: Path, arms="A,B"):
    return argparse.Namespace(
        run_dir=root / "measure",
        source_root=Path("candidate_v3"),
        audit_root=root / "audit_v1",
        python=root / ".venv/bin/python",
        arms=arms,
        modes="normal",
        repeats=1,
        clips="",
        summary=root / "measure" / "supplemental_summary.json",
    )


def _fake_call(monkeypatch, calls):
    def call(command, *, cwd, env, stdout, stderr):
        calls.append({"command": command, "cwd": Path(cwd), "env": env,
                      "stdout": stdout, "stderr": stderr})
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        if command[1].endswith("review_training_collision.py"):
            motion = Path(command[command.index("--motion") + 1])
            report = {"inputs": {"motion": str(motion)}, "acceptance_status": "unverified"}
        elif command[1].endswith("review_joint_smoothness.py"):
            report = {
                "before": str(Path(command[command.index("--before") + 1])),
                "after": str(Path(command[command.index("--after") + 1])),
                "all_individual_budgets_pass": True,
            }
        else:
            report = {"clip": output.parent.name}
        output.write_text(json.dumps(report))
        return 0

    monkeypatch.setattr(runner.subprocess, "call", call)


def test_source_split_invariants_and_supplemental_reference(tmp_path, monkeypatch):
    root = tmp_path
    baseline, candidate, audit = _paths(root)
    _clip(root / "measure", "A", 1, "walk")
    _clip(root / "measure", "C", 1, "walk")
    trace = root / "measure" / "A" / "trace" / "repeat_1" / "walk"
    trace.mkdir(parents=True)
    (trace / "receipt.json").write_text(json.dumps({
        "arm": "A", "mode": "trace", "repeat": 1, "clip": "walk",
        "solver_returncode": 0, "end_to_end_seconds": 99.0,
    }))
    # C inherits A's immutable prepared/source geometry, but has a changed target.
    a = root / "measure" / "A" / "normal" / "repeat_1" / "walk"
    c = root / "measure" / "C" / "normal" / "repeat_1" / "walk"
    _write_prepared(c / "prepared.npy")
    (c / "static_scene.obj").write_bytes((a / "static_scene.obj").read_bytes())
    calls = []
    _fake_call(monkeypatch, calls)
    summary = runner.run_supplemental_audit(_args(root, arms="A,C"))
    assert summary["incomplete_or_failed"] == 1
    assert summary["clips"][0]["invariants_passed"] is True
    assert summary["clips"][0]["task_statuses"]["invariants"] == "passed"
    assert summary["clips"][1]["invariants_passed"] is False
    assert "contact_targets_w" in summary["clips"][1]["invariant_mismatches"]
    assert summary["clips"][1]["task_statuses"]["invariants"] == "failed"

    smooth = [call for call in calls if call["command"][1].endswith("review_joint_smoothness.py")]
    quality = [call for call in calls if call["command"][1].endswith("review_spline_quality.py")]
    training = [call for call in calls if call["command"][1].endswith("review_training_collision.py")]
    assert len(smooth) == len(quality) == len(training) == 2
    assert [call["cwd"] for call in smooth] == [baseline, candidate]
    assert all(call["cwd"] == audit for call in quality + training)
    assert all(Path(call["env"]["PYTHONPATH"]) == call["cwd"] for call in calls)
    assert all(
        all(call["env"][name] == "1" for name in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        ))
        for call in calls
    )
    candidate_quality = next(
        call for call in quality
        if call["command"][call["command"].index("--final") + 1] == str(c / "final_retargeted.npz")
    )
    command = candidate_quality["command"]
    assert command[command.index("--reference") + 1] == str(c / "supplemental_fixed_reference.npz")
    assert command[command.index("--baseline") + 1] == str(a / "final_retargeted.npz")
    assert command[command.index("--baseline-reference") + 1] == str(a / "supplemental_fixed_reference.npz")
    with np.load(c / "supplemental_fixed_reference.npz") as saved:
        assert float(saved["framerate"]) == 30.0
        assert int(saved["frame_start"]) == 0
        np.testing.assert_array_equal(saved["qpos"], np.arange(8, dtype=float).reshape(2, 4))
    with np.load(c / "final_retargeted.npz") as saved:
        assert not np.array_equal(saved["qpos"], np.arange(8, dtype=float).reshape(2, 4))
    receipt = json.loads((c / "supplemental_receipt.json").read_text())
    assert receipt["seconds"] == receipt["supplemental_seconds"] > 0.0
    format_task = next(task for task in receipt["tasks"] if task.get("name") == "format_fixed_reference")
    assert format_task["status"] == "created"
    assert format_task["qpos_sha256"] == hashlib.sha256(
        np.ascontiguousarray(np.arange(8, dtype=float).reshape(2, 4)).tobytes()
    ).hexdigest()
    invariant = json.loads((c / "supplemental_receipt.json").read_text())["tasks"][0]
    assert invariant["final"]["arrays"]["source_joints_w"]["nan_equal_allowed"] is True
    assert invariant["final"]["arrays"]["source_joints_w"]["equal"] is True
    assert invariant["final"]["source_file_paths"]["candidate"] != invariant["final"]["source_file_paths"]["A"]
    assert "source_file" not in invariant["mismatches"]


def test_all_failed_receipts_are_counted_without_heavy_subprocess(tmp_path, monkeypatch):
    root = tmp_path
    _paths(root)
    for arm in ("A", "B", "C", "D"):
        _clip(root / "measure", arm, 1, "failed", final=False, solver_code=2)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("missing-motion receipts must not launch heavy audits")

    monkeypatch.setattr(runner.subprocess, "call", forbidden)
    summary = runner.run_supplemental_audit(_args(root, arms="A,B,C,D"))
    assert summary["receipt_count"] == 4
    assert summary["initial_failure_receipts"] == 4
    assert summary["incomplete_or_failed"] == 4
    assert summary["newly_processed"] == 4
    assert set(summary["by_arm"]) == {"A", "B", "C", "D"}
    assert all(value["receipts"] == 1 for value in summary["by_arm"].values())
    assert all(value["incomplete_supplemental_receipts"] == 1 for value in summary["by_arm"].values())
    assert summary["supplemental_seconds_sum"] > 0.0
    assert summary["comparison_total_seconds_sum"] == pytest.approx(
        summary["initial_end_to_end_seconds_sum"] + summary["supplemental_seconds_sum"]
    )
    for arm in ("A", "B", "C", "D"):
        receipt = json.loads((
            root / "measure" / arm / "normal" / "repeat_1" / "failed" / "supplemental_receipt.json"
        ).read_text())
        assert receipt["status"] == "incomplete"
        assert receipt["initial_solver_returncode"] == 2
        assert receipt["motion_modified"] is False


def test_supplemental_outputs_and_summary_are_never_overwritten(tmp_path, monkeypatch):
    root = tmp_path
    _paths(root)
    output = _clip(root / "measure", "D", 1, "keep")
    _write_prepared(output / "prepared.npy")
    a = root / "measure" / "A" / "normal" / "repeat_1" / "keep"
    _clip(root / "measure", "A", 1, "keep")
    _write_prepared(output / "prepared.npy")
    (output / "static_scene.obj").write_bytes((a / "static_scene.obj").read_bytes())
    sentinel = {
        "schema_version": 1, "arm": "D", "mode": "normal", "repeat": 1, "clip": "keep",
        "initial_end_to_end_seconds": 7.0, "supplemental_seconds": 1.25,
        "comparison_total_seconds": 8.25, "status": "completed", "tasks": [],
    }
    receipt = output / "supplemental_receipt.json"
    receipt.write_text(json.dumps(sentinel))
    calls = []
    _fake_call(monkeypatch, calls)
    summary = runner.run_supplemental_audit(_args(root, arms="A,D"))
    assert summary["already_present_skipped"] == 1
    assert summary["newly_processed"] == 1
    assert json.loads(receipt.read_text()) == sentinel
    assert all(str(output) not in " ".join(call["command"]) for call in calls)

    summary_path = root / "measure" / "supplemental_summary.json"
    before = summary_path.read_text()
    with pytest.raises(FileExistsError):
        runner.run_supplemental_audit(_args(root, arms="A,D"))
    assert summary_path.read_text() == before


def test_missing_snapshot_script_is_a_real_failure_and_never_overwritten(tmp_path, monkeypatch):
    root = tmp_path
    _paths(root)
    _clip(root / "measure", "A", 1, "keep")
    _clip(root / "measure", "D", 1, "keep")
    d = root / "measure" / "D" / "normal" / "repeat_1" / "keep"
    a = root / "measure" / "A" / "normal" / "repeat_1" / "keep"
    _write_prepared(d / "prepared.npy")
    (d / "static_scene.obj").write_bytes((a / "static_scene.obj").read_bytes())
    candidate_script = root / "measure" / "candidate_v3" / "scripts" / "review_joint_smoothness.py"
    candidate_script.unlink()
    calls = []
    _fake_call(monkeypatch, calls)
    summary = runner.run_supplemental_audit(_args(root, arms="A,D"))
    assert summary["incomplete_or_failed"] == 1
    receipt = json.loads((d / "supplemental_receipt.json").read_text())
    smoothness = next(
        task for task in receipt["tasks"]
        if task.get("name") == "joint_smoothness_supplemental"
    )
    assert smoothness["status"] == "failed"
    assert smoothness["reason"] == f"missing audit dependency: {candidate_script}"
    assert smoothness["returncode"] == 127
    assert not (d / "supplemental_joint_smoothness.log").exists()
    before = (d / "supplemental_receipt.json").read_text()
    args = _args(root, arms="D")
    args.summary = root / "measure" / "second-summary.json"
    second = runner.run_supplemental_audit(args)
    assert second["already_present_skipped"] == 1
    assert (d / "supplemental_receipt.json").read_text() == before


def test_orchestrator_exception_preserves_elapsed_cost_and_normal_mode_default(tmp_path, monkeypatch):
    root = tmp_path
    _paths(root)
    _clip(root / "measure", "A", 1, "boom")
    trace = root / "measure" / "A" / "trace" / "repeat_1" / "boom"
    trace.mkdir(parents=True)
    (trace / "receipt.json").write_text(json.dumps({
        "arm": "A", "mode": "trace", "repeat": 1, "clip": "boom",
        "solver_returncode": 0, "end_to_end_seconds": 99.0,
    }))

    def explode(*_args, **_kwargs):
        time.sleep(0.02)
        raise RuntimeError("synthetic orchestrator failure")

    monkeypatch.setattr(runner, "_process_clip", explode)
    args = _args(root, arms="A")
    del args.modes
    summary = runner.run_supplemental_audit(args)
    assert summary["receipt_count"] == 1
    assert summary["modes"] == ["normal"]
    assert summary["incomplete_or_failed"] == 1
    assert summary["supplemental_seconds_sum"] >= 0.015
    output = root / "measure" / "A" / "normal" / "repeat_1" / "boom"
    receipt = json.loads((output / "supplemental_receipt.json").read_text())
    assert receipt["status"] == "orchestrator_error"
    assert "synthetic orchestrator failure" in receipt["reason"]
    assert receipt["seconds"] == receipt["supplemental_seconds"] >= 0.015
    assert receipt["comparison_total_seconds"] == pytest.approx(
        receipt["initial_end_to_end_seconds"] + receipt["seconds"]
    )
    assert summary["comparison_total_seconds_sum"] == pytest.approx(
        summary["initial_end_to_end_seconds_sum"] + summary["supplemental_seconds_sum"]
    )


def test_venv_executable_and_explicit_retry_preserve_failed_cost(tmp_path, monkeypatch):
    _paths(tmp_path)
    output = _clip(tmp_path / "measure", "A", 1, "walk")
    executable = tmp_path / "system-python"
    executable.touch()
    args = _args(tmp_path, arms="A")
    args.python.parent.mkdir(parents=True)
    args.python.symlink_to(executable)
    calls = []
    _fake_call(monkeypatch, calls)
    # A failed earlier attempt retains its logs and all elapsed cost.
    original = {"seconds": 1.75, "status": "incomplete"}
    original_path = output / "supplemental_receipt.json"
    original_path.write_text(json.dumps(original))
    previous_log = output / "supplemental_training_collision.log"
    previous_log.write_text("ModuleNotFoundError: No module named 'mujoco'\n")
    args.attempt = "retry_2"
    result = runner.run_supplemental_audit(args)
    assert result["incomplete_or_failed"] == 0
    assert result["python"] == str(args.python.absolute())
    assert all(call["command"][0] == str(args.python.absolute()) for call in calls)
    retry_path = output / "supplemental_receipt_retry_2.json"
    retry = json.loads(retry_path.read_text())
    assert retry["audit_attempt"] == "retry_2"
    assert json.loads(original_path.read_text()) == original
    assert previous_log.read_text().startswith("ModuleNotFoundError")
    assert all("retry_2" in Path(call["stdout"].name).stem for call in calls)
    assert result["supplemental_seconds_sum"] == pytest.approx(1.75 + retry["seconds"])
    assert result["comparison_total_seconds_sum"] == pytest.approx(
        result["initial_end_to_end_seconds_sum"] + 1.75 + retry["seconds"]
    )
