"""Serial supplemental audits for frozen structural-spline benchmark runs.

The optimization snapshot and audit snapshot are deliberately separate.  ``--source-root``
selects the frozen implementation used for the original joint-smoothness audit; ``--audit-root``
selects the newer quality/training-collision scripts.  No motion is modified or optimized.
"""

from __future__ import annotations

import argparse
from contextvars import ContextVar
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
_ATTEMPT = ContextVar("supplemental_audit_attempt", default="base")


def _receipt_target(output: Path) -> Path:
    attempt = _ATTEMPT.get()
    suffix = "" if attempt == "base" else f"_{attempt}"
    return output / f"supplemental_receipt{suffix}.json"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_new_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _motion_fields(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        if "qpos" not in saved:
            raise ValueError(f"{path}: qpos missing")
        if "framerate" not in saved:
            raise ValueError(f"{path}: framerate missing")
        qpos = np.asarray(saved["qpos"])
        return {
            "qpos": qpos,
            "qpos_sha256": hashlib.sha256(
                np.ascontiguousarray(qpos).tobytes()
            ).hexdigest(),
            "framerate": float(np.asarray(saved["framerate"]).item()),
            "frame_start": int(np.asarray(saved["frame_start"]).item() if "frame_start" in saved else 0),
        }


def _same_qpos(first: np.ndarray, second: np.ndarray) -> bool:
    return (
        first.dtype == second.dtype
        and first.shape == second.shape
        and np.ascontiguousarray(first).tobytes()
        == np.ascontiguousarray(second).tobytes()
    )


def _selected_source(run_dir: Path, arm: str, requested: Path) -> Path:
    if arm == "A":
        return (run_dir / "baseline_source").resolve()
    candidate = requested if requested.is_absolute() else run_dir / requested
    if not candidate.exists():
        candidate = requested
    return candidate.resolve()


def _receipts(
    run_dir: Path, arms: list[str], repeats: int, clips: set[str] | None, modes: list[str]
) -> list[tuple[Path, dict[str, Any]]]:
    result: list[tuple[Path, dict[str, Any]]] = []
    for arm in arms:
        for repeat in range(1, repeats + 1):
            for mode in modes:
                pattern = Path(arm) / mode / f"repeat_{repeat}" / "*" / "receipt.json"
                for path in sorted(run_dir.glob(str(pattern))):
                    clip = path.parent.name
                    if clips is None or clip in clips:
                        result.append((path, _read_json(path)))
    return result


def _model_path(output: Path) -> str | None:
    invocation = output / "invocation.json"
    if not invocation.is_file():
        return None
    try:
        argv = _read_json(invocation).get("argv", [])
        profile = Path(argv[argv.index("--robot-config") + 1])
        return str(_read_json(profile).get("urdf_path", ""))
    except (ValueError, OSError, json.JSONDecodeError, KeyError, IndexError):
        return None


def _environment(source: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("OMNIRETARGETING_")}
    env["PYTHONPATH"] = str(source)
    for name in (
        "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"
    return env


def _run_subprocess(
    command: list[str],
    *,
    cwd: Path,
    source: Path,
    log: Path,
    output: Path,
    validate,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    if _ATTEMPT.get() != "base":
        log = log.with_name(f"{log.stem}_{_ATTEMPT.get()}{log.suffix}")
    if not cwd.is_dir():
        return {
            "status": "failed", "reason": f"missing working snapshot: {cwd}",
            "command": command, "returncode": 127, "seconds": 0.0,
            "log": str(log), "output": str(output),
        }, 0.0
    if not Path(command[1]).is_file():
        return {
            "status": "failed", "reason": f"missing audit dependency: {command[1]}",
            "command": command, "returncode": 127, "seconds": 0.0,
            "log": str(log), "output": str(output),
        }, 0.0
    if log.exists():
        return {
            "status": "blocked_existing_log", "seconds": 0.0,
            "command": command,
            "log": str(log),
            "output": str(output),
        }, 0.0
    if output.exists():
        try:
            validate(_read_json(output))
            return {
                "status": "already_present",
                "seconds": 0.0,
                "command": None,
                "log": str(log),
                "output": str(output),
            }, 0.0
        except Exception as error:
            return {
                "status": "invalid_existing_output",
                "reason": f"{type(error).__name__}: {error}",
                "seconds": 0.0,
                "command": None,
                "log": str(log),
                "output": str(output),
            }, 0.0
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x", encoding="utf-8") as stream:
        code = subprocess.call(
            command,
            cwd=str(cwd),
            env=_environment(source),
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    seconds = time.perf_counter() - started
    status = "failed" if code else "completed"
    if not code and not output.is_file():
        status = "missing_output"
    elif not code:
        try:
            validate(_read_json(output))
        except Exception as error:
            status = "invalid_output"
            return {
                "status": status,
                "reason": f"{type(error).__name__}: {error}",
                "command": command,
                "returncode": int(code),
                "seconds": seconds,
                "log": str(log),
                "output": str(output),
            }, seconds
    return {
        "status": status,
        "command": command,
        "returncode": int(code),
        "seconds": seconds,
        "log": str(log),
        "output": str(output),
    }, seconds


def _format_reference(output: Path) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    original = output / "fixed_reference.npz"
    final = output / "final_retargeted.npz"
    target = output / "supplemental_fixed_reference.npz"
    task: dict[str, Any] = {
        "status": "failed",
        "original": str(original),
        "final": str(final),
        "output": str(target),
    }
    try:
        fixed = _motion_qpos(original)
        final_fields = _motion_fields(final)
        if fixed.shape != final_fields["qpos"].shape:
            raise ValueError(
                "fixed_reference and final qpos shapes differ; frame/model dimensions do not match"
            )
        if final_fields["framerate"] <= 0:
            raise ValueError("final framerate is not positive")
        task.update(
            framerate=final_fields["framerate"],
            frame_start=final_fields["frame_start"],
            qpos_sha256=hashlib.sha256(np.ascontiguousarray(fixed).tobytes()).hexdigest(),
        )
        if target.exists():
            existing = _motion_fields(target)
            if not _same_qpos(existing["qpos"], fixed):
                raise ValueError("existing supplemental reference has different qpos")
            if (
                existing["framerate"] != final_fields["framerate"]
                or existing["frame_start"] != final_fields["frame_start"]
            ):
                raise ValueError("existing supplemental reference metadata differs from final")
            task["status"] = "already_present"
        else:
            np.savez(
                target,
                qpos=fixed,
                framerate=np.asarray(final_fields["framerate"]),
                frame_start=np.asarray(final_fields["frame_start"], dtype=int),
            )
            task["status"] = "created"
        task["reason"] = (
            "original fixed_reference qpos retained byte-for-byte; framerate/frame_start copied from final"
        )
    except Exception as error:
        task["reason"] = f"{type(error).__name__}: {error}"
    task["seconds"] = time.perf_counter() - started
    return task, task["seconds"]


def _motion_qpos(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as saved:
        return np.asarray(saved["qpos"])


def _dependency_failure(script: Path, source: Path, output: Path) -> tuple[dict[str, Any], float]:
    reason = (
        f"missing working snapshot: {source}"
        if not source.is_dir()
        else f"missing audit dependency: {script}"
    )
    return {
        "status": "failed", "reason": reason, "returncode": 127,
        "command": [str(script)], "log": None, "output": str(output), "seconds": 0.0,
    }, 0.0


def _training_task(
    output: Path, audit_root: Path, python: Path
) -> tuple[dict[str, Any], float]:
    prepared = output / "prepared.npy"
    final = output / "final_retargeted.npz"
    objects = output / "objects.npy"
    scene = output / "scene_metrics.json"
    target = output / "training_collision.json"
    script = audit_root / "scripts" / "review_training_collision.py"
    if not audit_root.is_dir() or not script.is_file():
        return _dependency_failure(script, audit_root, target)
    missing = [str(path) for path in (prepared, final, objects, scene) if not path.is_file()]
    if missing:
        return {
            "status": "not_run_missing_inputs",
            "missing": missing,
            "output": str(target),
            "seconds": 0.0,
        }, 0.0

    def validate(report: dict[str, Any]) -> None:
        recorded = report.get("inputs", {}).get("motion")
        if recorded and Path(recorded).resolve() != final.resolve():
            raise ValueError(f"training report belongs to {recorded}, not {final}")

    command = [
        str(python),
        str(script),
        "--prepared", str(prepared),
        "--motion", str(final),
        "--objects", str(objects),
        "--scene-metrics", str(scene),
        "--output", str(target),
    ]
    return _run_subprocess(
        command,
        cwd=audit_root,
        source=audit_root,
        log=output / "supplemental_training_collision.log",
        output=target,
        validate=validate,
    )


def _smoothness_task(
    output: Path, source_root: Path, python: Path
) -> tuple[dict[str, Any], float]:
    model = _model_path(output)
    before = output / "supplemental_fixed_reference.npz"
    after = output / "final_retargeted.npz"
    target = output / "joint_smoothness_supplemental.json"
    script = source_root / "scripts" / "review_joint_smoothness.py"
    if not source_root.is_dir() or not script.is_file():
        return _dependency_failure(script, source_root, target)
    missing = [
        str(path)
        for path in (before, after)
        if not path.is_file()
    ]
    if model is None or missing:
        return {
            "status": "not_run_missing_inputs",
            "missing": missing + ([] if model else ["invocation/model profile"]),
            "source_root": str(source_root),
            "output": str(target),
            "seconds": 0.0,
        }, 0.0

    def validate(report: dict[str, Any]) -> None:
        if Path(report.get("before", "")).resolve() != before.resolve():
            raise ValueError("smoothness report belongs to another before trajectory")
        if Path(report.get("after", "")).resolve() != after.resolve():
            raise ValueError("smoothness report belongs to another after trajectory")

    command = [
        str(python),
        str(script),
        "--model", model,
        "--before", str(before),
        "--after", str(after),
        "--output", str(target),
    ]
    return _run_subprocess(
        command,
        cwd=source_root,
        source=source_root,
        log=output / "supplemental_joint_smoothness.log",
        output=target,
        validate=validate,
    )


def _audit_map(
    output: Path,
    baseline: Path,
    target: Path,
) -> dict[str, Any]:
    def roles(directory: Path) -> dict[str, str]:
        result = {
            "scene": str(directory / "scene_metrics.json"),
            "contact": str(directory / "contact_metrics.json"),
            "smoothness": str(directory / "joint_smoothness_supplemental.json"),
            "self_collision": str(directory / "training_collision.json"),
        }
        return {key: value for key, value in result.items() if Path(value).is_file()}

    value = {"final": roles(output), "baseline": roles(baseline)}
    _write_new_json(target, value)
    return value


def _quality_task(
    run_dir: Path,
    output: Path,
    baseline: Path,
    arm: str,
    repeat: int,
    mode: str,
    clip: str,
    audit_root: Path,
    python: Path,
) -> tuple[dict[str, Any], float]:
    reference = output / "supplemental_fixed_reference.npz"
    final = output / "final_retargeted.npz"
    baseline_reference = baseline / "supplemental_fixed_reference.npz"
    baseline_final = baseline / "final_retargeted.npz"
    target = output / "spline_quality.json"
    script = audit_root / "scripts" / "review_spline_quality.py"
    if not audit_root.is_dir() or not script.is_file():
        return _dependency_failure(script, audit_root, target)
    required = [reference, final]
    if arm != "A":
        required += [baseline_reference, baseline_final]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return {
            "status": "not_run_missing_inputs",
            "missing": missing,
            "baseline": str(baseline),
            "output": str(target),
            "seconds": 0.0,
        }, 0.0

    suffix = '' if _ATTEMPT.get() == 'base' else f'_{_ATTEMPT.get()}'
    audit_map_path = output / f"spline_quality_audit_map{suffix}.json"
    try:
        audit_map = _audit_map(output, baseline, audit_map_path)
    except FileExistsError:
        audit_map = _read_json(audit_map_path)

    def validate(report: dict[str, Any]) -> None:
        if report.get("clip") != clip:
            raise ValueError(f"quality report belongs to {report.get('clip')}, not {clip}")

    command = [
        str(python),
        str(audit_root / "scripts" / "review_spline_quality.py"),
        "--prepared", str(output / "prepared.npy"),
        "--reference", str(reference),
        "--final", str(final),
        "--baseline", str(baseline_final),
        "--baseline-reference", str(baseline_reference),
        "--clip", clip,
        "--audits", str(audit_map_path),
        "--output", str(target),
    ]
    task, seconds = _run_subprocess(
        command,
        cwd=audit_root,
        source=audit_root,
        log=output / "supplemental_spline_quality.log",
        output=target,
        validate=validate,
    )
    task.update(audit_map=audit_map, baseline_repeat=repeat, baseline_mode=mode)
    return task, seconds


def _array_descriptor(value: np.ndarray) -> dict[str, Any]:
    value = np.asarray(value)
    contiguous = np.ascontiguousarray(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }


def _invariant_array_row(
    key: str,
    first: np.ndarray | None,
    second: np.ndarray | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "candidate_present": first is not None,
        "A_present": second is not None,
    }
    if first is not None:
        row["candidate"] = _array_descriptor(first)
    if second is not None:
        row["A"] = _array_descriptor(second)
    if first is None or second is None:
        row["equal"] = False
        return row
    allow_nan = key.startswith("source_") and first.dtype.kind in "fc"
    row["nan_equal_allowed"] = bool(allow_nan)
    row["equal"] = bool(np.array_equal(first, second, equal_nan=allow_nan))
    return row


def _is_production_invariant(key: str) -> bool:
    return (
        key == "framerate"
        or key in {
            "smplx_joints_w", "end_effector_reference_w", "end_effector_world_tracking",
        }
        or key.startswith((
            "source_", "scene_object_", "terrain_", "contact_", "surface_contact_",
            "self_contact_", "surface_pose_",
        ))
    )


def _scalar_text(value: np.ndarray) -> str:
    return str(value.item() if value.ndim == 0 else value.tolist())


def _final_invariants(candidate: Path, baseline: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "candidate_final": str(candidate),
        "A_final": str(baseline),
        "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "A_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
        "excluded_solution_arrays": ["qpos", "base_*", "joint_*"],
    }
    with np.load(candidate, allow_pickle=False) as first, np.load(baseline, allow_pickle=False) as second:
        selected = sorted({key for key in first.files + second.files if _is_production_invariant(key)})
        arrays: dict[str, dict[str, Any]] = {}
        for key in selected:
            arrays[key] = _invariant_array_row(
                key,
                first[key] if key in first.files else None,
                second[key] if key in second.files else None,
            )
        if "source_file" in first.files and "source_file" in second.files:
            candidate_path = _scalar_text(first["source_file"])
            baseline_path = _scalar_text(second["source_file"])
            report["source_file_paths"] = {"candidate": candidate_path, "A": baseline_path}
            report["source_file_path_difference_is_not_alone_a_defect"] = candidate_path != baseline_path
        frame_shapes = (first["qpos"].shape, second["qpos"].shape)
        report["frame_count"] = {"candidate": frame_shapes[0][0], "A": frame_shapes[1][0]}
        report["frame_count_equal"] = frame_shapes[0][0] == frame_shapes[1][0]
    report["arrays"] = arrays
    mismatches = [
        key for key, row in arrays.items()
        if key != "source_file" and not row["equal"]
    ]
    if not report["frame_count_equal"]:
        mismatches.append("frame_count")
    report["mismatches"] = mismatches
    report["passed"] = not mismatches
    return report


def _canonical_prepared(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _array_descriptor(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _canonical_prepared(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical_prepared(item) for item in value]
    return value


def _body_orientation_rows(first: Any, second: Any) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for index, (candidate_item, baseline_item) in enumerate(zip(first, second)):
        for field, candidate_value in candidate_item.items():
            if isinstance(candidate_value, np.ndarray):
                baseline_value = baseline_item.get(field)
                rows[f"body_orientation_targets[{index}].{field}"] = _invariant_array_row(
                    f"body_orientation_targets[{index}].{field}",
                    candidate_value,
                    baseline_value if isinstance(baseline_value, np.ndarray) else None,
                )
    return rows


def _prepared_invariants(candidate: Path, baseline: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "candidate_prepared": str(candidate),
        "A_prepared": str(baseline),
        "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "A_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
    }
    candidate_data = np.load(candidate, allow_pickle=True).item()
    baseline_data = np.load(baseline, allow_pickle=True).item()
    rows: dict[str, dict[str, Any]] = {}
    for key in ("roots", "positions", "root_orientation_targets"):
        rows[key] = _invariant_array_row(key, candidate_data.get(key), baseline_data.get(key))
    body_first = candidate_data.get("body_orientation_targets", [])
    body_second = baseline_data.get("body_orientation_targets", [])
    rows.update(_body_orientation_rows(body_first, body_second))
    report["body_orientation_targets_canonical"] = {
        "candidate": _canonical_prepared(body_first),
        "A": _canonical_prepared(body_second),
    }
    report["arrays"] = rows
    mismatches = [key for key, row in rows.items() if not row["equal"]]
    if _canonical_prepared(body_first) != _canonical_prepared(body_second):
        mismatches.append("body_orientation_targets_metadata")
    report["mismatches"] = sorted(set(mismatches))
    report["passed"] = not report["mismatches"]
    return report


def _invariant_task(candidate: Path, baseline: Path) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    task: dict[str, Any] = {
        "name": "invariants", "status": "unverified", "mismatches": [],
        "candidate": str(candidate), "A_baseline": str(baseline),
    }
    try:
        final = _final_invariants(candidate / "final_retargeted.npz", baseline / "final_retargeted.npz")
        task["final"] = final
        mismatches = list(final["mismatches"])
        candidate_scene = candidate / "static_scene.obj"
        baseline_scene = baseline / "static_scene.obj"
        if candidate_scene.is_file() and baseline_scene.is_file():
            scene_hashes = (
                hashlib.sha256(candidate_scene.read_bytes()).hexdigest(),
                hashlib.sha256(baseline_scene.read_bytes()).hexdigest(),
            )
            task["static_scene"] = {
                "candidate": str(candidate_scene), "A": str(baseline_scene),
                "candidate_sha256": scene_hashes[0], "A_sha256": scene_hashes[1],
                "equal": scene_hashes[0] == scene_hashes[1],
            }
            if scene_hashes[0] != scene_hashes[1]:
                mismatches.append("static_scene.obj")
        else:
            task["static_scene"] = {"status": "missing", "candidate": str(candidate_scene), "A": str(baseline_scene)}
            task["status"] = "unverified"
        candidate_prepared = candidate / "prepared.npy"
        baseline_prepared = baseline / "prepared.npy"
        if candidate_prepared.is_file() and baseline_prepared.is_file():
            prepared = _prepared_invariants(candidate_prepared, baseline_prepared)
            task["prepared"] = prepared
            mismatches.extend(f"prepared:{key}" for key in prepared["mismatches"])
        else:
            task["prepared"] = {"status": "missing"}
            task["status"] = "unverified"
        task["mismatches"] = sorted(set(mismatches))
        task["status"] = (
            "unverified"
            if task["static_scene"].get("status") == "missing" or task["prepared"].get("status") == "missing"
            else "failed" if task["mismatches"] else "passed"
        )
        task["reason"] = "source_file path difference is recorded separately and is not alone a ground-truth defect"
    except Exception as error:
        task["status"] = "unverified"
        task["reason"] = f"{type(error).__name__}: {error}"
    task["seconds"] = time.perf_counter() - started
    return task, task["seconds"]


def _process_clip(
    run_dir: Path,
    receipt_path: Path,
    initial: dict[str, Any],
    args: argparse.Namespace,
    audit_root: Path,
    python: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    output = receipt_path.parent
    arm = str(initial.get("arm", output.parts[-4]))
    mode = str(initial.get("mode", output.parts[-3]))
    repeat = int(initial.get("repeat", output.parts[-2].split("_")[-1]))
    clip = str(initial.get("clip", output.name))
    source_root = _selected_source(run_dir, arm, args.source_root)
    receipt_target = _receipt_target(output)
    if receipt_target.exists():
        existing = _read_json(receipt_target)
        existing["_orchestrator_rerun"] = "skipped_not_overwritten"
        return existing, existing

    baseline = run_dir / "A" / mode / f"repeat_{repeat}" / clip
    tasks: list[dict[str, Any]] = []
    invariant_task, _invariant_seconds = _invariant_task(output, baseline)
    tasks.append(invariant_task)
    format_task, format_seconds = _format_reference(output)
    format_task["name"] = "format_fixed_reference"
    tasks.append(format_task)
    training_task, _training_seconds = _training_task(output, audit_root, python)
    training_task["name"] = "training_collision_increment"
    tasks.append(training_task)
    smoothness_task, _smoothness_seconds = _smoothness_task(output, source_root, python)
    smoothness_task["name"] = "joint_smoothness_supplemental"
    tasks.append(smoothness_task)
    quality_task, _quality_seconds = _quality_task(
        run_dir, output, baseline, arm, repeat, mode, clip, audit_root, python
    )
    quality_task["name"] = "spline_quality"
    tasks.append(quality_task)
    seconds = time.perf_counter() - started
    incomplete = any(
        task.get("status") not in {"completed", "already_present", "created", "passed"}
        for task in tasks
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "audit_attempt": _ATTEMPT.get(),
        "arm": arm,
        "mode": mode,
        "repeat": repeat,
        "clip": clip,
        "initial_receipt": str(receipt_path),
        "initial_solver_returncode": initial.get("solver_returncode"),
        "initial_end_to_end_seconds": initial.get("end_to_end_seconds"),
        "optimization_source_root": str(source_root),
        "audit_source_root": str(audit_root),
        "tasks": tasks,
        "seconds": seconds,
        "supplemental_seconds": seconds,
        "comparison_total_seconds": float(initial.get("end_to_end_seconds", 0.0) + seconds),
        "status": "incomplete" if incomplete else "completed",
        "motion_modified": False,
    }
    _write_new_json(receipt_target, receipt)
    return receipt, receipt


def _summary_entry(receipt: dict[str, Any]) -> dict[str, Any]:
    tasks = receipt.get("tasks", [])
    return {
        "arm": receipt.get("arm"),
        "mode": receipt.get("mode"),
        "repeat": receipt.get("repeat"),
        "clip": receipt.get("clip"),
        "initial_solver_returncode": receipt.get("initial_solver_returncode"),
        "initial_end_to_end_seconds": receipt.get("initial_end_to_end_seconds"),
        "supplemental_seconds": receipt.get(
            "seconds", receipt.get("supplemental_seconds", receipt.get("total_supplemental_seconds", 0.0))
        ),
        "comparison_total_seconds": receipt.get("comparison_total_seconds"),
        "status": receipt.get("status"),
        "invariants_passed": next((
            task.get("status") == "passed"
            for task in tasks
            if task.get("name") == "invariants"
        ), None),
        "invariant_mismatches": next((
            task.get("mismatches", [])
            for task in tasks
            if task.get("name") == "invariants"
        ), []),
        "task_statuses": {
            task.get("name", task.get("status")): task.get("status")
            for task in tasks
        },
    }


def _failure_receipt(
    receipt_path: Path, initial: dict[str, Any], error: Exception, seconds: float
) -> dict[str, Any]:
    output = receipt_path.parent
    initial_seconds = float(initial.get("end_to_end_seconds", 0.0) or 0.0)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "arm": initial.get("arm", output.parts[-4]),
        "audit_attempt": _ATTEMPT.get(),
        "mode": initial.get("mode", output.parts[-3]),
        "repeat": initial.get("repeat", int(output.parts[-2].split("_")[-1])),
        "clip": initial.get("clip", output.name),
        "initial_receipt": str(receipt_path),
        "initial_solver_returncode": initial.get("solver_returncode"),
        "initial_end_to_end_seconds": initial.get("end_to_end_seconds"),
        "tasks": [],
        "seconds": seconds,
        "supplemental_seconds": seconds,
        "comparison_total_seconds": initial_seconds + seconds,
        "status": "orchestrator_error",
        "reason": f"{type(error).__name__}: {error}",
        "motion_modified": False,
    }
    target = _receipt_target(output)
    if target.exists():
        return _read_json(target)
    _write_new_json(target, receipt)
    return receipt


def run_supplemental_audit(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    audit_root = args.audit_root.resolve()
    # Resolving the venv executable symlink selects the system installation.
    python = args.python.absolute()
    attempt = getattr(args, "attempt", "base")
    if not re.fullmatch(r"[A-Za-z0-9_]+", attempt):
        raise ValueError("attempt must contain only letters, digits and underscores")
    arms = [item.strip() for item in args.arms.split(",") if item.strip()]
    clips = None if not args.clips else {
        item.strip() for item in args.clips.split(",") if item.strip()
    }
    modes = [
        item.strip()
        for item in str(getattr(args, "modes", "normal")).split(",")
        if item.strip()
    ]
    invalid_modes = sorted(set(modes) - {"normal", "trace"})
    if invalid_modes:
        raise ValueError(f"unsupported modes {invalid_modes}; expected normal and/or trace")
    receipts = _receipts(run_dir, arms, args.repeats, clips, modes)
    entries: list[dict[str, Any]] = []
    processed = 0
    skipped = 0
    for receipt_path, initial in receipts:
        clip_started = time.perf_counter()
        token = _ATTEMPT.set(attempt)
        try:
            receipt, _summary = _process_clip(
                run_dir, receipt_path, initial, args, audit_root, python
            )
            if receipt.get("_orchestrator_rerun") == "skipped_not_overwritten":
                skipped += 1
            else:
                processed += 1
            entries.append(_summary_entry(receipt))
        except Exception as error:
            receipt = _failure_receipt(
                receipt_path, initial, error, time.perf_counter() - clip_started
            )
            entries.append({
                **_summary_entry(receipt), "receipt": str(receipt_path),
            })
        finally:
            _ATTEMPT.reset(token)
        # Preserve the cost of failed attempts, including explicit versioned retries.
        attempts = [_read_json(path) for path in sorted(
            receipt_path.parent.glob("supplemental_receipt*.json")
        )]
        entries[-1]["audit_attempts"] = attempts
        entries[-1]["supplemental_seconds"] = sum(
            float(value.get("seconds", value.get("supplemental_seconds", 0.0)))
            for value in attempts
        )
        entries[-1]["comparison_total_seconds"] = (
            float(initial.get("end_to_end_seconds") or 0.0)
            + entries[-1]["supplemental_seconds"]
        )
    totals: dict[str, dict[str, Any]] = {}
    for arm in arms:
        arm_entries = [entry for entry in entries if entry.get("arm") == arm]
        initial_seconds = sum(
            float(entry.get("initial_end_to_end_seconds") or 0.0)
            for entry in arm_entries
        )
        supplemental_seconds = sum(
            float(entry.get("supplemental_seconds") or 0.0)
            for entry in arm_entries
        )
        totals[arm] = {
            "receipts": len(arm_entries),
            "initial_failure_receipts": sum(
                entry.get("initial_solver_returncode", 0) not in (0, None)
                for entry in arm_entries
            ),
            "incomplete_supplemental_receipts": sum(
                entry.get("status") not in {"completed", "already_present_not_overwritten"}
                for entry in arm_entries
            ),
            "initial_end_to_end_seconds": initial_seconds,
            "supplemental_seconds": supplemental_seconds,
            "comparison_total_seconds": initial_seconds + supplemental_seconds,
        }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir),
        "optimization_source_root_requested": str(args.source_root),
        "audit_source_root": str(audit_root),
        "python": str(python),
        "audit_attempt": attempt,
        "arms": arms,
        "modes": modes,
        "repeats": args.repeats,
        "clips": sorted(clips) if clips is not None else None,
        "receipt_count": len(receipts),
        "newly_processed": processed,
        "already_present_skipped": skipped,
        "initial_failure_receipts": sum(
            entry.get("initial_solver_returncode", 0) not in (0, None) for entry in entries
        ),
        "incomplete_or_failed": sum(
            entry.get("status") not in {"completed", "already_present_not_overwritten"}
            for entry in entries
        ),
        "initial_end_to_end_seconds_sum": sum(
            float(entry.get("initial_end_to_end_seconds") or 0.0) for entry in entries
        ),
        "supplemental_seconds_sum": sum(
            float(entry.get("supplemental_seconds") or 0.0) for entry in entries
        ),
        "comparison_total_seconds_sum": sum(
            float(entry.get("comparison_total_seconds") or 0.0) for entry in entries
        ),
        "by_arm": totals,
        "clips": entries,
        "physics": "not_run_by_design",
        "training_integration": "not_run_by_design; geometry/self-collision only with missing provenance unverified",
        "motion_modified": False,
    }
    _write_new_json(args.summary, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("measure"))
    parser.add_argument("--source-root", type=Path, default=Path("candidate_v3"))
    parser.add_argument("--audit-root", type=Path, default=ROOT)
    parser.add_argument("--python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument("--arms", default="A,B,C,D")
    parser.add_argument("--modes", default="normal", help="comma-separated subset: normal,trace")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--clips", default="")
    parser.add_argument("--attempt", default="base", help="explicit retry label; preserves earlier receipts/logs")
    parser.add_argument(
        "--summary",
        type=Path,
        help="strict-new aggregate JSON; default is RUN_DIR/supplemental_summary.json",
    )
    args = parser.parse_args()
    args.summary = args.summary or args.run_dir / "supplemental_summary.json"
    summary = run_supplemental_audit(args)
    print(
        "SAVED",
        summary["receipt_count"],
        summary["incomplete_or_failed"],
        summary["comparison_total_seconds_sum"],
    )


if __name__ == "__main__":
    main()
