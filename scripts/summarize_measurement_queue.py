#!/usr/bin/env python3
"""Read-only report for the spline-window measurement queue.

The reporter consumes receipts and completed audit JSON/NUMPY artifacts.  It
never invokes the retargeting pipeline.  For B/C/D it can invoke the existing
read-only ``review_spline_quality.py`` comparison and stores that comparison
under the report directory rather than modifying measurement directories.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_structural_spline import CLIPS
from scripts.measurement_nonregression import build as build_nonregression
from scripts.measurement_nonregression import markdown as nonregression_markdown


SPLINE_EXPERIMENT_CLIPS = ["sit180", "chair150", "omomo_78", "fridge957"]
FULL_REPEAT_ARMS = ("A", "E", "E2")
SPLINE_EXPERIMENT_ARMS = ("B", "C", "D")
DEFAULT_ARMS = ("A", "B", "C", "D", "E", "E2")
EXPECTED_AUDITS = ("scene_metrics", "training_collision", "source_pose", "contact_metrics", "joint_smoothness")
TIMING_STAGES = {
    "acquisition": "contact_acquisition",
    "reweight": "reweight",
    "free_bodies": "free_bodies",
    "original_mesh_repair": "original_mesh_repair",
}
QUALITY_FILES = (
    "receipt.json", "timing.json", "fast_report.json", "scene_metrics.json",
    "training_collision.json", "source_pose.json", "contact_metrics.json",
    "joint_smoothness.json", "final_retargeted.npz",
)


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "missing"
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return None, f"invalid: {error}"
    return (value, None) if isinstance(value, dict) else (None, "invalid: root is not an object")


def _read_gzip_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError, UnicodeError) as error:
        return None, f"invalid: {error}"
    return (value, None) if isinstance(value, dict) else (None, "invalid: root is not an object")


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _length(value: Any) -> int | None:
    return len(value) if isinstance(value, (list, tuple)) else None


def _millimeters(block: dict[str, Any], key: str = "max_m") -> float | None:
    value = _number(block.get(key)) if isinstance(block, dict) else None
    return value * 1000 if value is not None else None


def _stats(values: list[float]) -> dict[str, float] | None:
    numbers = [item for item in values if item is not None and math.isfinite(item)]
    if not numbers:
        return None
    return {"mean": sum(numbers) / len(numbers), "min": min(numbers), "max": max(numbers)}


def _repeat_number(name: str) -> int | None:
    try:
        return int(name.removeprefix("repeat_"))
    except ValueError:
        return None


def _discover_grid(run_dir: Path, arms: list[str], args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    expected_clips = {
        **{arm: list(CLIPS) for arm in FULL_REPEAT_ARMS},
        **{arm: list(SPLINE_EXPERIMENT_CLIPS) for arm in SPLINE_EXPERIMENT_ARMS},
    }
    expected_repeats = {
        "A": args.a_repeats,
        "E": args.e_repeats,
        "E2": args.e_repeats,
        **{arm: 1 for arm in SPLINE_EXPERIMENT_ARMS},
    }
    grid: dict[str, dict[str, Any]] = {}
    for arm in arms:
        normal = run_dir / arm / "normal"
        custom = arm not in expected_clips and arm in args.candidate_arms
        discovered_clips: set[str] = set(expected_clips.get(arm, [] if custom else CLIPS))
        discovered_repeats: set[int] = set()
        for repeat_dir in normal.glob("repeat_*") if normal.is_dir() else []:
            repeat = _repeat_number(repeat_dir.name)
            if repeat is not None:
                discovered_repeats.add(repeat)
            discovered_clips.update(path.name for path in repeat_dir.iterdir() if path.is_dir())
        default_repeats = 0 if custom else args.a_repeats
        wanted = set(range(1, expected_repeats.get(arm, default_repeats) + 1)) | discovered_repeats
        grid[arm] = {"clips": sorted(discovered_clips), "repeats": sorted(wanted)}
    return grid


def _timing_row(identity: dict[str, Any], directory: Path) -> dict[str, Any]:
    receipt, receipt_error = _read_json(directory / "receipt.json")
    timing, timing_error = _read_json(directory / "timing.json")
    report, report_error = _read_json(directory / "fast_report.json")
    acceptance = receipt.get("acceptance", {}) if receipt else {}
    receipts = acceptance.get("receipts", []) if isinstance(acceptance, dict) else []
    audit_seconds = {name: None for name in EXPECTED_AUDITS}
    audit_seconds.update({
        str(item.get("name", "unknown")): _number(item.get("seconds"))
        for item in receipts if isinstance(item, dict)
    })
    audit_returncodes = {}
    audit_returncodes.update({
        str(item.get("name", "unknown")): _integer(item.get("returncode"))
        for item in receipts if isinstance(item, dict)
    })
    production = _number(receipt.get("production_process_wall_seconds")) if receipt else None
    end_to_end = _number(receipt.get("end_to_end_seconds")) if receipt else None
    stage_seconds = timing.get("stage_seconds", {}) if isinstance(timing, dict) else {}
    stage_seconds = stage_seconds if isinstance(stage_seconds, dict) else {}
    explicit = set(TIMING_STAGES.values())
    other = sum(value for key, value in stage_seconds.items() if key not in explicit and _number(value) is not None)
    contact_scene = report.get("stages", {}).get("contact_scene", {}) if isinstance(report, dict) else {}
    mesh_repair = contact_scene.get("original_mesh_repair", {}) if isinstance(contact_scene, dict) else {}
    failed_audits = [name for name, code in audit_returncodes.items() if code not in (0, None)]
    row: dict[str, Any] = {
        **identity,
        "status": "missing" if receipt is None else "present",
        "path": str(directory),
        "receipt_error": receipt_error,
        "timing_error": timing_error,
        "fast_report_error": report_error,
        "solver_returncode": _integer(receipt.get("solver_returncode")) if receipt else None,
        "production_process_wall_seconds": production,
        "end_to_end_seconds": end_to_end,
        "backend_seconds": _number(timing.get("fast_backend_seconds")) if timing else None,
        "solver_wall_seconds": _number(timing.get("wall_seconds")) if timing else None,
        "source_load_seconds": _number(timing.get("source_load_seconds")) if timing else None,
        "contact_scene_seconds": _number(contact_scene.get("seconds")) if isinstance(contact_scene, dict) else None,
        "stage_seconds": {
            **{key: _number(stage_seconds.get(source)) for key, source in TIMING_STAGES.items()},
            "other": other if other else (other if isinstance(other, float) else None),
            "original_mesh_repair_report_seconds": _number(mesh_repair.get("seconds")) if isinstance(mesh_repair, dict) else None,
        },
        "audit_seconds": audit_seconds,
        "audit_returncodes": audit_returncodes,
        "audit_wall_clock_seconds": (
            end_to_end - production
            if end_to_end is not None and production is not None else None
        ),
        "audit_seconds_sum": (
            sum(value for value in audit_seconds.values() if value is not None)
            if any(value is not None for value in audit_seconds.values()) else None
        ),
        "concurrent": acceptance.get("concurrent") if isinstance(acceptance, dict) else None,
        "scene_workers": _integer(acceptance.get("scene_workers")) if isinstance(acceptance, dict) else None,
        "failed_audits": failed_audits,
        "missing_files": [name for name in QUALITY_FILES if not (directory / name).is_file()],
        "budgets": contact_scene.get("budgets") if isinstance(contact_scene, dict) else None,
        "skipped_tasks": contact_scene.get("skipped_tasks") if isinstance(contact_scene, dict) else None,
    }
    if row["stage_seconds"]["other"] == 0:
        row["stage_seconds"]["other"] = 0.0
    return row


def _worst_stat(parts: dict[str, Any], field: str) -> dict[str, Any] | None:
    rows = []
    for part, values in parts.items() if isinstance(parts, dict) else []:
        if not isinstance(values, dict):
            continue
        value = _number(values.get(field))
        if value is not None:
            rows.append({"part": str(part), "value": value,
                         "worst_frame": _integer(values.get("worst_frame"))})
    return max(rows, key=lambda item: item["value"]) if rows else None


def _quality_row(identity: dict[str, Any], directory: Path) -> dict[str, Any]:
    scene, scene_error = _read_json(directory / "scene_metrics.json")
    training, training_error = _read_json(directory / "training_collision.json")
    pose, pose_error = _read_json(directory / "source_pose.json")
    contact, contact_error = _read_json(directory / "contact_metrics.json")
    smooth, smooth_error = _read_json(directory / "joint_smoothness.json")
    static = scene.get("static", {}) if isinstance(scene, dict) else {}
    moving = scene.get("moving", {}) if isinstance(scene, dict) else {}
    self_collision = training.get("self_collision", {}) if isinstance(training, dict) else {}
    reset = training.get("reset", {}) if isinstance(training, dict) else {}
    pose_parts = pose.get("parts", {}) if isinstance(pose, dict) else {}
    orientation = pose.get("calibrated_body_orientation", {}) if isinstance(pose, dict) else {}
    contact_parts = contact.get("surface_sticking_target_relative_slip_m_s", {}) if isinstance(contact, dict) else {}
    joints = smooth.get("joints", {}) if isinstance(smooth, dict) else {}
    joint_chatter, velocity_joint, effort_joint = [], [], []
    if isinstance(joints, dict):
        for name, values in joints.items():
            if not isinstance(values, dict):
                continue
            chatter = values.get("source_relative_chatter", {})
            limits = values.get("mechanical_limits", {})
            if isinstance(chatter, dict):
                joint_chatter.append({"part": str(name), "value": _integer(chatter.get("violations")) or 0,
                                      "max_excess": _number(chatter.get("max_excess")),
                                      "worst_frame": _integer(chatter.get("worst_frame"))})
            if isinstance(limits, dict):
                for category, target in (("velocity", velocity_joint), ("effort", effort_joint)):
                    block = limits.get(category, {})
                    if isinstance(block, dict):
                        target.append({"part": str(name), "value": _integer(block.get("violations")) or 0,
                                       "peak": _number(block.get("peak_per_second", block.get("peak"))),
                                       "worst_frame": _integer(block.get("worst_frame"))})
    root_chatter = smooth.get("root_source_relative_chatter", {}) if isinstance(smooth, dict) else {}
    root_chatter = root_chatter if isinstance(root_chatter, dict) else {}
    root_xy = contact.get("root_xy_offset_m", {}) if isinstance(contact, dict) else {}
    root_z = contact.get("root_z_offset_m", {}) if isinstance(contact, dict) else {}
    unknown_breakdown = {
        "static_uncertain_samples": _integer(static.get("uncertain_samples")) if isinstance(static, dict) else None,
        "moving_unknown_components": _integer(moving.get("unknown_components")) if isinstance(moving, dict) else None,
        "self_unknown_query_count": _integer(self_collision.get("unknown_query_count")) if isinstance(self_collision, dict) else None,
    }
    unknown_values = [value for value in unknown_breakdown.values() if value is not None]
    return {
        **identity,
        "path": str(directory),
        "errors": {
            "scene_metrics": scene_error, "training_collision": training_error,
            "source_pose": pose_error, "contact_metrics": contact_error,
            "joint_smoothness": smooth_error,
        },
        "static_penetration": {
            "max_mm": _millimeters(static),
            "worst_body": static.get("worst_body") if isinstance(static, dict) else None,
            "worst_frame": _integer(static.get("worst_frame")) if isinstance(static, dict) else None,
        },
        "moving_penetration": {
            "max_mm": _millimeters(moving),
            "worst_body": moving.get("worst_body") if isinstance(moving, dict) else None,
            "worst_frame": _integer(moving.get("worst_frame")) if isinstance(moving, dict) else None,
            "failure_count": _length(moving.get("failures")) if isinstance(moving, dict) else None,
        },
        "self_collision": {
            "max_mm": _millimeters(self_collision),
            "worst_body": self_collision.get("worst_body") if isinstance(self_collision, dict) else None,
            "worst_frame": _integer(self_collision.get("worst_frame")) if isinstance(self_collision, dict) else None,
            "status": self_collision.get("status") if isinstance(self_collision, dict) else None,
        },
        "unknown_count": sum(unknown_values) if unknown_values else (0 if unknown_values else None),
        "unknown_breakdown": unknown_breakdown,
        "reset_geometry_failed_frames": _length(reset.get("failed_frames")) if isinstance(reset, dict) else None,
        "reset_status": reset.get("status") if isinstance(reset, dict) else None,
        "joint_chatter_max": max(joint_chatter, key=lambda item: item["value"], default=None),
        "root_chatter_translation_violations": (
            _integer(root_chatter.get("translation", {}).get("violations"))
            if isinstance(root_chatter.get("translation"), dict) else None
        ),
        "root_chatter_rotation_violations": (
            _integer(root_chatter.get("rotation", {}).get("violations"))
            if isinstance(root_chatter.get("rotation"), dict) else None
        ),
        "velocity_limit_violations": _integer(smooth.get("velocity_limit_violations")) if smooth else None,
        "effort_limit_violations": _integer(smooth.get("effort_limit_violations")) if smooth else None,
        "worst_velocity_joint": max(velocity_joint, key=lambda item: item["value"], default=None),
        "worst_effort_joint": max(effort_joint, key=lambda item: item["value"], default=None),
        "source_pose_worst_part": _worst_stat(pose_parts, "max_rad"),
        "calibrated_orientation_worst_part": _worst_stat(orientation, "max_rad"),
        "contact_sticking_worst_p95": _worst_stat(contact_parts, "p95"),
        "contact_sticking_worst_max": _worst_stat(contact_parts, "max"),
        "root_xy_offset_max_m": _number(root_xy.get("max")) if isinstance(root_xy, dict) else None,
        "root_z_offset_max_m": _number(root_z.get("max")) if isinstance(root_z, dict) else None,
    }


def _compare_npz(candidate: Path, baseline: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "candidate": str(candidate), "baseline": str(baseline),
        "candidate_present": candidate.is_file(), "baseline_present": baseline.is_file(),
        "all_arrays_bitwise_equal": None, "missing_or_invalid_npz": None,
        "candidate_arrays": [], "baseline_arrays": [], "mismatched_arrays": [],
    }
    try:
        with np.load(candidate, allow_pickle=False) as left, np.load(baseline, allow_pickle=False) as right:
            result["candidate_arrays"] = list(left.files)
            result["baseline_arrays"] = list(right.files)
            only_candidate = sorted(set(left.files) - set(right.files))
            only_baseline = sorted(set(right.files) - set(left.files))
            differences = []
            for name in sorted(set(left.files) & set(right.files)):
                first, second = left[name], right[name]
                if first.dtype == second.dtype and first.shape == second.shape and first.tobytes() == second.tobytes():
                    continue
                row: dict[str, Any] = {"array": name, "shape_equal": first.shape == second.shape,
                                       "dtype_equal": str(first.dtype) == str(second.dtype)}
                if first.shape == second.shape and np.issubdtype(first.dtype, np.number) and np.issubdtype(second.dtype, np.number):
                    delta = np.abs(first.astype(np.float64) - second.astype(np.float64))
                    finite = np.isfinite(delta)
                    row["max_abs_diff"] = _number(delta[finite].max()) if finite.any() else None
                    row["nonfinite_or_nan_mismatch"] = int((~finite).sum()) if not np.array_equal(np.isfinite(first), np.isfinite(second)) else 0
                differences.append(row)
            result["mismatched_arrays"] = differences
            result["arrays_only_in_candidate"] = only_candidate
            result["arrays_only_in_baseline"] = only_baseline
            result["all_arrays_bitwise_equal"] = not (only_candidate or only_baseline or differences)
    except (OSError, ValueError) as error:
        result["missing_or_invalid_npz"] = str(error)
    return result


def run_spline_quality_review(
    *, prepared: Path, reference: Path, final: Path, baseline: Path,
    baseline_reference: Path, output: Path,
) -> dict[str, Any]:
    """Invoke the existing read-only per-part quality comparison."""
    command = [
        sys.executable, str(Path(__file__).resolve().parent / "review_spline_quality.py"),
        "--prepared", str(prepared.resolve()), "--reference", str(reference.resolve()),
        "--final", str(final.resolve()), "--baseline", str(baseline.resolve()),
        "--baseline-reference", str(baseline_reference.resolve()),
        "--output", str(output.resolve()),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    return {
        "command": command, "returncode": completed.returncode,
        "stdout": completed.stdout, "stderr": completed.stderr,
        "output": str(output), "output_present": output.is_file(),
    }


def _flatten_quality_report(report: dict[str, Any], identity: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comparisons = report.get("comparisons", {})
    sections = {
        **{f"comparisons.{section}": block for section, block in (comparisons.items() if isinstance(comparisons, dict) else [])},
        "non_degradation_against_A": report.get("non_degradation_against_A", {}),
    }
    for section, block in sections.items():
        metrics = block.get("metrics", {}) if isinstance(block, dict) else {}
        for metric, parts in (metrics.items() if isinstance(metrics, dict) else []):
            for part, stats in (parts.items() if isinstance(parts, dict) else []):
                if not isinstance(stats, dict):
                    continue
                a_stats = stats.get("A", {}) if isinstance(stats.get("A"), dict) else {}
                final = stats.get("final", {}) if isinstance(stats.get("final"), dict) else {}
                delta = stats.get("delta", {}) if isinstance(stats.get("delta"), dict) else {}
                rows.append({
                    **identity, "section": section, "metric": str(metric), "part": str(part),
                    "mean": _number(stats.get("mean", final.get("mean"))),
                    "p95": _number(stats.get("p95", final.get("p95"))),
                    "max": _number(stats.get("max", final.get("max"))),
                    "worst_frame": _integer(stats.get("worst_frame", final.get("worst_frame"))),
                    "abnormal_intervals": stats.get("abnormal_intervals", final.get("abnormal_intervals", [])),
                    "A_mean": _number(a_stats.get("mean")), "A_p95": _number(a_stats.get("p95")),
                    "A_max": _number(a_stats.get("max")),
                    "delta_mean": _number(delta.get("mean")), "delta_p95": _number(delta.get("p95")),
                    "delta_max": _number(delta.get("max")),
                    "worse_than_A_intervals": stats.get("worse_than_A_intervals", []),
                })
    return rows


def _quality_part_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["section"] == "non_degradation_against_A":
            # Deltas in different metrics (and units) are never reordered to
            # choose one "worst" number; retain part x metric granularity.
            grouped[f"{row['part']}\0{row['metric']}"].append(row)
    summary = []
    for key, candidates in sorted(grouped.items()):
        part, metric = key.split("\0", 1)
        by_max = max(candidates, key=lambda row: row["delta_max"] if row["delta_max"] is not None else -math.inf)
        intervals = []
        for row in candidates:
            value = row.get("worse_than_A_intervals")
            intervals.extend(value if isinstance(value, list) else [])
        summary.append({
        "part": part, "worst_delta_max": by_max["delta_max"],
            "metric": metric, "worst_delta_max_section": by_max["section"],
            "worst_delta_mean": by_max["delta_mean"], "worst_delta_mean_metric": metric,
            "worst_delta_p95": by_max["delta_p95"], "worst_delta_p95_metric": metric,
            "worst_frame": by_max["worst_frame"], "worse_than_A_interval_count": len(intervals),
            "worse_than_A_interval_frames": sum(_integer(item.get("frames")) or 0 for item in intervals if isinstance(item, dict)),
        })
    return summary


def _baseline_defects(row: dict[str, Any]) -> list[dict[str, Any]]:
    defects: list[dict[str, Any]] = []

    def add(kind: str, severity: str, **details: Any) -> None:
        defects.append({"kind": kind, "severity": severity, **details})

    for category in ("static_penetration", "moving_penetration", "self_collision"):
        value = row.get(category, {}).get("max_mm")
        if value is None:
            add(category, "missing")
        elif value > 10:
            add(category, "defect_over_1cm", max_mm=value)
        elif value > 5:
            add(category, "residual_over_5mm", max_mm=value)
    if row.get("reset_geometry_failed_frames") is None:
        add("reset_geometry", "missing")
    elif row["reset_geometry_failed_frames"] > 0:
        add("reset_geometry", "failed", failed_frames=row["reset_geometry_failed_frames"])
    chatter = row.get("joint_chatter_max")
    if chatter is None:
        add("joint_chatter", "missing")
    elif chatter["value"] > 0:
        add("joint_chatter", "violated", **chatter)
    for key in ("root_chatter_translation_violations", "root_chatter_rotation_violations",
                "velocity_limit_violations", "effort_limit_violations"):
        value = row.get(key)
        if value is None:
            add(key, "missing")
        elif value > 0:
            add(key, "violated", violations=value)
    if row.get("unknown_count") is None:
        add("unknown_queries", "missing")
    elif row["unknown_count"] > 0:
        add("unknown_queries", "unknown", count=row["unknown_count"])
    for key, error in row.get("errors", {}).items():
        if error is not None:
            add(key, "missing_or_invalid", error=error)
    return defects


def _aggregate_timing(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "available_repeats": sum(row.get("end_to_end_seconds") is not None for row in rows),
        "run_count": len(rows),
        "receipt_missing_runs": sum(row.get("receipt_error") == "missing" for row in rows),
        "solver_failures": sum(row.get("solver_returncode") not in (0, None) for row in rows),
        "audit_failures": sum(bool(row.get("failed_audits")) for row in rows),
    }
    for field in ("production_process_wall_seconds", "end_to_end_seconds", "backend_seconds",
                  "audit_wall_clock_seconds", "contact_scene_seconds"):
        output[field] = _stats([row.get(field) for row in rows])
    for field in ("acquisition", "reweight", "free_bodies", "original_mesh_repair", "other"):
        output[f"stage_{field}"] = _stats([row.get("stage_seconds", {}).get(field) for row in rows])
    audits = defaultdict(list)
    for row in rows:
        for name, value in row.get("audit_seconds", {}).items():
            audits[name].append(value)
    output["audit_seconds"] = {name: _stats(values) for name, values in sorted(audits.items())}
    return output


def collect_report(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    arms = [item.strip() for item in args.arms.split(",") if item.strip()]
    candidate_arms = [item.strip() for item in args.candidate_arms.split(",") if item.strip()]
    unknown = sorted((set(arms) | set(candidate_arms)) - set(DEFAULT_ARMS) - set(candidate_arms))
    if unknown:
        raise ValueError(f"unknown measurement arms {unknown}")
    arms = list(dict.fromkeys([*arms, "A", *candidate_arms]))
    grid = _discover_grid(run_dir, arms, args)
    missing: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for arm in arms:
        for repeat in grid[arm]["repeats"]:
            for clip in grid[arm]["clips"]:
                directory = run_dir / arm / "normal" / f"repeat_{repeat}" / clip
                identity = {"arm": arm, "repeat": repeat, "clip": clip}
                if not (directory / "receipt.json").is_file():
                    missing.append({**identity, "path": str(directory), "reason": "receipt.json missing"})
                timing_rows.append(_timing_row(identity, directory))
                quality_rows.append(_quality_row(identity, directory))

    timing_aggregates: dict[str, dict[str, Any]] = {}
    ratio_rows: list[dict[str, Any]] = []
    arm_clip_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in timing_rows:
        arm_clip_groups[(row["arm"], row["clip"])].append(row)
    for (arm, clip), rows in sorted(arm_clip_groups.items()):
        aggregate = _aggregate_timing(rows)
        timing_aggregates[f"{clip}/{arm}"] = aggregate
        baseline_rows = arm_clip_groups.get(("A", clip), [])
        baseline_values = {
            field: {
                repeat: row.get(field)
                for repeat, row in ((item["repeat"], item) for item in baseline_rows)
            }
            for field in ("end_to_end_seconds", "production_process_wall_seconds", "backend_seconds")
        }
        baseline_means = {
            field: aggregate_item["mean"] if (aggregate_item := _stats(list(values.values()))) else None
            for field, values in baseline_values.items()
        }
        for row in rows:
            ratios: dict[str, Any] = {}
            for field in baseline_values:
                same_repeat = baseline_values[field].get(row["repeat"])
                baseline_value = same_repeat if same_repeat is not None else baseline_means[field]
                ratios[field] = (
                    baseline_value / row[field] if baseline_value is not None and row.get(field) not in (None, 0)
                    else None
                )
            ratio_rows.append({
                "arm": arm, "repeat": row["repeat"], "clip": clip,
                "A_end_to_end_seconds": baseline_values["end_to_end_seconds"].get(row["repeat"]),
                "A_production_seconds": baseline_values["production_process_wall_seconds"].get(row["repeat"]),
                "A_backend_seconds": baseline_values["backend_seconds"].get(row["repeat"]),
                "used_A_mean_fallback": all(
                    baseline_values[field].get(row["repeat"]) is None and baseline_means[field] is not None
                    for field in baseline_values
                ),
                **{f"A_over_arm_{field}": ratios[field] for field in baseline_values},
            })
        aggregate["ratios"] = {
            f"A_over_arm_{field}": _stats([
                row[f"A_over_arm_{field}"] for row in ratio_rows
                if row["arm"] == arm and row["clip"] == clip
            ]) for field in baseline_values
        }

    identity_rows = [row for row in timing_rows if row["arm"] != "A"]
    comparisons = [row for row in identity_rows]
    for row in comparisons:
        baseline = run_dir / "A" / "normal" / "repeat_1" / row["clip"] / "final_retargeted.npz"
        candidate = Path(row["path"]) / "final_retargeted.npz"
        row["bitwise_final_vs_A_repeat_1"] = _compare_npz(candidate, baseline)

    review_results: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    review_summaries: list[dict[str, Any]] = []
    if args.run_quality_review:
        for row in quality_rows:
            if row["arm"] not in SPLINE_EXPERIMENT_ARMS:
                continue
            directory = Path(row["path"])
            baseline_directory = run_dir / "A" / "normal" / "repeat_1" / row["clip"]
            output = output_dir / "review_quality" / row["arm"] / f"repeat_{row['repeat']}" / f"{row['clip']}.json"
            archive = output.with_suffix(".json.gz")
            output.parent.mkdir(parents=True, exist_ok=True)
            identity = {"arm": row["arm"], "repeat": row["repeat"], "clip": row["clip"]}
            record: dict[str, Any] = {
                **identity, "output": str(output), "archive": str(archive),
                "status": "cached" if output.is_file() or archive.is_file() else "launched",
            }
            if not output.is_file() and not archive.is_file():
                record.update(run_spline_quality_review(
                    prepared=directory / "prepared.npy", reference=directory / "fixed_reference.npz",
                    final=directory / "final_retargeted.npz", baseline=baseline_directory / "final_retargeted.npz",
                    baseline_reference=baseline_directory / "fixed_reference.npz", output=output,
                ))
                record["status"] = "completed" if record.get("returncode") == 0 and output.is_file() else "failed"
            if output.is_file():
                report, error = _read_json(output)
            elif archive.is_file():
                report, error = _read_gzip_json(archive)
            else:
                report, error = None, "missing"
            record["output_error"] = error
            if report is not None:
                flattened = _flatten_quality_report(report, identity)
                review_rows.extend(flattened)
                review_summaries.extend({**item, **identity} for item in _quality_part_summary(flattened))
                if output.is_file() and not getattr(args, "keep_quality_raw", False):
                    with output.open("rb") as source, gzip.open(archive, "wt", encoding="utf-8", compresslevel=9) as target:
                        target.write(source.read().decode("utf-8"))
                    output.unlink()
            review_results.append(record)

    baseline_rows = [row for row in quality_rows if row["arm"] == "A"]
    for row in baseline_rows:
        row["baseline_defects"] = _baseline_defects(row)
    nonregression = build_nonregression(run_dir, grid, candidate_arms)
    expected_runs = sum(len(value["clips"]) * len(value["repeats"]) for value in grid.values())
    completed_runs = sum(row["status"] == "present" for row in timing_rows)
    return {
        "schema_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_directory": str(run_dir),
        "report_directory": str(output_dir),
        "arms": arms,
        "expected_grid": grid,
        "expected_normal_runs": expected_runs,
        "receipt_present_runs": completed_runs,
        "missing_runs": missing,
        "provisional": bool(missing) or completed_runs != expected_runs,
        "pipeline_invoked": False,
        "quality_review_invoked": bool(args.run_quality_review),
        "time_runs": timing_rows,
        "timing_aggregates": timing_aggregates,
        "timing_ratios": ratio_rows,
        "bitwise_final_comparisons": comparisons,
        "quality_rows": quality_rows,
        "spline_quality_review_runs": review_results,
        "spline_quality_rows": review_rows,
        "spline_quality_part_worst": review_summaries,
        "clip_nonregression": nonregression,
        "baseline_defect_statement": "A 的下列数值是基线缺陷与未验证项，只用于对照，不宣称合格。",
    }


def _cell(value: Any, digits: int = 2) -> str:
    if value is None:
        return "missing"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _stats_cell(stats: dict[str, float] | None, digits: int = 2) -> str:
    if stats is None:
        return "missing"
    return f"{stats['mean']:.{digits}f}/{stats['min']:.{digits}f}/{stats['max']:.{digits}f}"


def _ratio_cell(stats: dict[str, float] | None) -> str:
    return "missing" if stats is None else f"{stats['mean']:.3f}×"


def _geo_cell(row: dict[str, Any], key: str) -> str:
    block = row.get(key, {})
    if not isinstance(block, dict) or block.get("max_mm") is None:
        return "missing"
    return f"{block['max_mm']:.2f}@{block.get('worst_body', '?')}#{block.get('worst_frame', '?')}"


def _optional_part(row: dict[str, Any], key: str, value_key: str = "value") -> str:
    block = row.get(key)
    if not isinstance(block, dict):
        return "missing"
    value = block.get(value_key)
    rendered = f"{value:.6g}" if isinstance(value, float) else str(value)
    return f"{rendered}@{block.get('part', '?')}"


def markdown(result: dict[str, Any]) -> str:
    lines = [
        "# spline_windowed_20261005 测量队列报告", "",
        f"生成时间：{result['generated_utc']}。只读取已有 receipt/audit/npz；未运行求解流水线。",
        f"receipt 覆盖：{result['receipt_present_runs']}/{result['expected_normal_runs']}；缺失 {len(result['missing_runs'])} 条，未剔除。",
        "所有比值均为 A/arm（数值大于 1 表示 A 更慢）。审计墙钟 = `end_to_end_seconds - production_process_wall_seconds`，并发时不是各审计 seconds 的加和。", "",
        "## 时间表（跨 repeat 的 mean/min/max）", "",
        "| clip | arm | n | 端到端 (s) | A/arm | 生产 (s) | A/arm | backend (s) | A/arm | 审计墙钟 | acquisition | reweight | free | repair | other | scene | training | pose | contact | smooth | 失败 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in sorted(result["timing_aggregates"].items()):
        clip, arm = key.rsplit("/", 1)
        audits = row["audit_seconds"]
        def audit(name: str) -> str:
            return _stats_cell(audits.get(name))
        lines.append(
            f"| {clip} | {arm} | {row['available_repeats']}/{row['run_count']} | "
            f"{_stats_cell(row['end_to_end_seconds'])} | {_ratio_cell(row['ratios']['A_over_arm_end_to_end_seconds'])} | "
            f"{_stats_cell(row['production_process_wall_seconds'])} | {_ratio_cell(row['ratios']['A_over_arm_production_process_wall_seconds'])} | "
            f"{_stats_cell(row['backend_seconds'])} | {_ratio_cell(row['ratios']['A_over_arm_backend_seconds'])} | "
            f"{_stats_cell(row['audit_wall_clock_seconds'])} | {_stats_cell(row['stage_acquisition'])} | "
            f"{_stats_cell(row['stage_reweight'])} | {_stats_cell(row['stage_free_bodies'])} | "
            f"{_stats_cell(row['stage_original_mesh_repair'])} | {_stats_cell(row['stage_other'])} | "
            f"{audit('scene_metrics')} | {audit('training_collision')} | {audit('source_pose')} | "
            f"{audit('contact_metrics')} | {audit('joint_smoothness')} | "
            f"missing={row['receipt_missing_runs']},solver={row['solver_failures']},audit={row['audit_failures']} |"
        )
    lines.extend([
        "", "## 逐位一致性（相对 A/repeat_1 final_retargeted.npz）", "",
        "| arm | repeat | clip | bitwise equal | 不等数组 | max|diff| |",
        "|---|---:|---|---|---|---:|",
    ])
    for row in result["bitwise_final_comparisons"]:
        comparison = row["bitwise_final_vs_A_repeat_1"]
        mismatches = comparison.get("mismatched_arrays", [])
        names = ",".join(item["array"] for item in mismatches)
        max_diff = max((item.get("max_abs_diff") for item in mismatches if item.get("max_abs_diff") is not None), default=None)
        if comparison.get("arrays_only_in_candidate") or comparison.get("arrays_only_in_baseline"):
            names = (names + ",key-set-differs").strip(",")
        lines.append(f"| {row['arm']} | {row['repeat']} | {row['clip']} | {comparison.get('all_arrays_bitwise_equal')} | {_cell(names)} | {_cell(max_diff, 6)} |")
    lines.extend([
        "", "## 质量表（每 clip/arm/repeat）", "",
        "| clip | arm | rep | static mm | moving mm | self mm | unknown | reset fail | joint chatter | root t/r | vel | effort | pose max | slip P95/max | root xy/z |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|---|",
    ])
    for row in result["quality_rows"]:
        root = (row.get("root_chatter_translation_violations"), row.get("root_chatter_rotation_violations"))
        root_offsets = (row.get("root_xy_offset_max_m"), row.get("root_z_offset_max_m"))
        lines.append(
            f"| {row['clip']} | {row['arm']} | {row['repeat']} | {_geo_cell(row, 'static_penetration')} | "
            f"{_geo_cell(row, 'moving_penetration')} | {_geo_cell(row, 'self_collision')} | {_cell(row.get('unknown_count'))} | "
            f"{_cell(row.get('reset_geometry_failed_frames'))} | {_optional_part(row, 'joint_chatter_max')} | "
            f"{_cell(root[0])}/{_cell(root[1])} | {_cell(row.get('velocity_limit_violations'))} | "
            f"{_cell(row.get('effort_limit_violations'))} | {_optional_part(row, 'source_pose_worst_part', 'value')} | "
            f"{_optional_part(row, 'contact_sticking_worst_p95')}/{_optional_part(row, 'contact_sticking_worst_max')} | "
            f"{_cell(root_offsets[0], 4)}/{_cell(root_offsets[1], 4)} |"
        )
    lines.extend([
        "", "## B/C/D 相对 A 的逐部位×指标最坏退化（不跨指标/单位排序）", "",
        "| arm | rep | clip | part | metric | worst Δmax | worst Δmean | worst Δp95 | worse frames |",
        "|---|---:|---|---|---:|---|---:|---:|---:|",
    ])
    for row in result["spline_quality_part_worst"]:
        lines.append(
            f"| {row['arm']} | {row['repeat']} | {row['clip']} | {row['part']} | {_cell(row['metric'])} | "
            f"{_cell(row['worst_delta_max'], 6)} | "
            f"{_cell(row['worst_delta_mean'], 6)} | {_cell(row['worst_delta_p95'], 6)} | "
            f"{_cell(row['worse_than_A_interval_frames'])} |"
        )
    if not result["spline_quality_part_worst"]:
        lines.append("| missing | - | - | - | missing | missing | missing | missing | missing |")
    lines.extend([
        "", "## A 基线缺陷清单", "",
        result["baseline_defect_statement"], "",
        "| clip | rep | static mm | moving mm | self mm | reset fail | chatter max | effort | velocity | unknown | 缺陷/未验证 |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ])
    for row in result["quality_rows"]:
        if row["arm"] != "A":
            continue
        if not any(error is None for error in row.get("errors", {}).values()):
            continue
        defects = row.get("baseline_defects", [])
        compact = ",".join(f"{item['kind']}:{item['severity']}" for item in defects)
        lines.append(
            f"| {row['clip']} | {row['repeat']} | {_geo_cell(row, 'static_penetration')} | "
            f"{_geo_cell(row, 'moving_penetration')} | {_geo_cell(row, 'self_collision')} | "
            f"{_cell(row.get('reset_geometry_failed_frames'))} | {_optional_part(row, 'joint_chatter_max')} | "
            f"{_cell(row.get('effort_limit_violations'))} | {_cell(row.get('velocity_limit_violations'))} | "
            f"{_cell(row.get('unknown_count'))} | {_cell(compact)} |"
        )
    lines.extend(nonregression_markdown(result["clip_nonregression"]).splitlines())
    lines.extend([
        "", "## 口径与限制", "",
        "- 穿透为几何报告值，unknown/reset/physics 未验证不合并为通过；5–10 mm 标为 residual_over_5mm，>10 mm 标为 defect_over_1cm。",
        "- joint chatter 为逐关节 violations 的最大值及最差关节，不做跨关节平均；速度/力矩为审计计数。",
        "- B/C/D 的 review_quality 在 report 目录生成；缺失 A repeat_1 输入时保持 failed/missing。",
        "- `budgets` 与 `skipped_tasks` 原文在 JSON 的 `time_runs` 中；本报告不新增通过判定。", "",
    ])
    return "\n".join(lines)


def write_report(args: argparse.Namespace) -> dict[str, Any]:
    result = collect_report(args)
    args.output_dir.resolve().mkdir(parents=True, exist_ok=True)
    (args.output_dir / "measurement_summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (args.output_dir / "measurement_summary.md").write_text(markdown(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    parser.add_argument("--candidate-arms", default="B,C,D,E,E2",
                        help="non-regression candidates relative to A; unknown arm names are discovered from the queue")
    parser.add_argument("--a-repeats", type=int, default=4)
    parser.add_argument("--e-repeats", type=int, default=2)
    parser.add_argument("--no-quality-review", action="store_true")
    parser.add_argument("--keep-quality-raw", action="store_true",
                        help="keep the very large review_spline_quality JSON files after flattening")
    args = parser.parse_args()
    if args.a_repeats < 1 or args.e_repeats < 1:
        parser.error("repeat counts must be positive")
    args.run_quality_review = not args.no_quality_review
    args.keep_quality_raw = args.keep_quality_raw
    result = write_report(args)
    print(json.dumps({
        "receipt_present_runs": result["receipt_present_runs"],
        "expected_normal_runs": result["expected_normal_runs"],
        "missing_runs": len(result["missing_runs"]),
        "quality_reviews": len(result["spline_quality_review_runs"]),
        "output": str(args.output_dir / "measurement_summary.md"),
    }))


if __name__ == "__main__":
    main()
