"""Noise-bounded, priority-ordered measurement comparisons against arm A."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PRIORITY_NAMES = {
    1: "P1 时间连续性", 2: "P2 非穿透与合格率", 3: "P3 接触",
    4: "P4 根路径", 5: "P5 姿态",
}
NOISE_FLOOR = 1e-9


def _read(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


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


def _frames(value: Any) -> set[int]:
    return {int(item) for item in value} if isinstance(value, list) else set()


def _length(value: Any) -> int | None:
    return len(value) if isinstance(value, (list, tuple)) else None


def _percentile(values: np.ndarray) -> float | None:
    return _number(np.quantile(values, .95)) if values.size else None


def _millimeters(value: Any) -> float | None:
    number = _number(value)
    return number * 1000 if number is not None else None


_MODEL_CACHE: dict[str, Any] = {}


def _scene_per_body(directory: Path, kind: str) -> dict[str, dict[str, float]]:
    arrays = directory / "scene_metrics.npz"
    prepared = directory / "prepared.npy"
    if not arrays.is_file() or not prepared.is_file():
        return {}
    try:
        inputs = np.load(prepared, allow_pickle=True).item()
        model_path = str(inputs["model_path"])
        if model_path not in _MODEL_CACHE:
            import mujoco
            from omniretargeting.scene_geometry import collision_geom_ids
            model = mujoco.MjModel.from_xml_path(model_path)
            _MODEL_CACHE[model_path] = (model, np.asarray(collision_geom_ids(model)))
        model, geom_ids = _MODEL_CACHE[model_path]
        with np.load(arrays, allow_pickle=False) as saved:
            depths = np.asarray(saved[f"{kind}_depths"], float)
        if depths.ndim != 2 or depths.shape[1] != model.ngeom:
            return {}
        result = {}
        for body_id in np.unique(model.geom_bodyid[geom_ids]):
            subset = geom_ids[model.geom_bodyid[geom_ids] == body_id]
            body_depth = depths[:, subset].max(axis=1)
            result[model.body(int(body_id)).name] = {
                "max_mm": _number(body_depth.max()) * 1000,
                "p95_mm": _percentile(body_depth) * 1000,
            }
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def _observe(
    observations: dict[str, dict[str, Any]], priority: int, metric: str,
    part: str, value: Any, *, higher: bool = False, unit: str = "",
) -> None:
    number = _number(value)
    if number is None:
        return
    key = f"P{priority}|{metric}|{part}"
    observations[key] = {
        "key": key, "priority": priority, "metric": metric, "part": part,
        "value": number, "higher_is_better": higher, "unit": unit,
    }


def _add_contact_metrics(observations: dict[str, dict[str, Any]], contact: dict[str, Any]) -> None:
    units = {
        "surface_sticking_target_relative_slip_m_s": "m/s",
        "surface_vector_error_m": "m", "surface_normal_error_m": "m",
    }
    for metric, unit in units.items():
        parts = contact.get(metric, {})
        for part, values in parts.items() if isinstance(parts, dict) else []:
            if not isinstance(values, dict):
                continue
            for field in ("p95", "max"):
                _observe(observations, 3, f"{metric}.{field}", str(part), values.get(field), unit=unit)
    for part in ("root_xy_offset_m", "root_z_offset_m"):
        _observe(observations, 4, f"{part}.max", "root", contact.get(part, {}).get("max") if isinstance(contact.get(part), dict) else None, unit="m")
    root = contact.get("source_root_correspondence", {})
    if isinstance(root, dict):
        for axis in ("xy", "z"):
            block = root.get(f"{axis}_error_m", {})
            _observe(observations, 4, f"source_root_correspondence.{axis}.max", axis,
                     block.get("max") if isinstance(block, dict) else None, unit="m")


def _add_pose(observations: dict[str, dict[str, Any]], pose: dict[str, Any], report: dict[str, Any]) -> None:
    for category in ("parts", "calibrated_body_orientation"):
        parts = pose.get(category, {})
        for part, values in parts.items() if isinstance(parts, dict) else []:
            if not isinstance(values, dict):
                continue
            for field in ("mean_rad", "p95_rad", "max_rad"):
                _observe(observations, 5, f"{category}.{field}", str(part), values.get(field), unit="rad")
    shape = report.get("source_shape", {})
    residuals = report.get("final_residuals", {})
    _observe(observations, 5, "layout_flip.frames", "layout", _length(shape.get("layout_flip_frames")) if isinstance(shape, dict) else None)
    _observe(observations, 5, "layout_flip.max", "layout", residuals.get("layout_flip_max") if isinstance(residuals, dict) else None)


def row_observations(directory: Path) -> dict[str, dict[str, Any]]:
    scene = _read(directory / "scene_metrics.json") or {}
    training = _read(directory / "training_collision.json") or {}
    contact = _read(directory / "contact_metrics.json") or {}
    pose = _read(directory / "source_pose.json") or {}
    smooth = _read(directory / "joint_smoothness.json") or {}
    report = _read(directory / "fast_report.json") or {}
    observations: dict[str, dict[str, Any]] = {}

    joints = smooth.get("joints", {})
    chatter_values = []
    if isinstance(joints, dict):
        for name, values in joints.items():
            if not isinstance(values, dict):
                continue
            chatter = values.get("source_relative_chatter", {})
            value = _integer(chatter.get("violations")) if isinstance(chatter, dict) else None
            if value is not None:
                chatter_values.append(value)
                _observe(observations, 1, "joint_chatter.violations", str(name), value)
    if chatter_values and len(chatter_values) == len(joints):
        _observe(observations, 1, "joint_chatter.total_violations", "all_joints", sum(chatter_values))
    root = smooth.get("root_source_relative_chatter", {})
    if isinstance(root, dict):
        for axis in ("translation", "rotation"):
            _observe(observations, 1, f"root_chatter.{axis}.violations", axis,
                     root.get(axis, {}).get("violations") if isinstance(root.get(axis), dict) else None)
    _observe(observations, 1, "velocity_limit.violations", "all_joints", smooth.get("velocity_limit_violations"))
    _observe(observations, 1, "effort_limit.violations", "all_joints", smooth.get("effort_limit_violations"))

    frames = _integer((training.get("inputs") or {}).get("input_frames")) or _integer(contact.get("frames")) or _integer(scene.get("frames"))
    reset = training.get("reset", {})
    failed = _frames(reset.get("failed_frames")) if isinstance(reset, dict) else set()
    if frames is not None and failed is not None:
        _observe(observations, 2, "reset.passed_frames", "trajectory", frames - len(failed), higher=True)
    static_training = training.get("static_scene", {})
    moving_training = training.get("moving_objects", {})
    self_training = training.get("self_collision", {})
    over_1cm, over_5mm = set(), set()
    for block in (static_training, moving_training, self_training):
        if isinstance(block, dict):
            over_1cm |= _frames(block.get("frames_over_1cm"))
            over_5mm |= _frames(block.get("frames_over_5mm"))
    _observe(observations, 2, "penetration.frames_over_1cm_union", "trajectory", len(over_1cm))
    _observe(observations, 2, "penetration.frames_over_5mm_union", "trajectory", len(over_5mm))
    static_scene, moving_scene = scene.get("static", {}), scene.get("moving", {})
    unknown = sum(_integer(block.get(key)) or 0 for block, key in (
        (static_scene, "uncertain_samples"), (moving_scene, "unknown_components"),
        (self_training, "unknown_query_count"),
    ) if isinstance(block, dict))
    if any(isinstance(block, dict) and block.get(key) is not None for block, key in (
        (static_scene, "uncertain_samples"), (moving_scene, "unknown_components"),
        (moving_scene, "unknown_components"), (self_training, "unknown_query_count"),
    )):
        _observe(observations, 2, "unknown.total_count", "queries",
                 unknown + (_length(moving_scene.get("failures")) if isinstance(moving_scene, dict) else 0))

    for kind, report_block, computed in (
        ("static", static_scene, _scene_per_body(directory, "static")),
        ("moving", moving_scene, _scene_per_body(directory, "moving")),
    ):
        rows = computed or {}
        if not rows and isinstance(report_block, dict):
            rows = {str(report_block.get("worst_body", "aggregate")): {
                "max_mm": _millimeters(report_block.get("max_m")),
                "p95_mm": _millimeters(report_block.get("p95_m")),
            }}
        for part, values in rows.items():
            for field in ("max_mm", "p95_mm"):
                _observe(observations, 2, f"{kind}_penetration.{field}", part, values.get(field), unit="mm")
    self_bodies = self_training.get("per_body", {}) if isinstance(self_training, dict) else {}
    for part, values in self_bodies.items() if isinstance(self_bodies, dict) else []:
        if not isinstance(values, dict):
            continue
        for field in ("max_m", "p95_m"):
            _observe(observations, 2, f"self_penetration.{field.removesuffix('_m')}_mm", str(part),
                     _millimeters(values.get(field)), unit="mm")

    _add_contact_metrics(observations, contact)
    _add_pose(observations, pose, report)
    return observations


def _compare_observations(rows: dict[str, list[dict[str, Any]]], key: str, metadata: dict[str, Any]) -> dict[str, Any]:
    baseline = [row.get(key) for row in rows.get("A", [])]
    candidate = [row.get(key) for row in rows.get("candidate", [])]
    baseline_values = [item["value"] for item in baseline if item is not None]
    candidate_values = [item["value"] for item in candidate if item is not None]
    if not baseline_values or not candidate_values:
        return {**metadata, "status": "missing", "A": None, "candidate": None,
                "A_repeats": len(baseline_values), "candidate_repeats": len(candidate_values)}
    a_value, candidate_value = statistics.median(baseline_values), statistics.median(candidate_values)
    tolerance = max(NOISE_FLOOR, max(baseline_values) - min(baseline_values), max(candidate_values) - min(candidate_values))
    delta = candidate_value - a_value
    if metadata["higher_is_better"]:
        verdict = "better" if delta > tolerance else "worse" if delta < -tolerance else "flat"
    else:
        verdict = "worse" if delta > tolerance else "better" if delta < -tolerance else "flat"
    return {**metadata, "status": verdict, "A": a_value, "candidate": candidate_value,
            "delta": delta, "noise_tolerance": tolerance,
            "A_repeat_values": baseline_values, "candidate_repeat_values": candidate_values,
            "A_repeats": len(baseline_values), "candidate_repeats": len(candidate_values)}


def _clip_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped["A" if row["arm"] == "A" else "candidate"].append(row["observations"])
    keys = sorted({key for row in rows for key in row["observations"]})
    compared = [_compare_observations(grouped, key, next(
        row["observations"][key] for row in rows if key in row["observations"]
    ) | {"key": key}) for key in keys]
    priorities = {}
    direction = "all_flat"
    for priority, name in PRIORITY_NAMES.items():
        items = [item for item in compared if item["priority"] == priority]
        worse = [item for item in items if item["status"] == "worse"]
        better = [item for item in items if item["status"] == "better"]
        priorities[str(priority)] = {
            "name": name, "worse": worse, "better": better,
            "flat_count": sum(item["status"] == "flat" for item in items),
            "missing_count": sum(item["status"] == "missing" for item in items),
            "indicator_count": len(items),
        }
        if not items:
            continue
        if worse and direction in ("all_flat",):
            direction = f"P{priority} worse"
        elif better and direction == "all_flat":
            direction = f"P{priority} better"
    return {"priorities": priorities, "summary_direction": direction, "comparisons": compared}


def build(run_dir: Path, grid: dict[str, Any], candidate_arms: list[str]) -> dict[str, Any]:
    rows = []
    for arm in ["A", *candidate_arms]:
        for repeat in grid.get(arm, {}).get("repeats", []):
            for clip in grid.get(arm, {}).get("clips", []):
                directory = run_dir / arm / "normal" / f"repeat_{repeat}" / clip
                rows.append({"arm": arm, "repeat": repeat, "clip": clip,
                             "directory": directory, "observations": row_observations(directory)})
    clip_results: dict[str, dict[str, Any]] = {}
    clips = sorted({row["clip"] for row in rows})
    for clip in clips:
        for candidate in candidate_arms:
            selected = [row for row in rows if row["clip"] == clip and row["arm"] in ("A", candidate)]
            if selected:
                comparison = _clip_comparison(selected)
                comparison["candidate_arm"] = candidate
                comparison["A_repeats"] = sorted(row["repeat"] for row in selected if row["arm"] == "A")
                comparison["candidate_repeats"] = sorted(row["repeat"] for row in selected if row["arm"] == candidate)
                clip_results[f"{candidate}/{clip}"] = comparison
    cross = {}
    aggregate_keys = {
        "reset.passed_frames": "P2|reset.passed_frames|trajectory",
        "frames_over_1cm": "P2|penetration.frames_over_1cm_union|trajectory",
        "joint_chatter_total": "P1|joint_chatter.total_violations|all_joints",
    }
    for candidate in candidate_arms:
        values: dict[str, dict[str, list[float]]] = {name: {"A": [], "candidate": []} for name in aggregate_keys}
        for label, key in aggregate_keys.items():
            for clip in clips:
                result = clip_results.get(f"{candidate}/{clip}")
                if result is None:
                    continue
                item = next((entry for entry in result["comparisons"] if entry["key"] == key), None)
                if item and item["status"] != "missing":
                    values[label]["A"].append(item["A"])
                    values[label]["candidate"].append(item["candidate"])
        cross[candidate] = {
            "method": "sum of per-clip median representatives; no aggregate verdict",
            "A_clips": [clip for clip in clips if f"{candidate}/{clip}" in clip_results],
            **{label: {"A": sum(parts["A"]), "candidate": sum(parts["candidate"]),
                       "clip_count": len(parts["A"])} for label, parts in values.items()},
        }
    return {"noise_floor": NOISE_FLOOR,
            "noise_rule": "max(1e-9, A repeat spread, candidate repeat spread)",
            "representative": "median of available repeats; incomplete data remains labeled",
            "clips": clip_results, "cross_clip": cross}


def _value(value: Any) -> str:
    return "missing" if value is None else f"{value:.9g}" if isinstance(value, float) else str(value)


def markdown(result: dict[str, Any]) -> str:
    lines = ["", "## 逐 clip 非退化判定（候选相对 A）", "",
             "代表值为各 repeat 的中位数；噪声容差为 `max(1e-9, A repeat 极差, 候选 repeat 极差)`。本节只标注变好/变差/持平，不判定采用或拒绝。", "",
             "### 跨 clip 代表值汇总", "",
             "| arm | 合格帧 A/候选 | >1cm 帧 A/候选 | chatter 总数 A/候选 | clip 数 |",
             "|---|---:|---:|---:|---:|"]
    for arm, row in result["cross_clip"].items():
        reset, over, chatter = row["reset.passed_frames"], row["frames_over_1cm"], row["joint_chatter_total"]
        lines.append(f"| {arm} | {reset['A']:.3g}/{reset['candidate']:.3g} | "
                     f"{over['A']:.3g}/{over['candidate']:.3g} | "
                     f"{chatter['A']:.3g}/{chatter['candidate']:.3g} | {reset['clip_count']} |")
    for key, clip in result["clips"].items():
        arm, clip_name = key.split("/", 1)
        lines.extend(["", f"### {arm} / {clip_name}", "",
                      f"最高优先级变化：`{clip['summary_direction']}`。", "",
                      "| 优先级 | 变差 | 变好 | 持平 | 缺失 |", "|---|---:|---:|---:|---:|"])
        for priority in sorted(clip["priorities"]):
            row = clip["priorities"][priority]
            lines.append(f"| {row['name']} | {len(row['worse'])} | {len(row['better'])} | {row['flat_count']} | {row['missing_count']} |")
        changed = [item for row in clip["priorities"].values() for item in row["worse"] + row["better"]]
        if changed:
            lines.extend(["", "| 方向 | 优先级 | 指标 | 部位 | A | 候选 | Δ | 容差 |", "|---|---|---|---|---:|---:|---:|---:|"])
            for item in changed:
                lines.append(f"| {'变差' if item['status'] == 'worse' else '变好'} | P{item['priority']} | "
                             f"{item['metric']} | {item['part']} | {_value(item['A'])} | {_value(item['candidate'])} | "
                             f"{_value(item['delta'])} | {_value(item['noise_tolerance'])} |")
    return "\n".join(lines)
