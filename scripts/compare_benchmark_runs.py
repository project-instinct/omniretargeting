#!/usr/bin/env python3
"""Compare two audited benchmark arms clip by clip.

The report is deliberately worst-case oriented: every headline metric keeps its
per-clip value and the available worst body/joint/frame, rather than replacing
them with an arm average. Missing audit files are reported and do not abort the
comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


FILES = (
    "timing.json",
    "receipt.json",
    "fast_report.json",
    "scene_metrics.json",
    "contact_metrics.json",
    "joint_smoothness.json",
    "source_pose.json",
    "training_collision.json",
    "penetration_persistence.json",
)

# The non-regression rule follows harness/compare.py: a change must exceed both
# an absolute floor for its unit and 2% of the before value.
FLOOR = {
    "": 0.5,
    "count": 0.5,
    "frame": 0.5,
    "frames": 0.5,
    "m": 5e-5,
    "m/s": 1e-3,
    "rad": 1e-3,
    "ratio": 0.005,
    "s": 0.05,
    "seconds": 0.05,
}

DEFECT_LABELS = {"deep", "persistent"}


def _read_json(path: Path) -> dict[str, Any]:
    value: dict[str, Any] = {"present": False, "data": None, "error": None}
    if not path.is_file():
        return value
    value["present"] = True
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError("JSON root is not an object")
        value["data"] = data
    except Exception as error:  # malformed output is a report field, not a crash
        value["error"] = f"{type(error).__name__}: {error}"
    return value


def load_clip(root: Path, clip: str) -> dict[str, Any]:
    directory = root / clip
    return {
        "directory": directory,
        "files": {name: _read_json(directory / name) for name in FILES},
    }


def data(run: dict[str, Any], name: str) -> dict[str, Any] | None:
    item = run["files"][name]
    return item["data"] if item["present"] and item["error"] is None else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _fmt(value: Any, digits: int = 6) -> str:
    number = _number(value)
    if number is not None:
        if float(number).is_integer() and abs(number) < 1e12:
            return str(int(number))
        return f"{number:.{digits}g}"
    if value is None:
        return "缺失"
    if isinstance(value, (list, tuple)):
        return f"list[{len(value)}]"
    if isinstance(value, dict):
        return f"dict[{len(value)}]"
    return str(value).replace("|", "\\|").replace("\n", " ")


def _delta(before: Any, after: Any, unit: str = "") -> str:
    a, b = _number(before), _number(after)
    if a is None or b is None:
        return "—"
    return f"{_fmt(b - a)}{unit}"


def _row(label: str, before: Any, after: Any, unit: str = "") -> str:
    suffix = f" {unit}" if unit else ""
    return (
        f"| {label} | {_fmt(before)}{suffix} | {_fmt(after)}{suffix} | "
        f"{_delta(before, after, unit)} |"
    )


def _safe_name(value: Any) -> str:
    return _fmt(value)


def _nested(record: dict[str, Any] | None, *keys: str, default=None) -> Any:
    value: Any = record
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    return value


def _frames(run: dict[str, Any]) -> Any:
    fast = data(run, "fast_report.json")
    persistence = data(run, "penetration_persistence.json")
    contact = data(run, "contact_metrics.json")
    smooth = data(run, "joint_smoothness.json")
    return (
        _nested(fast, "frames")
        or _nested(persistence, "inputs", "frames")
        or _nested(contact, "frames")
        or _nested(smooth, "frames")
    )


def _timing(run: dict[str, Any]) -> dict[str, Any]:
    timing = data(run, "timing.json") or {}
    receipt = data(run, "receipt.json") or {}
    stages = timing.get("stage_seconds", {})
    stages = stages if isinstance(stages, dict) else {}
    fast_seconds = _number(timing.get("fast_backend_seconds"))
    frames = _number(_frames(run))
    return {
        "production": _number(receipt.get("production_process_wall_seconds")),
        "fast": fast_seconds,
        "frames": frames,
        "per_frame": None if fast_seconds is None or not frames else fast_seconds / frames,
        "stages": {
            str(name): _number(value) for name, value in stages.items()
        },
    }

def _pair_row(pair: dict[str, Any]) -> dict[str, Any]:
    thresholds = pair.get("thresholds_m", {})

    def threshold(value: float) -> dict[str, Any]:
        for key, item in thresholds.items():
            if math.isclose(_number(key) or math.nan, value, rel_tol=0.0, abs_tol=1e-12):
                return item if isinstance(item, dict) else {}
        return {}

    t5, t2 = threshold(0.005), threshold(0.002)
    robot = pair.get("robot_geom", {})
    return {
        "pair_id": pair.get("pair_id"),
        "kind": pair.get("kind"),
        "body": robot.get("body") or robot.get("name") or "unknown",
        "other": pair.get("other"),
        "Dmax_m": _number(pair.get("max_depth_m")),
        "L5_frames": _number(t5.get("longest_run_frames")),
        "F5_frames": (
            len(t5.get("frames", []))
            if isinstance(t5.get("frames", []), list)
            else _number(t5.get("frame_fraction"))
        ),
        "L2_frames": _number(t2.get("longest_run_frames")),
        "worst_frame": pair.get("worst_frame"),
        "labels": pair.get("policy_v1", []),
    }


def _penetration(run: dict[str, Any]) -> dict[str, Any]:
    record = data(run, "penetration_persistence.json")
    if record is None:
        return {"missing": True}
    pairs = [pair for pair in record.get("pairs", []) if isinstance(pair, dict)]
    rows = [_pair_row(pair) for pair in pairs]
    defects = [row for row in rows if DEFECT_LABELS.intersection(row["labels"])]
    summary = record.get("summary", {})
    reset = record.get("reset", {})
    usable_count = _number(reset.get("usable_frame_count"))
    unusable_count = _number(reset.get("unusable_frame_count"))
    frames = _number(_nested(record, "inputs", "frames"))
    unknown_reset = None
    if usable_count is not None and unusable_count is not None and frames is not None:
        unknown_reset = max(0, frames - usable_count - unusable_count)
    return {
        "missing": False,
        "defect_count": _number(_nested(record, "verdict", "policy_v1_defect_pairs")),
        "counts": _nested(summary, "policy_v1_pair_counts", default={}),
        "depths": summary.get("max_depth_by_kind_m", {}),
        "over_5mm": len(_nested(summary, "union_frames", "over_5mm", default=[])),
        "over_1cm": len(_nested(summary, "union_frames", "over_1cm", default=[])),
        "unknown": _nested(summary, "unknown", "total"),
        "reset": {
            "usable": usable_count,
            "unusable": unusable_count,
            "unknown": unknown_reset,
        },
        "pairs": rows,
        "defects": sorted(
            defects, key=lambda row: -(row["Dmax_m"] or 0.0)
        ),
    }


def _body_family(run: dict[str, Any], field: str) -> dict[str, Any]:
    record = data(run, "contact_metrics.json") or {}
    values = record.get(field, {})
    values = values if isinstance(values, dict) else {}
    rows = {
        str(body): {
            "p95": _number(item.get("p95")) if isinstance(item, dict) else None,
            "max": _number(item.get("max")) if isinstance(item, dict) else None,
        }
        for body, item in values.items()
    }
    worst = max(rows, key=lambda name: rows[name]["max"] or -math.inf, default=None)
    worst_p95 = max(
        rows, key=lambda name: rows[name]["p95"] or -math.inf, default=None
    )
    return {
        "bodies": rows,
        "worst_max_body": worst,
        "worst_p95_body": worst_p95,
        "p95": rows[worst_p95]["p95"] if worst_p95 else None,
        "max": rows[worst]["max"] if worst else None,
    }


def _contact(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "normal_gap": _body_family(run, "surface_normal_error_m"),
        "slip": _body_family(
            run, "surface_sticking_target_relative_slip_m_s"
        ),
    }


def _joint_stats(run: dict[str, Any], field: str) -> tuple[Any, dict[str, Any]]:
    record = data(run, "joint_smoothness.json") or {}
    joints = record.get("joints", {})
    joints = joints if isinstance(joints, dict) else {}
    total = 0
    worst: dict[str, Any] = {}
    worst_key = None
    for name, row in joints.items():
        if not isinstance(row, dict):
            continue
        item = row.get(field, {})
        if not isinstance(item, dict):
            continue
        count = _number(item.get("violations"))
        total += int(count or 0)
        magnitude = _number(item.get("max_excess")) or _number(item.get("peak")) or 0.0
        if magnitude > (_number(worst.get("magnitude")) or -math.inf):
            worst = dict(item)
            worst["magnitude"] = magnitude
            worst_key = name
    return total, {"joint": worst_key, **worst}


def _smoothness(run: dict[str, Any]) -> dict[str, Any]:
    record = data(run, "joint_smoothness.json") or {}
    chatter, chatter_worst = _joint_stats(run, "source_relative_chatter")
    _, velocity_worst = _joint_stats(run, "mechanical_limits")
    # mechanical_limits is nested; make the summary explicit below instead.
    joints = record.get("joints", {})
    effort_worst: dict[str, Any] = {}
    velocity_explicit: dict[str, Any] = {}
    if isinstance(joints, dict):
        for name, row in joints.items():
            if not isinstance(row, dict):
                continue
            limits = row.get("mechanical_limits", {})
            if not isinstance(limits, dict):
                continue
            for field, target in (
                ("velocity", velocity_explicit),
                ("effort", effort_worst),
            ):
                item = limits.get(field, {})
                if not isinstance(item, dict):
                    continue
                magnitude = _number(item.get("peak")) or _number(
                    item.get("peak_per_second")
                )
                if magnitude is None:
                    continue
                if magnitude > (_number(target.get("peak")) or -math.inf):
                    target.clear()
                    target.update(item)
                    target["joint"] = name
                    target["peak"] = magnitude
    root = record.get("root_source_relative_chatter", {})
    root = root if isinstance(root, dict) else {}
    return {
        "chatter": chatter,
        "chatter_worst": chatter_worst,
        "velocity": record.get("velocity_limit_violations"),
        "velocity_worst": velocity_explicit,
        "effort": record.get("effort_limit_violations"),
        "effort_worst": effort_worst,
        "effort_unknown_frames": _nested(record, "effort_dynamics", "unknown_frame_count"),
        "effort_infeasible_frames": len(_nested(record, "effort_dynamics", "infeasible_frames"))
                                    if _nested(record, "effort_dynamics", "infeasible_frames") is not None else None,
        "effort_status_counts": _nested(record, "effort_dynamics", "frame_status_counts"),
        "effort_audit_error_frames": len(_nested(record, "effort_dynamics", "audit_error_frames"))
                                     if _nested(record, "effort_dynamics", "audit_error_frames") is not None else None,
        "effort_unaudited_endpoints": _nested(record, "effort_dynamics", "unaudited_endpoint_frames"),
        "root_translation_chatter": _nested(root, "translation", "violations"),
        "root_rotation_chatter": _nested(root, "rotation", "violations"),
    }


def _pose(run: dict[str, Any]) -> dict[str, Any]:
    record = data(run, "source_pose.json") or {}
    parts = record.get("parts", {})
    parts = parts if isinstance(parts, dict) else {}
    worst = max(
        parts,
        key=lambda name: _number(parts[name].get("max_rad")) if isinstance(parts[name], dict) else -math.inf,
        default=None,
    )
    return {
        "mean_rad": _number(record.get("mean_rad")),
        "p95_rad": _number(record.get("p95_rad")),
        "worst_part": worst,
        "worst_max_rad": _number(parts[worst].get("max_rad")) if worst else None,
        "worst_frame": _nested(parts, worst, "worst_frame") if worst else None,
    }


def _budgets(run: dict[str, Any]) -> dict[str, Any]:
    record = data(run, "fast_report.json") or {}
    values = _nested(record, "stages", "contact_scene", "budgets", default={})
    values = values if isinstance(values, dict) else {}
    selected = {
        key: value for key, value in values.items()
        if key.startswith("relation_") or key.startswith("layout_flip_")
    }
    return {
        "values": selected,
        "skipped": _nested(record, "stages", "contact_scene", "skipped_tasks", default=[]),
    }


def _compact(value: Any, depth: int = 0) -> str:
    if isinstance(value, dict):
        if depth >= 2:
            return f"dict[{len(value)}]"
        return "; ".join(f"{key}={_compact(item, depth + 1)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        return f"list[{len(value)}]: " + _compact(value[0], depth + 1)
    return _fmt(value)


def _find_key(value: Any, key: str, path: str = "") -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for child, item in value.items():
            location = f"{path}.{child}" if path else str(child)
            if child == key:
                found.append((location, item))
            found.extend(_find_key(item, key, location))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_find_key(item, key, f"{path}[{index}]"))
    return found


def _optional_fast_sections(run: dict[str, Any]) -> dict[str, list[tuple[str, Any]]]:
    record = data(run, "fast_report.json") or {}
    return {
        name: _find_key(record, name)
        for name in ("quality", "defects", "deadline", "solver_trace")
    }


def _headline(run: dict[str, Any]) -> dict[str, Any]:
    penetration = _penetration(run)
    depths = penetration.get("depths", {})
    smooth = _smoothness(run)
    contact = _contact(run)
    return {
        "per_frame_s": _timing(run)["per_frame"],
        "defect_pairs": penetration.get("defect_count"),
        "max_depth_m": max(
            (_number(value) for value in depths.values()), default=None
        ),
        "chatter": smooth["chatter"],
        "velocity": smooth["velocity"],
        "relation_p95_rad": _pose(run)["p95_rad"],
        "support_gap_p95_m": contact["normal_gap"]["p95"],
    }


def discover_clips(before: Path, after: Path, requested: str | None) -> list[str]:
    if requested is not None:
        return [name.strip() for name in requested.split(",") if name.strip()]
    before_clips = {p.name for p in before.iterdir() if p.is_dir()} if before.is_dir() else set()
    after_clips = {p.name for p in after.iterdir() if p.is_dir()} if after.is_dir() else set()
    return sorted(before_clips & after_clips)


def _add_comparison(
    comparisons: list[dict[str, Any]], clip: str, metric: str, before: Any,
    after: Any, unit: str, higher_is_worse: bool, detail: str = "",
) -> None:
    a, b = _number(before), _number(after)
    if a is None or b is None:
        return
    comparisons.append({
        "clip": clip,
        "metric": metric,
        "before": a,
        "after": b,
        "delta": b - a,
        "unit": unit,
        "higher_is_worse": higher_is_worse,
        "detail": detail,
    })


def compare(before_root: Path, after_root: Path, clips: list[str]) -> dict[str, Any]:
    runs: dict[str, dict[str, dict[str, Any]]] = {}
    headlines: dict[str, dict[str, Any]] = {}
    comparisons: list[dict[str, Any]] = []
    for clip in clips:
        runs[clip] = {
            "before": load_clip(before_root, clip),
            "after": load_clip(after_root, clip),
        }
        headlines[clip] = {
            side: _headline(runs[clip][side]) for side in ("before", "after")
        }
        before_run, after_run = runs[clip]["before"], runs[clip]["after"]
        bt, at = _timing(before_run), _timing(after_run)
        bp, ap = _penetration(before_run), _penetration(after_run)
        bc, ac = _contact(before_run), _contact(after_run)
        bs, as_ = _smoothness(before_run), _smoothness(after_run)
        bpose, apose = _pose(before_run), _pose(after_run)
        for metric, a, b, unit, worse in (
            ("秒/帧", bt["per_frame"], at["per_frame"], "s", True),
            ("v1缺陷对", bp.get("defect_count"), ap.get("defect_count"), "count", True),
            ("static最大穿透", _number(bp.get("depths", {}).get("static")), _number(ap.get("depths", {}).get("static")), "m", True),
            ("moving最大穿透", _number(bp.get("depths", {}).get("moving")), _number(ap.get("depths", {}).get("moving")), "m", True),
            ("self最大穿透", _number(bp.get("depths", {}).get("self")), _number(ap.get("depths", {}).get("self")), "m", True),
            (">5mm帧并集", bp.get("over_5mm"), ap.get("over_5mm"), "frames", True),
            (">1cm帧并集", bp.get("over_1cm"), ap.get("over_1cm"), "frames", True),
            ("unknown查询", bp.get("unknown"), ap.get("unknown"), "count", True),
            ("reset可用帧", bp.get("reset", {}).get("usable"), ap.get("reset", {}).get("usable"), "frames", False),
            ("source chatter违例", bs["chatter"], as_["chatter"], "count", True),
            ("速度超限", bs["velocity"], as_["velocity"], "count", True),
            ("力矩超限", bs["effort"], as_["effort"], "count", True),
            ("relation p95", bpose["p95_rad"], apose["p95_rad"], "rad", True),
            ("支撑normal-gap p95", bc["normal_gap"]["p95"], ac["normal_gap"]["p95"], "m", True),
            ("支撑slip p95", bc["slip"]["p95"], ac["slip"]["p95"], "m/s", True),
        ):
            _add_comparison(comparisons, clip, metric, a, b, unit, worse)
    worse = []
    for item in comparisons:
        floor = FLOOR.get(item["unit"], 1e-9)
        threshold = max(floor, 0.02 * abs(item["before"]))
        delta = item["delta"]
        degraded = delta > 0 if item["higher_is_worse"] else delta < 0
        if degraded and abs(delta) > threshold:
            worse.append({**item, "threshold": threshold})
    worse.sort(key=lambda item: (-abs(item["delta"]), item["clip"], item["metric"]))
    return {
        "before_root": str(before_root),
        "after_root": str(after_root),
        "clips": clips,
        "runs": runs,
        "headlines": headlines,
        "comparisons": comparisons,
        "worse": worse,
    }


def _file_status(run: dict[str, Any]) -> str:
    return ", ".join(sorted(
        name + ("(parse error)" if item["error"] else "")
        for name, item in run["files"].items()
        if not item["present"] or item["error"]
    ))


def _render_contact_family(
    lines: list[str], label: str, before: dict[str, Any], after: dict[str, Any], unit: str
) -> None:
    lines.append(f"#### {label}（{unit}）")
    lines.append("| body | before p95 | after p95 | Δp95 | before max | after max | Δmax |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    bodies = sorted(set(before["bodies"]) | set(after["bodies"]))
    for body in bodies:
        b, a = before["bodies"].get(body, {}), after["bodies"].get(body, {})
        lines.append(
            f"| {body} | {_fmt(b.get('p95'))} | {_fmt(a.get('p95'))} | "
            f"{_delta(b.get('p95'), a.get('p95'), unit)} | {_fmt(b.get('max'))} | "
            f"{_fmt(a.get('max'))} | {_delta(b.get('max'), a.get('max'), unit)} |"
        )
    lines.append(
        f"- 最差 p95 body：{_safe_name(before['worst_p95_body'])} → "
        f"{_safe_name(after['worst_p95_body'])}；最差 max body："
        f"{_safe_name(before['worst_max_body'])} → {_safe_name(after['worst_max_body'])}。"
    )
    lines.append("- 最差帧：contact_metrics.json 未保存逐 body 最差帧，未确认。\n")


def _render_optional(
    lines: list[str], before: dict[str, Any], after: dict[str, Any]
) -> None:
    before_optional = _optional_fast_sections(before)
    after_optional = _optional_fast_sections(after)
    if not any(before_optional.values()) and not any(after_optional.values()):
        return
    lines.append("### 求解器新增摘要字段")
    for name in before_optional:
        for side, optional in (("before", before_optional), ("after", after_optional)):
            if optional[name]:
                lines.append(f"- {name} [{side}]：")
                for location, value in optional[name][:8]:
                    lines.append(f"  - `{location}`：{_compact(value)}")
                if len(optional[name]) > 8:
                    lines.append(f"  - …另有 {len(optional[name]) - 8} 处")
    lines.append("")


def render(result: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Benchmark arm 对比报告",
        "",
        f"- before：`{result['before_root']}`",
        f"- after：`{result['after_root']}`",
        f"- clips：{', '.join(result['clips']) if result['clips'] else '缺失'}",
        "- 变差判定：`|Δ| > max(单位 floor, 2%×|before|)`；floor 采用 "
        "count/frame=0.5、m=5e-5、m/s=1e-3、rad=1e-3、s=0.05。",
        "",
    ]
    for clip in result["clips"]:
        before, after = result["runs"][clip]["before"], result["runs"][clip]["after"]
        lines.extend([f"## {clip}", ""])
        lines.append(f"- 缺失文件 before：{_file_status(before) or '无'}")
        lines.append(f"- 缺失文件 after：{_file_status(after) or '无'}")

        bt, at = _timing(before), _timing(after)
        lines.extend(["### 耗时", "", "| 指标 | before | after | Δ |", "|---|---:|---:|---:|"])
        lines.append(_row("production_process_wall_seconds", bt["production"], at["production"], "s"))
        lines.append(_row("fast_backend_seconds", bt["fast"], at["fast"], "s"))
        lines.append(_row("帧数", bt["frames"], at["frames"], "frames"))
        lines.append(_row("秒/帧", bt["per_frame"], at["per_frame"], "s"))
        lines.extend(["", "| stage | before s | after s |", "|---|---:|---:|"])
        for stage in sorted(set(bt["stages"]) | set(at["stages"])):
            lines.append(f"| {stage} | {_fmt(bt['stages'].get(stage))} | {_fmt(at['stages'].get(stage))} |")

        bp, ap = _penetration(before), _penetration(after)
        lines.extend(["", "### 穿透 persistence / policy_v1"])
        if bp["missing"] or ap["missing"]:
            lines.append(
                f"- penetration_persistence.json：before={'缺失' if bp['missing'] else '存在'}；"
                f"after={'缺失' if ap['missing'] else '存在'}。"
            )
        lines.extend(["", "| 指标 | before | after | Δ |", "|---|---:|---:|---:|"])
        lines.append(_row("defect pairs", bp.get("defect_count"), ap.get("defect_count")))
        for label in ("deep", "persistent", "support_residual", "mild_brief", "negligible"):
            lines.append(_row(label, bp.get("counts", {}).get(label), ap.get("counts", {}).get(label)))
        for kind in ("static", "moving", "self"):
            lines.append(_row(f"{kind} max depth", bp.get("depths", {}).get(kind), ap.get("depths", {}).get(kind), "m"))
        lines.append(_row(">5mm帧并集", bp.get("over_5mm"), ap.get("over_5mm"), "frames"))
        lines.append(_row(">1cm帧并集", bp.get("over_1cm"), ap.get("over_1cm"), "frames"))
        lines.append(_row("unknown", bp.get("unknown"), ap.get("unknown")))
        lines.append(_row("reset usable", bp.get("reset", {}).get("usable"), ap.get("reset", {}).get("usable"), "frames"))
        lines.append(_row("reset unusable", bp.get("reset", {}).get("unusable"), ap.get("reset", {}).get("unusable"), "frames"))

        pair_header = "| pair | kind | Dmax(m) | L5(frames) | F5(frames) | L2(frames) | worst_frame | labels |"
        lines.extend(["", "after 全部 defect pair：", "", pair_header, "|---|---|---:|---:|---:|---:|---:|---|"])
        for row in ap.get("defects", []):
            lines.append(
                f"| {row['body']}→{row['other']} | {row['kind']} | {_fmt(row['Dmax_m'])} | "
                f"{_fmt(row['L5_frames'])} | {_fmt(row['F5_frames'])} | {_fmt(row['L2_frames'])} | "
                f"{_fmt(row['worst_frame'])} | {','.join(map(str, row['labels']))} |"
            )
        lines.extend(["", "before 最深 top-3 defect pair：", "", pair_header, "|---|---|---:|---:|---:|---:|---:|---|"])
        for row in bp.get("defects", [])[:3]:
            lines.append(
                f"| {row['body']}→{row['other']} | {row['kind']} | {_fmt(row['Dmax_m'])} | "
                f"{_fmt(row['L5_frames'])} | {_fmt(row['F5_frames'])} | {_fmt(row['L2_frames'])} | "
                f"{_fmt(row['worst_frame'])} | {','.join(map(str, row['labels']))} |"
            )

        bc, ac = _contact(before), _contact(after)
        lines.extend(["", "### 接触"])
        _render_contact_family(lines, "支撑 normal gap", bc["normal_gap"], ac["normal_gap"], "m")
        _render_contact_family(lines, "sticking target-relative slip", bc["slip"], ac["slip"], "m/s")
        lines.append("- 接触丢失计数：当前 contact_metrics.json 未提供该字段。\n")

        bs, as_ = _smoothness(before), _smoothness(after)
        lines.extend(["### 抖动与可执行性", "", "| 指标 | before | after | Δ |", "|---|---:|---:|---:|"])
        lines.append(_row("source-relative chatter违例", bs["chatter"], as_["chatter"]))
        lines.append(_row("速度超限", bs["velocity"], as_["velocity"]))
        lines.append(_row("力矩超限（effort screen）", bs["effort"], as_["effort"]))
        lines.append(_row("力矩未判定帧", bs["effort_unknown_frames"], as_["effort_unknown_frames"]))
        lines.append(_row("支撑确定不可行帧", bs["effort_infeasible_frames"], as_["effort_infeasible_frames"]))
        lines.append(_row("力矩审计数值错误帧", bs["effort_audit_error_frames"], as_["effort_audit_error_frames"]))
        for side, smooth in (("before", bs), ("after", as_)):
            if smooth["effort_status_counts"] is not None:
                lines.append(f"- {side} 力矩/支撑帧状态：{smooth['effort_status_counts']}。")
            if smooth["effort_unaudited_endpoints"] is not None:
                lines.append(f"- {side} 力矩未审计端点帧：{smooth['effort_unaudited_endpoints']}；不能据超限计数为零判定通过。")
        lines.append(_row("root世界HF translation", bs["root_translation_chatter"], as_["root_translation_chatter"]))
        lines.append(_row("root世界HF rotation", bs["root_rotation_chatter"], as_["root_rotation_chatter"]))
        for side, smooth in (("before", bs), ("after", as_)):
            chatter_worst = smooth["chatter_worst"]
            effort_worst = smooth["effort_worst"]
            lines.append(
                f"- {side} chatter 最差关节：{_safe_name(chatter_worst.get('joint'))}，"
                f"max_excess={_fmt(chatter_worst.get('max_excess'))}，"
                f"frame={_fmt(chatter_worst.get('worst_frame'))}；effort 最差关节："
                f"{_safe_name(effort_worst.get('joint'))}，peak={_fmt(effort_worst.get('peak'))}，"
                f"frame={_fmt(effort_worst.get('worst_frame'))}。"
            )

        bpose, apose = _pose(before), _pose(after)
        lines.extend(["", "### 姿态", "", "| 指标 | before | after | Δ |", "|---|---:|---:|---:|"])
        lines.append(_row("relation mean", bpose["mean_rad"], apose["mean_rad"], "rad"))
        lines.append(_row("relation p95", bpose["p95_rad"], apose["p95_rad"], "rad"))
        lines.append(_row("relation worst max", bpose["worst_max_rad"], apose["worst_max_rad"], "rad"))
        lines.append(_row("layout flip帧", _nested(data(before, 'fast_report.json'), 'stages', 'contact_scene', 'budgets', 'layout_flip_frames'), _nested(data(after, 'fast_report.json'), 'stages', 'contact_scene', 'budgets', 'layout_flip_frames'), "frames"))
        lines.append(
            f"- 最差 landmark：{_safe_name(bpose['worst_part'])} → {_safe_name(apose['worst_part'])}；"
            f"最差帧：{_fmt(bpose['worst_frame'])} → {_fmt(apose['worst_frame'])}。"
        )

        lines.extend(["", "### fast budgets / skipped tasks"])
        for side, run in (("before", before), ("after", after)):
            budgets = _budgets(run)
            values = "; ".join(f"{key}={_compact(value)}" for key, value in budgets["values"].items())
            lines.append(f"- {side} budgets：{values or '缺失'}")
            lines.append(f"- {side} skipped_tasks：{_compact(budgets['skipped'])}")
        lines.append("")
        _render_optional(lines, before, after)

    lines.extend([
        "## 全 clip 汇总",
        "",
        "| clip | 秒/帧 B→A | v1缺陷 B→A | 最大穿透 B→A(m) | chatter B→A | 速度 B→A | relation p95 B→A(rad) | 支撑gap p95 B→A(m) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for clip, sides in result["headlines"].items():
        b, a = sides["before"], sides["after"]
        def pair(key, digits=6):
            return f"{_fmt(b[key], digits)}→{_fmt(a[key], digits)}"
        lines.append(
            f"| {clip} | {pair('per_frame_s')} | {pair('defect_pairs')} | "
            f"{pair('max_depth_m')} | {pair('chatter')} | {pair('velocity')} | "
            f"{pair('relation_p95_rad')} | {pair('support_gap_p95_m')} |"
        )
    lines.extend(["", "## 变差指标（超过 floor 且超过 before 的 2%）", ""])
    if not result["worse"]:
        lines.append("无。")
    else:
        lines.append("| clip | 指标 | before | after | Δ | 单位 | 判定阈值 |")
        lines.append("|---|---|---:|---:|---:|---|---:|")
        for item in result["worse"]:
            lines.append(
                f"| {item['clip']} | {item['metric']} | {_fmt(item['before'])} | "
                f"{_fmt(item['after'])} | {_fmt(item['delta'])} | {item['unit']} | "
                f"{_fmt(item['threshold'])} |"
            )
    return "\n".join(lines) + "\n"


def _json_result(result: dict[str, Any]) -> dict[str, Any]:
    details = {
        clip: {
            side: {
                "timing": _timing(run),
                "penetration": _penetration(run),
                "contact": _contact(run),
                "smoothness": _smoothness(run),
                "pose": _pose(run),
                "budgets": _budgets(run),
                "optional_fast_sections": _optional_fast_sections(run),
            }
            for side, run in sides.items()
        }
        for clip, sides in result["runs"].items()
    }
    return {
        "before_root": result["before_root"],
        "after_root": result["after_root"],
        "clips": result["clips"],
        "headlines": result["headlines"],
        "details": details,
        "comparisons": result["comparisons"],
        "worse": result["worse"],
        "missing_files": {
            clip: {
                side: _file_status(result["runs"][clip][side])
                for side in ("before", "after")
            }
            for clip in result["clips"]
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--clips", default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    clips = discover_clips(args.before, args.after, args.clips)
    result = compare(args.before.resolve(), args.after.resolve(), clips)
    report = render(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(_json_result(result), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(f"SAVED {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
