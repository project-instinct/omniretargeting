"""Export frozen per-part spline-quality audits to review tables.

The exporter is read-only: it never reruns geometry, kinematics, contact, or
smoothness. Missing or invalid reports remain in coverage/defects tables; no
exit status or aggregate improvement is interpreted as a pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_structural_spline import CLIPS



CSV_FIELDS = [
    "run_id", "arm", "mode", "repeat", "clip", "audit_kind", "section",
    "metric", "part", "unit", "unit_conversion", "count", "mean", "p95", "max",
    "A_mean", "A_p95", "A_max", "delta_mean", "delta_p95", "delta_max",
    "reference_mean", "reference_p95", "reference_max",
    "worst_frame_clip_local", "original_source_frame", "source_frame_start",
    "abnormal_interval_count", "abnormal_frames", "abnormal_intervals_json",
    "worse_than_A_frames", "worse_than_A_intervals_json", "summary_increased",
    "excess_tolerance", "violating_frames", "violations", "excess_definition",
    "budget_provenance", "correction_definition", "report_units",
    "binding_id", "binding_kind", "binding_index", "contact_role", "active_frames",
    "active_violating_frames", "source_status", "source_provenance",
    "source_path",
]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _json_text(value: Any) -> str:
    if value in (None, []):
        return ""
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _interval_frames(intervals: Any) -> int:
    if not isinstance(intervals, list):
        return 0
    return sum(int(item.get("frames", 0)) for item in intervals if isinstance(item, dict))


def _native_unit(metric: str, fallback: str = "report-native") -> str:
    lowered = metric.lower()
    if lowered.endswith("_m") or "_meters" in lowered:
        return "m"
    if lowered.endswith("_rad") or "_radians" in lowered:
        return "rad"
    if lowered.endswith("_deg") or "_degrees" in lowered:
        return "deg"
    if lowered.endswith("_s") or "_seconds" in lowered:
        return "s"
    return fallback


def _row(
    identity: dict[str, Any], audit_kind: str, section: str, metric: str,
    part: str, values: dict[str, Any], *, unit: str | None = None,
) -> dict[str, Any]:
    stats = values.get("final", values.get("after", values))
    baseline = values.get("A", values.get("baseline", {}))
    delta = values.get("delta", {})
    reference = values.get("reference", {})
    if not isinstance(stats, dict):
        stats = {}
    if not isinstance(baseline, dict):
        baseline = {}
    if not isinstance(delta, dict):
        delta = {}
    if not isinstance(reference, dict):
        reference = {}
    for target, source in (
        ("mean", "mean_m"), ("p95", "p95_m"), ("max", "max_m"),
        ("worst_frame", "worst_frame_m"),
    ):
        stats.setdefault(target, stats.get(source))
        baseline.setdefault(target, baseline.get(source))
        delta.setdefault(target, delta.get(source))
    worst = stats.get("worst_frame", stats.get("worst_frame_clip_local"))
    start = identity.get("source_frame_start", 0)
    intervals = stats.get("abnormal_intervals", stats.get("persistent_intervals"))
    return {
        **{key: identity.get(key, "") for key in (
            "run_id", "arm", "mode", "repeat", "clip", "source_frame_start",
        )},
        "audit_kind": audit_kind,
        "section": section,
        "metric": metric,
        "part": part,
        "unit": unit or _native_unit(metric),
        "unit_conversion": "none",
        "count": stats.get("count", values.get("count", "")),
        "mean": _number(stats.get("mean", values.get("mean"))),
        "p95": _number(stats.get("p95", values.get("p95"))),
        "max": _number(stats.get("max", values.get("max", values.get("max_excess")))),
        "A_mean": _number(baseline.get("mean", values.get("A_mean"))),
        "A_p95": _number(baseline.get("p95", values.get("A_p95"))),
        "A_max": _number(baseline.get("max", values.get("A_max"))),
        "delta_mean": _number(delta.get("mean", values.get("delta_mean"))),
        "delta_p95": _number(delta.get("p95", values.get("delta_p95"))),
        "delta_max": _number(delta.get("max", values.get("delta_max"))),
        "reference_mean": _number(reference.get("mean", values.get("reference_mean"))),
        "reference_p95": _number(reference.get("p95", values.get("reference_p95"))),
        "reference_max": _number(reference.get("max", values.get("reference_max"))),
        "worst_frame_clip_local": worst,
        "original_source_frame": (
            int(worst) + int(start)
            if worst is not None and start is not None and str(worst).lstrip("-").isdigit()
            else ""
        ),
        "abnormal_interval_count": len(intervals) if isinstance(intervals, list) else values.get("abnormal_interval_count", ""),
        "abnormal_frames": _interval_frames(intervals),
        "abnormal_intervals_json": _json_text(intervals),
        "worse_than_A_frames": values.get("worse_than_A_frames", ""),
        "worse_than_A_intervals_json": _json_text(values.get("worse_than_A_intervals")),
        "summary_increased": values.get("summary_increased", ""),
        "excess_tolerance": values.get("excess_tolerance", ""),
        "violating_frames": values.get("violating_frames", ""),
        "violations": values.get("violations", ""),
        "excess_definition": "abs(high_frequency(candidate-reference)) - chatter_budget(reference)",
        "budget_provenance": values.get("budget_provenance", ""),
        "correction_definition": values.get("correction_definition", ""),
        "report_units": values.get("units", unit or ""),
        "source_path": values.get("source_path", identity.get("source_path", "")),
    }


def _comparison_rows(identity: dict[str, Any], report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comparisons = report.get("comparisons", {})
    if not isinstance(comparisons, dict):
        return rows
    for section, comparison in comparisons.items():
        metrics = comparison.get("metrics", {}) if isinstance(comparison, dict) else {}
        for metric, parts in (metrics if isinstance(metrics, dict) else {}).items():
            for part, stats in (parts if isinstance(parts, dict) else {}).items():
                rows.append(_row(identity, "spline_quality", section, metric, part, {
                    **stats, "source_path": identity["quality_path"],
                }))
    return rows


def _non_degradation_rows(identity: dict[str, Any], report: dict[str, Any]) -> list[dict[str, Any]]:
    block = report.get("non_degradation_against_A", {})
    rows: list[dict[str, Any]] = []
    metrics = block.get("metrics", {}) if isinstance(block, dict) else {}
    for metric, parts in (metrics if isinstance(metrics, dict) else {}).items():
        for part, values in (parts if isinstance(parts, dict) else {}).items():
            rows.append(_row(identity, "spline_quality", "non_degradation_against_A", metric, part, {
                **values, "source_path": identity["quality_path"],
            }))
    return rows


def _contact_rows(identity: dict[str, Any], report: dict[str, Any]) -> list[dict[str, Any]]:
    block = report.get("contact_and_root_per_part", report.get("contact_and_root_drift", {}))
    rows: list[dict[str, Any]] = []
    if not isinstance(block, dict):
        return rows
    compared = block.get("final_against_A", block)
    metrics = compared.get("metrics", {}) if isinstance(compared, dict) else {}
    for metric, parts in (metrics if isinstance(metrics, dict) else {}).items():
        for part, values in (parts if isinstance(parts, dict) else {}).items():
            rows.append(_row(identity, "contact", "contact_and_root", metric, part, {
                **values, "source_path": identity["quality_path"],
            }))
    roles = block.get("roles", {})
    if isinstance(roles, dict):
        for role, role_report in roles.items():
            role_metrics = role_report.get("metrics", {}) if isinstance(role_report, dict) else {}
            for metric, parts in (role_metrics if isinstance(role_metrics, dict) else {}).items():
                for part, stats in (parts if isinstance(parts, dict) else {}).items():
                    rows.append(_row(identity, "contact", f"role:{role}", metric, part, {
                        **stats, "source_path": identity["quality_path"],
                    }))
    return rows


def _binding_context(
    values: dict[str, Any], *, binding_id: str = "", role: str = "",
    source_status: str = "", source_provenance: str = "",
) -> dict[str, Any]:
    return {
        "binding_id": binding_id,
        "binding_kind": values.get("kind", ""),
        "binding_index": values.get("binding_index", ""),
        "contact_role": role,
        "active_frames": values.get("active_frames", ""),
        "active_violating_frames": values.get("active_violating_frames", ""),
        "source_status": source_status,
        "source_provenance": source_provenance,
    }


def _contact_chatter_rows(identity: dict[str, Any], report: dict[str, Any]) -> list[dict[str, Any]]:
    block = report.get("contact_and_root_per_part", {})
    chatter = block.get("chatter", {}) if isinstance(block, dict) else {}
    rows: list[dict[str, Any]] = []
    if not isinstance(chatter, dict):
        return rows
    for role, bindings in (chatter.get("roles", {}) or {}).items():
        if not isinstance(bindings, dict):
            continue
        for binding_id, values in bindings.items():
            if not isinstance(values, dict):
                continue
            context = _binding_context(values, binding_id=str(binding_id), role=str(role))
            whole = values.get("whole_trajectory", {})
            if isinstance(whole, dict):
                rows.extend(_chatter_signal_rows(
                    identity, "contact_chatter", f"role:{role}:whole_trajectory",
                    str(binding_id), whole, str(values.get("unit", "m")), context=context,
                ))
            active = values.get("active_contact", {})
            if isinstance(active, dict):
                rows.append(_row(
                    identity, "contact_chatter", f"role:{role}:active_contact",
                    "active_contact_high_frequency_m", str(binding_id), {
                        **active, "source_path": identity["quality_path"],
                    }, unit=str(values.get("unit", "m")),
                ) | context)
                rows.append(_row(
                    identity, "contact_chatter", f"role:{role}:active_contact",
                    "active_contact_chatter_excess_m", str(binding_id), {
                        **active, "mean": None, "p95": None, "max": None,
                        "count": values.get("active_violating_frames", ""),
                        "source_path": identity["quality_path"],
                    }, unit=str(values.get("unit", "m")),
                ) | context)
            coordinates = values.get("active_per_coordinate", [])
            for axis, stats in enumerate(coordinates if isinstance(coordinates, list) else []):
                if not isinstance(stats, dict):
                    continue
                axis_part = f"{binding_id}/axis_{axis}"
                rows.append(_row(
                    identity, "contact_chatter", f"role:{role}:active_contact",
                    "active_contact_high_frequency_m", axis_part, {
                        **stats, "source_path": identity["quality_path"],
                    }, unit=str(values.get("unit", "m")),
                ) | context)
                rows.append(_row(
                    identity, "contact_chatter", f"role:{role}:active_contact",
                    "active_contact_chatter_excess_m", axis_part, {
                        **stats, "mean": None, "p95": None, "max": None,
                        "count": values.get("active_violating_frames", ""),
                        "source_path": identity["quality_path"],
                    }, unit=str(values.get("unit", "m")),
                ) | context)
    final = chatter.get("final_against_A", {})
    metrics = final.get("metrics", {}) if isinstance(final, dict) else {}
    for metric, parts in (metrics if isinstance(metrics, dict) else {}).items():
        for part, values in (parts if isinstance(parts, dict) else {}).items():
            binding_id, _, axis = str(part).partition("/axis_")
            current = values.get("final", {})
            context = _binding_context(
                current if isinstance(current, dict) else {},
                binding_id=binding_id,
                role="final_minus_A",
            )
            if axis:
                context["binding_index"] = axis
            rows.append(_row(
                identity, "contact_chatter", "final_minus_A", metric, str(part), {
                    **values, "source_path": identity["quality_path"],
                }, unit="m",
            ) | context)
    for role, bindings in (chatter.get("original_source_audits", {}) or {}).items():
        if not isinstance(bindings, dict):
            continue
        for binding_id, values in bindings.items():
            if not isinstance(values, dict):
                continue
            context = _binding_context(
                values, binding_id=str(binding_id), role=str(role),
                source_status=str(values.get("status", "")),
                source_provenance=str(values.get("provenance", values.get("reason", ""))),
            )
            if values.get("status") != "measured":
                continue
            rows.extend(_chatter_signal_rows(
                identity, "contact_source_chatter", f"role:{role}", str(binding_id),
                values.get("report", {}), "m", context=context,
            ))
    return rows


def _contact_chatter_defects(identity: dict[str, Any], quality: dict[str, Any]) -> list[dict[str, str]]:
    block = quality.get("contact_and_root_per_part", {})
    chatter = block.get("chatter", {}) if isinstance(block, dict) else None
    if not isinstance(chatter, dict) or chatter.get("status") != "measured":
        return [_defect(
            identity, "contact_chatter", "unknown", "", "contact_and_root_per_part.chatter",
            "contact chatter report missing, invalid, or not measured; no values are fabricated",
        )]
    defects: list[dict[str, str]] = []
    source_audits = chatter.get("original_source_audits", {})
    for role, bindings in (source_audits if isinstance(source_audits, dict) else {}).items():
        for binding_id, values in (bindings if isinstance(bindings, dict) else {}).items():
            if isinstance(values, dict) and values.get("status") != "measured":
                defects.append(_defect(
                    identity, "contact_source_chatter", "unknown", str(binding_id),
                    f"role:{role}", str(values.get("reason", "source correspondence unverified")),
                ))
    if not isinstance(chatter.get("final_against_A"), dict):
        defects.append(_defect(
            identity, "contact_chatter", "unknown", "", "final_against_A",
            "contact-chatter A comparison missing; no pass inferred",
        ))
    return defects


def _part_items(value: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        if "body" in value and any(key in value for key in ("mean_m", "max_m", "mean")):
            return [str(value.get("body", "body")), value]
        return [(str(part), child) for part, child in value.items() if isinstance(child, dict)]
    if isinstance(value, list):
        return [(str(item.get("body", item.get("part", "body"))), item) for item in value if isinstance(item, dict)]
    return []


def _geometry_rows(identity: dict[str, Any], quality: dict[str, Any], training: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    block = quality.get("original_geometry_per_body", {})
    categories = block.get("categories", block) if isinstance(block, dict) else {}
    for category, category_report in (categories if isinstance(categories, dict) else {}).items():
        per_body = category_report.get("per_body", {}) if isinstance(category_report, dict) else {}
        for part, values in _part_items(per_body):
            rows.append(_row(
                identity, "original_geometry", f"geometry:{category}",
                f"{category}_depth", part, {**values, "source_path": identity["training_path"]},
                unit="m",
            ))
    if rows:
        return rows
    for section, audit_kind in (("static_scene", "static"), ("moving_objects", "moving")):
        report = training.get(section, {})
        if isinstance(report, dict):
            rows.append(_row(
                identity, "original_geometry", f"geometry:{audit_kind}_aggregate",
                f"{audit_kind}_depth", "<aggregate>", {
                    **report, "source_path": identity["training_path"],
                }, unit="m",
            ))
    return rows


def _self_rows(identity: dict[str, Any], training: dict[str, Any]) -> list[dict[str, Any]]:
    report = training.get("self_collision", {})
    rows: list[dict[str, Any]] = []
    if not isinstance(report, dict):
        return rows
    for part, values in _part_items(report.get("per_body", {})):
        rows.append(_row(
            identity, "self_collision", "per_body", "self_collision_depth",
            part, {**values, "source_path": identity["training_path"]}, unit="m",
        ))
    for part, values in _part_items(report.get("per_pair", {})):
        rows.append(_row(
            identity, "self_collision", "per_pair", "self_collision_depth",
            part, {**values, "source_path": identity["training_path"]}, unit="m",
        ))
    rows.append(_row(
        identity, "self_collision", "aggregate", "self_collision_depth", "<aggregate>", {
            **report, "source_path": identity["training_path"],
        }, unit="m",
    ))
    return rows


def _quality_self_rows(identity: dict[str, Any], quality: dict[str, Any]) -> list[dict[str, Any]]:
    report = quality.get("original_self_collision_per_body")
    rows: list[dict[str, Any]] = []
    if not isinstance(report, dict) or report.get("status") != "measured":
        return rows
    for group in ("per_body", "per_pair"):
        block = report.get(group)
        if block is None or not isinstance(block, (dict, list)):
            continue
        for part, values in _part_items(block):
            rows.append(_row(
                identity, "original_self_collision", group, "self_collision_depth",
                part, {**values, "source_path": identity["quality_path"]}, unit="m",
            ))
    return rows


def _quality_self_defects(identity: dict[str, Any], quality: dict[str, Any]) -> list[dict[str, str]]:
    report = quality.get("original_self_collision_per_body")
    if not isinstance(report, dict):
        return [_defect(
            identity, "quality_self_collision", "unknown", "", "original_self_collision_per_body",
            "quality report field missing or invalid; A/delta comparison unavailable",
        )]
    if report.get("status") != "measured":
        return [_defect(
            identity, "quality_self_collision", "unknown", "", str(report.get("status", "unknown")),
            str(report.get("reason", "quality self-collision comparison is not measured")),
        )]
    defects: list[dict[str, str]] = []
    for group in ("per_body", "per_pair"):
        if group not in report:
            defects.append(_defect(
                identity, "quality_self_collision", "unknown", "", group,
                f"measured report lacks {group}; no A/delta values are fabricated",
            ))
            continue
        for part, values in _part_items(report[group]):
            if not isinstance(values, dict) or "A" not in values or not isinstance(values.get("delta"), dict):
                defects.append(_defect(
                    identity, "quality_self_collision", "unknown", str(part), group,
                    "part lacks A or delta; no A/delta values are fabricated",
                ))
    return defects


def _chatter_signal_rows(
    identity: dict[str, Any], audit_kind: str, section: str, part: str,
    values: dict[str, Any], unit: str, context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(values, dict):
        return []
    common = {**values, "source_path": identity["quality_path"]}
    content = {**common, "max": common.get("max")}
    excess = {
        **common,
        "mean": None, "p95": None,
        "max": values.get("max_excess"),
        "count": values.get("violating_frames", values.get("violations", "")),
    }
    rows = [
        _row(identity, audit_kind, section, "high_frequency_content", part, content, unit=unit),
        _row(identity, audit_kind, section, "chatter_excess", part, excess, unit=unit),
    ]
    if context is not None:
        rows = [row | context for row in rows]
    coordinates = values.get("per_coordinate", [])
    for axis, axis_values in enumerate(coordinates if isinstance(coordinates, list) else []):
        if not isinstance(axis_values, dict):
            continue
        axis_part = f"{part} / axis_{axis}"
        axis_common = {**axis_values, "source_path": identity["quality_path"]}
        axis_rows = [
            _row(identity, audit_kind, section, "high_frequency_content", axis_part, axis_common, unit=unit),
            _row(identity, audit_kind, section, "chatter_excess", axis_part, {
                **axis_common, "mean": None, "p95": None,
                "max": axis_values.get("max_excess"),
                "count": axis_values.get("violating_frames", ""),
            }, unit=unit),
        ]
        if context is not None:
            axis_rows = [row | context for row in axis_rows]
        rows.extend(axis_rows)
    return rows


def _chatter_rows(identity: dict[str, Any], quality: dict[str, Any]) -> list[dict[str, Any]]:
    chatter = quality.get("chatter", {})
    rows: list[dict[str, Any]] = []
    if not isinstance(chatter, dict):
        return rows
    correction = chatter.get("actual_final_minus_fixed_ref", {})
    if isinstance(correction, dict):
        for part, values in (correction.get("joints", {}) or {}).items():
            unit = values.get("units", "native") if isinstance(values, dict) else "native"
            rows.extend(_chatter_signal_rows(
                identity, "chatter", "actual_final_minus_fixed_ref:joints", part, values, unit,
            ))
        for joint, signals in (correction.get("root", {}) or {}).items():
            for signal, values in (signals if isinstance(signals, dict) else {}).items():
                rows.extend(_chatter_signal_rows(
                    identity, "chatter", "actual_final_minus_fixed_ref:root",
                    f"{joint} / {signal}", values, "m" if signal == "translation" else "rad",
                ))
        for part, values in (correction.get("production_motion_budgets_audit", {}) or {}).items():
            rows.extend(_chatter_signal_rows(
                identity, "chatter", "production_motion_budgets_audit", part, values, "native",
            ))
    for role, source_audit in (chatter.get("original_source_audits", {}) or {}).items():
        if not isinstance(source_audit, dict):
            continue
        for group in ("landmarks", "calibrated_body_orientation", "root_vs_original_source"):
            block = source_audit.get(group, {})
            for part, signals in (block if isinstance(block, dict) else {}).items():
                if group == "root_vs_original_source":
                    items = signals.items() if isinstance(signals, dict) else [(str(part), signals)]
                else:
                    items = [(str(part), signals)]
                for signal, values in items:
                    if group == "root_vs_original_source":
                        child = f"{part} / {signal}"
                        unit = "m" if signal == "translation" else "rad"
                    elif group == "calibrated_body_orientation":
                        child, unit = str(part), "rad"
                    else:
                        child, unit = str(part), "native"
                    rows.extend(_chatter_signal_rows(
                        identity, "chatter", f"source_chatter:{role}:{group}", child, values, unit,
                    ))
    return rows


def _smoothness_rows(identity: dict[str, Any], smoothness: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    joints = smoothness.get("joints", {})
    for part, details in (joints if isinstance(joints, dict) else {}).items():
        if not isinstance(details, dict):
            continue
        chatter = details.get("source_relative_chatter", {})
        if isinstance(chatter, dict):
            rows.append(_row(
                identity, "joint_smoothness", "source_relative_chatter",
                "chatter_excess", part, {
                    **chatter, "source_path": identity["smoothness_path"],
                }, unit=details.get("units", "native"),
            ))
        orders = details.get("orders", {})
        for order, stats in (orders if isinstance(orders, dict) else {}).items():
            rows.append(_row(
                identity, "joint_smoothness", f"derivative_order_{order}",
                f"derivative_increase_order_{order}", part, {
                    **stats, "source_path": identity["smoothness_path"],
                }, unit=details.get("units", "native"),
            ))
        limits = details.get("mechanical_limits", {})
        for limit_name in ("velocity", "effort"):
            stats = limits.get(limit_name, {})
            if isinstance(stats, dict):
                rows.append(_row(
                    identity, "joint_smoothness", f"mechanical_limit_{limit_name}",
                    limit_name, part, {
                        **stats, "source_path": identity["smoothness_path"],
                    }, unit=stats.get("unit", "native"),
                ))
    return rows


def _defect(
    identity: dict[str, Any], kind: str, severity: str, part: str,
    metric: str, detail: str, frame: Any = "",
) -> dict[str, str]:
    return {
        "run_id": identity.get("run_id", ""), "arm": identity.get("arm", ""),
        "mode": identity.get("mode", ""), "repeat": identity.get("repeat", ""),
        "clip": identity.get("clip", ""), "kind": kind, "severity": severity,
        "part": part, "metric": metric, "frame": frame, "detail": detail,
    }


def _quality_defects(identity: dict[str, Any], report: dict[str, Any]) -> list[dict[str, str]]:
    block = report.get("non_degradation_against_A", {})
    regressions = block.get("regressions", []) if isinstance(block, dict) else []
    return [
        _defect(
            identity, "quality_regression", "per_part", str(item.get("part", "")),
            str(item.get("metric", "")),
            "A-to-final mean/P95/max delta has at least one positive native-unit value",
        )
        for item in regressions if isinstance(item, dict)
    ]


def _report_defects(
    identity: dict[str, Any], quality: dict[str, Any] | None,
    training: dict[str, Any] | None, smoothness: dict[str, Any] | None,
) -> list[dict[str, str]]:
    defects: list[dict[str, str]] = []
    if quality is None:
        defects.append(_defect(identity, "missing_audit", "unknown", "", "spline_quality", "missing or invalid spline_quality.json"))
    else:
        defects.extend(_quality_defects(identity, quality))
        defects.extend(_quality_self_defects(identity, quality))
        defects.extend(_contact_chatter_defects(identity, quality))
    if training is None:
        defects.append(_defect(identity, "missing_audit", "unknown", "", "training_collision", "missing or invalid training_collision.json"))
    else:
        for section in ("static_scene", "moving_objects", "self_collision"):
            report = training.get(section, {})
            if not isinstance(report, dict):
                continue
            unknown = report.get("unknown_query_count", report.get("uncertain_samples", 0))
            if unknown:
                defects.append(_defect(identity, "unknown_query", "unknown", report.get("worst_body", ""), section, f"unknown_query_count={unknown}"))
            if report.get("status") == "unknown":
                defects.append(_defect(identity, "audit_status", "unknown", report.get("worst_body", ""), section, str(report.get("reason", "unknown"))))
        self_report = training.get("self_collision", {})
        if isinstance(self_report, dict):
            for threshold in ("frames_over_1mm", "frames_over_5mm", "frames_over_1cm"):
                frames = self_report.get(threshold, [])
                if frames:
                    defects.append(_defect(identity, "self_collision", "defect", "<aggregate>", threshold, _json_text(frames), frames[-1]))
    if smoothness is None:
        defects.append(_defect(identity, "missing_audit", "unknown", "", "joint_smoothness", "missing or invalid joint_smoothness_supplemental.json"))
    elif smoothness.get("all_individual_budgets_pass") is False:
        defects.append(_defect(identity, "joint_budget", "defect", "", "all_individual_budgets", "at least one native chatter/velocity/effort budget failed"))
    return defects


def _coverage_row(
    identity: dict[str, Any], receipt: dict[str, Any] | None,
    quality: dict[str, Any] | None, training: dict[str, Any] | None,
    smoothness: dict[str, Any] | None,
) -> dict[str, Any]:
    certification = quality.get("certification", {}) if isinstance(quality, dict) else {}
    return {
        **{key: identity.get(key, "") for key in ("run_id", "arm", "mode", "repeat", "clip", "source_frame_start")},
        "receipt_present": receipt is not None,
        "initial_solver_returncode": (receipt or {}).get("solver_returncode", ""),
        "quality_present": quality is not None,
        "training_collision_present": training is not None,
        "smoothness_present": smoothness is not None,
        "quality_certification": certification.get("all_quality_criteria_pass", "missing") if certification else "missing",
        "training_acceptance": training.get("acceptance_status", "missing") if isinstance(training, dict) else "missing",
        "smoothness_budgets_pass": smoothness.get("all_individual_budgets_pass", "missing") if isinstance(smoothness, dict) else "missing",
        "pass_inferred": False,
    }


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _markdown(
    coverage: list[dict[str, Any]], parts: list[dict[str, Any]],
    defects: list[dict[str, str]],
) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|") if value is not None else ""

    lines = [
        "# Structural spline quality export", "",
        "Read-only export of existing JSON audits. Missing, failed, and unknown reports remain listed; no pass is inferred.", "",
        "- Complete per-part table: [`quality_parts.csv`](quality_parts.csv)",
        "- Audit coverage: [`coverage.csv`](coverage.csv)",
        "- Coverage failures / defects / unknowns: [`defects.csv`](defects.csv)", "",
        "Native report units are retained with `unit_conversion=none`. No angular cap, percentage tolerance, or pass threshold is added.", "",
        "For TRUMANS `source.json.frame_start`, `worst_frame_clip_local` remains clip-local and `original_source_frame = worst_frame_clip_local + frame_start`. This is provenance indexing only; no frame timing or motion alignment is changed.", "",
        "## Existing audit status by run group", "",
        "| arm | mode | repeat | clips | quality | training | smoothness |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    groups: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in coverage:
        groups[(row["arm"], row["mode"], row["repeat"])].append(row)
    for (arm, mode, repeat), group in sorted(groups.items(), key=lambda item: str(item[0])):
        lines.append(
            f"| {arm} | {mode} | {repeat} | {len(group)} | "
            f"{sum(bool(row['quality_present']) for row in group)} | "
            f"{sum(bool(row['training_collision_present']) for row in group)} | "
            f"{sum(bool(row['smoothness_present']) for row in group)} |"
        )
    lines.extend([
        "",
        "## Worst part within each clip, metric, and unit",
        "",
        "Deltas are never compared across metrics or units. Rows remain per clip; this is not an arm-level average.", "",
        "| arm | mode | repeat | clip | audit | metric | unit | worst part | Δmean | Δp95 | Δmax |",
        "|---|---|---|---|---|---|---|---|---:|---:|---:|",
    ])
    metric_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in parts:
        metric_groups[(
            row["arm"], row["mode"], row["repeat"], row["clip"],
            row["audit_kind"], row["metric"], row["unit"],
        )].append(row)
    for key, group in sorted(metric_groups.items(), key=lambda item: str(item[0])):
        arm, mode, repeat, clip, audit_kind, metric, unit = key
        worst = max(
            group,
            key=lambda row: max(row.get(key) or 0 for key in ("delta_mean", "delta_p95", "delta_max")),
            default=None,
        )
        lines.append(
            f"| {arm} | {mode} | {repeat} | {clip} | {audit_kind} | {metric} | {unit} | "
            f"{cell(worst['part'] if worst else 'none')} | {cell(worst.get('delta_mean') if worst else '')} | "
            f"{cell(worst.get('delta_p95') if worst else '')} | {cell(worst.get('delta_max') if worst else '')} |"
        )
    lines.extend(["", "## Defect and unknown counts", ""])
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for item in defects:
        counts[(item["kind"], item["severity"])] += 1
    if counts:
        lines.extend(["| kind | severity | count |", "|---|---|---:|"])
        lines.extend(f"| {kind} | {severity} | {count} |" for (kind, severity), count in sorted(counts.items()))
    else:
        lines.append("No reported defects or unknowns (this is not a collision-free or quality pass certificate).")
    return "\n".join(lines) + "\n"


def export_tables(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    arms = [item.strip() for item in args.arms.split(",") if item.strip()]
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    expected_clips = list(CLIPS)
    if args.clips:
        requested = [item.strip() for item in args.clips.split(",") if item.strip()]
        unknown = sorted(set(requested) - set(CLIPS))
        if unknown:
            raise ValueError(f"unknown benchmark clips {unknown}")
        expected_clips = requested
    coverage: list[dict[str, Any]] = []
    parts: list[dict[str, Any]] = []
    defects: list[dict[str, str]] = []
    index = 0
    for arm in arms:
        for mode in modes:
            for repeat in range(1, args.repeats + 1):
                for clip in expected_clips:
                    index += 1
                    directory = run_dir / arm / mode / f"repeat_{repeat}" / clip
                    receipt_path = directory / "receipt.json"
                    receipt = _read_json(receipt_path)
                    fallback_identity = {
                        "run_id": f"{index:04d}", "arm": arm, "mode": mode,
                        "repeat": repeat, "clip": clip,
                    }
                    if receipt is None:
                        defects.append(_defect(
                            fallback_identity, "receipt", "missing", "", "receipt_json",
                            f"expected receipt absent: {receipt_path}",
                        ))
                    source = _read_json(directory / "source.json")
                    source_frame_start = source.get("frame_start", 0) if source is not None else None
                    identity = {
                        **fallback_identity,
                        "arm": receipt.get("arm", arm) if receipt is not None else arm,
                        "mode": receipt.get("mode", mode) if receipt is not None else mode,
                        "repeat": receipt.get("repeat", repeat) if receipt is not None else repeat,
                        "clip": receipt.get("clip", clip) if receipt is not None else clip,
                        "source_frame_start": source_frame_start,
                        "quality_path": str(directory / "spline_quality.json"),
                        "training_path": str(directory / "training_collision.json"),
                        "smoothness_path": str(directory / "joint_smoothness_supplemental.json"),
                    }
                    quality = _read_json(directory / "spline_quality.json")
                    training = _read_json(directory / "training_collision.json")
                    smoothness = _read_json(directory / "joint_smoothness_supplemental.json")
                    if receipt is not None and receipt.get("solver_returncode", 0) not in (0, None):
                        defects.append(_defect(identity, "initial_run", "failed", "", "solver_returncode", str(receipt.get("solver_returncode"))))
                    if source is None:
                        defects.append(_defect(identity, "source_metadata", "unknown", "", "source.json", "missing or invalid; original-source frame index is unknown"))
                    coverage.append(_coverage_row(identity, receipt, quality, training, smoothness))
                    defects.extend(_report_defects(identity, quality, training, smoothness))
                    if quality is not None:
                        parts.extend(_comparison_rows(identity, quality))
                        parts.extend(_non_degradation_rows(identity, quality))
                        parts.extend(_contact_rows(identity, quality))
                        parts.extend(_contact_chatter_rows(identity, quality))
                        parts.extend(_geometry_rows(identity, quality, training or {}))
                        parts.extend(_chatter_rows(identity, quality))
                        parts.extend(_quality_self_rows(identity, quality))
                    if training is not None:
                        parts.extend(_self_rows(identity, training))
                    if smoothness is not None:
                        parts.extend(_smoothness_rows(identity, smoothness))
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "quality_parts.csv", CSV_FIELDS, parts)
    _write_csv(output_dir / "coverage.csv", [
        "run_id", "arm", "mode", "repeat", "clip", "source_frame_start",
        "receipt_present", "initial_solver_returncode", "quality_present",
        "training_collision_present", "smoothness_present", "quality_certification",
        "training_acceptance", "smoothness_budgets_pass", "pass_inferred",
    ], coverage)
    _write_csv(output_dir / "defects.csv", [
        "run_id", "arm", "mode", "repeat", "clip", "kind", "severity",
        "part", "metric", "frame", "detail",
    ], defects)
    (output_dir / "README.md").write_text(_markdown(coverage, parts, defects), encoding="utf-8")
    return {
        "clips": len(coverage), "part_rows": len(parts), "defect_rows": len(defects),
        "output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arms", default="A,B,C,D")
    parser.add_argument("--modes", default="normal")
    parser.add_argument("--clips", default="")
    args = parser.parse_args()
    print("SAVED", json.dumps(export_tables(args), sort_keys=True))


if __name__ == "__main__":
    main()
