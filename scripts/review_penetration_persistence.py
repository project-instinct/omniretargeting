"""Per-pair penetration persistence and per-frame reset usability audit.

The input is a benchmark output directory.  Scene/object depths are consumed
from the frozen ``scene_metrics`` audit; self-collision depths are recomputed
with the same model filtering and configuration as the training-collision
audit.  The script is read-only except for its requested JSON output.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from omniretargeting.cpu_budget import available_cpus
from omniretargeting.moving_scene import MovingSceneQuery
from omniretargeting.quality_policy import (
    THRESHOLDS_M,
    classify as classify_shared,
    evaluate as evaluate_shared,
    pair_statistics,
    reset_states,
)
from omniretargeting.robot_config import load_robot_config
from omniretargeting.self_collision import SelfCollisionQuery
from omniretargeting.training_collision_parallel import query_self_frames


# Reporting and classification constants are deliberately local to this new
# acceptance view.  They can be tuned only with before/after data in hand.
LIST_MIN_DEPTH_M = 0.001
MILD_M = 0.005
DEEP_M = 0.015
PERSIST_SECONDS = 0.5
PERSIST_FRACTION = 0.25
DEFAULT_FPS = 30.0

# Training-policy calibration view.  Unlike the acceptance classes above,
# these thresholds intentionally distinguish support residuals from persistent
# non-support penetration.
# A deliberately stricter, support-blind control policy.
POLICY_STRICT_DEEP_M = 0.010
POLICY_STRICT_TWO_MM_SECONDS = 0.2
POLICY_STRICT_TWO_MM_FRACTION = 0.05
POLICY_STRICT_HALF_MM_M = 0.0005
POLICY_STRICT_HALF_MM_SECONDS = 0.5
POLICY_STRICT_HALF_MM_FRACTION = 0.10

# Keep reset geometry aligned with scripts/review_training_collision.py.  The
# epsilon there excludes exact threshold values from floating-point failures.
RESET_SCENE_M = 0.005
RESET_SELF_M = 0.001
DEPTH_EPSILON_M = 1e-6


def compress_frame_indices(labels: np.ndarray) -> list[list[int]]:
    """Compress sorted frame labels into inclusive integer intervals."""
    labels = np.asarray(labels, dtype=int)
    if len(labels) == 0:
        return []
    if np.any(np.diff(labels) <= 0):
        raise ValueError("frame labels must be strictly increasing")
    intervals: list[list[int]] = []
    start = previous = int(labels[0])
    for label in labels[1:]:
        label = int(label)
        if label != previous + 1:
            intervals.append([start, previous])
            start = label
        previous = label
    intervals.append([start, previous])
    return intervals


def _load_support_index(
    prepared_path: Path,
    model: mujoco.MjModel,
    model_path: Path,
    *,
    start: int,
    frames: int,
) -> dict[str, Any]:
    """Index prepared support contacts by ``(robot body, surface id)``.

    A contact row counts only when its source contact mask and its assigned
    support role are both true.  Matching later uses the robot body of the
    collision geom and the exact counterpart surface id (0 for static scene;
    moving objects are offset by one when the static scene is present).
    """
    try:
        inputs = np.load(prepared_path, allow_pickle=True).item()
        if Path(str(inputs["model_path"])).resolve() != model_path.resolve():
            raise ValueError("prepared.npy uses a different robot model")
        surface = inputs.get("surface", {})
        contact = np.asarray(surface.get("surface_contact_mask", ()), dtype=bool)
        support = np.asarray(
            surface.get("contact_role_surface_support", np.zeros_like(contact)),
            dtype=bool,
        )
        surface_ids = np.asarray(surface.get("surface_contact_surface_ids", ()), dtype=int)
        geom_ids = np.asarray(surface.get("surface_contact_robot_geoms", ()), dtype=int)
        if contact.ndim != 2 or support.shape != contact.shape or surface_ids.shape != contact.shape:
            raise ValueError("prepared surface contact arrays have inconsistent shapes")
        if len(geom_ids) != contact.shape[1]:
            raise ValueError("prepared surface contact geom count differs from columns")
        selected = contact[start : start + frames] & support[start : start + frames]
        ids_selected = surface_ids[start : start + frames]
        if selected.shape != (frames, contact.shape[1]):
            raise ValueError("prepared surface contact timeline does not cover motion frames")
        by_body_surface: dict[tuple[int, int], np.ndarray] = {}
        for column, geom_id in enumerate(geom_ids):
            body_id = int(model.geom_bodyid[int(geom_id)])
            for surface_id in np.unique(ids_selected[selected[:, column], column]):
                key = (body_id, int(surface_id))
                mask = selected[:, column] & (ids_selected[:, column] == surface_id)
                by_body_surface[key] = by_body_surface.get(
                    key, np.zeros(frames, dtype=bool)
                ) | mask
        return {
            "by_body_surface": by_body_surface,
            "any_surface": (
                np.logical_or.reduce(list(by_body_surface.values()))
                if by_body_surface
                else np.zeros(frames, dtype=bool)
            ),
            "source": "prepared_surface_contact_mask_and_contact_role_surface_support",
            "object_surface_offset": 1,
        }
    except Exception as error:
        return {
            "by_body_surface": {},
            "any_surface": np.zeros(frames, dtype=bool),
            "source": "unavailable_no_prepared_support_evidence",
            "reason": f"{type(error).__name__}: {error}",
            "object_surface_offset": 1,
        }


def _support_fields(
    pair: dict[str, Any],
    depths: np.ndarray,
    labels: np.ndarray,
    support_index: dict[str, Any],
    scene_record: dict[str, Any],
) -> dict[str, Any]:
    penetrating = depths > 0.0
    if support_index["source"] == (
        "prepared_surface_contact_mask_and_contact_role_surface_support"
    ):
        if pair["kind"] == "static":
            surface_id = 0
        elif pair.get("object_index") is not None:
            surface_id = int(pair["object_index"]) + int(
                support_index.get("object_surface_offset", 1)
            )
        else:
            surface_id = None
        support_contact = (
            support_index["by_body_surface"].get(
                (int(pair["robot_geom"]["body_id"]), surface_id),
                np.zeros(len(labels), dtype=bool),
            )
            if surface_id is not None
            else np.zeros(len(labels), dtype=bool)
        )
        selected = penetrating & support_contact
        return {
            "support_frames": support_contact,
            "support_normal": bool(np.any(selected)),
            "support_contact_frame_count": int(np.sum(selected)),
            "support_fraction_of_penetrating_frames": (
                float(np.sum(selected) / np.sum(penetrating))
                if np.any(penetrating)
                else 0.0
            ),
            "support_surface_id": surface_id,
            "evidence_source": support_index["source"],
        }
    fallback_frames = np.zeros(len(labels), dtype=bool)
    return {
        "support_frames": fallback_frames,
        "support_normal": False,
        "support_contact_frame_count": None,
        "support_fraction_of_penetrating_frames": None,
        "support_surface_id": None,
        "evidence_source": (
            "body_name_and_scene_metrics_aggregate_worst_exit_z_heuristic"
            if support_index.get("source") != "unavailable_no_prepared_support_evidence"
            else support_index["source"]
        ),
    }


def _geom_description(model: mujoco.MjModel, geom_id: int) -> dict[str, Any]:
    geom_id = int(geom_id)
    geom_name = str(model.geom(geom_id).name) or f"geom_{geom_id}"
    body_id = int(model.geom_bodyid[geom_id])
    return {
        "id": geom_id,
        "name": geom_name,
        "body": str(model.body(body_id).name),
        "body_id": body_id,
    }


def _threshold_key(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _localized_stats(stats: dict[str, Any], start: int) -> dict[str, Any]:
    result = dict(stats)
    if result["worst_frame"] is not None:
        result["worst_frame"] = int(result["worst_frame"]) + start
    result["unknown_frames"] = [int(x) + start for x in result["unknown_frames"]]
    thresholds = {}
    for key, value in result["thresholds_m"].items():
        value = dict(value)
        value["frames"] = [int(x) + start for x in value["frames"]]
        value["runs"] = [[int(x) + start for x in run] for run in value["runs"]]
        if value["longest_run_start"] is not None:
            value["longest_run_start"] = int(value["longest_run_start"]) + start
        value["upper"] = {
            **value["upper"],
            "runs": [[int(x) + start for x in run] for run in value["upper"]["runs"]],
        }
        thresholds[key] = value
    result["thresholds_m"] = thresholds
    return result


def _acceptance_classes(stats: dict[str, Any]) -> list[str]:
    mild = stats["thresholds_m"][f"{MILD_M:.3f}"]
    persistent = bool(
        mild["longest_run_seconds"] > PERSIST_SECONDS
        or mild["frame_fraction"] > PERSIST_FRACTION
    )
    classes = []
    if stats["max_depth_m"] > DEEP_M:
        classes.append("deep")
    if persistent:
        classes.append("persistent")
    if not classes:
        classes.append("mild_brief" if stats["max_depth_m"] > MILD_M else "negligible")
    return classes


def _strict_classes(stats: dict[str, Any]) -> list[str]:
    two = stats["thresholds_m"]["0.002"]
    half = stats["thresholds_m"]["0.0005"]
    persistent = bool(
        two["longest_run_seconds"] > POLICY_STRICT_TWO_MM_SECONDS
        or two["frame_fraction"] > POLICY_STRICT_TWO_MM_FRACTION
        or half["longest_run_seconds"] > POLICY_STRICT_HALF_MM_SECONDS
        or half["frame_fraction"] > POLICY_STRICT_HALF_MM_FRACTION
    )
    classes = []
    if stats["max_depth_m"] > POLICY_STRICT_DEEP_M:
        classes.append("deep")
    if persistent:
        classes.append("persistent")
    return classes


def _pair_json(
    pair: dict[str, Any],
    *,
    labels: np.ndarray,
    frames_per_second: float,
    support_index: dict[str, Any],
    scene_record: dict[str, Any],
    unknown: np.ndarray,
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    depths = np.asarray(pair["depths"], dtype=float)
    stats = pair_statistics(depths, frames_per_second, unknown)
    support = _support_fields(pair, depths, labels, support_index, scene_record)
    decision = classify_shared(
        stats,
        lateral_or_self=pair["kind"] == "self",
        support_frames=support["support_frames"],
        depth=depths,
    )
    classes = _acceptance_classes(stats)
    penetrating = depths > 0.0
    support_evidence = {
        **support,
        "support_frames": support["support_frames"].tolist(),
    }
    result = {
        "pair_id": pair["key"],
        "kind": pair["kind"],
        "robot_geom": pair["robot_geom"],
        "other": pair["other"],
        "max_depth_m": float(depths.max()),
        "worst_frame": int(labels[int(stats["worst_frame"])]),
        "mean_depth_when_penetrating_m": (
            float(np.mean(depths[penetrating])) if np.any(penetrating) else 0.0
        ),
        "thresholds_m": _localized_stats(stats, int(labels[0]))["thresholds_m"],
        "classification": classes,
        "support_normal": support["support_normal"],
        # Saved depth arrays do not carry per-frame exit normals/components.
        "support_residual_verified": False,
        "support_residual_pending_verification": "support_residual" in decision["classes"],
        "support_evidence": support_evidence,
        "policy_v1": decision["classes"],
        "policy_strict": _strict_classes(stats),
        "events": [
            {**event, "frames": event["frames"].tolist()}
            for event in decision["events"]
        ],
    }
    series = {
        "depth": depths,
        "unknown": np.asarray(unknown, dtype=bool),
        "kind": pair["kind"],
        "support_frames": support["support_frames"],
        "lateral": pair["kind"] == "self",
    }
    return result, classes, series


def _read_profile_model_path(invocation_path: Path) -> Path:
    invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
    argv = invocation.get("argv")
    if not isinstance(argv, list) or "--robot-config" not in argv:
        raise ValueError("invocation.json has no --robot-config argument")
    index = argv.index("--robot-config")
    if index + 1 >= len(argv):
        raise ValueError("invocation.json has no value after --robot-config")
    configured = Path(str(argv[index + 1])).expanduser()
    if configured.is_absolute():
        profile_path = configured
    else:
        candidates = [
            (Path.cwd() / configured).resolve(),
            (invocation_path.parent / configured).resolve(),
        ]
        profile_path = next((path for path in candidates if path.is_file()), candidates[0])
    profile = load_robot_config(profile_path)
    return Path(profile["urdf_path"]).expanduser().resolve()


def _motion_framerate(payload: np.lib.npyio.NpzFile, override: float | None) -> float:
    if override is not None:
        if not np.isfinite(override) or override <= 0:
            raise ValueError("--frames-per-second must be finite and positive")
        return float(override)
    try:
        value = float(np.asarray(payload["framerate"]).reshape(()))
    except (KeyError, TypeError, ValueError):
        value = 0.0
    return value if np.isfinite(value) and value > 0 else DEFAULT_FPS


def _slice_object(spec: dict[str, Any], start: int, frames: int) -> dict[str, Any]:
    result = dict(spec)
    for field in ("translations", "rotations", "scales"):
        if field in result:
            result[field] = np.asarray(result[field])[start : start + frames]
    return result


def _query_moving_by_object(
    model: mujoco.MjModel,
    objects: list[dict[str, Any]],
    qpos: np.ndarray,
    *,
    start: int,
) -> tuple[list[np.ndarray], list[str], float, int]:
    started = time.perf_counter()
    local_objects = [_slice_object(obj, start, len(qpos)) for obj in objects]
    query = MovingSceneQuery(model, local_objects, len(qpos))
    depths = [np.zeros((len(qpos), model.ngeom), dtype=float) for _ in objects]
    failures = 0
    for frame, pose in enumerate(np.asarray(qpos, dtype=float)):
        result = query.query(pose, frame)
        failures += len(result.get("failures", []))
        combined = np.maximum(result["forward_depth"], result["reverse_depth"])
        for object_id, object_depth in enumerate(combined):
            depths[object_id][frame] = object_depth
    names = []
    seen: dict[str, int] = {}
    for object_id, obj in enumerate(objects):
        name = str(obj.get("name") or f"object_{object_id}")
        count = seen.get(name, 0)
        seen[name] = count + 1
        names.append(name if count == 0 else f"{name}#{count + 1}")
    return depths, names, time.perf_counter() - started, failures


def _query_self(
    model: mujoco.MjModel, qpos: np.ndarray, *, workers: int
) -> tuple[dict[tuple[int, int], np.ndarray], np.ndarray, list[int], dict[str, Any]]:
    started = time.perf_counter()
    frame_depth = np.zeros(len(qpos), dtype=float)
    errors: list[int] = []
    try:
        query = SelfCollisionQuery(model)
        config = dict(query.cfg)
        structural_pairs = query.use_model_pairs()
        from omniretargeting.structural_overlap import StructuralOverlap
        structural = StructuralOverlap(query, len(qpos))
        pair_depths: dict[tuple[int, int], np.ndarray] = {}
        for frame, result in enumerate(query_self_frames(query, qpos, workers)):
            if result.get("error") is not None:
                structural.unknown[frame] = True
                errors.append(frame)
                continue
            for row in structural.collect(result.get("rows", []), frame):
                depth = max(-float(row["clearance"]), 0.0)
                if depth <= 0.0:
                    continue
                pair = tuple(sorted(map(int, row["pair"])))
                pair_depths.setdefault(pair, np.zeros(len(qpos), dtype=float))[frame] = depth
                frame_depth[frame] = max(frame_depth[frame], depth)
        details = {
            "status": "measured" if not errors else "unknown",
            "config": config,
            "eligible_pair_count": int(len(query.pairs)),
            "structural_pair_count": int(len(structural_pairs)),
            "structural_pairs": structural_pairs,
            "structural_pairs_included": False,
            "model_structural_overlap": structural.describe(),
            "pair_policy": "compiled_MJCF_measurement_separate_structural_overlaps",
        }
    except Exception as error:  # Keep unknown data explicit rather than passing.
        pair_depths = {}
        errors = list(range(len(qpos)))
        details = {
            "status": "unknown",
            "initialization_error": f"{type(error).__name__}: {error}",
        }
    details["query_seconds"] = time.perf_counter() - started
    details["workers"] = int(workers)
    details["unknown_query_count"] = len(errors)
    details["error_frames"] = np.asarray(errors, dtype=int).tolist()
    return pair_depths, frame_depth, errors, details


def _unknown_record(
    scene_record: dict[str, Any], self_errors: int, moving_recomputed_failures: int = 0
) -> dict[str, int]:
    static = scene_record.get("static", {})
    moving = scene_record.get("moving", {})
    static_count = (
        1
        if not isinstance(static, dict)
        else int(static.get("uncertain_samples", 0) or 0)
    )
    moving_components = (
        1
        if not isinstance(moving, dict)
        else int(moving.get("unknown_components", 0) or 0)
    )
    moving_failures = len(moving.get("failures", []) or [])
    return {
        "static_uncertain_samples": static_count,
        "moving_unknown_components": moving_components,
        "moving_query_failures": moving_failures,
        "moving_recomputed_failures": int(moving_recomputed_failures),
        "self_query_errors": int(self_errors),
        "total": (
            static_count
            + moving_components
            + moving_failures
            + int(moving_recomputed_failures)
            + int(self_errors)
        ),
    }


def _reset_report(
    *,
    static_depth: np.ndarray,
    moving_depth: np.ndarray,
    self_depth: np.ndarray,
    labels: np.ndarray,
    scene_unknown: bool,
    self_error_frames: set[int],
    lateral: np.ndarray,
    support_unverified: np.ndarray | None = None,
) -> dict[str, Any]:
    static_depth = np.asarray(static_depth, dtype=float)
    moving_depth = np.asarray(moving_depth, dtype=float)
    self_depth = np.asarray(self_depth, dtype=float)
    unknown = np.zeros(len(labels), dtype=bool)
    if scene_unknown:
        unknown[:] = True
    unknown[list(self_error_frames)] = True
    if support_unverified is not None:
        unknown |= np.asarray(support_unverified, bool)
    states = reset_states(static_depth, moving_depth, self_depth, lateral, unknown)
    usable = states == "candidate"
    local_unusable = np.flatnonzero(~usable)
    over_scene = np.maximum(static_depth, moving_depth) > RESET_SCENE_M + DEPTH_EPSILON_M
    over_self = self_depth > RESET_SELF_M + DEPTH_EPSILON_M
    reasons = {
        "scene_depth_over_threshold": compress_frame_indices(
            labels[np.flatnonzero(over_scene)]
        ),
        "self_depth_over_threshold": compress_frame_indices(
            labels[np.flatnonzero(over_self)]
        ),
        "unknown_geometry": compress_frame_indices(labels[np.flatnonzero(unknown)]),
        "support_normal_component_unverified": compress_frame_indices(labels[np.flatnonzero(
            np.zeros(len(labels), bool) if support_unverified is None else support_unverified)]),
    }
    return {
        "usable": usable.tolist(),
        "states": states.tolist(),
        "usable_frame_count": int(np.sum(usable)),
        "usable_frame_fraction": float(np.mean(usable)) if len(labels) else 0.0,
        "unusable_frame_count": int(len(local_unusable)),
        "unusable_frames": compress_frame_indices(labels[local_unusable]),
        "unusable_reasons": reasons,
        "failure_thresholds_m": {
            "static_or_moving": RESET_SCENE_M,
            "self": RESET_SELF_M,
            "comparison": "strictly greater than threshold + 1e-6 m",
        },
    }


def _lateral_reset_mask(
    model: mujoco.MjModel,
    static_depth: np.ndarray,
    moving_sequences: list[tuple[str, np.ndarray]],
    moving_depth: np.ndarray,
    support_index: dict[str, Any],
) -> np.ndarray:
    static = np.asarray(static_depth, dtype=float)
    moving = np.asarray(moving_depth, dtype=float)
    moving_by_object = [item[1] for item in moving_sequences] or [moving]
    result = np.zeros(len(static), dtype=bool)
    # A shallower lateral pair must not hide behind a deeper support residual.
    for surface_id, depths in enumerate([static, *moving_by_object]):
        for geom_id in np.flatnonzero(np.any(depths > RESET_SELF_M + DEPTH_EPSILON_M, axis=0)):
            body_id = int(model.geom_bodyid[geom_id])
            supported = support_index["by_body_surface"].get(
                (body_id, surface_id), np.zeros(len(static), dtype=bool))
            result |= (depths[:, geom_id] > RESET_SELF_M + DEPTH_EPSILON_M) & ~supported

    return result


def _summary(
    pairs: list[dict[str, Any]],
    classifications: dict[str, list[str]],
    quality: dict[str, Any],
    *,
    frame_peak: np.ndarray,
    labels: np.ndarray,
    frames_per_second: float,
    unknown: dict[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    mild_key = f"{MILD_M:.3f}"
    names = ("deep", "persistent", "mild_brief", "negligible")
    acceptance_counts = {name: sum(name in values for values in classifications.values()) for name in names}
    policy_counts = quality["summary"]["classification_pair_counts"]
    strict_counts = {
        name: sum(name in pair["policy_strict"] for pair in pairs)
        for name in ("deep", "persistent")
    }
    deep = [pair for pair in pairs if "deep" in classifications[pair["pair_id"]]]
    persistent = [pair for pair in pairs if "persistent" in classifications[pair["pair_id"]]]
    worst_deep = max(deep, key=lambda pair: pair["max_depth_m"], default=None)
    duration = lambda pair: (pair["thresholds_m"][mild_key]["longest_run_seconds"], pair["max_depth_m"])
    worst_persistent = max(persistent, key=duration, default=None)

    def brief(pair: dict[str, Any] | None):
        if pair is None:
            return None
        mild = pair["thresholds_m"][mild_key]
        return {
            "pair_id": pair["pair_id"],
            "kind": pair["kind"],
            "robot_geom": pair["robot_geom"],
            "other": pair["other"],
            "max_depth_m": pair["max_depth_m"],
            "longest_mild_run_seconds": mild["longest_run_seconds"],
        }

    summary = {
        "listed_pair_count": len(pairs),
        "classification_pair_counts": acceptance_counts,
        "policy_v1_pair_counts": policy_counts,
        "policy_strict_pair_counts": strict_counts,
        "worst_pairs": {
            "deep_by_depth": brief(worst_deep),
            "persistent_by_mild_run_duration": brief(worst_persistent),
        },
        "max_depth_by_kind_m": quality["summary"]["max_depth_by_kind_m"],
        "union_frames": {
            "over_5mm": labels[np.flatnonzero(frame_peak > MILD_M + DEPTH_EPSILON_M)].tolist(),
            "over_1cm": labels[np.flatnonzero(frame_peak > 0.010 + DEPTH_EPSILON_M)].tolist(),
        },
        "unknown": unknown,
        "duration_definition": "run_frames / frames_per_second",
        "frames_per_second": float(frames_per_second),
    }
    acceptance_defects = [pair["pair_id"] for pair in pairs if {"deep", "persistent"} & set(classifications[pair["pair_id"]])]
    verdict = {
        "defect_pairs": len(acceptance_defects),
        "defect_pair_ids": acceptance_defects,
        "policy_v1_defect_pairs": len(quality["defect_pairs"]),
        "policy_v1_defect_pair_ids": quality["defect_pairs"],
        "policy_strict_defect_pairs": sum(bool(set(pair["policy_strict"]) & {"deep", "persistent"}) for pair in pairs),
        "passed": not acceptance_defects and unknown["total"] == 0,
        "policy_v1_passed": quality["passed"],
        "policy_v1_passed_known": quality["passed_known"],
    }
    pending = [p['pair_id'] for p in pairs if p.get('support_residual_pending_verification')]
    summary['unverified_support_residual_pairs'] = pending
    verdict['support_residual_verification'] = 'unverified' if pending else 'not_required'
    if pending:
        verdict.update(passed=False, policy_v1_passed=False, policy_v1_passed_known=False)
    return summary, verdict


def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    clip = args.clip_dir.resolve()
    scene_json_path = clip / "scene_metrics.json"
    scene_arrays_path = clip / "scene_metrics.npz"
    motion_path = clip / "final_retargeted.npz"
    invocation_path = clip / "invocation.json"
    for path in (scene_json_path, scene_arrays_path, motion_path, invocation_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    scene_record = json.loads(scene_json_path.read_text(encoding="utf-8"))
    with np.load(scene_arrays_path, allow_pickle=False) as scene_arrays:
        static_depth = np.asarray(scene_arrays["static_depths"], dtype=float).copy()
        if "moving_depths" in scene_arrays:
            moving_depth = np.asarray(scene_arrays["moving_depths"], dtype=float).copy()
        else:
            moving_depth = np.zeros_like(static_depth)
    with np.load(motion_path, allow_pickle=False) as motion:
        qpos = np.asarray(motion["qpos"], dtype=float).copy()
        frames_per_second = _motion_framerate(motion, args.frames_per_second)
        motion_start = int(motion.get("frame_start", 0))

    start = int(scene_record.get("start", motion_start))
    frames = len(qpos)
    labels = np.arange(frames, dtype=int) + start
    if int(scene_record.get("frames", frames)) != frames:
        raise ValueError("scene_metrics.json and final_retargeted.npz frame counts differ")
    for name, depths in (("static_depths", static_depth), ("moving_depths", moving_depth)):
        if depths.ndim != 2 or depths.shape[0] != frames:
            raise ValueError(f"{name} must be a two-dimensional (frames, model.ngeom) array")
    if moving_depth.shape != static_depth.shape:
        raise ValueError(
            "static_depths and moving_depths must have identical (frames, geom) shapes"
        )

    model_started = time.perf_counter()
    model_path = _read_profile_model_path(invocation_path)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model_seconds = time.perf_counter() - model_started
    if static_depth.shape != (frames, model.ngeom):
        raise ValueError(
            f"scene depth width {static_depth.shape[1]} != model.ngeom {model.ngeom}"
        )
    if qpos.shape != (frames, model.nq):
        raise ValueError(f"qpos shape {qpos.shape} != expected {(frames, model.nq)}")
    support_index = _load_support_index(
        clip / "prepared.npy",
        model,
        model_path,
        start=start,
        frames=frames,
    )

    moving_attribution: dict[str, Any] = {
        "mode": "aggregate",
        "reason": (
            "frozen moving maxima have no penetrations above the listing threshold"
            if moving_depth.max() <= LIST_MIN_DEPTH_M
            else "objects.npy unavailable"
        ),
    }
    moving_sequences: list[tuple[str, np.ndarray]] = []
    objects_path = clip / "objects.npy"
    should_split = (
        args.moving_detail != "aggregate"
        and moving_depth.max() > LIST_MIN_DEPTH_M
        and objects_path.is_file()
    )
    if should_split:
        objects = np.load(objects_path, allow_pickle=True).tolist()
        if objects:
            object_depths, names, moving_seconds, moving_failures = _query_moving_by_object(
                model, objects, qpos, start=start
            )
            recomputed = np.maximum.reduce(object_depths)
            difference = float(np.max(np.abs(recomputed - moving_depth), initial=0.0))
            if difference > 1e-12:
                raise ValueError(
                    "recomputed moving object depths disagree with frozen scene_metrics: "
                    f"max |difference|={difference:.12g} m"
                )
            moving_sequences = list(zip(names, object_depths))
            moving_attribution = {
                "mode": "per_object_recomputed",
                "objects_path": str(objects_path),
                "object_count": len(objects),
                "frozen_agreement_max_abs_m": difference,
                "query_seconds": moving_seconds,
                "recomputed_failure_count": moving_failures,
            }
        else:
            moving_attribution["reason"] = "objects.npy contains no objects"
    elif args.moving_detail == "objects":
        moving_attribution["reason"] = (
            "no object file or no listed aggregate penetration; aggregate depths retained"
        )

    self_pairs, self_frame_depth, self_errors, self_details = _query_self(
        model, qpos, workers=args.workers
    )
    candidates: list[tuple[str, str, Any, Any, int | None, np.ndarray]] = []
    for geom_id, sequence in enumerate(static_depth.T):
        candidates.append((f"static:geom_{geom_id}", "static", geom_id, "static_scene", None, sequence))
    if moving_sequences:
        for object_id, (name, depths) in enumerate(moving_sequences):
            for geom_id, sequence in enumerate(depths.T):
                candidates.append((f"moving:{object_id}:geom_{geom_id}", "moving", geom_id, name, object_id, sequence))
    else:
        for geom_id, sequence in enumerate(moving_depth.T):
            candidates.append((f"moving:aggregate:geom_{geom_id}", "moving", geom_id, "moving_objects", None, sequence))
    for (geom_a, geom_b), sequence in self_pairs.items():
        candidates.append((f"self:geom_{geom_a}:geom_{geom_b}", "self", geom_a, geom_b, None, sequence))
    internal_pairs = [
        {
            "key": key,
            "kind": kind,
            "robot_geom": _geom_description(model, geom_a),
            "other": _geom_description(model, int(other)) if kind == "self" else other,
            "object_index": object_index,
            "depths": depths,
        }
        for key, kind, geom_a, other, object_index, depths in candidates
        if depths.max() > LIST_MIN_DEPTH_M
    ]


    pair_records: list[dict[str, Any]] = []
    classifications: dict[str, list[str]] = {}
    policy_v1: dict[str, list[str]] = {}
    policy_strict: dict[str, list[str]] = {}
    quality_series: dict[str, dict[str, Any]] = {}
    unknown = _unknown_record(
        scene_record,
        len(self_errors),
        int(moving_attribution.get("recomputed_failure_count", 0)),
    )
    scene_unknown = bool(
        unknown["static_uncertain_samples"]
        or unknown["moving_unknown_components"]
        or unknown["moving_query_failures"]
        or unknown["moving_recomputed_failures"]
    )
    self_unknown = np.zeros(frames, dtype=bool)
    self_unknown[self_errors] = True
    for pair in internal_pairs:
        pair_unknown = (
            self_unknown
            if pair["kind"] == "self"
            else np.full(frames, scene_unknown, dtype=bool)
        )
        pair_json, classes, series = _pair_json(
            pair,
            labels=labels,
            frames_per_second=frames_per_second,
            support_index=support_index,
            scene_record=scene_record,
            unknown=pair_unknown,
        )
        pair_records.append(pair_json)
        quality_series[pair_json["pair_id"]] = series
        classifications[pair_json["pair_id"]] = classes
        policy_v1[pair_json["pair_id"]] = pair_json["policy_v1"]
        policy_strict[pair_json["pair_id"]] = pair_json["policy_strict"]
    pair_records.sort(key=lambda pair: (-pair["max_depth_m"], pair["kind"], pair["pair_id"]))

    frame_peak = np.maximum.reduce(
        [static_depth.max(axis=1), moving_depth.max(axis=1), self_frame_depth]
    )
    quality = evaluate_shared(quality_series, frames_per_second)
    # Source roles establish contact evidence, not which obstacle face caused
    # a measured penetration. Missing witness certificates cannot certify reset.
    support_unverified = np.zeros(frames, bool)
    for surface_id, depths in enumerate([static_depth, *([v for _, v in moving_sequences] or [moving_depth])]):
        for geom_id in np.flatnonzero(np.any(depths > RESET_SELF_M, axis=0)):
            mask = support_index['by_body_surface'].get(
                (int(model.geom_bodyid[geom_id]), surface_id), np.zeros(frames, bool))
            support_unverified |= mask & (depths[:, geom_id] > RESET_SELF_M) & (depths[:, geom_id] <= RESET_SCENE_M)

    reset = _reset_report(
        static_depth=static_depth.max(axis=1),
        moving_depth=moving_depth.max(axis=1),
        self_depth=self_frame_depth,
        labels=labels,
        scene_unknown=scene_unknown,
        self_error_frames=set(self_errors),
        support_unverified=support_unverified,
        lateral=_lateral_reset_mask(
            model, static_depth, moving_sequences, moving_depth, support_index
        ),
    )
    summary, verdict = _summary(
        pair_records,
        classifications,
        quality,
        frame_peak=frame_peak,
        labels=labels,
        frames_per_second=frames_per_second,
        unknown=unknown,
    )
    summary["max_depth_by_kind_m"] = {
        "static": float(static_depth.max(initial=0.0)),
        "moving": float(moving_depth.max(initial=0.0)),
        "self": float(self_frame_depth.max(initial=0.0)),
    }
    return {
        "schema_version": 1,
        "inputs": {
            "clip_dir": str(clip),
            "scene_metrics": str(scene_json_path),
            "scene_metrics_arrays": str(scene_arrays_path),
            "motion": str(motion_path),
            "invocation": str(invocation_path),
            "robot_model": str(model_path),
            "frame_start": start,
            "frames": frames,
        },
        "pairs": pair_records,
        "classification": classifications,
        "policy_classification": {
            "policy_v1": policy_v1,
            "policy_strict": policy_strict,
        },
        "support_normal_method": {
            "primary": (
                "prepared.npy surface_contact_mask AND "
                "contact_role_surface_support; match the collision geom's robot "
                "body and counterpart surface id over frames where this pair "
                "depth is positive. Static scene is surface id 0; moving "
                "object ids are one-based in these benchmark outputs."
            ),
            "per_frame_verification": (
                "source support roles are a provisional body/surface proxy only; saved depth arrays "
                "lack per-frame witness exit-normal and finite-component evidence. Support residual "
                "classification is provisional, never verified or a passed verdict; reset is unknown "
                "when normal/component evidence is required, unless another pair proves invalid."
            ),
            "source": support_index["source"],
            "reason": support_index.get("reason"),
        },
        "summary": summary,
        "reset": reset,
        "verdict": verdict,
        "self_collision": self_details,
        "model_structural_overlap": self_details.get("model_structural_overlap", {}),
        "moving_object_attribution": moving_attribution,
        "timing": {
            "model_load_seconds": model_seconds,
            "self_collision_seconds": self_details["query_seconds"],
            "moving_object_seconds": (
                moving_attribution.get("query_seconds", 0.0)
                if moving_attribution.get("mode") == "per_object_recomputed"
                else 0.0
            ),
            "total_seconds": time.perf_counter() - started,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-per-second", type=float)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, available_cpus()),
        help="self-collision query workers (default: min(8, available CPUs))",
    )
    parser.add_argument(
        "--moving-detail",
        choices=("auto", "objects", "aggregate"),
        default="auto",
        help="attribute moving penetration to objects when source objects are available",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    report = audit(args)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("SAVED", args.output, flush=True)


if __name__ == "__main__":
    main()
