"""Shared NumPy-only production quality policy for solving and auditing.

The functions deliberately accept plain arrays and mappings so the fast solver
can call them without importing MuJoCo models or performing file IO.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


THRESHOLDS_M = (0.0005, 0.002, 0.005, 0.010, 0.020)
DEEP_M = 0.010
SUPPORT_RESIDUAL_M = 0.005
MILD_BRIEF_M = 0.002

MILD_SECONDS = 0.3
MILD_FRACTION = 0.10
TWO_MM_SECONDS = 1.0
TWO_MM_FRACTION = 0.25

RESET_SCENE_M = 0.005
RESET_SELF_M = 0.001
DEPTH_EPSILON_M = 1e-6
HALO_FRAMES = 5


def _vector(name: str, value: np.ndarray, *, length: int | None = None) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if length is not None and len(result) != length:
        raise ValueError(f"{name} has length {len(result)}, expected {length}")
    return result


def _runs(mask: np.ndarray) -> list[list[int]]:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("run mask must be one-dimensional")
    if not np.any(mask):
        return []
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return [[int(start), int(stop) - 1] for start, stop in zip(edges[::2], edges[1::2])]


def _expanded(indices: np.ndarray, frames: int) -> np.ndarray:
    result = np.zeros(frames, dtype=bool)
    selected = np.asarray(indices, dtype=int)
    selected = selected[(selected >= 0) & (selected < frames)]
    if len(selected):
        low = max(0, int(selected.min()) - HALO_FRAMES)
        high = min(frames, int(selected.max()) + HALO_FRAMES + 1)
        view = result[low:high]
        for index in selected:
            view[max(0, int(index) - low - HALO_FRAMES) : min(len(view), int(index) - low + HALO_FRAMES + 1)] = True
    return result


def _threshold_key(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _mask_from_frames(frames: Any, length: int) -> np.ndarray:
    if isinstance(frames, np.ndarray) and frames.dtype == bool:
        result = np.asarray(frames, dtype=bool)
    else:
        result = np.zeros(length, dtype=bool)
        result[np.asarray(frames, dtype=int)] = True
    if result.shape != (length,):
        raise ValueError("threshold frame mask has the wrong length")
    return result


def pair_statistics(depth: np.ndarray, fps: float, unknown: np.ndarray | None = None) -> dict:
    """Compute complete threshold statistics, including unknown-query upper bounds."""
    depth = _vector("depth", depth)
    if len(depth) and (not np.isfinite(depth).all() or np.any(depth < 0)):
        raise ValueError("depth must contain finite nonnegative metres")
    fps = float(fps)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    unknown_mask = np.zeros(len(depth), dtype=bool) if unknown is None else _vector("unknown", unknown, length=len(depth)).astype(bool)
    result: dict[str, Any] = {
        "frame_count": int(len(depth)),
        "frames_per_second": fps,
        "max_depth_m": float(depth.max(initial=0.0)),
        "worst_frame": int(np.argmax(depth)) if len(depth) else None,
        "unknown_frames": np.flatnonzero(unknown_mask).tolist(),
        "unknown_frame_count": int(np.sum(unknown_mask)),
        "thresholds_m": {},
    }
    for threshold in THRESHOLDS_M:
        measured = depth > threshold
        upper = measured | unknown_mask
        measured_runs = _runs(measured)
        upper_runs = _runs(upper)
        measured_longest = max(measured_runs, key=lambda run: run[1] - run[0] + 1, default=None)
        upper_longest = max(upper_runs, key=lambda run: run[1] - run[0] + 1, default=None)
        result["thresholds_m"][_threshold_key(threshold)] = {
            "frames": np.flatnonzero(measured).tolist(),
            "frame_fraction": float(np.mean(measured)) if len(depth) else 0.0,
            "longest_run_frames": 0 if measured_longest is None else measured_longest[1] - measured_longest[0] + 1,
            "longest_run_seconds": 0.0 if measured_longest is None else (measured_longest[1] - measured_longest[0] + 1) / fps,
            "longest_run_start": None if measured_longest is None else measured_longest[0],
            "runs": measured_runs,
            "run_count": len(measured_runs),
            "upper": {
                "frame_fraction": float(np.mean(upper)) if len(depth) else 0.0,
                "longest_run_frames": 0 if upper_longest is None else upper_longest[1] - upper_longest[0] + 1,
                "longest_run_seconds": 0.0 if upper_longest is None else (upper_longest[1] - upper_longest[0] + 1) / fps,
                "runs": upper_runs,
            },
        }
    return result


def classify(stats: Mapping[str, Any], *, lateral_or_self: bool, support_frames: np.ndarray | None = None, depth: np.ndarray | None = None) -> dict:
    """Apply production policy V1 and return event masks for solver repair."""
    if not isinstance(stats, Mapping):
        raise TypeError("stats must be a mapping")
    count = int(stats.get("frame_count", 0))
    fps = float(stats.get("frames_per_second", 0.0))
    if count < 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("stats is missing frame_count or valid frames_per_second")
    support = np.zeros(count, dtype=bool) if support_frames is None else _vector("support_frames", support_frames, length=count).astype(bool)
    if depth is not None:
        depth_values = _vector("depth", depth, length=count)
    else:
        depth_values = None
    if depth_values is not None and (
        not np.isfinite(depth_values).all() or np.any(depth_values < 0)
    ):
        raise ValueError("depth must contain finite nonnegative metres")

    thresholds = stats.get("thresholds_m", {})
    if not isinstance(thresholds, Mapping):
        raise ValueError("stats thresholds_m must be a mapping")
    five = thresholds.get(_threshold_key(SUPPORT_RESIDUAL_M), {})
    two = thresholds.get(_threshold_key(MILD_BRIEF_M), {})
    over_five = _mask_from_frames(five.get("frames", ()), count)
    over_two = _mask_from_frames(two.get("frames", ()), count)
    max_depth = float(stats.get("max_depth_m", 0.0))
    support_residual = bool(
        max_depth <= SUPPORT_RESIDUAL_M and np.any(over_two) and np.all(support[over_two])
    )
    mild_persistent = (
        float(five.get("longest_run_seconds", 0.0)) > MILD_SECONDS
        or float(five.get("frame_fraction", 0.0)) > MILD_FRACTION
    )
    lateral_persistent = (
        float(two.get("longest_run_seconds", 0.0)) > TWO_MM_SECONDS
        or float(two.get("frame_fraction", 0.0)) > TWO_MM_FRACTION
    )
    deep = max_depth > DEEP_M
    persistent = bool(
        (mild_persistent and not support_residual)
        or ((lateral_or_self or not support_residual) and lateral_persistent)
    )
    classes: list[str] = []
    if deep:
        classes.append("deep")
    if persistent:
        classes.append("persistent")
    if not classes:
        if support_residual:
            classes.append("support_residual")
        elif max_depth > MILD_BRIEF_M:
            classes.append("mild_brief")
        else:
            classes.append("negligible")

    events: list[dict[str, Any]] = []
    if deep:
        over_deep = (
            np.flatnonzero(depth_values > DEEP_M)
            if depth_values is not None
            else np.asarray(thresholds.get(_threshold_key(DEEP_M), {}).get("frames", ()), dtype=int)
        )
        events.append({"eps_m": DEEP_M, "frames": _expanded(over_deep, count), "reason": "deep"})
    if persistent and mild_persistent and not support_residual:
        events.append({"eps_m": SUPPORT_RESIDUAL_M, "frames": _expanded(np.flatnonzero(over_five), count), "reason": "persistent_5mm"})
    if persistent and (lateral_or_self or not support_residual) and lateral_persistent:
        events.append({"eps_m": MILD_BRIEF_M, "frames": _expanded(np.flatnonzero(over_two), count), "reason": "persistent_2mm"})
    return {"classes": classes, "events": events}


def evaluate(series: Mapping[Any, Mapping[str, Any]], fps: float) -> dict:
    """Evaluate all collision series and summarize production policy V1."""
    if not isinstance(series, Mapping):
        raise TypeError("series must be a mapping")
    pairs: dict[Any, dict[str, Any]] = {}
    over_5mm: set[int] = set()
    over_1cm: set[int] = set()
    unknown_total = 0
    kind_maxima = {"static": 0.0, "moving": 0.0, "self": 0.0}
    for key, spec in series.items():
        if not isinstance(spec, Mapping):
            raise TypeError(f"series[{key!r}] must be a mapping")
        kind = str(spec.get("kind", ""))
        if kind not in kind_maxima:
            raise ValueError(f"series[{key!r}] has invalid kind {kind!r}")
        depth = _vector(f"series[{key!r}].depth", spec.get("depth"))
        unknown = spec.get("unknown")
        stats = pair_statistics(depth, fps, unknown)
        support = spec.get("support_frames")
        decision = classify(
            stats,
            lateral_or_self=bool(spec.get("lateral", False)) or kind == "self",
            support_frames=support,
            depth=depth,
        )
        pairs[key] = {**dict(spec), "stats": stats, **decision}
        over_5mm.update(stats["thresholds_m"][_threshold_key(SUPPORT_RESIDUAL_M)]["frames"])
        over_1cm.update(stats["thresholds_m"][_threshold_key(DEEP_M)]["frames"])
        unknown_total += int(stats["unknown_frame_count"])
        kind_maxima[kind] = max(kind_maxima[kind], float(stats["max_depth_m"]))
    counts = {
        name: sum(name in pair["classes"] for pair in pairs.values())
        for name in ("deep", "persistent", "support_residual", "mild_brief", "negligible")
    }
    defects = [key for key, pair in pairs.items() if {"deep", "persistent"} & set(pair["classes"])]
    return {
        "pairs": pairs,
        "summary": {
            "pair_count": len(pairs),
            "classification_pair_counts": counts,
            "max_depth_by_kind_m": kind_maxima,
            "over_5mm_frame_count": len(over_5mm),
            "over_1cm_frame_count": len(over_1cm),
            "unknown_frame_count": unknown_total,
        },
        "defect_pairs": defects,
        "passed": not defects and unknown_total == 0,
        "passed_known": not defects,
    }


def reset_states(static: np.ndarray, moving: np.ndarray, self_depth: np.ndarray, lateral: np.ndarray, unknown: np.ndarray) -> np.ndarray[str]:
    """Return per-frame reset labels in invalid > unknown > candidate priority."""
    static = _vector("static", static).astype(float)
    moving = _vector("moving", moving, length=len(static)).astype(float)
    self_depth = _vector("self_depth", self_depth, length=len(static)).astype(float)
    lateral = _vector("lateral", lateral, length=len(static)).astype(bool)
    unknown = _vector("unknown", unknown, length=len(static)).astype(bool)
    scene = np.maximum(static, moving)
    invalid = (
        (self_depth > RESET_SELF_M + DEPTH_EPSILON_M)
        | (scene > RESET_SCENE_M + DEPTH_EPSILON_M)
        | (lateral & (scene > RESET_SELF_M + DEPTH_EPSILON_M))
    )
    return np.where(invalid, "invalid", np.where(unknown, "unknown", "candidate"))


CHECKPOINT_RESOLUTION = (1e-3, 1e-3, 1e-3, 0., .1, .002, .002, 0., 0.)


def checkpoint_quantized_key(key, *, anchor=None) -> tuple:
    """Fixed zero-origin bins give a transitive, resolution-limited ordering.

    Raw metrics and audit thresholds remain unchanged. Counts and F0 stay exact.
    Values straddling a fixed bin boundary can differ by less than its width.
    """
    if len(key) != len(CHECKPOINT_RESOLUTION) or not np.isfinite(key).all():
        raise ValueError("Checkpoint key must contain nine finite coordinates")
    if anchor is not None:
        if len(anchor) != len(key) or not np.isfinite(anchor).all():
            raise ValueError('Checkpoint anchor must contain nine finite coordinates')
        # The processed initializer defines one immutable origin. Improvements
        # smaller than the stated resolution cannot win by straddling zero-
        # origin bins. Never rebase bins to a later candidate (non-transitive).
        return tuple(int(np.trunc((float(value)-float(base))/scale)) if scale else value
                     for value, base, scale in zip(key, anchor, CHECKPOINT_RESOLUTION))
    return tuple(int(np.floor(float(value) / scale + 1e-10)) if scale else value
                 for value, scale in zip(key, CHECKPOINT_RESOLUTION))


def checkpoint_compare(a, b) -> int:
    first, second = checkpoint_quantized_key(a), checkpoint_quantized_key(b)
    return -1 if first < second else 1 if first > second else 0


def checkpoint_key(quality: Mapping[str, Any], *, velocity_excess: float, chatter_excess: float, support_excess: float, control_excess: float, posture_excess: float, base_cost: float) -> tuple:
    """Construct the V1 §7 lexicographic solver checkpoint key."""
    if not isinstance(quality, Mapping):
        raise TypeError("quality must be a mapping")
    pairs = quality.get("pairs", {})
    if not isinstance(pairs, Mapping):
        raise ValueError("quality.pairs must be a mapping")
    deep_excess = 0.0
    deep_count = 0
    persistent_severity = 0.0
    five_key = _threshold_key(SUPPORT_RESIDUAL_M)
    two_key = _threshold_key(MILD_BRIEF_M)
    for pair in pairs.values():
        stats = pair.get("stats", pair)
        thresholds = stats.get("thresholds_m", {})
        five = thresholds.get(five_key, {})
        two = thresholds.get(two_key, {})
        max_depth = float(stats.get("max_depth_m", 0.0))
        deep_excess += max(max_depth - DEEP_M, 0.0)
        classes = set(pair.get("classes", ()))
        deep_count += int("deep" in classes)
        ratios = (
            float(five.get("longest_run_seconds", 0.0)) / MILD_SECONDS,
            float(five.get("frame_fraction", 0.0)) / MILD_FRACTION,
            float(two.get("longest_run_seconds", 0.0)) / TWO_MM_SECONDS,
            float(two.get("frame_fraction", 0.0)) / TWO_MM_FRACTION,
        )
        if "persistent" in classes:
            persistent_severity += max(max(ratios) - 1.0, 0.0)
    return (
        float(velocity_excess),
        float(chatter_excess),
        float(deep_excess),
        int(deep_count),
        float(persistent_severity),
        float(support_excess),
        float(control_excess),
        float(posture_excess),
        float(base_cost),
    )
