"""Saved-frame self audit, including separately measured structural overlaps."""
from __future__ import annotations
import math
from typing import Any
import mujoco
import numpy as np
from omniretargeting.self_collision import SelfCollisionQuery
from omniretargeting.training_collision_parallel import query_self_frames

def audit_self(
    model: mujoco.MjModel,
    trajectory: TrainingTrajectory,
    geom_ids: np.ndarray,
    workers: int = 1,
) -> tuple[dict[str, Any], SelfCollisionQuery | None]:
    from scripts.review_training_collision import (SelfCollisionQuery, query_self_frames,
        _depth_summary, _body_name, DEPTH_EPSILON_M, RESET_STATIC_FAILURE_M, RESET_DEFECT_M)
    try:
        query = SelfCollisionQuery(model)
        restored_structural_pairs = query.use_model_pairs()
        from omniretargeting.structural_overlap import StructuralOverlap
        structural = StructuralOverlap(query, len(trajectory.qpos), trajectory.labels)
    except Exception as error:
        return {
            **_depth_summary(np.zeros((len(trajectory.qpos), len(geom_ids))), model, geom_ids, trajectory.labels),
            "status": "unknown",
            "reason": f"initialization failed: {type(error).__name__}: {error}",
            "unknown_query_count": len(trajectory.qpos),
        }, None
    depths = np.zeros((len(trajectory.qpos), model.ngeom), dtype=float)
    frame_depth = np.zeros(len(trajectory.qpos), dtype=float)
    min_clearance = np.full(len(trajectory.qpos), np.inf)
    errors: list[dict[str, Any]] = []
    error_frames: set[int] = set()
    pair_depths = {}
    for frame, frame_result in enumerate(query_self_frames(query, trajectory.qpos, workers)):
        rows = structural.collect(frame_result["rows"], frame)
        error = frame_result.get("error")
        if error is not None:
            structural.unknown[frame] = True
            error_frames.add(frame)
            errors.append({"frame": int(trajectory.labels[frame]), "message": error})
            continue
        if rows:
            pair_depth = max(max(-row["clearance"], 0.0) for row in rows)
            frame_depth[frame] = pair_depth
            for row in rows:
                pair = tuple(map(int, row['pair']))
                pair_depths.setdefault(pair, np.zeros(len(trajectory.qpos)))[frame] = max(-row['clearance'], 0.)
                for gid in pair:
                    depths[frame, gid] = max(depths[frame, gid], -row['clearance'], 0.)
            min_clearance[frame] = min(float(row["clearance"]) for row in rows)
    peak = frame_depth
    max_frame = int(np.argmax(peak)) if len(peak) else 0
    from scripts.spline_collision_details import self_details
    return {
        "mean_m": float(np.mean(peak)),
        "p95_m": float(np.quantile(peak, 0.95)),
        "max_m": float(np.max(peak, initial=0.0)),
        "worst_frame": int(trajectory.labels[max_frame]) if len(trajectory.labels) else None,
        "worst_body": _body_name(model, int(np.argmax(depths[max_frame]))) if depths.size else None,
        "frames_over_1mm": np.flatnonzero(peak > query.cfg["penetration_tolerance"] + DEPTH_EPSILON_M).tolist(),
        "frames_over_5mm": np.flatnonzero(peak > RESET_STATIC_FAILURE_M + DEPTH_EPSILON_M).tolist(),
        "frames_over_1cm": np.flatnonzero(peak > RESET_DEFECT_M + DEPTH_EPSILON_M).tolist(),
        "status": "unknown" if errors else "measured",
        "unknown_query_count": len(error_frames),
        "errors": errors,
        "frame_depths_max_m": frame_depth.tolist(),
        "pair_policy": "compiled_MJCF_measurement_separate_structural_overlaps",
        "model_structural_overlap": structural.describe(),
        "frame_min_clearance_m": [None if not math.isfinite(x) else x for x in min_clearance],
        **self_details(pair_depths, model, trajectory.labels, query.cfg['penetration_tolerance']),
        **{key: value for key, value in query.describe().items() if key != 'status'},
        "query_status": query.describe()['status'],
    }, query
