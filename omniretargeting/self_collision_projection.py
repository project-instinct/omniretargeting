"""Project complete motion proposals with both endpoints available to move.

Contact blending and chronological IK are proposal generators. Their allowed
initialization residuals are not an acceptance criterion for the trajectory.
Only small, jointly validated corrections are retained here.
"""

from __future__ import annotations

import numpy as np

from .collision_window import _optimize_window
from .local_collision_repair import _event_windows, _point_corrections
from .scene_geometry import SceneQuery
from .self_collision import SelfCollisionQuery
from .trajectory_validation import QueryBudgetExceeded, TrajectoryValidator, local_repair_config


def _path_stats(query, trajectory):
    depths = np.zeros((len(trajectory), len(query.pairs)))
    records, limited = [], []
    maximum = 0.0
    tolerance = query.cfg["penetration_tolerance"]

    def record(row, frame, alpha):
        if -row["clearance"] > tolerance:
            records.append({"frame": frame, "alpha": alpha, "pair": row["pair"],
                            "body_id": row["body_ids"][0], "obstacle_id": -2})

    for frame, q in enumerate(trajectory):
        for row in query.query(q):
            depths[frame, row["pair_index"]] = max(-row["clearance"], 0.0)
            record(row, frame, 0.0)
    maximum = float(np.max(depths, initial=0))
    for frame, (first, second) in enumerate(zip(trajectory[:-1], trajectory[1:])):
        path = query.segment(first, second)
        maximum = max(maximum, max(path["depths"].values(), default=0.0))
        if path["subdivision_limit"]:
            limited.append(frame)
        for row in path["rows"]:
            if row["alpha"] < 1:
                record(row, frame, row["alpha"])
    return {"self_depth": depths, "records": records, "max_depth": maximum,
            "subdivision_limit_intervals": limited}


def project_self_collision_trajectory(model, trajectory, scene, tasks, priority,
                                      config, *, framerate=30.0, scene_query=None,
                                      body_relation_tasks=None, self_contact_tasks=None, surface_contact_tasks=None, surface_pose_tasks=None):
    """Resolve saved-frame and playback-path collisions before later stages.

    No scene work is needed for a clear trajectory. For residuals, query only
    bounded windows and their neighbouring seams, using the full trajectory's
    original contact-marker choices and source timeline. The root stays fixed.
    """
    reference = np.asarray(trajectory, dtype=float).copy()
    cfg = local_repair_config({
        **dict(config.get("local_collision_repair", {}) or {}),
        "self_collision_enabled": True, "self_collision": config.get("self_collision", {}),
    }, priority=priority,
        joint_limit_safety_fraction=float(config.get("joint_limit_safety_fraction", 1.0)))
    query = SelfCollisionQuery(model, cfg["self_collision"])
    before = _path_stats(query, reference)
    report = {"status": "passed", "before_max_depth": before["max_depth"],
              "after_max_depth": before["max_depth"], "events": [],
              "subdivision_limit_intervals": before["subdivision_limit_intervals"]}
    if not before["records"]:
        if before["subdivision_limit_intervals"]:
            report["status"] = "incomplete"
        return reference, report
    scene_query = scene_query if scene_query is not None else SceneQuery.from_tasks(
        scene, tasks, full_geometry=cfg["full_geometry"],
    )
    validator = TrajectoryValidator(model, reference, scene_query, tasks, priority, cfg,
                                    framerate, body_relation_tasks=body_relation_tasks, self_contact_tasks=self_contact_tasks, surface_contact_tasks=surface_contact_tasks, surface_pose_tasks=surface_pose_tasks)
    result = reference.copy()
    events = _event_windows(validator, before)
    used = []
    try:
        for index, event in enumerate(events):
            a, b = event["window"]
            log = {"window": [a, b], "joints": event["joints"]}
            report["events"].append(log)
            if index >= cfg["max_windows"] or any(a < y and b > x for x, y in used):
                log["reason"] = "window_budget"
                continue
            start, stop = max(0, a - 1), min(len(result), b + 2)
            with validator.segment(start, stop) as view:
                local = result[start:stop].copy()
                stats = view.geometry(local)
                local_event = dict(event, window=[a - start, b - start])
                trial, checked, optimization = _optimize_window(
                    view, local, local, local_event, stats, [],
                )
                log.update(optimization)
                log["retained_unknown_scene_frames"] = len(stats["uncertain_frames"])
                if checked is None or np.array_equal(local, trial):
                    continue
                assembled = result.copy()
                assembled[start:stop] = trial
                fidelity = validator.fidelity(assembled)
                if not fidelity["ok"]:
                    log["reason"] = "assembly_fidelity:" + ",".join(fidelity["reasons"])
                    continue
                log.update(_point_corrections(view, local, trial, local_event, stats["records"]))
                result = assembled
                used.append((a, b))
    except QueryBudgetExceeded as exc:
        report["budget_reason"] = str(exc)
    after = _path_stats(query, result)
    report.update(after_max_depth=after["max_depth"],
                  subdivision_limit_intervals=after["subdivision_limit_intervals"],
                  status="incomplete" if after["subdivision_limit_intervals"] else
                  "residual" if after["records"] else "passed",
                  fidelity=validator.fidelity(result),
                  max_joint_correction=float(np.max(np.abs(result[:, validator.qi] - reference[:, validator.qi]), initial=0)),
                  query_points=scene_query.query_points - validator.query_start,
                  geometry_calls=validator.geometry_calls)
    if not np.array_equal(result[:, validator.fixed_qi], reference[:, validator.fixed_qi]):
        raise RuntimeError("Self-collision projection modified fixed coordinates")
    return result, report
