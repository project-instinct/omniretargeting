"""Interpolation-first local joint repair and final postprocessing acceptance.

All candidates use the same immutable reference and TrajectoryValidator. Only
window interior chain joints can change; targets, root and timing remain fixed.
"""

from __future__ import annotations

from time import perf_counter

import mujoco
import numpy as np

from .collision_window import (
    _check_corridor as _check_corridor,
    _collision_score as _collision_score,
    _corridor_planes as _corridor_planes,
    _local_fidelity as _local_fidelity,
    _local_geometry as _local_geometry,
    _optimize_window as _optimize_window,
    _window_stats as _window_stats,
)
from .contact import _is_surface_only_contact_mode
from .scene_geometry import SceneQuery
from .scene_witness_guard import preserve_known_scene_samples
from .trajectory_validation import (
    QueryBudgetExceeded,
    TrajectoryValidator,
    _pose_velocity,
    _runs,
    local_repair_config,
)


def _interpolate(reference, start, end, joints, model):
    candidate = reference.copy()
    if end - start < 2:
        return candidate
    for qi in joints:
        jid = int(np.flatnonzero(model.jnt_qposadr == qi)[0])
        a, b = reference[start, qi], reference[end, qi]
        if (
            model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE
            and not model.jnt_limited[jid]
        ):
            b = (
                a
                + _pose_velocity(model, reference[start], reference[end])[
                    int(model.jnt_dofadr[jid])
                ]
            )
        # Limited hinges stay inside their mechanical chart; no unwrap/clip.
        candidate[start + 1 : end, qi] = np.linspace(a, b, end - start + 1)[1:-1]
    return candidate


def _event_windows(v, stats):
    groups = {}
    for record in stats["records"]:
        joints = v.chain(record["body_id"])
        if record["obstacle_id"] == -2:
            bodies = [int(v.model.geom_bodyid[g]) for g in record["pair"]]
            paths = [set(v.ancestor_coordinates(b)) for b in bodies]
            relative = paths[0] ^ paths[1]
            common = paths[0] & paths[1]
            joints = set(relative)
            for body, path in zip(bodies, paths):
                if path & relative:
                    joints.update(v.chain(body))
            joints = tuple(sorted(joints - common))
        key = (joints, record["obstacle_id"])
        groups.setdefault(key, set()).add(record["frame"])
        if record["alpha"]:
            groups[key].add(record["frame"] + 1)
    events = []
    max_frames = max(0, int(np.floor(v.cfg["max_window_seconds"] * v.fps)))
    extension = max(1, int(np.floor(v.cfg["max_extension_seconds"] * v.fps)))
    for (joints, obstacle), frames in sorted(groups.items()):
        focus_self = obstacle == -2
        mask = np.zeros(len(v.reference), dtype=bool)
        mask[list(frames)] = True
        for first, last in _runs(mask):
            # Split persistent collisions into bounded attempts; never extrapolate.
            if max_frames < 2:
                events.append(
                    {
                        "window": [int(first), int(first)],
                        "affected": [int(first), int(last)],
                        "joints": list(joints),
                        "obstacles": [int(obstacle)],
                        "boundary_reason": "window_duration_below_two_intervals",
                    }
                )
                continue
            for lo in range(first, last + 1, max_frames - 1):
                hi = min(last, lo + max_frames - 2)
                start, end = max(0, lo - 1), min(len(mask) - 1, hi + 1)
                if focus_self:
                    # Correction derivatives need a collar on both sides.
                    # Contact modes are enforced per frame by the optimizer;
                    # a phase change is not a reason to skip self repair.
                    padding = min(extension, max(0, (max_frames - (end - start)) // 2))
                    start, end = max(0, start - padding), min(len(mask) - 1, end + padding)
                reason = ""
                # Boundaries must be collision/limit/contact-valid and same phase.
                phase = v.phase(lo, joints)

                def normal(f, joints=joints, phase=phase):
                    if focus_self:
                        return np.max(stats["self_depth"][f], initial=0) <= v.self_query.cfg["penetration_tolerance"]
                    if v.phase(f, joints) != phase:
                        return False
                    if (
                        np.max(stats["per_geom_depth"][f], initial=0)
                        > v.cfg["penetration_tolerance"]
                        or np.max(stats["self_depth"][f], initial=0)
                        > v.cfg["penetration_tolerance"]
                    ):
                        return False
                    q = v.reference[f, v.qi]
                    if np.any(q < v.lower) or np.any(q > v.upper):
                        return False
                    for name, spec in v.specs.items():
                        task = spec["task"]
                        if not task.contact_mask[f]:
                            continue
                        p = v.original_effectors[name]["position"][f]
                        delta = p - task.frame(f)["contact_position"]
                        if _is_surface_only_contact_mode(task.frame(f)["contact_mode"]):
                            delta = np.dot(delta, task.surface_normals[f])
                        if np.linalg.norm(delta) > v.cfg["contact_tolerance"]:
                            return False
                    return True

                for _ in range(extension):
                    if (
                        normal(start)
                        or start == 0
                        or end - start >= max_frames
                        or (not focus_self and v.phase(start, joints) != phase)
                    ):
                        break
                    start -= 1
                for _ in range(extension):
                    if (
                        normal(end)
                        or end == len(mask) - 1
                        or end - start >= max_frames
                        or (not focus_self and v.phase(end, joints) != phase)
                    ):
                        break
                    end += 1
                if not normal(start) or not normal(end):
                    reason = "missing_valid_boundaries"
                if not focus_self and any(v.phase(f, joints) != phase for f in range(start, end + 1)):
                    reason = "contact_phase_switch"
                events.append(
                    {
                        "window": [start, end],
                        "affected": [lo, hi],
                        "joints": list(joints),
                        "obstacles": [int(obstacle)],
                        "boundary_reason": reason,
                        "focus": "self" if focus_self else "scene",
                    }
                )
    # Coordinate overlapping windows once, including different obstacles/limbs.
    merged = []
    for event in sorted(events, key=lambda e: (e.get("focus") != "self", e["window"])):
        if (merged and event.get("focus") == merged[-1].get("focus")
                and event["window"][0] < merged[-1]["window"][1]):
            previous = merged[-1]
            if event["window"][1] - previous["window"][0] <= max_frames:
                previous["window"][1] = max(previous["window"][1], event["window"][1])
                previous["affected"][1] = max(
                    previous["affected"][1], event["affected"][1]
                )
                previous["joints"] = sorted(set(previous["joints"] + event["joints"]))
                previous["obstacles"] = sorted(
                    set(previous["obstacles"] + event["obstacles"])
                )
                previous["boundary_reason"] = (
                    previous["boundary_reason"] or event["boundary_reason"]
                )
                continue
        merged.append(event)
    return merged


def _point_corrections(v, before, after, event, records):
    a, b = event["window"]
    horizontal, vertical = 0.0, 0.0
    if np.array_equal(before[a : b + 1], after[a : b + 1]):
        return {
            "horizontal_point_correction": horizontal,
            "vertical_point_correction": vertical,
        }
    by_frame = {}
    for record in records:
        frame = record["frame"]
        if a <= frame <= b and not record["alpha"]:
            by_frame.setdefault(frame, []).append(record)
    d0, d1 = mujoco.MjData(v.model), mujoco.MjData(v.model)
    for frame, samples in by_frame.items():
        if np.array_equal(before[frame], after[frame]):
            continue
        d0.qpos[:] = before[frame]
        d1.qpos[:] = after[frame]
        mujoco.mj_forward(v.model, d0)
        mujoco.mj_forward(v.model, d1)
        for record in samples:
            gid = record["geom"]
            point = np.asarray(record["point"])
            local = d0.geom_xmat[gid].reshape(3, 3).T @ (
                point - d0.geom_xpos[gid]
            )
            delta = (
                d1.geom_xpos[gid]
                + d1.geom_xmat[gid].reshape(3, 3) @ local
                - point
            )
            horizontal = max(horizontal, float(np.linalg.norm(delta[:2])))
            vertical = max(vertical, abs(float(delta[2])))
    return {
        "horizontal_point_correction": horizontal,
        "vertical_point_correction": vertical,
    }


def repair_local_collisions(
    model,
    trajectory,
    scene_mesh,
    tasks,
    end_effector_config,
    config,
    *,
    framerate=30.0,
    scene_query=None,
    surface_tasks=None,
    geometry_cache=None,
    body_relation_tasks=None,
    self_contact_tasks=None,
    surface_contact_tasks=None,
    surface_pose_tasks=None,
    contact_offset_choice=None,
    candidate_acceptance=None,
):
    start_time = perf_counter()
    cfg = local_repair_config(config, priority=end_effector_config)
    reference = np.asarray(trajectory, dtype=float).copy()
    report = {
        "status": "disabled" if not cfg["enabled"] else "checked",
        "events": [],
        "config": cfg,
        "validated_effectors": sorted(tasks),
        "contact_plan_available": any(
            np.any(np.isfinite(task.distances)) for task in tasks.values()
        ),
        "unclassified_effectors": sorted(
            name
            for name, task in tasks.items()
            if not np.any(np.isfinite(task.distances))
        ),
        "surface_enhancement": (surface_tasks.diagnostics if surface_tasks is not None
                                else {"status": "disabled", "reason": "no_verified_surface_bindings"}),
    }
    if not cfg["enabled"] or len(reference) == 0:
        return reference, report
    query = (
        scene_query
        if scene_query is not None
        else SceneQuery.from_tasks(scene_mesh, tasks, full_geometry=cfg["full_geometry"])
    )
    v = TrajectoryValidator(
        model, reference, query, tasks, end_effector_config, cfg, framerate, surface_tasks,
        geometry_cache=geometry_cache,
        contact_offset_choice=contact_offset_choice,
        body_relation_tasks=body_relation_tasks, self_contact_tasks=self_contact_tasks, surface_contact_tasks=surface_contact_tasks, surface_pose_tasks=surface_pose_tasks,
    )
    try:
        original = v.geometry(reference)
    except QueryBudgetExceeded as exc:
        report.update(
            status="unvalidated_budget",
            unresolved=True,
            reason=str(exc),
            seconds=perf_counter() - start_time,
            query_points=query.query_points - v.query_start,
            geometry_calls=v.geometry_calls,
            failed_stage="initial_validation",
        )
        return reference, report
    result, final_stats = reference.copy(), original
    report["before"] = original
    report["reference_fidelity"] = v.fidelity(reference)
    events = _event_windows(v, original)
    used = []
    try:
        for index, event in enumerate(events):
            event_start = perf_counter()
            event.update(method="fallback", iterations=0)
            report["events"].append(event)
            if index >= cfg["max_windows"]:
                event["reason"] = "window_count_budget"
                continue
            a, b = event["window"]
            focus_self = event.get("focus") == "self"
            if any(a < y and b > x for x, y in used):
                event["reason"] = "overlap_exceeds_window_budget"
                continue
            if not event["joints"]:
                event["reason"] = "no_local_scalar_chain"
                continue
            event_reference, event_stats = result.copy(), final_stats
            # An earlier event can modify adjacent fixed boundaries. Build the
            # candidate from the current output, while reference budgets stay fixed.
            initial = _interpolate(result, a, b, event["joints"], model)
            if focus_self:
                initial = result.copy()
            planes = _corridor_planes(v, event, original)
            event["same_side_planes"] = len(planes)
            before = _window_stats(final_stats, a, b, model)
            reasons = [event["boundary_reason"]] if event["boundary_reason"] else []
            if focus_self:
                reasons = ["self_collision_projection"]
            if not reasons:
                if candidate_acceptance is not None and not candidate_acceptance(initial):
                    reasons.append("fixed_sequence_budget")
            if not reasons:
                fidelity = _local_fidelity(v, initial, a, b)
                reasons.extend(fidelity["reasons"])
                if not _check_corridor(v, initial, event, planes):
                    reasons.append("same_side_corridor")
                if not reasons:
                    local_after = _local_geometry(v, initial, a, b)
                    reasons.extend(
                        v.compare_geometry(
                            before,
                            local_after,
                            require_improvement=True,
                            allow_existing_uncertainty=True,
                        )
                    )
                    # Interpolation is accepted only if the event meets the
                    # detector's tolerance; partial progress uses bounded SQP.
                    if (
                        _collision_score(local_after)[0]
                        > cfg["penetration_tolerance"]
                    ):
                        reasons.append("interpolation_collision")
                    if not reasons:
                        reasons.extend(preserve_known_scene_samples(v, initial, original, a, b))
                    if not reasons:
                        reasons.extend(v.fidelity(initial)["reasons"])
                        check = v.geometry(initial)
                        reasons.extend(v.compare_geometry(
                            before, _window_stats(check, a, b, model),
                            require_improvement=True, allow_existing_uncertainty=True,
                        ))
                    if not reasons:
                        result, final_stats = initial, check
                        event.update(
                            method="interpolation", reason="joint_checks_passed"
                        )
            if reasons and not focus_self and cfg["bidirectional_enabled"] and cfg["iterations"]:
                from .local_candidates import bidirectional_window
                event["interpolation_rejected"] = sorted(set(reasons))
                candidate, directional = bidirectional_window(v, result, initial, event, planes)
                event["bidirectional"] = directional
                if candidate is not None:
                    rejected = (
                        ["fixed_sequence_budget"]
                        if candidate_acceptance is not None and not candidate_acceptance(candidate)
                        else []
                    )
                    if not rejected:
                        fidelity = _local_fidelity(v, candidate, a, b)
                        local_after = _local_geometry(v, candidate, a, b)
                        rejected.extend(fidelity["reasons"] + v.compare_geometry(
                            _window_stats(final_stats, a, b, model), local_after,
                            require_improvement=True,
                            allow_existing_uncertainty=True))
                    if not rejected:
                        rejected.extend(preserve_known_scene_samples(v, candidate, original, a, b))
                    if not _check_corridor(v, candidate, event, planes):
                        rejected.append("same_side_corridor")
                    if not rejected:
                        rejected.extend(v.fidelity(candidate)["reasons"])
                        check = v.geometry(candidate)
                        rejected.extend(v.compare_geometry(
                            final_stats, check, require_improvement=True,
                            allow_existing_uncertainty=True,
                        ))
                    if not rejected:
                        result, final_stats, initial = candidate, check, candidate
                        event.update(method="bidirectional_dp", reason="joint_checks_passed")
                        reasons = []
                        trial, polished, log = _optimize_window(
                            v, result, result, event, final_stats, planes,
                            min(cfg["iterations"], cfg["dp_polish_iterations"]),
                            polish=True, candidate_acceptance=candidate_acceptance)
                        event.update(optimization=log, iterations=log["iterations"])
                        if polished is not None:
                            result, final_stats = trial, polished
                    else:
                        directional.update(status="assembly_rejected", assembly_rejections=rejected)
            if reasons:
                event["interpolation_rejected"] = sorted(set(reasons))
                extension = max(0, int(cfg["max_extension_seconds"] * framerate))
                max_span = int(cfg["max_window_seconds"] * framerate)
                ea, eb = a, b
                for _ in range(0 if focus_self else extension):
                    na, nb = max(0, ea - 1), min(len(reference) - 1, eb + 1)
                    if nb - na > max_span or (na == ea and nb == eb):
                        break
                    if any(
                        v.phase(f, event["joints"]) != v.phase(a, event["joints"])
                        for f in range(na, nb + 1)
                    ):
                        break
                    if any(na < y and nb > x for x, y in used):
                        break
                    if any(
                        np.max(original["per_geom_depth"][f], initial=0)
                        > cfg["penetration_tolerance"]
                        for f in (na, nb)
                    ):
                        break
                    ea, eb = na, nb
                can_expand = (ea, eb) != (a, b)
                first_budget = (
                    max(1, cfg["iterations"] // 2)
                    if can_expand and cfg["iterations"]
                    else cfg["iterations"]
                )
                trial, check, log = _optimize_window(
                    v, result, initial, event, final_stats, planes, first_budget,
                    candidate_acceptance=candidate_acceptance,
                )
                event.update(
                    iterations=log["iterations"],
                    optimization=log,
                    attempts=[dict(window=[a, b], **log)],
                )
                if check is not None and not np.array_equal(trial, result):
                    result, final_stats = trial, check
                    event.update(method="local_optimization", reason=log["reason"])
                else:
                    event["reason"] = log["reason"]
                remaining = cfg["iterations"] - event["iterations"]
                if (
                    can_expand
                    and remaining > 0
                    and _collision_score(_window_stats(final_stats, a, b, model))[0]
                    > cfg["penetration_tolerance"]
                ):
                    event["window"] = [ea, eb]
                    a, b = ea, eb
                    initial = _interpolate(result, a, b, event["joints"], model)
                    planes = _corridor_planes(v, event, original)
                    trial, check, extra = _optimize_window(
                        v, result, initial, event, final_stats, planes, remaining,
                        candidate_acceptance=candidate_acceptance,
                    )
                    event["attempts"].append(dict(window=[a, b], **extra))
                    event["iterations"] += extra["iterations"]
                    if check is not None and not np.array_equal(trial, result):
                        result, final_stats = trial, check
                        event.update(
                            method="local_optimization", reason=extra["reason"]
                        )
                    elif event["method"] == "fallback":
                        event["reason"] = extra["reason"]
                    before = _window_stats(original, a, b, model)
            if (event["method"] != "fallback" and candidate_acceptance is not None
                    and not candidate_acceptance(result)):
                result, final_stats = event_reference, event_stats
                event.update(method="fallback", reason="fixed_sequence_budget")
            event["before"] = before
            event["after"] = _window_stats(final_stats, a, b, model)
            event["fidelity"] = v.fidelity(result)
            event["seconds"] = perf_counter() - event_start
            if event["method"] != "fallback":
                used.append((a, b))
        # Verify the exact returned array after all windows have been composed.
        if used:
            checked = v.geometry(result)
            reasons = (
                v.compare_geometry(original, checked, allow_existing_uncertainty=True)
                + v.fidelity(result)["reasons"]
                + preserve_known_scene_samples(v, result, original)
            )
            if reasons:
                # Revalidate/revert individual windows, including adjacent seams.
                for event in reversed(report["events"]):
                    if event["method"] == "fallback":
                        continue
                    a, b = event["window"]
                    joints = event["joints"]
                    rollback = result.copy()
                    rollback[np.ix_(np.arange(a + 1, b), joints)] = reference[
                        np.ix_(np.arange(a + 1, b), joints)
                    ]
                    checked = v.geometry(rollback)
                    result, final_stats = rollback, checked
                    event.update(
                        method="fallback",
                        reason="final_composition:" + ",".join(reasons),
                    )
                    reasons = (
                        v.compare_geometry(original, checked, allow_existing_uncertainty=True)
                        + v.fidelity(result)["reasons"]
                        + preserve_known_scene_samples(v, result, original)
                    )
                    if not reasons:
                        break
            final_stats = checked
    except QueryBudgetExceeded as exc:
        # Retain only the most recent fully checked assembly. Queries never
        # silently turn an unexamined candidate into an accepted trajectory.
        report["status"] = "budget_exhausted"
        report["reason"] = str(exc)
    for event in report["events"]:
        a, b = event["window"]
        event["after"] = _window_stats(final_stats, a, b, model)
        if (
            event["method"] == "local_optimization"
            and _collision_score(event["after"], focus_self=event.get("focus") == "self")[0] > cfg["penetration_tolerance"]
        ):
            event["reason"] = "bounded_improvement_with_residual"
        event.update(
            _point_corrections(v, reference, result, event, original["records"])
        )
    report["unresolved_reasons"] = (
        (
            ["collision_residual"]
            if _collision_score(final_stats)[0] > cfg["penetration_tolerance"]
            else []
        )
        + (
            ["uncertain_scene_side"]
            if final_stats["uncertain_frames"] or final_stats["uncertain_intervals"]
            else []
        )
        + (["subdivision_budget"] if final_stats["subdivision_limit_intervals"] else [])
    )
    report.update(
        after=final_stats,
        fidelity=v.fidelity(result),
        contact_offset_choice=v.offset_choice,
        scene_query=query.describe(),
        collision_geoms=v.probe.geom_ids.tolist(),
        seconds=perf_counter() - start_time,
        query_points=query.query_points - v.query_start,
        geometry_calls=v.geometry_calls,
        unresolved=bool(
            _collision_score(final_stats)[0] > cfg["penetration_tolerance"]
            or final_stats["uncertain_frames"]
            or final_stats["uncertain_intervals"]
            or final_stats["subdivision_limit_intervals"]
        ),
    )
    editable = np.zeros(reference.shape, dtype=bool)
    for event in report["events"]:
        if event["method"] != "fallback":
            a, b = event["window"]
            editable[np.ix_(np.arange(a+1, b), event["joints"])] = True
        event["status"] = ("unresolved" if event["method"] == "fallback" else
                           "partial" if _collision_score(event["after"], focus_self=event.get("focus") == "self")[0] > cfg["penetration_tolerance"] else "resolved")
    report["invariants"] = {
        "fixed_coordinates_unchanged": np.array_equal(result[:, v.fixed_qi], reference[:, v.fixed_qi]),
        "outside_windows_and_unrelated_joints_unchanged": np.array_equal(result[~editable], reference[~editable]),
        "frame_count_unchanged": len(result) == len(reference),
        "fixed_qpos_columns": v.fixed_qi.tolist(),
    }
    if not all(report["invariants"][name] for name in (
        "fixed_coordinates_unchanged", "outside_windows_and_unrelated_joints_unchanged", "frame_count_unchanged")):
        raise RuntimeError("Local repair modified protected coordinates")
    return result, report


def select_safe_postprocessing(
    model,
    baseline,
    candidate,
    scene,
    tasks,
    priority,
    config,
    *,
    framerate=30,
    scene_query=None,
    geometry_cache=None,
    body_relation_tasks=None,
    self_contact_tasks=None,
    surface_contact_tasks=None,
    surface_pose_tasks=None,
    contact_offset_choice=None,
    candidate_acceptance=None,
):
    """Jointly accept existing refinement/postprocessing; never mix joint angles.

    Fast path accepts the complete candidate. On regression, contiguous changed
    intervals are checked in bounded blocks and individually retained/reverted.
    Every splice is checked with its neighbours through full FK and geometry.
    """
    start_time = perf_counter()
    cfg = local_repair_config(config, priority=priority)
    query = (
        scene_query if scene_query is not None else SceneQuery.from_tasks(scene, tasks, full_geometry=cfg["full_geometry"])
    )
    v = TrajectoryValidator(model, baseline, query, tasks, priority, cfg, framerate,
                            geometry_cache=geometry_cache, body_relation_tasks=body_relation_tasks, self_contact_tasks=self_contact_tasks, surface_contact_tasks=surface_contact_tasks, surface_pose_tasks=surface_pose_tasks,
                            contact_offset_choice=contact_offset_choice)
    result = np.asarray(baseline).copy()
    decisions = []
    try:
        before = v.geometry(baseline)
        fidelity = v.fidelity(candidate)
        reasons = fidelity["reasons"]
        if not reasons and candidate_acceptance is not None and not candidate_acceptance(candidate):
            reasons = ["fixed_sequence_budget"]
        if not reasons:
            after = v.geometry(candidate)
            reasons = v.compare_geometry(before, after, allow_existing_uncertainty=True)
            if not reasons:
                reasons.extend(preserve_known_scene_samples(v, candidate, before))
        if not reasons:
            return (
                candidate.copy(),
                after,
                {
                    "decisions": [],
                    "accepted_fraction": 1.0,
                    "geometry_calls": v.geometry_calls,
                    "seconds": perf_counter() - start_time,
                    "before": before,
                    "fidelity": v.fidelity(candidate),
                },
            )
        size = max(1, int(cfg["max_window_seconds"] * framerate))
        changed = np.any(candidate != baseline, axis=1)
        checked = before
        for a, b in _runs(changed):
            for start in range(a, b + 1, size):
                end = min(b + 1, start + size)
                if len(decisions) >= cfg["max_windows"]:
                    decisions.append(
                        {
                            "window": [int(start), int(end - 1)],
                            "accepted": False,
                            "reasons": ["window_count_budget"],
                        }
                    )
                    continue
                trial = result.copy()
                trial[start:end] = candidate[start:end]
                fidelity = _local_fidelity(v, trial, start, end - 1)
                failure = fidelity["reasons"]
                if not failure and candidate_acceptance is not None and not candidate_acceptance(trial):
                    failure = ["fixed_sequence_budget"]
                geometry_evaluated = not failure
                if geometry_evaluated:
                    local_after = _local_geometry(v, trial, start, end - 1)
                    failure = v.compare_geometry(
                        _window_stats(before, start, end - 1, model),
                        local_after, allow_existing_uncertainty=True,
                    )
                    if not failure:
                        failure.extend(preserve_known_scene_samples(
                            v, trial, before, max(0, start - 1), min(len(trial) - 1, end)
                        ))
                decisions.append(
                    {
                        "window": [int(start), int(end - 1)],
                        "accepted": not failure,
                        "reasons": failure,
                        "geometry_evaluated": geometry_evaluated,
                    }
                )
                if not failure:
                    result = trial
        # Window decisions are provisional. Audit the composed full trajectory
        # once, including all cross-window temporal and scene constraints.
        checked = v.geometry(result)
        final_failure = (
            v.fidelity(result)["reasons"]
            + v.compare_geometry(before, checked, allow_existing_uncertainty=True)
            + preserve_known_scene_samples(v, result, before)
        )
        if candidate_acceptance is not None and not candidate_acceptance(result):
            final_failure.append("fixed_sequence_budget")
        if final_failure:
            # Rare coupled failure: replay the accepted blocks with the original
            # full-trajectory contract, keeping every independently safe block.
            result, checked = np.asarray(baseline).copy(), before
            for decision in decisions:
                if not decision["accepted"]:
                    continue
                start, last = decision["window"]
                trial = result.copy()
                trial[start : last + 1] = candidate[start : last + 1]
                failure = v.fidelity(trial)["reasons"]
                if not failure and candidate_acceptance is not None and not candidate_acceptance(trial):
                    failure = ["fixed_sequence_budget"]
                if not failure:
                    check = v.geometry(trial)
                    failure = (
                        v.compare_geometry(before, check, allow_existing_uncertainty=True)
                        + preserve_known_scene_samples(v, trial, before)
                    )
                if failure:
                    decision.update(accepted=False, reasons=sorted(set(failure)))
                else:
                    result, checked = trial, check
        accepted = sum(
            int(np.count_nonzero(changed[start : last + 1]))
            for decision in decisions if decision["accepted"]
            for start, last in [decision["window"]]
        )
        return (
            result,
            checked,
            {
                "decisions": decisions,
                "before": before,
                "accepted_fraction": accepted / max(int(np.count_nonzero(changed)), 1),
                "seconds": perf_counter() - start_time,
                "fidelity": v.fidelity(result),
            },
        )
    except QueryBudgetExceeded as exc:
        return (
            result,
            locals().get("checked", locals().get("before", {})),
            {
                "decisions": decisions,
                "accepted_fraction": 0.0,
                "reason": str(exc),
                "status": "budget_exhausted",
                "query_points": query.query_points - v.query_start,
                "geometry_calls": v.geometry_calls,
            },
        )


def diagnostics_json(value):
    """Portable JSON, including explicit unknown values rather than NaN tokens."""
    import json

    def portable(x):
        if isinstance(x, np.ndarray):
            return portable(x.tolist())
        if isinstance(x, np.generic):
            return portable(x.item())
        if isinstance(x, dict):
            return {str(k): portable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [portable(v) for v in x]
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x

    return json.dumps(portable(value), indent=2, ensure_ascii=False, allow_nan=False)


def trajectory_digest(trajectory):
    import hashlib

    return hashlib.sha256(
        np.ascontiguousarray(trajectory, dtype=np.float64).tobytes()
    ).hexdigest()
