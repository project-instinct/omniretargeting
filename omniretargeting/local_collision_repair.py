"""Interpolation-first local joint repair and final postprocessing acceptance.

All candidates use the same immutable reference and TrajectoryValidator. Only
window interior chain joints can change; targets, root and timing remain fixed.
"""

from __future__ import annotations

from time import perf_counter

import cvxpy as cp
import mujoco
import numpy as np

from .contact import _is_surface_only_contact_mode
from .scene_geometry import SceneQuery
from .trajectory_validation import (
    QueryBudgetExceeded,
    TrajectoryValidator,
    _pose_velocity,
    _runs,
    _summary,
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
        key = (v.chain(record["body_id"]), record["obstacle_id"])
        groups.setdefault(key, set()).add(record["frame"])
        if record["alpha"]:
            groups[key].add(record["frame"] + 1)
    events = []
    max_frames = max(0, int(np.floor(v.cfg["max_window_seconds"] * v.fps)))
    extension = max(1, int(np.floor(v.cfg["max_extension_seconds"] * v.fps)))
    for (joints, obstacle), frames in sorted(groups.items()):
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
                reason = ""
                # Boundaries must be collision/limit/contact-valid and same phase.
                phase = v.phase(lo, joints)

                def normal(f, joints=joints, phase=phase):
                    if v.phase(f, joints) != phase:
                        return False
                    if f in stats["uncertain_frames"]:
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
                        or v.phase(start, joints) != phase
                    ):
                        break
                    start -= 1
                for _ in range(extension):
                    if (
                        normal(end)
                        or end == len(mask) - 1
                        or end - start >= max_frames
                        or v.phase(end, joints) != phase
                    ):
                        break
                    end += 1
                if not normal(start) or not normal(end):
                    reason = "missing_valid_boundaries"
                if any(v.phase(f, joints) != phase for f in range(start, end + 1)):
                    reason = "contact_phase_switch"
                events.append(
                    {
                        "window": [start, end],
                        "affected": [lo, hi],
                        "joints": list(joints),
                        "obstacles": [int(obstacle)],
                        "boundary_reason": reason,
                    }
                )
    # Coordinate overlapping windows once, including different obstacles/limbs.
    merged = []
    for event in sorted(events, key=lambda e: e["window"]):
        if merged and event["window"][0] < merged[-1]["window"][1]:
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


def _window_stats(stats, start, end, model):
    result = dict(stats)
    result.update(_summary(stats["per_geom_depth"][start : end + 1], model))
    for key in ("interframe_depth", "interframe_self_depth"):
        result[key] = stats[key][max(0, start - 1) : end + 1]
    result["self_depth"] = stats["self_depth"][start : end + 1]
    result["interframe_contact_failures"] = [
        failure
        for failure in stats["interframe_contact_failures"]
        if max(0, start - 1) <= failure["frame"] <= end
    ]
    result["interframe_max_depth"] = float(
        np.max(result["interframe_depth"], initial=0)
    )
    for key in (
        "uncertain_frames",
        "uncertain_intervals",
        "subdivision_limit_intervals",
    ):
        result[key] = [f for f in stats[key] if max(0, start - 1) <= f <= end]
    return result


def _collision_score(stats):
    values = np.concatenate(
        [
            stats[k].ravel()
            for k in (
                "per_geom_depth",
                "interframe_depth",
                "self_depth",
                "interframe_self_depth",
            )
        ]
    )
    return float(np.max(values, initial=0)), float(np.sum(values**2))


def _corridor_planes(v, event, original):
    """Fixed same-side hints from reliable boundary geom centres.

    These are halfspaces on *added* motion, not an alternate scene SDF. At a
    corner the full sum of the boundary exit vectors retains both components.
    Opposite boundary sides provide no local channel and are reported.
    """
    a, b = event["window"]
    data = mujoco.MjData(v.model)
    gids = sorted(
        {
            r["geom"]
            for r in original["records"]
            if a <= r["frame"] <= b and r["obstacle_id"] in event["obstacles"]
        }
    )
    centers = []
    for f in (a, b):
        data.qpos[:] = v.reference[f]
        mujoco.mj_forward(v.model, data)
        centers.append(data.geom_xpos[gids].copy())
    queries = [v.query.query(points) for points in centers]
    planes = []
    for k, gid in enumerate(gids):
        if any(
            q["uncertain"][k]
            or not np.isfinite(q["clearance"][k])
            or q["clearance"][k] <= 0
            for q in queries
        ):
            continue
        if np.dot(queries[0]["directions"][k], queries[1]["directions"][k]) < -0.1:
            continue
        normal = queries[0]["directions"][k] + queries[1]["directions"][k]
        normal /= max(np.linalg.norm(normal), 1e-9)
        planes.append((gid, normal))
    return planes


def _check_corridor(v, candidate, event, planes):
    data, original_data = mujoco.MjData(v.model), mujoco.MjData(v.model)
    start, end = event["window"]
    for f in range(start + 1, end):
        data.qpos[:] = candidate[f]
        original_data.qpos[:] = v.reference[f]
        mujoco.mj_forward(v.model, data)
        mujoco.mj_forward(v.model, original_data)
        for gid, normal in planes:
            if (
                np.dot(data.geom_xpos[gid] - original_data.geom_xpos[gid], normal)
                < -1e-6
            ):
                return False
    return True


def _optimize_window(
    v, current, initial, event, original_stats, planes, iteration_budget=None,
    *, polish=False,
):
    start, end = event["window"]
    joints = np.asarray(event["joints"], dtype=int)
    frames = np.arange(start + 1, end)
    if not len(frames) or not len(joints):
        return (
            current,
            None,
            {"iterations": 0, "reason": "no_interior_chain_variables", "slack": 0.0},
        )
    # Acceptance includes both endpoints, which this solve cannot modify.
    # An uncertain fixed endpoint cannot become verified through interior SQP.
    # The caller may still try a permitted expansion with different endpoints.
    uncertain_endpoints = sorted(
        {start, end}.intersection(original_stats["uncertain_frames"])
    )
    if uncertain_endpoints:
        return current, None, {
            "iterations": 0, "reason": "uncertain_fixed_boundary", "slack": 0.0,
            "uncertain_fixed_frames": uncertain_endpoints,
        }
    columns = np.array([np.flatnonzero(v.qi == qi)[0] for qi in joints])
    dofs = v.di[columns]
    reference = v.reference[np.ix_(frames, joints)]
    best, best_stats = current.copy(), original_stats
    iterate = initial.copy()
    log = {
        "iterations": 0,
        "reason": "iteration_budget",
        "slack": 0.0,
        "slack_sum_peak": 0.0,
        "immovable_collision_rows": 0,
        "nonlinear_rejections": [],
    }
    cfg = v.cfg
    local_before = _window_stats(original_stats, start, end, v.model)
    data, original_data = mujoco.MjData(v.model), mujoco.MjData(v.model)
    # First exhaust a tight endpoint budget; then allow the explicit free-foot
    # robot-side budget. Contacts keep the same bounds in both passes.
    iteration_count = int(
        cfg["iterations"] if iteration_budget is None else iteration_budget
    )
    for iteration in range(iteration_count):
        log["iterations"] += 1
        step = cp.Variable((len(frames), len(joints)))
        correction = iterate[np.ix_(frames, joints)] - reference + step
        full_delta = cp.vstack(
            [np.zeros((1, len(joints))), correction, np.zeros((1, len(joints)))]
        )
        trust = min(0.08, cfg["max_joint_correction"])
        constraints = [
            step <= trust,
            step >= -trust,
            correction <= cfg["max_joint_correction"],
            correction >= -cfg["max_joint_correction"],
        ]
        lower, upper = v.lower[columns], v.upper[columns]
        finite = np.isfinite(lower)
        if np.any(finite):
            constraints.append(
                reference[:, finite] + correction[:, finite] >= lower[finite]
            )
        finite = np.isfinite(upper)
        if np.any(finite):
            constraints.append(
                reference[:, finite] + correction[:, finite] <= upper[finite]
            )
        diff = (full_delta[1:] - full_delta[:-1]) * v.fps
        constraints += [
            diff <= cfg["correction_velocity"],
            diff >= -cfg["correction_velocity"],
        ]
        acceleration = (diff[1:] - diff[:-1]) * v.fps
        constraints += [
            acceleration <= cfg["correction_acceleration"],
            acceleration >= -cfg["correction_acceleration"],
        ]
        objective = cp.sum_squares(
            correction / cfg["max_joint_correction"]
        ) + cp.sum_squares(diff / cfg["correction_velocity"])
        objective += cp.sum_squares(acceleration / cfg["correction_acceleration"])
        estimated = len(frames) * v.probe.query_work_bound
        point_limit = cfg["max_query_points"]
        if point_limit is not None and v.query.query_points - v.query_start + estimated > point_limit:
            raise QueryBudgetExceeded("scene query point budget before SQP")
        collision_frames = v.probe.probe_trajectory(iterate[frames])
        if point_limit is not None and v.query.query_points - v.query_start > point_limit:
            raise QueryBudgetExceeded("scene query point budget in SQP")
        slacks, linear_contacts = [], {}
        for row, f in enumerate(frames):
            data.qpos[:] = iterate[f]
            original_data.qpos[:] = v.reference[f]
            mujoco.mj_forward(v.model, data)
            mujoco.mj_forward(v.model, original_data)
            for weight, residual, jac in v.surface_rows(iterate[f], f, dofs):
                objective += weight * cp.sum_squares(residual + jac @ step[row])
            for name, spec in v.specs.items():
                payload = spec["task"].frame(f)
                point = v.point(data, name, f)
                p0 = v.original_effectors[name]["position"][f]
                jac = np.zeros((3, v.model.nv))
                mujoco.mj_jac(v.model, data, jac, None, point, spec["body_id"])
                p = point + jac[:, dofs] @ step[row]
                normal = np.array(payload["surface_normal"], dtype=float, copy=True)
                normal /= max(np.linalg.norm(normal), 1e-12)
                projector = (
                    np.outer(normal, normal)
                    if payload["contact"]
                    and _is_surface_only_contact_mode(payload["contact_mode"])
                    else np.eye(3)
                )
                if payload["contact"]:
                    target = payload["contact_position"]
                    budget = (
                        max(
                            float(np.linalg.norm(projector @ (p0 - target))),
                            cfg["contact_tolerance"],
                        )
                        + cfg["contact_regression"]
                    )
                    constraints += [
                        cp.norm(projector @ (p - target)) <= budget,
                        cp.norm(projector @ (p - p0))
                        <= v.contact_change_budget(
                            float(np.linalg.norm(projector @ (p0 - target)))
                        ),
                    ]
                    if payload["contact_mode"] == "sticking":
                        linear_contacts[name, int(f)] = p - p0
                else:
                    budget = (
                        min(v.swing_budget(name, f), 0.001)
                        if iteration < max(1, iteration_count // 2)
                        else v.swing_budget(name, f)
                    )
                    constraints.append(cp.norm(p - p0) <= budget)
            for gid, normal in planes:
                point = data.geom_xpos[gid]
                jac = np.zeros((3, v.model.nv))
                mujoco.mj_jac(
                    v.model, data, jac, None, point, int(v.model.geom_bodyid[gid])
                )
                constraints.append(
                    normal
                    @ (point - original_data.geom_xpos[gid] + jac[:, dofs] @ step[row])
                    >= 0
                )
            frame = collision_frames[row]
            indices = np.flatnonzero((frame["clearance"] < 0.01) & ~frame["uncertain"])
            for k in indices:
                jac = np.zeros((3, v.model.nv))
                mujoco.mj_jac(
                    v.model,
                    data,
                    jac,
                    None,
                    frame["points"][k],
                    int(frame["body_ids"][k]),
                )
                gradient = frame["directions"][k] @ jac[:, dofs]
                if not np.any(np.abs(gradient) > 1e-12):
                    log["immovable_collision_rows"] += 1
                slack = cp.Variable(nonneg=True)
                # Explicit bounded slack: infeasible scene constraints are never
                # described as hard nonpenetration, even with a large penalty.
                constraints += [
                    frame["clearance"][k] + gradient @ step[row]
                    >= -cfg["penetration_tolerance"] - slack,
                    slack
                    <= min(cfg["max_collision_slack"], max(-frame["clearance"][k], 0) + cfg["collision_regression"]),
                ]
                slacks.append(slack)
                objective += cp.square(slack / 0.005)
            if cfg["self_collision_enabled"]:
                for contact in data.contact:
                    if contact.dist >= 0.01:
                        continue
                    jacobians = []
                    for gid in contact.geom:
                        jac = np.zeros((3, v.model.nv))
                        mujoco.mj_jac(
                            v.model,
                            data,
                            jac,
                            None,
                            contact.pos,
                            int(v.model.geom_bodyid[gid]),
                        )
                        jacobians.append(jac[:, dofs])
                    gradient = contact.frame[:3] @ (jacobians[1] - jacobians[0])
                    slack = cp.Variable(nonneg=True)
                    constraints += [
                        contact.dist + gradient @ step[row]
                        >= -cfg["penetration_tolerance"] - slack,
                        slack <= min(cfg["max_collision_slack"], max(-contact.dist, 0) + cfg["collision_regression"]),
                    ]
                    slacks.append(slack)
                    objective += cp.square(slack / 0.005)
        inter_stats = v.geometry(iterate)
        for record in inter_stats["records"]:
            f, alpha = record["frame"], record["alpha"]
            if not alpha or not start <= f < end or record["confidence"] <= 0:
                continue
            dq = _pose_velocity(v.model, iterate[f], iterate[f + 1])
            q = iterate[f].copy()
            mujoco.mj_integratePos(v.model, q, dq, alpha)
            data.qpos[:] = q
            mujoco.mj_forward(v.model, data)
            jac = np.zeros((3, v.model.nv))
            mujoco.mj_jac(
                v.model, data, jac, None, np.asarray(record["point"]), record["body_id"]
            )
            increment = 0
            if start < f < end:
                increment = increment + (1 - alpha) * step[f - start - 1]
            if start < f + 1 < end:
                increment = increment + alpha * step[f - start]
            if isinstance(increment, int):
                continue
            slack = cp.Variable(nonneg=True)
            if record["obstacle_id"] == -2:
                pair_jacobians = []
                for gid in record["pair"]:
                    pair_jac = np.zeros((3, v.model.nv))
                    mujoco.mj_jac(
                        v.model,
                        data,
                        pair_jac,
                        None,
                        np.asarray(record["point"]),
                        int(v.model.geom_bodyid[gid]),
                    )
                    pair_jacobians.append(pair_jac[:, dofs])
                gradient = np.asarray(record["direction"]) @ (
                    pair_jacobians[1] - pair_jacobians[0]
                )
            else:
                gradient = np.asarray(record["direction"]) @ jac[:, dofs]
            constraints += [
                record["clearance"] + gradient @ increment
                >= -cfg["penetration_tolerance"] - slack,
                slack <= min(cfg["max_collision_slack"], record["depth"] + cfg["collision_regression"]),
            ]
            slacks.append(slack)
            objective += cp.square(slack / 0.005)
        for (name, f), pdelta in linear_contacts.items():
            previous = linear_contacts.get((name, f - 1))
            if previous is not None:
                constraints.append(
                    cp.norm((pdelta - previous) * v.fps)
                    <= cfg["support_slip_regression"]
                )
        problem = cp.Problem(cp.Minimize(objective), constraints)
        try:
            problem.solve(
                solver="CLARABEL", max_iter=80, canon_backend=cp.SCIPY_CANON_BACKEND
            )
        except cp.error.SolverError as exc:
            log["reason"] = f"solver_error:{exc}"
            continue
        if step.value is None:
            log["reason"] = f"linear_subproblem:{problem.status}"
            # Restore a feasible original start for the next bounded attempt.
            iterate = best.copy()
            continue
        values = [float(s.value) for s in slacks if s.value is not None]
        log["slack"] = max(log["slack"], max(values, default=0.0))
        log["slack_sum_peak"] = max(log["slack_sum_peak"], sum(values))
        accepted = False
        for alpha in (1.0, 0.5, 0.25, 0.125):
            trial = iterate.copy()
            trial[np.ix_(frames, joints)] += alpha * step.value
            fidelity = v.fidelity(trial)
            if not fidelity["ok"] or not _check_corridor(v, trial, event, planes):
                log["nonlinear_rejections"].append(
                    fidelity["reasons"] or ["same_side_corridor"]
                )
                continue
            stats = v.geometry(trial)
            local_after = _window_stats(stats, start, end, v.model)
            reasons = v.compare_geometry(
                local_before, local_after, require_improvement=not polish
            )
            # Keep only bounded, jointly admissible progress. A later iteration
            # may relinearize an improved residual; this is not a proof of infeasibility.
            def correction_cost(q):
                delta = q[:, joints] - v.reference[:, joints]
                return (v.surface_score(q) + np.sum((delta / cfg["max_joint_correction"])**2)
                        + np.sum((np.diff(delta, axis=0)*v.fps / cfg["correction_velocity"])**2)
                        + np.sum((np.diff(delta, n=2, axis=0)*v.fps**2 / cfg["correction_acceleration"])**2))

            improves = (
                _collision_score(local_after)[0] <= cfg["penetration_tolerance"]
                and correction_cost(trial) < correction_cost(best) - 1e-8
            ) if polish else _collision_score(local_after) < _collision_score(
                _window_stats(best_stats, start, end, v.model))
            if reasons or not improves:
                log["nonlinear_rejections"].append(reasons or ["no_progress"])
                continue
            best, best_stats, iterate = trial, stats, trial
            accepted = True
            break
        if (
            accepted
            and _collision_score(_window_stats(best_stats, start, end, v.model))[0]
            <= cfg["penetration_tolerance"]
        ):
            log["reason"] = "target_reached"
            break
        if not accepted:
            iterate = best.copy()
    return best, best_stats, log


def _point_corrections(v, before, after, event, records):
    a, b = event["window"]
    horizontal, vertical = 0.0, 0.0
    d0, d1 = mujoco.MjData(v.model), mujoco.MjData(v.model)
    for record in records:
        f, gid = record["frame"], record["geom"]
        if not a <= f <= b or record["alpha"]:
            continue
        d0.qpos[:] = before[f]
        d1.qpos[:] = after[f]
        mujoco.mj_forward(v.model, d0)
        mujoco.mj_forward(v.model, d1)
        local = d0.geom_xmat[gid].reshape(3, 3).T @ (
            np.asarray(record["point"]) - d0.geom_xpos[gid]
        )
        delta = (
            d1.geom_xpos[gid]
            + d1.geom_xmat[gid].reshape(3, 3) @ local
            - np.asarray(record["point"])
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
            event.update(method="fallback", iterations=0)
            report["events"].append(event)
            if index >= cfg["max_windows"]:
                event["reason"] = "window_count_budget"
                continue
            a, b = event["window"]
            if any(a < y and b > x for x, y in used):
                event["reason"] = "overlap_exceeds_window_budget"
                continue
            if not event["joints"]:
                event["reason"] = "no_local_scalar_chain"
                continue
            # An earlier event can modify adjacent fixed boundaries. Build the
            # candidate from the current output, while reference budgets stay fixed.
            initial = _interpolate(result, a, b, event["joints"], model)
            planes = _corridor_planes(v, event, original)
            event["same_side_planes"] = len(planes)
            before = _window_stats(final_stats, a, b, model)
            reasons = [event["boundary_reason"]] if event["boundary_reason"] else []
            if not reasons:
                fidelity = v.fidelity(initial)
                reasons.extend(fidelity["reasons"])
                if not _check_corridor(v, initial, event, planes):
                    reasons.append("same_side_corridor")
                if not reasons:
                    check = v.geometry(initial)
                    reasons.extend(
                        v.compare_geometry(
                            before,
                            _window_stats(check, a, b, model),
                            require_improvement=True,
                        )
                    )
                    # Interpolation is accepted only if the event meets the
                    # detector's tolerance; partial progress uses bounded SQP.
                    if (
                        _collision_score(_window_stats(check, a, b, model))[0]
                        > cfg["penetration_tolerance"]
                    ):
                        reasons.append("interpolation_collision")
                    if not reasons:
                        result, final_stats = initial, check
                        event.update(
                            method="interpolation", reason="joint_checks_passed"
                        )
            if reasons and cfg["bidirectional_enabled"] and cfg["iterations"]:
                from .local_candidates import bidirectional_window
                event["interpolation_rejected"] = sorted(set(reasons))
                candidate, directional = bidirectional_window(v, result, initial, event, planes)
                event["bidirectional"] = directional
                if candidate is not None:
                    fidelity = v.fidelity(candidate)
                    check = v.geometry(candidate)
                    rejected = fidelity["reasons"] + v.compare_geometry(
                        final_stats, check, require_improvement=True)
                    if not _check_corridor(v, candidate, event, planes):
                        rejected.append("same_side_corridor")
                    if not rejected:
                        result, final_stats, initial = candidate, check, candidate
                        event.update(method="bidirectional_dp", reason="joint_checks_passed")
                        reasons = []
                        trial, polished, log = _optimize_window(
                            v, result, result, event, final_stats, planes,
                            min(cfg["iterations"], cfg["dp_polish_iterations"]), polish=True)
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
                for _ in range(extension):
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
                        or f in original["uncertain_frames"]
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
                    v, result, initial, event, final_stats, planes, first_budget
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
                        v, result, initial, event, final_stats, planes, remaining
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
            event["before"] = before
            event["after"] = _window_stats(final_stats, a, b, model)
            event["fidelity"] = v.fidelity(result)
            event.update(
                _point_corrections(v, reference, result, event, original["records"])
            )
            if event["method"] != "fallback":
                used.append((a, b))
        # Verify the exact returned array after all windows have been composed.
        if used:
            checked = v.geometry(result)
            reasons = (
                v.compare_geometry(original, checked) + v.fidelity(result)["reasons"]
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
                        v.compare_geometry(original, checked)
                        + v.fidelity(result)["reasons"]
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
            and _collision_score(event["after"])[0] > cfg["penetration_tolerance"]
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
                           "partial" if _collision_score(event["after"])[0] > cfg["penetration_tolerance"] else "resolved")
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
                            geometry_cache=geometry_cache)
    result = np.asarray(baseline).copy()
    decisions = []
    try:
        before = v.geometry(baseline)
        fidelity = v.fidelity(candidate)
        reasons = fidelity["reasons"]
        if not reasons:
            after = v.geometry(candidate)
            reasons = v.compare_geometry(before, after)
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
        accepted = 0
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
                fidelity = v.fidelity(trial)
                failure = fidelity["reasons"]
                geometry_evaluated = not failure
                if geometry_evaluated:
                    check = v.geometry(trial)
                    failure = v.compare_geometry(before, check)
                decisions.append(
                    {
                        "window": [int(start), int(end - 1)],
                        "accepted": not failure,
                        "reasons": failure,
                        "geometry_evaluated": geometry_evaluated,
                    }
                )
                if not failure:
                    result, checked = trial, check
                    accepted += int(np.count_nonzero(changed[start:end]))
        # ``checked`` is the exact last accepted assembly; no modification follows.
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
