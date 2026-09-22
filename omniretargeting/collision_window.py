"""Bounded local joint optimization within a collision-event window."""

from __future__ import annotations

import cvxpy as cp
import mujoco
import numpy as np

from .contact import _is_surface_only_contact_mode
from .self_collision import bounded_collision_constraints
from .trajectory_validation import QueryBudgetExceeded, _pose_velocity, _summary


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
            and r["obstacle_id"] != -2
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
    focus_self = event.get("focus") == "self"
    joints = np.asarray(event["joints"], dtype=int)
    frames = np.arange(start + 1, end)
    if not len(frames) or not len(joints):
        return (
            current,
            None,
            {"iterations": 0, "reason": "no_interior_chain_variables", "slack": 0.0},
        )
    # Unknown scan sides are retained as unresolved. They need not prevent an
    # independent, bounded correction of a known intersection. Every formerly
    # known material sample is rechecked below, including intermediate poses.
    allow_unknown = bool(original_stats["uncertain_frames"] or original_stats["uncertain_intervals"])
    witness_guard = None
    if allow_unknown:
        from .scene_witness_guard import SceneWitnessGuard
        witness_guard = SceneWitnessGuard(v)
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
    trust = min(0.08, cfg["max_joint_correction"])
    for iteration in range(iteration_count):
        log["iterations"] += 1
        step = cp.Variable((len(frames), len(joints)))
        correction = iterate[np.ix_(frames, joints)] - reference + step
        full_delta = cp.vstack(
            [np.zeros((1, len(joints))), correction, np.zeros((1, len(joints)))]
        )
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
        slacks, self_slacks, linear_contacts = [], [], {}
        for row, f in enumerate(frames):
            data.qpos[:] = iterate[f]
            original_data.qpos[:] = v.reference[f]
            mujoco.mj_forward(v.model, data)
            mujoco.mj_forward(v.model, original_data)
            for weight, residual, jac in v.surface_rows(iterate[f], f, dofs):
                objective += weight * cp.sum_squares(residual + jac @ step[row])
            for weight, residual, jac in v.relation_rows(iterate[f], f, dofs):
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
                    <= min(cfg["max_collision_slack"],
                           max(-frame["clearance"][k] - cfg["penetration_tolerance"], 0)),
                ]
                slacks.append(slack)
                objective += cp.square(slack / 0.005)
            if v.self_query is not None:
                self_constraints, self_penalty = bounded_collision_constraints(
                    v.self_query, v.self_query.query(data=data), data, step[row], dofs, slack_variables=self_slacks
                )
                constraints.extend(self_constraints)
                objective += self_penalty
        inter_stats = v.geometry(iterate)
        if v.self_query is not None:
            # Include near, currently healthy pairs between saved poses too.
            # Penetration records alone omit precisely the new collisions that
            # would otherwise repeatedly reject the same nonlinear step.
            for f in range(start, end):
                path = v.self_query.segment(iterate[f], iterate[f + 1], jacobians=True)
                for collision in path["rows"]:
                    alpha = collision["alpha"]
                    if alpha >= 1:
                        continue
                    increment = 0
                    if start < f < end:
                        increment = increment + (1 - alpha) * step[f - start - 1]
                    if start < f + 1 < end:
                        increment = increment + alpha * step[f - start]
                    if isinstance(increment, int):
                        continue
                    row = dict(collision, jacobian=collision["jacobian"] / alpha)
                    halfspaces, penalty = bounded_collision_constraints(
                        v.self_query, [row], data, increment, dofs, slack_variables=self_slacks,
                    )
                    constraints.extend(halfspaces)
                    objective += penalty
        for record in inter_stats["records"]:
            f, alpha = record["frame"], record["alpha"]
            if (not alpha or not start <= f < end or record["confidence"] <= 0
                    or record.get("uncertain", False) or record["obstacle_id"] == -2):
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
                collision = {
                    "points": np.asarray(record["pair_points"]),
                    "body_ids": [int(v.model.geom_bodyid[g]) for g in record["pair"]],
                    "direction": np.asarray(record["pair_direction"]),
                }
                gradient = v.self_query.jacobian(collision, data)[dofs]
                target = v.self_query.cfg["minimum_distance"]
                self_slacks.append(slack)
            else:
                gradient = np.asarray(record["direction"]) @ jac[:, dofs]
                target = -cfg["penetration_tolerance"]
            constraints += [
                record["clearance"] + gradient @ increment
                >= target - slack,
                slack <= min(cfg["max_collision_slack"], max(target - record["clearance"], 0)),
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
        try:
            if focus_self and self_slacks:
                # Contact and correction budgets are constraints. Within
                # those budgets, first remove self penetration; posture cost
                # cannot purchase collision slack in the subsequent solve.
                level = cp.norm(cp.hstack(self_slacks), 2)
                recovery = cp.Problem(cp.Minimize(level), constraints)
                recovery.solve(solver="CLARABEL", max_iter=80,
                               canon_backend=cp.SCIPY_CANON_BACKEND)
                if recovery.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    log["reason"] = f"collision_recovery:{recovery.status}"
                    iterate = best.copy()
                    continue
                constraints.append(level <= max(float(level.value), 0.0) + 1e-6)
            problem = cp.Problem(cp.Minimize(objective), constraints)
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
        values = [float(s.value) for s in slacks + self_slacks if s.value is not None]
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
                local_before, local_after, require_improvement=not polish, focus_self=focus_self,
                allow_existing_uncertainty=allow_unknown,
            )
            if not reasons and witness_guard is not None:
                reasons.extend(witness_guard.check(trial, start, end))
                log["known_scene_witnesses"] = dict(witness_guard.report)
            # Keep only bounded, jointly admissible progress. A later iteration
            # may relinearize an improved residual; this is not a proof of infeasibility.
            def correction_cost(q):
                delta = q[:, joints] - v.reference[:, joints]
                return (v.surface_score(q) + v.relation_score(q) + np.sum((delta / cfg["max_joint_correction"])**2)
                        + np.sum((np.diff(delta, axis=0)*v.fps / cfg["correction_velocity"])**2)
                        + np.sum((np.diff(delta, n=2, axis=0)*v.fps**2 / cfg["correction_acceleration"])**2))

            improves = (
                _collision_score(local_after, focus_self=focus_self)[0] <= cfg["penetration_tolerance"]
                and correction_cost(trial) < correction_cost(best) - 1e-8
            ) if polish else _collision_score(local_after, focus_self=focus_self) < _collision_score(
                _window_stats(best_stats, start, end, v.model), focus_self=focus_self)
            if reasons or not improves:
                log["nonlinear_rejections"].append(reasons or ["no_progress"])
                continue
            best, best_stats, iterate = trial, stats, trial
            accepted = True
            break
        if (
            accepted
            and _collision_score(_window_stats(best_stats, start, end, v.model), focus_self=focus_self)[0]
            <= cfg["penetration_tolerance"]
        ):
            log["reason"] = "target_reached"
            break
        if not accepted:
            iterate = best.copy()
            trust *= 0.5
    return best, best_stats, log


def _window_stats(stats, start, end, model):
    result = dict(stats)
    result.update(_summary(stats["per_geom_depth"][start : end + 1], model))
    for key in ("interframe_depth", "interframe_self_depth"):
        result[key] = stats[key][max(0, start - 1) : end + 1]
    for key in ("reliable_scene_depth", "uncertain_geoms"):
        result[key] = stats[key][start : end + 1]
    for key in ("interframe_reliable_scene_depth", "interframe_uncertain_geoms"):
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


def _collision_score(stats, *, focus_self=False):
    keys = ("self_depth", "interframe_self_depth") if focus_self else (
        "per_geom_depth", "interframe_depth", "self_depth", "interframe_self_depth")
    values = np.concatenate(
        [
            stats[k].ravel()
            for k in keys
        ]
    )
    return float(np.max(values, initial=0)), float(np.sum(values**2))
