"""Bounded bidirectional candidates and edge-checked local selection.

Independent algorithm implementation inspired by UMR's solve_sequence and
select_bidirectional_dp (8c4db7db); see docs/umr_scene_interaction.md.
Directions only change warm starts. Contact tasks, marker choices, root poses,
and frame indices are frozen by the parent TrajectoryValidator.
"""

from __future__ import annotations

import cvxpy as cp
import mujoco
import numpy as np

from .contact import _is_surface_only_contact_mode
from .geometry_query import relative_point_jacobian
from .trajectory_validation import QueryBudgetExceeded


def _depth(stats):
    return max(
        float(np.max(stats[key], initial=0))
        for key in (
            "per_geom_depth",
            "interframe_depth",
            "self_depth",
            "interframe_self_depth",
        )
    )


def _evaluate(v, poses, start):
    """The same nonlinear definition for all nodes/edges, after solving."""
    with v.segment(start, start + len(poses)) as view:
        fidelity = view.fidelity(poses)
        if not fidelity["ok"]:
            return np.inf, fidelity["reasons"], None
        stats = view.geometry(poses)
        reasons = view.compare_geometry(stats, stats)
        if _depth(stats) > v.cfg["penetration_tolerance"] + 1e-8:
            reasons.append("collision_budget")
        # These are absolute, common residuals. No directional solver cost,
        # active-row count, q_prev smoothness, or linearized residual is reused.
        delta = poses[:, v.qi] - view.reference[:, v.qi]
        score = float(np.sum((delta / v.cfg["max_joint_correction"]) ** 2))
        current = view.effectors(poses)
        for name, spec in view.specs.items():
            for f in range(len(poses)):
                payload = spec["task"].frame(f)
                p = current[name]["position"][f]
                if payload["contact"]:
                    residual = p - payload["contact_position"]
                    if _is_surface_only_contact_mode(payload["contact_mode"]):
                        normal = np.asarray(payload["surface_normal"])
                        residual = np.dot(
                            residual, normal / max(np.linalg.norm(normal), 1e-12)
                        )
                    score += float(
                        np.sum(residual**2) / v.cfg["contact_tolerance"] ** 2
                    )
                else:
                    # Preserve the existing robot-side body-relative hand pose;
                    # world-tracking feet also score their immutable world target.
                    target = (
                        payload["tracking_position"]
                        if spec.get("preserve_world_target", "foot" in name)
                        else view.original_effectors[name]["position"][f]
                    )
                    score += float(
                        np.sum((p - target) ** 2) / v.cfg["swing_position_budget"] ** 2
                    )
                if payload["contact"] and spec.get("track_orientation", False):
                    axes = (
                        ["up"]
                        if _is_surface_only_contact_mode(payload["contact_mode"])
                        else ["up", "forward"]
                    )
                    for axis in axes:
                        target = getattr(spec["task"], axis + "_directions")
                        if target is not None:
                            score += float(
                                np.sum((current[name][axis][f] - target[f]) ** 2)
                            )
        score += (_depth(stats) / v.cfg["penetration_tolerance"]) ** 2
        score += view.surface_score(poses)
        return (np.inf if reasons else score), sorted(set(reasons)), stats


def _solve_pose(v, frame, warm, previous_delta, joints, planes):
    """Small SQP around a directional warm start; no temporal projection.

    Velocity/acceleration of the added correction are checked in chronological
    order on DP edges/triples. A backward q_prev is never a forward velocity.
    """
    cfg = v.cfg
    cols = np.array([np.flatnonzero(v.qi == qi)[0] for qi in joints])
    dofs = v.di[cols]
    original = v.reference[frame]
    pose = original.copy()
    pose[joints] = warm[joints]
    data = mujoco.MjData(v.model)
    refdata = mujoco.MjData(v.model)
    refdata.qpos[:] = original
    mujoco.mj_forward(v.model, refdata)
    log = {
        "frame": int(frame),
        "iterations": 0,
        "slack": 0.0,
        "status": "iteration_budget",
    }
    for _ in range(cfg["directional_iterations"]):
        log["iterations"] += 1
        data.qpos[:] = pose
        mujoco.mj_forward(v.model, data)
        point_limit = cfg["max_query_points"]
        if (
            point_limit is not None
            and v.query.query_points - v.query_start + v.probe.query_work_bound
            > point_limit
        ):
            raise QueryBudgetExceeded("directional scene query budget")
        collision = v.probe.probe_trajectory(pose[None])[0]
        if point_limit is not None and v.query.query_points - v.query_start > point_limit:
            raise QueryBudgetExceeded("directional scene query budget")
        if np.any(collision["uncertain"]):
            log["status"] = "uncertain_geometry"
            break
        step = cp.Variable(len(joints))
        correction = pose[joints] - original[joints] + step
        constraints = [
            cp.abs(step) <= min(0.08, cfg["max_joint_correction"]),
            cp.abs(correction) <= cfg["max_joint_correction"],
        ]
        for k, col in enumerate(cols):
            if np.isfinite(v.lower[col]):
                constraints.append(pose[joints[k]] + step[k] >= v.lower[col])
            if np.isfinite(v.upper[col]):
                constraints.append(pose[joints[k]] + step[k] <= v.upper[col])
        # Correction differences preserve deliberate source motion. The prior
        # comes from the adjacent solved frame, in either traversal direction.
        objective = cp.sum_squares(correction / cfg["max_joint_correction"])
        objective += 0.25 * cp.sum_squares(
            (correction - previous_delta) / cfg["max_joint_correction"]
        )
        for weight, residual, jac in v.surface_rows(pose, frame, dofs):
            objective += weight * cp.sum_squares(residual + jac @ step)
        for name, spec in v.specs.items():
            payload = spec["task"].frame(frame)
            point = v.point(data, name, frame)
            p0 = v.original_effectors[name]["position"][frame]
            jac = np.zeros((3, v.model.nv))
            mujoco.mj_jac(v.model, data, jac, None, point, spec["body_id"])
            predicted = point + jac[:, dofs] @ step
            if payload["contact"]:
                normal = np.asarray(payload["surface_normal"], dtype=float)
                normal = normal / max(np.linalg.norm(normal), 1e-12)
                projector = (
                    np.outer(normal, normal)
                    if _is_surface_only_contact_mode(payload["contact_mode"])
                    else np.eye(3)
                )
                target = payload["contact_position"]
                old_error = np.linalg.norm(projector @ (p0 - target))
                constraints += [
                    cp.norm(projector @ (predicted - target))
                    <= max(old_error, cfg["contact_tolerance"])
                    + cfg["contact_regression"],
                    cp.norm(projector @ (predicted - p0))
                    <= v.contact_change_budget(old_error),
                ]
            else:
                constraints.append(
                    cp.norm(predicted - p0) <= v.swing_budget(name, frame)
                )
        for gid, normal in planes:
            point = data.geom_xpos[gid]
            jac = relative_point_jacobian(
                v.model, data, point, v.model.geom_bodyid[gid], normal
            )
            constraints.append(
                normal @ (point - refdata.geom_xpos[gid]) + jac[dofs] @ step >= 0
            )
        rows = []
        for k in np.flatnonzero(collision["clearance"] < 0.01):
            row = relative_point_jacobian(
                v.model,
                data,
                collision["points"][k],
                collision["body_ids"][k],
                collision["directions"][k],
            )
            rows.append((float(collision["clearance"][k]), row[dofs]))
        if cfg["self_collision_enabled"]:
            for contact in data.contact:
                if contact.dist >= 0.01:
                    continue
                g1, g2 = contact.geom
                row = relative_point_jacobian(
                    v.model,
                    data,
                    contact.pos,
                    v.model.geom_bodyid[g2],
                    contact.frame[:3],
                    contact.pos,
                    v.model.geom_bodyid[g1],
                )
                rows.append((float(contact.dist), row[dofs]))
        slacks = []
        for phi, gradient in rows:
            slack = cp.Variable(nonneg=True)
            constraints += [
                phi + gradient @ step >= -slack,
                slack
                <= min(
                    cfg["max_collision_slack"],
                    max(-phi, 0) + cfg["collision_regression"],
                ),
            ]
            objective += cp.square(slack / 0.005)
            slacks.append(slack)
        try:
            cp.Problem(cp.Minimize(objective), constraints).solve(
                solver="CLARABEL", max_iter=80
            )
        except cp.error.SolverError as exc:
            log["status"] = f"solver_error:{exc}"
            break
        if step.value is None:
            log["status"] = "infeasible_linear_subproblem"
            break
        log["slack"] = max(
            log["slack"], max((float(s.value) for s in slacks), default=0.0)
        )
        pose[joints] += step.value
        # Recompute nonlinear node admissibility after every update. Inadmissible
        # iterates may be relinearized but never become accepted DP nodes.
        score, reasons, _ = _evaluate(v, pose[None], frame)
        log["node_rejections"] = reasons
        if np.isfinite(score):
            log["status"] = "admissible_node"
            break
    return pose, log


def select_candidate_path(v, candidates, start, joints, planes=()):
    """Second-order DP with forbidden edges and correction acceleration bounds.

    candidates has shape (frames, states, nq). Endpoint states must already be
    fixed by the caller. Returns no path if any hard node/edge budget fails.
    """
    n, count, _ = candidates.shape
    if count > 4:
        raise ValueError("At most original/interpolated/forward/backward candidates")
    log = {
        "node_rejections": [],
        "edge_rejections": [],
        "evaluations": 0,
        "max_candidates": count,
        "method": "adaptive_discrete",
        "motion_spacing": v.cfg["motion_spacing"],
        "max_substeps": v.cfg["max_substeps"],
    }
    node_costs = np.full((n, count), np.inf)
    cache = {}

    def evaluate(poses, frame):
        key = (frame, poses.tobytes())
        if key not in cache:
            if log["evaluations"] >= v.cfg["dp_max_evaluations"]:
                raise QueryBudgetExceeded("DP node/edge evaluation budget")
            log["evaluations"] += 1
            cache[key] = _evaluate(v, poses, frame)
        return cache[key]

    for t in range(n):
        for k in range(count):
            score, reasons, _ = evaluate(candidates[t, k : k + 1], start + t)
            # Same-side restrictions also apply to interpolation/original nodes.
            if np.isfinite(score) and planes:
                data, refdata = mujoco.MjData(v.model), mujoco.MjData(v.model)
                data.qpos[:] = candidates[t, k]
                refdata.qpos[:] = v.reference[start + t]
                mujoco.mj_forward(v.model, data)
                mujoco.mj_forward(v.model, refdata)
                if any(
                    normal @ (data.geom_xpos[gid] - refdata.geom_xpos[gid]) < -1e-6
                    for gid, normal in planes
                ):
                    score, reasons = np.inf, ["same_side_corridor"]
            node_costs[t, k] = score
            if reasons:
                log["node_rejections"].append(
                    {"frame": start + t, "state": k, "reasons": reasons}
                )
    correction = candidates[:, :, joints] - v.reference[start : start + n, None, joints]
    states = {
        (None, k): (node_costs[0, k], [k])
        for k in range(count)
        if np.isfinite(node_costs[0, k])
    }
    for t in range(1, n):
        next_states = {}
        for (h, j), (cost, path) in states.items():
            for k in range(count):
                if not np.isfinite(node_costs[t, k]):
                    continue
                edge = np.stack([candidates[t - 1, j], candidates[t, k]])
                edge_score, reasons, _ = evaluate(edge, start + t - 1)
                if not np.isfinite(edge_score):
                    record = {
                        "frame": start + t - 1,
                        "states": [j, k],
                        "reasons": reasons,
                    }
                    if record not in log["edge_rejections"]:
                        log["edge_rejections"].append(record)
                    continue
                if h is not None:
                    acceleration = (
                        correction[t, k]
                        - 2 * correction[t - 1, j]
                        + correction[t - 2, h]
                    ) * v.fps**2
                    if (
                        np.max(np.abs(acceleration), initial=0)
                        > v.cfg["correction_acceleration"] + 1e-8
                    ):
                        log["edge_rejections"].append(
                            {
                                "frame": start + t - 1,
                                "states": [h, j, k],
                                "reasons": ["correction_acceleration"],
                            }
                        )
                        continue
                velocity = (correction[t, k] - correction[t - 1, j]) * v.fps
                transition = float(
                    np.sum((velocity / v.cfg["correction_velocity"]) ** 2)
                ) + 0.01 * (j != k)
                total = cost + node_costs[t, k] + transition
                if total < next_states.get((j, k), (np.inf, None))[0]:
                    next_states[j, k] = total, path + [k]
        states = next_states
        if not states:
            log["status"] = "no_admissible_path"
            return None, log
    if not states:
        log["status"] = "no_admissible_nodes"
        return None, log
    cost, path = min(states.values(), key=lambda value: (value[0], value[1]))
    log.update(status="selected", states=path, score=float(cost), node_costs=node_costs)
    return candidates[np.arange(n), path].copy(), log


def bidirectional_window(v, current, interpolated, event, planes):
    a, b = event["window"]
    joints = np.asarray(event["joints"], dtype=int)
    log = {"directions": [], "status": "missing_valid_boundaries"}
    if event["boundary_reason"] or b - a < 2:
        return None, log
    candidates = [current[a : b + 1].copy(), interpolated[a : b + 1].copy()]
    for direction in (1, -1):
        sequence = current[a : b + 1].copy()
        previous = a if direction == 1 else b
        frames = range(a + 1, b) if direction == 1 else range(b - 1, a, -1)
        logs = []
        for frame in frames:
            warm = v.reference[frame].copy()
            # Warm with the adjacent correction plus the original chronological
            # displacement. This preserves source velocity in either direction.
            previous_delta = (
                sequence[previous - a, joints] - v.reference[previous, joints]
            )
            warm[joints] += previous_delta
            if previous in (a, b):
                warm[joints] = sequence[previous - a, joints]
            pose, entry = _solve_pose(v, frame, warm, previous_delta, joints, planes)
            sequence[frame - a, joints] = pose[joints]
            logs.append(entry)
            previous = frame
        log["directions"].append(
            {"direction": "forward" if direction == 1 else "backward", "frames": logs}
        )
        candidates.append(sequence)
    poses, selection = select_candidate_path(
        v, np.stack(candidates, axis=1), a, joints, planes
    )
    log.update(selection=selection, status=selection["status"])
    if poses is None:
        return None, log
    result = current.copy()
    result[np.ix_(np.arange(a + 1, b), joints)] = poses[1:-1, joints]
    return result, log
