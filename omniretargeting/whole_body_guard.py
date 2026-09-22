"""Fixed-baseline protection for coupled articulated whole-body correction."""

import cvxpy as cp
import mujoco
import numpy as np
from scipy import sparse

from ._refinement.guard import RefinementGuard
from .contact import _is_surface_only_contact_mode
from .moving_scene import interpolate_object, moving_regression_reasons
from .scene_witness_guard import preserve_known_scene_samples
from .trajectory_validation import TrajectoryValidator, _pose_velocity


def interpolated_pose(model, poses, frame, alpha):
    pose = poses[frame].copy()
    if alpha:
        mujoco.mj_integratePos(
            model, pose, _pose_velocity(model, pose, poses[frame + 1]), alpha
        )
    return pose


class WholeBodyValidator(TrajectoryValidator):
    """Keep exact marker identities and protect sliding tangent as well as normal."""

    def __init__(self, *args, whole_body_config, **kwargs):
        self.whole_cfg = whole_body_config
        self.object_surfaces = ()
        super().__init__(*args, **kwargs)

    def swing_budget(self, name, frame):
        return self.whole_cfg[
            "foot_position_budget" if "foot" in name else "free_position_budget"
        ]

    def _point_reasons(self, name, payload, position, original):
        reasons = []
        budget = self.whole_cfg["contact_regression"]
        if (
            "foot" in name
            and np.linalg.norm(position - original)
            > self.whole_cfg["foot_position_budget"] + 1e-8
        ):
            reasons.append(f"whole_body_foot_position:{name}")
        if payload["contact"]:
            target = payload["contact_position"]
            projectors = [np.eye(3)]
            if _is_surface_only_contact_mode(payload["contact_mode"]):
                normal = np.array(payload["surface_normal"], dtype=float, copy=True)
                normal /= max(np.linalg.norm(normal), 1e-12)
                projectors = [normal[None], np.eye(3) - np.outer(normal, normal)]
            for projector in projectors:
                old = np.linalg.norm(projector @ (original - target))
                new = np.linalg.norm(projector @ (position - target))
                if new > old + budget + 1e-8:
                    reasons.append(f"whole_body_contact:{name}")
        return reasons

    def fidelity(self, candidate):
        report = super().fidelity(candidate)
        if not np.isfinite(candidate).all():
            return report
        values = self.effectors(candidate)
        for name, spec in self.specs.items():
            for f in range(len(candidate)):
                report["reasons"].extend(
                    self._point_reasons(
                        name,
                        spec["task"].frame(f),
                        values[name]["position"][f],
                        self.original_effectors[name]["position"][f],
                    )
                )
        correction = candidate[:, self.qi] - self.reference[:, self.qi]
        jerk = float(
            np.max(np.abs(np.diff(correction, n=3, axis=0)), initial=0) * self.fps**3
        )
        if jerk > self.whole_cfg["correction_jerk"] + 1e-7:
            report["reasons"].append("whole_body_correction_jerk")
        report["ok"] = not report["reasons"]
        report["reasons"] = sorted(set(report["reasons"]))
        return report

    def interframe_contact(self, trajectory, frame, alpha, q):
        reasons, maximum = [], 0.0
        original = interpolated_pose(self.model, self.reference, frame, alpha)
        a, b = mujoco.MjData(self.model), mujoco.MjData(self.model)
        for data, pose in ((a, original), (b, q)):
            data.qpos[:] = pose
            mujoco.mj_forward(self.model, data)
        for name, spec in self.specs.items():
            payload = dict(spec["task"].frame(frame))
            following = spec["task"].frame(frame + 1)
            if (
                payload["contact"]
                and following["contact"]
                and payload["environment_id"] == following["environment_id"]
            ):
                # At subframes the comparison uses a shared authored target;
                # stored task arrays and contact material identities stay fixed.
                payload["contact_position"] = (1 - alpha) * payload[
                    "contact_position"
                ] + alpha * following["contact_position"]
                oid = payload["environment_id"] - 1
                if 0 <= oid < len(self.object_surfaces):
                    surface = self.object_surfaces[oid]
                    task = spec["task"]
                    t0, r0, s0 = interpolate_object(surface, self.frame_start + frame)
                    t1, r1, s1 = interpolate_object(
                        surface, self.frame_start + frame + 1
                    )
                    t, r, s = interpolate_object(
                        surface, self.frame_start + frame + alpha
                    )
                    local0 = (task.surface_points[frame] - t0) @ r0 / s0
                    local1 = (task.surface_points[frame + 1] - t1) @ r1 / s1
                    offset0 = (
                        task.contact_positions[frame] - task.surface_points[frame]
                    ) @ r0
                    offset1 = (
                        task.contact_positions[frame + 1]
                        - task.surface_points[frame + 1]
                    ) @ r1
                    payload["contact_position"] = (
                        ((1 - alpha) * local0 + alpha * local1) @ r.T * s
                        + t
                        + ((1 - alpha) * offset0 + alpha * offset1) @ r.T
                    )
                    payload["surface_normal"] = r @ r0.T @ task.surface_normals[frame]
            point, original_point = (
                self.point(b, name, frame),
                self.point(a, name, frame),
            )
            reasons.extend(self._point_reasons(name, payload, point, original_point))
            change = np.linalg.norm(point - original_point)
            maximum = max(maximum, float(change))
            if (
                not payload["contact"]
                and change > self.swing_budget(name, frame) + 1e-8
            ):
                reasons.append(f"swing_budget:{name}")
            if payload["contact"] and spec.get("track_orientation", False):
                for axis in (
                    ("up",)
                    if _is_surface_only_contact_mode(payload["contact_mode"])
                    else ("up", "forward")
                ):
                    local = spec.get(
                        "orientation_local_" + axis,
                        [0, 0, 1] if axis == "up" else [1, 0, 0],
                    )
                    first = a.xmat[spec["body_id"]].reshape(3, 3) @ local
                    second = b.xmat[spec["body_id"]].reshape(3, 3) @ local
                    cosine = (
                        first
                        @ second
                        / max(np.linalg.norm(first) * np.linalg.norm(second), 1e-12)
                    )
                    if np.arccos(np.clip(cosine, -1, 1)) > 0.02 + 1e-8:
                        reasons.append(f"contact_orientation:{name}")
        return sorted(set(reasons)), maximum


class WholeBodyGuard(RefinementGuard):
    """One convex subproblem; nonlinear FK and both geometries decide acceptance."""

    def __init__(self, validator, moving_query=None):
        super().__init__(validator)
        self.moving = moving_query
        self.collision_rows = []
        self.last_moving = None
        self.last_geometry = None

    def _strict_endpoints(self, current, step):
        v, constraints = self.validator, []
        data = mujoco.MjData(v.model)
        for f, pose in enumerate(current):
            data.qpos[:] = pose
            mujoco.mj_forward(v.model, data)
            for name, spec in v.specs.items():
                payload = spec["task"].frame(f)
                point = v.point(data, name, f)
                original = v.original_effectors[name]["position"][f]
                jac = np.zeros((3, v.model.nv))
                mujoco.mj_jac(v.model, data, jac, None, point, spec["body_id"])
                linear = point + jac[:, v.di] @ step[f]
                if "foot" in name:
                    constraints.append(
                        cp.norm(linear - original)
                        <= v.whole_cfg["foot_position_budget"]
                    )
                if not payload["contact"]:
                    continue
                projectors = [np.eye(3)]
                if _is_surface_only_contact_mode(payload["contact_mode"]):
                    n = np.array(payload["surface_normal"], dtype=float, copy=True)
                    n /= max(np.linalg.norm(n), 1e-12)
                    projectors = [n[None], np.eye(3) - np.outer(n, n)]
                target = payload["contact_position"]
                for projector in projectors:
                    budget = (
                        np.linalg.norm(projector @ (original - target))
                        + v.whole_cfg["contact_regression"]
                    )
                    constraints.append(cp.norm(projector @ (linear - target)) <= budget)
        return constraints

    def solve(self, system, rhs, current, active_coordinates):
        v, cfg = self.validator, self.validator.whole_cfg
        frames, joints = len(current), len(v.qi)
        step = cp.Variable((frames, joints))
        flat = cp.vec(step, order="C")
        correction = current[:, v.qi] - v.reference[:, v.qi] + step
        constraints = [
            cp.abs(step) <= cfg["trust_region"],
            cp.abs(correction) <= cfg["max_joint_correction"],
        ]
        inactive = ~np.asarray(active_coordinates, dtype=bool)
        if np.any(inactive):
            constraints.append(step[:, inactive] == 0)
        for bounds, lower in ((v.lower, True), (v.upper, False)):
            finite = np.isfinite(bounds)
            if np.any(finite):
                q = current[:, v.qi[finite]] + step[:, finite]
                constraints.append(
                    q >= bounds[finite] if lower else q <= bounds[finite]
                )
        for order, key in (
            (1, "correction_velocity"),
            (2, "correction_acceleration"),
            (3, "correction_jerk"),
        ):
            if frames > order:
                constraints.append(
                    cp.abs(cp.diff(correction, k=order, axis=0))
                    <= cfg[key] / v.fps**order
                )
        for task in (v.self_contact_tasks, v.surface_contact_tasks):
            if task is None:
                continue
            old = task.errors(v.reference, v.frame_start)
            for f, pose in enumerate(current):
                for index, residual, jac in task.position_rows(
                    pose, v.frame_start + f, v.di
                ):
                    budget = (
                        max(old[f, index], task.cfg["tolerance"])
                        + task.cfg["regression"]
                    )
                    constraints.append(cp.norm(residual + jac @ step[f]) <= budget)
        constraints.extend(self._endpoint_constraints(current, step))
        constraints.extend(self._strict_endpoints(current, step))
        constraints.extend(self._scene_constraints(current, flat))
        constraints.extend(self._self_constraints(current, step))
        objective = cp.sum_squares(system @ flat - rhs)
        active_rows = [
            (f, alpha, r)
            for f, alpha, r in self.collision_rows
            if np.linalg.norm(r["jacobian"]) > 1e-10
        ]
        if active_rows:
            # Nonnegative residual slack permits incompatible geometry. Hard
            # endpoint constraints cannot be relaxed by increasing this weight.
            row_indices, columns, values, clearances = [], [], [], []
            groups, group_indices = {}, []
            for index, (f, alpha, row) in enumerate(active_rows):
                key = (f, bool(alpha), row["object_id"], row["geom_id"], row["kind"])
                group_indices.append(groups.setdefault(key, len(groups)))
                jac = row["jacobian"]
                nonzero = np.flatnonzero(np.abs(jac) > 1e-14)
                for frame, fraction in ((f, 1 - alpha), (f + 1, alpha)):
                    if fraction:
                        row_indices.extend([index] * len(nonzero))
                        columns.extend(frame * joints + nonzero)
                        values.extend(fraction * jac[nonzero])
                clearances.append(row["clearance"])
            matrix = sparse.coo_matrix(
                (values, (row_indices, columns)),
                shape=(len(active_rows), frames * joints),
            ).tocsr()
            residual = -np.asarray(clearances) - matrix @ flat
            # One worst-depth slack per body/object/time interval. Sample or
            # facet counts cannot arbitrarily multiply collision priority.
            collision_slack = cp.Variable(len(groups), nonneg=True)
            constraints.append(residual <= collision_slack[np.asarray(group_indices)])
            objective += cfg["collision_weight"] * cp.sum_squares(
                collision_slack / cfg["collision_sigma"]
            )
        problem = cp.Problem(cp.Minimize(objective), constraints)
        try:
            problem.solve(
                solver="CLARABEL", max_iter=100, canon_backend=cp.SCIPY_CANON_BACKEND
            )
        except cp.error.SolverError as exc:
            return None, {"status": "solver_failed", "reason": str(exc)}
        report = {
            "status": str(problem.status),
            "linearized_collision_rows": len(active_rows),
        }
        if (
            problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
            or step.value is None
        ):
            return None, report
        if active_rows:
            slack = np.maximum(
                -np.asarray(clearances) - matrix @ np.asarray(step.value).ravel(), 0
            )
            report.update(
                linearized_collision_slack_max_m=float(slack.max(initial=0)),
                linearized_collision_slack_positive_rows=int(
                    np.count_nonzero(slack > 1e-8)
                ),
                collision_slack_groups=len(groups),
            )
        return np.asarray(step.value), report

    def _self_constraints(self, current, step):
        v, constraints = self.validator, []
        query = v.self_query
        if query is None:
            return constraints
        before = self._baseline_geometry()

        def append(row, gradient, frame, alpha=0.0):
            key = "interframe_self_depth" if alpha else "self_depth"
            old = before[key][frame, row["pair_index"]]
            tolerance = query.cfg["penetration_tolerance"]
            allowance = (
                tolerance
                if old <= tolerance
                else old + query.cfg["regression_tolerance"]
            )
            change = (1 - alpha) * step[frame]
            if alpha:
                change = change + alpha * step[frame + 1]
            constraints.append(row["clearance"] + gradient[v.di] @ change >= -allowance)

        for f, pose in enumerate(current):
            for row in query.query(pose):
                append(row, query.jacobian(row, query.data), f)
            if f + 1 < len(current):
                segment = query.segment(pose, current[f + 1], jacobians=True)
                for row in segment["rows"]:
                    alpha = row["alpha"]
                    if 0 < alpha < 1:
                        append(row, row["jacobian"] / alpha, f, alpha)
        return constraints

    def moving_check(self, candidate):
        if self.moving is None:
            return [], None
        v, query = self.validator, self.moving
        reasons, count = [], 0
        summaries = {
            key: 0.0
            for key in (
                "forward_max_depth",
                "reverse_max_depth",
                "interframe_forward_max_depth",
                "interframe_reverse_max_depth",
            )
        }
        per_geom = {
            key: np.zeros((len(query.objects), v.model.ngeom))
            for key in (
                "forward",
                "reverse",
                "interframe_forward",
                "interframe_reverse",
            )
        }
        for f in range(len(candidate)):
            steps = 1
            if f + 1 < len(candidate):
                steps = max(
                    query.interval_steps(
                        q[f],
                        q[f + 1],
                        v.frame_start + f,
                        v.cfg["motion_spacing"],
                        v.cfg["max_substeps"],
                    )
                    for q in (v.reference, candidate)
                )
            for index in range(steps):
                alpha = index / steps
                old = interpolated_pose(v.model, v.reference, f, alpha)
                new = interpolated_pose(v.model, candidate, f, alpha)
                time = v.frame_start + f + alpha
                before = query.cached(old, time)
                after = (
                    before if np.array_equal(old, new) else query.query(new, time, [])
                )
                reasons.extend(
                    moving_regression_reasons(
                        before,
                        after,
                        v.whole_cfg["penetration_tolerance"],
                        v.whole_cfg["collision_regression"],
                    )
                )
                prefix = "interframe_" if alpha else ""
                for kind in ("forward", "reverse"):
                    key = prefix + kind + "_max_depth"
                    summaries[key] = max(
                        summaries[key], float(np.max(after[kind + "_depth"], initial=0))
                    )
                    per_geom[prefix + kind] = np.maximum(
                        per_geom[prefix + kind], after[kind + "_depth"]
                    )
                if alpha:
                    contact_reasons, _ = v.interframe_contact(candidate, f, alpha, new)
                    reasons.extend(contact_reasons)
                count += 1
                if reasons:
                    return sorted(set(reasons)), {
                        **summaries,
                        "checked_poses": count,
                        "complete": False,
                    }
        return [], {
            **summaries,
            "checked_poses": count,
            "complete": True,
            "unknown_components": query.unknown_components,
            "max_depth_by_object_geom": {
                key: value.tolist() for key, value in per_geom.items()
            },
        }

    def accept(self, current, change, merit):
        v = self.validator
        before = self._baseline_geometry()
        old_cost = merit(current)
        record = {"accepted_alpha": 0.0, "rejections": []}
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
            trial = current.copy()
            trial[:, v.qi] += alpha * change
            fidelity = v.fidelity(trial)
            reasons = list(fidelity["reasons"])
            if not reasons:
                cost = merit(trial)
                if cost >= old_cost - max(1e-10, abs(old_cost) * 1e-10):
                    reasons.append("nonlinear_objective_not_improved")
            if not reasons:
                after = v.geometry(trial)
                reasons.extend(
                    v.compare_geometry(before, after, allow_existing_uncertainty=True)
                )
                if not reasons:
                    reasons.extend(preserve_known_scene_samples(v, trial, before))
            if not reasons:
                reasons, moving = self.moving_check(trial)
            if not reasons:
                record.update(
                    accepted_alpha=alpha,
                    objective_before=old_cost,
                    objective_after=cost,
                    fidelity=fidelity["metrics"],
                    moving_collision=moving,
                )
                self.last_moving = moving
                self.last_geometry = after
                return trial, record
            record["rejections"].append(
                {"alpha": alpha, "reasons": sorted(set(reasons))}
            )
        return current.copy(), record
