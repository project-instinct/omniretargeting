"""Constrain refinement to the final validator's immutable motion reference.

CVXPY/CLARABEL solves the existing sparse least-squares objective with endpoint
and correction budgets. Real FK and the shared dense/interframe scene checks
validate each accepted update; a linearized constraint is never a certificate.
"""

from __future__ import annotations

from types import SimpleNamespace

import cvxpy as cp
import mujoco
import numpy as np
from scipy import sparse

from ..contact import _is_surface_only_contact_mode
from ..scene_witness_guard import preserve_known_scene_samples
from ..trajectory_validation import QueryBudgetExceeded
from .math import _skew_symmetric


class RefinementGuard:
    def __init__(self, validator):
        self.validator = validator
        self.before = None
        self.report = {"status": "ready", "iterations": []}

    def _baseline_geometry(self):
        if self.before is None:
            v = self.validator
            fidelity = v.fidelity(v.reference)
            if not fidelity["ok"]:
                raise ValueError(
                    f"Refinement baseline fails fidelity: {fidelity['reasons']}"
                )
            self.before = v.geometry(v.reference)
            self.report["geometry_policy"] = "nonregression_with_existing_uncertainty"
            self.report["baseline_uncertain_frames"] = len(
                self.before["uncertain_frames"]
            )
            self.report["baseline_uncertain_intervals"] = len(
                self.before["uncertain_intervals"]
            )
        return self.before

    def _endpoint_constraints(self, current, step):
        v, constraints = self.validator, []
        data = mujoco.MjData(v.model)
        for name, spec in v.specs.items():
            task = spec["task"]
            points, jacobians, rotations, rotation_jacobians = [], [], [], []
            for f, q in enumerate(current):
                data.qpos[:] = q
                mujoco.mj_forward(v.model, data)
                point = v.point(data, name, f)
                jp, jr = np.zeros((3, v.model.nv)), np.zeros((3, v.model.nv))
                mujoco.mj_jac(v.model, data, jp, jr, point, spec["body_id"])
                points.append(point)
                jacobians.append(jp[:, v.di])
                rotations.append(data.xmat[spec["body_id"]].reshape(3, 3).copy())
                rotation_jacobians.append(jr[:, v.di])
            p0 = v.original_effectors[name]["position"]
            linear = np.asarray(points) + cp.reshape(
                sparse.block_diag(jacobians, format="csr") @ cp.vec(step, order="C"),
                (len(current), 3),
                order="C",
            )
            sticking = []
            for f in range(len(current)):
                payload = task.frame(f)
                if not payload["contact"]:
                    budget = v.swing_budget(name, f)
                    if not np.isfinite(budget):
                        continue
                    constraints.append(cp.norm(linear[f] - p0[f]) <= budget)
                    if spec.get("preserve_world_target", "foot" in name):
                        target = payload["tracking_position"]
                        constraints.append(
                            cp.norm(linear[f] - target)
                            <= np.linalg.norm(p0[f] - target) + budget
                        )
                    continue
                projector = np.eye(3)
                surface_only = _is_surface_only_contact_mode(payload["contact_mode"])
                if surface_only:
                    n = np.asarray(payload["surface_normal"], dtype=float)
                    n = n / max(np.linalg.norm(n), 1e-12)
                    projector = n[None, :]
                target = payload["contact_position"]
                old_error = np.linalg.norm(projector @ (p0[f] - target))
                constraints += [
                    cp.norm(projector @ (linear[f] - target))
                    <= max(old_error, v.cfg["contact_tolerance"])
                    + v.cfg["contact_regression"],
                    cp.norm(projector @ (linear[f] - p0[f]))
                    <= v.contact_change_budget(old_error),
                ]
                if payload["contact_mode"] == "sticking":
                    if (sticking and sticking[-1] == f - 1
                            and payload["environment_id"] == task.frame(f - 1)["environment_id"]):
                        constraints.append(
                            cp.norm(
                                (linear[f] - p0[f] - linear[f - 1] + p0[f - 1]) * v.fps
                            )
                            <= v.cfg["support_slip_regression"]
                        )
                    sticking.append(f)
                if not spec.get("track_orientation", False):
                    continue
                for axis in ["up"] if surface_only else ["up", "forward"]:
                    targets = getattr(task, f"{axis}_directions")
                    if targets is None:
                        continue
                    local = spec.get(
                        "orientation_local_" + axis,
                        [0, 0, 1] if axis == "up" else [1, 0, 0],
                    )
                    direction = rotations[f] @ local
                    direction /= max(np.linalg.norm(direction), 1e-12)
                    original = np.asarray(
                        v.original_effectors[name][axis][f], dtype=float
                    )
                    original = original / max(np.linalg.norm(original), 1e-12)
                    target_axis = np.asarray(targets[f], dtype=float)
                    target_axis = target_axis / max(np.linalg.norm(target_axis), 1e-12)
                    linear_axis = (
                        direction
                        - _skew_symmetric(direction) @ rotation_jacobians[f] @ step[f]
                    )
                    angle = np.arccos(np.clip(original @ target_axis, -1, 1))
                    # Unit-vector chord length corresponding to the validator's
                    # angle budget. Nonlinear angle checks still run below.
                    constraints += [
                        cp.norm(linear_axis - original) <= 2 * np.sin(0.02 / 2),
                        cp.norm(linear_axis - target_axis)
                        <= 2 * np.sin(min(angle + 0.02, np.pi) / 2),
                    ]
        return constraints

    def _scene_constraints(self, current, flat_step):
        v = self.validator
        before = self._baseline_geometry()
        limit = v.cfg["max_query_points"]
        work = len(current) * v.probe.query_work_bound
        if limit is not None and v.query.query_points - v.query_start + work > limit:
            raise QueryBudgetExceeded(
                "scene query point budget before refinement constraints"
            )
        frames = v.probe.probe_trajectory(current)
        if limit is not None and v.query.query_points - v.query_start > limit:
            raise QueryBudgetExceeded(
                "scene query point budget in refinement constraints"
            )
        rows, cols, values, rhs = [], [], [], []
        data = mujoco.MjData(v.model)
        for f, frame in enumerate(frames):
            data.qpos[:] = current[f]
            mujoco.mj_forward(v.model, data)
            active = np.flatnonzero((frame["clearance"] < 0.01) & ~frame["uncertain"])
            for k in active:
                gid = int(frame["geom_ids"][k])
                old = before["reliable_scene_depth"][f, gid]
                allowance = (
                    v.cfg["penetration_tolerance"]
                    if old <= v.cfg["penetration_tolerance"]
                    else old + v.cfg["collision_regression"]
                )
                jac = np.zeros((3, v.model.nv))
                mujoco.mj_jac(
                    v.model,
                    data,
                    jac,
                    None,
                    frame["points"][k],
                    int(frame["body_ids"][k]),
                )
                gradient = frame["directions"][k] @ jac[:, v.di]
                nonzero = np.flatnonzero(np.abs(gradient) > 1e-14)
                rows.extend([len(rhs)] * len(nonzero))
                cols.extend(f * len(v.qi) + nonzero)
                values.extend(gradient[nonzero])
                rhs.append(-allowance - frame["clearance"][k])
        if not rhs:
            return []
        matrix = sparse.coo_matrix(
            (values, (rows, cols)), shape=(len(rhs), current.shape[0] * len(v.qi))
        ).tocsr()
        return [matrix @ flat_step >= np.asarray(rhs)]

    def solve(self, system, rhs, current, lower, upper):
        v = self.validator
        flat = cp.Variable(system.shape[1])
        step = cp.reshape(flat, (len(current), len(v.qi)), order="C")
        correction = current[:, v.qi] - v.reference[:, v.qi] + step
        constraints = [
            flat >= lower,
            flat <= upper,
            correction <= v.cfg["max_joint_correction"],
            correction >= -v.cfg["max_joint_correction"],
        ]
        # Trust-region box bounds admit zero for standalone legacy inputs.
        # The guarded pipeline additionally enforces exact mechanical bounds.
        for bounds, sense in ((v.lower, "lower"), (v.upper, "upper")):
            mask = np.isfinite(bounds)
            if np.any(mask):
                joint = current[:, v.qi[mask]] + step[:, mask]
                constraints.append(
                    joint >= bounds[mask] if sense == "lower" else joint <= bounds[mask]
                )
        if len(current) > 1:
            velocity = cp.diff(correction, axis=0) * v.fps
            constraints += [
                velocity <= v.cfg["correction_velocity"],
                velocity >= -v.cfg["correction_velocity"],
            ]
        if len(current) > 2:
            acceleration = cp.diff(correction, k=2, axis=0) * v.fps**2
            constraints += [
                acceleration <= v.cfg["correction_acceleration"],
                acceleration >= -v.cfg["correction_acceleration"],
            ]
        for task in (v.self_contact_tasks, v.surface_contact_tasks):
            if task is None:
                continue
            old = task.errors(v.reference, v.frame_start)
            for f, pose in enumerate(current):
                for index, residual, jacobian in task.position_rows(pose, v.frame_start + f, v.di):
                    budget = max(old[f, index], task.cfg["tolerance"]) + task.cfg["regression"]
                    constraints.append(cp.norm(residual + jacobian @ step[f]) <= budget)
        constraints.extend(self._endpoint_constraints(current, step))
        constraints.extend(self._scene_constraints(current, flat))
        problem = cp.Problem(
            cp.Minimize(0.5 * cp.sum_squares(system @ flat - rhs)), constraints
        )
        try:
            problem.solve(
                solver="CLARABEL", max_iter=100, canon_backend=cp.SCIPY_CANON_BACKEND
            )
        except cp.error.SolverError as exc:
            self.report.update(status="solver_failed", reason=str(exc))
            return None
        if (
            problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
            or flat.value is None
        ):
            self.report.update(status="solver_failed", reason=problem.status)
            return None
        return SimpleNamespace(
            x=np.asarray(flat.value), cost=float(problem.value), status=1
        )

    def preserve_safe_smoothing(self, current, candidate):
        """Keep validated refinement if optional smoothing violates its budgets."""
        if np.array_equal(current, candidate):
            return candidate
        v = self.validator
        try:
            reasons = v.fidelity(candidate)["reasons"]
            if not reasons:
                reasons = v.compare_geometry(
                    self._baseline_geometry(),
                    v.geometry(candidate),
                    allow_existing_uncertainty=True,
                )
                if not reasons:
                    reasons.extend(preserve_known_scene_samples(v, candidate, self.before))
        except QueryBudgetExceeded as exc:
            reasons = [str(exc)]
        self.report["optional_smoothing"] = {
            "accepted": not reasons,
            "reasons": reasons,
        }
        return current.copy() if reasons else candidate

    def accept(self, current, candidate, system, rhs):
        v = self.validator
        before = self._baseline_geometry()
        change = candidate[:, v.qi] - current[:, v.qi]
        initial_cost = float(rhs @ rhs)
        record = {"accepted_alpha": 0.0, "rejections": []}
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625):
            trial = current.copy()
            trial[:, v.qi] += alpha * change
            predicted = system @ (alpha * change).reshape(-1) - rhs
            if float(predicted @ predicted) > initial_cost + 1e-8:
                reasons = ["linear_objective_regression"]
            else:
                fidelity = v.fidelity(trial)
                reasons = fidelity["reasons"]
                if not reasons:
                    after = v.geometry(trial)
                    reasons = v.compare_geometry(
                        before, after, allow_existing_uncertainty=True
                    )
                    if not reasons:
                        reasons.extend(preserve_known_scene_samples(v, trial, before))
            if not reasons:
                record.update(
                    accepted_alpha=alpha,
                    fidelity=fidelity["metrics"],
                    uncertain_frames=len(after["uncertain_frames"]),
                    uncertain_intervals=len(after["uncertain_intervals"]),
                    reliable_scene_max_depth=float(
                        np.max(after["reliable_scene_depth"], initial=0)
                    ),
                    reliable_interframe_max_depth=float(
                        np.max(after["interframe_reliable_scene_depth"], initial=0)
                    ),
                )
                self.report["iterations"].append(record)
                self.report["status"] = "accepted_updates"
                return trial
            record["rejections"].append({"alpha": alpha, "reasons": reasons})
        self.report["iterations"].append(record)
        self.report["status"] = "no_feasible_update"
        return current.copy()
