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

from ..conic_solver import solve_conic
from ..scene_witness_guard import preserve_known_scene_samples
from ..trajectory_validation import QueryBudgetExceeded


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
        from ..endpoint_constraints import endpoint_constraints

        return endpoint_constraints(self.validator, current, cp.vec(step, order="C"))

    def _scene_constraints(self, current, flat_step):
        v = self.validator
        before = self._baseline_geometry()
        limit = v.cfg["max_query_points"]
        work = len(current) * v.probe.query_work_bound
        if limit is not None and v.query.query_points - v.query_start + work > limit:
            raise QueryBudgetExceeded(
                "scene query point budget before refinement constraints"
            )
        if limit is None:
            from ..scene_probe_batch import probe_batches

            frames = probe_batches(v.probe, current)
        else:
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
            solve_conic(problem,
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
