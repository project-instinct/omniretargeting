"""Fit source directions before endpoint errors can select a different action."""

from types import SimpleNamespace

import cvxpy as cp
import mujoco
import numpy as np
from ..conic_solver import solve_conic

from ..conic_constraints import LocalNormBounds, squared_residual_objective
from .surface_guard import _stack


class SourceShapeGuard:
    """Kinematic-reference hierarchy on source directions, per axis.

    Primary direction fitting finds a model-reachable branch. The complete
    source/endpoint/temporal objective then operates without worsening any
    achieved source direction. No angular allowance, anatomy labels or new
    physical tolerances.
    """

    def __init__(self, tasks):
        self.tasks = tasks
        self.bounds = {}
        self.failures = 0
        self.preserve_contacts = False
        self.pose = self.continuity = None
        self.priority_tolerance = 1e-7

    def rows(self, poses, frames=None):
        tasks = self.tasks
        for f in range(len(poses)) if frames is None else frames:
            tasks.data.qpos[:] = poses[f]
            mujoco.mj_forward(tasks.model, tasks.data)
            points, jacs = zip(*(tasks.point(body) for body in tasks.bodyids))
            points, jacs = np.asarray(points), np.asarray(jacs)
            for k, (weights, _) in enumerate(tasks.axes):
                current, target = weights @ points, weights @ tasks.human[f]
                length, target_length = np.linalg.norm(current), np.linalg.norm(target)
                if min(length, target_length) < 1e-10:
                    continue
                direction = current / length
                jac = (
                    (np.eye(3) - np.outer(direction, direction))
                    @ np.einsum("i,ijk->jk", weights, jacs)
                    / length
                )
                yield f, k, direction - target / target_length, jac

    def priority_errors(self, poses, frames=None):
        window = range(len(poses)) if frames is None else frames
        values = np.zeros((len(window), 1))
        for f, _, residual, _ in self.rows(poses, frames):
            values[f - window.start, 0] += np.linalg.norm(residual) ** 2
        return np.sqrt(values)

    def diagnostics(self, poses):
        angles = np.full((len(poses), len(self.tasks.axes)), np.nan)
        for f, k, residual, _ in self.rows(poses):
            angles[f, k] = 2 * np.arcsin(np.clip(np.linalg.norm(residual) / 2, 0, 1))
        finite = angles[np.isfinite(angles)]
        return {
            "source_modified": False,
            "angular_envelope_rad": None,
            "maximum_source_axis_angle_rad": float(finite.max(initial=0)),
            "mean_source_axis_angle_rad": float(finite.mean()) if len(finite) else None,
            "axes": [
                {self.tasks.names[i]: float(w) for i, w in enumerate(weights) if w}
                for weights, _ in self.tasks.axes
            ],
        }

    def accepts(self, poses, frames=None):
        # There is no acquired reference yet. The caller checks nonlinear
        # source-direction progress per frame; imposing independent monotonic
        # errors at a neutral seed can forbid the coupled rotations needed to
        # reach any source-shaped branch. Later stages use SequencePoseGuard.
        return bool(np.isfinite(poses).all())

    def feasible_step(self, poses, step, frames=None):
        """Backtrack source-fit frames independently before temporal acceptance.

        A nearly fitted frame on a curved angular boundary must not shrink the
        entire clip's step and strand unrelated frames on the wrong IK branch.
        The outer line search still checks the full nonlinear temporal objective
        and per-frame source priority. This is only reference acquisition; the
        fixed posture/continuity envelope in later passes is unchanged.
        ``frames`` gives the window that ``step`` spans; others stay fixed.
        """
        window = range(len(poses)) if frames is None else frames
        before = self.priority_errors(poses, frames)[:, 0]
        scales = np.ones(len(window))
        unresolved = np.ones(len(window), bool)
        full = np.zeros((len(poses), step.shape[1]))
        for _ in range(9):
            full[window.start : window.stop] = step * scales[:, None]
            candidate = self.tasks.coords.integrate(poses, full, 1)
            after = self.priority_errors(candidate, frames)[:, 0]
            unresolved = after > before + self.priority_tolerance
            if not unresolved.any():
                break
            scales[unresolved] *= 0.5
        scales[unresolved] = 0
        return step * scales[:, None]

    def solve(self, matrix, rhs, lower, upper, poses, frames=None):
        count, width = lower.shape
        start = 0 if frames is None else frames.start
        rows = list(self.rows(poses, frames))
        if not rows:
            from .linear import bounded_step

            return bounded_step(
                matrix, rhs, lower.ravel(), upper.ravel(), self.tasks.config
            )
        self.bounds = {(f, k): np.linalg.norm(r) for f, k, r, _ in rows}
        flat = cp.Variable(count * width)
        constraints = [flat >= lower.ravel(), flat <= upper.ravel()]
        linear, offset, _ = _stack(
            [(f - start, r, j) for f, _, r, j in rows], count, width
        )
        regularizer = 1e-8 * cp.sum_squares(flat)
        primary = cp.Problem(
            cp.Minimize(
                squared_residual_objective(linear, -offset, flat) + regularizer
            ),
            constraints,
        )
        try:
            solve_conic(primary, solver=cp.CLARABEL, direct_solve_method="qdldl")
            if flat.value is None or primary.status not in {
                cp.OPTIMAL,
                cp.OPTIMAL_INACCURATE,
            }:
                raise cp.error.SolverError("Source direction fit did not converge")
            proposal = np.asarray(flat.value).reshape(count, width)
            bounds = LocalNormBounds(count, width)
            for f, _, residual, jac in rows:
                achieved = np.linalg.norm(residual + jac @ proposal[f - start])
                bounds.add(residual, achieved + 1e-7, (f - start, jac))
            constraints += bounds.constraints(flat)
            secondary = cp.Problem(
                cp.Minimize(
                    (squared_residual_objective(matrix, rhs, flat) + regularizer)
                    / max(float(rhs @ rhs), 1.0)
                ),
                constraints,
            )
            solve_conic(secondary, solver=cp.CLARABEL, direct_solve_method="qdldl")
            if flat.value is not None and secondary.status in {
                cp.OPTIMAL,
                cp.OPTIMAL_INACCURATE,
            }:
                return SimpleNamespace(
                    x=np.asarray(flat.value), status=1,
                    backend=f"source_shape:{secondary.solver_stats.solver_name}",
                )
        except cp.error.SolverError:
            pass
        self.failures += 1
        return SimpleNamespace(x=np.zeros(count * width), status=0)
