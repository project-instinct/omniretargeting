"""Source fitting and vertical registration inside one immutable seed envelope.

This opt-in helper does not freeze or rebase the final robot reference. Every
accepted initializer uses the existing executed-derivative continuity guard.
Layout, posture, source data, contacts and clearance remain the caller's work.
"""

from types import SimpleNamespace

import cvxpy as cp
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from ..conic_solver import solve_conic
from .continuity import CorrectionContinuity, ROUND_OFF, audit_continuity, derivatives


class InitializationConflict(RuntimeError):
    """No certified initializer; ``report`` retains the original seed comparison."""

    def __init__(self, report):
        self.report = report
        super().__init__(f"Initialization {report.get('stage')} is not certified: "
                         f"{report.get('reason', 'seed continuity budget')}")


class _ContinuityOnlyShape:
    """Enable the existing constrained fallback without a second reference shape.

The sole continuity guard is in candidate_guards. Direct assembly dispatches
on continuity_only before accessing ReferenceShape-specific attributes.
"""

    continuity_only = True

    def __init__(self, tasks):
        self.pose = SimpleNamespace(tasks=tasks)
        self.rejections = 0

    def accepts(self, poses):
        return bool(np.isfinite(poses).all())

    def violation_frames(self, poses):
        return ~np.isfinite(poses).all(axis=1)

    def constraints(self, poses, flat, box=None):
        return []

    def qp_bounds(self, poses):
        return [], []

    def project(self, poses):
        return poses


class SeedInitialization:
    """One read-only orientation seed, shared by fit, normalization and lifts."""

    def __init__(self, coords, seed):
        self.coords = coords
        self.seed = np.array(seed, dtype=float, copy=True)
        if self.seed.ndim != 2 or not np.isfinite(self.seed).all():
            raise ValueError('Initialization seed must be a finite trajectory')
        self.seed.setflags(write=False)
        self.guard = CorrectionContinuity(coords, self.seed, correction=False)

    def _failures(self, poses):
        values = derivatives(self.coords, poses)
        failures = []
        for name, bounds in self.guard.bounds.items():
            size = np.abs(values[name])
            excess = size - bounds - ROUND_OFF
            if not np.any(excess > 0):
                continue
            frame, coordinate = np.unravel_index(np.argmax(excess), excess.shape)
            label = ('x', 'y', 'z')[coordinate] if not name.startswith('joints') else None
            if label is None:
                model = self.coords.model
                joint = int(model.dof_jntid[self.coords.di[coordinate]])
                label = model.joint(joint).name
            else:
                model = self.coords.model
                joint = int(model.dof_jntid[self.coords.root_d])
            body = model.body(int(model.jnt_bodyid[joint])).name
            order = int(name[-1])
            failures.append(dict(channel=name, coordinate=label, body=body,
                                 coordinate_index=int(coordinate),
                                 stencil=list(range(int(frame), int(frame) + order + 1)),
                                 actual=float(size[frame, coordinate]),
                                 budget=float(bounds[frame, coordinate]),
                                 slack=ROUND_OFF, excess=float(excess[frame, coordinate])))
        return failures

    def audit(self, poses, *, stage):
        poses = np.asarray(poses)
        if poses.shape != self.seed.shape or not np.isfinite(poses).all():
            return dict(stage=stage, reference_rebased=False, accepted=False,
                        reason='nonfinite or mismatched trajectory', failures=[])
        try:
            result = audit_continuity(self.coords, self.seed, poses, correction=False)
        except ValueError as error:
            return dict(stage=stage, reference_rebased=False, accepted=False,
                        reason=str(error), failures=[])
        result.update(stage=stage, reference_rebased=False,
                      accepted_semantics='immutable_seed_windowed_continuity_only',
                      accepted=not bool(self.guard.violation_frames(poses).any()),
                      failures=self._failures(poses))
        return result

    def check_normalized(self, poses, *, stage):
        """Certify the supplied state; clipping never replaces the stored seed."""
        report = self.audit(poses, stage=stage)
        if not report['accepted']:
            raise InitializationConflict(report)
        return report

    def attach_source_objective(self, objective):
        self.check_normalized(objective.reference, stage='source_objective_reference')
        if getattr(objective, 'source_shape', None) is not None:
            raise ValueError('Source initialization already has a reference shape')
        objective.source_shape = _ContinuityOnlyShape(objective.tasks)
        guards = list(getattr(objective, 'candidate_guards', ()))
        if not any(guard is self.guard for guard in guards):
            guards.append(self.guard)
        objective.candidate_guards = guards

    def _vertical_constraints(self, poses, shift):
        # Source fitting may lock xy while height registration does not. Use
        # the original guard's columns, not the later objective's column index.
        column = self.coords.inverse.get(self.coords.root_d + 2)
        if column is None:
            raise InitializationConflict(dict(stage='vertical_registration', accepted=False,
                reason='seed coordinates lock root height', reference_rebased=False))
        indices = np.arange(len(poses)) * self.coords.width + column
        linear, norms = self.guard.repair_bounds(poses)
        constraints = []
        for matrix, lower, upper in linear:
            selected = matrix[:, indices].tocsr()
            active = np.flatnonzero(np.diff(selected.indptr))
            if len(active):
                # Preserve the guard's own interval and roundoff reserve. A
                # zero step remains feasible for an accepted boundary state.
                center = (upper[active] + lower[active]) / 2
                half = (upper[active] - lower[active]) / 2
                residual = sparse.diags(1 / half) @ selected[active] @ shift - center / half
                constraints += [residual <= 1., residual >= -1.]
        for matrix, values, bound in norms:
            selected = matrix[:, indices]
            if selected.nnz:
                residual = selected @ shift + values.ravel()
                constraints.append(cp.SOC(bound,
                    cp.reshape(residual, (values.shape[1], len(values)), order='F'), axis=0))
        return constraints

    def _solve_vertical(self, poses, matrix, gradient, *, stage, upper=None):
        shift = cp.Variable(len(poses))
        constraints = self._vertical_constraints(poses, shift)
        if upper is not None:
            constraints += [shift >= 0., shift <= upper]
        scale = max(float(np.linalg.norm(gradient, ord=np.inf)), 1.)
        problem = cp.Problem(cp.Minimize(
            (cp.quad_form(shift, cp.psd_wrap(matrix)) + 2 * gradient @ shift) / scale), constraints)
        try:
            solve_conic(problem, cpu_only=True, solver=cp.CLARABEL, direct_solve_method='qdldl')
        except cp.error.SolverError as error:
            raise InitializationConflict(dict(stage=stage, accepted=False,
                reason=str(error), solver_status=problem.status, reference_rebased=False)) from error
        if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or shift.value is None:
            raise InitializationConflict(dict(stage=stage, accepted=False,
                reason='vertical QP has no certified candidate', solver_status=problem.status,
                reference_rebased=False))
        value = np.asarray(shift.value)
        if upper is not None:
            # Enforce physical request bounds exactly, then certify derivatives
            # again; clipping alone is never a continuity certificate.
            value = np.clip(value, 0., upper)
        result = np.array(poses, copy=True)
        result[:, self.coords.root_q + 2] += value
        report = self.check_normalized(result, stage=stage)
        report['solver_status'] = problem.status
        return value, result, report

    def bounded_height_registration(self, objective, poses):
        self.check_normalized(poses, stage='height_origin')
        coords = objective.coords
        column = (None if coords.root_d is None else coords.inverse.get(coords.root_d + 2))
        if column is None:
            report = self.audit(poses, stage='postheight_vs_seed')
            report.update(status='height_locked', constrained=False)
            return np.array(poses, copy=True), np.zeros(len(poses)), report
        acc, residual, jacobian, _, _ = objective.evaluate(poses)
        indices = np.arange(len(poses)) * coords.width + column
        temporal = jacobian[:, indices]
        matrix = sparse.diags(acc.hessian[:, column, column]) + temporal.T @ temporal
        gradient = acc.gradient[:, column] + temporal.T @ residual
        # Same diagonal regularizer as reference.register_height.
        matrix += sparse.eye(len(poses)) * 1e-9
        requested = spsolve(matrix.tocsc(), -gradient)
        if not np.isfinite(requested).all():
            raise InitializationConflict(dict(stage='height_registration', accepted=False,
                reason='nonfinite unconstrained height', reference_rebased=False))
        trial = np.array(poses, copy=True)
        trial[:, coords.root_q + 2] += requested
        request_report = self.audit(trial, stage='unconstrained_height_vs_seed')
        if request_report['accepted']:
            shift, result = requested, trial
            report = self.audit(result, stage='postheight_vs_seed')
            report.update(status='unconstrained_height_certified', constrained=False)
        else:
            shift, result, report = self._solve_vertical(poses, matrix, gradient,
                                                       stage='postheight_vs_seed')
            report.update(status='height_constrained_by_seed', constrained=True)
        report.update(request_audit=request_report, requested_shift_m=requested.tolist(),
                      applied_shift_m=shift.tolist())
        return result, shift, report

    def bounded_reference_lift(self, poses, requested_lift):
        self.check_normalized(poses, stage='lift_origin')
        request = np.array(requested_lift, dtype=float, copy=True)
        if request.shape != (len(poses),) or not np.isfinite(request).all() or np.any(request < 0):
            raise ValueError('Reference lift must be finite nonnegative (frames,)')
        if self.coords.root_q is None:
            if request.any():
                raise InitializationConflict(dict(stage='reference_lift', accepted=False,
                    reason='model has no floating root', reference_rebased=False))
            return request, dict(stage='postlift_vs_seed', accepted=True,
                                 status='no_lift_requested', reference_rebased=False)
        trial = np.array(poses, copy=True)
        trial[:, self.coords.root_q + 2] += request
        request_report = self.audit(trial, stage='requested_lift_vs_seed')
        height_locked = self.coords.root_d + 2 not in self.coords.inverse
        if height_locked:
            shift = np.zeros(len(poses))
            report = self.audit(poses, stage='postlift_vs_seed')
            report['binding_constraint'] = 'locked root height'
        elif request_report['accepted']:
            shift = request
            report = self.audit(trial, stage='postlift_vs_seed')
        else:
            shift, _, report = self._solve_vertical(poses, sparse.eye(len(poses)), -request,
                                                   stage='postlift_vs_seed', upper=request)
        residual = request - shift
        limited = np.flatnonzero(residual > 0.)
        report.update(status=('lift_limited_by_locked_height' if height_locked and len(limited)
                              else 'lift_limited_by_seed' if len(limited) else 'requested_lift_certified'),
                      requested_lift_m=request.tolist(), applied_lift_m=shift.tolist(),
                      unachieved_lift_m=residual.tolist(), limited_frames=limited.tolist(),
                      requested_lift_achieved=not bool(len(limited)),
                      support_clearance_certified=False,
                      request_audit=request_report)
        return shift, report
