"""Fixed-boundary adapters for the original LM loop and conic restoration.

The solver sees local variables, while every evaluation sees absolute frames
of the original trajectory. No objective, guard, reference or budget is rebuilt.
"""

from contextlib import contextmanager
from dataclasses import replace
import time

import cvxpy as cp
import numpy as np

from .frame_kinematics import frame_range
from .proposals import guards_of


def guard_failures(objective, poses, frames=None):
    failures = []
    if not np.isfinite(poses).all():
        return ['nonfinite']
    values = poses[:, objective.coords.qi]
    if np.any(values < objective.lower) or np.any(values > objective.upper):
        failures.append('joint_limits')
    for guard in guards_of(objective)[1]:
        if not (guard.accepts(poses) if frames is None else guard.accepts(poses, frames=frames)):
            failures.append(type(guard).__name__)
    return failures


def commit_verdict(objective, origin, candidate, frames, *, costs=None):
    """The original parent decision, evaluated against a worker's snapshot."""
    started = time.monotonic()
    if costs is None:
        before = objective.evaluate(origin, False, frames=frames)[0].cost
        after = objective.evaluate(candidate, False, frames=frames)[0].cost
    else:
        before, after = costs
    failures = guard_failures(objective, candidate, frames)
    accepted = not failures and np.isfinite(after) and after < before
    reason = failures or ([] if accepted else ['no_actual_decrease'])
    return dict(cost=before, after=after if accepted else before, candidate_cost=after,
                accepted=bool(accepted), parent_rejections=reason,
                certification_seconds=time.monotonic() - started)


class WindowGuard:
    def __init__(self, window, guard):
        self.window, self.guard = window, guard
        self.rejections = 0

    def __getattr__(self, name):
        return getattr(self.guard, name)

    def accepts(self, poses, frames=None):
        good = self.guard.accepts(self.window.embed(poses), frames=self.window.variable_frames)
        self.rejections += not good
        return good

    def violation_frames(self, poses):
        bad = self.guard.violation_frames(self.window.embed(poses), frames=self.window.variable_frames)
        return bad[self.window.span]

    def qp_bounds(self, poses, *, frames=None):
        return self.guard.qp_bounds(self.window.embed(poses), frames=self.window.variable_frames)

    def constraints(self, poses, flat, box=None):
        # Keep complete vector cones, including stencils crossing either end.
        # Do not use shape_step's finite-interval merge for one-sided contacts.
        linear, cones = self.qp_bounds(poses)
        constraints = []
        for matrix, lower, upper in linear:
            lower, upper = np.asarray(lower), np.asarray(upper)
            lo, hi = np.isfinite(lower), np.isfinite(upper)
            if lo.any():
                constraints.append(matrix[lo] @ flat >= lower[lo])
            if hi.any():
                constraints.append(matrix[hi] @ flat <= upper[hi])
        for matrix, residual, allowed in cones:
            if len(allowed):
                value = cp.reshape(matrix @ flat, residual.shape, order='C') + residual
                scale = 1 / np.maximum(allowed, 1e-3)
                constraints.append(cp.norm(cp.multiply(scale[:, None], value), axis=1) <= allowed * scale)
        return constraints


class WindowObjective:
    """Local-variable facade consumed by solver._optimize unchanged."""
    def __init__(self, objective, poses, frames):
        self.global_objective = objective
        permits = getattr(objective, '_window_restoration_permits', None)
        if permits is not None:
            self._window_restoration_permits = permits
        # This facade's origin, absolute frame slice, weights and guards are
        # immutable for one solve. Retain evaluated costs for certification.
        self.evaluated_costs = {}
        self.evaluation_seconds = 0.
        self.evaluation_calls = self.evaluation_jacobian_calls = 0
        self.variable_frames = frame_range(frames, len(poses))
        self.span = slice(frames.start, frames.stop)
        self.origin = np.array(poses, copy=True)
        self.origin.setflags(write=False)
        self.frames, self.width = len(frames), objective.width
        self.coords, self.cfg = objective.coords, objective.cfg
        self.priority_clearance = None
        self.lower, self.upper = (self._slice_bound(objective.lower), self._slice_bound(objective.upper))
        fixed = getattr(objective, 'fixed_root', None)
        self.fixed_root = None if fixed is None else fixed[self.span]
        self.source_shape = (None if objective.source_shape is None else
                             WindowGuard(self, objective.source_shape))
        self.candidate_guards = [WindowGuard(self, guard) for guard in objective.candidate_guards]

    def _slice_bound(self, value):
        value = np.asarray(value)
        return value[self.span] if value.ndim == 2 else value

    def embed(self, poses):
        poses = np.asarray(poses)
        if poses.shape != (self.frames, self.origin.shape[1]):
            raise ValueError('Window pose shape changed')
        full = self.origin.copy()
        full[self.span] = poses
        return full

    def evaluate(self, poses, with_jacobian=True):
        started = time.monotonic()
        result = self.global_objective.evaluate(self.embed(poses), with_jacobian,
                                                frames=self.variable_frames)
        self.evaluation_seconds += time.monotonic() - started
        self.evaluation_calls += 1
        self.evaluation_jacobian_calls += bool(with_jacobian)
        self.evaluated_costs[np.asarray(poses).tobytes()] = result[0].cost
        return result

    def project(self, poses):
        # Project against global root/reference bounds; only the window returns.
        return self.global_objective.project(self.embed(poses))[self.span].copy()

    def root_boundaries(self, poses):
        return [(offset, normals[self.span]) for offset, normals in
                self.global_objective.root_boundaries(self.embed(poses))]


@contextmanager
def serial_proposals():
    """Outer windows own concurrency; preserve serial proposal mathematics."""
    from . import proposals, runtime_options
    speculative, options = proposals.SPECULATE, runtime_options.OPTIONS
    proposals.SPECULATE = False
    runtime_options.OPTIONS = replace(options, parallel_shortening=False)
    try:
        yield
    finally:
        proposals.SPECULATE = speculative
        runtime_options.OPTIONS = options


@contextmanager
def observe_convergence(state):
    """Observe the actual legacy verdict, including its shortened-step rule.

    History alone omits the ``shortened`` argument. Delegate the existing
    function unchanged in this private, single-thread solve and restore it.
    """
    from . import solver
    original = solver._progress_converged

    def observed(*args, **kwargs):
        slow, done = original(*args, **kwargs)
        state['converged'] = bool(done)
        return slow, done

    solver._progress_converged = observed
    try:
        yield
    finally:
        solver._progress_converged = original


def solve_local(objective, poses, frames, iterations, convergence):
    """Run the established LM/fallback/restoration rules with local variables."""
    import os
    from .solver import _optimize
    started = time.monotonic()
    local = WindowObjective(objective, poses, frames)
    from .window_qoco import RESTORATION_DIAGNOSTICS
    RESTORATION_DIAGNOSTICS.clear()
    progress = dict(converged=False)
    result, history = poses[local.span].copy(), []

    def accepted(candidate, row):
        nonlocal result
        result = candidate
        history.append(row)

    from .runtime_options import OPTIONS
    barrier = None
    constrained = None
    with serial_proposals(), observe_convergence(progress):
        if OPTIONS.continuity_hinge_acquisition or OPTIONS.active_set_acquisition:
            from .constrained_solver import optimize as constrained_optimize
            candidate, history, progress['converged'], reason, constrained = constrained_optimize(
                local, result, iterations, convergence)
            if candidate is not None:
                result = candidate
        elif OPTIONS.barrier_acquisition:
            from .barrier_solver import optimize as barrier_optimize
            candidate, history, progress['converged'], reason, barrier = barrier_optimize(
                local, result, iterations, convergence)
            if candidate is not None:
                result = candidate
        if ((barrier is None and constrained is None) or
                (barrier is not None and barrier['mode'] == 'legacy_fallback') or
                (constrained is not None and constrained['mode'] == 'legacy_fallback')):
            result, history = _optimize(local, result, iterations, accepted,
                                       'contact_scene', convergence)
            reason = ('rejected_iteration' if history and history[-1]['alpha'] == 0 else
                      'converged' if progress['converged'] else 'iteration_quota')
            if barrier is not None and history:
                history[-1]['barrier'] = barrier
            if constrained is not None and history:
                history[-1]['constrained'] = constrained
    lm_seconds = time.monotonic() - started
    costs = (local.evaluated_costs[poses[local.span].tobytes()],
             local.evaluated_costs[result.tobytes()])
    verdict = commit_verdict(objective, poses, local.embed(result), frames, costs=costs)
    return dict(poses=result, history=history, seconds=time.monotonic() - started,
                worker_pid=os.getpid(), converged=progress['converged'], stop_reason=reason,
                commit=verdict, lm_seconds=lm_seconds,
                evaluation_seconds=local.evaluation_seconds, evaluation_calls=local.evaluation_calls,
                evaluation_jacobian_calls=local.evaluation_jacobian_calls,
                parallel_restoration=getattr(local, 'parallel_shortening_stats', {}),
                qoco_restoration=dict(RESTORATION_DIAGNOSTICS), barrier=barrier, constrained=constrained)


def grid_depths(objective, poses, *, chunk_size=64):
    """Current static/moving acquisition-grid depths by absolute frame.

    No original-mesh service or temporary deepest_fields pool. These are the
    same shell samples and SDFs used by acquisition, not a mesh certificate.
    Positive infinity in the existing grid kernel means far OR unknown;
    preserve that kernel's evidence scope rather than inventing coverage.
    """
    from .window_objective import _views
    depths, families = np.zeros(len(poses)), set()
    for first in range(0, len(poses), chunk_size):
        frames = range(first, min(len(poses), first + chunk_size))
        local, _, _ = _views(objective, frames)
        local.kin = objective.window_kinematics(poses, frames, False)
        for name, *field in local._clearance_fields():
            families.add(name)
            clearance = local._deepest_samples(*field)[0]
            value = np.max(np.maximum(-clearance, 0.), axis=1, initial=0.)
            depths[frames.start:frames.stop] = np.maximum(depths[frames.start:frames.stop], value)
    return depths, sorted(families)
