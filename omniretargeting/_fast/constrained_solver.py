"""Constrained acquisition proposals, with original physical/guard acceptance."""

from copy import deepcopy
import time

import numpy as np
from scipy.linalg import LinAlgError

from .barrier_assembly import physical_system
from .barrier_rows import UnsupportedBarrier, minimum
from .barrier_solver import _domains
from .constrained_rows import ConstrainedRows
from .constrained_step import hinge_directions, active_directions
from .constrained_active import ActiveSetFailure
from .proposals import guards_of, ALPHAS
from .solver import (INITIAL_DAMPING, MIN_DAMPING, _model_decrease,
                     _conic_proposal, _shorten_conic)


def trial(objective, rows, poses, families, direction, evaluation, gradient, diagnostics):
    """Exact slack values first, then unchanged guard predicates and C descent."""
    acc, _, temporal, _, _ = evaluation
    for alpha in ALPHAS:
        candidate = objective.project(objective.coords.integrate(
            poses, direction.reshape(objective.frames, objective.width), alpha))
        if not _domains(objective, candidate):
            continue
        values = rows.evaluate(candidate, False)
        failures = [f for f in values if not np.isfinite(f.slack).all() or np.any(f.slack < 0)]
        diagnostics['guard_trials'] += 1
        if failures:
            diagnostics['binding_row'] = minimum(failures)
            if alpha == 1. and getattr(rows, 'second_order', False) and not rows._soc_last:
                rows._soc_trial = (values, objective.coords.difference(poses, candidate).ravel())
                return None
            continue
        if not all(g.accepts(candidate) for g in guards_of(objective)[1]):
            continue
        # Tiny fractions request a better constrained direction first.
        if alpha < .25:
            diagnostics['short_feasible_alpha'] = alpha
            break
        displacement = objective.coords.difference(poses, candidate).ravel()
        predicted = _model_decrease(acc, temporal, displacement, gradient)
        value = objective.evaluate(candidate, False)
        diagnostics['physical_trials'] += 1
        ratio = (acc.cost - value[0].cost) / predicted if predicted > 0 else -1.
        if np.isfinite(value[0].cost) and value[0].cost < acc.cost and ratio > 0:
            return candidate, alpha, value, ratio
    if getattr(rows, 'banded_restoration', False):
        from .constrained_restoration import restore
        for alpha in (1., .5, .25):
            try:
                candidate = restore(objective, rows, poses, direction, alpha, diagnostics)
            except (LinAlgError, ActiveSetFailure) as error:
                diagnostics.setdefault('restoration_failures', []).append(str(error))
                continue
            if candidate is None:
                continue
            displacement = objective.coords.difference(poses, candidate).ravel()
            predicted = _model_decrease(acc, temporal, displacement, gradient)
            value = objective.evaluate(candidate, False)
            diagnostics['physical_trials'] += 1
            ratio = (acc.cost - value[0].cost) / predicted if predicted > 0 else -1.
            diagnostics.setdefault('restored_trials', []).append(dict(alpha=alpha,
                before=acc.cost, after=value[0].cost, ratio=float(ratio)))
            if np.isfinite(value[0].cost) and value[0].cost < acc.cost and ratio > 0:
                return candidate, alpha, value, ratio
    return None


def escape(objective, poses, evaluation, physical, gradient, diagonal, damping, diagnostics):
    acc, _, temporal, _, _ = evaluation
    guards = guards_of(objective)[1]
    diagnostics['conic_calls'] += 1
    proposal = _conic_proposal(objective, poses, physical, gradient, diagonal, damping, guards)
    if proposal is None:
        return None
    step, candidate, alpha = proposal
    if candidate is not None:
        actual = objective.coords.difference(poses, candidate).ravel()
        predicted = _model_decrease(acc, temporal, actual, gradient)
        value = objective.evaluate(candidate, False)
        ratio = (acc.cost - value[0].cost) / predicted if predicted > 0 else -1.
        if np.isfinite(value[0].cost) and value[0].cost < acc.cost and ratio > 0:
            return candidate, alpha, value, ratio
    return _shorten_conic(objective, poses, step, 1. if candidate is None else alpha,
                          acc, temporal, gradient, acc.cost, metric=(physical, diagonal, damping))


def optimize(objective, poses, iterations, convergence):
    started = time.monotonic()
    from .runtime_options import OPTIONS
    active_mode = OPTIONS.active_set_acquisition
    diagnostics = dict(mode='active_set' if active_mode else 'continuity_hinge',
        factorizations=0, hinge_rounds=0, guard_trials=0, physical_trials=0, conic_calls=0,
        true_alphas=[], numerical_failures=[], band_solves=0, schur_iterations=0,
        schur_incomplete=0, working_sets=0, added_rows=0, dropped_rows=0)
    try:
        rows = ConstrainedRows(objective)
    except UnsupportedBarrier as error:
        return None, [], False, 'unsupported_constrained', dict(mode='legacy_fallback', binding_row=str(error))
    if active_mode:
        # The opt-in B route uses the quadratic primal-dual KKT component.
        # Explicit class overrides are retained for isolated offline probes.
        rows.primal_dual = getattr(rows, 'primal_dual', True)
        rows.banded_restoration = getattr(rows, 'banded_restoration', rows.primal_dual)
    history, slow, converged, damping = [], 0, False, INITIAL_DAMPING
    warm = {}
    reason = 'iteration_quota'
    for iteration in range(iterations):
        evaluation = objective.evaluate(poses)
        acc, residual, temporal, stats, _ = evaluation
        gradient = acc.gradient.ravel() + temporal.T @ residual
        physical = physical_system(acc, temporal)
        diagonal = np.maximum(np.diagonal(acc.hessian, axis1=1, axis2=2).ravel(), 1e-9)
        families = rows.evaluate(poses, True)
        accepted = None
        backend = 'active_set_banded' if active_mode else 'continuity_hinge_banded'
        try:
            arguments = (objective, poses, rows, families, physical, gradient, diagonal, damping, diagnostics)
            directions = active_directions(*arguments, warm) if active_mode else hinge_directions(*arguments)
            for direction in directions:
                accepted = trial(objective, rows, poses, families, direction, evaluation, gradient, diagnostics)
                if accepted is not None:
                    break
        except (LinAlgError, ActiveSetFailure) as error:
            diagnostics['numerical_failures'].append(str(error))
        if accepted is None:
            backend = 'shape_conic'
            accepted = escape(objective, poses, evaluation, physical, gradient, diagonal, damping, diagnostics)
        if accepted is None:
            reason = 'rejected_iteration'
            history.append(dict(phase='contact_scene', iteration=iteration, cost=acc.cost, after=acc.cost,
                alpha=0., damping=damping, step_backend=backend, constrained=deepcopy(diagnostics),
                seconds=time.monotonic() - started, **stats))
            break
        before_poses=poses
        poses, alpha, value, ratio = accepted
        warm['previous_step']=objective.coords.difference(before_poses,poses).ravel()
        after = value[0].cost
        diagnostics['true_alphas'].append(float(alpha))
        damping = max(damping * max(1 / 3, 1 - (2 * ratio - 1)**3), MIN_DAMPING)
        history.append(dict(phase='contact_scene', iteration=iteration, cost=acc.cost, after=after,
            alpha=float(alpha), damping=damping, step_backend=backend,
            constrained=deepcopy(diagnostics),
            seconds=time.monotonic() - started, **value[3]))
        decrease = (acc.cost - after) / max(acc.cost, 1e-12)
        slow = slow + 1 if alpha == 1. and decrease < convergence else 0
        converged = slow >= 2
        if converged:
            reason = 'converged'
            break
    diagnostics.update(slack_evaluations=rows.evaluations, slack_jacobian_evaluations=rows.jacobian_evaluations)
    if history:
        history[-1]['constrained'] = diagnostics
    return poses, history, converged, reason, diagnostics
