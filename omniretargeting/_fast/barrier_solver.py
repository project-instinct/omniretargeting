"""Primal interior-point LM for legacy contact-acquisition windows.

The barrier is numerical; exact guards and the parent physical-cost verdict
remain authoritative. No attempt speculation or nonlinear restoration runs.
"""

import time

import numpy as np

from .barrier_assembly import equilibrated_solve, physical_system
from .barrier_rows import (BarrierRows, UnsupportedBarrier, accumulate,
                           boundary_fraction, minimum, strictly_interior)
from .box_step import BoxStepFailure, direction_bounds, solve_box_banded
from .proposals import guards_of
from .solver import (DIAGONAL_FLOOR, INITIAL_DAMPING, MAX_DAMPING, MIN_DAMPING,
                     _banded_solve, _hold, _hold_radial, _root_holds)

MAX_ATTEMPTS = 8
MAX_BACKTRACKS = 24
ARMIJO = 1e-4
RETAINED_MARGIN = .005
MU_REDUCTION = .2


def _equalities(objective, poses):
    coords, width = objective.coords, objective.width
    held = []
    if objective.fixed_root is not None:
        columns = [coords.inverse[d] for d in range(coords.root_d, coords.root_d + 6)
                   if d in coords.inverse]
        held.extend((np.arange(len(poses))[:, None] * width + columns).ravel())
    # A zero translation domain is an equality, even without fixed_root.
    global_objective = objective.global_objective
    shape = global_objective.source_shape
    if coords.root_d is not None:
        axes = (0, 1, 2) if getattr(global_objective, 'root_budget', None) == 0 else (
            (0, 1) if shape is not None and shape.xy == 0 else ())
        columns = [coords.inverse[coords.root_d + axis] for axis in axes
                   if coords.root_d + axis in coords.inverse]
        held.extend((np.arange(len(poses))[:, None] * width + columns).ravel())
    return np.array(held, dtype=int)


def _radial(objective, poses, gradient):
    holds = _root_holds(objective, poses, gradient)
    # ReferenceShape.project also has an XY domain. Its active radial rows
    # must participate in the coupled solve; it is not a logarithmic guard.
    global_objective, coords = objective.global_objective, objective.coords
    shape = global_objective.source_shape
    if shape is not None and coords.root_q is not None:
        columns = [coords.inverse.get(coords.root_d + axis) for axis in range(3)]
        if all(c is not None for c in columns):
            span, q = objective.span, coords.root_q
            delta = poses[:, q:q + 2] - shape.seed[span, q:q + 2]
            size = np.linalg.norm(delta, axis=1)
            normals = np.zeros((len(poses), 3))
            normals[:, :2] = delta / np.maximum(size[:, None], 1e-12)
            slope = np.einsum('fi,fi->f', gradient.reshape(len(poses), -1)[:, columns], normals)
            frames = np.flatnonzero((size >= shape.xy - 1e-9) & (slope < 0))
            holds.append((frames, np.asarray(columns), normals[frames]))
    return holds


def _trial(objective, rows, poses, families, step, alpha, cost, mu, gradient,
           diagnostics):
    """Values first; a rejected trial never requests objective Jacobians."""
    merit = cost / 2 + mu * rows.merit(families)
    for _ in range(MAX_BACKTRACKS):
        raw = objective.coords.integrate(poses, step.reshape(poses.shape[0], -1), alpha)
        candidate = objective.project(raw)
        if not _domains(objective, candidate):
            alpha *= .5
            diagnostics['boundary_contractions'] += 1
            continue
        trial_rows = rows.evaluate(candidate, False)
        feasible = all(np.isfinite(new.slack).all() and
                       np.all(new.slack >= RETAINED_MARGIN * old.slack)
                       for old, new in zip(families, trial_rows))
        if not feasible:
            alpha *= .5
            diagnostics['boundary_contractions'] += 1
            continue
        # Projection is part of the executed step and changes the merit slope.
        actual = objective.coords.difference(poses, candidate).ravel()
        slope = float(gradient @ actual)
        if slope >= 0 or not all(g.accepts(candidate) for g in guards_of(objective)[1]):
            alpha *= .5
            diagnostics['boundary_contractions'] += 1
            continue
        value = objective.evaluate(candidate, False)
        new_cost = value[0].cost
        diagnostics['physical_value_evaluations'] += 1
        after = new_cost / 2 + mu * rows.merit(trial_rows)
        if (np.isfinite(new_cost) and new_cost < cost and
                after <= merit + ARMIJO * slope):
            diagnostics['projection_changes'] += bool(np.any(candidate != raw))
            return candidate, alpha, new_cost, trial_rows, value[3]
        diagnostics['merit_contractions'] += 1
        alpha *= .5
    return None


def _domains(objective, poses):
    """The actual joint/root projection domains, independently of soft holds."""
    if not np.isfinite(poses).all():
        return False
    coords = objective.coords
    values = poses[:, coords.qi]
    if np.any(values < objective.lower) or np.any(values > objective.upper):
        return False
    owner = objective.global_objective
    q, span = coords.root_q, objective.span
    if q is not None:
        if objective.fixed_root is not None and not np.array_equal(poses[:, q:q + 7], objective.fixed_root):
            return False
        budget = getattr(owner, 'root_budget', None)
        if budget is not None and np.any(np.linalg.norm(
                poses[:, q:q + 3] - owner.reference[span, q:q + 3], axis=1) > budget):
            return False
        shape = owner.source_shape
        if shape is not None and np.any(np.linalg.norm(
                poses[:, q:q + 2] - shape.seed[span, q:q + 2], axis=1) > shape.xy):
            return False
    return True


def _mu(cost, families, unit_gradient, physical_gradient, diagonal):
    terms = sum(len(f.slack) for f in families)
    value = 1e-3 * max(cost / 2, 1.) / max(terms, 1)
    scale = 1 / np.sqrt(np.maximum(diagonal, DIAGONAL_FLOOR))
    barrier_norm = np.linalg.norm(unit_gradient * scale)
    if barrier_norm > 0:
        value = min(value, .1 * np.linalg.norm(physical_gradient * scale) / barrier_norm)
    return max(value, np.finfo(float).tiny)


def optimize(objective, poses, iterations, convergence):
    """Return None plus a binding diagnostic when the entry lacks an interior."""
    started = time.monotonic()
    diagnostics = dict(mode='barrier', mu=None, min_slack=None, boundary_contractions=0,
        merit_contractions=0, projection_changes=0, factorizations=0, escapes=0,
        escape_reasons=[], physical_value_evaluations=0, true_alphas=[], mu_reductions=0)
    try:
        rows = BarrierRows(objective)
    except UnsupportedBarrier as error:
        diagnostics.update(mode='legacy_fallback', binding_row=str(error))
        return None, [], False, 'unsupported_barrier', diagnostics
    families = rows.evaluate(poses, False)
    diagnostics['min_slack'] = minimum(families)
    if not strictly_interior(families):
        diagnostics.update(mode='legacy_fallback', binding_row=minimum(families),
                           slack_evaluations=rows.evaluations, slack_jacobian_evaluations=0)
        return None, [], False, 'no_strict_interior', diagnostics
    rows.initialize(families)
    history, mu, slow, converged = [], None, 0, False
    damping = INITIAL_DAMPING
    evaluation = objective.evaluate(poses)
    reason = 'iteration_quota'

    def solve(ab, rhs):
        diagnostics['factorizations'] += 1
        return equilibrated_solve(ab, rhs, _banded_solve)

    for iteration in range(iterations):
        acc, residual, temporal, stats, _ = evaluation
        cost = acc.cost
        physical_gradient = acc.gradient.ravel() + temporal.T @ residual
        physical = physical_system(acc, temporal)
        diagonal = np.maximum(np.diagonal(acc.hessian, axis1=1, axis2=2).ravel(), DIAGONAL_FLOOR)
        families = rows.evaluate(poses, True)
        unit = np.zeros_like(physical, order='F')
        unit_gradient = accumulate(families, 1., unit)
        initial_mu = _mu(cost, families, unit_gradient, physical_gradient, physical[0])
        mu = initial_mu if mu is None else min(mu, initial_mu)
        held = _equalities(objective, poses)
        lower, upper = direction_bounds(objective, poses, held)
        accepted, step_backend, last_failure = None, 'barrier_banded', 'no_physical_descent'
        for attempt in range(MAX_ATTEMPTS):
            gradient = physical_gradient + mu * unit_gradient
            system = physical + mu * unit
            system[0] += damping * diagonal + DIAGONAL_FLOOR
            _hold_radial(system, objective.width, _radial(objective, poses, gradient))
            try:
                step = solve_box_banded(system, -gradient, lower, upper, solve, _hold)
            except BoxStepFailure as error:
                last_failure = str(error)
                damping = min(damping * 10, MAX_DAMPING)
                continue
            alpha = boundary_fraction(families, step)
            diagnostics['boundary_contractions'] += alpha < 1.
            accepted = _trial(objective, rows, poses, families, step, alpha, cost,
                              mu, gradient, diagnostics)
            if accepted is not None:
                break
            # Centering may oppose C descent; reduce mu before adding stiffness.
            mu *= MU_REDUCTION
            diagnostics['mu_reductions'] += 1
            if attempt >= 2:
                damping = min(damping * 10, MAX_DAMPING)
        if accepted is None and diagnostics['escapes'] == 0:
            # One raw constrained direction. Never call _conic_proposal: that
            # invokes restoration/fraction fans and can solve repeatedly.
            from .shape_step import bounded_shape_step
            diagnostics['escapes'] += 1
            diagnostics['escape_reasons'].append(last_failure)
            step = bounded_shape_step(objective, poses, physical, physical_gradient, diagonal, damping)
            if step is not None:
                gradient = physical_gradient + mu * unit_gradient
                accepted = _trial(objective, rows, poses, families, step,
                    boundary_fraction(families, step), cost, mu, gradient, diagnostics)
                step_backend = 'barrier_conic_escape'
        if accepted is None:
            reason = 'barrier_stall'
            diagnostics['stall_reason'] = last_failure
            diagnostics['binding_row'] = minimum(families)
            history.append(dict(phase='contact_scene', iteration=iteration, cost=cost, after=cost,
                alpha=0., damping=damping, step_backend='barrier_stall', barrier=dict(diagnostics),
                seconds=time.monotonic() - started, **stats))
            break
        poses, alpha, after, families, stats = accepted
        diagnostics['true_alphas'].append(float(alpha))
        diagnostics.update(mu=float(mu), min_slack=minimum(families),
            slack_evaluations=rows.evaluations, slack_jacobian_evaluations=rows.jacobian_evaluations)
        history.append(dict(phase='contact_scene', iteration=iteration, cost=cost, after=after,
            alpha=float(alpha), damping=damping, step_backend=step_backend,
            barrier={**diagnostics, 'true_alphas': list(diagnostics['true_alphas']),
                     'escape_reasons': list(diagnostics['escape_reasons'])},
            seconds=time.monotonic() - started, **stats))
        # Boundary-clipped steps and numerical centering are unfinished work.
        decrease = (cost - after) / max(cost, 1e-12)
        slow = slow + 1 if alpha == 1. and decrease < convergence else 0
        converged = slow >= 2
        mu *= MU_REDUCTION
        diagnostics['mu_reductions'] += 1
        damping = max(damping / 3, MIN_DAMPING)
        if converged:
            reason = 'converged'
            break
        if iteration + 1 < iterations:
            # Accepted-pose FK is retained in the shared per-frame cache.
            evaluation = objective.evaluate(poses)
    diagnostics.update(mu=float(mu) if mu is not None else None, min_slack=minimum(families),
        slack_evaluations=rows.evaluations, slack_jacobian_evaluations=rows.jacobian_evaluations)
    if history:
        history[-1]['barrier'] = diagnostics
    return poses, history, converged, reason, diagnostics
