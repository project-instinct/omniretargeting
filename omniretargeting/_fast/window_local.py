"""One private window's complete local LM search, with pure exact trial checks."""

from copy import copy
import time

import numpy as np

from .proposals import ALPHAS, guards_of
from .priority_merit import report_value
from .qp_types import PriorityMerit
from .window_assembly import prepare_window_model, solve_window_model
from .window_schedule import MAX_LOCAL_ITERATIONS
from .window_soc import corrected_model
from .window_refinement import GeometryRefinement, MAX_REFINEMENTS

MAX_ATTEMPTS = 3
MAX_ESCALATIONS = 3
MAX_SOC = 2
STEP_TOLERANCE = 1e-9


def private_objective(objective, poses, frames):
    """No parent ledger mutation, even in the serial execution fallback."""
    local = copy(objective)
    parent = objective.priority_clearance
    child = parent.fork_window(poses, frames=frames)
    local.priority_clearance = child
    local.candidate_guards = []
    for guard in getattr(objective, 'candidate_guards', ()):
        if guard is parent:
            local.candidate_guards.append(child)
        elif getattr(guard, 'priority', None) is parent:
            guard = copy(guard)
            guard.priority = child
            local.candidate_guards.append(guard)
        else:
            local.candidate_guards.append(guard)
    child.objective = local
    # Frame caches are private mutable dictionaries; large immutable arrays
    # remain shared/COW. No inherited legacy worker pool may be used here.
    local.__dict__.pop('_window_views', None)
    local.pool, local.worker_output = None, None
    return local


def candidate_poses(objective, origin, frames, step, alpha=1.):
    span = slice(frames.start, frames.stop)
    candidate = origin.copy()
    candidate[span] = objective.coords.integrate(origin[span], step.reshape(len(frames), -1), alpha)
    projected = objective.project(candidate)
    candidate[span] = projected[span]
    return candidate


def exact_trial(objective, poses, frames):
    coordinator = objective.priority_clearance
    merit = PriorityMerit.coerce(coordinator.evaluate_trial(poses, frames=frames))
    # Coordinator checks the complete nonlinear priority contract, including
    # original hard guards. Do not repeat geometry through delegated guards.
    hard = all(guard.accepts(poses, frames=frames) for guard in guards_of(objective)[1]
               if guard is not coordinator and getattr(guard, 'priority', None) is not coordinator)
    return merit, bool(hard and merit.valid)


def penetration_regressed(merit):
    d = merit.details
    return bool(d.get('penetration_regressions') or d.get('body_regressions')
                or d.get('total_penetration_regression') or d.get('reason') == 'penetration_regression')


def tracking_margin_cost(model, merit):
    """Use complete nonlinear cost, including newly appearing collision keys."""
    if 'tracking_margin_cost' in merit.details:
        try:
            value = float(merit.details['tracking_margin_cost'])
        except (ValueError, TypeError):
            return None
        return value if np.isfinite(value) and value >= 0 else None
    # Older/empty-margin coordinators remain compatible. Unknown witnesses
    # have unknown weights here, so absence of their full cost fails closed.
    if any(isinstance(key, tuple) and key and key[0] == 'clearance_margin'
           and key not in model.soft._index for key in merit.slacks):
        return None
    return model.soft.tracking_penalty(merit.slacks)


def assess_window(model, delta, before, after, cost, value, coefficients):
    margin_before, margin_after = tracking_margin_cost(model, before), tracking_margin_cost(model, after)
    if margin_before is None or margin_after is None:
        return False, -1., dict(tracking_margin_cost_valid=False,
                               tracking_rejection='missing_or_invalid_complete_margin_cost', improved_level=None)
    reductions = model.predicted_reductions(delta)
    margin_predicted = model.soft.tracking_penalty(reductions)
    margin_actual = margin_before - margin_after
    predicted = model.tracking_decrease(delta) + margin_predicted
    actual = float(cost - value) + margin_actual
    gain = float(before.levels[0] - after.levels[0])
    tolerance = max(before.tolerances[0], after.tolerances[0])
    improved = gain > tolerance
    ratio = actual / predicted if predicted > 0 else -1.
    accepted = after.valid and np.isfinite(value) and (improved or (actual > 0 and ratio > 0))
    if improved:
        pred_pen = model.soft.level_values(reductions)[0]
        ratio = gain / pred_pen if pred_pen > 0 else 1.
    # Keep original-unit L1 actual/predicted merit in diagnostics even though
    # higher-level exact progress may override a tracking loss.
    predicted_l1 = model.soft.penalty(reductions, coefficients)
    actual_l1 = model.soft.penalty(before.slacks, coefficients) - model.soft.penalty(after.slacks, coefficients)
    details = dict(tracking_actual_decrease=actual, tracking_predicted_decrease=predicted,
                   tracking_cost_before=float(cost) + margin_before, tracking_cost_after=float(value) + margin_after,
                   tracking_margin_cost_valid=True, tracking_margin_actual_decrease=margin_actual,
                   tracking_margin_predicted_decrease=margin_predicted,
                   actual_merit_decrease=actual + actual_l1,
                   predicted_merit_decrease=predicted + predicted_l1,
                   penetration_decrease=gain, improved_level=0 if improved else None)
    return bool(accepted), float(ratio), details


def solve_local_window(objective, poses, frames, iterations, converged_decrease):
    """Return final private poses/claims; only the outer parent may commit."""
    from .solver import INITIAL_DAMPING, MIN_DAMPING, MAX_DAMPING
    from .runtime_options import OPTIONS
    start = time.monotonic()
    local = private_objective(objective, poses, frames)
    coordinator, coords = local.priority_clearance, local.coords
    poses = poses.copy()
    span = slice(frames.start, frames.stop)
    damping, growth, multiplier, slow = INITIAL_DAMPING, 2., 1., 0
    history = []
    for iteration in range(min(iterations, MAX_LOCAL_ITERATIONS)):
        iteration_start = time.monotonic()
        tick = time.monotonic()
        evaluation = local.evaluate(poses, frames=frames)
        objective_seconds = time.monotonic() - tick
        cost = evaluation[0].cost
        tick = time.monotonic()
        model, mandatory = prepare_window_model(local, poses, frames, evaluation)
        before = PriorityMerit.coerce(coordinator.evaluate_trial(poses, frames=frames))
        assembly_seconds = time.monotonic() - tick
        accepted, escalations, soc_count = False, 0, 0
        refiner = GeometryRefinement(coordinator, enabled=getattr(OPTIONS, 'window_geometry_generation', False))
        trace, checks, details, alpha = [], [], {}, 0.
        for attempt in range(MAX_ATTEMPTS + MAX_ESCALATIONS + (MAX_REFINEMENTS if refiner.enabled else 0)):
            if attempt - escalations - refiner.rounds >= MAX_ATTEMPTS:
                break
            solution, solves = solve_window_model(model, damping, mandatory,
                                                  penetration_multiplier=multiplier)
            for solve in solves:
                solve.update(attempt=attempt, mode='qp', damping=damping, penetration_multiplier=multiplier,
                             geometry_generation=refiner.rounds)
            trace.extend(solves)
            if solution is None:
                damping, growth = min(damping * growth, MAX_DAMPING), growth * 2
                continue
            directions = [(solution, model, 'qp')]
            full_regression, refined, block_escalation = False, False, False
            while directions and not accepted:
                proposed, proposal_model, mode = directions.pop(0)
                for fraction in (ALPHAS[1:] if mode == 'backtrack' else ALPHAS):
                    tick = time.monotonic()
                    candidate = candidate_poses(local, poses, frames, proposed.step, fraction)
                    after, valid = exact_trial(local, candidate, frames)
                    check = dict(alpha=fraction, mode=mode, valid=valid,
                                 attempt=attempt, damping=damping, penetration_multiplier=multiplier,
                                 reason=after.details.get('reason'), seconds=time.monotonic() - tick,
                                 details=report_value(after.details))
                    checks.append(check)
                    if refiner.enabled and penetration_regressed(after):
                        updated, keep, generation = refiner.refine(model, mandatory, candidate)
                        check['geometry_refinement'] = generation
                        block_escalation |= generation['unresolved']
                        if updated is not None:
                            model, mandatory, refined = updated, keep, True
                            break  # Same objective/box/damping/weights, new geometry rows first.
                    if fraction == 1. and mode == 'qp' and penetration_regressed(after):
                        full_regression = True
                        if escalations < MAX_ESCALATIONS and not block_escalation:
                            break
                    delta = coords.difference(poses, candidate)[span].ravel()
                    if not valid:
                        if fraction == 1. and soc_count < MAX_SOC:
                            correction = corrected_model(model, candidate, delta)
                            if correction is not None:
                                corrected, more = solve_window_model(correction, damping, mandatory,
                                                                     penetration_multiplier=multiplier)
                                for solve in more:
                                    solve.update(attempt=attempt, mode='soc', damping=damping,
                                                 penetration_multiplier=multiplier,
                                                 geometry_generation=refiner.rounds)
                                trace.extend(more)
                                soc_count += 1
                                if corrected is not None:
                                    directions.insert(0, (corrected, correction, 'soc'))
                                    directions.append((proposed, proposal_model, 'backtrack'))
                                    break  # Try the objective-keeping full SOC before shortening.
                        continue
                    tick = time.monotonic()
                    value = local.evaluate(candidate, with_jacobian=False, frames=frames)[0].cost
                    objective_seconds += time.monotonic() - tick
                    accepted, ratio, details = assess_window(model, delta, before, after, cost, value,
                                                             proposed.certificate['coefficients'])
                    if accepted and np.max(np.abs(delta), initial=0.) > STEP_TOLERANCE:
                        alpha = fraction
                        break
                    accepted = False
                if refined or (full_regression and escalations < MAX_ESCALATIONS and not block_escalation):
                    break
            if refined:
                continue
            if accepted:
                break
            if full_regression and escalations < MAX_ESCALATIONS and not block_escalation:
                multiplier *= 10.
                escalations += 1
            else:
                damping, growth = min(damping * growth, MAX_DAMPING), growth * 2
        record = dict(iteration=iteration, frames=(frames.start, frames.stop), cost=cost,
                      after=value if accepted else cost, alpha=alpha, damping=damping,
                      priority_status='accepted_local' if accepted else 'no_acceptable_window_proposal',
                      step_backend='windowed_priority', optimality_certified=False,
                      penetration_multiplier=multiplier, escalations=escalations, soc_solves=soc_count,
                      qp_variables=model.program.A.shape[1], qp_rows=len(model.program.b),
                      qp_cones=len(model.program.soc), objective_seconds=objective_seconds,
                      assembly_seconds=assembly_seconds, qp_seconds=sum(v['seconds'] for v in trace),
                      acceptance_seconds=sum(v['seconds'] for v in checks), qp_trace=trace,
                      geometry_generations=refiner.rounds, geometry_refinement_trace=refiner.trace,
                      geometry_refinement_seconds=sum(v['seconds'] for v in refiner.trace),
                      stats=report_value(evaluation[3]),
                      trial_trace=checks, seconds=time.monotonic() - iteration_start, **details)
        history.append(record)
        if not accepted:
            break
        tick = time.monotonic()
        coordinator.advance_local(candidate, frames=frames)
        record['advance_seconds'] = time.monotonic() - tick
        record['seconds'] = time.monotonic() - iteration_start
        poses = candidate
        update = float(np.clip(ratio, 0., 2.))
        damping = max(damping * max(1 / 3, 1 - (2 * update - 1) ** 3), MIN_DAMPING)
        growth = 2.
        small = (details['improved_level'] is None and
                 details['tracking_actual_decrease'] / max(abs(details['tracking_cost_before']), 1e-12)
                 < converged_decrease)
        slow = slow + 1 if small else 0
        if slow >= 2:
            break
    return dict(poses=poses[span].copy(), transactions=coordinator.export_transactions(frames=frames),
                trace=history, seconds=time.monotonic() - start,
                changed=bool(not np.array_equal(poses[span], objective.priority_clearance.current[span])))
