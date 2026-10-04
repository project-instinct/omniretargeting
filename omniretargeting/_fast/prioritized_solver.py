"""Coordinator-driven acquisition LM; no ledger mutation during proposals."""

import os
import time

import numpy as np
from scipy import sparse

from .globalization import globalize
from .priority_merit import assess, check_trial, evaluate_priority, report_value
from .proposals import ALPHAS, guards_of
from .qp_assembly import prepare_model, solve_model
from .priority_policy import CERTIFICATE_STEP_TOLERANCE, TRUST_SHRINK, iteration_policy


def optimize_priority(objective, poses, iterations, callback, label, converged_decrease):
    from .solver import (INITIAL_DAMPING, DIAGONAL_FLOOR, MIN_DAMPING, MAX_DAMPING,
                         MAX_SHAPE_SOLVES, banded_system, _progress_converged)
    from .runtime_options import OPTIONS
    from .descent_step import NORMALIZED_PENALTY, PENALTY_GROWTH, original_primal_certificate
    owner = os.getpid()
    coordinator = objective.priority_clearance
    descent = OPTIONS.priority_zero_yield_descent
    coords = objective.coords
    damping, growth, slow = INITIAL_DAMPING, 2., 0
    history, costs = [], []
    start = time.monotonic()
    acc, temporal_r, temporal_j, stats, _ = objective.evaluate(poses)
    for iteration in range(iterations):
        gradient = acc.gradient.ravel() + temporal_j.T @ temporal_r
        system = banded_system(objective.frames, objective.width, acc.hessian, temporal_j)
        diagonal = np.maximum(np.diagonal(acc.hessian, axis1=1, axis2=2).ravel(), DIAGONAL_FLOOR)
        model = prepare_model(objective, poses, system, gradient, diagonal)
        policy = iteration_policy(damping, growth)
        repeated = getattr(coordinator, 'repeated_stall', lambda model, policy: None)(model, policy)
        if repeated is not None:
            history.append(dict(phase=label, iteration=iteration, cost=acc.cost, after=acc.cost, alpha=0.,
                                damping=damping, seconds=time.monotonic() - start,
                                priority_status='repeated_stalled_problem', reused_failure=repeated, **stats))
            break
        before = evaluate_priority(objective, poses)
        cost = acc.cost
        if not costs:
            costs.append(cost)
        guards = guards_of(objective)[1]
        accepted = False
        details = {}
        trust_fraction = 1.
        # Both paths retain the original conic domain. Descent candidates get
        # their distinct nonlinear certificate only at the executed trial.
        for attempt in range(MAX_SHAPE_SOLVES):
            if trust_fraction == 1.:
                kwargs = (dict(descent_penalty=NORMALIZED_PENALTY * PENALTY_GROWTH ** attempt)
                          if descent else {})
                solution = solve_model(model, damping, priority=True, **kwargs)
            else:
                from .qp_reduction import solve_restricted
                restricted = solve_restricted(
                    model, damping, sparse.eye(gradient.size, format='csc'), np.zeros(gradient.size),
                    priority=True, lower=trust_fraction * model.step_lower.ravel(),
                    upper=trust_fraction * model.step_upper.ravel())
                solution = None if restricted is None else restricted[0]
            candidate_ready = (solution is not None and solution.certificate.get(
                'primal_feasible' if descent else 'certified', False))
            if candidate_ready:
                solution, candidate, alpha = globalize(
                    objective, model, solution, damping, guards,
                    soc=OPTIONS.objective_soc, frame_lengths=OPTIONS.frame_step_lengths)
                if candidate is not None:
                    direction = coords.difference(poses, candidate)
                    for fraction in ALPHAS:
                        trial_poses = (candidate if fraction == 1. else
                            objective.project(coords.integrate(poses, direction, fraction)))
                        if not check_trial(objective, trial_poses, guards):
                            continue
                        after = evaluate_priority(objective, trial_poses)
                        delta = coords.difference(poses, trial_poses).ravel()
                        if descent:
                            primal = original_primal_certificate(model, delta)
                            if not primal['primal_feasible']:
                                continue
                            proof = coordinator.certify_zero_yield(
                                trial_poses, predicted_slacks=model.predicted_slacks(delta))
                            if (proof is None or not proof.get('descent_certified', False)
                                    or proof.get('optimality_certified', True)
                                    or proof.get('kind') != 'zero_additional_contact_yield_descent'):
                                continue
                            reductions = model.predicted_reductions(delta)
                            solution.certificate = dict(solution.certificate, **proof)
                            solution.certificate.update(
                                certified=False, original_primal=primal,
                                predicted_reductions=dict(keys=reductions,
                                    levels=model.soft.level_values(reductions).tolist()))
                        elif (after.details.get('requires_priority_certificate', False)
                                and not np.allclose(delta, solution.step, rtol=0., atol=CERTIFICATE_STEP_TOLERANCE)):
                            # Uniform shortening can create avoidable contact
                            # yield. Certify a new full hierarchy optimum in
                            # the smaller trust box before granting credit.
                            trust_fraction *= min(TRUST_SHRINK, alpha * fraction)
                            break
                        value = objective.evaluate(trial_poses, with_jacobian=False)[0].cost
                        accepted, ratio, details = assess(model, delta, before, after, cost, value,
                                                          solution.certificate)
                        if accepted:
                            candidate = trial_poses
                            alpha *= fraction
                            break
            if accepted:
                # A higher-level gain supplies its own gain ratio; tracking
                # cost increases never veto it. Clamp only the damping update.
                update_ratio = float(np.clip(ratio, 0., 2.))
                damping = max(damping * max(1 / 3, 1 - (2 * update_ratio - 1) ** 3), MIN_DAMPING)
                growth = 2.
                break
            damping, growth = min(damping * growth, MAX_DAMPING), growth * 2
        if not accepted:
            history.append(dict(phase=label, iteration=iteration, cost=cost, after=cost, alpha=0.,
                                damping=damping, seconds=time.monotonic() - start,
                                priority_status='no_certified_accepted_proposal',
                                priority_solve_diagnostics=report_value(model.solve_diagnostics), **stats))
            record_stall = getattr(coordinator, 'record_stall', None)
            if record_stall is not None:
                record_stall(model, policy, history[-1])
            break
        assert os.getpid() == owner, 'Speculative workers cannot commit priority credit'
        # Re-evaluation in the parent precedes commit; the pure coordinator can
        # also validate funding against the unchanged begin_iteration model.
        after = evaluate_priority(objective, candidate)
        if not after.valid:
            raise RuntimeError('Priority trial changed without a commit')
        coordinator.certificate = solution.certificate
        coordinator.commit(candidate)
        poses = candidate
        acc, temporal_r, temporal_j, stats, _ = objective.evaluate(poses)
        record = dict(phase=label, iteration=iteration, cost=cost, after=acc.cost, alpha=alpha,
                      damping=damping, step_backend=('priority_zero_yield_descent' if descent else 'priority_conic'),
                      seconds=time.monotonic() - start,
                      priority_certificate=report_value(solution.certificate),
                      priority_slacks=report_value(model.predicted_slacks(delta)),
                      priority_merit=report_value(after.__dict__ if hasattr(after, '__dict__') else after),
                      **details, **stats)
        history.append(record)
        if callback is not None:
            callback(poses.copy(), dict(record))
        if details['improved_level'] is not None:
            # Tracking can rise during penetration/contact progress; it cannot
            # be mistaken for a converged tracking-cost window.
            costs, slow = [acc.cost], 0
        else:
            slow, done = _progress_converged(costs, acc.cost, slow, alpha, True, converged_decrease)
            if done:
                break
    return poses, history
