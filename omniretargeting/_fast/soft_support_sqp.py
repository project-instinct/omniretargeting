"""Private whole-trajectory banded projection with support-only protection.

Free neighbours avoid the infeasible short-region boundary seen when a support
is reached after soft bulk. Intermediate SQP states never leave this module.
"""
import time
import numpy as np
from .acquisition_guard import AcquisitionNormalGuard
from .barrier_assembly import physical_system
from .barrier_solver import _equalities
from .constrained_ip import QuadraticDomain, solve
from .constrained_rows import ConstrainedRows
from .constrained_step import step_boxes
from .legacy_window_qp import WindowObjective
from .soft_acquisition_v4 import final_guards
from .soft_bulk import bulk_context
from .soft_projection import certify, feasibility_merit
from .soft_support_guard import support_guard
from .soft_support_rows import SupportWeights
from .solver import _hold


def project(objective, poses, entry, *, rounds=6, private_callback=None):
    started = time.monotonic()
    report = dict(mode='whole_banded_support_sqp', evaluations=0, evaluation_seconds=0.,
                  factorizations=0, band_solves=0, rounds=[])
    frozen = next(g for g in objective.candidate_guards if isinstance(g, AcquisitionNormalGuard))
    selected = support_guard(objective, frozen)
    weights = SupportWeights(anchor_sigma=.0005, observed_anchor_sigma=.002,
                             support_velocity_sigma=.0006, sticking_velocity_sigma=.001)
    original = poses.copy()
    warm = {}
    feasible = None
    with bulk_context(objective, poses, report, clearance_multiplier=4., support_weights=weights) as physical, final_guards(objective):
        objective.candidate_guards = [*objective.candidate_guards, selected]
        if not certify(objective, poses):feasible = poses.copy()
        for number in range(rounds):
            tick = time.monotonic()
            window = WindowObjective(objective, poses, range(len(poses)))
            rows = ConstrainedRows(window)
            families = rows.evaluate(poses, True)
            merit = feasibility_merit(families)
            ev = physical.evaluate(poses)
            acc, residual, temporal, _, _ = ev
            system = physical_system(acc, temporal)
            gradient = acc.gradient.ravel() + temporal.T @ residual
            system[0] += .001 * np.maximum(system[0], 1.)
            lower, upper = step_boxes(window, poses)
            held = _equalities(window, poses)
            _hold(system, held); gradient[held] = 0.
            domain = QuadraticDomain(families, lower, upper, held)
            direction = solve(system, gradient, domain, warm, report)
            accepted = False
            history = dict(iteration=number, feasibility_before=merit, trials=[])
            for alpha in (1., .5, .25, .125):
                trial = objective.project(objective.coords.integrate(poses, direction.reshape(len(poses), -1), alpha))
                values = rows.evaluate(trial, False)
                after = feasibility_merit(values)
                record = dict(alpha=alpha, feasibility_after=after)
                if after < merit * (1.-1e-4*alpha) or after == 0.:
                    value = physical.evaluate(trial, False)
                    record.update(cost=physical.physical_cost(value), stats=value[3], support_cost=value[4]['support_cost'])
                    exact = not certify(objective, trial)
                    if (not exact and merit > 0.) or physical.acceptance_cost(value) < physical.acceptance_cost(ev):
                        poses = trial; accepted = True
                        if exact:feasible = trial.copy()
                history['trials'].append(record)
                if accepted:break
            history.update(seconds=time.monotonic()-tick, accepted=accepted)
            report['rounds'].append(history)
            if private_callback is not None:
                private_callback(poses.copy(), history)
            if not accepted:break
        report['remaining_failures'] = certify(objective, poses)
        if feasible is None:
            if certify(objective, entry):raise RuntimeError('Original entry lost immutable support feasibility')
            feasible = entry.copy(); report['fallback'] = 'immutable_feasible_entry'
    report.update(seconds=time.monotonic()-started, support_points=sum(int(g[5].sum()) for g in selected.groups),
                  correction_distance=float(np.linalg.norm(objective.coords.difference(original, feasible))))
    return feasible, report
