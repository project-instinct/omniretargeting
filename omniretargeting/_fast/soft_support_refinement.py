"""Coupled banded support refinement inside the frozen final guard set.

Private KKT trials contain the original physical rows and source-world support
rows. Numerical weights never change continuity, layout or root domains.
"""
import time
import numpy as np

from .barrier_assembly import physical_system
from .barrier_solver import _equalities
from .constrained_ip import QuadraticDomain, solve
from .constrained_rows import ConstrainedRows
from .constrained_step import step_boxes
from .legacy_window_qp import WindowObjective
from .soft_acquisition_v4 import final_guards
from .soft_bulk import bulk_context
from .soft_projection import certify, project_roots
from .soft_support_rows import SupportWeights
from .solver import _hold


def refine(objective, poses, *, rounds=3, roots=True, damping=.0001):
    started = time.monotonic()
    report = dict(mode='banded_coupled_support_refinement', evaluations=0,
                  evaluation_seconds=0., factorizations=0, band_solves=0, trials=[])
    weights = SupportWeights(anchor_sigma=.0002, support_velocity_sigma=.0006,
                             sticking_velocity_sigma=.001, observed_anchor_sigma=.002)
    warm = {}
    with bulk_context(objective, poses, report, clearance_multiplier=4.,
                      support_weights=weights) as physical, final_guards(objective):
        evaluation = physical.evaluate(poses)
        for number in range(rounds):
            tick = time.monotonic()
            window = WindowObjective(objective, poses, range(len(poses)))
            rows = ConstrainedRows(window)
            acc, residual, temporal, stats, _ = evaluation
            system = physical_system(acc, temporal)
            gradient = acc.gradient.ravel() + temporal.T @ residual
            diagonal = np.maximum(system[0], 1.)
            system[0] += damping * diagonal
            lower, upper = step_boxes(window, poses)
            held = _equalities(window, poses)
            if not roots:
                root_cols = sorted(set(range(window.width)) - set(window.coords.scalar_cols))
                from .soft_support_projection import ContactProjection
                dependency = ContactProjection(objective, poses, np.ones((len(poses), window.width)))
                root_cols += np.asarray(window.coords.scalar_cols)[~dependency.enabled].tolist()
                held = np.unique(np.r_[held, (np.arange(len(poses))[:, None] * window.width + root_cols).ravel()])
            _hold(system, held); gradient[held] = 0.
            families = rows.evaluate(poses, True)
            domain = QuadraticDomain(families, lower, upper, held)
            direction = solve(system, gradient, domain, warm, report)
            accepted = False
            for alpha in (1., .5, .25, .125):
                trial = objective.project(objective.coords.integrate(poses, direction.reshape(len(poses), -1), alpha))
                failures = certify(objective, trial)
                record = dict(round=number, alpha=alpha, failures=failures)
                if alpha == 1. and roots and failures and 'ReferenceShape' not in failures:
                    correction_report = {}
                    trial = project_roots(objective, trial, trial.copy(), poses, report=correction_report)
                    failures = certify(objective, trial)
                    record.update(root_correction=correction_report, corrected_failures=failures)
                if failures:
                    from .barrier_rows import minimum
                    record['binding'] = minimum(rows.evaluate(trial, False))
                if not failures:
                    value = physical.evaluate(trial, False)
                    record.update(cost=physical.physical_cost(value), stats=value[3],
                                  support_cost=value[4]['support_cost'])
                    clearance = (value[3]['scene_depth_m'] <= stats['scene_depth_m'] + .001 and
                                 value[3]['self_depth_m'] <= stats['self_depth_m'] + .0005)
                    if clearance and physical.acceptance_cost(value) < physical.acceptance_cost(evaluation):
                        poses = trial
                        evaluation = physical.evaluate(poses)
                        accepted = True
                report['trials'].append(record)
                if accepted: break
            report.setdefault('round_seconds', []).append(time.monotonic()-tick)
            if not accepted: break
    report['seconds'] = time.monotonic() - started
    return poses, report
