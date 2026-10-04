"""Primal-feasible clearance proposals; nonlinear descent certification is R1's.

These proposals do not claim lexicographic or tracking optimality. The full
original program and exact nonlinear zero-yield contract still veto trials.
"""

import numpy as np

from .qp_residual import cone_violation
from .qp_types import QPSolution, settings, solve_program

NORMALIZED_PENALTY = 10.
PENALTY_GROWTH = 10.
STEP_BOX_TOLERANCE = 1e-9


def original_primal_certificate(model, delta):
    """Check the complete original linearized domain at the executed step.

    Epigraph values are their exact per-key max across all scalar/norm
    witnesses. No row/cone, joint box or coordinate bound is omitted.
    This is a feasibility check, not an optimality/convergence certificate.
    """
    delta = np.asarray(delta, float).ravel()
    invalid = dict(primal_feasible=False, optimality_certified=False)
    if delta.shape != (model.gradient.size,) or not np.isfinite(delta).all():
        return dict(invalid, reason='invalid_step')
    lower, upper = model.step_lower.ravel(), model.step_upper.ravel()
    if lower.shape != delta.shape or upper.shape != delta.shape:
        return dict(invalid, reason='unknown_step_domain')
    values = model.predicted_slacks(delta)
    x = np.r_[delta, [values[key] for key in model.soft.keys]]
    if not np.isfinite(x).all():
        return dict(invalid, reason='nonfinite_epigraph')
    program = model.program
    ax = program.A @ x
    if not np.isfinite(ax).all() or not np.isfinite(program.b).all():
        return dict(invalid, reason='nonfinite_program_evaluation')
    violation = cone_violation(program.b - ax, program.nonneg, program.soc)
    scale = max(1., float(np.max(abs(program.b), initial=0.)), float(np.max(abs(ax), initial=0.)))
    primal = violation / scale
    box = max(float(np.max(lower - delta, initial=0.)), float(np.max(delta - upper, initial=0.)))
    tolerance = settings().tol_feas
    valid = np.isfinite(primal) and primal <= tolerance and box <= STEP_BOX_TOLERANCE
    return dict(primal_feasible=bool(valid), optimality_certified=False,
                primal=float(primal), raw_cone_violation=float(violation), primal_scale=scale,
                primal_tolerance=tolerance, box_violation=box, box_tolerance=STEP_BOX_TOLERANCE)


def solve_descent(model, P, c, *, penalty=NORMALIZED_PENALTY):
    """Minimize the original GN model plus L1 to propose an improving step.

    The generic normalized penalty is only a proposal parameter. Strict
    measured/predicted clearance descent and no extra contact yield must
    subsequently be certified by the coordinator for the actual trial.
    """
    coefficients = np.array([penalty * model.scale, 1., 1., 1.])
    weighted = np.array(c, copy=True)
    weighted[model.gradient.size:] += (np.asarray(model.soft.weights) *
                                      model.soft.high_coefficients(coefficients) / model.scale)
    for precision in (False, True):
        result = solve_program(model.program, P, weighted, precision=precision)
        if result is None:
            check = dict(primal_feasible=False, reason='solver_did_not_solve')
        else:
            step = np.asarray(result.x[:model.gradient.size])
            check = original_primal_certificate(model, step)
        model.solve_diagnostics.append(dict(stage='zero_yield_candidate',
                                            **model.program.last_solve, **check))
        if check['primal_feasible']:
            reductions = model.predicted_reductions(step)
            certificate = dict(kind='zero_additional_contact_yield_candidate', certified=False,
                               descent_certified=False, optimality_certified=False,
                               primal_feasible=True, original_primal=check,
                               coefficients=coefficients.tolist(), proposal_penalty=penalty,
                               predicted_reductions=dict(keys=reductions,
                                   levels=model.soft.level_values(reductions).tolist()))
            return QPSolution(step, model.predicted_slacks(step), certificate,
                              np.asarray(result.x), np.asarray(result.z))
    return None
