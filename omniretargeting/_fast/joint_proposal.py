"""Finite globalization and one all-row second-order joint correction."""
import cvxpy as cp
import numpy as np
from scipy import sparse

from ..conic_solver import solve_conic
from .joint_model import pair_indices


def correct(joint, origin, step):
    from .shape_step import _guard_constraints, _pin_root
    obj, coords = joint.objective, joint.objective.coords
    trial = obj.project(coords.integrate(origin, np.asarray(step).reshape(len(origin), -1), 1.))
    # Calibrate nonlinear remainder before assembling the correction cones.
    joint.support.accepts(trial)
    measured = joint.measure(trial, True)
    keys, raw, jac, frames = joint.rows(measured)
    pairs = pair_indices(keys)
    size, width = len(origin) * obj.width, obj.width
    cols = frames[:, None] * width + np.arange(width)
    matrix = sparse.csr_matrix((jac.ravel(), (np.repeat(np.arange(len(raw)), width), cols.ravel())), shape=(len(raw), size))
    delta = cp.Variable(size)
    displacement = coords.difference(origin, trial)
    lo, hi = coords.bounds(origin, obj.cfg['angular_step_rad'], obj.cfg['translation_step_m'])
    lo, hi = lo - displacement, hi - displacement
    lo[:, coords.scalar_cols] = np.maximum(lo[:, coords.scalar_cols], obj.lower - trial[:, coords.qi])
    hi[:, coords.scalar_cols] = np.minimum(hi[:, coords.scalar_cols], obj.upper - trial[:, coords.qi])
    _pin_root(obj, trial, lo, hi)
    constraints = [delta >= lo.ravel(), delta <= hi.ravel()]
    constraints += obj.source_shape.constraints(trial, delta, np.maximum(abs(lo), abs(hi)))
    constraints += _guard_constraints(obj.candidate_guards, trial, delta)
    penalty = 0.
    if len(raw):
        z = cp.Variable(int(pairs.max()) + 1, nonneg=True)
        constraints.append(raw + matrix @ delta <= z[pairs])
        penalty = cp.sum_squares(z / joint.sigma)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(delta) + penalty), constraints)
    try:
        solve_conic(problem, solver=cp.CLARABEL, direct_solve_method='qdldl')
    except cp.error.SolverError:
        return None
    if delta.value is None or problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return None
    candidate = obj.project(coords.integrate(trial, delta.value.reshape(len(origin), width), 1.))
    from .proposals import guards_of
    return candidate if all(g.accepts(candidate) for g in guards_of(obj)[1]) else None


def globalize(objective, origin, step, guards):
    from .proposals import backtrack
    candidate, alpha = backtrack(objective, origin, step, guards)
    if candidate is not None:
        return step, candidate, alpha
    restored = objective.joint_witness.correct(origin, step)
    return step, restored, 1. if restored is not None else 0.


def shorten(objective, origin, step, alpha, acc, temporal_j, gradient, cost):
    from .solver import _model_decrease, CONIC_FRACTIONS
    from .proposals import guards_of
    joint, coords = objective.joint_witness, objective.coords
    guards = guards_of(objective)[1]
    for fraction in CONIC_FRACTIONS:
        scaled = (alpha or 1.) * fraction
        candidate = objective.project(coords.integrate(origin, np.asarray(step).reshape(len(origin), -1), scaled))
        if not all(g.accepts(candidate) for g in guards):
            candidate = joint.correct(origin, np.asarray(step) * scaled)
            if candidate is None:
                continue
        delta = coords.difference(origin, candidate).ravel()
        prediction = _model_decrease(acc, temporal_j, delta, gradient)
        trial = objective.evaluate(candidate)
        ratio = (cost - trial[0].cost) / prediction if prediction > 0 else -1.
        if np.isfinite(trial[0].cost) and trial[0].cost < cost and ratio > 0:
            return candidate, scaled, trial, ratio
        joint.rejections['exact_cost_no_decrease'] += 1
    return None


def banded_candidate(joint, poses, step, guards):
    from .proposals import ALPHAS
    objective = joint.objective
    for alpha in ALPHAS:
        candidate = objective.project(objective.coords.integrate(
            poses, step.reshape(objective.frames, objective.width), alpha))
        if joint.linear_feasible(candidate) and all(g.accepts(candidate) for g in guards):
            return candidate, alpha
    return None, 0.
