"""Adaptive exact L1 proposals, with an explicit lexicographic fallback.

Saturation of weighted-QP multipliers is a warning, not a proof of hierarchy.
Positive higher-level optima require solving/certifying those levels directly.
"""

from dataclasses import replace
from copy import copy

import numpy as np
from scipy import sparse

from .qp_types import QPSolution, TRACKING_LEVEL, settings, solve_program
from .qp_residual import kkt_residual
from .priority_bounds import box_level_lower_bounds

MAX_ESCALATIONS = 2
ESCALATION = 10.


def _certified(model, program, P, c, result, stage):
    residual = (dict(certified=False, reason='solver_did_not_solve') if result is None else
                kkt_residual(P, c, program.A, program.b, program.nonneg, program.soc,
                             result.x, result.z, settings()))
    model.solve_diagnostics.append(dict(stage=stage, **program.last_solve, **residual))
    return residual['certified']


def _solve_certified(model, program, P, c, stage):
    result = solve_program(program, P, c)
    if _certified(model, program, P, c, result, stage):
        return result
    # No altered rows, objective, cone or certificate tolerance. A numerical
    # retry is not a feasibility or convergence claim if it fails again.
    result = solve_program(program, P, c, precision=True)
    return result if _certified(model, program, P, c, result, stage + '_precision') else None


def _level_rows(model):
    soft = model.soft
    rows = np.zeros((4, model.program.A.shape[1]))
    for index, (weight, level) in enumerate(zip(soft.weights, soft.levels)):
        if level < TRACKING_LEVEL:
            rows[level, model.gradient.size + index] = weight
    return rows


def _pressure(model, result):
    values = np.zeros(len(model.soft.keys))
    dual = np.asarray(result.z)
    for heads, indices in model.soft.heads:
        np.add.at(values, indices, dual[heads])
    # Back out the original objective scaling and each key's normalization.
    values *= model.scale / np.asarray(model.soft.weights)
    bounds = np.zeros(4)
    levels = np.asarray(model.soft.levels, int)
    high = levels < TRACKING_LEVEL
    np.maximum.at(bounds, levels[high], values[high])
    return bounds


def _freeze(program, row, upper):
    # Insert before the SOCs; keeping whole cones preserves their coordinates.
    at = program.nonneg
    A = sparse.vstack((program.A[:at], sparse.csr_matrix(row[None]), program.A[at:]), format='csc')
    soft = copy(program.soft)
    soft.heads = [(heads + (heads >= at), keys) for heads, keys in soft.heads]
    return replace(program, A=A, b=np.r_[program.b[:at], upper, program.b[at:]], nonneg=at + 1,
                   soft=soft, last_solve={})


def _solution(model, result, coefficients, bounds, method, certified, **extra):
    primal = np.asarray(result.x)
    step = primal[:model.gradient.size]
    values = model.soft.values(step)
    zero = model.soft.values(np.zeros_like(step))
    reductions = {key: zero[key] - values[key] for key in model.soft.keys}
    certificate = dict(coefficients=np.asarray(coefficients).tolist(),
                       multiplier_bounds=np.asarray(bounds).tolist(),
                       certified=bool(certified), method=method,
                       predicted_reductions=dict(levels=model.soft.level_values(reductions).tolist(),
                                                 keys=reductions), **extra)
    return QPSolution(step, dict(zip(model.soft.keys, primal[model.gradient.size:].tolist())),
                      certificate, primal, np.asarray(result.z))


def solve_priority(model, P, c, *, strict=True):
    soft, program = model.soft, model.program
    levels = _level_rows(model)
    coefficients = np.ones(4)
    tolerance = settings()
    normalizers = np.maximum(np.max(levels, axis=1), 1.)
    bounds = np.zeros(4)
    cached = model.priority_cache
    if strict and cached is not None and cached['program'] is program:
        return _finish_strict(model, P, c, cached, reused=True)
    lower_bounds = box_level_lower_bounds(model) if strict else np.zeros(4)
    normalized_lower = np.nextafter(lower_bounds / normalizers, -np.inf)
    skip_weighted = bool(np.any(normalized_lower > tolerance.tol_gap_abs))
    attempts = 0 if skip_weighted else (MAX_ESCALATIONS + 1 if strict else 1)
    for escalation in range(attempts):
        weighted = c + coefficients @ levels / model.scale
        result = solve_program(program, P, weighted)
        if result is None:
            _certified(model, program, P, weighted, result, f'weighted_{escalation}')
            return None
        bounds = _pressure(model, result)
        if not strict:
            return _solution(model, result, coefficients, bounds, 'weighted', False)
        values = levels @ np.asarray(result.x) / normalizers
        # Nonnegative levels have a known zero lower bound. No further solves
        # are needed to establish their optima when all are zero to tolerance.
        zero = values <= tolerance.tol_gap_abs
        saturated = bounds >= coefficients * (1 - 1e-5)
        if zero.all() and not saturated.any() and _certified(model, program, P, weighted, result, 'zero_lower_bound'):
            return _solution(model, result, coefficients, bounds, 'zero_lower_bound', True,
                             escalations=escalation)
        increase = ~zero | saturated
        coefficients[increase] = np.maximum(ESCALATION * coefficients[increase],
                                            2 * bounds[increase] + 1.)

    # Strict hierarchy is the conservative fallback. A level's scalar optimum
    # (sum of unique, normalized key slacks) is frozen, never every witness.
    work = program
    attained, freezes, level_duals = [], [], []
    zero_P = sparse.csc_matrix(P.shape)
    freeze_rows = []
    for level, row in enumerate(levels):
        if not np.any(row):
            attained.append(0.)
            freezes.append(0.)
            level_duals.append(np.zeros(4))
            continue
        normalized = row / normalizers[level]
        result = _solve_certified(model, work, zero_P, normalized, f'level_{level}')
        if result is None:
            return None
        value = max(float(normalized @ result.x), 0.)
        limit = value + tolerance.tol_gap_abs + tolerance.tol_gap_rel * abs(value)
        duals = np.zeros(4)
        for earlier, index in freeze_rows:
            duals[earlier] = result.z[index] * normalizers[level] / normalizers[earlier]
        level_duals.append(duals)
        attained.append(value * normalizers[level])
        freezes.append(limit * normalizers[level])
        freeze_rows.append((level, work.nonneg))
        work = _freeze(work, normalized, limit)
    cached = dict(program=program, work=work, levels=levels, normalizers=normalizers,
                  attained=attained, freezes=freezes, level_duals=level_duals, freeze_rows=freeze_rows,
                  box_lower_bounds=lower_bounds.tolist(), weighted_attempts=attempts)
    model.priority_cache = cached
    return _finish_strict(model, P, c, cached, reused=False)


def _finish_strict(model, P, c, cached, *, reused):
    """Only the tracking QP depends on damping once level optima are certified."""
    work = cached['work']
    normalizers = cached['normalizers']
    freeze_rows, level_duals = cached['freeze_rows'], cached['level_duals']
    tolerance = settings()
    result = _solve_certified(model, work, P, c, 'tracking')
    if result is None:
        return None
    # Relevant lower-level multipliers now have well-defined optimal-value
    # constraints. Include their propagation through the hierarchy when
    # reporting sufficient exact-penalty coefficients for this frozen model.
    bounds = np.zeros(4)
    for level, index in freeze_rows:
        bounds[level] = result.z[index] * model.scale / normalizers[level]
    coefficients = np.zeros(4)
    for level in reversed(range(4)):
        bounds[level] += sum(coefficients[k] * level_duals[k][level] for k in range(level + 1, 4))
        coefficients[level] = max(1., 1.1 * bounds[level] + 1.)
    actual = cached['levels'] @ np.asarray(result.x)
    allowed = np.asarray(cached['freezes']) + normalizers * tolerance.tol_feas
    certified = np.all(actual <= allowed) and np.isfinite(coefficients).all()
    if not certified:
        return None
    return _solution(model, result, coefficients, bounds, 'lexicographic', True,
                     escalations=max(0, cached['weighted_attempts'] - 1), level_optima=cached['attained'],
                     box_lower_bounds=cached['box_lower_bounds'], weighted_attempts=cached['weighted_attempts'],
                     level_upper_bounds=cached['freezes'], reused_level_optima=reused)
