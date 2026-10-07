"""Opt-in acquisition trial-point correction; production entry is unchanged.

A rejected full constrained direction can cross frozen contact bounds through
kinematic curvature. Replace only violated contact rows/cones by their trial
linearization, translated into the origin's step coordinates, then solve the
same objective once. Every original nonlinear guard still decides acceptance.
The benchmark harness selects this prototype explicitly (mode='soc').
"""
from contextlib import contextmanager
import time

import numpy as np
from scipy import sparse

INWARD_RESERVE_M = 1e-6


def translated_bounds(origin_bounds, trial_bounds, displacement,
                      reserve=INWARD_RESERVE_M, *, all_rows=False):
    """Replace violated rows, keeping cone order and unviolated origin rows.

    For a trial x+d and a new origin step p, the trial correction is p-d:
    lo <= J_trial(p-d) <= hi; ||r_trial+J_trial(p-d)|| <= allowance.
    The reserve tightens a proposal only; no acceptance tolerance changes.
    """
    linear, norms = [], []
    changed = 0
    for (a, lo, hi), (j, trial_lo, trial_hi) in zip(origin_bounds[0], trial_bounds[0], strict=True):
        bad_lo, bad_hi = trial_lo > 0., trial_hi < 0.
        bad = np.ones_like(bad_lo) if all_rows else bad_lo | bad_hi
        near_lo = trial_lo > -reserve if all_rows else bad_lo
        near_hi = trial_hi < reserve if all_rows else bad_hi
        shift = np.asarray(j @ displacement).ravel()
        matrix = sparse.diags((~bad).astype(float)) @ a + sparse.diags((bad).astype(float)) @ j
        lower = np.where(bad, trial_lo + shift + reserve * near_lo, lo)
        upper = np.where(bad, trial_hi + shift - reserve * near_hi, hi)
        linear.append((matrix.tocsr(), lower, upper))
        changed += int(bad.sum())
    for (a, r, allowed), (j, trial_r, trial_allowed) in zip(origin_bounds[1], trial_bounds[1], strict=True):
        gap = np.linalg.norm(trial_r, axis=1) - trial_allowed
        bad = np.ones(len(gap), bool) if all_rows else gap > 0.
        near = gap > -reserve if all_rows else bad
        dimensions = r.shape[1]
        mask = np.repeat(bad, dimensions)
        matrix = sparse.diags((~mask).astype(float)) @ a + sparse.diags((mask).astype(float)) @ j
        shifted = trial_r - np.asarray(j @ displacement).reshape(trial_r.shape)
        norms.append((matrix.tocsr(), np.where(bad[:, None], shifted, r),
                      np.where(bad, trial_allowed - reserve * near, allowed)))
        changed += int(bad.sum())
    return (linear, norms), changed


@contextmanager
def _override_bounds(guard, bounds):
    saved = guard.__dict__.get('repair_bounds')
    guard.repair_bounds = lambda poses: bounds
    try:
        yield
    finally:
        if saved is None:
            guard.__dict__.pop('repair_bounds', None)
        else:
            guard.repair_bounds = saved


def transport_trial_bounds(coords, origin, trial, displacement, bounds):
    """Express trial Jacobians in origin step coordinates, including SO(3).

    Scalar displacements add; free-root angular increments do not. Differentiate
    integrate(origin, p) at p=d into the trial tangent basis. Central differences
    only form the 3x3 root block; all joint and translation blocks stay identity.
    """
    root_d = getattr(coords, 'root_d', None)
    if root_d is None:
        return bounds
    columns = [coords.inverse[d] for d in range(root_d + 3, root_d + 6)
               if d in coords.inverse]
    if not columns:
        return bounds
    frames, width = len(origin), coords.width
    d = np.asarray(displacement).reshape(frames, width)
    epsilon = 1e-6
    block = np.empty((frames, len(columns), len(columns)))
    for j, column in enumerate(columns):
        plus, minus = d.copy(), d.copy()
        plus[:, column] += epsilon
        minus[:, column] -= epsilon
        forward = coords.difference(trial, coords.integrate(origin, plus, 1.))
        backward = coords.difference(trial, coords.integrate(origin, minus, 1.))
        block[:, :, j] = ((forward - backward) / (2 * epsilon))[:, columns]
    indices = np.arange(frames)[:, None] * width + columns
    row = np.broadcast_to(indices[:, :, None], block.shape).ravel()
    col = np.broadcast_to(indices[:, None, :], block.shape).ravel()
    correction = block - np.eye(len(columns))[None]
    transport = sparse.eye(frames * width, format='csr') + sparse.csr_matrix(
        (correction.ravel(), (row, col)), shape=(frames * width, frames * width))
    linear = [(matrix @ transport, lo, hi) for matrix, lo, hi in bounds[0]]
    norms = [(matrix @ transport, residual, allowance)
             for matrix, residual, allowance in bounds[1]]
    return linear, norms


def projection_metric(objective, poses, step):
    """Closest-to-trial correction has no temporal Gauss-Newton Hessian band.

    Original velocity/layout/contact constraints remain in the canonical
    program. Scaling uses the unchanged per-iteration coordinate box.
    """
    lo, hi = objective.coords.bounds(poses, objective.cfg['angular_step_rad'],
                                    objective.cfg['translation_step_m'])
    scale = np.maximum(np.maximum(abs(lo), abs(hi)), 1e-6).ravel()
    metric = (1 / scale**2)[None, :]
    return metric, -metric[0] * np.asarray(step).ravel()


def conic_proposal(objective, poses, system, gradient, diagonal, damping, guards,
                   *, mode='soc'):
    """One corrected re-solve, falling back to the original admissible fraction."""
    from . import solver
    from .shape_step import bounded_shape_step, restore_shape_step
    from .acquisition_guard import AcquisitionNormalGuard
    from .whole_canonical import supported
    if mode not in ('soc', 'soc_all', 'soc_projection', 'soc_projection2'):
        raise ValueError(f'Unknown acquisition prototype: {mode}')
    trace = getattr(objective, 'performance_trace', None)
    if trace is not None:
        trace.record_call('proposal', 'shape_conic')
    step = bounded_shape_step(objective, poses, system, gradient, diagonal, damping)
    if step is None:
        return None
    candidate, alpha = solver.backtrack(objective, poses, step, guards)
    events = objective.__dict__.setdefault('_acquisition_soc_events', [])
    record = dict(original_alpha=alpha, correction_solves=0, corrected_full=False)
    events.append(record)
    if alpha < 1. and supported(objective):
        trial = objective.project(objective.coords.integrate(
            poses, step.reshape(len(poses), -1), 1.))
        contact = next((g for g in guards if type(g) is AcquisitionNormalGuard), None)
        if contact is not None and not contact.accepts(trial):
            tick = time.monotonic()
            correction_target = step
            for correction_index in range(2 if mode == 'soc_projection2' else 1):
                displacement = objective.coords.difference(poses, trial).ravel()
                trial_bounds = contact.repair_bounds(trial)
                if mode != 'soc':
                    trial_bounds = transport_trial_bounds(objective.coords, poses, trial,
                                                          displacement, trial_bounds)
                bounds, changed = translated_bounds(contact.repair_bounds(poses),
                                                     trial_bounds, displacement,
                                                     all_rows=mode != 'soc')
                record['changed_rows_or_cones'] = changed
                if not changed:
                    break
                record['correction_solves'] += 1
                if trace is not None:
                    trace.record_call('proposal', 'acquisition_soc')
                with _override_bounds(contact, bounds):
                    if mode.startswith('soc_projection'):
                        metric, target_gradient = projection_metric(objective, poses,
                                                                    correction_target)
                        corrected = bounded_shape_step(objective, poses, metric, target_gradient,
                                                       metric[0], 0.)
                    else:
                        corrected = bounded_shape_step(objective, poses, system, gradient,
                                                       diagonal, damping)
                if corrected is None:
                    break
                full = objective.project(objective.coords.integrate(
                    poses, corrected.reshape(len(poses), -1), 1.))
                record['corrected_normal'] = contact.diagnostics(full)
                record['corrected_guards'] = {type(g).__name__:bool(g.accepts(full)) for g in guards}
                if all(record['corrected_guards'].values()):
                    full_cost = float(objective.evaluate(full, with_jacobian=False)[0].cost)
                    fractional_cost = (float('inf') if candidate is None else
                                       float(objective.evaluate(candidate, with_jacobian=False)[0].cost))
                    record.update(full_cost=full_cost, fractional_cost=fractional_cost)
                    if full_cost < fractional_cost:
                        step, candidate, alpha = corrected, full, 1.
                        record['corrected_full'] = True
                    break
                trial = full
                correction_target = displacement = objective.coords.difference(poses, full).ravel()
            record['correction_seconds'] = time.monotonic() - tick
    if alpha < .25:
        restored = restore_shape_step(objective, poses, step)
        if restored is not None and all(g.accepts(restored) for g in guards):
            candidate, alpha = restored, 1.
    return step, candidate, alpha
