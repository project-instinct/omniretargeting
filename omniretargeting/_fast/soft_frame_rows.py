"""Parallel per-frame layout penalties for every frozen shape envelope."""

import numpy as np


def add_clearance(objective, poses, accumulator, stats):
    """Globally weight clearance families, retaining their original cost."""
    from .rows import _Accumulator
    clearance = _Accumulator(objective.frames, objective.width, accumulator.jacobian)
    objective._clearance_terms(clearance, stats, poses)
    objective._plane_terms(clearance, stats)
    objective._self_collision_terms(poses, clearance, stats)
    multiplier = getattr(objective, '_soft_clearance_multiplier', 1.)
    accumulator.frame_cost += multiplier * clearance.frame_cost
    accumulator.cost = float(accumulator.frame_cost.sum())
    if accumulator.jacobian:
        accumulator.hessian += multiplier * clearance.hessian
        accumulator.gradient += multiplier * clearance.gradient
    accumulator.soft_clearance_cost = clearance.cost
    accumulator.soft_priority_extra = (multiplier - 1.) * clearance.cost


def add_layout(objective, accumulator, stats):
    """Numerical inner-margin rows, separate from original physical terms."""
    from .terms import ENVELOPE_SIGMA
    frames = getattr(objective, 'frame_slice', slice(0, objective.frames))
    weight = objective.envelope_weight * np.sqrt(getattr(objective, '_soft_guard_stiffness', 1.))
    for envelope in objective._soft_bulk_layouts:
        envelope = envelope.for_frames(frames)
        _, _, (change, pairs) = envelope.measure(objective.kin, accumulator.jacobian)
        value, derivative = envelope.layout(change, pairs)
        if value is None:
            continue
        reference = envelope.layout_sign * np.einsum('fpk,fak->fpa', envelope.reference, envelope.axes)
        margin = .01 * np.maximum(reference, 0.) + 1e-7
        residual = np.where(envelope.layout_sign != 0, np.maximum(margin - value, 0.), 0.)
        scale = weight[:, None, None] / ENVELOPE_SIGMA
        jacobian = None
        if accumulator.jacobian:
            jacobian = -derivative * ((residual > 0.) * scale)[..., None]
            jacobian = jacobian.reshape(objective.frames, -1, objective.width)
        residual = (residual * scale).reshape(objective.frames, -1)
        accumulator.add(residual, jacobian)
        accumulator.soft_guard_cost = (getattr(accumulator, 'soft_guard_cost', 0.) +
                                       float(np.square(residual).sum()))
        stats['layout_flip_max'] = max(stats.get('layout_flip_max', 0.),
                                     float(np.maximum(-value, 0.).max(initial=0.)))
    for guard in getattr(objective, 'hinge_guards', ()):
        residual, jacobian = guard.hinge_rows(objective.kin, frames, weight,
            ENVELOPE_SIGMA, accumulator.jacobian, inner_margin=.00025)
        if residual is not None:
            accumulator.add(residual, jacobian)
            accumulator.soft_guard_cost = (getattr(accumulator, 'soft_guard_cost', 0.) +
                                           float(np.square(residual).sum()))
