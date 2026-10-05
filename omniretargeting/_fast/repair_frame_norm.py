"""Exact point cones without Jacobians of fixed, valid frames.

Only pure coordinate projections qualify. Movable frames retain their full
Jacobian columns and the original full-box redundancy calculation. Invalid
constant cones retain their residual/bound and zero projected coefficients.
"""

from copy import copy
from dataclasses import dataclass

import numpy as np
from scipy import sparse


def enabled():
    from .runtime_options import OPTIONS
    return OPTIONS.repair_main_cache


@dataclass
class FrameJacobian:
    values: np.ndarray
    frames: np.ndarray


def point_jacobians(limits, bounds, bodies, points):
    from .repair_projected import ProjectedBounds
    if (not enabled() or not isinstance(bounds, ProjectedBounds)
            or bounds.selected is None):
        return limits.kin.point_jacobians(bodies, points)
    frames = np.flatnonzero(bounds.selected.any(axis=1))
    original = limits.kin
    state = getattr(limits, '_frame_point_kin', None)
    key = frames.tobytes()
    if (state is None or state[0] is not original.jacp
            or state[1] is not original.jacr or state[2] != key):
        kin = copy(original)
        for name in ('xpos', 'xmat', 'geom_xpos', 'geom_xmat', 'jacp', 'jacr'):
            setattr(kin, name, getattr(original, name)[frames])
        state = limits._frame_point_kin = original.jacp, original.jacr, key, kin, {}
    kin, cache = state[3:]
    bodies = np.asarray(bodies)
    local_bodies = bodies[frames] if bodies.ndim == 2 else bodies
    local_points = points[frames]
    signature = (local_bodies.dtype.str, local_bodies.shape, local_bodies.tobytes(),
                 local_points.dtype.str, local_points.shape, local_points.tobytes())
    value = cache.get(signature)
    if value is None:
        value = kin.point_jacobians(local_bodies, local_points)
        value.setflags(write=False)
        cache[signature] = value
    return FrameJacobian(value, frames)


def root_relative(jacobian, share, kin, body):
    if isinstance(jacobian, FrameJacobian):
        frames = jacobian.frames
        return FrameJacobian(jacobian.values - share[frames, None, None, None]
                             * kin.jacp[frames, body, None], frames)
    jacobian -= share[:, None, None, None] * kin.jacp[:, body, None]
    return jacobian


def masked_norm(bounds, jacobian, residual, allowed, mask):
    """The existing np.where arithmetic; Jacobian masking is frame-local."""
    residual = np.where(mask[..., None], residual, 0.)
    allowed = np.where(mask, allowed, 1.)
    if not isinstance(jacobian, FrameJacobian):
        bounds.norm(np.where(mask[..., None, None], jacobian, 0.), residual, allowed)
        return
    values = np.where(mask[jacobian.frames, ..., None, None], jacobian.values, 0.)
    frame_norm(bounds, FrameJacobian(values, jacobian.frames), residual, allowed)


def frame_norm(bounds, jacobian, residual, allowed):
    """Same projected norm rows, residuals and limits in their original order."""
    shape, width = residual.shape, bounds.width
    dimension = shape[-1]
    residual = residual.reshape(-1, dimension)
    base = np.broadcast_to(allowed, shape[:-1]).ravel()
    length = np.linalg.norm(residual, axis=1)
    allowed = np.minimum(np.maximum(base + 5e-8, length), base + 1e-7)
    points = int(np.prod(shape[1:-1]))
    values, frames = jacobian.values, jacobian.frames
    varying = np.zeros((bounds.frames, points), bool)
    varying[frames] = np.any((values != 0.) & bounds.selected[frames, None, None, :], axis=(2, 3))
    keep = varying.ravel() | ~(length <= allowed + 1e-12)
    indices = np.flatnonzero(keep)
    row_frames = indices // points
    local = np.searchsorted(frames, row_frames)
    active = local < len(frames)
    active[active] &= frames[local[active]] == row_frames[active]
    data = np.zeros((len(indices), dimension, width))
    data[active] = values[local[active], indices[active] % points]
    columns = np.broadcast_to(row_frames[:, None, None] * width
                              + np.arange(width), data.shape).ravel().copy()
    pointer = np.arange(len(indices) * dimension + 1) * width
    matrix = sparse.csr_matrix((data.ravel(), columns, pointer),
                              shape=(len(indices) * dimension, bounds.size))
    matrix.eliminate_zeros()
    bounds.norms.append((matrix, residual[keep], allowed[keep]))
