"""Exact window restrictions of immutable contact groups; no target rebasing."""

from copy import copy
from dataclasses import replace

import numpy as np

from .kinematics import TrajectoryKinematics


def checked_frames(frames, total):
    if (not isinstance(frames, range) or frames.step != 1 or not len(frames)
            or frames.start < 0 or frames.stop > total):
        raise ValueError('frames must be a nonempty absolute unit-step range')
    return slice(frames.start, frames.stop)


def evaluate_contacts(adapter, poses, frames, *, jacobian=True):
    part = checked_frames(frames, len(poses))
    local = copy(adapter)
    guard = copy(adapter.guard)
    local.guard = guard
    # A bounded local FK cache must not replace a shared whole-trajectory kin.
    window_kin = getattr(adapter.objective, 'window_kinematics', None)
    guard.kin = (TrajectoryKinematics(adapter.model, adapter.objective.coords.cols)
                 if window_kin is None else window_kin(poses, frames, with_jacobian=jacobian))
    groups = []
    acquired = []
    for group in adapter.guard.groups:
        acquired.append(not isinstance(group, tuple))
        if isinstance(group, tuple):
            kind, indices, offsets, target, normal, active, held, bound, own = group
            groups.append((kind, indices, offsets if kind == 'surface' else offsets[part],
                           target[part], normal[part], active[part], held[part], bound[part], own[part]))
        else:
            names = ('targets', 'active', 'projected', 'normals', 'acquired',
                     'positions', 'established', 'yielded')
            values = {name: getattr(group, name)[part] for name in names}
            if group.kind == 'endpoint':
                values['local'] = group.local[part]
            groups.append(replace(group, **values))
    guard.groups = tuple(groups)
    local.ranks = [rank[part] for rank in adapter.ranks]
    blocks = local.evaluate(np.asarray(poses)[part], jacobian=jacobian)
    result = []
    for block in blocks:
        identities = tuple((name, group, f + frames.start, sample, channel)
                           for name, group, f, sample, channel in block.identities)
        block = replace(block, frames=block.frames + frames.start, identities=identities)
        result.append(block)
        if block.scalar and identities[0][-1] == 'normal':
            # Restore both sides of the ORIGINAL guard. Acquisition inward
            # motion has no own-depth credit; ContactGuard's absolute normal
            # bound grants its frozen allowance on either side.
            inward = tuple((*identity[:-1], 'normal_inward') for identity in identities)
            own = block.own.copy() if acquired[identities[0][1]] else np.zeros_like(block.own)
            result.append(replace(block, matrix=None if block.matrix is None else -block.matrix,
                                  residual=-block.residual, own=own, identities=inward))
    return result


def frozen_bounds(adapter, poses, frames):
    """Original guard tolerances, with its half-tolerance reserve."""
    linear, cones = [], []
    for block in adapter.evaluate(poses, frames=frames):
        base = block.bound + block.own
        allowed = np.minimum(np.maximum(base + 5e-8, block.values), base + 1e-7)
        if block.scalar:
            linear.append((block.matrix, np.full(len(base), -np.inf), allowed - block.residual))
        else:
            cones.append((block.matrix, block.residual, allowed))
    return linear, cones


def frozen_violations(adapter, poses, frames):
    bad = np.zeros(len(poses), bool)
    for block in adapter.evaluate(poses, jacobian=False, frames=frames):
        values = block.values
        violated = ~np.isfinite(values) | (values > block.bound + block.own + 1e-7)
        bad[block.frames[violated]] = True
    return bad
