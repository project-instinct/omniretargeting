"""Exact boolean reuse for the two immutable contact guard families."""

from collections import OrderedDict
from inspect import unwrap

import numpy as np


def accepts(guard, poses):
    from .acquisition_guard import AcquisitionNormalGuard
    from .contact_guard import ContactGuard, NUMERICAL_SLACK
    from .repair_frame_norm import enabled
    if (not enabled() or type(guard) not in (AcquisitionNormalGuard, ContactGuard)
            or guard.priority is not None):
        return guard.accepts(poses)
    if getattr(guard.accepts, '__func__', None) is not type(guard).accepts:
        return guard.accepts(poses)
    for name in ('accepts', 'diagnostics', '_points'):
        method = unwrap(getattr(guard, name))
        if (method.__module__ != type(guard).__module__
                or method.__qualname__ != type(guard).__name__ + '.' + name):
            return guard.accepts(poses)
    groups = tuple(guard.groups)
    config = (getattr(guard, 'shape', None), getattr(guard, 'preserve_positions', None), getattr(guard, 'ablate_contacts', None),
              getattr(guard, 'numerical_slack', NUMERICAL_SLACK),
              guard.kin.columns.tobytes(), guard.accepts.__func__,
              getattr(guard.diagnostics, '__func__', guard.diagnostics),
              getattr(guard._points, '__func__', guard._points))
    state = getattr(guard, '_repair_acceptance_cache', None)
    if (state is None or state[1] != config or len(state[0]) != len(groups)
            or any(a is not b for a, b in zip(state[0], groups))):
        state = guard._repair_acceptance_cache = groups, config, OrderedDict(), {}
    poses = np.asarray(poses)
    key = poses.dtype.str, poses.shape, poses.tobytes()
    cache = state[2]
    if key not in cache:
        cache[key] = _chunk_accepts(guard, poses, state[3])
    cache.move_to_end(key)
    while len(cache) > 4:
        cache.popitem(last=False)
    return cache[key]


def _chunk_accepts(guard, poses, caches):
    """Cache independent frame predicates; retain full-trajectory FK dispatch.

    Slicing FK results rather than evaluating little trajectories preserves
    the established native/fallback choice. Norms/einsum contract only spatial
    coordinates; no arithmetic or decision depends on another chunk.
    """
    from copy import copy
    from .contact_guard import ContactGuard
    from .repair_contact_rows import CHUNK_FRAMES, _slice
    if (poses.ndim != 2 or not np.isfinite(poses).all()
            or isinstance(guard, ContactGuard) and poses.shape != guard.shape):
        return guard.accepts(poses)
    pending = []
    for start in range(0, len(poses), CHUNK_FRAMES):
        part = slice(start, min(start + CHUNK_FRAMES, len(poses)))
        key = poses.dtype.str, poses[part].shape, poses[part].tobytes()
        cache = caches.setdefault(start, OrderedDict())
        if key in cache:
            cache.move_to_end(key)
            if not cache[key]:
                return False
        else:
            pending.append((part, key, cache))
    if not pending:
        return True
    guard.kin.evaluate(poses, with_jacobian=False)
    for part, key, cache in pending:
        local = _slice(guard, part)
        local.kin = copy(guard.kin)
        for name in ('xpos', 'xmat', 'geom_xpos', 'geom_xmat'):
            setattr(local.kin, name, getattr(guard.kin, name)[part])
        # FK is already evaluated, including the original full-frame choice.
        local.kin.evaluate = lambda *args, **kwargs: local.kin
        if isinstance(guard, ContactGuard):
            local.shape = poses[part].shape
            accepted = local._accepts_without_report(poses[part])
        else:
            excess = 0.
            for kind, indices, offsets, target, normal, active, held, bound, yielded in local.groups:
                offset = local._points(kind, indices, offsets) - target
                along = np.sum(offset * normal, axis=-1)
                gap = np.where(held, np.maximum(along - bound - yielded, -along - bound),
                               np.linalg.norm(offset, axis=-1) - bound - yielded)
                if not np.isfinite(gap[active]).all():
                    excess = np.inf
                    break
                if active.any():
                    excess = max(excess, float(np.max(gap[active])))
            accepted = bool(np.isfinite(excess) and excess <= guard.numerical_slack)
        cache[key] = accepted
        while len(cache) > 4:
            cache.popitem(last=False)
        if not accepted:
            return False
    return True
