"""Exact material points reused only for identical rigid transforms."""

import numpy as np


def points(limits, index, kind, indices, offsets):
    """Keep the original per-frame einsum; no transform tolerance is used."""
    kin = limits.kin
    indices = np.asarray(indices)
    unique = np.unique(indices)
    if kind == 'surface':
        origin, rotation = kin.geom_xpos[:, unique], kin.geom_xmat[:, unique]
    else:
        origin, rotation = kin.xpos[:, unique], kin.xmat[:, unique]
    identity = (kind, indices.dtype.str, indices.shape, indices.tobytes(),
                offsets.dtype.str, offsets.shape, offsets.tobytes())
    caches = getattr(limits, '_repair_points_cache', None)
    if caches is None:
        caches = limits._repair_points_cache = {}
    previous = caches.get(index)
    if previous is None or previous[0] != identity or previous[1].shape != origin.shape:
        changed = np.ones(len(origin), bool)
        value = None
    else:
        # Equality of IEEE bytes also preserves signed zeros and NaN payloads.
        changed = (np.any(origin.view(np.uint64) != previous[1].view(np.uint64), axis=(1, 2))
                   | np.any(rotation.view(np.uint64) != previous[2].view(np.uint64), axis=(1, 2, 3)))
        value = previous[3].copy()
    if value is None:
        value = np.empty((len(origin), len(indices), 3), dtype=float)
    if changed.any():
        if kind == 'surface':
            value[changed] = kin.geom_xpos[changed][:, indices] + np.einsum(
                'fkij,kj->fki', kin.geom_xmat[changed][:, indices], offsets)
        else:
            body = int(indices[0])
            value[changed, 0] = kin.xpos[changed, body] + np.einsum(
                'fij,fj->fi', kin.xmat[changed, body], offsets[changed])
    stored = value.copy()
    stored.setflags(write=False)
    caches[index] = identity, origin, rotation, stored
    return value
