"""Reliable per-body collision depths for frozen repair contact budgets."""

import numpy as np


def reliable_body_depth(model, frames, measurement):
    if measurement.get('failures') or not measurement.get('complete', True):
        raise ValueError('Contact yield requires a complete, reliable measurement')
    depth = np.zeros((frames, model.nbody))
    for key, value in measurement['depths'].items():
        if key[1] not in ('static', 'moving'):
            continue
        if not np.isfinite(value) or value < 0:
            raise ValueError('Contact yield requires finite, nonnegative penetration')
        body = model.body_weldid[model.geom_bodyid[key[-1]]]
        depth[key[0], body] = max(depth[key[0], body], value)
    depth.setflags(write=False)
    return depth
