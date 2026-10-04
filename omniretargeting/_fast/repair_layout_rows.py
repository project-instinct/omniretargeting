"""Exact frame-chunk reuse of fixed-reference layout Jacobians and margins."""

from collections import OrderedDict

import numpy as np
from scipy import sparse

from .kinematics import TrajectoryKinematics
from .repair_contact_rows import CHUNK_FRAMES, _shift
from .reference import ReferenceShape

_REPAIR_BOUNDS = ReferenceShape.repair_bounds


def _identity(envelope, part):
    """Every array read by measure/layout/linear_bounds, including the graph."""
    values = [envelope.bodies, envelope.offsets, envelope.graph.pairs,
              envelope.graph.robot_lengths, envelope.reference[part],
              envelope.axes[part] if envelope.axes is not None else None,
              envelope.layout_sign[part] if envelope.layout_sign is not None else None]
    return (envelope.SLACK, *(None if a is None else (a.shape, a.dtype.str, a.tobytes())
                             for a in values))


def layout_bounds(guard, poses, *, restore=False):
    """Full-coordinate ReferenceShape rows with the established family order."""
    if (type(guard) is not ReferenceShape
            or getattr(guard.repair_bounds, "__func__", None) is not _REPAIR_BOUNDS):
        return None
    state = getattr(guard, "_repair_layout_cache", None)
    if state is None or state[0] != guard.coords.cols.tobytes():
        state = guard._repair_layout_cache = guard.coords.cols.tobytes(), {}
    caches = state[1]
    collected = [[] for _ in guard.envelopes]
    guard.kin.evaluate(poses)
    for start in range(0, len(poses), CHUNK_FRAMES):
        part = slice(start, min(start + CHUNK_FRAMES, len(poses)))
        key = (poses.dtype.str, poses[part].tobytes(), restore,
               tuple(_identity(e, part) for e in guard.envelopes))
        cache = caches.setdefault(start, OrderedDict())
        value = cache.get(key)
        if value is None:
            # Slice the same full-trajectory FK arrays, so chunks retain even
            # the established native/fallback choice and its floating point.
            kin = TrajectoryKinematics(guard.kin.model, guard.kin.columns)
            for name in ("xpos", "xmat", "geom_xpos", "geom_xmat", "jacp", "jacr"):
                setattr(kin, name, getattr(guard.kin, name)[part])
            kin.with_jacobian = True
            value = [e.for_frames(part).linear_bounds(kin, guard.coords.width, restore=restore)
                     for e in guard.envelopes]
            cache[key] = value
        cache.move_to_end(key)
        while len(cache) > 3:
            cache.popitem(last=False)
        for output, rows in zip(collected, value):
            for matrix, low, high in rows:
                output.append((_shift(matrix, start * guard.coords.width,
                                      len(poses) * guard.coords.width), low, high))
    linear = []
    for chunks in collected:
        if chunks:
            matrices, low, high = zip(*chunks)
            linear.append((sparse.vstack(matrices, format="csr"),
                           np.concatenate(low), np.concatenate(high)))
    return linear, []
