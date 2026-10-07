"""Exact chunk caches for immutable frame-local contact guard rows.

Calls each guard's existing repair_bounds implementation on a private slice,
then restores its original family and frame order. No temporal or layout
guard is sliced here. Those guards keep their complete crossing stencils.
"""

from collections import OrderedDict
from copy import copy
from dataclasses import replace

import numpy as np
from scipy import sparse

from .kinematics import TrajectoryKinematics

CHUNK_FRAMES = 64


def _slice(guard, part):
    local = copy(guard)
    local.kin = TrajectoryKinematics(guard.kin.model, guard.kin.columns)
    groups = []
    for group in guard.groups:
        if isinstance(group, tuple):
            kind, indices, offsets, target, normal, active, held, bound, own = group
            groups.append((kind, indices, offsets if kind == "surface" else offsets[part],
                           target[part], normal[part], active[part], held[part], bound[part], own[part]))
        else:
            names = ("targets", "active", "projected", "normals", "acquired",
                     "positions", "established", "yielded")
            values = {name: getattr(group, name)[part] for name in names}
            if group.kind == "endpoint":
                values["local"] = group.local[part]
            groups.append(replace(group, **values))
    local.groups = tuple(groups)
    return local


def _families(guard):
    linear, norms = [], []
    for index, group in enumerate(guard.groups):
        if isinstance(group, tuple):
            active, held = group[5], group[6]
            if (active & held).any():
                linear.append((index, "normal"))
            if (active & ~held).any():
                norms.append((index, "reaching"))
        else:
            held = group.established
            if held.any():
                linear.append((index, "normal"))
            norms.append((index, "active"))
            if (held & group.projected).any():
                norms.append((index, "tangent"))
            if guard.preserve_positions and held.any():
                norms.append((index, "positions"))
    return linear, norms


def _shift(matrix, start, width):
    value = sparse.csr_matrix(matrix, copy=True)
    value.indices += start
    return sparse.csr_matrix((value.data, value.indices, value.indptr), shape=(value.shape[0], width))


def contact_bounds(guard, poses, *, selector=None):
    """Full-coordinate, byte-equal guard rows, or None for an unknown guard."""
    from .acquisition_guard import AcquisitionNormalGuard
    from .contact_guard import ContactGuard
    if type(guard) not in (AcquisitionNormalGuard, ContactGuard):
        return None
    if guard.priority is not None:
        return [], []
    # Both supported guard constructors own readonly arrays in immutable
    # groups. Retain those objects, so recycled IDs cannot certify new state.
    identity = tuple(guard.groups)
    config = (getattr(guard, "preserve_positions", None), guard.kin.columns.tobytes(), CHUNK_FRAMES,
              False)
    config += (len(poses), getattr(guard, 'numerical_slack', None), selector_key(selector))
    state = getattr(guard, "_repair_contact_cache", None)
    if (state is None or state[1] != config or len(state[0]) != len(identity)
            or any(a is not b for a, b in zip(state[0], identity))):
        state = guard._repair_contact_cache = identity, config, {}
    caches = state[2]
    families = _families(guard)
    collected = [{family: [] for family in names} for names in families]
    width = guard.objective.width
    for start in range(0, len(poses), CHUNK_FRAMES):
        part = slice(start, min(start + CHUNK_FRAMES, len(poses)))
        key = (poses[part].dtype.str, poses[part].tobytes())
        cache = caches.setdefault(start, OrderedDict())
        value = cache.get(key)
        if value is None:
            local = _slice(guard, part)
            result = local.repair_bounds(poses[part])
            result = [[(shift_project(matrix, start * width, len(poses) * width, selector),
                        first, second) for matrix, first, second in rows] for rows in result]
            value = [(names, rows) for names, rows in zip(_families(local), result)]
            cache[key] = value
        cache.move_to_end(key)
        while len(cache) > 3:
            cache.popitem(last=False)
        for output, (names, rows) in zip(collected, value):
            assert len(names) == len(rows), "Contact row families changed"
            for family, (matrix, first, second) in zip(names, rows):
                output[family].append((matrix,
                                       first, second))
    result = []
    for output in collected:
        rows = []
        for chunks in output.values():
            matrices, first, second = zip(*chunks)
            rows.append((sparse.vstack(matrices, format="csr"),
                         np.concatenate(first), np.concatenate(second)))
        result.append(rows)
    return tuple(result)


def selector_key(selector):
    if selector is None:
        return None
    return (selector.shape, selector.dtype.str, selector.data.tobytes(),
            selector.indices.tobytes(), selector.indptr.tobytes())


def shift_project(matrix, start, width, selector):
    matrix = _shift(matrix, start, width)
    return matrix if selector is None else matrix @ selector
