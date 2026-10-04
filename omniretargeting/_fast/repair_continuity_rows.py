"""Reuse identical temporal Jacobians without changing derivative arithmetic.

Residuals, allowances and complete crossing stencils are recomputed by the
established expressions. Only the sparse operator/Jacobian product is cached,
against every coefficient it reads; immutable references are never rebased.
"""

from collections import OrderedDict

import numpy as np
from scipy import sparse

from .._sequence.continuity_guard import (
    ACCEPTANCE_SLACK, LINEARIZATION_SLACK, SequenceContinuityGuard, _change_size,
)
from .guard_signals import CachedSequenceContinuityGuard

_REPAIR_BOUNDS = SequenceContinuityGuard.repair_bounds


def continuity_bounds(guard, poses):
    """Established full-coordinate rows, or None for a custom implementation."""
    if (type(guard) not in (SequenceContinuityGuard, CachedSequenceContinuityGuard)
            or getattr(guard.repair_bounds, "__func__", None) is not _REPAIR_BOUNDS):
        return None
    signals = guard.signals(poses)
    caches = getattr(guard, "_repair_continuity_cache", None)
    if caches is None:
        caches = guard._repair_continuity_cache = {}
    count = len(poses)
    linear, norms, signatures = [], [], {}
    for (name, order), bound in guard.bounds.items():
        if count <= order:
            continue
        values, jac = signals[name]
        cache = caches.setdefault((name, order), OrderedDict())
        if name not in signatures:
            # Zero frame stride proves every frame reads the same coefficients.
            # Certify that representative's current bytes, including mutations.
            # Other signals serialize once, shared across their derivative orders.
            broadcast = jac.strides[0] == 0
            coefficients = jac[0].tobytes() if broadcast else jac.tobytes()
            signatures[name] = (count, values.shape[1], jac.shape, jac.dtype.str,
                                broadcast, coefficients)
        key = signatures[name]
        matrix = cache.get(key)
        if matrix is None:
            coefficients = {1: [-1, 1], 2: [1, -2, 1], 3: [-1, 3, -3, 1]}[order]
            difference = sparse.diags(coefficients, np.arange(order + 1),
                                      shape=(count - order, count), format="csr", dtype=float)
            operator = sparse.kron(difference, sparse.eye(values.shape[1]), format="csr")
            matrix = operator @ sparse.block_diag(list(jac), format="csr")
            cache[key] = matrix
        cache.move_to_end(key)
        while len(cache) > 2:
            cache.popitem(last=False)
        residual = np.diff(values, n=order, axis=0)
        allowance = np.minimum(
            np.maximum(bound + LINEARIZATION_SLACK, _change_size(name, residual)),
            bound + ACCEPTANCE_SLACK,
        )
        if name == "joints":
            linear.append((matrix, -allowance.ravel() - residual.ravel(),
                           allowance.ravel() - residual.ravel()))
        else:
            norms.append((matrix, residual, allowance))
    return linear, norms
