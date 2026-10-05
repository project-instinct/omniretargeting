"""Original scalar-joint temporal budgets on an unrebased private slice.

Crossing boundary stencils and non-joint signals are omitted in the reduced
predictor. The full QP and nonlinear guards still enforce all of them.
"""
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from .._refinement.math import _difference_operator


def joint_guard(guard, part, sequence):
    coords = guard.tasks.coords if sequence else guard.coords
    count = part.stop - part.start

    def repair_bounds(poses):
        values = poses[:, coords.qi]
        if guard.correction:
            fixed = guard.reference['joints'][0] if sequence else guard.fixed[:, coords.qi]
            values = values - fixed[part]
        linear = []
        for order in (1, 2, 3):
            if count <= order:
                continue
            key = ('joints', order) if sequence else f'joints_order_{order}'
            if key not in guard.bounds:
                continue
            rows = slice(part.start, part.stop-order)
            residual = np.diff(values, n=order, axis=0)
            operator = sparse.kron(_difference_operator(count, order), coords.selector, format='csr')
            if sequence:
                from .._sequence.continuity_guard import LINEARIZATION_SLACK, ACCEPTANCE_SLACK
                budget = guard.bounds[key][rows]
                allowed = np.minimum(np.maximum(budget+LINEARIZATION_SLACK, np.abs(residual)),
                                     budget+ACCEPTANCE_SLACK)
            else:
                from .continuity import ROUND_OFF
                budget = guard.bounds[key][rows]
                allowed = np.minimum(np.maximum(budget+ROUND_OFF/2, np.abs(residual)), budget+ROUND_OFF)
            linear.append((operator, (-allowed-residual).ravel(), (allowed-residual).ravel()))
        return linear, []

    return SimpleNamespace(repair_bounds=repair_bounds)


def differences(limits, part):
    result = []
    width = limits.width
    for operator, budget in limits.differences:
        order = limits.frames - operator.shape[0]//width
        if part.stop-part.start <= order:
            continue
        rows = slice(part.start*width, (part.stop-order)*width)
        cols = slice(part.start*width, part.stop*width)
        result.append((operator[rows, cols], budget[rows]))
    return result
