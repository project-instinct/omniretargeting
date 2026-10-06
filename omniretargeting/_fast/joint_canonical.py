"""Reuse the whole acquisition packing for joint proposals, adding split cones."""
from copy import copy

import cvxpy as cp
import numpy as np


def constraints(objective, poses, lower, upper, flat):
    from .whole_canonical import _constraint_data
    base = copy(objective)
    base.joint_witness = None
    base.candidate_guards = [g for g in objective.candidate_guards if g is not objective.joint_witness]
    data = _constraint_data(base, poses, lower, upper)
    matrix, bound, dims = data['A'], data['b'], data['dims']
    count = dims.nonneg
    result = [matrix[:count] @ flat <= bound[:count]]
    starts = count + np.r_[0, np.cumsum(dims.soc[:-1])]
    sizes = np.asarray(dims.soc)
    for size in np.unique(sizes):
        heads = starts[sizes == size]
        tails = (heads[:, None] + np.arange(1, size)).ravel()
        vectors = cp.reshape(bound[tails] - matrix[tails] @ flat, (size - 1, len(heads)), order='F')
        result.append(cp.SOC(bound[heads] - matrix[heads] @ flat, vectors, axis=0))
    return result + objective.joint_witness.constraints(poses, flat)
