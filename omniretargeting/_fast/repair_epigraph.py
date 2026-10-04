"""Default-off elimination of restoration's norm epigraph variables.

Each scalar t has only t <= budget and (t, affine vector) in a SOC. Existential
elimination replaces it with (budget, affine vector) in the same SOC. This
preserves the projected feasible set but changes the numerical solver path.
"""
from types import SimpleNamespace

import numpy as np
from scipy import sparse

REDUCED_LARGE_RESTORATION = False
MIN_VARIABLES = 2048


def eliminate_epigraph(data, variables):
    auxiliary = len(data['c']) - variables
    sizes = data['dims'].soc
    assert auxiliary == len(sizes)
    assert not np.any(data['c'][variables:])
    assert data['P'][variables:].nnz == 0 and data['P'][:, variables:].nnz == 0
    nonnegative = data['dims'].nonneg
    linear = nonnegative - auxiliary
    bound = data['A'][linear:nonnegative]
    assert bound[:, :variables].nnz == 0
    assert (bound[:, variables:] - sparse.eye(auxiliary, format='csc')).nnz == 0
    cone = data['A'][nonnegative:]
    leading = np.r_[0, np.cumsum(sizes)[:-1]]
    assert cone[leading, :variables].nnz == 0
    assert not np.any(data['b'][nonnegative:][leading])
    expected = sparse.csc_matrix((-np.ones(auxiliary), (leading, np.arange(auxiliary))),
                                 shape=(cone.shape[0], auxiliary))
    assert (cone[:, variables:] - expected).nnz == 0
    offset = data['b'][nonnegative:].copy()
    offset[leading] = data['b'][linear:nonnegative]
    return dict(P=data['P'][:variables, :variables], c=data['c'][:variables],
                A=sparse.vstack([data['A'][:linear, :variables], cone[:, :variables]], format='csc'),
                b=np.r_[data['b'][:linear], offset],
                dims=SimpleNamespace(zero=0, nonneg=linear, soc=list(sizes),
                                     psd=[], exp=0, p3d=[], pnd=[]))
