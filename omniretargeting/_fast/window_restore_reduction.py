"""Exact elimination of unpenalized norm epigraphs in window restorations.

The canonical program has ||J x + r|| <= t <= radius for every norm.
Substitution gives ||J x + r|| <= radius, with precisely the same pose
domain and quadratic objective. In particular, constraint generation can
now screen norms by their actual slack, rather than by the artificial t=0
starting point. Proposal programs and nonlinear guards are unchanged.
"""

from collections import Counter
import os
from types import SimpleNamespace

import numpy as np
from scipy import sparse

EXACT_WINDOW_RESTORATION = os.environ.get(
    'OMNIRETARGETING_FAST_EXACT_WINDOW_RESTORATION', '0').lower() in ('1', 'true', 'yes')
DIAGNOSTICS = Counter()


def reduce_epigraphs(data, variables):
    """Eliminate only the established canonical assembler's private t block.

    Reject unexpected packing rather than silently modifying another conic
    contract. No approximate coefficient comparison or budget change occurs.
    """
    dims = data['dims']
    sizes = np.asarray(dims.soc, dtype=int)
    count = len(data['c']) - variables
    if dims.zero or count != len(sizes) or np.any(sizes < 2):
        raise ValueError('Unexpected window norm epigraph packing')
    if not count:
        return data
    matrix = data['A'].tocsr()
    linear = matrix[:dims.nonneg]
    aux = linear[:, variables:].tocoo()
    if (aux.nnz != count or not np.array_equal(np.sort(aux.col), np.arange(count))
            or np.any(aux.data != 1.) or len(np.unique(aux.row)) != count):
        raise ValueError('Unexpected window epigraph upper bounds')
    order = np.argsort(aux.col)
    rows = aux.row[order]
    if linear[rows, :variables].nnz:
        raise ValueError('Window epigraph bounds depend on pose coordinates')
    starts = dims.nonneg + np.r_[0, np.cumsum(sizes)[:-1]]
    keep = np.ones(matrix.shape[0], dtype=bool)
    keep[rows] = False
    expected = sparse.csr_matrix((-np.ones(count), (starts, np.arange(count))),
                                 shape=(matrix.shape[0], count))
    # The t upper rows are removed, leaving one -t coefficient per SOC head.
    if (matrix[keep, variables:] != expected[keep]).nnz:
        raise ValueError('Unexpected window epigraph cone coefficients')
    if (data['P'][variables:].nnz or data['P'][:, variables:].nnz
            or np.any(data['c'][variables:] != 0.) or np.any(data['b'][starts] != 0.)):
        raise ValueError('Window norm epigraph participates in objective or offset')
    offset = data['b'].copy()
    offset[starts] = offset[rows]
    result = dict(data, P=data['P'][:variables, :variables].tocsc(),
                  c=data['c'][:variables].copy(), A=matrix[keep, :variables].tocsc(),
                  b=offset[keep], dims=SimpleNamespace(**vars(dims)))
    result['dims'].nonneg -= count
    DIAGNOSTICS.update(programs=1, eliminated_variables=count, eliminated_rows=count)
    return result
