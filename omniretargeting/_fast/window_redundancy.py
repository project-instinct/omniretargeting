"""Conservative box proofs and exact whole-cone duplicates for a frozen QP.

The original program stays intact for separation. Every proof depends on
explicit retained domain rows; a metadata box alone is not a certificate.
"""

import numpy as np


def cone_groups(program):
    sizes = np.asarray(program.soc, int)
    starts = program.nonneg + np.r_[0, np.cumsum(sizes)[:-1]].astype(int)
    return sizes, starts


def _box_domain(model, matrix):
    program, size = model.program, model.gradient.size
    lower, upper = model.step_lower.ravel(), model.step_upper.ravel()
    protected = np.zeros(program.nonneg, bool)
    safe_extra = np.zeros(matrix.shape[1] - size, bool)
    if (lower.shape != (size,) or upper.shape != (size,) or
            not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper)):
        return None
    single = np.flatnonzero(np.diff(matrix.indptr[:program.nonneg + 1]) == 1)
    offsets = matrix.indptr[single]
    columns, values = matrix.indices[offsets], matrix.data[offsets]
    found_lo, found_hi = np.zeros(size, bool), np.zeros(size, bool)
    for row, col, value in zip(single, columns, values):
        bound = program.b[row]
        if col < size:
            if value == 1. and bound == upper[col] and not found_hi[col]:
                protected[row], found_hi[col] = True, True
            elif value == -1. and bound == -lower[col] and not found_lo[col]:
                protected[row], found_lo[col] = True, True
        elif value == -1. and bound == 0. and not safe_extra[col - size]:
            protected[row], safe_extra[col - size] = True, True
    if not found_lo.all() or not found_hi.all():
        return None
    return lower, upper, protected, safe_extra


def box_redundancy(model):
    """Prove A*x<=b / whole norm membership for every delta in the step box.

    Unbounded epigraphs can only help scalar/head inequalities, and cannot
    occur in norm tails. Outward reserves cover products, sparse dot sums,
    subtractions and norm rounding. Uncertain/boundary cases are retained.
    """
    program = model.program
    matrix = program.A.tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    sizes, starts = cone_groups(program)
    linear, cones = np.zeros(program.nonneg, bool), np.zeros(len(sizes), bool)
    domain = _box_domain(model, matrix)
    if domain is None:
        return linear, cones, np.zeros(program.nonneg, bool)
    lower, upper, protected, safe_extra = domain
    step, extra = matrix[:, :model.gradient.size], matrix[:, model.gradient.size:]
    pos, neg = step.maximum(0.), step.minimum(0.)
    maximum, minimum = pos @ upper + neg @ lower, pos @ lower + neg @ upper
    magnitude = abs(step) @ np.maximum(abs(lower), abs(upper))
    operations = np.diff(step.indptr).astype(float) + 4.
    eps = np.finfo(float).eps
    reserve = 16. * eps * operations * (abs(program.b) + magnitude + 1.)
    min_slack = np.nextafter(program.b - maximum - reserve, -np.inf)
    tail_size = np.nextafter(np.maximum(abs(program.b - maximum), abs(program.b - minimum))
                             + reserve, np.inf)
    helpful = np.asarray((extra > 0.).getnnz(axis=1) == 0)
    if not safe_extra.all():
        helpful &= np.asarray(extra[:, ~safe_extra].getnnz(axis=1) == 0)
    linear[:] = helpful[:program.nonneg] & (min_slack[:program.nonneg] > 0.)
    empty_extra = np.asarray(extra.getnnz(axis=1) == 0)
    for size in np.unique(sizes):
        mask = sizes == size
        rows = starts[mask, None] + np.arange(size)
        upper_norm = np.linalg.norm(tail_size[rows[:, 1:]], axis=1)
        upper_norm = np.nextafter(upper_norm * (1. + 16. * size * eps), np.inf)
        cones[mask] = (helpful[rows[:, 0]] & empty_extra[rows[:, 1:]].all(axis=1)
                       & (min_slack[rows[:, 0]] > upper_norm))
    linear[protected] = False
    return linear, cones, protected


def duplicate_cones(program):
    """Map identical affine norms/heads to the tightest scalar bound.

    Complete sparse coefficients include epigraph columns. Sharing just a
    slack key, norm dimension or sample body never establishes duplication.
    No approximate row matching, tail rotation or rounded hashes are used.
    """
    matrix = program.A.tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    sizes, starts = cone_groups(program)
    representatives = np.arange(len(sizes))
    groups = {}
    for index, (size, start) in enumerate(zip(sizes, starts)):
        first, last = matrix.indptr[start], matrix.indptr[start + size]
        signature = (int(size), (matrix.indptr[start:start + size + 1] - first).tobytes(),
                     matrix.indices[first:last].tobytes(), matrix.data[first:last].tobytes(),
                     program.b[start + 1:start + size].tobytes())
        groups.setdefault(signature, []).append(index)
    for indices in groups.values():
        if len(indices) > 1:
            tight = min(indices, key=lambda k: program.b[starts[k]])
            representatives[indices] = tight
    return representatives
