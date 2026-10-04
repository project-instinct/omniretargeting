"""Row accumulation for per-frame normal equations of the fast backend."""

import numpy as np
from scipy import sparse


def _unit(vectors):
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, norm, out=np.zeros_like(vectors), where=norm > 1e-12)


def _block_rows(blocks, width):
    """Sparse block-diagonal matrix from per-frame (frames, k, width) blocks."""
    frames, rows, _ = blocks.shape
    from .guard_acceleration import enabled

    if enabled():
        # Unique columns already occur in ascending order in every row.
        # COO sorting/duplicate summation cannot change these coefficients.
        # Copy data: eliminate_zeros compacts it in place, and callers may
        # still use their dense Jacobian to construct another cone.
        size = frames * width
        dtype = np.int32 if max(size, blocks.size) < 2**31 else np.int64
        columns = np.broadcast_to(
            np.arange(frames, dtype=dtype)[:, None, None] * width
            + np.arange(width, dtype=dtype), blocks.shape).reshape(-1).copy()
        pointer = np.arange(frames * rows + 1, dtype=dtype) * width
        matrix = sparse.csr_matrix((blocks.ravel().copy(), columns, pointer),
                                   shape=(frames * rows, size))
        matrix.eliminate_zeros()
        return matrix
    r = np.arange(frames * rows).repeat(width)
    c = (np.arange(frames)[:, None, None] * width + np.arange(width)).repeat(rows, 1)
    matrix = sparse.csr_matrix(
        (blocks.ravel(), (r, c.ravel())), shape=(frames * rows, frames * width)
    )
    # Root rows touch very few coordinates. Explicit zero entries otherwise
    # turn sparse temporal products into nearly dense block multiplications.
    matrix.eliminate_zeros()
    return matrix


def _merge_identical(matrix, lower, upper):
    """One row per distinct coefficient row, with the intersected interval.

    Several guards bound the same temporal stencils. The feasible set is
    unchanged; rows keep their first position.
    """
    matrix = sparse.csr_matrix(matrix, copy=True)
    matrix.sum_duplicates()
    if matrix.shape[0] < 2:
        return matrix, lower, upper
    # A fixed probe only groups candidates; equality is then checked exactly.
    key = matrix @ np.random.default_rng(0).uniform(1., 2., matrix.shape[1])
    count = np.diff(matrix.indptr)
    order = np.lexsort((key, count))
    same = (count[order[1:]] == count[order[:-1]]) & (key[order[1:]] == key[order[:-1]])
    pairs = np.flatnonzero(same)
    if not len(pairs):
        return matrix, lower, upper
    difference = matrix[order[pairs]] - matrix[order[pairs + 1]]
    difference.eliminate_zeros()
    same[pairs] = np.diff(difference.indptr) == 0
    starts = np.flatnonzero(np.r_[True, ~same])
    first = np.minimum.reduceat(order, starts)
    low = np.maximum.reduceat(lower[order], starts)
    high = np.minimum.reduceat(upper[order], starts)
    rank = np.argsort(first)
    return matrix[first[rank]], low[rank], high[rank]


class _Accumulator:
    """Per-frame cost, normal equations and gradient of row families.

    Every value is accumulated per frame, so a frame's result does not depend
    on which other frames share its chunk; ``cost`` sums the frames in order.
    """

    def __init__(self, frames, width, with_jacobian):
        self.frame_cost = np.zeros(frames)
        self.cost = 0.0
        self.jacobian = with_jacobian
        if with_jacobian:
            self.hessian = np.zeros((frames, width, width))
            self.gradient = np.zeros((frames, width))

    def _add_cost(self, residual):
        residual = np.asarray(residual, dtype=float)
        self.frame_cost += np.square(residual).reshape(len(residual), -1).sum(axis=1)
        self.cost = float(self.frame_cost.sum())

    def add(self, residual, jacobian=None):
        self._add_cost(residual)
        if self.jacobian:
            self.hessian += np.matmul(jacobian.transpose(0, 2, 1), jacobian)
            self.gradient += np.matmul(residual[:, None, :], jacobian)[:, 0]

    def add_frame_rows(self, frame, residual, jacobian=None):
        """Rows of one frame; their count may differ between frames."""
        residual = np.asarray(residual, dtype=float)
        self.frame_cost[frame] += float(np.square(residual).sum())
        self.cost = float(self.frame_cost.sum())
        if self.jacobian and len(residual):
            self.hessian[frame] += jacobian.T @ jacobian
            self.gradient[frame] += residual @ jacobian

    def add_body_rows(self, residual, coefficients, bodies, kin):
        """Rows ``a^T [Jp_b; Jr_b]`` of points rigidly attached to ``bodies``.

        ``residual`` (frames, k), ``coefficients`` (frames, k, 6), ``bodies``
        (k,). Per-body 6x6 moments give the exact J^T J and J^T r without
        forming per-point (frames, k, width) Jacobians.
        """
        self._add_cost(residual)
        if not self.jacobian:
            return
        for body in np.unique(bodies):
            index = np.flatnonzero(bodies == body)
            rows = coefficients[:, index]
            spatial = np.concatenate([kin.jacp[:, body], kin.jacr[:, body]], axis=1)
            moment = np.matmul(rows.transpose(0, 2, 1), rows)
            pull = np.matmul(residual[:, None, index], rows)
            self.hessian += np.matmul(spatial.transpose(0, 2, 1), np.matmul(moment, spatial))
            self.gradient += np.matmul(pull, spatial)[:, 0]


class _DeferredRows:
    """Prepare independent families early; insert them in reference order.

    Combining their normal equations before insertion changes round-off.
    Keep each add separate so a collision query can overlap later families
    without changing any per-frame addition or BLAS reduction.
    """

    def __init__(self, with_jacobian):
        self.jacobian = with_jacobian
        self.parts = []

    def add(self, residual, jacobian=None):
        residual = np.asarray(residual, dtype=float)
        cost = np.square(residual).reshape(len(residual), -1).sum(axis=1)
        hessian = gradient = None
        if self.jacobian:
            hessian = np.matmul(jacobian.transpose(0, 2, 1), jacobian)
            gradient = np.matmul(residual[:, None, :], jacobian)[:, 0]
        self.parts.append((cost, hessian, gradient))

    def insert(self, accumulator):
        for cost, hessian, gradient in self.parts:
            accumulator.frame_cost += cost
            if self.jacobian:
                accumulator.hessian += hessian
                accumulator.gradient += gradient
        accumulator.cost = float(accumulator.frame_cost.sum())


def _point_rows(directions, levers):
    """Coefficients ``[d, r x d]``: d . (velocity of a point at lever r)."""
    return np.concatenate([directions, np.cross(levers, directions)], -1)
