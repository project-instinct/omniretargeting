"""Early cone assembly for coordinate-projected repair programs.

The full joint box still decides redundancy and row equilibration. In
particular, projecting that box first would change the numerical program.
Only cones the established projected solver already discards are omitted
before sparse assembly. Linear rows retain full-coordinate identities until
duplicate merging; crossing temporal and support stencils remain whole.
"""

import numpy as np
from scipy import sparse

from .repair_limits import LinearBounds
from .rows import _merge_identical


class ProjectedBounds(LinearBounds):
    def __init__(self, size, projection, frames, width):
        super().__init__(size)
        self.projection = sparse.csc_matrix(projection)
        self.frames, self.width = frames, width
        selector = self.projection
        self.selected = None
        if (np.all(np.diff(selector.indptr) == 1) and np.all(selector.data == 1.)
                and len(np.unique(selector.indices)) == selector.shape[1]):
            self.selected = np.zeros(size, dtype=bool)
            self.selected[selector.indices] = True
            self.selected = self.selected.reshape(frames, width)

    def packed(self):
        if self.selected is None or self.box is None:
            return super().packed()
        matrix = sparse.vstack(self.matrices, format="csr")
        lower, upper = np.concatenate(self.lower), np.concatenate(self.upper)
        projected = (matrix @ self.projection).tocsr()
        projected.eliminate_zeros()
        varying = np.diff(projected.indptr) != 0
        # A constant violated row can merge with a satisfied duplicate and
        # acquire different equilibration. Keep the original assembly in that
        # case, including every duplicate. Otherwise every dropped row remains
        # constant-valid even after intersection, so none reaches the solver.
        if np.any(~varying & ((lower > 0.) | (upper < 0.))):
            return super().packed()
        # The established duplicate merger checks adjacent equal probes. A
        # fixed row with the same probe as a different varying row can split
        # their run. Removing it must not make the merger discover extra
        # duplicates, even for deliberately constructed floating-point ties.
        check = matrix.copy()
        check.sum_duplicates()
        key = check @ np.random.default_rng(0).uniform(1., 2., self.size)
        count = np.diff(check.indptr)
        order = np.lexsort((key, count))
        tied = (count[order[1:]] == count[order[:-1]]) & (key[order[1:]] == key[order[:-1]])
        if np.any(tied & (varying[order[1:]] != varying[order[:-1]])):
            return super().packed()
        first = sum(m.shape[0] for m in self.matrices[:self.box[0]])
        box_row = (np.flatnonzero(varying) >= first) & (np.flatnonzero(varying) < first + self.size)
        matrix, lower, upper = matrix[varying], lower[varying], upper[varying]
        # Redundancy still uses the full joint box and full coefficient rows.
        bottom, top = self._range(matrix)
        needed = (bottom < lower) | (top > upper) | box_row
        matrix, lower, upper = _merge_identical(matrix[needed], lower[needed], upper[needed])
        scale = 1 / np.maximum(np.maximum(np.abs(lower), np.abs(upper)), 1e-4)
        scale[~np.isfinite(scale) | (scale == 0)] = 1.
        return sparse.diags(scale) @ matrix.tocsc(), lower * scale, upper * scale

    def norm(self, jacobian, residual, allowed):
        if self.selected is None:
            return super().norm(jacobian, residual, allowed)
        shape = residual.shape
        dimension = shape[-1]
        residual = residual.reshape(-1, dimension)
        base = np.broadcast_to(allowed, shape[:-1]).ravel()
        allowed = np.minimum(np.maximum(base + 5e-8, np.linalg.norm(residual, axis=1)),
                             base + 1e-7)
        blocks = jacobian.reshape(self.frames, -1, dimension, self.width)
        # Test every vector component together; a zero component never
        # licenses shortening a norm cone. Invalid fixed cones survive.
        varying = np.any((blocks != 0.) & self.selected[:, None, None, :], axis=(2, 3))
        keep = varying.ravel() | ~(np.linalg.norm(residual, axis=1) <= allowed + 1e-12)
        indices = np.flatnonzero(keep)
        data = blocks.reshape(-1, dimension, self.width)[indices]
        frame = indices // blocks.shape[1]
        columns = np.broadcast_to(frame[:, None, None] * self.width
                                  + np.arange(self.width), data.shape).ravel().copy()
        pointer = np.arange(data.shape[0] * dimension + 1) * self.width
        matrix = sparse.csr_matrix((data.ravel(), columns, pointer),
                                   shape=(len(indices) * dimension, self.size))
        matrix.eliminate_zeros()
        self.norms.append((matrix, residual[keep], allowed[keep]))
