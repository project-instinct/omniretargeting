"""One sparse correction interpolant shared by coarse guards and transfer."""

import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation

from .kinematics import skew


def _unwrap_vectors(vectors):
    result = vectors.copy()
    for frame in range(1, len(result)):
        size = np.linalg.norm(result[frame])
        if size > 1e-12:
            axis = result[frame] / size
            turns = int(np.rint(np.dot(result[frame - 1] - result[frame], axis) / (2 * np.pi)))
            result[frame] += turns * (2 * np.pi) * axis
    return result


def _right_jacobian(vectors):
    """SO(3) exp derivative in right (MuJoCo local angular) coordinates."""
    squared = np.sum(vectors**2, axis=1)
    angle = np.sqrt(squared)
    small = squared < 1e-8
    a = np.empty(len(vectors))
    b = np.empty(len(vectors))
    a[small] = .5 - squared[small] / 24 + squared[small]**2 / 720
    b[small] = 1 / 6 - squared[small] / 120 + squared[small]**2 / 5040
    a[~small] = (1 - np.cos(angle[~small])) / squared[~small]
    b[~small] = (angle[~small] - np.sin(angle[~small])) / angle[~small]**3
    matrix = skew(vectors)
    return np.eye(3) - a[:, None, None] * matrix + b[:, None, None] * (matrix @ matrix)


class LinearCorrectionEmbedding:
    def __init__(self, coords, fixed, indices):
        self.coords, self.fixed = coords, np.asarray(fixed)
        self.indices = np.asarray(indices)
        count = len(fixed)
        if not np.array_equal(self.indices, np.arange(0, count, 2)):
            raise ValueError('Invalid regular half-rate correction knots')
        self.count, self.knots = count, len(indices)
        frames = np.arange(count)
        self.left = np.minimum(frames // 2, self.knots - 1)
        self.right = np.minimum((frames + 1) // 2, self.knots - 1)
        self.L = sparse.csr_matrix((np.full(2 * count, .5),
                                   (np.repeat(frames, 2), np.c_[self.left, self.right].ravel())),
                                  shape=(count, self.knots))
        d = coords.root_d
        self.angular = ([] if d is None else
                        [coords.inverse.get(d + axis) for axis in (3, 4, 5)])
        self.rotate = bool(self.angular) and all(c is not None for c in self.angular)
        if self.rotate:
            q = coords.root_q
            self.rotations = Rotation.from_quat(fixed[:, q + 3:q + 7], scalar_first=True)
            self.axes = self.rotations.as_matrix()
        self._cache = None
        self._jacobian_cache = None

    def evaluate(self, poses):
        poses = np.asarray(poses)
        if poses.shape != self.fixed[self.indices].shape:
            raise ValueError('Coarse pose shape changed')
        key = poses.dtype.str, poses.tobytes()
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        values = self.coords.difference(self.fixed[self.indices], poses)
        if self.rotate:
            values[:, self.angular] = _unwrap_vectors(
                self.rotations[self.indices].apply(values[:, self.angular]))
        world = self.L @ values
        delta = world.copy()
        if self.rotate:
            delta[:, self.angular] = self.rotations.inv().apply(world[:, self.angular])
        full = self.fixed.copy()
        full[:, self.coords.qi] += delta[:, self.coords.scalar_cols]
        q, d = self.coords.root_q, self.coords.root_d
        if q is not None:
            for axis in range(3):
                column = self.coords.inverse.get(d + axis)
                if column is not None:
                    full[:, q + axis] += delta[:, column]
            if self.rotate:
                full[:, q + 3:q + 7] = (Rotation.from_rotvec(world[:, self.angular])
                                         * self.rotations).as_quat(scalar_first=True)
                zero = np.all(world[:, self.angular] == 0., axis=1)
                full[zero, q + 3:q + 7] = self.fixed[zero, q + 3:q + 7]
        # Preserve knots bit-for-bit, including exact active scalar boxes and
        # root quaternion signs. All other frames keep their authored reference.
        full[self.indices] = poses
        result = full, delta, values, world
        self._cache = key, result
        return result

    def embed(self, poses):
        return self.evaluate(poses)[0]

    def correction(self, poses):
        return self.evaluate(poses)[1]

    def affected_frames(self, frames):
        # Interior odd frames immediately outside both knot endpoints change
        # too. At even N the held tail belongs to the final knot.
        return range(max(0, 2 * frames.start - 1), min(self.count, 2 * frames.stop))

    def contributors(self, frames):
        return range(frames.start // 2, min(self.knots, frames.stop // 2 + 1))

    def map_violations(self, bad):
        result = np.zeros(self.knots, bool)
        frames = np.flatnonzero(bad)
        result[self.left[frames]] = True
        result[self.right[frames]] = True
        return result

    def jacobian(self, poses, coarse_frames=None, full_frames=None):
        """d(full local tangent)/d(coarse local tangent), sparse and banded."""
        coarse_frames = range(self.knots) if coarse_frames is None else coarse_frames
        full_frames = range(self.count) if full_frames is None else full_frames
        span = slice(full_frames.start, full_frames.stop)
        knot_span = slice(coarse_frames.start, coarse_frames.stop)
        self.evaluate(poses)
        key = (self._cache[0], coarse_frames.start, coarse_frames.stop,
               full_frames.start, full_frames.stop)
        if self._jacobian_cache is not None and self._jacobian_cache[0] == key:
            return self._jacobian_cache[1]
        weights = self.L[span, knot_span]
        width = self.coords.width
        if not self.rotate:
            result = sparse.kron(weights, sparse.eye(width), format='csr')
            self._jacobian_cache = key, result
            return result
        _, _, values, world = self.evaluate(poses)
        incoming = np.linalg.inv(_right_jacobian(values[:, self.angular])) @ self.axes[self.indices]
        outgoing = self.axes.transpose(0, 2, 1) @ _right_jacobian(world[:, self.angular])
        weights = weights.tocoo()
        blocks = (outgoing[weights.row + full_frames.start]
                  @ incoming[weights.col + coarse_frames.start]) * weights.data[:, None, None]
        diagonal = np.eye(width)
        diagonal[np.ix_(self.angular, self.angular)] = 0.
        linear = sparse.kron(weights.tocsr(), diagonal, format='csr')
        rows = (weights.row[:, None, None] * width + np.asarray(self.angular)[None, :, None])
        columns = (weights.col[:, None, None] * width + np.asarray(self.angular)[None, None, :])
        angular = sparse.coo_matrix((blocks.ravel(),
            (np.broadcast_to(rows, blocks.shape).ravel(),
             np.broadcast_to(columns, blocks.shape).ravel())), shape=linear.shape).tocsr()
        result = linear + angular
        self._jacobian_cache = key, result
        return result
