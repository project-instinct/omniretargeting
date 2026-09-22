"""Model-derived tangent coordinates and time-normalized differences."""

import mujoco
import numpy as np
from scipy import sparse

from .._refinement.math import _difference_operator, _scalar_joint_coordinates


class Coordinates:
    def __init__(
        self, model, lock_root_xy=True, lock_root_z=False, lock_root_orientation=False
    ):
        self.model = model
        self.qi, self.di, self.lower, self.upper = _scalar_joint_coordinates(model)
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) > 1 or np.any(model.jnt_type == mujoco.mjtJoint.mjJNT_BALL):
            raise ValueError(
                "Sequence solver supports scalar joints and at most one free root"
            )
        self.root_q = int(model.jnt_qposadr[free[0]]) if len(free) else None
        self.root_d = int(model.jnt_dofadr[free[0]]) if len(free) else None
        excluded = (
            []
            if self.root_d is None or not lock_root_xy
            else [self.root_d, self.root_d + 1]
        )
        if self.root_d is not None:
            if lock_root_z:
                excluded.append(self.root_d + 2)
            if lock_root_orientation:
                excluded.extend(range(self.root_d + 3, self.root_d + 6))
        self.cols = np.array(
            [d for d in range(model.nv) if d not in excluded], dtype=int
        )
        self.width = len(self.cols)
        self.inverse = {int(d): i for i, d in enumerate(self.cols)}
        self.scalar_cols = np.array([self.inverse[int(d)] for d in self.di], dtype=int)
        self.selector = sparse.csr_matrix(
            (np.ones(len(self.di)), (np.arange(len(self.di)), self.scalar_cols)),
            shape=(len(self.di), self.width),
        )

    def normalize(self, q):
        q = np.asarray(q, dtype=float).copy()
        if q.ndim != 2 or q.shape[1] != self.model.nq or not np.all(np.isfinite(q)):
            raise ValueError("Initial trajectory must be finite (frames, model.nq)")
        if self.root_q is not None:
            sl = slice(self.root_q + 3, self.root_q + 7)
            norms = np.linalg.norm(q[:, sl], axis=1)
            if np.any(norms < 1e-10):
                raise ValueError("Initial root quaternion must be nonzero")
            q[:, sl] /= norms[:, None]
            for f in range(1, len(q)):
                if np.dot(q[f - 1, sl], q[f, sl]) < 0:
                    q[f, sl] *= -1
        q[:, self.qi] = np.clip(q[:, self.qi], self.lower, self.upper)
        return q

    def integrate(self, q, step, alpha):
        result = q.copy()
        for f in range(len(q)):
            velocity = np.zeros(self.model.nv)
            velocity[self.cols] = step[f]
            mujoco.mj_integratePos(self.model, result[f], velocity, alpha)
        return result

    def bounds(self, q, angular_step, translation_step):
        lo = np.full((len(q), self.width), -angular_step)
        hi = -lo
        translation_dofs = set()
        if self.root_d is not None:
            translation_dofs.update(range(self.root_d, self.root_d + 3))
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                translation_dofs.add(int(self.model.jnt_dofadr[j]))
        for d in translation_dofs:
            if d in self.inverse:
                lo[:, self.inverse[d]], hi[:, self.inverse[d]] = (
                    -translation_step,
                    translation_step,
                )
        lo[:, self.scalar_cols] = np.maximum(
            lo[:, self.scalar_cols], self.lower - q[:, self.qi]
        )
        hi[:, self.scalar_cols] = np.minimum(
            hi[:, self.scalar_cols], self.upper - q[:, self.qi]
        )
        return lo, hi


def temporal_operators(frames, framerate, derivative_sigmas):
    """Use derivatives per second, independent of input sampling frequency."""
    if not np.isfinite(framerate) or framerate <= 0:
        raise ValueError("framerate must be finite and positive")
    sigmas = np.asarray(derivative_sigmas, dtype=float)
    if sigmas.shape != (3,) or not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0):
        raise ValueError("Derivative sigmas must be three finite positive values")
    return [
        _difference_operator(frames, order) * framerate**order / sigma
        for order, sigma in enumerate(sigmas, 1)
    ]
