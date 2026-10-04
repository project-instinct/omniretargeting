"""Whole-trajectory forward kinematics and Jacobians in one pass.

MuJoCo evaluates each frame in C; this module only stores the results as
arrays so every task can be evaluated for all frames with vectorized numpy
instead of per-frame Python loops. Point Jacobians follow from body Jacobians
through ``J_point = J_origin - [p - x_body]_x J_rotation``.
"""

from collections import OrderedDict

import mujoco
import numpy as np


def cross_columns(vectors, matrices):
    """``skew(vectors) @ matrices`` for (..., 3) vectors and (..., 3, n) matrices."""
    x, y, z = (vectors[..., axis, None] for axis in range(3))
    a, b, c = matrices[..., 0, :], matrices[..., 1, :], matrices[..., 2, :]
    return np.stack([y * c - z * b, z * a - x * c, x * b - y * a], -2)


def skew(vectors):
    """Batched cross-product matrices of vectors with shape (..., 3)."""
    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
    zero = np.zeros_like(x)
    return np.stack(
        [
            np.stack([zero, -z, y], -1),
            np.stack([z, zero, -x], -1),
            np.stack([-y, x, zero], -1),
        ],
        -2,
    )


class TrajectoryKinematics:
    """Body poses and Jacobians of every frame of a trajectory.

    ``columns`` selects the tangent coordinates that the optimizer moves.
    """

    def __init__(self, model, columns, *, cache=None):
        self.model = model
        self.data = mujoco.MjData(model)
        self.columns = np.asarray(columns, dtype=int)
        # Shared only by tasks using this immutable model within one solve.
        # Four exact candidates retain an accepted state alongside trial states.
        self.cache = OrderedDict() if cache is None else cache

    def evaluate(self, poses, with_jacobian=True, *, native=None):
        model, data = self.model, self.data
        poses = np.asarray(poses)
        if native is None:
            from .guard_acceleration import enabled
            native = enabled() and len(poses) >= 64
        key = (self.columns.tobytes(), poses.shape, poses.dtype.str, poses.tobytes())
        previous = self.cache.get(key)
        if previous is not None and (not with_jacobian or 'jacp' in previous):
            for name, value in previous.items():
                setattr(self, name, value)
            self.with_jacobian = with_jacobian
            self.cache.move_to_end(key)
            return self
        frames = len(poses)
        self.xpos = np.empty((frames, model.nbody, 3))
        self.xmat = np.empty((frames, model.nbody, 3, 3))
        self.geom_xpos = np.empty((frames, model.ngeom, 3))
        self.geom_xmat = np.empty((frames, model.ngeom, 3, 3))
        if with_jacobian:
            jacp = np.zeros((frames, model.nbody, 3, model.nv))
            jacr = np.zeros((frames, model.nbody, 3, model.nv))
        computed = False
        if native:
            from .native_kinematics import evaluate_kinematics
            computed = evaluate_kinematics(self, poses, jacp if with_jacobian else None,
                                          jacr if with_jacobian else None)
        for frame, pose in enumerate(() if computed else poses):
            data.qpos[:] = pose
            mujoco.mj_kinematics(model, data)
            self.xpos[frame] = data.xpos
            self.xmat[frame] = data.xmat.reshape(-1, 3, 3)
            self.geom_xpos[frame] = data.geom_xpos
            self.geom_xmat[frame] = data.geom_xmat.reshape(-1, 3, 3)
            if not with_jacobian:
                continue
            mujoco.mj_comPos(model, data)
            for body in range(1, model.nbody):
                mujoco.mj_jacBody(model, data, jacp[frame, body], jacr[frame, body], body)
        if with_jacobian:
            full = len(self.columns) == model.nv and np.array_equal(self.columns, np.arange(model.nv))
            self.jacp = jacp if full else jacp[..., self.columns]
            self.jacr = jacr if full else jacr[..., self.columns]
        self.with_jacobian = with_jacobian
        names = ['xpos', 'xmat', 'geom_xpos', 'geom_xmat']
        if with_jacobian:
            names += ['jacp', 'jacr']
        result = {name: getattr(self, name) for name in names}
        for value in result.values():
            value.setflags(write=False)
        self.cache[key] = result
        self.cache.move_to_end(key)
        while len(self.cache) > 4:
            self.cache.popitem(last=False)
        return self

    def points(self, bodies, local):
        """World points ``x_b + R_b @ local`` for (frames, k) bodies/offsets."""
        bodies = np.asarray(bodies, dtype=int)
        local = np.broadcast_to(np.asarray(local, dtype=float), (*bodies.shape, 3))
        frames = np.arange(len(self.xpos))[:, None]
        rotation = self.xmat[frames, bodies] if bodies.ndim == 2 else self.xmat[:, bodies]
        origin = self.xpos[frames, bodies] if bodies.ndim == 2 else self.xpos[:, bodies]
        return origin + np.einsum("...ij,...j->...i", rotation, local)

    def point_jacobians(self, bodies, points):
        """Translational Jacobians of world ``points`` rigidly fixed to bodies.

        ``bodies`` has shape (k,) or (frames, k); ``points`` (frames, k, 3).
        Returns (frames, k, 3, width).
        """
        bodies = np.asarray(bodies, dtype=int)
        from .guard_acceleration import enabled

        if (enabled() and self.jacp.shape[-1] and np.asarray(points).dtype == np.float64
                and bodies.ndim in (1, 2) and np.all((bodies >= 0) & (bodies < self.model.nbody))):
            from .batched_self_collision import NativeUnavailable
            from .native_point_rows import NativePointRows

            try:
                previous = getattr(self, '_native_point_inputs', None)
                if previous is None or previous[0] is not self.jacp or previous[1] is not self.jacr:
                    self._native_point_rows = NativePointRows(self)
                    # Subset Jacobians need contiguous copies. Retain their
                    # original arrays as well, so recycled object IDs cannot
                    # mistake a later pose for the previously copied inputs.
                    self._native_point_inputs = self.jacp, self.jacr
                return self._native_point_rows.jacobians(bodies, points)
            except NativeUnavailable:
                pass
        if bodies.ndim == 1:
            jacp, jacr, origin = (
                self.jacp[:, bodies],
                self.jacr[:, bodies],
                self.xpos[:, bodies],
            )
        else:
            frames = np.arange(len(self.xpos))[:, None]
            jacp, jacr, origin = (
                self.jacp[frames, bodies],
                self.jacr[frames, bodies],
                self.xpos[frames, bodies],
            )
        return jacp - cross_columns(points - origin, jacr)

    def geom_points(self, geoms, local):
        """World points on geoms: ``g_x + g_R @ local`` for (k,) geoms."""
        geoms = np.asarray(geoms, dtype=int)
        return self.geom_xpos[:, geoms] + np.einsum(
            "fkij,kj->fki", self.geom_xmat[:, geoms], np.asarray(local, dtype=float)
        )
