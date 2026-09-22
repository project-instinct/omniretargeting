# SPDX-License-Identifier: Apache-2.0
"""mjbatch dynamics with a previous-control state for true slew penalties.

Quaternion chart and batched finite differences adapted from mjbatch's
examples/g1_flip.py at 2f169146b643adda65e195d8ad13d997abdd2a2e.
See LICENSE.mjbatch. Only free + scalar joints and stateless motors are supported.
"""

import mujoco
import numpy as np


def quat_mul(a, b):
    w = a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdims=True)
    v = (
        a[..., :1] * b[..., 1:]
        + b[..., :1] * a[..., 1:]
        + np.cross(a[..., 1:], b[..., 1:])
    )
    return np.concatenate([w, v], axis=-1)


def quat_exp(v):
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    axis = np.divide(v, angle, out=np.zeros_like(v), where=angle > 0)
    return np.concatenate([np.cos(angle / 2), np.sin(angle / 2) * axis], axis=-1)


def quat_log(q):
    norm = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    axis = np.divide(q[..., 1:], norm, out=np.zeros_like(q[..., 1:]), where=norm > 0)
    angle = 2 * np.arctan2(norm, q[..., :1])
    return np.where(angle > np.pi, angle - 2 * np.pi, angle) * axis


class StateChart:
    """Stored state is (qpos, qvel, previous ctrl); tangent is (dq, dv, du_prev)."""

    def __init__(self, model):
        self.nq, self.nv, self.nu = model.nq, model.nv, model.nu
        self.physical_nx = 2 * model.nv
        self.nx = self.physical_nx + model.nu
        if np.any(model.jnt_type == mujoco.mjtJoint.mjJNT_BALL):
            raise ValueError(
                "Experimental iLQR currently supports free and scalar joints only"
            )
        jt = model.jnt_type
        self.free = list(zip(model.jnt_qposadr[jt == 0], model.jnt_dofadr[jt == 0]))
        quat = [a + i for a, _ in self.free for i in range(3, 7)]
        spin = [d + i for _, d in self.free for i in range(3, 6)]
        self.lin_q = np.setdiff1d(range(self.nq), quat)
        self.lin_v = np.setdiff1d(range(self.nv), spin)

    def integrate(self, state, delta):
        out = np.array(state, dtype=float, copy=True)
        out[..., self.lin_q] += delta[..., self.lin_v]
        for a, d in self.free:
            quat = quat_mul(
                state[..., a + 3 : a + 7], quat_exp(delta[..., d + 3 : d + 6])
            )
            out[..., a + 3 : a + 7] = quat / np.linalg.norm(
                quat, axis=-1, keepdims=True
            )
        out[..., self.nq :] += delta[..., self.nv :]
        return out

    def difference(self, first, second):
        dq = np.empty(
            (*np.broadcast_shapes(first.shape[:-1], second.shape[:-1]), self.nv)
        )
        dq[..., self.lin_v] = second[..., self.lin_q] - first[..., self.lin_q]
        for a, d in self.free:
            conjugate = first[..., a + 3 : a + 7] * [1, -1, -1, -1]
            dq[..., d + 3 : d + 6] = quat_log(
                quat_mul(conjugate, second[..., a + 3 : a + 7])
            )
        return np.concatenate(
            [dq, second[..., self.nq :] - first[..., self.nq :]], axis=-1
        )


class BatchedDynamics(StateChart):
    def __init__(self, model, horizon, substeps, threads=8, epsilon=1e-6):
        super().__init__(model)
        from mjbatch import Batch

        if model.na or not model.nu or not np.all(model.actuator_ctrllimited):
            raise ValueError("iLQR requires bounded, stateless actuators")
        if horizon < 1 or substeps < 1 or not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("Invalid horizon, substeps or finite-difference epsilon")
        self.model, self.T, self.sub = model, int(horizon), int(substeps)
        self.epsilon = epsilon
        self.lo, self.hi = model.actuator_ctrlrange.T.copy()
        self.alphas = 0.5 ** np.arange(9)
        count = max(
            (horizon + 1) * (1 + self.physical_nx),
            horizon * (1 + self.physical_nx + self.nu),
        )
        self.batch = Batch(model, count, num_threads=threads)
        self.qpos, self.qvel = self.batch.bind("qpos"), self.batch.bind("qvel")
        self.ctrl, self.warm = (
            self.batch.bind("ctrl"),
            self.batch.bind("qacc_warmstart"),
        )
        self.line = Batch(model, len(self.alphas), num_threads=threads)
        self.line_fields = [
            self.line.bind(f) for f in ("qpos", "qvel", "ctrl", "qacc_warmstart")
        ]
        self.warning = self.line.bind("warning")

    def step(self, states, controls):
        n = len(states)
        self.qpos[:n] = states[:, : self.nq]
        self.qvel[:n] = states[:, self.nq : self.nq + self.nv]
        self.ctrl[:n], self.warm[:n] = controls, 0.0
        self.batch.step(np.arange(n), nstep=self.sub)
        return np.concatenate([self.qpos[:n], self.qvel[:n], controls], axis=-1)

    def advance(self, states, controls):
        qpos, qvel, ctrl, warm = self.line_fields
        qpos[:], qvel[:] = states[:, : self.nq], states[:, self.nq : self.nq + self.nv]
        ctrl[:], warm[:] = controls, 0.0
        self.line.step(nstep=self.sub)
        return np.concatenate([qpos, qvel, controls], axis=-1)

    def linearize(self, states, controls):
        physical, nu, eps = self.physical_nx, self.nu, self.epsilon
        cols = 1 + physical + nu
        x = np.repeat(states[:-1], cols, axis=0)
        u = np.repeat(controls, cols, axis=0)
        perturbation = eps * np.tile(np.eye(cols)[:, 1:], (self.T, 1))
        perturbation[:, physical:] *= np.where(u + eps > self.hi, -1.0, 1.0)
        dx = np.zeros((len(x), self.nx))
        dx[:, :physical] = perturbation[:, :physical]
        out = self.step(self.integrate(x, dx), u + perturbation[:, physical:])
        out = out.reshape(self.T, cols, -1)
        h = perturbation.reshape(self.T, cols, -1)[:, 1:].sum(axis=-1)
        jac = self.difference(out[:, :1], out[:, 1:]) / h[..., None]
        jac = jac.swapaxes(1, 2)
        A = np.zeros((self.T, self.nx, self.nx))
        A[:, :, :physical] = jac[:, :, :physical]
        B = jac[:, :, physical:]
        # Previous controls do not affect current physics. Their next value is u.
        A[:, physical:, :] = 0
        B[:, physical:, :] = np.eye(nu)
        return A, B

    def rollout(self, initial, controls, gains=None):
        n = len(self.alphas)
        state = np.tile(initial, (n, 1))
        warned = self.warning.copy()
        xs = np.empty((self.T + 1, n, len(initial)))
        us, total = np.empty((self.T, n, self.nu)), np.zeros(n)
        xs[0] = state
        for t in range(self.T):
            u = np.tile(controls[t], (n, 1))
            if gains is not None:
                reference, k, K = gains
                u += (
                    self.alphas[:, None] * k[t]
                    + self.difference(reference[t], state) @ K[t].T
                )
            u = np.clip(u, self.lo, self.hi)
            total += self.cost(t, state, u)
            state = self.advance(state, u)
            xs[t + 1], us[t] = state, u
        total += self.cost(self.T, state, np.zeros((n, self.nu)))
        total[(self.warning != warned).any(axis=(1, 2))] = np.inf
        total[~np.isfinite(xs).all(axis=(0, 2))] = np.inf
        total[~np.isfinite(total)] = np.inf
        return xs, us, total
