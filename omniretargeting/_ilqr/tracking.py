"""Robot-reference tracking costs and inverse-dynamics initialization.

The input is an already retargeted robot trajectory. Feet remain world-relative;
free hand markers are expressed in the robot torso frame. No human data changes.
This first experimental tracker has no semantic scene-contact task adapter.
"""

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from .dynamics import BatchedDynamics


def reference_velocity(model, poses, dt):
    """Finite differences in MuJoCo's tangent coordinates, at original cadence."""
    poses = np.asarray(poses, dtype=float)
    if len(poses) < 2 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("At least two poses and a positive dt are required")
    velocity = np.empty((len(poses), model.nv))
    for frame in range(len(poses)):
        left, right = max(frame - 1, 0), min(frame + 1, len(poses) - 1)
        mujoco.mj_differentiatePos(
            model, velocity[frame], (right - left) * dt, poses[left], poses[right]
        )
    return velocity


def inverse_seed(model, poses, velocity, dt):
    """Fit contact forces and motor torques, leaving the root unactuated.

    Like mjbatch's G1 example, use inverse dynamics for the required force and
    fit nonnegative pyramid contact forces rather than equating penetration
    depth with the desired supporting force. Return the unresolved root force.
    """
    if model.opt.cone != mujoco.mjtCone.mjCONE_PYRAMIDAL:
        raise ValueError("This initialization requires pyramidal contacts")
    data = mujoco.MjData(model)
    acceleration = np.gradient(velocity, dt, axis=0)
    act_dofs = np.array([model.jnt_dofadr[j] for j in model.actuator_trnid[:, 0]])
    unactuated = np.setdiff1d(np.arange(model.nv), act_dofs)
    lower, upper = model.actuator_ctrlrange.T
    limit = np.maximum(abs(lower), abs(upper))
    torques, root_residual, contact_counts = [], [], []
    for q, v, a in zip(poses, velocity, acceleration):
        data.qpos[:], data.qvel[:], data.qacc[:] = q, v, a
        mujoco.mj_inverse(model, data)
        required = data.qfrc_inverse + data.qfrc_constraint
        rows = (
            data.efc_type[: data.nefc] == mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL
        )
        J = data.efc_J.reshape(-1, model.nv)[: data.nefc][rows]
        force = np.zeros(model.nv)
        if len(J):
            matrix = np.vstack(
                [
                    J[:, unactuated].T,
                    J[:, act_dofs].T / limit[:, None],
                    1e-3 * np.eye(len(J)),
                ]
            )
            rhs = np.r_[
                required[unactuated], required[act_dofs] / limit, np.zeros(len(J))
            ]
            fit = lsq_linear(matrix, rhs, bounds=(0, np.inf), tol=1e-8, max_iter=100)
            if np.isfinite(fit.x).all():
                force = J.T @ fit.x
        torques.append(np.clip((required - force)[act_dofs], lower, upper))
        root_residual.append((required - force)[unactuated])
        contact_counts.append(data.ncon)
    return np.asarray(torques), {
        "unactuated_dofs": unactuated,
        "unactuated_force_residual": np.asarray(root_residual),
        "reference_physics_contacts": np.asarray(contact_counts),
    }


@dataclass(frozen=True)
class TrackingWeights:
    joint_sigma: float = 0.15
    root_position_sigma: float = 0.005
    root_rotation_sigma: float = 0.03
    joint_velocity_sigma: float = 3.0
    root_velocity_sigma: float = 0.2
    foot_sigma: float = 0.005
    hand_local_sigma: float = 0.02
    limit_sigma: float = 0.02
    control_weight: float = 0.01
    slew_weight: float = 1.0


class ReferenceTracker(BatchedDynamics):
    """Soft tracking costs; candidate acceptance is evaluated independently."""

    def __init__(
        self,
        model,
        poses,
        velocity,
        effectors,
        torso,
        substeps,
        weights=None,
        threads=8,
    ):
        weights = TrackingWeights() if weights is None else weights
        if any(
            not np.isfinite(x) or x <= 0
            for k, x in vars(weights).items()
            if k.endswith("sigma")
        ):
            raise ValueError("Tracking sigmas must be finite and positive")
        if any(
            not np.isfinite(x) or x < 0
            for k, x in vars(weights).items()
            if k.endswith("weight")
        ):
            raise ValueError("Control weights must be finite and nonnegative")
        super().__init__(model, len(poses) - 1, substeps, threads)
        self.weights = weights
        self.reference = np.c_[poses, velocity, np.zeros((len(poses), model.nu))]
        self.position_scale = np.full(self.nv, 1 / weights.joint_sigma)
        self.velocity_scale = np.full(self.nv, 1 / weights.joint_velocity_sigma)
        for _, dof in self.free:
            self.position_scale[dof : dof + 3] = 1 / weights.root_position_sigma
            self.position_scale[dof + 3 : dof + 6] = 1 / weights.root_rotation_sigma
            self.velocity_scale[dof : dof + 6] = 1 / weights.root_velocity_sigma
        self.scale = np.r_[self.position_scale, self.velocity_scale]
        self.xpos, self.xmat = self.batch.bind("xpos"), self.batch.bind("xmat")
        self.torso_id = model.body(torso).id
        self.effectors = {
            name: (
                model.body(spec["robot_body"]).id,
                np.asarray(spec.get("tracking_offset", [0, 0, 0])),
            )
            for name, spec in effectors.items()
        }
        if set(self.effectors) != {
            "left_hand",
            "right_hand",
            "left_foot",
            "right_foot",
        }:
            raise ValueError("The experiment requires both hands and both feet")
        self.targets = self.points(self.reference)
        self.limited_q = model.jnt_qposadr[np.asarray(model.jnt_limited, dtype=bool)]
        self.limited_range = model.jnt_range[np.asarray(model.jnt_limited, dtype=bool)]
        self.control_scale = 1 / np.maximum(abs(self.lo), abs(self.hi)) ** 2

    def points(self, states):
        n = len(states)
        self.qpos[:n] = states[:, : self.nq]
        self.qvel[:n] = states[:, self.nq : self.nq + self.nv]
        self.batch.forward(np.arange(n))
        rotation = self.xmat[:n].reshape(n, -1, 3, 3)
        points = {}
        for name, (body, offset) in self.effectors.items():
            p = self.xpos[:n, body] + np.einsum("nij,j->ni", rotation[:, body], offset)
            if "hand" in name:
                p = np.einsum(
                    "nji,nj->ni",
                    rotation[:, self.torso_id],
                    p - self.xpos[:n, self.torso_id],
                )
            points[name] = p.copy()
        return points

    def residual(self, frames, states):
        reference = self.reference[frames]
        state_error = (
            self.difference(reference, states)[:, : self.physical_nx] * self.scale
        )
        rows = [state_error]
        for name, p in self.points(states).items():
            sigma = (
                self.weights.hand_local_sigma
                if "hand" in name
                else self.weights.foot_sigma
            )
            rows.append((p - self.targets[name][frames]) / sigma)
        q = states[:, self.limited_q]
        lo, hi = self.limited_range.T
        rows += [
            np.maximum(lo - q, 0) / self.weights.limit_sigma,
            np.maximum(q - hi, 0) / self.weights.limit_sigma,
        ]
        return np.concatenate(rows, axis=-1)

    def cost(self, frame, states, controls):
        r = self.residual(frame, states)
        cost = np.sum(r * r, axis=-1)
        if frame < self.T:
            delta = controls - states[:, self.nq + self.nv :]
            cost += np.sum(
                self.control_scale
                * (
                    self.weights.control_weight * controls**2
                    + self.weights.slew_weight * delta**2
                ),
                axis=-1,
            )
        return cost

    def expand(self, states, controls):
        physical, eps = self.physical_nx, self.epsilon
        columns = 1 + physical
        delta = np.zeros((columns, self.nx))
        delta[1:, :physical] = eps * np.eye(physical)
        x = np.repeat(states, columns, axis=0)
        x = self.integrate(x, np.tile(delta, (self.T + 1, 1)))
        frames = np.repeat(np.arange(self.T + 1), columns)
        r = self.residual(frames, x).reshape(self.T + 1, columns, -1)
        J = (r[:, 1:] - r[:, :1]) / eps
        lx, lxx = (
            np.zeros((self.T + 1, self.nx)),
            np.zeros((self.T + 1, self.nx, self.nx)),
        )
        lx[:, :physical] = 2 * np.einsum("tkr,tr->tk", J, r[:, 0])
        lxx[:, :physical, :physical] = 2 * np.einsum(
            "tkr,tlr->tkl", J, J, optimize=True
        )
        delta_u = controls - states[:-1, self.nq + self.nv :]
        slew = self.weights.slew_weight * self.control_scale
        magnitude = self.weights.control_weight * self.control_scale
        lx[:-1, physical:] = -2 * slew * delta_u
        lxx[:-1, physical:, physical:] = np.diag(2 * slew)
        lu = 2 * magnitude * controls + 2 * slew * delta_u
        luu = np.tile(np.diag(2 * (magnitude + slew)), (self.T, 1, 1))
        lux = np.zeros((self.T, self.nu, self.nx))
        lux[:, :, physical:] = -np.diag(2 * slew)
        return lx, lxx, lu, luu, lux


def replay(model, initial_state, controls, substeps):
    """Independent serial MuJoCo rollout; no per-frame reset or external forces."""
    data = mujoco.MjData(model)
    data.qpos[:] = initial_state[: model.nq]
    data.qvel[:] = initial_state[model.nq : model.nq + model.nv]
    poses = [data.qpos.copy()]
    velocities = [data.qvel.copy()]
    for control in controls:
        data.ctrl[:], data.qacc_warmstart[:] = control, 0.0
        mujoco.mj_step(model, data, nstep=substeps)
        poses.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
    return np.c_[poses, velocities]
