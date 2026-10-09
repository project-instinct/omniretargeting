"""Exact mechanical screen for an added correction, matching the saved audit.

This is a kinematic inverse-dynamics screen, not physical replay or a contact
force feasibility certificate. Pre-existing violations remain visible; a
refinement may reduce them but may not create or enlarge one.
"""
import mujoco
import numpy as np

from .._sequence.continuity_guard import ACCEPTANCE_SLACK
from . import deadline_state


def inverse_forces(model, poses, fps):
    velocity = np.empty((len(poses) - 1, model.nv))
    for frame in range(len(poses) - 1):
        mujoco.mj_differentiatePos(model, velocity[frame], 1. / fps,
                                 poses[frame], poses[frame + 1])
    acceleration = np.diff(velocity, axis=0) * fps
    data = mujoco.MjData(model)
    force = np.empty((len(poses) - 2, model.nv))
    for frame in range(len(poses) - 2):
        deadline_state.check('projection_effort_screen')
        data.qpos[:], data.qvel[:], data.qacc[:] = poses[frame + 2], velocity[frame + 1], acceleration[frame]
        mujoco.mj_inverse(model, data)
        force[frame] = data.qfrc_inverse
    return force


class MechanicalScreen:
    def __init__(self, objective, entry):
        self.model, self.fps = objective.model, objective.framerate
        joints = np.flatnonzero(self.model.jnt_actfrclimited)
        self.joints, self.dofs = joints, self.model.jnt_dofadr[joints]
        limits = np.max(np.abs(self.model.jnt_actfrcrange[joints]), axis=1)
        initial = np.abs(inverse_forces(self.model, entry, self.fps)[:, self.dofs])
        self.bounds = np.maximum(initial, limits[None] + ACCEPTANCE_SLACK * self.fps)
        self.limits = limits

    def rejection(self, poses):
        force = np.abs(inverse_forces(self.model, poses, self.fps)[:, self.dofs])
        excess = force - self.bounds
        if not np.any(excess > 1e-7):
            return None
        frame, column = np.unravel_index(np.argmax(excess), excess.shape)
        data = mujoco.MjData(self.model)
        data.qpos[:] = poses[frame + 2]
        mujoco.mj_forward(self.model, data)
        contacts = sorted((c for c in data.contact if c.dist < 0), key=lambda c: c.dist)
        overlap = [dict(geoms=[self.model.geom(int(g)).name for g in c.geom],
                        bodies=[self.model.body(int(self.model.geom_bodyid[g])).name for g in c.geom],
                        depth_m=float(-c.dist)) for c in contacts[:6]]
        return dict(kind='effort_screen', frame=int(frame + 2), native_self_contacts=overlap,
                    joint=self.model.joint(int(self.joints[column])).name,
                    required=float(force[frame, column]), allowed=float(self.bounds[frame, column]),
                    actuator_limit=float(self.limits[column]))


def velocity_jacobians(model, poses, fps):
    """Execution-stencil derivatives in MuJoCo integration coordinates.

    Scalar joints and translations are linear; only free-root rotation needs
    differentiating its quaternion subtraction. No pose/sample is modified.
    """
    velocity = np.empty((len(poses) - 1, model.nv))
    first = np.tile(-fps * np.eye(model.nv), (len(velocity), 1, 1))
    second = -first.copy()
    rotational = []
    for joint in range(model.njnt):
        if model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
            rotational.extend(range(model.jnt_dofadr[joint] + 3, model.jnt_dofadr[joint] + 6))
    eps = 1e-6
    for frame in range(len(velocity)):
        deadline_state.check('projection_velocity_jacobians')
        mujoco.mj_differentiatePos(model, velocity[frame], 1. / fps, poses[frame], poses[frame + 1])
        for side, jac in ((0, first), (1, second)):
            for column in rotational:
                delta = np.zeros(model.nv)
                delta[column] = 1.
                plus, minus = poses[frame + side].copy(), poses[frame + side].copy()
                mujoco.mj_integratePos(model, plus, delta, eps)
                mujoco.mj_integratePos(model, minus, delta, -eps)
                vp, vm = np.empty(model.nv), np.empty(model.nv)
                a, b = (plus, poses[frame + 1]) if side == 0 else (poses[frame], plus)
                mujoco.mj_differentiatePos(model, vp, 1. / fps, a, b)
                a, b = (minus, poses[frame + 1]) if side == 0 else (poses[frame], minus)
                mujoco.mj_differentiatePos(model, vm, 1. / fps, a, b)
                jac[frame, :, column] = (vp - vm) / (2 * eps)
    return velocity, first, second


def mechanical_rows(screen, coordinates, poses):
    """Differentiate the exact inverse-dynamics screen on its three-frame stencil."""
    from scipy import sparse
    model, fps, dofs = screen.model, screen.fps, screen.dofs
    frames, width = len(poses), coordinates.width
    count = len(dofs)
    if not count:
        return sparse.csc_matrix((0, frames * width)), np.empty(0), np.empty(0)
    velocity, first, second = velocity_jacobians(model, poses, fps)
    acceleration = np.diff(velocity, axis=0) * fps
    data = mujoco.MjData(model)
    dq, dv, da = (np.empty((model.nv, model.nv)) for _ in range(3))
    rows, columns, values, forces = [], [], [], []
    for frame in range(frames - 2):
        deadline_state.check('projection_mechanical_rows')
        data.qpos[:], data.qvel[:], data.qacc[:] = poses[frame + 2], velocity[frame + 1], acceleration[frame]
        mujoco.mj_inverse(model, data)
        forces.append(data.qfrc_inverse[dofs].copy())
        mujoco.mjd_inverseFD(model, data, 1e-6, False, dq, dv, da, None, None, None, None)
        pos, vel, acc = dq.T[dofs], dv.T[dofs], da.T[dofs]
        blocks = (
            -fps * acc @ first[frame],
            vel @ first[frame + 1] + fps * acc @ (first[frame + 1] - second[frame]),
            pos + vel @ second[frame + 1] + fps * acc @ second[frame + 1],
        )
        for offset, block in enumerate(blocks):
            block = block[:, coordinates.cols]
            row, col = np.nonzero(block)
            rows.extend(frame * count + row)
            columns.extend((frame + offset) * width + col)
            values.extend(block[row, col])
    jacobian = sparse.csc_matrix((values, (rows, columns)), shape=((frames - 2) * count, frames * width))
    force = np.asarray(forces).ravel()
    # The exact check keeps its cap. Reserve room for native FD/roundoff here.
    cap = screen.bounds.ravel() - np.tile(screen.limits * 1e-4, frames - 2)
    return jacobian, -cap - force, cap - force
