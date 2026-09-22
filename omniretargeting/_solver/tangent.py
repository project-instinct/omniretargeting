"""Physical SQP increments compatible with existing qpos task Jacobians.

The upstream OmniRetargeting solver uses MuJoCo tangent increments, physical
block limits, and mj_integratePos. This maps those increments into the existing
task expressions without changing their contact targets or priority levels.
"""

import cvxpy as cp
import mujoco
import numpy as np


def enabled(retargeter):
    mode = retargeter.end_effector_config.get("step_coordinates", "qpos")
    if mode not in ("qpos", "tangent"):
        raise ValueError("step_coordinates must be 'qpos' or 'tangent'.")
    return mode == "tangent"


def tangent_basis(retargeter, q):
    """Return qpos derivative embedding, active velocity indices and limits."""
    model = retargeter.robot_model
    indices = np.asarray(retargeter.q_a_indices)
    selected = {int(address): i for i, address in enumerate(indices)}
    retargeter.robot_data.qpos[:] = q
    transform = retargeter._build_transform_qdot_to_qvel_fast()
    columns, dofs, limits = [], [], []

    def append(values, dof, limit):
        columns.append(values[indices])
        dofs.append(dof)
        limits.append(limit)

    for joint in range(model.njnt):
        address = int(model.jnt_qposadr[joint])
        dof = int(model.jnt_dofadr[joint])
        kind = model.jnt_type[joint]
        if kind == mujoco.mjtJoint.mjJNT_FREE:
            for axis in range(3):
                if address + axis in selected:
                    column = np.zeros(model.nq)
                    column[address + axis] = 1.
                    append(column, dof + axis, min(retargeter.step_size, .05) if axis == 2 else retargeter.step_size)
            quaternion = np.arange(address + 3, address + 7)
            active = [int(x) in selected for x in quaternion]
            if any(active) and not all(active):
                raise ValueError("A tangent step requires all or no quaternion components.")
            if all(active):
                # For a unit quaternion T T^T = 4 I on the angular block.
                embedding = .25 * transform[dof + 3:dof + 6, quaternion].T
                for axis in range(3):
                    column = np.zeros(model.nq)
                    column[quaternion] = embedding[:, axis]
                    append(column, dof + 3 + axis, retargeter.step_size)
        elif kind in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            if address in selected:
                column = np.zeros(model.nq)
                column[address] = 1.
                limit = retargeter.step_size
                if model.jnt_limited[joint]:
                    limit = min(limit, .1 * np.ptp(model.jnt_range[joint]))
                append(column, dof, limit)
        else:
            raise NotImplementedError("Tangent steps support free and scalar joints.")
    return np.column_stack(columns), np.asarray(dofs), np.asarray(limits)


def make_step(retargeter, q):
    embedding, dofs, limits = tangent_basis(retargeter, q)
    velocity = cp.Variable(len(dofs), name="tangent_step")
    return embedding @ velocity, [velocity >= -limits, velocity <= limits]


def temporal_residual(retargeter, q, previous, dqa):
    difference = np.zeros(retargeter.robot_model.nv)
    mujoco.mj_differentiatePos(retargeter.robot_model, difference, 1., previous, q)
    _, dofs, _ = tangent_basis(retargeter, q)
    transform = retargeter._build_transform_qdot_to_qvel_fast()[:, retargeter.q_a_indices]
    return difference[dofs] + transform[dofs] @ dqa


def integrate_step(retargeter, q, step):
    retargeter.robot_data.qpos[:] = q
    transform = retargeter._build_transform_qdot_to_qvel_fast()[:, retargeter.q_a_indices]
    result = q.copy()
    mujoco.mj_integratePos(retargeter.robot_model, result, transform @ step, 1.)
    return result


def constraint_violation(problem):
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        return np.inf
    violations = []
    for constraint in problem.constraints:
        try:
            violation = np.asarray(constraint.violation(), dtype=float)
        except ValueError:
            return np.inf
        if not np.isfinite(violation).all():
            return np.inf
        violations.append(float(np.max(np.abs(violation), initial=0.)))
    return max(violations, default=0.)


def usable(problem, retargeter):
    if enabled(retargeter):
        return constraint_violation(problem) <= 1e-5
    return problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
