"""Motor-demand audit with unactuated-base balance and explicit force witnesses.

This is an instantaneous feasibility test, not a physical replay. No forces
from MuJoCo's soft collision/limit solver are credited as motor assistance.
"""

from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import linprog

from .collision_manifest import collision_manifest

FORCE_TOLERANCE = 1e-6  # solver roundoff, in N or Nm; independent of frame rate
FRICTION = 0.6
CONE_SIDES = 8


def mechanical_limits(model_path, model, limits_urdf=None):
    """Keep dynamics in the production MJCF; use the original URDF limits."""
    path = Path(model_path).resolve()
    if limits_urdf is None:
        if path.suffix == ".urdf":
            limits_urdf = path
        elif path.name.startswith("g1_29dof_"):
            candidate = path.parent / "g1_29dof_popsicle.urdf"
            if candidate.is_file():
                limits_urdf = candidate
    velocity, effort = {}, {}
    if limits_urdf is not None:
        limits_urdf = Path(limits_urdf).resolve()
        for joint in ET.parse(limits_urdf).getroot().findall(".//joint"):
            limit = joint.find("limit")
            if joint.get("name") is None or limit is None:
                continue
            for key, dest in (("velocity", velocity), ("effort", effort)):
                if limit.get(key) is not None:
                    value = float(limit.get(key))
                    if not np.isfinite(value) or value <= 0:
                        raise ValueError(f"Invalid URDF {key} for {joint.get('name')}")
                    dest[joint.get("name")] = value
    for index in range(model.njnt):
        name = model.joint(index).name
        if model.jnt_actfrclimited[index]:
            lower, upper = model.jnt_actfrcrange[index]
            if not np.isclose(-lower, upper):
                raise ValueError(f"Asymmetric effort range needs explicit bounds: {name}")
            cap = float(upper)
            if name in effort and not np.isclose(cap, effort[name]):
                raise ValueError(f"MJCF/URDF effort mismatch: {name}")
            effort.setdefault(name, cap)
    provenance = {
        "dynamics_model": str(path),
        "dynamics_model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "collision_model_sha256": collision_manifest(model)["sha256"],
        "limits_urdf": str(limits_urdf) if limits_urdf else None,
        "limits_urdf_sha256": (
            hashlib.sha256(limits_urdf.read_bytes()).hexdigest() if limits_urdf else None
        ),
        "mass_kg": float(model.body_mass.sum()),
        "gravity_m_s2": model.opt.gravity.tolist(),
        "inertia_source": "compiled production MJCF explicit body inertials",
        "armature_included": True,
        "passive_forces_subtracted": True,
        "dry_friction": "sign(qvel) * dof_frictionloss; zero at rest",
        "soft_contact_and_joint_limit_forces_credited": False,
    }
    return velocity, effort, provenance


def derivatives(model, poses, fps):
    """Interior centered stencils, quaternion tangent vectors in frame t.

    Both interval velocities are expressed at the middle orientation, rather
    than subtracting angular velocities expressed at different orientations.
    Endpoints have no symmetric acceleration evidence and remain unaudited.
    """
    frames = np.arange(1, len(poses) - 1)
    velocity = np.empty((len(frames), model.nv))
    acceleration = np.empty_like(velocity)
    backward, forward = np.zeros(model.nv), np.zeros(model.nv)
    for index, frame in enumerate(frames):
        mujoco.mj_differentiatePos(model, backward, 1 / fps, poses[frame], poses[frame - 1])
        mujoco.mj_differentiatePos(model, forward, 1 / fps, poses[frame], poses[frame + 1])
        velocity[index] = (forward - backward) / 2
        acceleration[index] = (forward + backward) * fps
    return frames, velocity, acceleration


def motor_demand(model, data, pose, velocity, acceleration):
    """M a + bias - passive + dry friction, without soft constraint forces."""
    data.qpos[:] = pose
    data.qvel[:] = velocity
    mujoco.mj_fwdPosition(model, data)
    mujoco.mj_fwdVelocity(model, data)
    demand = np.empty(model.nv)
    mujoco.mj_mulM(model, data, demand, acceleration)
    demand += data.qfrc_bias - data.qfrc_passive
    demand += np.sign(velocity) * model.dof_frictionloss
    return demand


def friction_rays(normal, friction=FRICTION, *, outer=False):
    """Inner/outer polygons bracket the circular Coulomb cone."""
    normal = np.array(normal, float, copy=True)
    normal /= np.linalg.norm(normal)
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    first = np.cross(normal, axis)
    first /= np.linalg.norm(first)
    second = np.cross(normal, first)
    angles = np.arange(CONE_SIDES) * 2 * np.pi / CONE_SIDES
    radius = friction / np.cos(np.pi / CONE_SIDES) if outer else friction
    return normal[:, None] + radius * (
        first[:, None] * np.cos(angles) + second[:, None] * np.sin(angles)
    )


def contact_matrix(model, data, contacts, friction=FRICTION, *, outer=False):
    columns = []
    for contact in contacts:
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacobian, None, contact["point"], contact["body"])
        columns.append(jacobian.T @ friction_rays(contact["normal"], friction, outer=outer))
    return np.concatenate(columns, axis=1) if columns else np.empty((model.nv, 0))


def allocate(demand, mapping, base_dofs, joint_dofs, limits):
    """Minimize max |motor torque| / URDF effort with *exact* base balance.

    The minimum-norm force split alone cannot prove overload: another admissible
    split may satisfy all motors. This minimax LP tests existence instead.
    """
    count = mapping.shape[1]
    if not count:
        if np.max(np.abs(demand[base_dofs]), initial=0) > FORCE_TOLERANCE:
            return {"status": "base_infeasible", "torque": None, "utilization": None}
        torque = demand[joint_dofs].copy()
        return {"status": "balanced", "torque": torque,
                "utilization": float(np.max(np.abs(torque) / limits, initial=0)),
                "coefficients": np.empty(0), "base_residual": 0.0}
    joint_map = mapping[joint_dofs]
    # demand - J'f <= s*effort, and its negative <= s*effort.
    inequality = np.r_[np.c_[-joint_map, -limits], np.c_[joint_map, -limits]]
    rhs = np.r_[-demand[joint_dofs], demand[joint_dofs]]
    objective = np.r_[np.zeros(count), 1.0]
    equality = np.c_[mapping[base_dofs], np.zeros(len(base_dofs))]
    result = linprog(objective, A_ub=inequality, b_ub=rhs,
                     A_eq=equality if len(base_dofs) else None,
                     b_eq=demand[base_dofs] if len(base_dofs) else None,
                     bounds=[(0, None)] * (count + 1), method="highs")
    if not result.success:
        return {"status": "base_infeasible" if result.status == 2 else "numerical_unknown",
                "torque": None, "utilization": None}
    coeff = result.x[:-1]
    residual = float(np.max(np.abs(mapping[base_dofs] @ coeff - demand[base_dofs]), initial=0))
    torque = demand[joint_dofs] - joint_map @ coeff
    if residual > FORCE_TOLERANCE or np.min(coeff, initial=0) < -FORCE_TOLERANCE:
        return {"status": "numerical_unknown", "torque": None, "utilization": None}
    return {"status": "balanced", "torque": torque,
            "utilization": float(np.max(np.abs(torque) / limits, initial=0)),
            "coefficients": coeff, "base_residual": residual}
