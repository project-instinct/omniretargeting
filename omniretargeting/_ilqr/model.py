"""Explicit actuator and scene adapter for the first G1 flat-box experiment."""

import re
from pathlib import Path

import mujoco
import numpy as np
import trimesh

# Values mirror InstinctMJ's BeyondMimic G1 actuator definitions. No PD spring
# is added: iLQR commands bounded direct motor torque, with passive damping.
G1_MOTORS = (
    (r"(?:left|right)_hip_(?:pitch|yaw)_joint|waist_yaw_joint", 88.0, 0.010177520),
    (r"(?:left|right)_(?:hip_roll|knee)_joint", 139.0, 0.025101925),
    (
        r"(?:left|right)_ankle_(?:pitch|roll)_joint|waist_(?:pitch|roll)_joint",
        50.0,
        0.007219450,
    ),
    (
        r"(?:left|right)_(?:shoulder_(?:pitch|roll|yaw)|elbow|wrist_roll)_joint",
        25.0,
        0.003609725,
    ),
    (r"(?:left|right)_wrist_(?:pitch|yaw)_joint", 5.0, 0.00425),
)


def box_scene_bounds(mesh):
    """Accept only an exact axis-aligned solid box, never approximate a scene."""
    if (
        not isinstance(mesh, trimesh.Trimesh)
        or not mesh.is_watertight
        or not mesh.is_convex
    ):
        raise ValueError(
            "First iLQR experiment supports an existing solid box scene only"
        )
    bounds = mesh.bounds
    extent = np.diff(bounds, axis=0)[0]
    if np.any(extent <= 0) or not np.isclose(
        abs(mesh.volume), np.prod(extent), rtol=1e-6
    ):
        raise ValueError(
            "Scene is not an axis-aligned box; no convex-hull substitution is allowed"
        )
    coordinates = np.asarray(mesh.vertices)
    on_boundary = np.isclose(coordinates, bounds[0], atol=1e-7) | np.isclose(
        coordinates, bounds[1], atol=1e-7
    )
    if not np.all(np.any(on_boundary, axis=1)):
        raise ValueError("Scene contains geometry inside the box boundary")
    return bounds


def build_g1_model(xml_path, mesh, fps, substeps=17):
    """Preserve kinematics, masses and collision shells; add all 29 motors.

    Preserve the exact source scene box, including its thickness and position.
    No lift, scaling, floor extension, root force, or kinematic root clamp.
    """
    if not np.isfinite(fps) or fps <= 0 or substeps < 1:
        raise ValueError("fps and substeps must be positive")
    bounds = box_scene_bounds(mesh)
    spec = mujoco.MjSpec.from_file(str(Path(xml_path).resolve()))
    original = spec.compile()
    if original.nu:
        raise ValueError("G1 experiment expects the unactuated retargeting MJCF")
    scalar_joints = [j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_HINGE]
    if len(scalar_joints) != 29 or original.nq != 36 or original.nv != 35:
        raise ValueError("Expected the complete free-base, 29-hinge G1 model")
    motors = []
    for joint in scalar_joints:
        matches = [row for row in G1_MOTORS if re.fullmatch(row[0], joint.name)]
        if len(matches) != 1:
            raise ValueError(f"No unique motor specification for {joint.name}")
        _, limit, armature = matches[0]
        damping = 2 * 2.0 * armature * (10 * 2 * np.pi)
        joint.armature = armature
        joint.damping = [damping, 0.0, 0.0]
        spec.add_actuator(
            name=joint.name,
            target=joint.name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gaintype=mujoco.mjtGain.mjGAIN_FIXED,
            gainprm=[1.0] + [0.0] * 9,
            biastype=mujoco.mjtBias.mjBIAS_NONE,
            dyntype=mujoco.mjtDyn.mjDYN_NONE,
            gear=[1, 0, 0, 0, 0, 0],
            ctrllimited=True,
            ctrlrange=[-limit, limit],
        )
        motors.append(
            {
                "joint": joint.name,
                "limit_nm": limit,
                "armature": armature,
                "damping": damping,
            }
        )
    for joint in spec.joints:
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            joint.armature = 0.0
            joint.damping = [0.0, 0.0, 0.0]
    # Force dense Jacobians for the small inverse-dynamics contact fit.
    spec.option.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    spec.option.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.timestep = 1 / (float(fps) * substeps)
    spec.worldbody.add_geom(
        name="ilqr_source_box",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=bounds.mean(axis=0),
        size=(bounds[1] - bounds[0]) / 2,
        contype=1,
        conaffinity=1,
        condim=3,
        friction=[0.6, 0.005, 0.0001],
        rgba=[0.7, 0.7, 0.7, 1],
    )
    model = spec.compile()
    # Resolve by names; never assume the free body is the pelvis or joint order
    # agrees with the upstream pelvis-base model.
    for j in range(original.njnt):
        name = original.joint(j).name
        other = model.joint(name).id
        if (
            original.jnt_qposadr[j] != model.jnt_qposadr[other]
            or original.jnt_dofadr[j] != model.jnt_dofadr[other]
        ):
            raise ValueError("Compiling the experiment reordered robot coordinates")
    return (
        original,
        model,
        {
            "source_xml": str(Path(xml_path).resolve()),
            "motors": motors,
            "motor_reference": "InstinctMJ assets/unitree_g1.py BEYONDMIMIC_G1_29DOF_*",
            "controller": "direct torque + specified passive damping, no PD position spring",
            "free_base_damping_and_armature": 0.0,
            "scene": "exact source axis-aligned box",
            "scene_bounds": bounds,
            "scene_friction_assumption": [0.6, 0.005, 0.0001],
            "timestep": model.opt.timestep,
            "substeps": substeps,
            "robot_scale": 1.0,
            "scene_scale": 1.0,
        },
    )
