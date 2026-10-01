"""Choose a continuous IK branch from already calibrated source rotations."""

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .coordinates import Coordinates
from .orientation import OrientationTargets


def source_orientation_seed(model, initial, roots, priority, specifications, config):
    """Kinematic initialization only; source, scene and target times stay fixed.

    Source rotations disambiguate axial twist and bent-limb branches that bone
    directions alone cannot identify. Consecutive frames warm-start from the
    previous scalar solution, then the coupled reference solve handles temporal
    objectives and endpoints. No contact, collision or action labels enter here.
    """
    q = np.asarray(initial, float).copy()
    if not specifications:
        return q
    coords = Coordinates(model)
    orientations = OrientationTargets(model, len(q), specifications)
    data = mujoco.MjData(model)
    root_body = (
        None
        if coords.root_d is None
        else int(model.jnt_bodyid[model.dof_jntid[coords.root_d]])
    )
    anchor = priority.get("source_root_robot_body")
    anchor_body = root_body if anchor is None else model.body(anchor).id
    for frame in range(len(q)):
        if frame:
            q[frame, coords.qi] = q[frame - 1, coords.qi]
        if root_body is not None and not config.get("lock_root_orientation", False):
            for body, parent, targets, _ in orientations.targets:
                if body == root_body and parent is None:
                    q[frame, coords.root_q + 3 : coords.root_q + 7] = (
                        Rotation.from_matrix(targets[frame]).as_quat(scalar_first=True)
                    )
        cached = {}

        def evaluate(values, frame=frame, cached=cached):
            key = values.tobytes()
            if cached.get("key") != key:
                data.qpos[:] = q[frame]
                data.qpos[coords.qi] = values
                mujoco.mj_kinematics(model, data)
                mujoco.mj_comPos(model, data)
                residuals, jacobians, _ = orientations.rows(data, coords.di, frame)
                cached.update(
                    key=key,
                    residual=np.concatenate(residuals),
                    jacobian=np.vstack(jacobians),
                )
            return cached

        if len(coords.qi):
            fitted = least_squares(
                lambda x: evaluate(x)["residual"],
                np.clip(q[frame, coords.qi], coords.lower, coords.upper),
                jac=lambda x: evaluate(x)["jacobian"],
                bounds=(coords.lower, coords.upper),
            )
            q[frame, coords.qi] = fitted.x
        if root_body is not None:
            data.qpos[:] = q[frame]
            mujoco.mj_kinematics(model, data)
            offset = np.asarray(roots[frame]) - data.xpos[anchor_body]
            if config.get("lock_root_xy", False):
                offset[:2] = 0
            if config.get("lock_root_z", False):
                offset[2] = 0
            q[frame, coords.root_q : coords.root_q + 3] += offset
    return q
