"""Calibrated robot-frame orientation targets, independent of root translation."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


class OrientationTargets:
    """World or parent-relative rotation residuals with exact axis Jacobians.

    Targets are robot-frame matrices supplied by the adapter/core. No source
    skeleton conventions, Euler angles or fitted robot poses are inferred here.
    """

    def __init__(self, model, frames, specifications):
        self.model = model
        self.targets = []
        for spec in specifications:
            rotations = np.asarray(spec["rotations"], dtype=float).copy()
            if rotations.shape != (frames, 3, 3) or not np.all(np.isfinite(rotations)):
                raise ValueError("Orientation targets must be finite (frames, 3, 3)")
            if not np.allclose(
                rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-6
            ) or not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-6):
                raise ValueError("Orientation targets must be proper rotation matrices")
            sigma = float(spec["sigma"])
            if not np.isfinite(sigma) or sigma <= 0:
                raise ValueError("Orientation sigma must be finite and positive")
            rotations.setflags(write=False)
            self.targets.append(
                (
                    model.body(spec["robot_body"]).id,
                    model.body(spec["parent_body"]).id
                    if spec.get("parent_body")
                    else None,
                    rotations,
                    sigma,
                )
            )
        self.bodies = np.array([body for body, _, _, _ in self.targets], dtype=int)
        self.parents = np.array([0 if parent is None else parent for _, parent, _, _ in self.targets], dtype=int)
        self.relative = np.array([parent is not None for _, parent, _, _ in self.targets], dtype=bool)
        self.jacobian_bodies = np.unique(np.concatenate([self.bodies, self.parents]))
        self.scales = np.sqrt(2.0) * np.array([sigma for _, _, _, sigma in self.targets])

    def rows(self, data, columns, frame):
        if not self.targets:
            return [], [], 0.0
        # A parent can appear in many targets: compute its angular Jacobian
        # once and batch the tiny axis products over all targets.
        body_angular = np.zeros((self.model.nbody, 3, self.model.nv))
        for body in self.jacobian_bodies:
            if body:
                mujoco.mj_jacBody(self.model, data, None, body_angular[body], int(body))
        current = data.xmat[self.bodies].reshape(-1, 3, 3).copy()
        angular = body_angular[self.bodies]
        relative = self.relative
        parent_rotation = data.xmat[self.parents[relative]].reshape(-1, 3, 3).transpose(0, 2, 1)
        current[relative] = parent_rotation @ current[relative]
        angular[relative] = parent_rotation @ (angular[relative] - body_angular[self.parents[relative]])
        targets = np.stack([targets[frame] for _, _, targets, _ in self.targets])
        # Three axes constrain pitch, roll and heading. Division by sqrt(2)
        # gives the small-angle rotation-vector norm, without an Euler wrap.
        residuals = (current - targets).transpose(0, 2, 1).reshape(-1, 9) / self.scales[:, None]
        axes = current.transpose(0, 2, 1)
        cross = np.zeros((*axes.shape, 3))
        x, y, z = (axes[..., axis] for axis in range(3))
        cross[..., 0, 1], cross[..., 0, 2] = z, -y
        cross[..., 1, 0], cross[..., 1, 2] = -z, x
        cross[..., 2, 0], cross[..., 2, 1] = y, -x
        matrices = (cross @ angular[:, None, :, columns]).reshape(len(self.targets), 9, len(columns))
        matrices /= self.scales[:, None, None]
        errors = targets @ current.transpose(0, 2, 1)
        return list(residuals), list(matrices), float(Rotation.from_matrix(errors).magnitude().max())
