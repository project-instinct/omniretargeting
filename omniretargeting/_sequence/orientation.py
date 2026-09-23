"""Calibrated robot-frame orientation targets, independent of root translation."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .._refinement.math import _skew_symmetric


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

    def rows(self, data, columns, frame):
        residuals, matrices, errors = [], [], []
        for body, parent, targets, sigma in self.targets:
            current = data.xmat[body].reshape(3, 3)
            angular = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, data, None, angular, body)
            if parent is not None:
                parent_rotation = data.xmat[parent].reshape(3, 3)
                parent_angular = np.zeros_like(angular)
                mujoco.mj_jacBody(self.model, data, None, parent_angular, parent)
                current = parent_rotation.T @ current
                angular = parent_rotation.T @ (angular - parent_angular)
            target = targets[frame]
            # Three axes constrain pitch, roll and heading. Division by sqrt(2)
            # gives the small-angle rotation-vector norm, without an Euler wrap.
            scale = np.sqrt(2.0) * sigma
            residuals.append((current - target).T.ravel() / scale)
            matrices.append(
                np.vstack(
                    [
                        -_skew_symmetric(current[:, axis]) @ angular[:, columns]
                        for axis in range(3)
                    ]
                )
                / scale
            )
            errors.append(float(Rotation.from_matrix(target @ current.T).magnitude()))
        return residuals, matrices, max(errors, default=0.0)
