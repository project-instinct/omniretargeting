"""Bound cumulative translation after acquiring the source/contact pose.

Posture has no angular limit (``motion_budgets``): joint, body-rotation and
per-bone source-direction changes are reported against the fixed reference and
the immutable source, not bounded.
"""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..conic_constraints import LocalNormBounds
from ..scene_geometry import collision_geom_ids


class SequencePoseGuard:
    def __init__(self, tasks, reference):
        self.tasks, self.reference = tasks, np.array(reference, copy=True)
        self.reference.setflags(write=False)
        # The root and slide joints keep the configured translation bound.
        self.translation = tasks.config["translation_step_m"]
        self.joints = tasks.model.dof_jntid[tasks.coords.di]
        self.slide = tasks.model.jnt_type[self.joints] == mujoco.mjtJoint.mjJNT_SLIDE
        bodies = (
            set(tasks.bodyids)
            | {spec["body_id"] for spec in tasks.effectors.values()}
            | set(map(int, tasks.model.geom_bodyid[collision_geom_ids(tasks.model)]))
        )
        # Welded descendants share exactly the same world rotation change.
        self.bodies = sorted({int(tasks.model.body_weldid[b]) for b in bodies})
        self.rotations = []
        self.data = mujoco.MjData(tasks.model)
        for pose in reference:
            self.data.qpos[:] = pose
            mujoco.mj_forward(tasks.model, self.data)
            self.rotations.append(self.data.xmat[self.bodies].copy().reshape(-1, 3, 3))
        # Mapped bone directions of the immutable human, for reporting.
        mapped = {body: i for i, body in enumerate(tasks.bodyids)}
        self.edges = []
        for body, child in mapped.items():
            parent = int(tasks.model.body_parentid[body])
            while parent and parent not in mapped:
                parent = int(tasks.model.body_parentid[parent])
            if parent in mapped:
                self.edges.append((mapped[parent], child))
        self.direction_targets = np.zeros((len(reference), len(self.edges), 3))
        self.direction_valid = np.zeros((len(reference), len(self.edges)), bool)
        for f, pose in enumerate(reference):
            self.data.qpos[:] = pose
            mujoco.mj_forward(tasks.model, self.data)
            for k, (a, b) in enumerate(self.edges):
                target = tasks.human[f, b] - tasks.human[f, a]
                actual = (
                    self.data.xpos[tasks.bodyids[b]] - self.data.xpos[tasks.bodyids[a]]
                )
                if min(np.linalg.norm(target), np.linalg.norm(actual)) <= 1e-10:
                    continue
                self.direction_targets[f, k] = target / np.linalg.norm(target)
                self.direction_valid[f, k] = True

    def constrain_step(self, poses, lower, upper, frames=None):
        """Clip a step box; with ``frames`` the box rows span that window."""
        window = slice(None) if frames is None else slice(frames.start, frames.stop)
        coords = self.tasks.coords
        for q, column, slide in zip(coords.qi, coords.scalar_cols, self.slide):
            if not slide:
                continue
            change = self.reference[window, q] - poses[window, q]
            lower[:, column] = np.maximum(lower[:, column], change - self.translation)
            upper[:, column] = np.minimum(upper[:, column], change + self.translation)

    def rows(self, poses, *, jacobians, frames=None):
        coords = self.tasks.coords
        if coords.root_q is None:
            return
        q = coords.root_q
        jac = np.zeros((3, coords.width))
        for axis in range(3):
            column = coords.inverse.get(coords.root_d + axis)
            if column is not None:
                jac[axis, column] = 1.0
        for f in range(len(poses)) if frames is None else frames:
            yield (
                f,
                poses[f, q : q + 3] - self.reference[f, q : q + 3],
                jac if jacobians else None,
                self.translation,
            )

    def constraints(self, poses, flat, frames=None):
        window = range(len(poses)) if frames is None else frames
        rows = LocalNormBounds(len(window), self.tasks.coords.width)
        for f, residual, jac, budget in self.rows(
            poses, jacobians=True, frames=frames
        ):
            rows.add(residual, budget, (f - window.start, jac))
        return rows.constraints(flat)

    def accepts(self, poses, frames=None):
        window = slice(None) if frames is None else slice(frames.start, frames.stop)
        coords = self.tasks.coords
        for q, slide in zip(coords.qi, self.slide):
            if not slide:
                continue
            change = abs(poses[window, q] - self.reference[window, q])
            if np.any(change > self.translation + 1e-7):
                return False
        return all(
            np.linalg.norm(residual) <= budget + 1e-7
            for _, residual, _, budget in self.rows(
                poses, jacobians=False, frames=frames
            )
        )

    def source_direction_angles(self, poses):
        """Angle (frames, edges) of each mapped bone to the source direction."""
        tasks = self.tasks
        angles = np.zeros(self.direction_valid.shape)
        for f, pose in enumerate(poses):
            self.data.qpos[:] = pose
            mujoco.mj_kinematics(tasks.model, self.data)
            for k, (a, b) in enumerate(self.edges):
                if not self.direction_valid[f, k]:
                    continue
                vector = self.data.xpos[tasks.bodyids[b]] - self.data.xpos[tasks.bodyids[a]]
                norm = np.linalg.norm(vector)
                if norm <= 1e-10:
                    angles[f, k] = np.pi
                    continue
                cosine = np.dot(vector / norm, self.direction_targets[f, k])
                angles[f, k] = np.arccos(np.clip(cosine, -1.0, 1.0))
        return angles

    def diagnostics(self, poses):
        """Report posture changes against the fixed reference and the source."""
        coords = self.tasks.coords
        joint_angles = np.abs(
            poses[:, coords.qi[~self.slide]] - self.reference[:, coords.qi[~self.slide]]
        )
        body_angles = np.zeros((len(poses), len(self.bodies)))
        for f, pose in enumerate(poses):
            self.data.qpos[:] = pose
            mujoco.mj_forward(self.tasks.model, self.data)
            body_angles[f] = Rotation.from_matrix(
                self.data.xmat[self.bodies].reshape(-1, 3, 3)
                @ np.swapaxes(self.rotations[f], -1, -2)
            ).magnitude()
        return {
            "reference": "fixed_before_contact_acquisition",
            "angular_bound": None,
            "maximum_joint_change_rad": float(joint_angles.max(initial=0)),
            "maximum_body_rotation_change_rad": float(body_angles.max(initial=0)),
            "maximum_source_bone_angle_rad": float(self.source_direction_angles(poses).max(initial=0)),
            "within_translation_bound": bool(self.accepts(poses)),
        }
