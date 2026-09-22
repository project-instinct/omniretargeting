"""Conservative collision-body travel bounds for saved-qpos interpolation."""

import mujoco
import numpy as np

from ._refinement.math import _scalar_joint_coordinates


class CollisionMotionBounds:
    """Bound each geom's travel using its ancestors and compiled dimensions."""

    def __init__(self, model, geom_ids):
        self.model = model
        self.geom_ids = np.asarray(geom_ids, dtype=int)
        self.qi, self.di, _, _ = _scalar_joint_coordinates(model)
        self.joint_ids = np.array([
            np.flatnonzero(model.jnt_qposadr == qi)[0] for qi in self.qi
        ], dtype=int)
        self.levers = np.zeros((len(self.geom_ids), len(self.qi)))
        self.root_levers = np.zeros(len(self.geom_ids))
        self.slide_ancestors = np.zeros_like(self.levers, dtype=bool)
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        common_root = int(model.jnt_bodyid[free[0]]) if len(free) == 1 else -1
        self.common_free_root = common_root >= 0
        for row, gid in enumerate(self.geom_ids):
            bid = int(model.geom_bodyid[gid])
            radius = float(model.geom_rbound[gid]) + float(np.linalg.norm(model.geom_pos[gid]))
            ancestors = set()
            while bid > 0:
                ancestors.add(bid)
                for col, jid in enumerate(self.joint_ids):
                    if model.jnt_bodyid[jid] == bid:
                        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE:
                            self.levers[row, col] = 1.0
                            self.slide_ancestors[row, col] = True
                        else:
                            self.levers[row, col] = radius + float(np.linalg.norm(model.jnt_pos[jid]))
                radius += float(np.linalg.norm(model.body_pos[bid]))
                bid = int(model.body_parentid[bid])
            self.root_levers[row] = radius
            self.common_free_root &= common_root in ancestors
        self.geom_rows = {int(g): i for i, g in enumerate(self.geom_ids)}

    def displacements(self, first, second, *, relative=False):
        model = self.model
        dq = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, dq, 1.0, first, second)
        slides = model.jnt_type[self.joint_ids] == mujoco.mjtJoint.mjJNT_SLIDE
        excursions = self.slide_ancestors @ np.maximum(np.abs(first[self.qi]), np.abs(second[self.qi]))
        levers = self.levers + excursions[:, None] * (self.levers > 0) * (~slides)[None, :]
        motion = levers @ np.abs(dq[self.di])
        for jid in range(model.njnt):
            dof = int(model.jnt_dofadr[jid])
            if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                if relative and self.common_free_root:
                    continue  # Common rigid transforms leave all self distances unchanged.
                motion += np.linalg.norm(dq[dof:dof+3]) + (
                    self.root_levers + excursions
                ) * np.linalg.norm(dq[dof+3:dof+6])
            elif model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_BALL:
                motion += (self.root_levers + excursions) * np.linalg.norm(dq[dof:dof+3])
        return motion

    def pair_bound(self, pairs, first, second):
        motion = self.displacements(first, second, relative=True)
        return max((motion[self.geom_rows[int(a)]] + motion[self.geom_rows[int(b)]]
                    for a, b in pairs), default=0.0)
