"""Shared, immutable learned surface identities for pose and contact tasks.

Source adapters own mesh extraction. This module accepts metric barycentric
surfaces from any adapter and never modifies its arrays or scene registration.
"""

import numpy as np

from .surface_learning import bind_whole_body
from .surface_tasks import SurfaceKinematics, _face_frames


class SurfaceMotion:
    def __init__(self, model, source, reference_qpos, config=None, mapping=None):
        self.model = model
        self.bindings, self.binding_valid, self.diagnostics = bind_whole_body(
            model, source, reference_qpos, config, joint_mapping=mapping
        )
        full = source["full_body"]
        self.world_tracking = np.array(
            full.get("sample_world_tracking", np.zeros(len(self.bindings))),
            dtype=bool,
            copy=True,
        )
        if self.world_tracking.shape != (len(self.bindings),):
            raise ValueError(
                "Surface world-tracking labels must follow sample identities"
            )
        triangles = np.asarray(full["triangles"], dtype=float)
        bary = np.asarray(full["barycentric"], dtype=float)
        if triangles.ndim != 4 or triangles.shape[1:] != (len(bary), 3, 3):
            raise ValueError("Surface motion must have shape (frames, samples, 3, 3)")
        self.frames, valid = _face_frames(triangles)
        rest = np.asarray(full["reference_vertices"])[
            np.asarray(full["faces"])[np.asarray(full["face_ids"])]
        ]
        self.reference_frames, rest_valid = _face_frames(rest)
        self.valid = valid & rest_valid[None, :] & self.binding_valid[None, :]
        self.points = np.einsum("tnij,ni->tnj", triangles, bary)
        self.points[~self.valid] = 0.0
        self.frames[~self.valid] = np.eye(3)
        self.reference_points = np.einsum("nij,ni->nj", rest, bary)
        self.reference_qpos = np.array(reference_qpos, dtype=float, copy=True)
        kin = SurfaceKinematics(model, self.bindings)
        values = [kin.at(self.reference_qpos, i) for i in range(len(bary))]
        self.robot_points = np.array([v[0] for v in values])
        self.robot_normals = np.array([v[1] for v in values])
        self.body_ids = np.array(
            [model.geom_bodyid[model.geom(b.robot_geom).id] for b in self.bindings],
            dtype=int,
        )
        self.robot_rotations = np.array(
            [
                kin.data.geom_xmat[model.geom(b.robot_geom).id].reshape(3, 3)
                for b in self.bindings
            ]
        )
        self.normal_targets = np.einsum(
            "tnij,nkj,nk->tni", self.frames, self.reference_frames, self.robot_normals
        )
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)

    @property
    def binding_result(self):
        return self.bindings, self.binding_valid, self.diagnostics
