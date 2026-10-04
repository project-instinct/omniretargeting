"""Original mesh hinges and Jacobians, restricted by a physical band."""

import mujoco
import numpy as np

from .batched_self_collision import BatchedSelfCollision, NativeUnavailable
from ..geometry_query import GeometryQueryError


class _RestrictedPlanes(BatchedSelfCollision):
    def _nearby(self, positions, rotations):
        # The caller's native physical query certifies every omitted pair.
        # Only this physical band mask enters the old support/Qhull kernel.
        return self.mask, np.zeros(self.mask.shape)


class BandMeshHinges:
    """Retain the old per-row force/curvature without far mesh queries."""

    def __init__(self, query):
        self.source = query
        self.query = query = query.fresh_state()
        self.mesh_pairs = np.all(query.model.geom_type[query.pairs] == mujoco.mjtGeom.mjGEOM_MESH, axis=1)
        try:
            self.batch = _RestrictedPlanes(query, workers=1)
        except NativeUnavailable:
            self.batch = None
        self.cutoff = None

    def rows(self, poses, kin, native_rows, with_jacobian, target, cutoff):
        self.query.cfg['detection_distance'] = self.source.cfg['detection_distance']
        mask = np.zeros((len(poses), len(self.query.pairs)), np.uint8)
        for frame, rows in enumerate(native_rows):
            for row in rows:
                if self.mesh_pairs[row['pair_index']]:
                    mask[frame, row['pair_index']] = 1
        if self.batch is not None:
            if self.cutoff != cutoff:
                self.batch.cache.clear()
                self.batch.previous = None
            self.cutoff = cutoff
            self.batch.mask = np.ascontiguousarray(mask)
            rows, _ = self.batch.rows(poses, kin, False, cutoff)
            if with_jacobian:
                from .native_collision_rows import collision_jacobians
                for pose, current in zip(poses, rows):
                    active = [row for row in current if row['clearance'] < target]
                    if not active:
                        continue
                    self.query.data.qpos[:] = pose
                    mujoco.mj_kinematics(self.query.model, self.query.data)
                    mujoco.mj_comPos(self.query.model, self.query.data)
                    for row, jac in zip(active, collision_jacobians(self.query, active)):
                        row['_jacobian'] = jac
            return rows, self.batch.unknown_frames, int(mask.sum())
        # Missing optional native kernel retains the original mesh authority.
        result, failures = [], []
        query = self.query
        for frame, pose in enumerate(poses):
            rows = []
            query.data.qpos[:] = pose
            mujoco.mj_kinematics(query.model, query.data)
            if with_jacobian:
                mujoco.mj_comPos(query.model, query.data)
            try:
                for row in native_rows[frame]:
                    a, b = row['pair']
                    if not np.all(query.model.geom_type[[a, b]] == mujoco.mjtGeom.mjGEOM_MESH):
                        continue
                    phi, points, direction = query.mesh_distance.query(a, b, query.data)
                    if phi < cutoff:
                        rows.append({k: v for k, v in {**row, 'clearance': phi, 'points': points,
                            'direction': direction}.items() if k != '_original_required'})
                if with_jacobian:
                    for row in rows:
                        if row['clearance'] < target:
                            row['_jacobian'] = query.jacobian(row, query.data)
            except GeometryQueryError:
                failures.append(frame)
                rows = []
            result.append(rows)
        return result, failures, int(mask.sum())
