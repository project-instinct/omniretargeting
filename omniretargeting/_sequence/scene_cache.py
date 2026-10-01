"""Batch identical static-scene queries across poses; never subsample geometry."""

import mujoco
import numpy as np


class SceneTrajectoryQueries:
    def __init__(self, model, query, shells, *, batch_points=1 << 18):
        self.data = mujoco.MjData(model)
        self.model, self.query, self.shells = model, query, shells
        self.batch_points = int(batch_points)  # memory bound, not a geometry tolerance
        if self.batch_points < 1:
            raise ValueError("Static query batch size must be positive")
        self.cached = {}

    def get(self, pose, frame):
        entry = self.cached.get(frame)
        key = np.ascontiguousarray(pose).tobytes()
        return entry[1] if entry is not None and entry[0] == key else None

    def prepare(self, poses):
        pending, points, count = [], [], 0

        def flush():
            if not pending:
                return
            # libigl rebuilds the winding hierarchy on every Python call.
            # One batch amortizes that work without changing any witness.
            measured = self.query.query(np.concatenate(points))
            begin = 0
            for frame, key, size in pending:
                self.cached[frame] = (
                    key,
                    {
                        name: value[begin : begin + size]
                        for name, value in measured.items()
                    },
                )
                begin += size
            pending.clear()
            points.clear()

        for frame, pose in enumerate(poses):
            if self.get(pose, frame) is not None:
                continue
            self.data.qpos[:] = pose
            mujoco.mj_forward(self.model, self.data)
            groups = [
                offsets @ self.data.geom_xmat[gid].reshape(3, 3).T
                + self.data.geom_xpos[gid]
                for gid, offsets in self.shells
            ]
            if not groups:
                continue
            current = np.concatenate(groups)
            if count + len(current) > self.batch_points:
                flush()
                count = 0
            pending.append((frame, np.ascontiguousarray(pose).tobytes(), len(current)))
            points.append(current)
            count += len(current)
        flush()
