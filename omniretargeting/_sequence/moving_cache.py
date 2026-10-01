"""Batch the identical moving-mesh winding queries over an entire trajectory."""

import mujoco
import numpy as np


class MovingForwardQueries:
    def __init__(self, query, *, batch_points=1 << 18):
        self.query = query
        self.batch_points = int(batch_points)  # Memory bound, not a geometry tolerance.
        if self.batch_points < 1:
            raise ValueError("Moving query batch size must be positive")
        self.cached = {}
        self.data = mujoco.MjData(query.model)

    def get(self, pose, frame, object_id):
        entry = self.cached.get(frame)
        key = np.ascontiguousarray(pose).tobytes()
        return entry[1][object_id] if entry is not None and entry[0] == key else None

    def prepare(self, poses):
        from ..moving_scene import interpolate_object

        query = self.query
        cached = dict(self.cached)
        pending = []
        for frame, pose in enumerate(poses):
            key = np.ascontiguousarray(pose).tobytes()
            if frame in self.cached and self.cached[frame][0] == key:
                continue
            pending.append(frame)
            self.data.qpos[:] = pose
            mujoco.mj_forward(query.model, self.data)
            worlds = {
                shell["geom_id"]: (
                    shell["offsets"]
                    @ self.data.geom_xmat[shell["geom_id"]].reshape(3, 3).T
                    + self.data.geom_xpos[shell["geom_id"]]
                )
                for shell in query.shells
            }
            groups = []
            for obj in query.objects:
                translation, rotation, scale = interpolate_object(obj["surface"], frame)
                groups.append(
                    {
                        gid: (world, (world - translation) @ rotation / scale, {})
                        for gid, world in worlds.items()
                    }
                )
            cached[frame] = (key, groups)

        for oid, obj in enumerate(query.objects):
            for cid in obj["closed"]:
                bounds = obj["query"].components[cid][0].bounds
                groups, points, count = [], [], 0

                def flush():
                    if not groups:
                        return
                    measured = obj["query"].query(
                        np.concatenate(points), component_ids=[cid]
                    )
                    begin = 0
                    for frame, gid, ids in groups:
                        end = begin + len(ids)
                        cached[frame][1][oid][gid][2][cid] = (
                            ids,
                            {
                                name: values[begin:end]
                                for name, values in measured.items()
                            },
                        )
                        begin = end
                    groups.clear()
                    points.clear()

                for frame in pending:
                    band = (
                        query.cfg["activation_distance"] / obj["surface"].scales[frame]
                    )
                    for gid, (_, local, _) in cached[frame][1][oid].items():
                        gap = np.maximum(
                            np.maximum(bounds[0] - local, local - bounds[1]), 0
                        )
                        ids = np.flatnonzero(np.linalg.norm(gap, axis=1) <= band)
                        if not len(ids):
                            continue
                        if count + len(ids) > self.batch_points:
                            flush()
                            count = 0
                        groups.append((frame, gid, ids))
                        points.append(local[ids])
                        count += len(ids)
                flush()
        self.cached = cached
