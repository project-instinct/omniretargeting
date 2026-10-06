"""SDF-screened original queries, with the existing exact-pose worker caches."""
from collections import OrderedDict

import mujoco
import numpy as np

from .mesh_repair import OriginalGeometry
from .original_rigid_fallback import (WITNESS_CLEARANCE, enabled as rigid_fallback_enabled,
                                      resolve_unknown_static, rigid_robot_data)

BAND = .008  # OriginalGeometry's existing witness activation distance.


class JointGeometry(OriginalGeometry):
    def __init__(self, objective, objects, scene, reference):
        super().__init__(objective, objects, scene, manifold=True,
                         columns=objective.coords.cols, reference_poses=reference)
        self._gpu_allowed = False  # Screening owns the selected CPU geom set.
        self._screens = {}
        if self.moving is not None:
            from .joint_moving import JointMoving
            self.moving.__class__ = JointMoving
            self.moving.material_cache.clear()
            self._moving_query = self.moving.query
            self.moving.query = self._screened_moving

    def screen(self, pose, frame):
        cache = self._screens.setdefault(frame, OrderedDict())
        token = np.asarray(pose).tobytes()
        if token in cache:
            cache.move_to_end(token)
            return cache[token]
        obj, data = self.objective, self.data
        data.qpos[:] = pose
        mujoco.mj_forward(self.model, data)
        geoms = np.asarray(obj.shell_geoms)
        selected = []
        fields = obj._clearance_fields()
        expected = int(self.scene is not None) + (0 if self.moving is None else len(self.moving.objects))
        if len(fields) != expected:
            # No matching proxy is uncertainty, not a zero-depth observation.
            selected = [set(map(int, geoms)) for _ in range(expected)]
            cache[token] = selected
            while len(cache) > 3:
                cache.popitem(last=False)
            return selected
        for _, sdf, rotation, translation, scales in fields:
            centres = data.geom_xpos[geoms].copy()
            if rotation is not None:
                centres = (centres - translation[frame]) @ rotation[frame]
            scale = 1. if scales is None else scales[frame]
            phi = sdf.query(centres / scale, gradient=False) * scale
            # Unknown proxy regions remain active; they never certify clearance.
            near = ~np.isfinite(phi) | (phi < obj.shell_radius + BAND)
            active = set()
            for column in np.flatnonzero(near):
                gid = geoms[column]
                ids = obj.shell_index[column]
                ids = ids[ids >= 0]
                points = obj.shell_local[ids] @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
                if rotation is not None:
                    points = (points - translation[frame]) @ rotation[frame]
                value = sdf.query(points / scale, gradient=False) * scale
                if np.any(~np.isfinite(value) | (value < BAND)):
                    active.add(int(gid))
            selected.append(active)
        cache[token] = selected
        while len(cache) > 3:
            cache.popitem(last=False)
        return selected

    def _static_frames(self, poses, frames=None, *, _gpu_ready=False):
        """Batch changed geoms; reuse exact-transform witnesses, never Jacobians."""
        if not _gpu_ready and getattr(self, "_gpu_pending_static", None) is not None:
            from .gpu_geometry import lazy_static_frames
            yield from lazy_static_frames(self, poses, frames)
            return
        objective, data = self.objective, self.data
        ids = objective.shell_sample_geom
        groups = [(int(geom), np.flatnonzero(ids == geom)) for geom in objective.shell_geoms]
        limit = max(1, int(self.SCENE_BATCH_POINTS))
        summaries_by_frame, updates, pending, count = [], [], [], 0
        for frame in range(len(poses)) if frames is None else frames:
            pose = poses[frame]
            active = self.screen(pose, frame)[0]
            data.qpos[:] = pose
            mujoco.mj_forward(self.model, data)
            summaries = []
            for geom, selected in groups:
                if geom not in active:
                    continue
                # Byte comparison intentionally distinguishes even signed zero;
                # no coordinate rounding or approximate cache acceptance.
                signature = (data.geom_xpos[geom].tobytes(), data.geom_xmat[geom].tobytes())
                cached = self._static_cache.get((frame, geom))
                if cached is not None and cached[0] == signature:
                    summary = cached[1]
                    if summary["unknown"]:
                        updates.append(((frame, geom), signature, summary,
                                        data.geom_xpos[geom].copy(), data.geom_xmat[geom].copy(),
                                        rigid_robot_data(data)))
                elif cached is not None and self._static_bound_reuse and self._clear_after_motion(cached, geom):
                    # The cache keeps the exactly queried transform and bound.
                    summary = cached[1]
                    self.static_reused += 1
                else:
                    summary = {"row": None, "unknown": False, "clearance": np.inf,
                               "rows": [], "origin": data.geom_xpos[geom].copy(), "_screen_band": .02}
                    if (getattr(self, "_fork_queries_prepared", False)
                            or getattr(self, "_repair_static_bounds", False)):
                        from .original_geometry_cpu import screened_static_samples
                        from .original_geometry_bounds import far_geom
                        selected = (np.empty(0, int) if far_geom(self, geom, summary) else
                                    screened_static_samples(self, geom, selected, cached, summary))
                    updates.append(((frame, geom), signature, summary,
                                    data.geom_xpos[geom].copy(), data.geom_xmat[geom].copy(),
                                        rigid_robot_data(data)))
                    first = 0
                    while first < len(selected):
                        indices = selected[first:first + limit - count]
                        points = data.geom_xpos[ids[indices]] + np.einsum(
                            "nij,nj->ni", data.geom_xmat[ids[indices]].reshape(-1, 3, 3),
                            objective.shell_local[indices])
                        first += len(indices)
                        if (getattr(self, "_fork_queries_prepared", False)
                                or getattr(self, "_repair_static_bounds", False)):
                            from .original_geometry_bounds import screen_points
                            points, indices = screen_points(self, summary, points, indices)
                        if len(indices):
                            pending.append((summary, points, indices))
                        count += len(indices)
                        if count == limit:
                            self._query_static_batch(pending)
                            pending, count = [], 0
                summaries.append((geom, summary))
            summaries_by_frame.append(summaries)
        if pending:
            self._query_static_batch(pending)
        # Never commit a partially queried summary if SceneQuery raises.
        for key, signature, summary, position, rotation, robot_data in updates:
            if summary["unknown"] and rigid_fallback_enabled(self.scene):
                resolve_unknown_static(self, key[0], key[1], summary, signature,
                                       robot_data, WITNESS_CLEARANCE)
            if summary.pop("_screen_sort", False):
                order = np.argsort(summary["_screen_ids"], kind="stable")
                summary["_screen_ids"] = summary["_screen_ids"][order]
                summary["_screen_phi"] = summary["_screen_phi"][order]
            summary.pop("_reuse_sign_ids", None)
            summary.pop("_reuse_sign_phi", None)
            self._static_cache[key] = (signature, summary, position, rotation)
        self._static_cache = {key: value for key, value in self._static_cache.items()
                              if key[0] < len(poses)}
        yield from summaries_by_frame

    def _screened_moving(self, pose, frame, dofs=None):
        offset = int(self.scene is not None)
        active = set().union(*self.screen(pose, int(frame))[offset:])
        # Mandatory frozen material observations are never screened away.
        required = self._moving_evidence_baseline.get(int(frame), ())
        active.update(int(item[1]) for item in required)
        shells = self.moving.shells
        try:
            self.moving.shells = [s for s in shells if s['geom_id'] in active]
            return self._moving_query(pose, frame, dofs)
        finally:
            self.moving.shells = shells
