"""Protect known material samples while unknown scan regions stay unresolved."""

from collections import OrderedDict

import mujoco
import numpy as np


def preserve_known_scene_samples(
    validator, candidate, baseline_stats, start=0, end=None
):
    """Share material-point validation across repair and postprocessing paths."""
    if not (
        baseline_stats["uncertain_frames"] or baseline_stats["uncertain_intervals"]
    ):
        return []
    guard = getattr(validator, "_scene_witness_guard", None)
    if guard is None:
        guard = validator._scene_witness_guard = SceneWitnessGuard(validator)
    return guard.check(candidate, start, len(candidate) - 1 if end is None else end)


class SceneWitnessGuard:
    """Recheck the same robot material points on a common temporal grid.

    A per-geom unknown flag alone cannot detect a formerly known sample becoming
    unknown when that geom already has other unknown samples. This guard checks
    individual, fixed collision-shell samples before accepting such repairs.
    """

    def __init__(self, validator):
        self.v = validator
        self.cache = OrderedDict()
        self.reference_data = mujoco.MjData(validator.model)
        self.candidate_data = mujoco.MjData(validator.model)
        self.report = {
            "checked_poses": 0,
            "checked_material_points": 0,
            "policy": "preserve_known_samples_keep_unknown_unresolved",
        }

    def _query(self, points, component):
        from .trajectory_validation import QueryBudgetExceeded

        limit = self.v.cfg["max_query_points"]
        if (
            limit is not None
            and self.v.query.query_points - self.v.query_start + len(points) > limit
        ):
            raise QueryBudgetExceeded("known scene witness budget")
        return self.v.query.query(points, component_ids=[component])

    def _pose(self, original, candidate):
        if np.array_equal(original, candidate):
            return []
        v, a, b = self.v, self.reference_data, self.candidate_data
        for data, q in ((a, original), (b, candidate)):
            data.qpos[:] = q
            mujoco.mj_forward(v.model, data)
        key = np.ascontiguousarray(original).tobytes()
        if key not in self.cache:
            world, owners = v.probe.world_points(a)
            witnesses = []
            for component in range(len(v.query.components)):
                query = self._query(world, component)
                known = ~query["uncertain"] & np.isfinite(query["clearance"])
                if np.any(known):
                    witnesses.append(
                        (
                            component,
                            owners[known],
                            query["clearance"][known].copy(),
                            world[known].copy(),
                        )
                    )
            self.cache[key] = witnesses
            if len(self.cache) > 32:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        reasons = []
        for component, owners, old_clearance, old_world in self.cache[key]:
            reasons.extend(
                self._component(a, b, component, owners, old_clearance, old_world)
            )
        return reasons

    def _component(self, a, b, component, owners, old_clearance, old_world):
        v = self.v
        gids = v.probe.geom_ids[owners]
        moved = (
            np.linalg.norm(a.geom_xpos[gids] - b.geom_xpos[gids], axis=1) > 1e-12
        ) | (np.linalg.norm(a.geom_xmat[gids] - b.geom_xmat[gids], axis=1) > 1e-12)
        if not np.any(moved):
            return []
        gids = gids[moved]
        local = np.einsum(
            "ni,nij->nj",
            old_world[moved] - a.geom_xpos[gids],
            a.geom_xmat[gids].reshape(-1, 3, 3),
        )
        new_world = (
            np.einsum("ni,nji->nj", local, b.geom_xmat[gids].reshape(-1, 3, 3))
            + b.geom_xpos[gids]
        )
        after = self._query(new_world, component)
        self.report["checked_poses"] += 1
        self.report["checked_material_points"] += len(new_world)
        reasons = []
        if np.any(after["uncertain"] | ~np.isfinite(after["clearance"])):
            reasons.append("known_scene_sample_became_uncertain")
        old_depth = np.maximum(-old_clearance[moved], 0)
        bound = np.where(
            old_depth <= v.cfg["penetration_tolerance"],
            v.cfg["penetration_tolerance"],
            old_depth + v.cfg["collision_regression"],
        )
        if np.any(-after["clearance"] > bound + 1e-8):
            reasons.append("known_scene_sample_regression")
        return reasons

    def check(self, candidate, start, end):
        from .trajectory_validation import _pose_velocity

        v = self.v
        reasons = []
        for frame in range(start, end + 1):
            reasons.extend(self._pose(v.reference[frame], candidate[frame]))
            if reasons:
                return sorted(set(reasons))
        for frame in range(start, end):
            bound = max(
                float(
                    np.max(
                        v.motion_bounds.displacements(q[frame], q[frame + 1]), initial=0
                    )
                )
                for q in (v.reference, candidate)
            )
            steps = max(1, int(np.ceil(bound / v.cfg["motion_spacing"])))
            if steps > v.cfg["max_substeps"]:
                return ["known_scene_witness_subdivision_budget"]
            velocities = [
                _pose_velocity(v.model, q[frame], q[frame + 1])
                for q in (v.reference, candidate)
            ]
            for index in range(1, steps):
                poses = [q[frame].copy() for q in (v.reference, candidate)]
                for pose, velocity in zip(poses, velocities):
                    mujoco.mj_integratePos(v.model, pose, velocity, index / steps)
                reasons.extend(self._pose(*poses))
                if reasons:
                    return sorted(set(reasons))
        return []
