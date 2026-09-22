"""Shared robot collision-body distances; visual-only geoms are never queried."""

from __future__ import annotations

from itertools import combinations
from copy import copy
from collections import OrderedDict

import mujoco
import numpy as np

from .geometry_query import distance_direction, relative_point_jacobian
from .collision_motion import CollisionMotionBounds


SELF_COLLISION_DEFAULTS = {
    "detection_distance": 0.03,
    "minimum_distance": 0.002,
    "penetration_tolerance": 0.001,
    "slack_sigma": 0.002,
    "regression_tolerance": 1e-6,
    "motion_spacing": 0.005,
    "max_substeps": 256,
}


def self_collision_config(config=None):
    cfg = {**SELF_COLLISION_DEFAULTS, **dict(config or {})}
    for key in SELF_COLLISION_DEFAULTS:
        cfg[key] = float(cfg[key])
        if not np.isfinite(cfg[key]) or cfg[key] < 0:
            raise ValueError(f"self_collision.{key} must be finite and nonnegative")
    if cfg["slack_sigma"] <= 0 or cfg["detection_distance"] <= cfg["minimum_distance"]:
        raise ValueError("Self-collision requires positive sigma and detection > minimum distance")
    if cfg["motion_spacing"] <= 0 or cfg["max_substeps"] < 1 or int(cfg["max_substeps"]) != cfg["max_substeps"]:
        raise ValueError("Self-collision requires positive spacing and an integer substep budget")
    cfg["max_substeps"] = int(cfg["max_substeps"])
    return cfg


class SelfCollisionQuery:
    """Cache filtered pairs and query signed distances without changing the model.

    Both geoms must have collision bits and belong to robot bodies. Explicit
    MuJoCo pairs retain their precedence over dynamic pair filtering, but cannot
    opt visual-only geoms in. Ordinary pairs respect welds, parent filtering,
    contact excludes and the contype/conaffinity masks.
    """

    def __init__(self, model, config=None):
        self.model = model
        self.cfg = self_collision_config(config)
        self.data = mujoco.MjData(model)
        self.geom_ids = np.flatnonzero(
            ((model.geom_contype | model.geom_conaffinity) != 0)
            & (model.geom_bodyid != 0)
        )
        explicit = {
            tuple(sorted((int(a), int(b))))
            for a, b in zip(model.pair_geom1, model.pair_geom2)
        }
        excludes = set(map(int, model.exclude_signature))
        filter_parent = not (
            model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
        )
        pairs = []
        for a, b in combinations(map(int, self.geom_ids), 2):
            ba, bb = int(model.geom_bodyid[a]), int(model.geom_bodyid[b])
            wa, wb = int(model.body_weldid[ba]), int(model.body_weldid[bb])
            if (a, b) in explicit:
                pairs.append((a, b))
                continue
            if wa == wb:
                continue
            low, high = sorted((ba, bb))
            if (low << 16) + high in excludes:
                continue
            if filter_parent and (
                model.body_weldid[model.body_parentid[wa]] == wb
                or model.body_weldid[model.body_parentid[wb]] == wa
            ):
                continue
            if not (
                (model.geom_contype[a] & model.geom_conaffinity[b])
                or (model.geom_contype[b] & model.geom_conaffinity[a])
            ):
                continue
            pairs.append((a, b))
        self.pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
        self.pair_ids = self.pairs[:, 0] * model.ngeom + self.pairs[:, 1]
        self.pair_queries = 0
        self.motion_bounds = CollisionMotionBounds(model, self.geom_ids)
        self._segment_cache = OrderedDict()

    def query(self, q=None, *, data=None, distance=None):
        """Return nearby pairs with two witnesses and a first-body exit vector.

        Passing data reuses the caller's current FK. Errors and indeterminate
        normals are raised, never reported as collision-free configurations.
        """
        model = self.model
        if data is None:
            data = self.data
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
        threshold = self.cfg["detection_distance"] if distance is None else float(distance)
        a, b = self.pairs.T
        radii = model.geom_rbound[a] + model.geom_rbound[b]
        bounded = (model.geom_rbound[a] > 0) & (model.geom_rbound[b] > 0)
        delta = data.geom_xpos[a] - data.geom_xpos[b]
        nearby = ~bounded | (np.linalg.norm(delta, axis=1) <= radii + threshold)
        rows = []
        fromto = np.zeros(6)
        for index in np.flatnonzero(nearby):
            g1, g2 = map(int, self.pairs[index])
            fromto[:] = 0
            phi = float(mujoco.mj_geomDistance(model, data, g1, g2, threshold, fromto))
            self.pair_queries += 1
            if phi >= threshold:
                continue
            normal = None
            if np.linalg.norm(fromto[:3] - fromto[3:]) <= 1e-10:
                normal_data = data
                # At numerical zero, normal-margin collision detection may
                # omit the touching pair. Ask the collision engine with a
                # positive margin on an isolated model, never the user's model.
                if not any(set(map(int, c.geom)) == {g1, g2} for c in data.contact):
                    if not hasattr(self, "_normal_model"):
                        self._normal_model = copy(model)
                        self._normal_model.geom_margin[:] = np.maximum(model.geom_margin, 1e-5)
                        self._normal_model.pair_margin[:] = np.maximum(model.pair_margin, 1e-5)
                        self._normal_data = mujoco.MjData(self._normal_model)
                    normal_data = self._normal_data
                    normal_data.qpos[:] = data.qpos
                    mujoco.mj_forward(self._normal_model, normal_data)
                for contact in normal_data.contact:
                    if tuple(map(int, contact.geom)) == (g1, g2):
                        normal = contact.frame[:3]
                        break
                    if tuple(map(int, contact.geom)) == (g2, g1):
                        normal = -contact.frame[:3]
                        break
            direction = distance_direction(phi, fromto, normal)
            rows.append({
                "pair": [g1, g2], "pair_id": int(self.pair_ids[index]), "pair_index": int(index),
                "body_ids": [int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])],
                "clearance": phi, "points": fromto.reshape(2, 3).copy(),
                "direction": direction,
            })
        return rows

    def jacobian(self, row, data):
        return relative_point_jacobian(
            self.model, data, row["points"][0], row["body_ids"][0],
            row["direction"], row["points"][1], row["body_ids"][1],
        )

    def depths(self, q):
        return {row["pair_id"]: max(-row["clearance"], 0.0) for row in self.query(q)}

    def segment(self, first, second, *, jacobians=False):
        """Sample the playback path and retain each pair's worst witness.

        Every eligible pair is considered. Closest witnesses supply the active
        linearization, while nonlinear acceptance checks every sampled subpose.
        This is an adaptive discrete check, not continuous collision detection.
        """
        cache_key = (np.ascontiguousarray(first).tobytes(), np.ascontiguousarray(second).tobytes(), jacobians)
        if cache_key in self._segment_cache:
            self._segment_cache.move_to_end(cache_key)
            return self._segment_cache[cache_key]
        travel = self.motion_bounds.pair_bound(self.pairs, first, second)
        required = max(1, int(np.ceil(travel / self.cfg["motion_spacing"])))
        steps = min(required, self.cfg["max_substeps"])
        tangent = np.zeros(self.model.nv)
        mujoco.mj_differentiatePos(self.model, tangent, 1.0, first, second)
        worst, depths = {}, {}
        for part in range(steps + 1):
            alpha = part / steps
            q = first.copy()
            mujoco.mj_integratePos(self.model, q, tangent, alpha)
            for row in self.query(q):
                key = row["pair_id"]
                depths[key] = max(depths.get(key, 0.0), -row["clearance"])
                if alpha == 0:
                    continue  # The previous saved pose is fixed.
                if key not in worst or row["clearance"] < worst[key]["clearance"]:
                    row["alpha"] = alpha
                    if jacobians:
                        row["jacobian"] = self.jacobian(row, self.data) * alpha
                    worst[key] = row
        result = {"rows": list(worst.values()), "depths": depths,
                  "samples": steps + 1, "subdivision_limit": required > steps}
        self._segment_cache[cache_key] = result
        if len(self._segment_cache) > 256:
            self._segment_cache.popitem(last=False)
        return result

    def acceptable(self, before, after):
        tolerance = self.cfg["penetration_tolerance"]
        regression = self.cfg["regression_tolerance"]
        return all(
            depth <= max(tolerance, before.get(pair, 0.0)) + regression
            for pair, depth in after.items()
        )

    def describe(self):
        return {
            "status": "enabled" if len(self.pairs) else "no_eligible_pairs",
            "geometry": "collision_geoms_only", "geom_ids": self.geom_ids.tolist(),
            "pair_count": len(self.pairs), "config": self.cfg,
        }


def bounded_collision_constraints(query, rows, data, step, columns, transform=None, *, slack_variables=None):
    """Return feasible-at-zero collision halfspaces and bounded slack cost.

    Slack is limited to the current clearance deficit: a deep starting
    penetration may recover over several SQP steps, but the linear model may
    not trade a larger intersection for posture. A nonlinear check must follow.
    """
    import cvxpy as cp

    constraints, slacks = [], []
    for row in rows:
        jac = row["jacobian"] if "jacobian" in row else query.jacobian(row, data)
        if transform is not None:
            jac = jac @ transform
        phi = row["clearance"]
        target = query.cfg["minimum_distance"]
        deficit = max(target - phi, 0.0)
        if deficit > 0:
            slack = cp.Variable(nonneg=True)
            constraints.append(slack <= deficit)
            slacks.append(slack)
        else:
            slack = 0.0
        constraints.append(phi + jac[columns] @ step >= target - slack)
    penalty = (
        cp.sum_squares(cp.hstack(slacks) / query.cfg["slack_sigma"])
        if slacks else cp.Constant(0.0)
    )
    if slack_variables is not None:
        slack_variables.extend(slacks)
    return constraints, penalty


def summarize_self_collisions(query, trajectory, *, include_interframes=False):
    """Saved-frame measurement independent of scene availability and repair caps."""
    records, frame_depths = [], []
    for frame, q in enumerate(trajectory):
        rows = query.query(q)
        frame_depths.append(max((max(-r["clearance"], 0.0) for r in rows), default=0.0))
        for row in rows:
            if -row["clearance"] > query.cfg["penetration_tolerance"]:
                records.append({
                    "frame": frame, "depth": -row["clearance"],
                    "pair": row["pair"],
                    "bodies": [query.model.body(b).name for b in row["body_ids"]],
                })
    report = {
        **query.describe(), "method": "saved_frames",
        "max_depth": max(frame_depths, default=0.0),
        "frames_over_tolerance": [
            f for f, depth in enumerate(frame_depths)
            if depth > query.cfg["penetration_tolerance"]
        ],
        "worst": max(records, key=lambda row: row["depth"], default=None),
        "records": records,
    }
    if include_interframes:
        intervals, limited, maximum, worst = [], [], 0.0, None
        for frame, (first, second) in enumerate(zip(trajectory[:-1], trajectory[1:])):
            path = query.segment(first, second)
            if path["subdivision_limit"]:
                limited.append(frame)
            depth = max(path["depths"].values(), default=0.0)
            if depth > query.cfg["penetration_tolerance"]:
                intervals.append(frame)
            if depth > maximum:
                maximum = depth
                row = min(path["rows"], key=lambda r: r["clearance"], default=None)
                worst = {"interval": frame, "depth": depth}
                if row is not None:
                    worst.update(alpha=row["alpha"], pair=row["pair"],
                                 bodies=[query.model.body(b).name for b in row["body_ids"]])
        report.update(method="adaptive_discrete", interframe_max_depth=maximum,
                      interframe_worst=worst, intervals_over_tolerance=intervals,
                      subdivision_limit_intervals=limited,
                      acceptance_status=("incomplete" if limited else "residual"
                                         if intervals or report["frames_over_tolerance"] else "passed"))
    return report
