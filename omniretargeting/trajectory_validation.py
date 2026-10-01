"""Shared FK, contact and geometry validation for trajectory changes.

The original robot trajectory is immutable reference. Only selected chain joints
inside a window are variables. Geometry and contact checks use real FK after
interpolation, each SQP line-search step, and the final assembly. No global blend,
root correction, retiming, clipping or filtering is performed here.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import copy
from dataclasses import fields, replace

import mujoco
import numpy as np

from .retargeting import (
    _apply_configured_joint_limits,
    _apply_joint_limit_safety_fraction,
)
from .scene_geometry import ScenePenetrationProbe, collision_geom_ids
from .collision_motion import CollisionMotionBounds
from .interval_geometry_cache import CachedInterval, interval_key
from .interframe_sampling import interval_samples, prepare_intervals
from .scene_probe_batch import QUERY_BATCH_POINTS
from .self_collision import SelfCollisionQuery, self_collision_config
from .trajectory_fidelity import (
    TrajectoryFidelityMixin,
    _pose_velocity as _pose_velocity,
)
from .trajectory_refinement import _lock_contact_offsets, _scalar_joint_coordinates

LOCAL_REPAIR_DEFAULTS = {
    "enabled": True,
    "max_window_seconds": 0.4,
    "max_extension_seconds": 0.1,
    # Locality of one secondary correction relative to the trajectory it
    # receives, not a posture limit against the source motion.
    "max_joint_correction": 0.2,
    "swing_position_budget": 0.02,
    "sample_spacing": 0.02,
    "motion_spacing": 0.01,
    "max_substeps": 64,
    "iterations": 2,
    "max_windows": 24,
    "max_query_points": 2000000,  # None permits complete offline point queries.
    "max_geometry_calls": 256,
    "full_geometry": True,
    "bidirectional_enabled": True,
    "directional_iterations": 3,
    "dp_max_evaluations": 512,
    "dp_polish_iterations": 2,
    "max_collision_slack": 0.1,
}


def local_repair_config(
    config,
    *,
    priority=None,
    joint_limit_safety_fraction=1.0,
    self_collision_enabled=False,
):
    from .config_validation import known_options

    known_options(config, LOCAL_REPAIR_DEFAULTS, "local_collision_repair", extra=('contact_tolerance', 'contact_regression', 'penetration_tolerance', 'collision_regression', 'joint_limit_safety_fraction', 'self_collision_enabled', 'self_collision', 'correction_velocity', 'correction_acceleration', 'support_slip_regression'))
    cfg = {**LOCAL_REPAIR_DEFAULTS, **dict(config or {})}
    priority = dict(priority or {})
    cfg.setdefault(
        "contact_tolerance", float(priority.get("contact_normal_tolerance", 0.005))
    )
    cfg.setdefault("contact_regression", 0.001)
    cfg.setdefault("penetration_tolerance", 0.001)
    cfg.setdefault("collision_regression", 0.001)
    cfg["joint_limit_safety_fraction"] = min(
        float(cfg.get("joint_limit_safety_fraction", joint_limit_safety_fraction)),
        float(joint_limit_safety_fraction),
    )
    cfg["self_collision_enabled"] = bool(
        self_collision_enabled or cfg.get("self_collision_enabled", False)
    )
    cfg["self_collision"] = self_collision_config(cfg.get("self_collision"))
    # Bound only added collision-repair corrections, not source motion rates.
    cfg.setdefault("correction_velocity", 5.4)
    cfg.setdefault("correction_acceleration", 72.0)
    cfg.setdefault("support_slip_regression", 0.02)  # m/s of added sticking slip
    for name in (
        "max_window_seconds",
        "max_extension_seconds",
        "max_joint_correction",
        "swing_position_budget",
        "sample_spacing",
        "motion_spacing",
        "contact_tolerance",
        "contact_regression",
        "penetration_tolerance",
        "collision_regression",
        "correction_velocity",
        "correction_acceleration",
        "max_collision_slack",
    ):
        if not np.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in (
        "max_substeps",
        "max_windows",
        "max_query_points",
        "max_geometry_calls",
        "directional_iterations",
        "dp_max_evaluations",
        "dp_polish_iterations",
    ):
        if name == "max_query_points" and cfg[name] is None:
            continue
        if int(cfg[name]) != cfg[name] or cfg[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if int(cfg["iterations"]) != cfg["iterations"] or cfg["iterations"] < 0:
        raise ValueError("iterations must be a nonnegative integer")
    return cfg


class QueryBudgetExceeded(RuntimeError):
    pass


def _runs(mask):
    changes = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1) - 1)


def _summary(depths, model):
    frame_depth = np.max(depths, axis=1, initial=0)
    worst = (
        np.unravel_index(np.argmax(depths), depths.shape)
        if depths.size and np.max(depths) > 0
        else (-1, -1)
    )
    return {
        "max_depth": float(np.max(frame_depth, initial=0)),
        "p95_depth": float(np.percentile(frame_depth, 95)) if len(frame_depth) else 0.0,
        "frames_penetrating": int(np.count_nonzero(frame_depth > 1e-6)),
        "worst_frame": int(worst[0]),
        "worst_body": model.body(int(model.geom_bodyid[worst[1]])).name
        if worst[1] >= 0
        else "",
        "per_geom_depth": depths,
        "per_frame_depth": frame_depth,
    }


class TrajectoryValidator(TrajectoryFidelityMixin):
    def __init__(self, model, reference, query, tasks, priority, config, framerate,
                 surface_tasks=None, *, geometry_cache=None, body_relation_tasks=None, self_contact_tasks=None, surface_contact_tasks=None, surface_pose_tasks=None, contact_offset_choice=None):
        self.model, self.reference, self.query = (
            model,
            np.asarray(reference).copy(),
            query,
        )
        self.cfg, self.fps, self.tasks = config, float(framerate), tasks
        self.surface_tasks, self.frame_start = surface_tasks, 0
        self.body_relation_tasks = body_relation_tasks
        self.surface_pose_tasks = surface_pose_tasks
        if surface_pose_tasks is not None and (
            surface_pose_tasks.model is not model or len(surface_pose_tasks.targets) != len(self.reference)
        ):
            raise ValueError("Surface pose must use the same model and original timeline")
        self.surface_contact_tasks = surface_contact_tasks
        from .contact_phase import phase_policy
        self.contact_phase = phase_policy(surface_contact_tasks, surface_pose_tasks, *tasks.values())
        if surface_contact_tasks is not None and (
            surface_contact_tasks.model is not model or len(surface_contact_tasks.targets) != len(self.reference)
        ):
            raise ValueError("Surface contact must use the same model and original timeline")
        self.self_contact_tasks = self_contact_tasks
        if self_contact_tasks is not None:
            if self_contact_tasks.model is not model or len(self_contact_tasks.targets) != len(self.reference):
                raise ValueError("Self contact must use the same model and original timeline")
        if body_relation_tasks is not None:
            if body_relation_tasks.model is not model or len(body_relation_tasks.targets) != len(self.reference):
                raise ValueError("Body relations must use the same model and original timeline")
        if surface_tasks is not None and surface_tasks.kinematics.model is not model:
            raise ValueError("Surface bindings belong to a different compiled robot model")
        if surface_tasks is not None and len(surface_tasks.targets) != len(self.reference):
            raise ValueError("Surface tasks and trajectory must have identical frame counts")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("framerate must be positive and finite")
        self.qi, self.di, lo, hi = _scalar_joint_coordinates(model)
        lo, hi = _apply_joint_limit_safety_fraction(
            model, self.qi, lo, hi, config["joint_limit_safety_fraction"]
        )
        self.lower, self.upper = _apply_configured_joint_limits(
            model, self.qi, lo, hi, priority.get("joint_posture_limits")
        )
        self.probe = ScenePenetrationProbe(
            model,
            query.mesh,
            collision_geom_ids(model),
            0.05,
            query=query,
            spacing=config["sample_spacing"],
        )
        self.query_start = query.query_points
        self.geometry_calls = 0
        self.self_query = (
            SelfCollisionQuery(model, config.get("self_collision"))
            if config["self_collision_enabled"] else None
        )
        # Reuse exact-pose geometry only while model and scene remain unchanged.
        # Keep their objects alive and isolate sampling/self-collision policies.
        # Contact residuals and frame-specific modes are re-evaluated below.
        self._pose_cache = {} if geometry_cache is None else geometry_cache.setdefault(
            (model, query, tuple(self.probe.geom_ids), self.probe.margin,
             config["sample_spacing"], config["penetration_tolerance"],
             config["self_collision_enabled"],
             tuple(sorted(self_collision_config(config.get("self_collision")).items()))), {}
        )
        # Local repairs change only a few poses. Reuse the adaptive interframe
        # scan for every unchanged pair, including self geometry and contact
        # checks. Segment views share this cache through their shallow copy.
        self._interval_cache = {}
        self.specs = {}
        for name, task in tasks.items():
            raw = dict(priority.get("effectors", {}).get(name, {}))
            bid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, str(raw.get("robot_body", ""))
            )
            if bid < 0:
                raise ValueError(
                    f"Cannot validate contact task {name}: missing robot_body"
                )
            tracking = np.asarray(raw.get("tracking_offset", [0, 0, 0]), dtype=float)
            self.specs[name] = dict(
                raw,
                task=task,
                body_id=bid,
                tracking_offset=tracking,
                contact_offsets=np.asarray(
                    raw.get("contact_offsets", [tracking]), dtype=float
                ).reshape(-1, 3),
            )
        self.offset_choice = (_lock_contact_offsets(model, self.reference, self.specs)
                              if contact_offset_choice is None else
                              {name: np.asarray(choice, dtype=int).copy() for name, choice in contact_offset_choice.items()})
        for name, spec in self.specs.items():
            choice = self.offset_choice.get(name)
            if choice is None or choice.shape != (len(self.reference),) or np.any(choice < 0) or np.any(choice >= len(spec["contact_offsets"])):
                raise ValueError(f"Invalid fixed contact marker choices for {name}")
        self.original_effectors = self.effectors(self.reference)
        self.fixed_qi = np.setdiff1d(np.arange(model.nq), self.qi)
        self.motion_bounds = CollisionMotionBounds(model, self.probe.geom_ids)
        self.joint_ids = self.motion_bounds.joint_ids

    @contextmanager
    def segment(self, start, stop):
        """Evaluate original-time slices without rerunning contact hysteresis
        or selecting new robot marker offsets. Share FK/query caches and budgets.
        """
        view = copy(self)
        # A witness guard owns its validator's reference and timeline. A
        # shallow-copied parent guard would check the wrong source frames.
        view._scene_witness_guard = None
        view.frame_start = self.frame_start + start
        view.reference = self.reference[start:stop]
        view.specs = {}
        for name, spec in self.specs.items():
            task = spec["task"]
            arrays = {field.name: getattr(task, field.name)[start:stop]
                      for field in fields(task) if isinstance(getattr(task, field.name), np.ndarray)}
            if task.approach_reference is not None:
                arrays["approach_reference"] = tuple(
                    value[start:stop] for value in task.approach_reference
                )
            view.specs[name] = dict(spec, task=replace(task, **arrays))
        view.tasks = {name: spec["task"] for name, spec in view.specs.items()}
        view.offset_choice = {name: choice[start:stop] for name, choice in self.offset_choice.items()}
        view.original_effectors = {name: {key: array[start:stop] for key, array in values.items()}
                                   for name, values in self.original_effectors.items()}
        try:
            yield view
        finally:
            self.geometry_calls = view.geometry_calls

    def surface_rows(self, pose, frame, dofs):
        if self.surface_tasks is None:
            return ()
        excluded = [spec["body_id"] for spec in self.specs.values() if spec.get("track_orientation", False)]
        return self.surface_tasks.rows(pose, self.frame_start + frame, dofs, excluded)

    def surface_score(self, poses):
        return sum(weight * float(residual @ residual)
                   for frame, pose in enumerate(poses)
                   for weight, residual, _ in self.surface_rows(pose, frame, self.di))

    def relation_rows(self, pose, frame, dofs):
        if self.surface_pose_tasks is not None:
            yield from self.surface_pose_tasks.rows(pose, self.frame_start + frame, dofs)
        if self.body_relation_tasks is not None:
            yield from self.body_relation_tasks.rows(pose, self.frame_start + frame, dofs)
        if self.self_contact_tasks is not None:
            yield from self.self_contact_tasks.rows(pose, self.frame_start + frame, dofs)
        if self.surface_contact_tasks is not None:
            yield from self.surface_contact_tasks.rows(pose, self.frame_start + frame, dofs)

    def relation_score(self, poses):
        return sum(weight * float(residual @ residual)
                   for frame, pose in enumerate(poses)
                   for weight, residual, _ in self.relation_rows(pose, frame, self.di))

    def phase(self, frame, joints=None):
        phases = []
        if self.contact_phase is not None:
            policy, f = self.contact_phase, self.frame_start + frame
            phases.append(("contact_roles", tuple(np.flatnonzero(policy.support[f])),
                           tuple(policy.role(n, f) for n in policy.roles)))
            phases.extend(("surface_role", role, tuple(np.flatnonzero(mask[f])))
                          for role, mask in policy.surface_roles.items()
                          if role != "support")
        for name, spec in self.specs.items():
            if joints is not None and not set(joints) & set(
                self.ancestor_coordinates(spec["body_id"])
            ):
                continue
            task = spec["task"]
            phases.append(
                (
                    name,
                    task.frame(frame)["contact_mode"],
                    bool(task.contact_mask[frame]),
                    task.frame(frame)["environment_id"],
                )
            )
        return tuple(phases)

    def ancestor_coordinates(self, bid):
        coords = []
        while bid > 0:
            for jid in range(
                int(self.model.body_jntadr[bid]),
                int(self.model.body_jntadr[bid] + self.model.body_jntnum[bid]),
            ):
                if jid >= 0 and self.model.jnt_qposadr[jid] in self.qi:
                    coords.append(int(self.model.jnt_qposadr[jid]))
            bid = int(self.model.body_parentid[bid])
        return coords

    def chain(self, bid):
        """Stop at a branching torso; add a downstream effector's joints."""
        coords, bodies = set(), set()
        current = int(bid)
        while current > 0:
            bodies.add(current)
            for jid in range(
                int(self.model.body_jntadr[current]),
                int(self.model.body_jntadr[current] + self.model.body_jntnum[current]),
            ):
                if jid >= 0 and self.model.jnt_qposadr[jid] in self.qi:
                    coords.add(int(self.model.jnt_qposadr[jid]))
            parent = int(self.model.body_parentid[current])
            if (
                parent == 0
                or np.count_nonzero(self.model.body_parentid[1:] == parent) > 1
            ):
                break
            current = parent
        for spec in self.specs.values():
            path, current = [], spec["body_id"]
            while current > 0 and current not in bodies:
                path.append(current)
                current = int(self.model.body_parentid[current])
            if current in bodies:
                for child in path:
                    for jid in range(
                        int(self.model.body_jntadr[child]),
                        int(
                            self.model.body_jntadr[child]
                            + self.model.body_jntnum[child]
                        ),
                    ):
                        if jid >= 0 and self.model.jnt_qposadr[jid] in self.qi:
                            coords.add(int(self.model.jnt_qposadr[jid]))
        return tuple(sorted(coords))

    def point(self, data, name, frame):
        spec, task = self.specs[name], self.specs[name]["task"]
        offset = (
            spec["contact_offsets"][self.offset_choice[name][frame]]
            if task.contact_mask[frame]
            else spec["tracking_offset"]
        )
        return (
            data.xpos[spec["body_id"]]
            + data.xmat[spec["body_id"]].reshape(3, 3) @ offset
        )

    def effectors(self, trajectory):
        values = {
            name: {"position": [], "up": [], "forward": []} for name in self.specs
        }
        data = mujoco.MjData(self.model)
        for f, q in enumerate(trajectory):
            data.qpos[:] = q
            mujoco.mj_forward(self.model, data)
            for name, spec in self.specs.items():
                rot = data.xmat[spec["body_id"]].reshape(3, 3)
                values[name]["position"].append(self.point(data, name, f))
                values[name]["up"].append(
                    rot @ spec.get("orientation_local_up", [0, 0, 1])
                )
                values[name]["forward"].append(
                    rot @ spec.get("orientation_local_forward", [1, 0, 0])
                )
        return {
            name: {k: np.asarray(v) for k, v in val.items()}
            for name, val in values.items()
        }

    def geometry(self, trajectory):
        if self.geometry_calls >= self.cfg["max_geometry_calls"]:
            raise QueryBudgetExceeded("geometry call budget")
        self.geometry_calls += 1
        n = len(trajectory)
        depths = np.zeros((n, self.model.ngeom))
        between = np.zeros((max(n - 1, 0), self.model.ngeom))
        reliable = np.zeros_like(depths)
        between_reliable = np.zeros_like(between)
        unknown_geoms = np.zeros_like(depths, dtype=bool)
        between_unknown_geoms = np.zeros_like(between, dtype=bool)
        unknown = np.zeros(n, dtype=bool)
        between_unknown = np.zeros(max(n - 1, 0), dtype=bool)
        pair_count = len(self.self_query.pairs) if self.self_query is not None else 1
        self_depth = np.zeros((n, pair_count))
        between_self = np.zeros((max(n - 1, 0), pair_count))
        samples, unresolved = 0, []
        interframe_contact_failures, interframe_contact_change = [], 0.0
        records = []
        data = mujoco.MjData(self.model)
        prepared_frames = {}

        def prepare_samples(poses):
            # Explicit work budgets retain sequential accounting. Unlimited
            # offline validation batches the identical original-mesh queries.
            if self.cfg["max_query_points"] is not None:
                return
            from .scene_probe_batch import batched_probe
            pending = {}
            for pose in poses:
                key = np.ascontiguousarray(pose).tobytes()
                if key not in self._pose_cache and key not in prepared_frames:
                    pending[key] = pose
            if pending:
                frames = batched_probe(self.probe, list(pending.values()))
                prepared_frames.update(zip(pending, frames))

        def sample(q, f, alpha=0.0):
            key = np.ascontiguousarray(q).tobytes()
            cached = self._pose_cache.get(key)
            if cached is not None:
                values, uncertain, sd, saved_records, trusted, unknown_ids = cached
                records.extend(
                    dict(
                        record,
                        frame=int(f),
                        alpha=float(alpha),
                        contact_modes=list(self.phase(f)),
                    )
                    for record in saved_records
                )
                return values, uncertain, sd, trusted, unknown_ids
            record_start = len(records)
            # Bound expensive queries before invoking the shared backend.
            estimated = self.probe.query_work_bound
            point_limit = self.cfg["max_query_points"]
            if (
                point_limit is not None
                and self.query.query_points - self.query_start + estimated
                > self.cfg["max_query_points"]
            ):
                raise QueryBudgetExceeded("scene query point budget")
            frame = prepared_frames.pop(key, None)
            if frame is None:
                frame = self.probe.probe_trajectory(q[None])[0]
            if point_limit is not None and self.query.query_points - self.query_start > point_limit:
                raise QueryBudgetExceeded("scene query point/pair budget")
            values = np.zeros(self.model.ngeom)
            trusted = np.zeros(self.model.ngeom)
            unknown_ids = np.zeros(self.model.ngeom, dtype=bool)
            for gid in np.unique(frame["geom_ids"]):
                mask = frame["geom_ids"] == gid
                unknown_ids[gid] = np.any(frame["uncertain"][mask])
                valid = mask & ~frame["uncertain"] & np.isfinite(frame["clearance"])
                trusted[gid] = np.max(-frame["clearance"][valid], initial=0)
                indices = np.flatnonzero(
                    (frame["geom_ids"] == gid) & np.isfinite(frame["clearance"])
                )
                if not len(indices):
                    continue
                k = indices[np.argmin(frame["clearance"][indices])]
                values[gid] = max(-float(frame["clearance"][k]), 0)
                if values[gid] > self.cfg["penetration_tolerance"]:
                    bid = int(frame["body_ids"][k])
                    records.append(
                        {
                            "frame": int(f),
                            "alpha": float(alpha),
                            "body_id": bid,
                            "body": self.model.body(bid).name,
                            "geom": int(gid),
                            "point": frame["points"][k].tolist(),
                            "depth": values[gid],
                            "clearance": float(frame["clearance"][k]),
                            "direction": frame["directions"][k].tolist(),
                            "confidence": float(frame["confidence"][k]),
                            "uncertain": bool(frame["uncertain"][k]),
                            "surface_id": int(frame["surface_ids"][k]),
                            "obstacle_id": int(frame["obstacle_ids"][k]),
                            "contact_modes": list(self.phase(f)),
                        }
                    )
            sd = np.zeros(pair_count)
            if self.self_query is not None:
                data.qpos[:] = q
                mujoco.mj_forward(self.model, data)
                for collision in self.self_query.query(data=data):
                    depth = max(-collision["clearance"], 0.0)
                    sd[collision["pair_index"]] = depth
                    if depth > self.self_query.cfg["penetration_tolerance"]:
                        for side, gid in enumerate(collision["pair"]):
                            bid = int(self.model.geom_bodyid[gid])
                            records.append({
                                "frame": int(f), "alpha": float(alpha),
                                "body_id": bid, "body": self.model.body(bid).name,
                                "geom": int(gid), "point": collision["points"][side].tolist(),
                                "depth": depth, "clearance": collision["clearance"],
                                "direction": ((1 if side == 0 else -1) * collision["direction"]).tolist(),
                                "confidence": 1.0, "surface_id": -1, "obstacle_id": -2,
                                "pair": collision["pair"],
                                "pair_points": collision["points"].tolist(),
                                "pair_direction": collision["direction"].tolist(),
                                "contact_modes": list(self.phase(f)),
                            })
            uncertain = bool(np.any(frame["uncertain"]))
            self._pose_cache[key] = (
                values,
                uncertain,
                sd,
                records[record_start:].copy(),
                trusted,
                unknown_ids,
            )
            return values, uncertain, sd, trusted, unknown_ids

        batch_size = max(1, QUERY_BATCH_POINTS // max(1, self.probe.query_work_bound))
        for f, q in enumerate(trajectory):
            if f % batch_size == 0:
                prepare_samples(trajectory[f : f + batch_size])
            depths[f], unknown[f], self_depth[f], reliable[f], unknown_geoms[f] = sample(q, f)
        # Interval envelopes include endpoints: a stationary penetrating pose
        # cannot masquerade as a previously healthy interval just because no
        # interior subdivision was needed.
        between[:] = np.maximum(depths[:-1], depths[1:])
        between_self[:] = np.maximum(self_depth[:-1], self_depth[1:])
        between_reliable[:] = np.maximum(reliable[:-1], reliable[1:])
        between_unknown_geoms[:] = unknown_geoms[:-1] | unknown_geoms[1:]
        between_unknown[:] = unknown[:-1] | unknown[1:]
        for f in range(n - 1):
            if f % 16 == 0 and self.cfg["max_query_points"] is None:
                prepare_intervals(
                    self, trajectory, f, min(f + 16, n - 1), batch_size, prepare_samples
                )
            key = interval_key(self.frame_start + f, trajectory[f], trajectory[f + 1])
            cached_interval = self._interval_cache.get(key)
            if cached_interval is not None:
                was_unresolved, change, count = cached_interval.replay(
                    f, between, between_self, between_reliable,
                    between_unknown_geoms, between_unknown,
                    interframe_contact_failures, records,
                )
                if was_unresolved:
                    unresolved.append(f)
                interframe_contact_change = max(interframe_contact_change, change)
                samples += count
                continue
            record_start = len(records)
            failure_start = len(interframe_contact_failures)
            interval_change = 0.0
            required, steps, interior = interval_samples(
                self, trajectory[f], trajectory[f + 1]
            )
            if required > steps:
                unresolved.append(f)
            for index, q in enumerate(interior):
                if index % batch_size == 0:
                    prepare_samples(interior[index : index + batch_size])
                alpha = (index + 1) / steps
                values, uncertain, sd, trusted, unknown_ids = sample(q, f, alpha)
                failure, change = self.interframe_contact(trajectory, f, alpha, q)
                interframe_contact_change = max(interframe_contact_change, change)
                interval_change = max(interval_change, change)
                if failure:
                    interframe_contact_failures.append(
                        {"frame": f, "alpha": alpha, "reasons": failure}
                    )
                between[f] = np.maximum(between[f], values)
                between_unknown[f] |= uncertain
                between_self[f] = np.maximum(between_self[f], sd)
                between_reliable[f] = np.maximum(between_reliable[f], trusted)
                between_unknown_geoms[f] |= unknown_ids
                samples += 1
            self._interval_cache[key] = CachedInterval.capture(
                between[f], between_self[f], between_reliable[f],
                between_unknown_geoms[f], between_unknown[f], required > steps,
                interframe_contact_failures[failure_start:], interval_change,
                len(interior), records[record_start:],
            )
        stats = _summary(depths, self.model)
        stats.update(
            interframe_max_depth=float(np.max(between, initial=0)),
            interframe_depth=between,
            reliable_scene_depth=reliable,
            interframe_reliable_scene_depth=between_reliable,
            uncertain_geoms=unknown_geoms,
            interframe_uncertain_geoms=between_unknown_geoms,
            uncertain_frames=np.flatnonzero(unknown).tolist(),
            uncertain_intervals=np.flatnonzero(between_unknown).tolist(),
            subdivision_limit_intervals=unresolved,
            self_depth=self_depth,
            self_collision_status="enabled" if self.self_query is not None else "disabled",
            self_collision_pairs=self.self_query.pairs.tolist() if self.self_query is not None else [],
            self_collision_max_depth=float(np.max(self_depth, initial=0)) if self.self_query is not None else None,
            interframe_self_depth=between_self,
            records=records,
            interframe_contact_failures=interframe_contact_failures,
            interframe_contact_change=interframe_contact_change,
            interframe_samples=samples,
            method="adaptive_discrete",
            sample_spacing=self.cfg["sample_spacing"],
            motion_spacing=self.cfg["motion_spacing"],
        )
        return stats

    def compare_geometry(self, before, after, *, require_improvement=False,
                         focus_self=False, allow_existing_uncertainty=False):
        reasons = []
        cfg = self.cfg
        allow_existing_uncertainty |= focus_self
        scene_keys = (("reliable_scene_depth", "interframe_reliable_scene_depth")
                      if allow_existing_uncertainty else ("per_geom_depth", "interframe_depth"))
        for key in (*scene_keys, "self_depth", "interframe_self_depth"):
            tolerance = (cfg["self_collision"]["penetration_tolerance"]
                         if "self" in key else cfg["penetration_tolerance"])
            regression = (cfg["self_collision"]["regression_tolerance"]
                          if "self" in key else cfg["collision_regression"])
            old, new = before[key], after[key]
            limit = np.where(
                old <= tolerance,
                tolerance,
                old + regression,
            )
            if np.any(new > limit + 1e-8):
                reasons.append("collision_regression")
        if allow_existing_uncertainty:
            # Retained unknown scene sides remain unresolved in the report.
            # They cannot veto independent self repair, but a newly unknown
            # geom/interval cannot be used to hide a scene regression either.
            for key in ("uncertain_geoms", "interframe_uncertain_geoms"):
                if np.any(after[key] & ~before[key]):
                    reasons.append("new_uncertain_scene_side")
        elif after["uncertain_frames"] or after["uncertain_intervals"]:
            reasons.append("uncertain_scene_side")
        if after["subdivision_limit_intervals"]:
            reasons.append("subdivision_budget")
        if after["interframe_contact_failures"]:
            reasons.append("contact_interframe")
        if require_improvement:
            if focus_self:
                keys = ("self_depth", "interframe_self_depth")
                old = max(np.max(before[k], initial=0) for k in keys)
                new = max(np.max(after[k], initial=0) for k in keys)
                if new > cfg["self_collision"]["penetration_tolerance"] and new >= old - 1e-5:
                    reasons.append("collision_not_improved")
                return sorted(set(reasons))
            old = max(
                before["max_depth"],
                before["interframe_max_depth"],
                np.max(before["self_depth"], initial=0),
                np.max(before["interframe_self_depth"], initial=0),
            )
            new = max(
                after["max_depth"],
                after["interframe_max_depth"],
                np.max(after["self_depth"], initial=0),
                np.max(after["interframe_self_depth"], initial=0),
            )
            # An unresolved event must actually improve, irrespective of a 10 cm
            # global reporting target. Fully repaired events meet detection tolerance.
            if new > cfg["penetration_tolerance"] and new >= old - 1e-5:
                reasons.append("collision_not_improved")
        return sorted(set(reasons))
