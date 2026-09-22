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

from .contact import _is_surface_only_contact_mode
from .retargeting import (
    _apply_configured_joint_limits,
    _apply_joint_limit_safety_fraction,
)
from .scene_geometry import ScenePenetrationProbe, collision_geom_ids
from .collision_motion import CollisionMotionBounds
from .self_collision import SelfCollisionQuery, self_collision_config
from .trajectory_refinement import _lock_contact_offsets, _scalar_joint_coordinates

LOCAL_REPAIR_DEFAULTS = {
    "enabled": True,
    "max_window_seconds": 0.4,
    "max_extension_seconds": 0.1,
    "max_joint_correction": 0.2,
    "swing_position_budget": 0.02,
    "sample_spacing": 0.02,
    "motion_spacing": 0.01,
    "max_substeps": 64,
    "iterations": 8,
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


def _pose_velocity(model, first, second):
    """Match mjviser/downstream scalar interpolation of the saved qpos values.

    MuJoCo handles quaternion manifolds; scalar hinges retain their numeric
    displacement, including principal-chart seams and intentional full turns.
    A validator-only short-arc unwrap would inspect a different playback path.
    """
    velocity = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, velocity, 1, first, second)
    return velocity


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


class TrajectoryValidator:
    def __init__(self, model, reference, query, tasks, priority, config, framerate,
                 surface_tasks=None, *, geometry_cache=None, body_relation_tasks=None, self_contact_tasks=None, surface_contact_tasks=None, contact_offset_choice=None):
        self.model, self.reference, self.query = (
            model,
            np.asarray(reference).copy(),
            query,
        )
        self.cfg, self.fps, self.tasks = config, float(framerate), tasks
        self.surface_tasks, self.frame_start = surface_tasks, 0
        self.body_relation_tasks = body_relation_tasks
        self.surface_contact_tasks = surface_contact_tasks
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
             config["sample_spacing"], config["self_collision_enabled"],
             tuple(sorted(self_collision_config(config.get("self_collision")).items()))), {}
        )
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

    def swing_budget(self, name, frame):
        task = self.specs[name]["task"]
        # A direct world-tracking task has infinite scene distance: no contact
        # classification was performed. Do not treat that as evidence of swing
        # and spend a stance foot's 2 cm budget. Keep its existing world target
        # within the tight endpoint allowance until a real contact plan exists.
        if "foot" in name and not np.isfinite(task.distances[frame]):
            return min(
                self.cfg["swing_position_budget"], self.cfg["contact_regression"]
            )
        return self.cfg["swing_position_budget"]

    def contact_change_budget(self, original_error):
        # Reached anchors stay put. An already inaccurate contact may move
        # toward its immutable target instead of freezing in the wrong pose.
        return self.cfg["contact_regression"] + max(
            original_error - self.cfg["contact_tolerance"], 0.0
        )

    def fidelity(self, candidate):
        reasons, metrics = [], {}
        cfg = self.cfg
        if not np.all(np.isfinite(candidate)):
            return {"ok": False, "reasons": ["nonfinite"], "metrics": {}}
        delta = candidate[:, self.qi] - self.reference[:, self.qi]
        if not np.array_equal(candidate[:, self.fixed_qi], self.reference[:, self.fixed_qi]):
            reasons.append("floating_base")
        joint = candidate[:, self.qi]
        if np.any(joint < self.lower - 1e-9) or np.any(joint > self.upper + 1e-9):
            reasons.append("joint_limits")
        magnitude = float(np.max(np.abs(delta), initial=0))
        velocity = float(np.max(np.abs(np.diff(delta, axis=0)), initial=0) * self.fps)
        acceleration = float(
            np.max(np.abs(np.diff(delta, n=2, axis=0)), initial=0) * self.fps**2
        )
        if magnitude > cfg["max_joint_correction"] + 1e-8:
            reasons.append("joint_correction_budget")
        if (
            velocity > cfg["correction_velocity"] + 1e-8
            or acceleration > cfg["correction_acceleration"] + 1e-8
        ):
            reasons.append("boundary_continuity")
        metrics.update(
            max_joint_correction=magnitude,
            correction_velocity=velocity,
            correction_acceleration=acceleration,
            reference_acceleration=float(
                np.max(
                    np.abs(np.diff(self.reference[:, self.qi], n=2, axis=0)), initial=0
                )
                * self.fps**2
            ),
        )
        metrics["correction_jerk"] = float(
            np.max(np.abs(np.diff(delta, n=3, axis=0)), initial=0) * self.fps**3
        )
        if self.self_contact_tasks is not None:
            task = self.self_contact_tasks
            old = task.errors(self.reference, self.frame_start)
            new = task.errors(candidate, self.frame_start)
            if np.any(new > np.maximum(old, task.cfg["tolerance"]) + task.cfg["regression"] + 1e-8):
                reasons.append("self_contact_regression")
            metrics["self_contact"] = task.metrics(candidate, self.frame_start)
        if self.surface_contact_tasks is not None:
            task = self.surface_contact_tasks
            old = task.errors(self.reference, self.frame_start)
            new = task.errors(candidate, self.frame_start)
            if np.any(new > np.maximum(old, task.cfg["tolerance"]) + task.cfg["regression"] + 1e-8):
                reasons.append("surface_contact_regression")
            metrics["surface_contact"] = task.metrics(candidate, self.frame_start)
        if self.surface_tasks is not None:
            metrics["surface_normal_cost"] = self.surface_score(candidate)
        if self.body_relation_tasks is not None:
            metrics["body_relations"] = self.body_relation_tasks.metrics(candidate, self.frame_start)
        metrics["joints"] = {
            self.model.joint(int(jid)).name: {
                "unit": "m"
                if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE
                else "rad",
                "max_correction": float(np.max(np.abs(delta[:, col]), initial=0)),
            }
            for col, jid in enumerate(self.joint_ids)
        }
        current = self.effectors(candidate)
        for name, spec in self.specs.items():
            task = spec["task"]
            original = self.original_effectors[name]
            p, p0 = current[name]["position"], original["position"]
            contact_errors, swing_errors, changes = [], [], []
            slip = []
            for f in range(len(candidate)):
                payload = task.frame(f)
                mode = payload["contact_mode"]
                target = (
                    payload["contact_position"]
                    if payload["contact"]
                    else payload["tracking_position"]
                )
                projector = np.eye(3)
                if payload["contact"] and _is_surface_only_contact_mode(mode):
                    n = np.asarray(payload["surface_normal"], dtype=float)
                    n = n / max(np.linalg.norm(n), 1e-12)
                    projector = np.outer(n, n)
                error = float(np.linalg.norm(projector @ (p[f] - target)))
                old_error = float(np.linalg.norm(projector @ (p0[f] - target)))
                change = float(np.linalg.norm(projector @ (p[f] - p0[f])))
                if payload["contact"]:
                    contact_errors.append(error)
                    if (
                        error
                        > max(old_error, cfg["contact_tolerance"])
                        + cfg["contact_regression"]
                        + 1e-8
                        or change > self.contact_change_budget(old_error) + 1e-8
                    ):
                        reasons.append(f"contact:{name}")
                    if (
                        f
                        and mode == "sticking"
                        and task.frame(f - 1)["contact_mode"] == "sticking"
                        and payload["environment_id"] == task.frame(f - 1)["environment_id"]
                    ):
                        # Remove the authored target motion. A hand traveling
                        # with a carried object has zero tracking slip even
                        # though its world velocity is large.
                        target_step = target - task.frame(f - 1)["contact_position"]
                        v = (p[f] - p[f - 1] - target_step) * self.fps
                        v0 = (p0[f] - p0[f - 1] - target_step) * self.fps
                        slip.append(float(np.linalg.norm(v)))
                        if (
                            np.linalg.norm(v - v0)
                            > cfg["support_slip_regression"] + 1e-8
                        ):
                            reasons.append(f"support_slip:{name}")
                else:
                    swing_errors.append(
                        float(np.linalg.norm(p[f] - task.tracking_positions[f]))
                    )
                    changes.append(float(np.linalg.norm(p[f] - p0[f])))
                    # Foot world references stay immutable. No reach-clamping or
                    # target transformation buys a larger repair allowance.
                    if change > self.swing_budget(name, f) + 1e-8:
                        reasons.append(f"swing_budget:{name}")
                    if (
                        spec.get("preserve_world_target", "foot" in name)
                        and error > old_error + self.swing_budget(name, f) + 1e-8
                    ):
                        reasons.append(f"swing_reference:{name}")
                if payload["contact"] and spec.get("track_orientation", False):
                    axes = (
                        ["up"]
                        if _is_surface_only_contact_mode(mode)
                        else ["up", "forward"]
                    )
                    for axis in axes:
                        refdir = getattr(task, f"{axis}_directions")
                        if refdir is None:
                            continue
                        a, b, target_axis = (
                            current[name][axis][f],
                            original[axis][f],
                            refdir[f],
                        )
                        angle = lambda x, y: float(
                            np.arccos(
                                np.clip(
                                    np.dot(x, y)
                                    / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12),
                                    -1,
                                    1,
                                )
                            )
                        )
                        if (
                            angle(a, target_axis) > angle(b, target_axis) + 0.02
                            or angle(a, b) > 0.02
                        ):
                            reasons.append(f"contact_orientation:{name}")
            metrics[name] = {
                "contact_position_error": max(contact_errors, default=0.0),
                "support_slip": max(slip, default=0.0),
                "support_slip_definition": "world contact-target residual velocity, m/s; same environment only",
                "swing_world_error": max(swing_errors, default=0.0),
                "swing_world_deviation": max(changes, default=0.0),
                "horizontal_correction": float(
                    np.max(np.linalg.norm((p - p0)[:, :2], axis=1), initial=0)
                ),
                "vertical_correction": float(np.max(np.abs((p - p0)[:, 2]), initial=0)),
            }
        return {"ok": not reasons, "reasons": sorted(set(reasons)), "metrics": metrics}

    def interframe_contact(self, trajectory, frame, alpha, q):
        """Check contact/swing constraints on the same discrete subposes as geometry.

        A switch occurs at its saved timestamp; samples before it use the left
        mode. Same-mode sliding/tracking targets are linearly interpolated in
        world space for validation only; the stored tasks are never edited.
        """
        if not self.specs:
            return [], 0.0
        refq = self.reference[frame].copy()
        velocity = _pose_velocity(
            self.model, self.reference[frame], self.reference[frame + 1]
        )
        mujoco.mj_integratePos(self.model, refq, velocity, alpha)
        if np.array_equal(refq, q):
            return [], 0.0
        data, refdata = mujoco.MjData(self.model), mujoco.MjData(self.model)
        data.qpos[:] = q
        refdata.qpos[:] = refq
        mujoco.mj_forward(self.model, data)
        mujoco.mj_forward(self.model, refdata)
        reasons, maximum = [], 0.0
        for name, spec in self.specs.items():
            payload = spec["task"].frame(frame)
            next_payload = spec["task"].frame(frame + 1)
            p, p0 = self.point(data, name, frame), self.point(refdata, name, frame)
            target = np.array(
                payload["contact_position"]
                if payload["contact"]
                else payload["tracking_position"],
                copy=True,
            )
            mode = payload["contact_mode"]
            if mode == next_payload["contact_mode"] and mode != "sticking":
                next_target = (
                    next_payload["contact_position"]
                    if payload["contact"]
                    else next_payload["tracking_position"]
                )
                target = (1 - alpha) * target + alpha * next_target
            projector = np.eye(3)
            if payload["contact"] and _is_surface_only_contact_mode(mode):
                normal = np.asarray(payload["surface_normal"])
                normal = normal / max(np.linalg.norm(normal), 1e-12)
                projector = np.outer(normal, normal)
            change = float(np.linalg.norm(projector @ (p - p0)))
            maximum = max(maximum, change if payload["contact"] else 0.0)
            if payload["contact"]:
                bound = (
                    max(
                        float(np.linalg.norm(projector @ (p0 - target))),
                        self.cfg["contact_tolerance"],
                    )
                    + self.cfg["contact_regression"]
                )
                if (
                    change
                    > self.contact_change_budget(
                        float(np.linalg.norm(projector @ (p0 - target)))
                    )
                    + 1e-8
                    or np.linalg.norm(projector @ (p - target)) > bound + 1e-8
                ):
                    reasons.append("contact:" + name)
                if spec.get("track_orientation", False):
                    rot, rot0 = (
                        data.xmat[spec["body_id"]].reshape(3, 3),
                        refdata.xmat[spec["body_id"]].reshape(3, 3),
                    )
                    axes = (
                        ["up"]
                        if _is_surface_only_contact_mode(mode)
                        else ["up", "forward"]
                    )
                    for axis in axes:
                        local = spec.get(
                            "orientation_local_" + axis,
                            [0, 0, 1] if axis == "up" else [1, 0, 0],
                        )
                        a, b = rot @ local, rot0 @ local
                        angle = np.arccos(
                            np.clip(
                                np.dot(a, b)
                                / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12),
                                -1,
                                1,
                            )
                        )
                        if angle > 0.02 + 1e-8:
                            reasons.append("contact_orientation:" + name)
            elif change > self.swing_budget(name, frame) + 1e-8:
                reasons.append("swing_budget:" + name)
        return sorted(set(reasons)), maximum

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

        for f, q in enumerate(trajectory):
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
            dq = _pose_velocity(self.model, trajectory[f], trajectory[f + 1])
            motion = self.motion_bounds.displacements(trajectory[f], trajectory[f + 1])
            bound = float(np.max(motion, initial=0))
            spacing = self.cfg["motion_spacing"]
            if self.self_query is not None:
                relative = self.self_query.motion_bounds.pair_bound(
                    self.self_query.pairs, trajectory[f], trajectory[f + 1])
                # A self distance changes with the motion of both bodies.
                bound = max(bound, relative * spacing / self.self_query.cfg["motion_spacing"])
            required = max(1, int(np.ceil(bound / self.cfg["motion_spacing"])))
            steps = min(required, int(self.cfg["max_substeps"]))
            if required > steps:
                unresolved.append(f)
            for j in range(1, steps):
                alpha = j / steps
                q = trajectory[f].copy()
                mujoco.mj_integratePos(self.model, q, dq, alpha)
                values, uncertain, sd, trusted, unknown_ids = sample(q, f, alpha)
                failure, change = self.interframe_contact(trajectory, f, alpha, q)
                interframe_contact_change = max(interframe_contact_change, change)
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
