"""Contact, swing and joint-correction fidelity checks for trajectory changes.

A candidate is compared with the validator's immutable reference on saved
frames and on the same interpolated subposes used by the geometry scan.
TrajectoryValidator owns all state: specs, budgets, the immutable reference
and its cached effector FK.
"""

from __future__ import annotations

import mujoco
import numpy as np

from ._solver.targets import _world_target_regime
from .contact import _is_surface_only_contact_mode
from .contact_error_cache import reference_contact_errors


def _pose_velocity(model, first, second):
    """Match mjviser/downstream scalar interpolation of the saved qpos values.

    MuJoCo handles quaternion manifolds; scalar hinges retain their numeric
    displacement, including principal-chart seams and intentional full turns.
    A validator-only short-arc unwrap would inspect a different playback path.
    """
    velocity = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, velocity, 1, first, second)
    return velocity


class TrajectoryFidelityMixin:
    """Contact, swing and joint-correction checks of a TrajectoryValidator.

    Internal methods sharing the owning validator instance; no separate state.
    """

    def swing_budget(self, name, frame):
        task = self.specs[name]["task"]
        # Unknown clearance cannot justify spending the free-motion budget.
        if not np.isfinite(task.distances[frame]):
            return min(self.cfg["swing_position_budget"], self.cfg["contact_regression"])
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
            old = reference_contact_errors(self, task)
            new = task.errors(candidate, self.frame_start)
            if np.any(new > np.maximum(old, task.cfg["tolerance"]) + task.cfg["regression"] + 1e-8):
                reasons.append("self_contact_regression")
            metrics["self_contact"] = task.metrics(candidate, self.frame_start, errors=new)
        if self.surface_contact_tasks is not None:
            task = self.surface_contact_tasks
            old = reference_contact_errors(self, task)
            new = task.errors(candidate, self.frame_start)
            if np.any(new > np.maximum(old, task.cfg["tolerance"]) + task.cfg["regression"] + 1e-8):
                reasons.append("surface_contact_regression")
            metrics["surface_contact"] = task.metrics(candidate, self.frame_start, errors=new)
        if self.surface_tasks is not None:
            metrics["surface_normal_cost"] = self.surface_score(candidate)
        if self.body_relation_tasks is not None:
            metrics["body_relations"] = self.body_relation_tasks.metrics(candidate, self.frame_start)
        if self.surface_pose_tasks is not None:
            metrics["surface_pose"] = self.surface_pose_tasks.metrics(candidate, self.frame_start)
        if self.contact_phase is not None:
            metrics["contact_phase_priority"] = self.contact_phase.diagnostics
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
                if payload["contact"] and payload.get("surface_managed", False):
                    continue
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
                    # World references stay immutable. No reach-clamping or
                    # target transformation buys a larger repair allowance.
                    if change > self.swing_budget(name, f) + 1e-8:
                        reasons.append(f"swing_budget:{name}")
                    if (
                        _world_target_regime(spec, payload)
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
            if payload["contact"] and payload.get("surface_managed", False):
                continue
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
