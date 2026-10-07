"""Immutable attained-contact limits for secondary fast-backend corrections.

Freeze once after contact acquisition, before clearance optimization. Targets
are the original per-frame world positions, including moving material targets;
neither a new candidate nor an objective mutation can reset these budgets.
"""

from dataclasses import dataclass

import numpy as np

from .kinematics import TrajectoryKinematics
from .repair_limits import CONTACT_DRIFT
from .rows import _block_rows
from .surface_presence import body_penetration, on_surface
from .contact_protection import endpoint_protection, surface_protection

NUMERICAL_SLACK = 1e-7


def _fixed(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class _Contact:
    name: str
    kind: str
    indices: np.ndarray
    local: np.ndarray
    targets: np.ndarray
    active: np.ndarray
    projected: np.ndarray
    normals: np.ndarray
    tolerance: float
    acquired: np.ndarray
    positions: np.ndarray
    established: np.ndarray
    yielded: np.ndarray


class ContactGuard:
    """Keep each acquired endpoint/material-sample error within its allowance.

    Only evidenced active contacts participate. Managed endpoints are covered
    by their material samples, and free tracking targets never become contacts.
    In priority mode only the original acquisition protection masks participate;
    other contacts remain tracking objectives. Legacy secondary corrections keep
    their existing active-contact scope. Full vectors preserve authored motion at each timestamp. For modes
    with independent tangent tasks, those errors are also bounded. The normal
    error of every contact point that lies on an original surface is bounded
    independently, including sticking, so vector improvement cannot conceal
    separation. A point off every surface (``surface_presence``) keeps only its
    full-distance bound: it may approach its target, not be pinned beside it.
    Non-penetration precedes contacts: full, normal and tangent errors (and a
    held position) may grow by the penetration depth of the point's own body
    at the acquired pose, and no further. This one frozen allowance belongs
    to that body only; it is never renewed at a candidate pose.
    """

    def __init__(self, objective, acquired, *, preserve_positions=False):
        self.ablate_contacts = False
        poses = np.asarray(acquired, dtype=float)
        self.shape = (objective.frames, objective.model.nq)
        if poses.shape != self.shape or not np.isfinite(poses).all():
            raise ValueError("Acquired contact trajectory must be finite (frames, model.nq)")
        # Free-body repair holds every attained point. Grouped contact repair
        # may instead approach the original target under its separate anchors.
        self.preserve_positions = bool(preserve_positions)
        self.coords = objective.coords
        self.model = objective.model
        self.objective = objective
        self.priority = None
        self.kin = TrajectoryKinematics(objective.model, objective.coords.cols, cache=objective.kin.cache)
        self.penetration = body_penetration(objective, poses)
        self.kin.evaluate(poses, with_jacobian=False)
        self.groups = []
        for effector in objective.effectors:
            active = np.asarray(effector["active"], bool) & ~np.asarray(effector["managed"], bool)
            if self.priority is not None:
                active = active & endpoint_protection(objective, effector)
            if not active.any():
                continue
            self._add(
                str(effector["name"]), "endpoint", [effector["body"]],
                np.asarray(effector["offset"])[:, None],
                np.asarray(effector["contact"])[:, None], active[:, None],
                np.asarray(effector["surface_only"], bool)[:, None],
                np.asarray(effector["normal"])[:, None], objective.cfg["endpoint_regression_m"],
            )
        for index, surface in enumerate(objective.surfaces):
            active = np.asarray(surface['active'], bool)
            if self.priority is not None:
                active = active & surface_protection(objective, surface)
            if not active.any():
                continue
            modes = np.asarray(surface["task"].modes)
            self._add(
                f"surface_{index}", "surface", surface["geoms"], surface["local"],
                surface["targets"], active,
                np.isin(modes, ("sliding", "touch", "transition")), surface["normals"],
                surface["task"].cfg["regression"],
            )
        self.groups = tuple(self.groups)
        if self.priority is not None:
            self.priority.register(self)

    def _add(self, name, kind, indices, local, targets, active, projected, normals, allowance):
        if not np.isfinite(allowance) or allowance < 0:
            raise ValueError("Contact regression allowance must be finite and nonnegative")
        tolerance = min(CONTACT_DRIFT, float(allowance))
        active = _fixed(active, bool)
        normals = np.asarray(normals, float)
        lengths = np.linalg.norm(normals, axis=-1)
        if np.any(active & projected & (~np.isfinite(lengths) | (lengths < 1e-12))):
            raise ValueError("Projected contact requires a finite nonzero surface normal")
        normals = normals / np.maximum(lengths[..., None], 1e-12)
        group = _Contact(
            name, kind, _fixed(indices, int), _fixed(local, float), _fixed(targets, float),
            active, _fixed(projected, bool), _fixed(normals), tolerance, np.empty(0), np.empty(0),
            np.empty(0, bool), np.empty(0),
        )
        points = self._points(group)
        errors = self._errors(group, points)
        if not np.isfinite(errors[active]).all():
            raise ValueError("Acquired contact errors must be finite on active samples")
        bodies = (self.model.geom_bodyid[group.indices] if kind == "surface"
                  else np.broadcast_to(group.indices, active.shape))
        yielded = self.penetration[np.arange(len(points))[:, None], self.model.body_weldid[bodies]]
        established = active & on_surface(self.objective, points)
        self.groups.append(_Contact(**{
            **vars(group), "acquired": _fixed(errors), "positions": _fixed(points),
            "established": _fixed(established, bool),
            "yielded": _fixed(np.broadcast_to(yielded, active.shape)),
        }))

    def _points(self, group):
        if group.kind == "endpoint":
            bodies = np.broadcast_to(group.indices, group.active.shape)
            return self.kin.points(bodies, group.local)
        return self.kin.geom_points(group.indices, group.local)

    @staticmethod
    def _errors(group, points):
        error = points - group.targets
        full = np.linalg.norm(error, axis=-1)
        along = np.einsum("fsi,fsi->fs", error, group.normals)
        # Even a sticking vector may shrink while its normal separation grows.
        normal = np.abs(along)
        tangent = np.where(
            group.projected, np.linalg.norm(error - along[..., None] * group.normals, axis=-1), 0.,
        )
        return np.stack((full, normal, tangent), axis=-1)

    def repair_bounds(self, poses):
        """Keep full, normal and tangent precision in the correction proposal."""
        if self.priority is not None:
            return [], []
        self.kin.evaluate(poses)
        linear, norms = [], []
        for group in self.groups:
            points = self._points(group)
            bodies = self.model.geom_bodyid[group.indices] if group.kind == "surface" else group.indices
            jac = self.kin.point_jacobians(bodies, points)
            error = points - group.targets
            active = group.active
            held = group.established
            normal = group.normals
            along = np.sum(error * normal, axis=-1)
            normal_jac = np.einsum('fsi,fsiw->fsw', normal, jac)
            if held.any():
                scalar = _block_rows(normal_jac, self.coords.width)[held.ravel()]
                base = group.acquired[..., 1][held] + group.tolerance + group.yielded[held]
                allowance = np.minimum(np.maximum(base + NUMERICAL_SLACK / 2, np.abs(along[held])),
                                       base + NUMERICAL_SLACK)
                lower = -allowance - along[held]
                linear.append((scalar, lower, allowance - along[held]))

            def add_norm(rows, residual, bound, mask):
                selected = np.repeat(mask.ravel(), 3)
                base = np.broadcast_to(bound, mask.shape)[mask]
                allowed = np.minimum(np.maximum(base + NUMERICAL_SLACK / 2,
                                                np.linalg.norm(residual[mask], axis=1)),
                                     base + NUMERICAL_SLACK)
                norms.append((_block_rows(rows.reshape(len(poses), -1, self.coords.width), self.coords.width)[selected],
                              residual[mask], allowed))

            add_norm(jac, error, group.acquired[..., 0] + group.tolerance + group.yielded, active)
            projected = held & group.projected
            if projected.any():
                tangent = error - along[..., None] * normal
                tangent_jac = jac - normal[..., None] * normal_jac[..., None, :]
                add_norm(tangent_jac, tangent,
                         group.acquired[..., 2] + group.tolerance + group.yielded, projected)
            if self.preserve_positions and held.any():
                add_norm(jac, points - group.positions, group.tolerance + group.yielded, held)
        return linear, norms

    def window_dependencies(self, frames):
        return frames

    def qp_bounds(self, poses, *, frames=None):
        if frames is None:
            return self.repair_bounds(poses)
        if self.priority is not None:
            return [], []  # Coordinator owns these soft rows and funded acceptance.
        from .priority_contacts import PriorityContacts
        from .priority_window_contacts import frozen_bounds
        return frozen_bounds(PriorityContacts(self, type(self).__name__), poses, frames)

    def accepts(self, candidate, *, frames=None):
        if frames is not None:
            return not self.violation_frames(candidate, frames=frames).any()
        if self.priority is not None:
            return self.priority.accepts(candidate)
        if type(self) is ContactGuard:
            from inspect import unwrap
            diagnostic = unwrap(self.diagnostics)
            if (diagnostic.__module__ == __name__
                    and diagnostic.__qualname__ == 'ContactGuard.diagnostics'):
                return self._accepts_without_report(candidate)
        return self.diagnostics(candidate)["accepted"]

    def _accepts_without_report(self, candidate):
        """The diagnostics rejection expressions, without reporting reductions."""
        poses = np.asarray(candidate, dtype=float)
        if poses.shape != self.shape or not np.isfinite(poses).all():
            return False
        self.kin.evaluate(poses, with_jacobian=False)
        for group in self.groups:
            points = self._points(group)
            errors = self._errors(group, points)
            checked = np.stack((group.active, group.established, group.established), axis=-1)
            regression = np.where(checked, errors - group.acquired, 0.)
            allowance = group.tolerance + group.yielded[..., None]
            invalid = checked & (~np.isfinite(errors) | (regression - allowance > NUMERICAL_SLACK))
            if invalid.any():
                return False
            if self.preserve_positions and group.established.any():
                drift = np.linalg.norm(points - group.positions, axis=-1)
                budget = group.tolerance + group.yielded
                if np.any(group.established & (~np.isfinite(drift) | (drift > budget + NUMERICAL_SLACK))):
                    return False
        return True

    def violation_frames(self, candidate, *, frames=None):
        if frames is not None:
            if self.priority is not None:
                return self.priority.violation_frames(candidate, frames=frames)
            from .priority_contacts import PriorityContacts
            from .priority_window_contacts import frozen_violations
            return frozen_violations(PriorityContacts(self, type(self).__name__), candidate, frames)
        """Frames ``accepts`` rejects (all of them for an invalid candidate)."""
        if self.priority is not None:
            return self.priority.violation_frames(candidate)
        report = self.diagnostics(candidate)
        bad = np.full(self.shape[0], "reason" in report)
        bad[report["violating_frames"]] = True
        return bad

    def diagnostics(self, candidate):
        if self.priority is not None:
            return self.priority.diagnostics(candidate)
        poses = np.asarray(candidate, dtype=float)
        report = {
            "fixed_acquisition": True, "targets_modified": False,
            "accepted": False, "numerical_slack_m": NUMERICAL_SLACK,
            "preserve_positions": self.preserve_positions,
            "protected_endpoint_frame_points": sum(int(g.active.sum()) for g in self.groups if g.kind == "endpoint"),
            "protected_surface_frame_points": sum(int(g.active.sum()) for g in self.groups if g.kind == "surface"),
            "max_regression_m": 0., "max_excess_m": 0.,
            "max_position_drift_m": 0., "max_position_excess_m": 0.,
            "violating_frames": [], "groups": [],
        }
        if poses.shape != self.shape or not np.isfinite(poses).all():
            report["reason"] = "nonfinite_or_wrong_shape"
            return report
        self.kin.evaluate(poses, with_jacobian=False)
        violated = np.zeros(self.shape[0], bool)
        for group in self.groups:
            points = self._points(group)
            errors = self._errors(group, points)
            regression = errors - group.acquired
            # Points off every surface keep only their full-distance bound.
            checked = np.stack((group.active, group.established, group.established), axis=-1)
            regression = np.where(checked, regression, 0.)
            allowance = group.tolerance + group.yielded[..., None]
            excess = regression - allowance
            invalid = checked & (~np.isfinite(errors) | (excess > NUMERICAL_SLACK))
            frames = np.any(invalid, axis=(1, 2))
            drift = np.linalg.norm(points - group.positions, axis=-1)
            if self.preserve_positions:
                position_budget = group.tolerance + group.yielded
                frames |= np.any(group.established & (~np.isfinite(drift)
                                                      | (drift > position_budget + NUMERICAL_SLACK)), axis=1)
            violated |= frames
            max_regression = float(np.max(np.where(group.active[..., None], regression, 0.), initial=0))
            max_excess = float(np.max(np.where(group.active[..., None], excess, 0.), initial=0))
            if not np.isfinite(errors[group.active]).all():
                max_regression = max_excess = float("inf")
            max_drift = float(drift[group.established].max(initial=0))
            if not np.isfinite(drift[group.established]).all():
                max_drift = float("inf")
            position_excess = max(0., max_drift - group.tolerance)
            if self.preserve_positions:
                max_excess = max(max_excess, position_excess)
            report["max_regression_m"] = max(report["max_regression_m"], max_regression)
            report["max_excess_m"] = max(report["max_excess_m"], max_excess)
            report["max_position_drift_m"] = max(report["max_position_drift_m"], max_drift)
            report["max_position_excess_m"] = max(report["max_position_excess_m"], position_excess)
            report["groups"].append({
                "name": group.name, "kind": group.kind, "tolerance_m": group.tolerance,
                "active_frame_points": int(group.active.sum()),
                "on_surface_frame_points": int(group.established.sum()),
                "acquired_max_error_m": float(group.acquired[group.active, 0].max(initial=0)),
                "max_regression_m": max_regression, "max_excess_m": max_excess,
                "max_position_drift_m": max_drift, "max_position_excess_m": position_excess,
                "violating_frames": np.flatnonzero(frames).tolist(),
            })
        report["violating_frames"] = np.flatnonzero(violated).tolist()
        report["accepted"] = not bool(violated.any())
        return report
