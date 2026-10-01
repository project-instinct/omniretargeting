"""Hold attained contact vectors while one rigid contact group is corrected.

A bound on distance to a target alone permits motion around that target. These
additional bounds keep other contacts at their attained positions. The original
contact, posture and per-joint continuity budgets remain independently active.
"""

from dataclasses import dataclass

import numpy as np

from ..conic_constraints import LocalNormBounds


def contact_body(tasks, key, frame):
    """Return a welded body identity for an active contact row, never a label."""
    if key[0] == "endpoint":
        spec = tasks.effectors[key[1]]
        if not spec["task"].frame(frame)["contact"]:
            return None
        body = spec["body_id"]
    elif key[0] == "surface_contact":
        task = tasks.extra_tasks[key[1]]
        if not hasattr(task, "surface_ids"):
            # A self-contact row constrains a pair in a material frame, not
            # one scene anchor. Its original vector budget remains active.
            return None
        binding = task.bindings[key[2]]
        body = tasks.model.geom_bodyid[tasks.model.geom(binding.robot_geom).id]
    else:
        return None
    return int(tasks.model.body_weldid[body])


class ContactHoldAnchors:
    """Immutable attained positions, updated only for the group being repaired."""

    def __init__(self, guard, reference):
        self.values = []
        self.bodies = []
        for frame, pose in enumerate(reference):
            values, bodies = {}, {}
            for key, residual, _, _, _ in guard.rows(
                pose, frame, full_surface_vectors=True
            ):
                body = contact_body(guard.tasks, key, frame)
                if body is not None:
                    values[key] = np.array(residual, copy=True)
                    values[key].setflags(write=False)
                    bodies[key] = body
            self.values.append(values)
            self.bodies.append(bodies)

    def acquired(self, guard, poses, body):
        """Freeze this corrected group without ratcheting any other anchor."""
        result = object.__new__(type(self))
        result.values = [dict(values) for values in self.values]
        result.bodies = self.bodies
        for frame, pose in enumerate(poses):
            for key, residual, _, _, _ in guard.rows(
                pose, frame, full_surface_vectors=True
            ):
                if self.bodies[frame].get(key) == body:
                    value = np.array(residual, copy=True)
                    value.setflags(write=False)
                    result.values[frame][key] = value
        return result


@dataclass(frozen=True)
class ContactHoldStage:
    anchors: ContactHoldAnchors
    movable_body: int | None
    tolerance: float
    # Optional per-frame lower/upper root heights, derived from geometric exit
    # evidence. XY/orientation locks are supplied through Coordinates as usual.
    root_height_bounds: np.ndarray | None = None
    objective: object | None = None
    continuity_reference: np.ndarray | None = None

    def bind(self, guard):
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("Contact hold tolerance must be finite and positive")
        if len(self.anchors.values) != guard.tasks.frames:
            raise ValueError("Contact hold anchors must share the complete timeline")
        return BoundContactHold(self, guard)


class BoundContactHold:
    def __init__(self, stage, guard):
        self.stage, self.guard = stage, guard
        self.continuity = None
        reference = stage.continuity_reference
        if reference is None and stage.objective is not None:
            reference = stage.objective.reference
        if reference is not None:
            from .continuity_guard import SequenceContinuityGuard

            if guard.continuity is None:
                raise ValueError("Stage continuity requires the fixed sequence guard")
            self.continuity = SequenceContinuityGuard(
                guard.tasks, np.array(reference, copy=True), guard.continuity.rotations
            )
        bounds = stage.root_height_bounds
        if bounds is not None:
            bounds = np.array(bounds, dtype=float, copy=True)
            if (
                bounds.shape != (guard.tasks.frames, 2)
                or not np.isfinite(bounds).all()
                or np.any(bounds[:, 0] > bounds[:, 1])
                or guard.tasks.coords.root_q is None
            ):
                raise ValueError(
                    "Root height bounds require a finite floating timeline"
                )
            bounds.setflags(write=False)
        self.root_bounds = bounds

    def rows(self, poses, frames=None):
        stage, guard = self.stage, self.guard
        for frame in range(len(poses)) if frames is None else frames:
            seen = set()
            for key, residual, jac, _, _ in guard.rows(
                poses[frame], frame, full_surface_vectors=True
            ):
                if key not in stage.anchors.values[frame]:
                    continue
                seen.add(key)
                if stage.anchors.bodies[frame][key] != stage.movable_body:
                    yield frame, residual - stage.anchors.values[frame][key], jac
            if seen != set(stage.anchors.values[frame]):
                raise ValueError(
                    "A contact stage changed the fixed contact row identities"
                )

    def constrain_step(self, poses, lower, upper, frames=None):
        if self.root_bounds is None:
            return
        coords = self.guard.tasks.coords
        column = coords.inverse.get(coords.root_d + 2)
        if column is None:
            return
        span = slice(None) if frames is None else slice(frames.start, frames.stop)
        height = poses[span, coords.root_q + 2]
        lower[:, column] = np.maximum(
            lower[:, column], self.root_bounds[span, 0] - height
        )
        upper[:, column] = np.minimum(
            upper[:, column], self.root_bounds[span, 1] - height
        )

    def constraints(self, poses, flat, frames=None):
        window = range(len(poses)) if frames is None else frames
        rows = LocalNormBounds(len(window), self.guard.tasks.coords.width)
        for frame, residual, jac in self.rows(poses, frames):
            rows.add(residual, self.stage.tolerance, (frame - window.start, jac))
        constraints = rows.constraints(flat)
        if self.continuity is not None:
            constraints.extend(self.continuity.constraints(poses, flat, frames))
        return constraints

    def accepts(self, poses, frames=None):
        if self.continuity is not None and not self.continuity.accepts(poses, frames):
            return False
        if self.root_bounds is not None:
            span = slice(None) if frames is None else slice(frames.start, frames.stop)
            height = poses[span, self.guard.tasks.coords.root_q + 2]
            if np.any(height < self.root_bounds[span, 0] - 1e-7) or np.any(
                height > self.root_bounds[span, 1] + 1e-7
            ):
                return False
        return all(
            np.linalg.norm(residual) <= self.stage.tolerance + 1e-7
            for _, residual, _ in self.rows(poses, frames)
        )

    def diagnostics(self, poses):
        maximum = max(
            (float(np.linalg.norm(r)) for _, r, _ in self.rows(poses)), default=0.0
        )
        return {
            "movable_body": (
                None
                if self.stage.movable_body is None
                else self.guard.tasks.model.body(self.stage.movable_body).name
            ),
            "held_contact_displacement_max_m": maximum,
            "held_contact_budget_m": self.stage.tolerance,
            "within_hold_budget": self.accepts(poses),
            "preserves_acquired_continuity": self.continuity is not None,
        }
