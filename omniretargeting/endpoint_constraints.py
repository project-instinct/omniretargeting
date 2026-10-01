"""Sparse endpoint budgets shared by bounded trajectory correction solvers."""

import mujoco
import numpy as np

from ._solver.targets import _world_target_regime

from ._refinement.math import _skew_symmetric
from .conic_constraints import LocalNormBounds
from .contact import _is_surface_only_contact_mode


def endpoint_constraints(v, current, flat):
    rows = LocalNormBounds(len(current), len(v.qi))
    data = mujoco.MjData(v.model)
    for name, spec in v.specs.items():
        task = spec["task"]
        previous = None
        for f, pose in enumerate(current):
            data.qpos[:] = pose
            mujoco.mj_forward(v.model, data)
            point = v.point(data, name, f)
            original = v.original_effectors[name]["position"][f]
            jp, jr = np.zeros((3, v.model.nv)), np.zeros((3, v.model.nv))
            mujoco.mj_jac(v.model, data, jp, jr, point, spec["body_id"])
            jac = jp[:, v.di]
            payload = task.frame(f)
            if payload["contact"] and payload.get("surface_managed", False):
                continue
            if not payload["contact"]:
                budget = v.swing_budget(name, f)
                if np.isfinite(budget):
                    rows.add(point - original, budget, (f, jac))
                    if _world_target_regime(spec, payload):
                        target = payload["tracking_position"]
                        rows.add(
                            point - target,
                            np.linalg.norm(original - target) + budget,
                            (f, jac),
                        )
                continue
            surface_only = _is_surface_only_contact_mode(payload["contact_mode"])
            projector = np.eye(3)
            if surface_only:
                normal = np.asarray(payload["surface_normal"], float)
                projector = (normal / max(np.linalg.norm(normal), 1e-12))[None]
            target = payload["contact_position"]
            old_error = np.linalg.norm(projector @ (original - target))
            projected = projector @ jac
            rows.add(
                projector @ (point - target),
                max(old_error, v.cfg["contact_tolerance"])
                + v.cfg["contact_regression"],
                (f, projected),
            )
            rows.add(
                projector @ (point - original),
                v.contact_change_budget(old_error),
                (f, projected),
            )
            if payload["contact_mode"] == "sticking":
                if (
                    previous is not None
                    and previous[0] == f - 1
                    and payload["environment_id"] == task.frame(f - 1)["environment_id"]
                ):
                    _, delta, old_jac = previous
                    rows.add(
                        (point - original - delta) * v.fps,
                        v.cfg["support_slip_regression"],
                        (f, jac * v.fps),
                        (f - 1, -old_jac * v.fps),
                    )
                previous = (f, point - original, jac)
            if not spec.get("track_orientation", False):
                continue
            rotation = data.xmat[spec["body_id"]].reshape(3, 3)
            for axis in ["up"] if surface_only else ["up", "forward"]:
                targets = getattr(task, f"{axis}_directions")
                if targets is None:
                    continue
                local = spec.get(
                    "orientation_local_" + axis,
                    [0, 0, 1] if axis == "up" else [1, 0, 0],
                )
                direction = rotation @ local
                direction /= max(np.linalg.norm(direction), 1e-12)
                original_axis = np.asarray(v.original_effectors[name][axis][f], float)
                original_axis = original_axis / max(
                    np.linalg.norm(original_axis), 1e-12
                )
                target_axis = np.asarray(targets[f], float)
                target_axis = target_axis / max(np.linalg.norm(target_axis), 1e-12)
                gradient = -_skew_symmetric(direction) @ jr[:, v.di]
                angle = np.arccos(np.clip(original_axis @ target_axis, -1, 1))
                rows.add(direction - original_axis, 2 * np.sin(0.02 / 2), (f, gradient))
                rows.add(
                    direction - target_axis,
                    2 * np.sin(min(angle + 0.02, np.pi) / 2),
                    (f, gradient),
                )
    return rows.constraints(flat)
