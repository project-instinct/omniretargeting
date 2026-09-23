"""Geometry-based initial guesses for floating-base motion solves."""

from __future__ import annotations

from collections.abc import Mapping

import mujoco
import numpy as np


def align_initial_base_height_to_feet(
    model: mujoco.MjModel,
    q_init: np.ndarray,
    effectors: Mapping[str, dict],
    tasks: Mapping[str, object],
    *,
    source_height_spread: float = 0.06,
) -> np.ndarray:
    """Lift only the initial guess if a low world-tracking foot starts below target.

    The robot foot marker is measured by FK at the source root pose.  This does
    not constrain or modify any solved frame, source target, or scene geometry.
    Among the available feet, use only those close to the lowest source foot;
    a lifted swing foot must not pull the initial base upward.
    """
    selected = []
    for name, task in tasks.items():
        if not name.endswith("_foot"):
            continue
        spec = effectors.get(name)
        if not spec or not spec.get("preserve_world_target", False):
            continue
        payload = task.frame(0)
        key = "contact_position" if payload.get("contact", False) else "tracking_position"
        target = payload.get(key)
        if target is None:
            continue
        target = np.asarray(target, dtype=float).reshape(3)
        if np.isfinite(target).all():
            selected.append((spec, target))
    if not selected:
        return q_init

    data = mujoco.MjData(model)
    data.qpos[:] = q_init
    mujoco.mj_forward(model, data)
    lowest_target_z = min(float(target[2]) for _, target in selected)
    lifts = []
    for spec, target in selected:
        if target[2] > lowest_target_z + source_height_spread:
            continue
        body_id = int(spec["body_id"])
        offset = np.asarray(spec["tracking_offset"], dtype=float).reshape(3)
        point = data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ offset
        lifts.append(float(target[2] - point[2]))
    if not lifts or max(lifts) <= 0.0:
        return q_init
    aligned = q_init.copy()
    aligned[2] += max(lifts)
    return aligned
