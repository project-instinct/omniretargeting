"""Prepare exact trajectory interpolation samples in scene-query sized batches."""

import mujoco
import numpy as np


def interval_samples(validator, first, second):
    model = validator.model
    velocity = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, velocity, 1, first, second)
    motion = validator.motion_bounds.displacements(first, second)
    bound = float(np.max(motion, initial=0))
    spacing = validator.cfg["motion_spacing"]
    if validator.self_query is not None:
        relative = validator.self_query.motion_bounds.pair_bound(
            validator.self_query.pairs, first, second
        )
        bound = max(bound, relative * spacing / validator.self_query.cfg["motion_spacing"])
    required = max(1, int(np.ceil(bound / spacing)))
    steps = min(required, int(validator.cfg["max_substeps"]))
    interior = []
    for j in range(1, steps):
        pose = first.copy()
        mujoco.mj_integratePos(model, pose, velocity, j / steps)
        interior.append(pose)
    return required, steps, interior


def prepare_intervals(validator, trajectory, start, stop, batch_size, prepare_samples):
    """Group nearby intervals without changing their poses or result ordering."""
    pending = []
    for frame in range(start, stop):
        from .interval_geometry_cache import interval_key
        if interval_key(
            validator.frame_start + frame, trajectory[frame], trajectory[frame + 1]
        ) in validator._interval_cache:
            continue
        _, _, interior = interval_samples(validator, trajectory[frame], trajectory[frame + 1])
        pending.extend(interior)
    for begin in range(0, len(pending), batch_size):
        prepare_samples(pending[begin : begin + batch_size])
