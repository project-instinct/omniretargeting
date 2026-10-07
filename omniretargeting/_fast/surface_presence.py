"""Whether protected contact points actually lie on an original surface.

A protected (support/hold) point is frozen at its normal distance to the
target's tangent plane. That plane stands for the surface only where the point
is on the surface: a hand far from its target, or buried in an object wall,
merely crosses the plane's extension. Presence is read from the solver's own
clearance fields; where no field knows the point, it is treated as present, so
unknown geometry keeps the existing protection.
"""

import numpy as np

from ..surface_contact import surface_contact_config

# The source detector's own contact distance: no new threshold.
CONTACT_DISTANCE_M = surface_contact_config()["distance"]


def on_surface(objective, points, distance=CONTACT_DISTANCE_M):
    """(frames, samples) mask of points within ``distance`` of an actual surface."""
    points = np.asarray(points, dtype=float)
    fields = objective._clearance_fields() if hasattr(objective, "_clearance_fields") else []
    if not fields or not points.size:
        return np.ones(points.shape[:2], bool)
    frames = np.repeat(np.arange(points.shape[0]), points.shape[1])
    flat = points.reshape(-1, 1, 3)
    present = np.zeros(len(flat), bool)
    known = np.zeros(len(flat), bool)
    for field in range(len(fields)):
        value = objective._inside(field, frames, flat)[:, 0]
        finite = np.isfinite(value)
        known |= finite
        present |= finite & (np.abs(value) <= distance)
    return (present | ~known).reshape(points.shape[:2])


def body_penetration(objective, poses):
    """(frames, nbody) deepest scene/object penetration of each welded body.

    Read from the solver's clearance fields at ``poses``: the depth a contact
    on that body may yield so that clearance, which precedes contacts, can be
    restored. Bodies away from every field have zero depth.
    """
    priority = None
    if priority is not None:
        return priority.frozen_depth(poses)
    provider = None
    if provider is not None:
        return provider(poses)
    model = objective.model
    depth = np.zeros((len(poses), model.nbody))
    fields = objective._clearance_fields() if hasattr(objective, "_clearance_fields") else []
    if not fields:
        return depth
    welds = model.body_weldid[model.geom_bodyid[np.asarray(objective.shell_geoms)]]
    # An explicit per-objective provider (CPU/device/window geometry, or a test
    # fixture) owns the measurement protocol. Do not let ``deepest_fields``
    # install the bulk helper over it before reading depth.
    provider = objective.__dict__.get("_deepest_samples")
    provider_is_bulk = getattr(provider, "__self__", None) is not None
    if hasattr(objective, "deepest_fields") and (provider is None or provider_is_bulk):
        deepest = objective.deepest_fields(poses)
    else:
        objective.kin.evaluate(poses, with_jacobian=False)
        deepest = [provider(*field[1:]) for field in fields]
    for clearance, *_ in deepest:
        sunk = np.where(np.isfinite(clearance), np.maximum(-clearance, 0.0), 0.0)
        for column, weld in enumerate(welds):
            depth[:, weld] = np.maximum(depth[:, weld], sunk[:, column])
    return depth
