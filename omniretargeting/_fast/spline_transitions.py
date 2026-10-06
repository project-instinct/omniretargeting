"""Semantic contact-transition allocation for adaptive spline knots.

Contact evidence and every per-binding target remain immutable.  This module
only decides where extra cubic controls can represent a genuine semantic
handoff of a robot body: contact onset/release, mode, environment/material,
role, or a different effector region on that body.
"""

from collections import defaultdict

import numpy as np


def _array(value, frames, default, dtype):
    if value is None:
        return np.full(frames, default, dtype=dtype)
    result = np.asarray(value, dtype=dtype)
    if result.ndim < 1 or result.shape[0] != frames:
        raise ValueError("Transition arrays must share the original timeline")
    return result


def _welded_bodies(objective, bodies):
    model = objective.model
    result = np.asarray(bodies, dtype=int)
    welds = getattr(model, "body_weldid", None)
    return result if welds is None else np.asarray(welds, dtype=int)[result]


def _endpoint_role(objective, effector, frames):
    policy = getattr(objective.tasks, "contact_phase", None)
    name = effector["name"]
    if policy is not None and name in getattr(policy, "roles", {}):
        return _array(policy.roles[name], frames, "free", object)
    if "role" in effector:
        role = np.asarray(effector["role"])
        if role.shape != (frames,):
            raise ValueError("Endpoint fallback role must share the timeline")
        return role
    return np.full(frames, "free", object)


def _add_endpoint_states(objective, states):
    frames = objective.frames
    for effector in objective.effectors:
        name = str(effector["name"])
        task = objective.tasks.effectors[name]["task"]
        active = np.asarray(effector["active"], bool) | np.asarray(
            effector["managed"], bool)
        modes = _array(task.contact_modes, frames, "sticking", object)
        environments = _array(task.environment_ids, frames, 0, int)
        roles = _endpoint_role(objective, effector, frames)
        body = int(_welded_bodies(objective, [effector["body"]])[0])
        for frame in range(frames):
            if active[frame]:
                states[frame][body].add((
                    name, str(modes[frame]), int(environments[frame]),
                    repr(roles[frame]),
                ))


def _surface_names(objective, surface):
    task = surface["task"]
    names = getattr(task, "effector_names", None)
    if names is not None:
        return np.asarray(names, dtype=object)
    return np.asarray([binding.part for binding in task.bindings], dtype=object)


def _add_surface_states(objective, surface, states):
    from .soft_support_rows import surface_roles

    frames = objective.frames
    task = surface["task"]
    active = np.asarray(surface["active"], bool)
    if active.shape != (frames, len(surface["bodies"])):
        raise ValueError("Surface contact map must share the original timeline")
    modes = _array(getattr(task, "modes", None), frames, "contact", object)
    if modes.shape == (frames,):
        modes = np.broadcast_to(modes[:, None], active.shape)
    elif modes.shape != active.shape:
        raise ValueError("Surface modes must share the contact timeline")
    identifiers = _array(getattr(task, "surface_ids", None), frames, 0, int)
    if identifiers.shape == (frames,):
        identifiers = np.broadcast_to(identifiers[:, None], active.shape)
    elif identifiers.shape != active.shape:
        raise ValueError("Surface ids must share the contact timeline")

    roles = surface_roles(objective, surface)
    names = _surface_names(objective, surface)
    bodies = _welded_bodies(objective, surface["bodies"])
    for frame in range(frames):
        for column in np.flatnonzero(active[frame]):
            states[frame][int(bodies[column])].add((
                str(names[column]), str(modes[frame, column]),
                int(identifiers[frame, column]), str(roles[frame, column]),
            ))


def semantic_transition_times(objective):
    """Return interior semantic handoff times, without changing contact data.

    Rows are aggregated by welded robot body.  An exchange of samples with the
    same effector, mode, environment and role is not a new body transition;
    a one-binding onset/release, a mode/material/role change, or a change to a
    different effector/body remains visible.
    """
    frames, framerate = int(objective.frames), float(objective.framerate)
    if frames < 2 or not np.isfinite(framerate) or framerate <= 0:
        raise ValueError("Transition allocation requires at least two frames and a positive rate")

    states = [defaultdict(set) for _ in range(frames)]
    _add_endpoint_states(objective, states)
    for surface in objective.surfaces:
        _add_surface_states(objective, surface, states)

    changes = np.zeros(frames, bool)
    previous = states[0]
    for frame in range(1, frames):
        current = states[frame]
        bodies = set(previous) | set(current)
        changes[frame] = any(previous.get(body, set()) != current.get(body, set())
                             for body in bodies)
        previous = current

    interior = np.flatnonzero(changes)
    interior = interior[(interior > 0) & (interior < frames - 1)]
    return interior / framerate
