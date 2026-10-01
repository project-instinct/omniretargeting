"""Kinematic variable scopes for local limb repair, independent of source rigs.

The shared ancestor stays fixed even on torso-rooted models. Selecting a limb
changes only the admissible coordinates, never contact evidence or priority.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LimbScope:
    anchor: int
    bodies: frozenset
    columns: frozenset


def paired_limb_scope(model, endpoints, dofs):
    """Find disjoint endpoint branches strictly below their nearest ancestor.

    No joint/body names, gravity axis, limb length or dataset convention enters
    this calculation. Descendants include fixed soles and articulated toes.
    Degenerate endpoints cannot justify moving their shared ancestor.
    """
    endpoints = tuple(dict.fromkeys(int(body) for body in endpoints))
    if len(endpoints) < 2:
        raise ValueError("A limb scope requires at least two distinct endpoints")
    paths = []
    for body in endpoints:
        if not 0 < body < model.nbody:
            raise ValueError("Limb endpoints must be non-world model bodies")
        path = [body]
        while path[-1] != 0:
            path.append(int(model.body_parentid[path[-1]]))
        paths.append(path)
    common = set(paths[0]).intersection(*map(set, paths[1:]))
    anchor = next(body for body in paths[0] if body in common)
    if any(anchor == body for body in endpoints):
        raise ValueError("Limb endpoints must lie on separate branches")
    roots = {path[path.index(anchor) - 1] for path in paths}
    if len(roots) != len(endpoints):
        raise ValueError("Limb endpoints must lie on separate branches")
    bodies = set()
    for body in range(1, model.nbody):
        parent = body
        while parent not in roots and parent not in (0, anchor):
            parent = int(model.body_parentid[parent])
        if parent in roots:
            bodies.add(body)
    joint_bodies = model.jnt_bodyid[model.dof_jntid[np.asarray(dofs, dtype=int)]]
    columns = np.flatnonzero(np.isin(joint_bodies, list(bodies)))
    return LimbScope(anchor, frozenset(bodies), frozenset(map(int, columns)))


def endpoint_limb_scopes(objective):
    """Resolve profile-supplied groups independently of effector/source names."""
    groups = {}
    specs = getattr(objective.tasks, "effectors", {})
    for eff in objective.effectors:
        group = specs.get(eff["name"], {}).get("local_repair_group")
        if group is not None:
            groups.setdefault(str(group), []).append(eff["body"])
    if not groups:
        # Backward compatibility for direct low-level callers whose explicit
        # endpoint names already specify robot anatomy.
        endpoints = {eff["name"]: eff["body"] for eff in objective.effectors}
        names = ("left_foot", "right_foot")
        if all(name in endpoints for name in names):
            groups["lower_limbs"] = [endpoints[n] for n in names]
    return tuple(paired_limb_scope(objective.model, bodies, objective.coords.di)
                 for bodies in groups.values() if len(set(bodies)) >= 2)


def scope_for_body(limits, body):
    for scope in getattr(limits, "limb_scopes", ()):
        if body in scope.bodies:
            return scope
    return None
