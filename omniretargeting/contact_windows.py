"""Slice full-timeline observations without recomputing contact handovers."""

from dataclasses import fields, replace

import numpy as np


def slice_contact_task(task, start, stop):
    """Keep global anchors and approach progress; attach window roles separately.

    Contact choices must likewise be planned once on the full trajectory and
    sliced by the caller. Re-inferring either near a crop boundary changes the
    optimization problem and can produce a visible seam.
    """
    frames = len(task.contact_mask)
    if not 0 <= start < stop <= frames:
        raise ValueError("Contact window must be a nonempty source interval")
    values = {}
    for field in fields(task):
        value = getattr(task, field.name)
        if isinstance(value, np.ndarray) and value.ndim and len(value) == frames:
            values[field.name] = value[start:stop].copy()
    values["contact_phase"] = None
    values["approach_reference"] = tuple(
        value[start:stop].copy() for value in task.approach
    )
    return replace(task, **values)


def restore_contact_roles(surface, primary_tasks, priority, payload):
    """Restore exported source roles for replay, without inferring new mechanics.

    The caller slices the exported timeline first. Missing free-space evidence
    stays unknown, so a replay cannot newly release a possible world hold.
    """
    from .contact_phase import CONTACT_ROLES, ROLES, attach_contact_phase

    if "contact_roles" not in payload:
        return None
    names = list(map(str, payload["contact_role_names"]))
    values = np.asarray(payload["contact_roles"])
    frames = len(surface.active)
    if values.shape != (frames, len(names)) or not np.isin(values, ROLES).all():
        raise ValueError("Exported contact roles must share the replay timeline")
    masks = {}
    for role in CONTACT_ROLES:
        mask = np.asarray(payload[f"contact_role_surface_{role}"], dtype=bool)
        if mask.shape != surface.active.shape or np.any(mask & ~surface.active):
            raise ValueError("Exported surface roles must refer to active contacts")
        masks[role] = mask.copy()
        masks[role].setflags(write=False)
    if np.any(np.stack(list(masks.values())).sum(axis=0) > 1):
        raise ValueError("Exported surface contact roles must be disjoint")
    policy = attach_contact_phase(surface, None, primary_tasks, priority)
    if set(names) != set(policy.roles):
        raise ValueError(
            "Exported contact role identities differ from the replay tasks"
        )
    policy.roles = {name: values[:, i].copy() for i, name in enumerate(names)}
    for value in policy.roles.values():
        value.setflags(write=False)
    policy.surface_roles = masks
    policy.support = masks["support"]
    policy.follow_surface = masks["follow"]
    policy.diagnostics = {
        **policy.diagnostics,
        "method": "restored_source_contact_roles",
        "source_support_evidence": "roles_restored_from_export",
        "source_control_evidence": "roles_restored_from_export",
        "surface_support_frame_points": int(policy.support.sum()),
        "surface_follow_frame_points": int(policy.follow_surface.sum()),
        "surface_role_frame_points": {r: int(m.sum()) for r, m in masks.items()},
        "role_frames": {
            name: {
                role: np.flatnonzero(value == role).tolist()
                for role in ROLES
                if np.any(value == role)
            }
            for name, value in policy.roles.items()
        },
    }
    # Recomputed phase descriptions must use the restored roles too.
    welds = np.array(
        [
            surface.model.body_weldid[
                surface.model.geom_bodyid[surface.model.geom(b.robot_geom).id]
            ]
            for b in surface.bindings
        ]
    )
    policy.diagnostics["phases"] = policy._phases(welds)
    policy.diagnostics["surface_support_bodies"] = sorted(
        {surface.model.body(int(w)).name for w in welds[policy.support.any(axis=0)]}
    )
    return policy
