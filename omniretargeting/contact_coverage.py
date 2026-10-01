"""Keep primary observations when learned surface correspondences drop them."""

from dataclasses import replace

import numpy as np


def surface_effector_samples(surface, name, spec):
    """Match source identity and robot rigid part, never an anatomy whitelist."""
    model = surface.model
    weld = model.body_weldid
    body = model.body(spec["robot_body"]).id
    bodies = [model.geom_bodyid[model.geom(b.robot_geom).id] for b in surface.bindings]
    regions = np.asarray([b.part for b in surface.bindings])
    names = getattr(surface, "effector_names", regions)
    return (np.asarray(names) == spec.get("source_contact_part", name)) & (
        weld[np.asarray(bodies, dtype=int)] == weld[body]
    )


def attach_surface_coverage(surface, primary_tasks, priority):
    """An active point must cover the same source region, object and mode.

    Coverage does not create or remove source contacts. It chooses which
    representation enforces them, including when contact roles are disabled.
    """
    for name, task in primary_tasks.items():
        if not task.surface_managed:
            continue
        covered = np.zeros(len(task.contact_mask), dtype=bool)
        spec = priority.get("effectors", {}).get(name)
        if surface is not None and spec is not None:
            selected = surface_effector_samples(surface, name, spec)
            if selected.any():
                environment = (np.zeros(len(covered), int) if task.environment_ids is None
                               else np.asarray(task.environment_ids))
                valid = surface.active[:, selected].copy()
                valid &= surface.surface_ids[:, selected] == environment[:, None]
                valid &= surface.confidence[:, selected] > 0
                sticking = (np.asarray(task.contact_mask) if task.contact_modes is None
                            else np.asarray(task.contact_modes) == "sticking")
                valid &= ~sticking[:, None] | (surface.modes[:, selected] == "sticking")
                covered = valid.any(axis=1)
        covered.setflags(write=False)
        primary_tasks[name] = replace(task, surface_managed_mask=covered)
