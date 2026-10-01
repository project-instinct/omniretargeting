"""Derive endpoint views of the same immutable whole-body contact table."""

import numpy as np
from dataclasses import replace

from ..contact import SceneContactTask
from ..contact_correspondence import coherent_sample_indices
from ..contact_schedule import _iter_true_runs
from ..source_markers import marker_observations
from ..source_contacts import build_source_contacts
from ..surface_contact import surface_contact_config


def endpoint_marker(markers, name, sample_labels, indices):
    """Use adapter-declared material identities when a profile renames a tip."""
    markers = markers or {}
    if name in markers:
        return markers[name]
    labels = set(np.asarray(sample_labels)[indices]) - {""}
    if len(labels) == 1:
        return markers.get(next(iter(labels)), {})
    return {}  # Mixed/unknown anatomy cannot justify choosing a semantic frame.


def unified_contact_inputs(owner, source, positions, scene, objects, markers, fps):
    dense = source is not None and "full_body" in source
    full, authored = (source["full_body"], None) if dense else marker_observations(markers or {})
    cfg = surface_contact_config(owner.retargeting_config.get("surface_contact"))
    table = None if full is None else build_source_contacts(
        full, scene, cfg, framerate=fps, scene_objects=objects,
        scene_enclosed_free_space=owner.retargeting_config.get("scene_enclosed_free_space", False),
        min_duration_s=cfg["min_contact_frames"] / fps, authored_mask=authored,
    )
    owner.last_source_contact_table = table
    full = full or {}
    count = 0 if table is None else len(table.regions)
    names = list(full.get("joint_names", owner.smplx_joint_names))
    parents = np.asarray(full.get("joint_parents", np.full(len(names), -1)))
    source_joints = np.asarray(full.get("sample_joint_indices", np.full(count, -1)))
    labels = np.asarray(full.get("sample_effectors", [""] * count))
    priority = owner.retargeting_config["end_effector_priority"]
    tasks, output_markers, clearances = {}, {}, {}
    for name, spec in priority.get("effectors", {}).items():
        source_name = spec.get("source_joint")
        group = labels == name
        if not group.any() and source_name in names:
            selected_joint = names.index(source_name)
            descendants = {selected_joint}
            for joint in range(len(parents)):
                parent = joint
                while parent >= 0 and parent not in descendants:
                    parent = int(parents[parent])
                if parent in descendants:
                    descendants.add(joint)
            group = np.isin(source_joints, list(descendants))
        ids = np.flatnonzero(group)
        supplied = endpoint_marker(markers, name, labels, ids)
        if not len(ids):
            # No material observations means unknown clearance, never a
            # fabricated contact. A mapped joint still has a free target.
            if "tracking_positions" in supplied:
                tracking = np.asarray(supplied["tracking_positions"], float)
            elif source_name in owner.smplx_joint_names:
                tracking = positions[:, owner.smplx_joint_names.index(source_name)]
            else:
                continue
            total = len(tracking)
            task = SceneContactTask(
                name,
                tracking,
                tracking.copy(),
                tracking.copy(),
                np.tile([0, 0, 1], (total, 1)),
                np.full(total, np.inf),
                np.zeros(total),
                np.zeros(total),
                np.zeros(total, bool),
                surface_managed=dense,
            )
        else:
            obs = table.observation
            distances = obs["distance"][:, ids]
            # Correspondence optimization follows actual skin motion. Inactive
            # samples cannot win solely because their backside is closer.
            selected = coherent_sample_indices(
                distances,
                obs["material_environment"][:, ids],
                obs["material_points"][:, ids],
                max_distance=cfg["distance"], valid=table.active[:, ids],
            )
            frame = np.arange(len(positions))
            chosen = ids[selected]
            contact = table.active[frame, chosen]
            tracking = np.asarray(
                supplied.get("tracking_positions", table.points[:, ids].mean(1)), float
            )
            task = SceneContactTask(
                name,
                tracking,
                obs["environment"][frame, chosen].copy(),
                obs["environment"][frame, chosen].copy(),
                obs["normals"][frame, chosen].copy(),
                obs["distance"][frame, chosen].copy(),
                np.linalg.norm(obs["relative_velocity"][frame, chosen], axis=1),
                obs["confidence"][frame, chosen].copy(),
                contact.copy(),
                forward_directions=supplied.get("forward_directions"),
                up_directions=supplied.get("up_directions"),
                contact_modes=obs["modes"][frame, chosen].copy(),
                environment_ids=obs["surface_ids"][frame, chosen].copy(),
                surface_managed=dense,
            )
            if not dense:
                # A marker-only patch has a single active robot point. Lock
                # its material placement over the same source contact runs.
                target = task.contact_positions.copy()
                for surface_id in np.unique(task.environment_ids[contact]):
                    mask = contact & (task.environment_ids == surface_id) & (task.contact_modes == "sticking")
                    rotation, translation, scale = obs["surface_poses"][surface_id]
                    for start, stop in _iter_true_runs(mask):
                        material = obs["material_environment"][frame, chosen][start:stop]
                        anchor = np.mean(material, axis=0)
                        target[start:stop] = np.einsum("tij,j->ti", rotation[start:stop], anchor) * scale[start:stop, None] + translation[start:stop]
                task = replace(task, contact_positions=target, preserve_contact_anchors=True)
            output_markers[name] = {
                "tracking_positions": tracking,
                "contact_samples": table.points[:, ids],
            }
            clearances[name] = np.where(
                (table.valid[:, ids] & obs["clear_side"][:, ids]).all(1),
                distances.min(1),
                np.nan,
            )
        tasks[name] = task
        known_clear = np.isfinite(clearances.get(name, np.full(len(positions), np.nan)))
        known_clear &= clearances.get(name, np.zeros(len(positions))) > cfg["distance"]
        world = ~known_clear
        world_tracking = source_name in set(full.get("world_target_joints", ()))
        if world_tracking:
            world[:] = True
        if "preserve_world_target" in spec:
            world[:] = bool(spec["preserve_world_target"])
            world_tracking = bool(spec["preserve_world_target"])
        tasks[name] = replace(task, swing_world_mask=world,
                              preserve_world_tracking=world_tracking)
    owner._contact_phase_clearances = clearances
    return tasks, output_markers
