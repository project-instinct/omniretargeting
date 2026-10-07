"""Export-only bridge from stored pipeline inputs to the contact/task contract."""

from ..contact_task_output import build_contact_task_output, contact_task_output_config
from ..source_contacts import build_source_contacts
from ..source_markers import marker_observations
from ..surface_contact import surface_contact_config


def contact_task_output(owner, *, source_markers=None):
    if not hasattr(owner, "_contact_output_frames"):
        raise RuntimeError("Run retarget_motion before requesting contact/task output")
    frames = owner._contact_output_frames
    fps = owner._source_framerate
    config = getattr(owner, "retargeting_config", {})
    cfg = contact_task_output_config(config.get("contact_task_output"))
    effectors = config.get("end_effector_priority", {}).get("effectors", {})
    markers = (owner._contact_output_source_markers if source_markers is None else source_markers) or {}
    cfg["candidate_parts"] = (*cfg["candidate_parts"], *effectors, *markers)
    # Profile correspondence is an explicit region declaration, not anatomy
    # guessed from robot link names. User export groups take precedence.
    cfg["body_part_regions"] = {
        **{name: (spec["source_contact_part"],) for name, spec in effectors.items()
           if "source_contact_part" in spec},
        **cfg["body_part_regions"],
    }
    table = getattr(owner, "last_source_contact_table", None)
    scene = owner._contact_output_scene
    objects = owner._scene_objects
    if table is None:
        full = (owner._body_surface_samples or {}).get("full_body")
        authored = None
        if full is None:
            full, authored = marker_observations(markers)
        if full is not None:
            detection = surface_contact_config(config.get("surface_contact"))
            table = build_source_contacts(
                full, scene, detection, framerate=fps, scene_objects=objects,
                scene_enclosed_free_space=config.get("scene_enclosed_free_space", False),
                min_duration_s=detection["min_contact_frames"] / fps, authored_mask=authored,
            )
    names = (["scene"] if scene is not None and len(scene.faces) else [])
    names += [str(spec.get("name", "object")) for spec in objects]
    from ..object_support_evidence import combine_support_roles
    from .surfaces import _unresolved_support_frames

    support = combine_support_roles(
        getattr(owner, "_source_support_evidence", None),
        (getattr(owner, "_object_support_evidence", None)
         if getattr(owner, "_object_support_built", False) else None),
    )
    return build_contact_task_output(
        table, frames=frames, framerate=fps, config=cfg,
        primary_tasks=getattr(owner, "last_end_effector_tasks", {}),
        scene_object_names=names, support_evidence=support,
        control_evidence=getattr(owner, "_source_control_evidence", None),
        unresolved_support_frames=_unresolved_support_frames(owner, frames),
    )
