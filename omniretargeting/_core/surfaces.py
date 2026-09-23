"""Prepare one surface correspondence shared by all whole-body objectives."""

from ..local_collision_repair import diagnostics_json
from ..surface_motion import SurfaceMotion


def prepare_surface_tasks(
    owner,
    retargeter,
    positions,
    scene,
    reference_positions,
    *,
    apply_scene_contacts=True,
    robot_seed_trajectory=None,
):
    from ..self_contact import build_self_contact_tasks
    from ..surface_contact import build_surface_contact_tasks
    from ..surface_pose import build_surface_pose_tasks

    config = owner.retargeting_config
    source = getattr(owner, "_body_surface_samples", None)
    reference = getattr(retargeter, "robot_reference_qpos", None)
    mapping = owner.valid_joint_mapping
    scene_tasks = getattr(owner, "last_end_effector_tasks", {})
    if not getattr(owner, "_surface_tasks_prepared", False):
        configs = [
            config.get(name, {})
            for name in ("surface_contact", "self_contact", "surface_pose")
        ]
        correspondence = [
            c["correspondence"]
            for c in configs
            if c.get("enabled", True) and "correspondence" in c
        ]
        if correspondence and any(c != correspondence[0] for c in correspondence):
            raise ValueError(
                "Whole-body pose and contacts must share one correspondence configuration"
            )
        motion = None
        if (
            source is not None
            and "full_body" in source
            and reference is not None
            and any(c.get("enabled", True) for c in configs)
        ):
            motion = SurfaceMotion(
                model=owner.robot_model,
                source=source,
                reference_qpos=reference,
                mapping=mapping,
                config=correspondence[0] if correspondence else None,
            )
        owner._learned_surface_motion = motion
        owner._surface_tasks_prepared = True
        owner._surface_pose_built = False
    motion = owner._learned_surface_motion
    shared = {"surface_motion": motion} if motion is not None else {}
    if getattr(owner, "last_self_contact_tasks", None) is None:
        owner.last_self_contact_tasks, owner.last_self_contact_diagnostics = (
            build_self_contact_tasks(
                owner.robot_model,
                source,
                mapping,
                reference,
                scene_tasks,
                config.get("self_contact"),
                **shared,
            )
        )
    scene_built = False
    if apply_scene_contacts and not getattr(owner, "_surface_contacts_built", False):
        owner.last_surface_contact_tasks, owner.last_surface_contact_diagnostics = (
            build_surface_contact_tasks(
                owner.robot_model,
                source,
                reference,
                scene,
                config.get("surface_contact"),
                joint_mapping=mapping,
                framerate=getattr(owner, "_source_framerate", 30.0),
                scene_objects=getattr(owner, "_scene_objects", ()),
                robot_seed_trajectory=robot_seed_trajectory,
                **shared,
            )
        )
        owner._surface_contacts_built = True
        scene_built = True
    if not owner._surface_pose_built or scene_built:
        root = owner._resolve_source_root_index()
        # Resolve the chosen source root in the surface's own shaped-rest
        # coordinates. A torso-based profile must not inherit a pelvis offset.
        full = (source or {}).get("full_body", {})
        names = list(full.get("joint_names", []))
        source_names = getattr(owner, "smplx_joint_names", ())
        root_name = source_names[root] if root < len(source_names) else None
        reference_root = None
        if root_name in names and "reference_joint_positions" in full:
            reference_root = full["reference_joint_positions"][names.index(root_name)]
        elif reference_positions is not None:
            reference_root = reference_positions[root]
        elif (
            root_name in names
            and "joint_parents" in full
            and full["joint_parents"][names.index(root_name)] < 0
        ):
            reference_root = full.get("reference_root_position")
        owner.last_surface_pose_tasks, owner.last_surface_pose_diagnostics = (
            build_surface_pose_tasks(
                owner.robot_model,
                source,
                reference,
                positions[:, root],
                reference_root,
                config.get("surface_pose"),
                mapping=mapping,
                scene_tasks=scene_tasks,
                priority=config.get("end_effector_priority", {}),
                surface_contacts=getattr(owner, "last_surface_contact_tasks", None),
                self_contacts=owner.last_self_contact_tasks,
                **shared,
            )
        )
        owner._surface_pose_built = True
        for name in ("self_contact", "surface_contact", "surface_pose"):
            report = getattr(owner, f"last_{name}_diagnostics", {})
            print(
                name
                + ": "
                + diagnostics_json(
                    {
                        k: report.get(k)
                        for k in (
                            "status",
                            "reason",
                            "event_count",
                            "contact_samples",
                            "samples",
                        )
                    }
                )
            )
    if apply_scene_contacts:
        from ..contact_phase import attach_contact_phase, contact_phase_config
        from ..contact_phase_observation import observe_effector_clearances

        phase_cfg = contact_phase_config(config.get("contact_phase_priority"))
        if (
            phase_cfg["enabled"]
            and getattr(owner, "_contact_phase_clearances", None) is None
        ):
            owner._contact_phase_clearances = observe_effector_clearances(
                getattr(owner, "_contact_phase_markers", {}),
                scene,
                config.get("end_effector_priority", {}),
                framerate=getattr(owner, "_source_framerate", 30.0),
                scene_objects=getattr(owner, "_scene_objects", ()),
            )
        owner.last_contact_phase = attach_contact_phase(
            getattr(owner, "last_surface_contact_tasks", None),
            owner.last_surface_pose_tasks,
            scene_tasks,
            config.get("end_effector_priority", {}),
            phase_cfg,
            model=owner.robot_model,
            source_clearances=getattr(owner, "_contact_phase_clearances", None),
        )
    retargeter.contact_phase = (
        getattr(owner, "last_contact_phase", None) if apply_scene_contacts else None
    )
    retargeter.self_contact_tasks = owner.last_self_contact_tasks
    retargeter.surface_contact_tasks = (
        getattr(owner, "last_surface_contact_tasks", None)
        if apply_scene_contacts
        else None
    )
    retargeter.surface_pose_tasks = owner.last_surface_pose_tasks
    for task in (
        retargeter.self_contact_tasks,
        retargeter.surface_contact_tasks,
        retargeter.surface_pose_tasks,
    ):
        if task is not None and len(task.targets) != len(positions):
            raise ValueError("Surface tasks must follow the original motion timeline")
