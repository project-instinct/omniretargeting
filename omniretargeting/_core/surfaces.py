"""Prepare one surface correspondence shared by all whole-body objectives."""

import numpy as np

from ..local_collision_repair import diagnostics_json
from ..surface_motion import SurfaceMotion


def _prepare_source_support(owner, source, positions, scene, config, *, map_function=None):
    """Derive source net-wrench support evidence once, unless supplied.

    It tells contacts the source dynamics need (support) from ones that only
    touch; ``contact_phase`` assigns roles from it and keeps geometric roles
    where the mechanics cannot explain a frame.
    """
    from ..contact_phase import contact_phase_config
    from ..source_support_evidence import build_source_support_evidence
    from ..surface_contact import surface_contact_config

    if getattr(owner, "_support_evidence_built", False):
        return
    owner._support_evidence_built = True
    if not contact_phase_config(config.get("contact_phase_priority"))["enabled"]:
        owner.last_source_support_diagnostics = {"status": "disabled", "reason": "contact_phase_disabled"}
        return
    evidence, table, report = build_source_support_evidence(
        (source or {}).get("full_body"),
        positions,
        getattr(owner, "smplx_joint_names", ()),
        getattr(owner, "_source_joint_orientations", None),
        scene,
        surface_contact_config(config.get("surface_contact")),
        framerate=getattr(owner, "_source_framerate", 30.0),
        scene_objects=getattr(owner, "_scene_objects", ()),
        scene_enclosed_free_space=config.get("scene_enclosed_free_space", False),
        table=getattr(owner, "last_source_contact_table", None),
        inertia_model=getattr(owner, "_source_inertia_model", None),
        map_function=map_function,
    )
    owner._source_support_evidence = evidence
    if table is not None:
        owner.last_source_contact_table = table
    owner.last_source_support_diagnostics = report
    print("source_support: " + diagnostics_json(
        {k: report.get(k) for k in ("status", "reason", "inertia_source", "inertia_assumptions", "explained_frames", "frames",
                                    "support_region_frames", "moving_object_candidates_excluded", "fixed_object_frames",
                                    "skin_sample_spacing_m", "support_patch_frames", "support_patch_demoted_candidates",
                                    "support_patch_restored_candidates", "support_demotion_window_frames")}))


def _prepare_object_support(owner, scene, config, *, map_function=None):
    """Ask each moving object, once, which skin contacts carry it.

    Uses the same source contact table as the human evidence; the result only
    promotes contacts to support in the role table (``object_support_evidence``).
    """
    from ..contact_phase import contact_phase_config
    from ..object_support_evidence import build_object_support_evidence
    from ..surface_contact import surface_contact_config

    if getattr(owner, "_object_support_built", False):
        return
    owner._object_support_built = True
    owner._object_support_evidence = None
    objects = list(getattr(owner, "_scene_objects", None) or ())
    if not objects or not contact_phase_config(config.get("contact_phase_priority"))["enabled"]:
        return
    evidence, report = build_object_support_evidence(
        getattr(owner, "last_source_contact_table", None), scene, objects,
        surface_contact_config(config.get("surface_contact")),
        framerate=getattr(owner, "_source_framerate", 30.0),
        scene_enclosed_free_space=config.get("scene_enclosed_free_space", False),
        map_function=map_function,
    )
    owner._object_support_evidence = evidence
    owner.last_object_support_diagnostics = report
    print("object_support: " + diagnostics_json(
        {k: report.get(k) for k in ("status", "reason", "support_region_frames")}))


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
    from .preprocessing_parallel import EvidencePrefetch
    sequence = owner.retargeting_config.get('sequence_solver', {})
    prefetch = getattr(owner,'_preprocess_evidence_prefetch',None) or EvidencePrefetch(
        owner, getattr(owner, '_body_surface_samples', None), positions, scene,
        owner.retargeting_config,
        apply_scene_contacts and sequence.get('enabled', False) and sequence.get('backend') == 'fast',
    )
    try:
        return _prepare_surface_tasks(
            owner, retargeter, positions, scene, reference_positions,
            apply_scene_contacts=apply_scene_contacts, robot_seed_trajectory=robot_seed_trajectory,
            prefetch=prefetch,
        )
    finally:
        prefetch.close()


def _prepare_surface_tasks(
    owner, retargeter, positions, scene, reference_positions, *,
    apply_scene_contacts=True, robot_seed_trajectory=None, prefetch,
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
        from ..support_parallel import SupportPool

        with SupportPool() as pool:
            if not prefetch.consume('source'):
                _prepare_source_support(owner, source, positions, scene, config, map_function=pool.map)
            if not prefetch.consume('object'):
                _prepare_object_support(owner, scene, config, map_function=pool.map)
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
                support_evidence=getattr(owner, "_source_support_evidence", None),
                source_contact_table=getattr(owner, "last_source_contact_table", None),
                scene_enclosed_free_space=config.get("scene_enclosed_free_space", False),
                **shared,
            )
        )
        owner._surface_contacts_built = True
        contact_task = owner.last_surface_contact_tasks
        owner.last_source_contact_table = getattr(contact_task, "source_contact_table", getattr(owner, "last_source_contact_table", None))
        scene_built = True
        from .._sequence.proximity import build_surface_proximity

        owner.last_surface_proximity_tasks = build_surface_proximity(
            owner.robot_model, source, motion, scene,
            getattr(owner, '_scene_objects', ()), config.get('surface_contact'),
            getattr(owner, '_source_framerate', 30.),
            source_contact_table=owner.last_source_contact_table,
        )
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
        from ..object_support_evidence import combine_support_roles

        phase_cfg = contact_phase_config(config.get("contact_phase_priority"))
        if (
            phase_cfg["enabled"]
            and getattr(owner, "_source_control_evidence", None) is None
            and not getattr(owner, "_control_evidence_built", False)
        ):
            # Evidence supplied by the caller is kept; otherwise derive it from
            # the same skin contact observations and prescribed object motion.
            from ..object_control_evidence import build_object_control_evidence

            surface_task = getattr(owner, "last_surface_contact_tasks", None)
            objects = list(getattr(owner, "_scene_objects", None) or ())
            if surface_task is not None and objects:
                owner._source_control_evidence = build_object_control_evidence(
                    surface_task, objects, len(positions),
                    getattr(owner, "_source_framerate", 30.0),
                    scene_present=scene is not None and len(scene.faces) > 0,
                )
            owner._control_evidence_built = True
            evidence = getattr(owner, "_source_control_evidence", None)
            if evidence is not None:
                print("object_control: " + diagnostics_json({
                    k: evidence.diagnostics[k]
                    for k in ("control_frames", "control_region_frames", "observation_budget_m")
                }))
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
            support_evidence=combine_support_roles(
                getattr(owner, "_source_support_evidence", None),
                getattr(owner, "_object_support_evidence", None),
            ),
            control_evidence=getattr(owner, "_source_control_evidence", None),
            unresolved_support_frames=_unresolved_support_frames(owner, len(positions)),
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


def _unresolved_support_frames(owner, frames):
    """Keep observed contacts protected where source mechanics is unknown."""
    report = getattr(owner, "last_source_support_diagnostics", {})
    if report.get("status") == "unresolved":
        return np.ones(frames, bool)
    evidence = getattr(owner, "_source_support_evidence", None)
    if evidence is None:
        return None
    unresolved = np.zeros(frames, bool)
    unresolved[evidence.diagnostics.get("unexplained_frames", [])] = True
    return unresolved
