"""Prepare the generic sequence solver from existing core correspondences."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .._sequence.coordinates import Coordinates
from ..sequence_retargeting import retarget_sequence
from .orientation_targets import sequence_body_orientation_targets


def solve_core_sequence(
    owner,
    retargeter,
    positions,
    tasks,
    base_orientations,
    object_points,
    reference_trajectory=None,
    source_joint_orientations=None,
    *, source_only=False,
):
    model = owner.robot_model
    frames = len(positions)
    coordinates = Coordinates(model)
    root_index = owner._resolve_source_root_index()
    config = dict(owner.retargeting_config.get("sequence_solver", {}))
    config.setdefault(
        "self_collision_enabled",
        owner.retargeting_config.get("self_collision_enabled", True),
    )
    config.setdefault(
        "self_collision", owner.retargeting_config.get("self_collision", {})
    )
    config.setdefault(
        "joint_limit_safety_fraction",
        owner.retargeting_config.get("joint_limit_safety_fraction", 1.0),
    )
    priority = owner.retargeting_config.get("end_effector_priority", {})
    config.setdefault(
        "lock_root_z", owner.retargeting_config.get("preserve_root_z", False)
    )
    config.setdefault(
        "lock_root_orientation", priority.get("lock_root_orientation_to_source", False)
    )
    mapping = dict(owner.valid_joint_mapping)
    # Torso axes use existing semantic correspondences, never source file names.
    shoulders = []
    for spec in priority.get("effectors", {}).values():
        arm = spec.get("arm_posture", {})
        name = arm.get("source_shoulder")
        if name in mapping and name not in shoulders:
            shoulders.append(name)
            if arm.get("shoulder_body"):
                mapping[name] = arm["shoulder_body"]
    source_root_name = owner.smplx_joint_names[root_index]
    if source_root_name in mapping:
        priority = dict(priority, source_root_robot_body=mapping[source_root_name])
        owner.retargeting_config["end_effector_priority"] = priority
        retargeter.end_effector_config = priority
    if len(shoulders) == 2 and source_root_name in mapping:
        config.setdefault(
            "pose_axes",
            [
                {
                    "weights": {
                        shoulders[0]: 0.5,
                        shoulders[1]: 0.5,
                        source_root_name: -1,
                    },
                    "sigma": 0.04,
                },
                {"weights": {shoulders[0]: 1, shoulders[1]: -1}, "sigma": 0.06},
            ],
        )
    # A model can float at its chest while the source root denotes the pelvis.
    # Use the calibrated source rotation of that actual floating body, when
    # available, rather than asking both chest and pelvis to share one rotation.
    body_orientation_targets = sequence_body_orientation_targets(
        retargeter, source_joint_orientations, None
    )
    if coordinates.root_d is not None:
        floating_body = model.body(
            int(model.jnt_bodyid[model.dof_jntid[coordinates.root_d]])
        ).name
        for target in body_orientation_targets:
            if target["robot_body"] == floating_body and not target.get("parent_body"):
                base_orientations = Rotation.from_matrix(target["rotations"]).as_quat(
                    scalar_first=True
                )
                break
    if reference_trajectory is None:
        seed = getattr(retargeter, "robot_reference_qpos", None)
        initial = np.tile(model.qpos0 if seed is None else seed, (frames, 1))
        if coordinates.root_q is not None:
            root = coordinates.root_q
            initial[:, root : root + 3] = positions[:, root_index]
            for f in range(frames):
                orientation = (
                    None if base_orientations is None else base_orientations[f]
                )
                if orientation is None:
                    xyzw = owner._estimate_base_orientation_from_joints(positions[f])
                    if xyzw is not None:
                        orientation = np.asarray(xyzw)[[3, 0, 1, 2]]
                if orientation is not None:
                    orientation = np.asarray(orientation)
                    if orientation.shape == (3,):
                        orientation = Rotation.from_rotvec(orientation).as_quat(
                            scalar_first=True
                        )
                    initial[f, root + 3 : root + 7] = orientation
            if source_root_name in mapping:
                data = mujoco.MjData(model)
                body = model.body(mapping[source_root_name]).id
                for frame, pose in enumerate(initial):
                    data.qpos[:] = pose
                    mujoco.mj_forward(model, data)
                    initial[frame, root : root + 3] += (
                        positions[frame, root_index] - data.xpos[body]
                    )
    else:
        initial = np.asarray(reference_trajectory, dtype=float).copy()
    # Freeze source orientation targets before bootstrap. Its output must not
    # become the next pass's new definition of the desired back angle.
    orientation_reference = base_orientations
    if orientation_reference is None and coordinates.root_q is not None:
        start = coordinates.root_q + 3
        orientation_reference = initial[:, start : start + 4].copy()
    elif orientation_reference is not None:
        orientation_reference = np.asarray(orientation_reference, dtype=float)
        if orientation_reference.shape == (frames, 3):
            orientation_reference = Rotation.from_rotvec(orientation_reference).as_quat(
                scalar_first=True
            )
    terrain = np.broadcast_to(
        retargeter.terrain_points, (frames, *retargeter.terrain_points.shape)
    )
    environment = (
        terrain
        if object_points is None
        else np.concatenate([terrain, object_points], axis=1)
    )

    def progress(_pose, record):
        if record.get("iteration", 0) % 5 == 0 or record.get("alpha") == 0:
            # Window records of the windowed solver carry no trajectory-wide
            # contact statistic; print it only when the record has it.
            contact = record.get("sticking_max_m")
            print(
                f"Sequence {record.get('phase')} {record.get('iteration')}: "
                f"cost={record.get('after', float('nan')):.3f}"
                + ("" if contact is None else f", contact={contact:.5f} m"),
                flush=True,
            )

    common = {
        "framerate": owner._source_framerate,
        "environment_points": environment,
        "scene_query": retargeter.scene_query,
        "root_orientation_targets": orientation_reference,
        "body_orientation_targets": body_orientation_targets,
    }
    if source_only:
        config.pop("backend",None)
        from .._fast.pipeline import prepare_fast_source
        return prepare_fast_source(
            owner, model, initial, positions, owner.smplx_joint_names, mapping,
            positions[:, root_index], tasks or {}, priority, config, common, (),

        )
    initialization_report = None
    extra_tasks = (
        getattr(owner, "last_body_relation_tasks", None),
        getattr(owner, "last_self_contact_tasks", None),
        getattr(owner, "last_surface_contact_tasks", None),
        getattr(owner, "last_surface_pose_tasks", None),
        getattr(owner, "last_surface_proximity_tasks", None),
    )
    backend = config.pop("backend", "sequence")
    if config.get("staged_contact_repair", False) and backend != "sequence":
        raise ValueError("staged_contact_repair requires the strict sequence backend")
    if getattr(owner, "_scene_objects", ()) and backend != "fast":
        from .._sequence.scene import PrescribedSceneClearance
        from ..sequence_retargeting import SEQUENCE_DEFAULTS

        extra_tasks += (
            PrescribedSceneClearance(
                model,
                owner._scene_objects,
                frames,
                clearance=config.get(
                    "scene_clearance_m", SEQUENCE_DEFAULTS["scene_clearance_m"]
                ),
                query_config=config.get("moving_scene"),
            ),
        )
    if backend == "fast":
        # Speed-first backend: same objective terms, banded Gauss-Newton,
        # scene grids, budgets verified afterwards (see _fast/pipeline.py).
        from .._fast.pipeline import solve_fast_sequence

        motion, report = solve_fast_sequence(
            owner, model, initial, positions, owner.smplx_joint_names, mapping,
            positions[:, root_index], tasks or {}, priority, config, common,
            extra_tasks, reference_trajectory, callback=progress,
        )
        owner.last_sequence_report = report
        owner.last_unprocessed_framewise_motion = motion.copy()
        return motion
    if backend == "mink":
        # UMR-style per-frame QPs warm-start from the previous frame, so the
        # relaxed whole-sequence bootstrap is not needed.
        from ..mink_retargeting import retarget_mink

        def frame_progress(_pose, record):
            if record["frame"] % 30 == 0 or record["frame"] == frames - 1:
                print(
                    f"Mink frame {record['frame']}/{frames}: cost={record['after']:.3f}, "
                    f"iterations={record['iterations']}, {record['seconds']:.1f} s",
                    flush=True,
                )

        motion, report = retarget_mink(
            model,
            initial,
            positions,
            owner.smplx_joint_names,
            mapping,
            positions[:, root_index],
            tasks or {},
            priority,
            config=config,
            callback=frame_progress,
            extra_tasks=extra_tasks,
            **common,
        )
        report["initialization"] = None
        owner.last_sequence_report = report
        owner.last_unprocessed_framewise_motion = motion.copy()
        return motion
    if reference_trajectory is None:
        from .._sequence.source_pose import acquire_source_pose

        initial, source_pose_report = acquire_source_pose(
            model,
            initial,
            positions,
            owner.smplx_joint_names,
            mapping,
            positions[:, root_index],
            tasks or {},
            priority,
            config=config,
            callback=progress,
            solve=retarget_sequence,
            **{
                key: common[key]
                for key in (
                    "framerate",
                    "root_orientation_targets",
                    "body_orientation_targets",
                )
            },
        )
        source_pose_reference = initial.copy()
        # Source-fit diagnostics do not abort optimization. Keep this reference
        # fixed and continue under cumulative angular bounds; incompatible
        # source/contact geometry remains a reported residual.
        # The source-only solve has already selected the kinematic branch.
        # Proceed directly to evidence-ordered contact acquisition: a second
        # relaxed pass with contacts demoted can detach the feet for clearance
        # and consume the same fixed posture budget before acquisition starts.
        initialization_report = {"source_pose": source_pose_report}

    contact_reference, contact_report = None, None
    correction_reference = (
        initial.copy() if reference_trajectory is not None else source_pose_reference
    )
    surface = getattr(owner, "last_surface_contact_tasks", None)
    if any(np.any(t.contact_mask) for t in (tasks or {}).values()) or (
        surface is not None and np.any(surface.active)
    ):
        # First acquire reachable world contacts with continuous root motion.
        # Subsequent geometry/posture improvement must preserve that measured
        # accuracy instead of buying clearance by detaching a hand or foot.
        contact_reference, contact_report = retarget_sequence(
            model,
            initial,
            positions,
            owner.smplx_joint_names,
            mapping,
            positions[:, root_index],
            tasks or {},
            priority,
            config=config,
            callback=progress,
            extra_tasks=extra_tasks,
            correction_reference_trajectory=correction_reference,
            **common,
        )
        initial = contact_reference.copy()

    motion, report = retarget_sequence(
        model,
        initial,
        positions,
        owner.smplx_joint_names,
        mapping,
        positions[:, root_index],
        tasks or {},
        priority,
        config=config,
        **common,
        callback=progress,
        extra_tasks=extra_tasks,
        endpoint_reference_trajectory=contact_reference,
        correction_reference_trajectory=correction_reference,
        contact_offset_choice=(
            None if contact_report is None else contact_report["contact_offset_choice"]
        ),
    )
    if config.get("staged_contact_repair", False):
        from ..staged_contact_repair import repair_contact_stages

        motion, repair_report = repair_contact_stages(
            model, motion, positions, owner.smplx_joint_names, mapping,
            positions[:, root_index], tasks or {}, priority,
            config=config, callback=progress, extra_tasks=extra_tasks,
            endpoint_reference_trajectory=motion if contact_reference is None else contact_reference,
            correction_reference_trajectory=correction_reference,
            contact_offset_choice=report["contact_offset_choice"],
            **common,
        )
        report["staged_contact_repair"] = repair_report
        for stage in repair_report["stages"]:
            if stage["accepted"]:
                for key in ("final_residuals", "posture_envelope", "contact_role_residuals"):
                    if key in stage["solver"]:
                        report[key] = stage["solver"][key]
    report["initialization"] = initialization_report
    report["contact_acquisition"] = contact_report
    from .sequence_guard import bind_sequence_guard

    bind_sequence_guard(
        owner,
        model,
        motion,
        motion if contact_reference is None else contact_reference,
        correction_reference,
        positions,
        owner.smplx_joint_names,
        mapping,
        positions[:, root_index],
        tasks or {},
        priority,
        config=config,
        extra_tasks=extra_tasks,
        contact_offset_choice=report["contact_offset_choice"],
        **{key: value for key, value in common.items() if key != "framerate"},
    )
    if config.get("staged_contact_repair", False):
        from .._sequence.contact_hold import ContactHoldAnchors, ContactHoldStage

        guard = owner._sequence_acceptance_guard
        if guard is not None:
            guard.contact_stage = ContactHoldStage(
                ContactHoldAnchors(guard, motion), None,
                min(0.0001, guard.allowance),
                continuity_reference=motion,
            ).bind(guard)
    owner.last_sequence_report = report
    owner.last_unprocessed_framewise_motion = motion.copy()
    return motion
