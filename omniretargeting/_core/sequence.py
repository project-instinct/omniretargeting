"""Prepare the generic sequence solver from existing core correspondences."""

import numpy as np
from scipy.spatial.transform import Rotation

from ..sequence_retargeting import retarget_sequence


def solve_core_sequence(
    owner,
    retargeter,
    positions,
    tasks,
    base_orientations,
    object_points,
    reference_trajectory=None,
):
    model = owner.robot_model
    frames = len(positions)
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
    if reference_trajectory is None:
        seed = getattr(retargeter, "robot_reference_qpos", None)
        initial = np.tile(model.qpos0 if seed is None else seed, (frames, 1))
        from .._sequence.coordinates import Coordinates

        coordinates = Coordinates(model)
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
    else:
        initial = np.asarray(reference_trajectory, dtype=float).copy()
    terrain = np.broadcast_to(
        retargeter.terrain_points, (frames, *retargeter.terrain_points.shape)
    )
    environment = (
        terrain
        if object_points is None
        else np.concatenate([terrain, object_points], axis=1)
    )

    def progress(_pose, record):
        if record["iteration"] % 5 == 0 or record["alpha"] == 0:
            print(
                f"Sequence {record['phase']} {record['iteration']}: "
                f"cost={record['after']:.3f}, contact={record['sticking_max_m']:.5f} m",
                flush=True,
            )

    common = dict(
        framerate=owner._source_framerate,
        environment_points=environment,
        scene_query=retargeter.scene_query,
    )
    initialization_report = None
    if reference_trajectory is None:
        # A neutral/calibrated pose is not an IK solution. First acquire the
        # source body/feet configuration before tightening interaction anchors;
        # otherwise strong hand contacts can trap the legs on a folded branch.
        # These are global continuation weights, never clip/frame overrides.
        bootstrap_config = dict(config)
        bootstrap_config.update(
            contact_sigma_m=0.05, contact_normal_sigma_m=0.05, source_side_support=True
        )
        initial, initialization_report = retarget_sequence(
            model,
            initial,
            positions,
            owner.smplx_joint_names,
            mapping,
            positions[:, root_index],
            tasks or {},
            priority,
            config=bootstrap_config,
            callback=progress,
            **common,
        )

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
        extra_tasks=(
            getattr(owner, "last_body_relation_tasks", None),
            getattr(owner, "last_self_contact_tasks", None),
            getattr(owner, "last_surface_contact_tasks", None),
        ),
    )
    report["initialization"] = initialization_report
    owner.last_sequence_report = report
    owner.last_unprocessed_framewise_motion = motion.copy()
    return motion
