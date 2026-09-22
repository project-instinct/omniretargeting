"""Source preparation and framewise retargeting execution."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from ..contact import SceneContactTask
from ..local_collision_repair import diagnostics_json


class FramewiseMotionMixin:
    """Source preparation and framewise retargeting execution.

    Internal methods sharing the owning OmniRetargeter instance; no separate state.
    """

    def _perform_retargeting(
        self,
        processed_trajectory: np.ndarray,
        scaled_terrain: trimesh.Trimesh,
        base_orientations: np.ndarray | None = None,
        base_translations: np.ndarray | None = None,
        source_joint_orientations: np.ndarray | None = None,
        source_reference_joint_orientations: np.ndarray | None = None,
        source_reference_joint_positions: np.ndarray | None = None,
        end_effector_tasks: Optional[Dict[str, SceneContactTask]] = None,
        reference_trajectory: np.ndarray | None = None,
        overlay_contact_names: Optional[set[str]] = None,
        apply_surface_contacts: bool = True,
        object_points: np.ndarray | None = None,
    ) -> np.ndarray:
        """Perform the actual motion retargeting using generic interaction mesh retargeting."""
        from ..retargeting import GenericInteractionRetargeter

        if reference_trajectory is not None and len(reference_trajectory) != len(
            processed_trajectory
        ):
            raise ValueError(
                "reference_trajectory must have the same number of frames as "
                "processed_trajectory."
            )

        # Create retargeter instance
        # Note: scaled_terrain is already scaled by _scale_terrain_mesh
        # CRITICAL: Pass only valid_joint_mapping to ensure consistent sizes
        # This ensures robot_points and human_joints have the same number of joints
        preserve_root_z = bool(self.retargeting_config.get("preserve_root_z", False))
        lock_root_orientation = bool(
            self.retargeting_config.get("end_effector_priority", {}).get(
                "lock_root_orientation_to_source", False
            )
        )
        retargeter_kwargs: Dict[str, Any] = {}
        if end_effector_tasks or source_joint_orientations is not None:
            # Body-orientation tracking is a source capability, not a dataset
            # mode.  Whenever a source supplies global joint rotations, make
            # that fact available to the generic solver.  Robot profiles may
            # tune task weights/body overrides, but do not need a PARC- or
            # SMPL-X-specific enable switch.
            end_effector_config = dict(
                self.retargeting_config.get("end_effector_priority", {}) or {}
            )
            body_orientation_config = dict(
                end_effector_config.get("body_orientation", {}) or {}
            )
            body_orientation_config["_source_orientations_available"] = (
                source_joint_orientations is not None
            )
            end_effector_config["body_orientation"] = body_orientation_config
            retargeter_kwargs["end_effector_config"] = end_effector_config
        else:
            # Positions-only callers historically omit the priority config.
            # Route only explicitly opted-in experiments without activating
            # unrelated contact/orientation settings on the legacy path.
            priority = self.retargeting_config.get("end_effector_priority", {}) or {}
            experiment_config = {}
            if priority.get("pose_initialization", {}).get("enabled", False):
                experiment_config["pose_initialization"] = priority["pose_initialization"]
            if priority.get("interaction_mesh", {}).get("weighting", "uniform") != "uniform":
                experiment_config["interaction_mesh"] = {
                    key: priority["interaction_mesh"][key]
                    for key in ("weighting", "kappa") if key in priority["interaction_mesh"]
                }
            if experiment_config:
                retargeter_kwargs["end_effector_config"] = experiment_config
        if preserve_root_z:
            retargeter_kwargs["preserve_root_z"] = True
        # Leave constructor defaults implicit unless a profile actually opts in.
        # Besides preserving the historical call contract, this keeps third-party
        # GenericInteractionRetargeter-compatible wrappers source-compatible.
        if "self_collision_enabled" in self.retargeting_config:
            retargeter_kwargs["self_collision_enabled"] = bool(
                self.retargeting_config["self_collision_enabled"]
            )
        if "self_collision" in self.retargeting_config:
            retargeter_kwargs["self_collision"] = dict(self.retargeting_config["self_collision"])
        if "joint_limit_safety_fraction" in self.retargeting_config:
            retargeter_kwargs["joint_limit_safety_fraction"] = float(
                self.retargeting_config["joint_limit_safety_fraction"]
            )
        retargeter = GenericInteractionRetargeter(
            self.robot_model,
            self.robot_data,
            scaled_terrain,
            self.valid_joint_mapping,  # Use filtered mapping, not full joint_mapping
            self.robot_height,
            collision_detection_threshold=float(self.retargeting_config.get("collision_detection_threshold", 0.1)),
            terrain_sample_points=int(self.retargeting_config.get("terrain_sample_points", 100)),
            terrain_sample_seed=self.retargeting_config.get("terrain_sample_seed"),
            valid_joint_names=self.valid_joint_names,  # CRITICAL: Pass ordered joint names for consistency
            replace_cylinders_with_capsules=bool(self.retargeting_config.get("replace_cylinders_with_capsules", False)),
            hard_penetration_constraint=self.retargeting_config.get("penetration_resolver", "hard_constraint") == "hard_constraint",
            **retargeter_kwargs,
        )
        retargeter.scene_query = getattr(self, "scene_query", None)
        coupled_projection = bool(getattr(self, "_coupled_self_collision_projection", False))
        # Full trajectories can correct both endpoints before returning.
        # Standalone frame solves retain the stricter greedy warm-start guard.
        retargeter._defer_self_collision_path_repair = coupled_projection
        if source_joint_orientations is not None:
            reference_orientations = source_reference_joint_orientations
            reference_positions = source_reference_joint_positions
            reference_label = "source bind pose"
            if reference_orientations is None:
                reference_orientations = source_joint_orientations[0]
                reference_positions = processed_trajectory[0]
                reference_label = "first source frame"

            reference_base_orientation = None
            if reference_positions is not None:
                reference_base_xyzw = self._estimate_base_orientation_from_joints(
                    reference_positions
                )
                if reference_base_xyzw is not None:
                    reference_base_orientation = np.asarray(
                        [
                            reference_base_xyzw[3],
                            reference_base_xyzw[0],
                            reference_base_xyzw[1],
                            reference_base_xyzw[2],
                        ],
                        dtype=float,
                    )
            retargeter.configure_source_orientation_reference(
                self.smplx_joint_names,
                reference_orientations,
                reference_positions,
                reference_base_orientation=reference_base_orientation,
                reference_label=reference_label,
                preserve_core_reference=(
                    source_reference_joint_orientations is not None
                    and source_reference_joint_positions is not None
                ),
            )

        if self.retargeting_config.get("body_relations", {}).get("enabled", False):
            from ..body_relations import build_body_relation_tasks

            # Reuse the SAME fixed graph/targets for a contact overlay pass.
            # No calibration from an already retargeted (possibly wrong) frame.
            if getattr(self, "last_body_relation_tasks", None) is None:
                tasks, report = build_body_relation_tasks(
                    self.robot_model, self.retargeting_config["body_relations"],
                    self.valid_joint_mapping, self.retargeting_config.get("end_effector_priority", {}),
                    self.smplx_joint_names, processed_trajectory, source_joint_orientations,
                    source_reference_joint_positions, source_reference_joint_orientations,
                    retargeter.robot_reference_qpos,
                )
                self.last_body_relation_tasks, self.last_body_relation_diagnostics = tasks, report
                print("Body relations: " + diagnostics_json(report))
            retargeter.body_relation_tasks = self.last_body_relation_tasks

        from ..self_contact import build_self_contact_tasks
        if getattr(self, "last_self_contact_tasks", None) is None:
            tasks, report = build_self_contact_tasks(
                self.robot_model, getattr(self, "_body_surface_samples", None),
                self.valid_joint_mapping, getattr(retargeter, "robot_reference_qpos", None),
                getattr(self, "last_end_effector_tasks", {}),
                self.retargeting_config.get("self_contact", {}),
            )
            self.last_self_contact_tasks, self.last_self_contact_diagnostics = tasks, report
            print("Self contact: " + diagnostics_json(report))
        retargeter.self_contact_tasks = self.last_self_contact_tasks

        from ..surface_contact import build_surface_contact_tasks
        if apply_surface_contacts and not getattr(self, "_surface_contacts_built", False):
            tasks, report = build_surface_contact_tasks(
                self.robot_model, getattr(self, "_body_surface_samples", None),
                getattr(retargeter, "robot_reference_qpos", None), scaled_terrain,
                self.retargeting_config.get("surface_contact", {}),
                joint_mapping=self.valid_joint_mapping,
                framerate=getattr(self, "_source_framerate", 30.0),
                scene_objects=getattr(self, "_scene_objects", ()),
            )
            self.last_surface_contact_tasks, self.last_surface_contact_diagnostics = tasks, report
            self._surface_contacts_built = True
            print("Whole-body surface contact: " + diagnostics_json({
                k: report.get(k) for k in ("status", "reason", "contact_regions", "contact_samples", "active_frame_points")
            }))
        retargeter.surface_contact_tasks = (
            self.last_surface_contact_tasks if apply_surface_contacts else None
        )
        if (retargeter.surface_contact_tasks is not None
                and len(retargeter.surface_contact_tasks.targets) != len(processed_trajectory)):
            raise ValueError("Whole-body surface contacts must follow the original motion timeline")

        if self.retargeting_config.get("sequence_solver", {}).get("enabled", False):
            from .sequence import solve_core_sequence
            return solve_core_sequence(
                self, retargeter, processed_trajectory, end_effector_tasks,
                base_orientations, object_points, reference_trajectory,
            )

        # Retarget each frame
        framewise_max_iterations = self.retargeting_config.get("framewise_max_iterations")
        if framewise_max_iterations is not None and (
            not isinstance(framewise_max_iterations, int)
            or isinstance(framewise_max_iterations, bool)
            or framewise_max_iterations < 1
        ):
            raise ValueError("framewise_max_iterations must be a positive integer.")
        retargeted_trajectory = []
        frame_reports = []
        previous_effector_directions = {}
        self._last_estimated_quat = None

        # Start from the model's nominal configuration.  Joint-range midpoints
        # are not a neutral pose for humanoids (for G1 they bend the knees by
        # about 1.4 rad), which previously forced the floating base downward to
        # compensate for artificially shortened legs.
        try:
            q_init = np.asarray(self.robot_model.qpos0, dtype=float).copy()
            if q_init.shape != (int(self.robot_model.nq),):
                raise ValueError("invalid model qpos0 shape")
        except (TypeError, ValueError):
            # Lightweight unit-test models and third-party wrappers may not
            # expose qpos0.  Preserve the historical zero initialization for
            # those callers while real MuJoCo models use their neutral pose.
            q_init = np.zeros(int(self.robot_model.nq), dtype=float)
        if len(q_init) >= 7 and np.linalg.norm(q_init[3:7]) < 1e-8:
            q_init[3:7] = [1, 0, 0, 0]  # Identity quaternion (wxyz)
        root_position_idx = self._resolve_source_root_index()
        base_position_cfg = dict(
            self.retargeting_config.get("end_effector_priority", {}).get(
                "base_position_constraint", {}
            )
            or {}
        )
        base_position_enabled = bool(base_position_cfg.get("enabled", False))
        use_initial_base_offset = bool(
            base_position_cfg.get("use_initial_offset", False)
        )
        lock_initial_base_xy = bool(
            base_position_cfg.get("lock_initial_xy", False)
        )
        source_to_robot_base_offset: np.ndarray | None = None
        if (
            reference_trajectory is not None
            and len(reference_trajectory)
            and base_position_enabled
            and use_initial_base_offset
        ):
            source_to_robot_base_offset = (
                np.asarray(reference_trajectory[0, :3], dtype=float)
                - processed_trajectory[0, root_position_idx]
            )

        # Initial height: Use processed (scaled) trajectory root height or default
        # Assuming joint 0 is root
        if len(processed_trajectory) > 0:
            q_init[:3] = processed_trajectory[0, root_position_idx] # Start at first frame root pos
            # Maintain initial rotation (identity or from trajectory if available)
        else:
            q_init[2] = self.robot_height * 0.5  # Default height (Z coordinate)

        for frame_idx, human_joints in enumerate(processed_trajectory):
            retargeter._surface_overlay_frame = None
            q_last = (
                retargeted_trajectory[-1] if retargeted_trajectory else None
            )
            if reference_trajectory is not None:
                contact_active = any(
                    name in end_effector_tasks
                    and bool(end_effector_tasks[name].contact_mask[frame_idx])
                    for name in (overlay_contact_names or set())
                )
                surface_active = (
                    not contact_active
                    and retargeter.surface_contact_tasks is not None
                    and bool(np.any(retargeter.surface_contact_tasks.active[frame_idx]))
                )
                if not contact_active and surface_active:
                    # A released hand does not release the actor's other skin
                    # contacts. Solve their articulated chains with the same
                    # fixed robot base instead of silently copying the baseline.
                    retargeter._surface_overlay_frame = frame_idx
                if not contact_active and not surface_active:
                    q_opt = np.asarray(
                        reference_trajectory[frame_idx], dtype=float
                    ).copy()
                    if (
                        base_position_enabled
                        and use_initial_base_offset
                        and source_to_robot_base_offset is None
                    ):
                        source_to_robot_base_offset = q_opt[:3] - human_joints[
                            root_position_idx
                        ]
                    checked = (q_opt if coupled_projection else
                               retargeter._self_collision_initial_guess(q_opt, q_last))
                    if np.array_equal(checked, q_opt):
                        retargeted_trajectory.append(q_opt)
                        q_init = q_opt.copy()
                        continue
                    # A contact-release splice also needs a current-frame
                    # solve when its path crosses a collision body.
                # Contact refinement is an overlay on the independently
                # solved source-faithful trajectory, not a new warm-start path.
                q_init = np.asarray(
                    reference_trajectory[frame_idx], dtype=float
                ).copy()
            # An overlay starts from the solved robot baseline, including its
            # base. Reinitializing from the actor here would discard that fit
            # while local contact constraints freeze the overwritten base.
            if frame_idx == 0 and reference_trajectory is None:
                # Initialize base position from SMPLX root
                if base_translations is not None:
                    q_init[:3] = base_translations[frame_idx]
                else:
                    # Use the configured root position as initial guess
                    q_init[:3] = human_joints[root_position_idx]

                if (
                    base_position_enabled
                    and lock_initial_base_xy
                    and bool(base_position_cfg.get("track_xy", True))
                ):
                    # A model translation is not necessarily its anatomical
                    # root position (SMPL-X includes a shaped joint offset).
                    # Start inside the requested registration constraint;
                    # otherwise the first trust region can be infeasible and
                    # its failed pose becomes a spurious permanent offset.
                    q_init[:2] = human_joints[root_position_idx, :2]

                # Initialize base orientation
                if base_orientations is not None:
                    source_orientation = np.asarray(
                        base_orientations[frame_idx], dtype=float
                    )
                    if source_orientation.shape == (4,):
                        norm = float(np.linalg.norm(source_orientation))
                        if norm < 1e-12:
                            raise ValueError(
                                "base_orientations contains a zero-length quaternion."
                            )
                        q_init[3:7] = source_orientation / norm
                    elif source_orientation.shape == (3,):
                        quat_xyzw = Rotation.from_rotvec(
                            source_orientation
                        ).as_quat()
                        q_init[3:7] = np.array(
                            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
                        )
                    else:
                        raise ValueError(
                            "base_orientations must contain wxyz quaternions or "
                            "axis-angle vectors; got per-frame shape "
                            f"{source_orientation.shape}."
                        )
                else:
                    quat_xyzw = self._estimate_base_orientation_from_joints(human_joints)
                    if quat_xyzw is not None:
                        q_init[3:7] = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            if (
                frame_idx > 0
                and reference_trajectory is None
                and self.retargeting_config.get("end_effector_priority", {}).get(
                    "source_root_prediction", False
                )
            ):
                predict_source_root = bool(
                    self.retargeting_config.get("end_effector_priority", {}).get(
                        "source_root_prediction", False
                    )
                )
                if predict_source_root:
                    q_init[:3] += (
                        human_joints[root_position_idx]
                        - processed_trajectory[frame_idx - 1, root_position_idx]
                    )
                if predict_source_root and base_orientations is not None:
                    # Predict the root's deliberate source rotation while
                    # retaining the previous tracking error. The optimizer
                    # still owns the result; this is not an orientation lock.
                    # q_last remains the actual prior output for all temporal
                    # and contact constraints.
                    current_source = Rotation.from_quat(
                        base_orientations[frame_idx], scalar_first=True
                    )
                    previous_source = Rotation.from_quat(
                        base_orientations[frame_idx - 1], scalar_first=True
                    )
                    previous_robot = Rotation.from_quat(
                        q_last[3:7], scalar_first=True
                    )
                    q_init[3:7] = (
                        current_source * previous_source.inv() * previous_robot
                    ).as_quat(scalar_first=True)

            if (
                frame_idx > 0
                and reference_trajectory is None
                and base_position_enabled
                and bool(base_position_cfg.get("track_xy", True))
                and bool(base_position_cfg.get("lock_xy", False))
            ):
                # Start on the prescribed affine path even when prediction is
                # disabled. Every bounded/backtracked increment then preserves
                # the XY equality. Shortening a step from the previous frame's
                # root would accumulate lag and conflict with temporal bounds.
                q_init[:2] = human_joints[root_position_idx, :2]
                if source_to_robot_base_offset is not None:
                    q_init[:2] += source_to_robot_base_offset[:2]

            # A configured hard lock remains distinct from the predictor.
            if lock_root_orientation and base_orientations is not None:
                source_root_quaternion = np.asarray(
                    base_orientations[frame_idx], dtype=float
                ).reshape(4).copy()
                source_root_quaternion /= max(
                    float(np.linalg.norm(source_root_quaternion)), 1e-12
                )
                if q_last is not None and np.dot(
                    source_root_quaternion, q_last[3:7]
                ) < 0.0:
                    source_root_quaternion *= -1.0
                q_init[3:7] = source_root_quaternion
            if preserve_root_z:
                if base_translations is not None:
                    q_init[2] = base_translations[frame_idx, 2]
                else:
                    q_init[2] = human_joints[root_position_idx, 2]

            # Extract mapped joints using the proper indices
            # CRITICAL: The order MUST match valid_joint_names exactly
            # mapped_joint_indices[i] corresponds to valid_joint_names[i] for all i
            # human_joints shape: (num_smplx_joints, 3)

            # Validate order consistency before extraction
            if len(self.mapped_joint_indices) != len(self.valid_joint_names):
                raise ValueError(
                    f"Order mismatch: mapped_joint_indices has {len(self.mapped_joint_indices)} elements, "
                    f"but valid_joint_names has {len(self.valid_joint_names)} elements."
                )

            # Verify that each index corresponds to the correct joint name
            for i, (joint_name, smplx_idx) in enumerate(zip(self.valid_joint_names, self.mapped_joint_indices)):
                expected_idx = self.smplx_joint_indices.get(joint_name)
                if expected_idx != smplx_idx:
                    raise ValueError(
                        f"Order mismatch at position {i}: joint '{joint_name}' has index {smplx_idx} "
                        f"in mapped_joint_indices, but smplx_joint_indices says it should be {expected_idx}."
                    )

            mapped_joints = human_joints[self.mapped_joint_indices]
            # mapped_joints shape: (num_mapped_joints, 3)
            # mapped_joints[i] corresponds to valid_joint_names[i] for all i

            # CRITICAL: Validate that we extracted the expected number of joints
            expected_num_joints = len(self.valid_joint_names)
            if len(mapped_joints) != expected_num_joints:
                raise ValueError(
                    f"Size mismatch: extracted {len(mapped_joints)} joints from human_joints, "
                    f"but expected {expected_num_joints} joints from valid_joint_names. "
                    f"This indicates an inconsistency in joint mapping."
                )

            # Debug: Print order and actual positions for first frame to verify matching
            import sys
            if not hasattr(sys, '_omni_human_joint_order_printed'):
                print(f"\n=== Human Joint Extraction Order ===")
                print(f"valid_joint_names: {self.valid_joint_names}")
                print(f"mapped_joint_indices: {self.mapped_joint_indices}")
                print(f"Extracted {len(mapped_joints)} joints in order: {self.valid_joint_names}")
                print(f"\n=== First Frame Joint Positions (to verify order) ===")
                for i, (name, idx) in enumerate(zip(self.valid_joint_names, self.mapped_joint_indices)):
                    pos = mapped_joints[i] if frame_idx == 0 else mapped_joints[i]
                    if frame_idx == 0:
                        print(f"  {i}: {name} (SMPLX idx {idx}) -> pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                # Specifically check knee vs ankle positions
                if frame_idx == 0:
                    l_knee_idx = self.valid_joint_names.index("L_Knee") if "L_Knee" in self.valid_joint_names else None
                    l_ankle_idx = self.valid_joint_names.index("L_Ankle") if "L_Ankle" in self.valid_joint_names else None
                    if l_knee_idx is not None and l_ankle_idx is not None:
                        l_knee_pos = mapped_joints[l_knee_idx]
                        l_ankle_pos = mapped_joints[l_ankle_idx]
                        print(f"\n=== Knee vs Ankle Position Check ===")
                        print(f"L_Knee (idx {l_knee_idx}, SMPLX {self.mapped_joint_indices[l_knee_idx]}): [{l_knee_pos[0]:.3f}, {l_knee_pos[1]:.3f}, {l_knee_pos[2]:.3f}]")
                        print(f"L_Ankle (idx {l_ankle_idx}, SMPLX {self.mapped_joint_indices[l_ankle_idx]}): [{l_ankle_pos[0]:.3f}, {l_ankle_pos[1]:.3f}, {l_ankle_pos[2]:.3f}]")
                        print(f"Knee should be HIGHER than ankle (knee.z > ankle.z): {l_knee_pos[2] > l_ankle_pos[2]}")
                sys._omni_human_joint_order_printed = True

            # Estimate target base orientation from human joints
            last_target_quat = None
            if frame_idx > 0 and len(retargeted_trajectory) > 0:
                # Get last orientation from previous frame result or previous target estimate
                # We use the result (wxyz) converted back to xyzw for continuity check
                # But wait, retargeted_trajectory has wxyz.
                # _estimate returns xyzw.
                # Better to store the previous estimated xyzw.
                pass

            # We need to maintain state of estimated quaternion to ensure continuity
            if not hasattr(self, '_last_estimated_quat'):
                self._last_estimated_quat = None

            target_quat_wxyz = None
            if base_orientations is not None:
                target_quat_wxyz = np.asarray(
                    base_orientations[frame_idx], dtype=float
                ).reshape(4).copy()
                target_quat_wxyz /= max(
                    float(np.linalg.norm(target_quat_wxyz)), 1e-12
                )
            else:
                target_quat_xyzw = self._estimate_base_orientation_from_joints(
                    human_joints, self._last_estimated_quat
                )
                self._last_estimated_quat = target_quat_xyzw
                if target_quat_xyzw is not None:
                    # Convert xyzw to wxyz for MuJoCo.
                    target_quat_wxyz = np.array(
                        [
                            target_quat_xyzw[3],
                            target_quat_xyzw[0],
                            target_quat_xyzw[1],
                            target_quat_xyzw[2],
                        ]
                    )

            # Retarget frame with previous frame as reference for smoothness
            frame_kwargs: Dict[str, Any] = dict(
                q_last=q_last,
                target_base_orientation=target_quat_wxyz,
            )
            if object_points is not None:
                frame_kwargs["object_points"] = object_points[frame_idx]
            if base_orientations is not None and frame_idx > 0:
                previous_target_quaternion = np.asarray(
                    base_orientations[frame_idx - 1], dtype=float
                ).reshape(4).copy()
                previous_target_quaternion /= max(
                    float(np.linalg.norm(previous_target_quaternion)), 1e-12
                )
                frame_kwargs["previous_target_base_orientation"] = (
                    previous_target_quaternion
                )
            if source_joint_orientations is not None:
                frame_kwargs["source_joint_orientations"] = (
                    source_joint_orientations[frame_idx]
                )
            if end_effector_tasks:
                if base_position_enabled and (
                    not use_initial_base_offset
                    or source_to_robot_base_offset is not None
                    or (frame_idx == 0 and lock_initial_base_xy)
                ):
                    target_base_position = np.asarray(
                        human_joints[root_position_idx], dtype=float
                    ).copy()
                    if source_to_robot_base_offset is not None:
                        target_base_position[:2] += source_to_robot_base_offset[:2]
                        if bool(base_position_cfg.get("track_z", True)):
                            target_base_position[2] += source_to_robot_base_offset[2]
                    frame_kwargs["target_base_position"] = target_base_position
                frame_kwargs["end_effector_targets"] = {}
                for name, task in end_effector_tasks.items():
                    target_payload = dict(task.frame(frame_idx))
                    directions = {}
                    for axis in ("forward_direction", "up_direction"):
                        value = target_payload.get(axis)
                        if value is not None:
                            directions[axis] = np.asarray(value, dtype=float).copy()
                        previous = previous_effector_directions.get(name, {}).get(axis)
                        if previous is not None:
                            target_payload["previous_" + axis] = previous.copy()
                    # Capture the chronological source plan before solving. Do
                    # not rerun contact inference or read a robot-derived pose.
                    previous_effector_directions[name] = directions
                    target_payload["source_root_position"] = human_joints[
                        root_position_idx
                    ]
                    frame_kwargs["end_effector_targets"][name] = target_payload
                frame_kwargs["q_prev"] = (
                    retargeted_trajectory[-2]
                    if len(retargeted_trajectory) > 1
                    else None
                )
                frame_kwargs["q_prevprev"] = (
                    retargeted_trajectory[-3]
                    if len(retargeted_trajectory) > 2
                    else None
                )
                frame_kwargs["previous_human_joints"] = (
                    processed_trajectory[frame_idx - 1, self.mapped_joint_indices]
                    if frame_idx > 0
                    else None
                )
                frame_kwargs["previous_previous_human_joints"] = (
                    processed_trajectory[frame_idx - 2, self.mapped_joint_indices]
                    if frame_idx > 1
                    else None
                )
            if (getattr(self, "last_body_relation_tasks", None) is not None
                    or getattr(self, "last_self_contact_tasks", None) is not None
                    or getattr(self, "last_surface_contact_tasks", None) is not None):
                frame_kwargs["relation_frame"] = frame_idx
            if framewise_max_iterations is not None:
                frame_kwargs["max_iter"] = framewise_max_iterations
            q_opt = retargeter.retarget_frame(mapped_joints, q_init, **frame_kwargs)
            report = getattr(retargeter, "last_frame_solve", None)
            if isinstance(report, dict):
                frame_reports.append(dict(frame=frame_idx, **report))
            if reference_trajectory is not None and coupled_projection:
                weight = max(
                    (float(end_effector_tasks[name].transition_weights[frame_idx])
                     if end_effector_tasks[name].transition_weights is not None else 1.0
                     for name in (overlay_contact_names or set())
                     if name in end_effector_tasks and end_effector_tasks[name].contact_mask[frame_idx]),
                    default=1.0,
                )
                baseline = np.asarray(reference_trajectory[frame_idx], dtype=float)
                overlay = np.asarray(q_opt, dtype=float).copy()
                if np.dot(overlay[3:7], baseline[3:7]) < 0:
                    overlay[3:7] *= -1
                q_opt = baseline + weight * (overlay - baseline)
                q_opt[3:7] /= max(float(np.linalg.norm(q_opt[3:7])), 1e-12)
                # This is a proposal. The coupled projection below checks
                # both blended poses and every playback connection.
            if lock_root_orientation and base_orientations is not None:
                q_opt[3:7] = q_init[3:7]
            if preserve_root_z:
                q_opt[2] = q_init[2]
            if (
                base_position_enabled
                and use_initial_base_offset
                and source_to_robot_base_offset is None
            ):
                # Animation retargeters calibrate a pelvis/root offset once and
                # then transfer the source root path relative to that aligned
                # pose.  This preserves morphology and initial registration
                # without letting framewise IK accumulate horizontal drift.
                source_to_robot_base_offset = q_opt[:3] - human_joints[
                    root_position_idx
                ]
            retargeted_trajectory.append(q_opt)

            # Update initial guess for next frame (smooth trajectory)
            q_init = q_opt.copy()

        self.last_frame_solve_reports = frame_reports
        if isinstance(getattr(self, "last_motion_solve_reports", None), list):
            self.last_motion_solve_reports.append(dict(
                pass_kind="contact_overlay" if reference_trajectory is not None else "baseline",
                surface_contact_enabled=retargeter.surface_contact_tasks is not None,
                frames=frame_reports,
            ))
        fallbacks = [r["initialization"]["reason"] for r in frame_reports
                     if r["initialization"].get("status") == "fallback"]
        if fallbacks:
            from collections import Counter
            print("Pose initialization fallbacks: " + diagnostics_json(dict(Counter(fallbacks))))
        motion = np.array(retargeted_trajectory)
        self.last_unprocessed_framewise_motion = motion.copy()
        if coupled_projection and getattr(retargeter, "self_collision_enabled", False) is True:
            from ..self_collision_projection import project_self_collision_trajectory

            priority = self.retargeting_config.get("end_effector_priority", {})
            tasks, priority, _ = self._validation_foot_references(
                motion, end_effector_tasks or {}, priority,
            )
            motion, report = project_self_collision_trajectory(
                self.robot_model, motion, scaled_terrain, tasks, priority,
                self._self_collision_projection_config,
                scene_query=getattr(self, "scene_query", None),
                framerate=self._projection_framerate,
                body_relation_tasks=getattr(self, "last_body_relation_tasks", None),
                self_contact_tasks=getattr(self, "last_self_contact_tasks", None),
                surface_contact_tasks=retargeter.surface_contact_tasks,
            )
            report["pass"] = "contact_overlay" if reference_trajectory is not None else "baseline"
            self.last_framewise_self_collision_reports.append(report)
            print("Framewise self-collision projection: " + diagnostics_json({
                key: report[key] for key in ("pass", "status", "before_max_depth", "after_max_depth")
            }))
        return motion


    def _estimate_base_orientation_from_joints(self, joints: np.ndarray, last_quat: np.ndarray | None = None) -> np.ndarray | None:
        """
        Estimate a base orientation from joint positions.

        Uses pelvis/hips/spine to build an approximate body frame:
        - up: pelvis -> spine1
        - right: left_hip -> right_hip
        - forward: right x up
        Returns quaternion in xyzw order.

        Args:
            joints: Joint positions array.
            last_quat: Quaternion from previous frame to ensure continuity (xyzw).
        """
        if joints.shape[0] < 4:
            return None

        pelvis_name = self.base_orientation_config.get("pelvis", "Pelvis")
        left_hip_name = self.base_orientation_config.get("left_hip", "L_Hip")
        right_hip_name = self.base_orientation_config.get("right_hip", "R_Hip")
        spine_name = self.base_orientation_config.get("spine", "Spine1")

        # A configured name that is missing from the source layout used to fall
        # back to a positional index, which silently built the body frame out of
        # whichever joints happened to sit at 0/1/2/3.  This frame is what the
        # whole upper body tracks, so a typo here has to be loud.
        frame_indices = {}
        for key, name in (
            ("pelvis", pelvis_name),
            ("left_hip", left_hip_name),
            ("right_hip", right_hip_name),
            ("spine", spine_name),
        ):
            if name not in self.smplx_joint_indices:
                raise ValueError(
                    f"base_orientation.{key} refers to '{name}', which is not in "
                    f"the source joint layout "
                    f"{list(self.smplx_joint_indices)[:8]}..."
                )
            frame_indices[key] = self.smplx_joint_indices[name]
        pelvis_idx = frame_indices["pelvis"]
        left_hip_idx = frame_indices["left_hip"]
        right_hip_idx = frame_indices["right_hip"]
        spine_idx = frame_indices["spine"]

        max_required_idx = max(pelvis_idx, left_hip_idx, right_hip_idx, spine_idx)
        if joints.shape[0] <= max_required_idx:
            return None

        pelvis = joints[pelvis_idx]
        left_hip = joints[left_hip_idx]
        right_hip = joints[right_hip_idx]
        spine1 = joints[spine_idx]

        up = spine1 - pelvis
        right = right_hip - left_hip

        up_norm = np.linalg.norm(up)
        right_norm = np.linalg.norm(right)
        if up_norm < 1e-8 or right_norm < 1e-8:
            return None

        up = up / up_norm
        right = right / right_norm

        # Build forward using right-hand rule
        forward = np.cross(up, right)
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-8:
            return None
        forward = forward / forward_norm

        # Re-orthogonalize right to ensure perfect orthogonality
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)

        # Rotation matrix with body axes in world coordinates.
        # Convention: X=forward, Y=right, Z=up
        rot = np.column_stack([forward, right, up])

        # Verify right-handed
        det = np.linalg.det(rot)
        if det < 0:
            forward = -forward
            right = -right
            rot = np.column_stack([forward, right, up])

        current_quat = Rotation.from_matrix(rot).as_quat() # xyzw

        # Ensure continuity if last quaternion is provided
        if last_quat is not None:
            # Check dot product to see if we need to flip sign
            dot = np.dot(current_quat, last_quat)
            if dot < 0:
                current_quat = -current_quat

        return current_quat


    def _resolve_source_root_index(self) -> int:
        """Resolve the highest available source torso joint for robot-root use."""
        indices = getattr(self, "smplx_joint_indices", {})
        if not indices:
            return 0
        requested = str(
            self.base_orientation_config.get("root_position", "Pelvis")
        )
        if requested in indices:
            return int(indices[requested])

        casefold_indices = {
            str(name).casefold(): int(idx) for name, idx in indices.items()
        }
        requested_lower = requested.casefold()
        if "spine" in requested_lower or any(
            token in requested_lower for token in ("chest", "torso", "neck")
        ):
            candidates = (
                "Spine3",
                "Spine2",
                "Spine1",
                "Chest",
                "Torso",
                "Pelvis",
            )
        else:
            candidates = (requested, "Pelvis", "Spine1")
        for candidate in candidates:
            match = casefold_indices.get(candidate.casefold())
            if match is not None:
                return int(match)
        return 0
