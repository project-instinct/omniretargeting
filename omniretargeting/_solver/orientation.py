"""Source bind calibration and body orientation residuals."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import cvxpy as cp
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from ..utils import calculate_laplacian_coordinates
from .targets import (
    _calibrated_orientation_target,
)


class BodyOrientationMixin:
    """Source bind calibration and body orientation residuals.

    Internal methods sharing the owning GenericInteractionRetargeter instance; no separate state.
    """

    def _setup_body_orientations(self) -> None:
        """Resolve generic source-joint to robot-body orientation tasks.

        Orientation tracking turns on automatically when the source contract
        supplies global joint rotations.  The existing positional
        ``joint_mapping`` is also the default orientation mapping, so this is
        not a PARC-only feature and a new source does not need its own solver
        branch.  Robot profiles may override a mapped body, sigma, or weight
        for morphology-specific link conventions.
        """
        cfg = dict(self.end_effector_config.get("body_orientation", {}) or {})
        source_orientations_available = bool(
            cfg.get("_source_orientations_available", False)
        )
        explicitly_disabled = cfg.get("enabled") is False
        self.body_orientation_enabled = (
            source_orientations_available and not explicitly_disabled
        )
        self.body_orientation_calibration_iterations = max(
            int(cfg.get("calibration_iterations", 30)), 1
        )
        # A positional fit cannot observe rotation about a bone's own axis, so
        # a solved bind pose leaves that twist wherever the optimizer stopped,
        # independently per side. A declared pose fixes it and is mirrorable.
        self.body_orientation_reference_pose = dict(
            cfg.get("reference_pose", {}) or {}
        )
        self.body_orientation_tasks: Dict[str, Dict[str, Any]] = {}
        self._source_orientation_indices: Dict[str, int] = {}
        self._source_orientation_references: Dict[str, np.ndarray] = {}
        self._robot_orientation_references: Dict[str, np.ndarray] = {}
        self.robot_reference_qpos = None
        self.body_relation_tasks = None
        self.self_contact_tasks = None
        self.surface_contact_tasks = None
        self.surface_pose_tasks = None
        if not self.body_orientation_enabled:
            return

        default_sigma = max(float(cfg.get("sigma", 0.2)), 1e-6)
        configured_tasks = dict(cfg.get("tasks", {}) or {})
        base_orientation_source_joint = self.end_effector_config.get(
            "_base_orientation_source_joint"
        )
        task_specs: Dict[str, Any] = {
            source_name: {"robot_body": robot_body}
            for source_name, robot_body in self.joint_mapping.items()
            if source_name != base_orientation_source_joint
        }
        for source_name, configured_spec in configured_tasks.items():
            if source_name == base_orientation_source_joint:
                continue
            inherited_spec = task_specs.get(source_name, {})
            if isinstance(configured_spec, str):
                task_specs[source_name] = {"robot_body": configured_spec}
            else:
                task_specs[source_name] = {
                    **dict(inherited_spec or {}),
                    **dict(configured_spec or {}),
                }

        for source_name, raw_spec in task_specs.items():
            spec = (
                {"robot_body": raw_spec}
                if isinstance(raw_spec, str)
                else dict(raw_spec or {})
            )
            if spec.get("enabled") is False:
                continue
            robot_body = str(
                spec.get("robot_body", self.joint_mapping.get(source_name, ""))
            )
            body_id = mujoco.mj_name2id(
                self.robot_model,
                mujoco.mjtObj.mjOBJ_BODY,
                robot_body,
            )
            if body_id < 0:
                raise ValueError(
                    f"body_orientation task '{source_name}' refers to missing "
                    f"robot body '{robot_body}'."
                )
            weight = max(float(spec.get("weight", 1.0)), 0.0)
            if source_name == self.end_effector_config.get("_position_anchor_source_joint"):
                # When the floating body is the chest, also preserve the
                # articulated pelvis/torso relationship. Resolve the anchor
                # through the source contract, never a clip or body name.
                weight *= max(float(cfg.get("position_anchor_weight", 1.0)), 0.0)
            if weight == 0.0:
                continue
            self.body_orientation_tasks[str(source_name)] = {
                "body_id": int(body_id),
                "body_name": robot_body,
                "sigma": max(float(spec.get("sigma", default_sigma)), 1e-6),
                "weight": weight,
            }

        if not self.body_orientation_tasks:
            self.body_orientation_enabled = False

        frame = str(cfg.get("frame", "world"))
        if frame not in {"world", "parent_relative"}:
            raise ValueError("body_orientation.frame must be 'world' or 'parent_relative'.")
        if frame == "parent_relative":
            # Several source landmarks can map to one rigid robot segment.
            # They cannot independently control its orientation. Prefer the
            # explicitly declared position anchor, otherwise the proximal
            # mapped body; all positional tasks remain intact.
            rigid_groups = {}
            for name, task in self.body_orientation_tasks.items():
                rigid_groups.setdefault(int(self.robot_model.body_weldid[task["body_id"]]), []).append(name)
            anchor = self.end_effector_config.get("_position_anchor_source_joint")
            for group in rigid_groups.values():
                if len(group) < 2:
                    continue
                if anchor in group:
                    keep = anchor
                else:
                    depths = {}
                    for name in group:
                        body, depth = self.body_orientation_tasks[name]["body_id"], 0
                        while body > 0:
                            depth += 1
                            body = int(self.robot_model.body_parentid[body])
                        depths[name] = depth
                    nearest = [name for name in group if depths[name] == min(depths.values())]
                    if len(nearest) != 1:
                        raise ValueError(f"Rigid segment has ambiguous source orientations {group}; declare one enabled orientation task.")
                    keep = nearest[0]
                for name in group:
                    if name != keep:
                        self.body_orientation_tasks.pop(name)
                print(f"Rigid orientation group {group}: using '{keep}', retaining other position mappings.")
            # Position mappings need not expose every intermediate joint. Find
            # the nearest *mapped* ancestor in the actual compiled body tree.
            # Include the floating body even if its duplicate task was removed.
            orientation_mapping = {name: spec["body_name"] for name, spec in self.body_orientation_tasks.items()}
            if base_orientation_source_joint in self.joint_mapping:
                orientation_mapping[base_orientation_source_joint] = self.joint_mapping[base_orientation_source_joint]
            by_body = {}
            for name, body_name in orientation_mapping.items():
                body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if body_id >= 0:
                    by_body.setdefault(int(body_id), []).append(name)
            for task in self.body_orientation_tasks.values():
                parent = int(self.robot_model.body_parentid[task["body_id"]])
                while parent > 0 and parent not in by_body:
                    parent = int(self.robot_model.body_parentid[parent])
                if parent in by_body:
                    if len(by_body[parent]) != 1:
                        raise ValueError(f"Ambiguous orientation parent on robot body {parent}: {by_body[parent]}")
                    task["parent_source"] = by_body[parent][0]
                    task["parent_body_id"] = parent


    def configure_source_orientation_reference(
        self,
        source_joint_names: List[str],
        source_reference_orientations: Optional[np.ndarray],
        source_reference_positions: Optional[np.ndarray] = None,
        *,
        reference_base_orientation: Optional[np.ndarray] = None,
        reference_label: str = "source reference pose",
        preserve_core_reference: bool = False,
    ) -> None:
        """Calibrate source joint frames against a morphology-matched pose."""
        if not self.body_orientation_enabled:
            return
        if source_reference_orientations is None:
            print(
                "Body-orientation tracking requested, but the source provides "
                "no joint_rest_orientations; continuing with position tasks."
            )
            return

        references = np.asarray(source_reference_orientations, dtype=float)
        if references.shape != (len(source_joint_names), 4):
            raise ValueError(
                "source_reference_orientations must have shape "
                f"({len(source_joint_names)}, 4), got {references.shape}."
            )
        name_to_index = {
            str(name): index for index, name in enumerate(source_joint_names)
        }
        reference_bodies = {name: task["body_id"] for name, task in self.body_orientation_tasks.items()}
        reference_bodies.update({task["parent_source"]: task["parent_body_id"]
                                 for task in self.body_orientation_tasks.values() if "parent_source" in task})
        missing = [name for name in reference_bodies if name not in name_to_index]
        if missing:
            raise ValueError(
                "body_orientation source joints are missing from the motion: "
                f"{missing}."
            )

        current_qpos = self.robot_data.qpos.copy()
        robot_reference_qpos = np.asarray(
            self.robot_model.qpos0, dtype=float
        ).copy()
        if source_reference_positions is not None:
            reference_positions = np.asarray(
                source_reference_positions, dtype=float
            )
            if reference_positions.shape != (len(source_joint_names), 3):
                raise ValueError(
                    "source_reference_positions must have shape "
                    f"({len(source_joint_names)}, 3), got "
                    f"{reference_positions.shape}."
                )
            mapped_reference_positions = np.asarray(
                [reference_positions[name_to_index[name]] for name in self.valid_joint_names],
                dtype=float,
            )
            root_name = "Pelvis" if "Pelvis" in name_to_index else source_joint_names[0]
            free_joint_ids = np.flatnonzero(
                self.robot_model.jnt_type == mujoco.mjtJoint.mjJNT_FREE
            )
            if len(free_joint_ids):
                free_body_id = int(
                    self.robot_model.jnt_bodyid[int(free_joint_ids[0])]
                )
                for source_name, robot_body_name in self.joint_mapping.items():
                    robot_body_id = mujoco.mj_name2id(
                        self.robot_model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        str(robot_body_name),
                    )
                    if robot_body_id == free_body_id and source_name in name_to_index:
                        root_name = source_name
                        break
            if len(robot_reference_qpos) >= 3:
                robot_reference_qpos[:3] = reference_positions[
                    name_to_index[root_name]
                ]
            if not self.body_orientation_reference_pose:
                # Reference-pose calibration is independent of the clip's scene.
                # Use a complete graph of reference joints, without environment
                # vertices or scene constraints. This also works for coplanar rest
                # poses where a tetrahedralization would be degenerate.
                adjacency = [
                    [j for j in range(len(mapped_reference_positions)) if j != i]
                    for i in range(len(mapped_reference_positions))
                ]
                reference_laplacian = calculate_laplacian_coordinates(
                    mapped_reference_positions, adjacency, uniform_weight=True
                )
                collision_policy = self.hard_penetration_constraint
                try:
                    self.hard_penetration_constraint = False
                    robot_reference_qpos = self._optimize_configuration(
                        robot_reference_qpos,
                        reference_laplacian,
                        adjacency,
                        np.empty((0, 3)),
                        target_human_joints=mapped_reference_positions,
                        max_iter=self.body_orientation_calibration_iterations,
                        target_base_orientation=reference_base_orientation,
                    )
                finally:
                    self.hard_penetration_constraint = collision_policy
                print(
                    "Solved robot body-orientation calibration pose from source "
                    f"reference ({reference_label}, "
                    f"{self.body_orientation_calibration_iterations} SQP iterations)."
                )

        if preserve_core_reference:
            # A positional fit between differently proportioned bind poses
            # must not redefine the core's neutral orientation. Otherwise its
            # incidental waist twist becomes an offset on every source frame.
            free_ids = np.flatnonzero(
                self.robot_model.jnt_type == mujoco.mjtJoint.mjJNT_FREE
            )
            if len(free_ids) == 1:
                free_id = int(free_ids[0])
                free_body = int(self.robot_model.jnt_bodyid[free_id])
                free_qpos = int(self.robot_model.jnt_qposadr[free_id])
                if reference_base_orientation is not None:
                    orientation = np.asarray(reference_base_orientation, dtype=float)
                    robot_reference_qpos[free_qpos + 3:free_qpos + 7] = (
                        orientation / np.linalg.norm(orientation)
                    )
                anchor = self.end_effector_config.get("_position_anchor_source_joint")
                body = reference_bodies.get(anchor)
                core_joints = []
                while body is not None and body > 0 and body != free_body:
                    first = int(self.robot_model.body_jntadr[body])
                    core_joints.extend(range(first, first + int(self.robot_model.body_jntnum[body])))
                    body = int(self.robot_model.body_parentid[body])
                if body == free_body:
                    for joint_id in core_joints:
                        qadr = int(self.robot_model.jnt_qposadr[joint_id])
                        width = 4 if self.robot_model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_BALL else 1
                        robot_reference_qpos[qadr:qadr + width] = self.robot_model.qpos0[qadr:qadr + width]

        if self.body_orientation_reference_pose:
            # Applied after any core reset, so a declared joint always wins and
            # a mirrored declaration stays an exact mirror on the robot side.
            free_ids = np.flatnonzero(
                self.robot_model.jnt_type == mujoco.mjtJoint.mjJNT_FREE
            )
            if len(free_ids) == 1 and reference_base_orientation is not None:
                free_qpos = int(self.robot_model.jnt_qposadr[int(free_ids[0])])
                orientation = np.asarray(reference_base_orientation, dtype=float)
                robot_reference_qpos[free_qpos + 3:free_qpos + 7] = (
                    orientation / np.linalg.norm(orientation)
                )
            for joint_name, angle in self.body_orientation_reference_pose.items():
                joint_id = mujoco.mj_name2id(
                    self.robot_model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name)
                )
                if joint_id < 0:
                    raise ValueError(
                        "body_orientation.reference_pose names missing robot "
                        f"joint '{joint_name}'."
                    )
                if self.robot_model.jnt_type[joint_id] not in (
                    mujoco.mjtJoint.mjJNT_HINGE,
                    mujoco.mjtJoint.mjJNT_SLIDE,
                ):
                    raise ValueError(
                        "body_orientation.reference_pose only accepts scalar "
                        f"joints; '{joint_name}' is not one."
                    )
                value = float(angle)
                lower, upper = (
                    float(x) for x in self.robot_model.jnt_range[joint_id]
                )
                if not np.isfinite(value) or (
                    bool(self.robot_model.jnt_limited[joint_id])
                    and not lower <= value <= upper
                ):
                    raise ValueError(
                        f"body_orientation.reference_pose['{joint_name}'] = "
                        f"{angle} is not a finite in-range joint value."
                    )
                robot_reference_qpos[
                    int(self.robot_model.jnt_qposadr[joint_id])
                ] = value
            print(
                "Declared robot body-orientation calibration pose from the robot "
                f"profile ({len(self.body_orientation_reference_pose)} joints); "
                "no positional fit, so bind twist is not left to the optimizer."
            )

        self.robot_reference_qpos = robot_reference_qpos.copy()
        self.robot_data.qpos[:] = robot_reference_qpos
        mujoco.mj_forward(self.robot_model, self.robot_data)
        try:
            for source_name, body_id in reference_bodies.items():
                source_index = int(name_to_index[source_name])
                source_wxyz = references[source_index]
                source_norm = float(np.linalg.norm(source_wxyz))
                if source_norm < 1e-12:
                    raise ValueError(
                        f"Reference orientation for '{source_name}' is zero."
                    )
                self._source_orientation_indices[source_name] = source_index
                self._source_orientation_references[source_name] = (
                    Rotation.from_quat(
                        source_wxyz / source_norm, scalar_first=True
                    ).as_matrix()
                )
                self._robot_orientation_references[source_name] = (
                    self.robot_data.xmat[int(body_id)]
                    .reshape(3, 3)
                    .copy()
                )
        finally:
            self.robot_data.qpos[:] = current_qpos
            mujoco.mj_forward(self.robot_model, self.robot_data)

        print(
            "Calibrated source body orientations: "
            + ", ".join(
                f"{name}->{task['body_name']}"
                for name, task in self.body_orientation_tasks.items()
            )
        )


    def _build_body_orientation_residual(
        self,
        dqa: cp.Variable,
        source_joint_orientations: Optional[np.ndarray],
        target_base_orientation: Optional[np.ndarray] = None,
    ) -> Optional[cp.Expression]:
        """Linearize calibrated world-frame body orientation targets."""
        if (
            not self.body_orientation_enabled
            or source_joint_orientations is None
            or not self._source_orientation_indices
        ):
            return None
        source_orientations = np.asarray(source_joint_orientations, dtype=float)
        if source_orientations.ndim != 2 or source_orientations.shape[1] != 4:
            raise ValueError(
                "source_joint_orientations for one frame must have shape (J, 4)."
            )

        residuals: List[cp.Expression] = []
        for source_name, task in self.body_orientation_tasks.items():
            source_index = self._source_orientation_indices[source_name]
            source_wxyz = source_orientations[source_index]
            source_norm = float(np.linalg.norm(source_wxyz))
            if source_norm < 1e-12:
                continue
            source_rotation = Rotation.from_quat(
                source_wxyz / source_norm, scalar_first=True
            ).as_matrix()
            target_rotation = _calibrated_orientation_target(
                source_rotation,
                self._source_orientation_references[source_name],
                self._robot_orientation_references[source_name],
            )
            body_id = int(task["body_id"])
            current_rotation = self.robot_data.xmat[body_id].reshape(3, 3)
            rotation_error = Rotation.from_matrix(
                target_rotation @ current_rotation.T
            ).as_rotvec()
            angular_jacobian = self._body_angular_jacobian(body_id)
            if "parent_source" in task:
                parent_name, parent_id = task["parent_source"], task["parent_body_id"]
                parent_source = Rotation.from_quat(
                    source_orientations[self._source_orientation_indices[parent_name]],
                    scalar_first=True,
                ).as_matrix()
                parent_reference = self._robot_orientation_references[parent_name]
                # Relative target equals calibrated-parent-world.T times
                # calibrated-child-world. Both reference coordinate changes
                # are explicit; source and robot joint axes need not match.
                target_relative = (
                    parent_reference.T
                    @ self._source_orientation_references[parent_name]
                    @ parent_source.T @ target_rotation
                )
                if target_base_orientation is not None and int(self.robot_model.body_parentid[parent_id]) == 0:
                    # A virtual anatomical torso may drive the base, instead
                    # of one raw spine joint. Use that same declared parent
                    # frame so the relative limb task solves the same problem.
                    target_relative = Rotation.from_quat(
                        target_base_orientation, scalar_first=True
                    ).as_matrix().T @ target_rotation
                parent_rotation = self.robot_data.xmat[parent_id].reshape(3, 3)
                current_relative = parent_rotation.T @ current_rotation
                rotation_error = Rotation.from_matrix(
                    target_relative @ current_relative.T
                ).as_rotvec()
                angular_jacobian = parent_rotation.T @ (
                    angular_jacobian - self._body_angular_jacobian(parent_id)
                )
            scale = np.sqrt(float(task["weight"])) / float(task["sigma"])
            residuals.append(
                scale * (angular_jacobian @ dqa - rotation_error)
            )
        return cp.hstack(residuals) if residuals else None
