"""Source-relative bone direction and temporal pose residuals."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple

import cvxpy as cp
import mujoco
import numpy as np


class PoseDirectionMixin:
    """Source-relative bone direction and temporal pose residuals.

    Internal methods sharing the owning GenericInteractionRetargeter instance; no separate state.
    """

    def _setup_pose_bones(self) -> None:
        """Cache robot bone lengths for source/robot direction matching."""
        pose_cfg = dict(self.end_effector_config.get("pose_direction", {}) or {})
        self.pose_direction_enabled = bool(pose_cfg.get("enabled", True))
        # The direction-velocity target transports an ambient difference
        # between different points of the unit sphere, which can fight the
        # position-level direction matching; profiles can keep only the
        # position-level term.
        self.pose_velocity_enabled = bool(pose_cfg.get("velocity_enabled", True))
        self.pose_direction_sigma = max(float(pose_cfg.get("sigma", 0.12)), 1e-6)
        self.pose_arm_weight = max(float(pose_cfg.get("arm_weight", 1.0)), 0.0)
        self.pose_leg_weight = max(float(pose_cfg.get("leg_weight", 1.0)), 0.0)
        self.pose_edge_weight_overrides: Dict[str, float] = {}
        for edge_name, raw_weight in dict(
            pose_cfg.get("edge_weight_overrides", {}) or {}
        ).items():
            edge_weight = float(raw_weight)
            if not np.isfinite(edge_weight) or edge_weight < 0.0:
                raise ValueError(
                    f"pose_direction.edge_weight_overrides['{edge_name}'] "
                    "must be a finite non-negative number."
                )
            self.pose_edge_weight_overrides[str(edge_name)] = edge_weight
        self.pose_edge_joint_effectors: Dict[str, str] = {
            str(edge_name): str(effector_name)
            for edge_name, effector_name in dict(
                pose_cfg.get("edge_joint_effectors", {}) or {}
            ).items()
        }
        for edge_name, effector_name in self.pose_edge_joint_effectors.items():
            if effector_name not in self.end_effectors:
                raise ValueError(
                    f"pose_direction.edge_joint_effectors['{edge_name}'] "
                    f"refers to unknown effector '{effector_name}'."
                )
        self.pose_velocity_sigma = max(
            float(pose_cfg.get("velocity_sigma", 0.035)), 1e-6
        )
        self.pose_acceleration_enabled = bool(
            pose_cfg.get("acceleration_enabled", False)
        )
        self.pose_acceleration_sigma = max(
            float(pose_cfg.get("acceleration_sigma", 0.03)), 1e-6
        )
        raw_edges = list(
            pose_cfg.get(
                "edges",
                [
                    ["Spine3", "L_Shoulder"],
                    ["L_Shoulder", "L_Elbow"],
                    ["L_Elbow", "L_Wrist"],
                    ["Spine3", "R_Shoulder"],
                    ["R_Shoulder", "R_Elbow"],
                    ["R_Elbow", "R_Wrist"],
                    ["Spine3", "Spine1"],
                    ["Pelvis", "L_Hip"],
                    ["L_Hip", "L_Knee"],
                    ["L_Knee", "L_Ankle"],
                    ["Pelvis", "R_Hip"],
                    ["R_Hip", "R_Knee"],
                    ["R_Knee", "R_Ankle"],
                ],
            )
        )
        base_orientation_source_joint = self.end_effector_config.get(
            "_base_orientation_source_joint"
        )
        torso_edge = ("Pelvis", str(base_orientation_source_joint))
        if (
            base_orientation_source_joint
            and all(name in self.joint_mapping for name in torso_edge)
            and torso_edge
            not in {(str(edge[0]), str(edge[1])) for edge in raw_edges}
        ):
            # A reduced source such as PARC may expose Spine1 but not the
            # profile's SMPL-X Spine3.  Restore the pelvis-to-base relationship
            # so waist posture remains defined after the base joint itself is
            # removed from duplicate world-orientation tracking.
            raw_edges.append(list(torso_edge))
        self.pose_bones: List[Tuple[str, str, float]] = []
        if not self.pose_direction_enabled and not self.end_effector_config.get("pose_initialization", {}).get("enabled", False):
            return

        current_qpos = self.robot_data.qpos.copy()
        self.robot_data.qpos[:] = self.robot_model.qpos0
        mujoco.mj_forward(self.robot_model, self.robot_data)
        for edge in raw_edges:
            if len(edge) != 2:
                raise ValueError(f"pose_direction edge must have two names, got {edge}.")
            parent_name, child_name = str(edge[0]), str(edge[1])
            if parent_name not in self.joint_mapping or child_name not in self.joint_mapping:
                continue
            parent_id = mujoco.mj_name2id(
                self.robot_model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.joint_mapping[parent_name],
            )
            child_id = mujoco.mj_name2id(
                self.robot_model,
                mujoco.mjtObj.mjOBJ_BODY,
                self.joint_mapping[child_name],
            )
            length = float(
                np.linalg.norm(
                    self.robot_data.xpos[child_id] - self.robot_data.xpos[parent_id]
                )
            )
            if length > 1e-4:
                self.pose_bones.append((parent_name, child_name, length))
        self.robot_data.qpos[:] = current_qpos
        mujoco.mj_forward(self.robot_model, self.robot_data)
        print(f"Pose-direction bones: {len(self.pose_bones)}")


    def _localize_pose_edge_jacobian(
        self,
        parent_name: str,
        child_name: str,
        jacobian: np.ndarray,
    ) -> np.ndarray:
        """Keep configured limb-direction corrections inside that limb chain."""
        effector_name = self.pose_edge_joint_effectors.get(
            f"{parent_name}->{child_name}"
        )
        if effector_name is None:
            return jacobian
        allowed = np.asarray(
            self.end_effectors[effector_name]["contact_joint_qa_indices"],
            dtype=int,
        )
        localized = np.zeros_like(jacobian)
        if len(allowed):
            localized[:, allowed] = jacobian[:, allowed]
        return localized


    def _build_pose_direction_residual(
        self,
        dqa: cp.Variable,
        robot_positions: Mapping[str, np.ndarray],
        robot_jacobian: np.ndarray,
        target_human_joints: Optional[np.ndarray],
    ) -> Optional[cp.Expression]:
        """Match robot bone directions to SMPL-X, independent of morphology."""
        if (
            not self.pose_direction_enabled
            or not self.pose_bones
            or target_human_joints is None
        ):
            return None
        target_human_joints = np.asarray(target_human_joints, dtype=float)
        name_to_index = {
            name: index for index, name in enumerate(self.valid_joint_names)
        }
        terms: List[cp.Expression] = []
        for parent_name, child_name, robot_length in self.pose_bones:
            parent_idx = name_to_index[parent_name]
            child_idx = name_to_index[child_name]
            human_vector = (
                target_human_joints[child_idx] - target_human_joints[parent_idx]
            )
            human_length = float(np.linalg.norm(human_vector))
            if human_length < 1e-8:
                continue
            target_direction = human_vector / human_length
            robot_vector = (
                robot_positions[child_name] - robot_positions[parent_name]
            )
            vector_jacobian = (
                robot_jacobian[3 * child_idx : 3 * child_idx + 3]
                - robot_jacobian[3 * parent_idx : 3 * parent_idx + 3]
            )
            vector_jacobian = self._localize_pose_edge_jacobian(
                parent_name, child_name, vector_jacobian
            )
            is_leg = any(
                token in parent_name or token in child_name
                for token in ("Hip", "Knee", "Ankle")
            )
            direction_weight = (
                self.pose_leg_weight if is_leg else self.pose_arm_weight
            )
            direction_weight *= self.pose_edge_weight_overrides.get(
                f"{parent_name}->{child_name}", 1.0
            )
            terms.append(
                np.sqrt(direction_weight)
                * (
                    (robot_vector + vector_jacobian @ dqa) / robot_length
                    - target_direction
                )
                / self.pose_direction_sigma
            )
        return cp.hstack(terms) if terms else None


    def _build_pose_velocity_residual(
        self,
        dqa: cp.Variable,
        robot_positions: Mapping[str, np.ndarray],
        robot_jacobian: np.ndarray,
        target_human_joints: Optional[np.ndarray],
        previous_human_joints: Optional[np.ndarray],
        last_directions: Mapping[Tuple[str, str], np.ndarray],
    ) -> Optional[cp.Expression]:
        """Track captured bone-direction velocity instead of chasing frames."""
        if (
            not self.pose_velocity_enabled
            or target_human_joints is None
            or previous_human_joints is None
            or not last_directions
        ):
            return None
        name_to_index = {
            name: index for index, name in enumerate(self.valid_joint_names)
        }
        current_human = np.asarray(target_human_joints, dtype=float)
        previous_human = np.asarray(previous_human_joints, dtype=float)
        terms: List[cp.Expression] = []
        for parent_name, child_name, _ in self.pose_bones:
            edge = (parent_name, child_name)
            if edge not in last_directions:
                continue
            parent_idx = name_to_index[parent_name]
            child_idx = name_to_index[child_name]
            human_current = current_human[child_idx] - current_human[parent_idx]
            human_previous = previous_human[child_idx] - previous_human[parent_idx]
            human_current /= max(float(np.linalg.norm(human_current)), 1e-10)
            human_previous /= max(float(np.linalg.norm(human_previous)), 1e-10)
            desired_direction = np.asarray(last_directions[edge]) + (
                human_current - human_previous
            )
            desired_direction /= max(
                float(np.linalg.norm(desired_direction)), 1e-10
            )

            robot_vector = robot_positions[child_name] - robot_positions[parent_name]
            robot_length = float(np.linalg.norm(robot_vector))
            if robot_length < 1e-8:
                continue
            robot_direction = robot_vector / robot_length
            vector_jacobian = (
                robot_jacobian[3 * child_idx : 3 * child_idx + 3]
                - robot_jacobian[3 * parent_idx : 3 * parent_idx + 3]
            )
            direction_jacobian = (
                np.eye(3) - np.outer(robot_direction, robot_direction)
            ) @ vector_jacobian / robot_length
            terms.append(
                (robot_direction + direction_jacobian @ dqa - desired_direction)
                / self.pose_velocity_sigma
            )
        return cp.hstack(terms) if terms else None


    def _build_pose_acceleration_residual(
        self,
        dqa: cp.Variable,
        robot_positions: Mapping[str, np.ndarray],
        robot_jacobian: np.ndarray,
        target_human_joints: Optional[np.ndarray],
        previous_human_joints: Optional[np.ndarray],
        previous_previous_human_joints: Optional[np.ndarray],
        last_directions: Mapping[Tuple[str, str], np.ndarray],
        previous_directions: Mapping[Tuple[str, str], np.ndarray],
    ) -> Optional[cp.Expression]:
        """Match second differences of visible robot and human bone directions.

        Position-only IK can alternate between redundant elbow, wrist, knee and
        ankle solutions while hitting every per-frame target.  Matching the
        captured bone acceleration removes that alternating mode without
        low-pass filtering away deliberate fast human motion.
        """
        if (
            not self.pose_acceleration_enabled
            or target_human_joints is None
            or previous_human_joints is None
            or previous_previous_human_joints is None
            or not last_directions
            or not previous_directions
        ):
            return None
        name_to_index = {
            name: index for index, name in enumerate(self.valid_joint_names)
        }
        human_current = np.asarray(target_human_joints, dtype=float)
        human_previous = np.asarray(previous_human_joints, dtype=float)
        human_previous_previous = np.asarray(
            previous_previous_human_joints, dtype=float
        )
        terms: List[cp.Expression] = []
        for parent_name, child_name, _ in self.pose_bones:
            edge = (parent_name, child_name)
            if edge not in last_directions or edge not in previous_directions:
                continue
            parent_idx = name_to_index[parent_name]
            child_idx = name_to_index[child_name]

            human_directions = []
            for joints in (
                human_current,
                human_previous,
                human_previous_previous,
            ):
                direction = joints[child_idx] - joints[parent_idx]
                direction /= max(float(np.linalg.norm(direction)), 1e-10)
                human_directions.append(direction)
            human_acceleration = (
                human_directions[0]
                - 2.0 * human_directions[1]
                + human_directions[2]
            )
            desired_direction = (
                2.0 * np.asarray(last_directions[edge], dtype=float)
                - np.asarray(previous_directions[edge], dtype=float)
                + human_acceleration
            )
            desired_direction /= max(
                float(np.linalg.norm(desired_direction)), 1e-10
            )

            robot_vector = (
                robot_positions[child_name] - robot_positions[parent_name]
            )
            robot_length = float(np.linalg.norm(robot_vector))
            if robot_length < 1e-8:
                continue
            robot_direction = robot_vector / robot_length
            vector_jacobian = (
                robot_jacobian[3 * child_idx : 3 * child_idx + 3]
                - robot_jacobian[3 * parent_idx : 3 * parent_idx + 3]
            )
            direction_jacobian = (
                np.eye(3) - np.outer(robot_direction, robot_direction)
            ) @ vector_jacobian / robot_length
            terms.append(
                (
                    robot_direction
                    + direction_jacobian @ dqa
                    - desired_direction
                )
                / self.pose_acceleration_sigma
            )
        return cp.hstack(terms) if terms else None
