"""Assembly of one framewise SQP step."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import cvxpy as cp
import mujoco
import numpy as np
from scipy import sparse as sp
from scipy.spatial.transform import Rotation

from ..utils import calculate_laplacian_matrix
from .constraints import build_step_constraints
from .hierarchy import solve_hierarchy
from . import tangent


class OptimizationStepMixin:
    """Assembly of one framewise SQP step.

    Internal methods sharing the owning GenericInteractionRetargeter instance; no separate state.
    """

    def _single_optimization_step(
        self,
        q: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: List[List[int]],
        terrain_points: np.ndarray,
        target_human_joints: Optional[np.ndarray] = None,
        q_last: Optional[np.ndarray] = None,
        q_prev: Optional[np.ndarray] = None,
        q_prevprev: Optional[np.ndarray] = None,
        previous_human_joints: Optional[np.ndarray] = None,
        previous_previous_human_joints: Optional[np.ndarray] = None,
        target_base_orientation: Optional[np.ndarray] = None,
        previous_target_base_orientation: Optional[np.ndarray] = None,
        source_joint_orientations: Optional[np.ndarray] = None,
        target_base_position: Optional[np.ndarray] = None,
        end_effector_targets: Optional[Mapping[str, Mapping[str, Any]]] = None,
        relation_frame: int | None = None,
        source_laplacian: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Single SQP optimization step.

        Args:
            q: Current configuration
            target_laplacian: Target Laplacian coordinates
            adj_list: Mesh adjacency list
            terrain_points: Terrain contact points
            q_last: Configuration at previous time step (for smoothness)

        Returns:
            Tuple of (optimized_config, cost)
        """
        # Update robot state
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)

        # Compute Jacobians for mapped joints
        J_V, p_V, _ = self._compute_robot_jacobians(q)

        # Create Laplacian matrices
        # CRITICAL: Ensure robot_points are in the SAME ORDER as human_joints passed to retarget_frame
        # The order MUST match: human_joints[i] corresponds to robot_points[i] for all i
        #
        # Order flow:
        #   1. In core.py: valid_joint_names is built by iterating joint_mapping.keys() in order
        #   2. In core.py: mapped_joints = human_joints[mapped_joint_indices] where mapped_joint_indices
        #      corresponds to valid_joint_names[i] for each i
        #   3. In core.py: valid_joint_mapping preserves valid_joint_names order
        #   4. Here: self.joint_mapping IS valid_joint_mapping (passed from core.py)
        #   5. Here: robot_points built by iterating self.joint_mapping.keys() which matches valid_joint_names
        #   6. J_V stacked by iterating self.joint_mapping.keys() in the same order
        #
        # So: human_joints[i] (from valid_joint_names[i]) should match robot_points[i] (from joint_mapping.keys()[i])
        # CRITICAL: Use self.valid_joint_names to ensure consistent ordering
        joint_names_ordered = self.valid_joint_names

        # CRITICAL: Verify that p_V has all joints in the correct order
        if set(joint_names_ordered) != set(p_V.keys()):
            missing = set(joint_names_ordered) - set(p_V.keys())
            extra = set(p_V.keys()) - set(joint_names_ordered)
            raise RuntimeError(
                f"Joint mismatch: p_V has different joints than joint_mapping. "
                f"Missing from p_V: {missing}, Extra in p_V: {extra}"
            )

        robot_points = []
        for joint_name in joint_names_ordered:
            robot_points.append(p_V[joint_name])

        robot_points = np.array(robot_points)

        # Debug: Print order for first frame to verify
        import sys
        if not hasattr(sys, '_omni_joint_order_printed'):
            print(f"\n=== Joint Order Verification ===")
            print(f"Joint mapping order: {joint_names_ordered}")
            print(f"p_V keys order: {list(p_V.keys())}")
            print(f"First 3 robot points correspond to: {joint_names_ordered[:3]}")
            sys._omni_joint_order_printed = True

        # CRITICAL: Validate that sizes match exactly
        expected_num_joints = len(joint_names_ordered)
        if len(robot_points) != expected_num_joints:
            raise ValueError(
                f"Size mismatch: robot_points has {len(robot_points)} joints, "
                f"but expected {expected_num_joints} joints from joint_mapping.keys()."
            )
        if J_V.shape[0] != 3 * expected_num_joints:
            raise ValueError(
                f"Jacobian dimension mismatch: J_V has {J_V.shape[0]//3} joints, "
                f"but expected {expected_num_joints} joints from joint_mapping. "
                f"J_V shape: {J_V.shape}, expected rows: {3 * expected_num_joints}"
            )
        if len(robot_points) != J_V.shape[0] // 3:
            raise ValueError(
                f"Size mismatch between robot_points ({len(robot_points)}) and J_V ({J_V.shape[0]//3} joints)."
            )
        if len(robot_points) == 0:
            # Handle empty robot points to avoid vstack error
            if len(terrain_points) == 0:
                # Both empty, raise error as we can't do anything
                raise ValueError("Both robot_points and terrain_points are empty")
            vertices = terrain_points
            print("WARNING: No robot points found! Only using terrain points.")
        else:
            vertices = np.vstack([robot_points, terrain_points])

        # CRITICAL: Use uniform_weight=True to match target_laplacian computation
        # This ensures consistent Laplacian computation between target and current
        L = (calculate_laplacian_matrix(vertices, adj_list, uniform_weight=True)
             if source_laplacian is None else source_laplacian)
        if L.shape != (len(vertices), len(vertices)):
            raise ValueError("Source Laplacian must match the ordered interaction vertices.")
        if not sp.issparse(L):
            L = sp.csr_matrix(L)

        # Kron shape: (3*num_vertices, 3*num_vertices)
        Kron = sp.kron(L, sp.eye(3, format="csr"), format="csr")

        # J_V shape: (3*num_joints, nq_a) - stacked Jacobians for each mapped joint
        # BUT Kron expects input vector of size 3*num_vertices (where vertices = mapped_joints + terrain_points)
        # J_V maps joint velocities (dqa) to velocities of mapped_joints (and 0 for terrain points)

        # We need to construct a full Jacobian J_full of shape (3*num_vertices, nq_a)
        # The top part corresponds to robot_points (mapped joints), bottom part (terrain) is zeros

        num_robot_points = len(robot_points)
        num_terrain_points = len(terrain_points)
        num_vertices = num_robot_points + num_terrain_points

        # Verify sizes match
        if J_V.shape[0] != 3 * num_robot_points:
             # This can happen if p_V has different number of points than J_V's stack
             # But they come from the same loop, so they should match
             print(f"Warning: J_V rows ({J_V.shape[0]}) != 3 * num_robot_points ({3*num_robot_points})")

        # Construct full Jacobian for all vertices
        # Top part: J_V (robot points), Bottom part: 0 (terrain points, static)
        J_full_vertices = sp.vstack([
            sp.csr_matrix(J_V),  # Jacobians for robot points
            sp.csr_matrix((3 * num_terrain_points, self.nq_a)) # Zeros for terrain points
        ])

        # Now J_L = Kron @ J_full_vertices
        # Kron: (3*V, 3*V)
        # J_full_vertices: (3*V, nq_a)
        # Result J_L: (3*V, nq_a)
        J_L = Kron @ J_full_vertices

        # Setup optimization problem
        dqa = cp.Variable(len(self.q_a_indices), name="dqa")

        # Constraints
        constraints = []
        if tangent.enabled(self):
            dqa, constraints = tangent.make_step(self, q)

        # Substitute the Laplacian residual directly, as in upstream's small
        # QP. The objective is unchanged; auxiliary graph equalities disappear.
        lap0_vec = (L @ vertices).reshape(-1)
        target_lap_vec = target_laplacian.reshape(-1)

        lap_var = lap0_vec + cp.Constant(J_L) @ dqa

        # Joint limits
        q_a_current = q[self.q_a_indices]
        constraints.extend([
            dqa >= (self.q_a_lb - q_a_current),
            dqa <= (self.q_a_ub - q_a_current),
        ])

        temporal_cfg, base_cfg, base_position_obj = build_step_constraints(
            retargeter=self,
            constraints=constraints,
            dqa=dqa,
            end_effector_targets=end_effector_targets,
            previous_human_joints=previous_human_joints,
            previous_previous_human_joints=previous_previous_human_joints,
            previous_target_base_orientation=previous_target_base_orientation,
            q=q,
            q_a_current=q_a_current,
            q_last=q_last,
            q_prev=q_prev,
            target_base_orientation=target_base_orientation,
            target_base_position=target_base_position,
            target_human_joints=target_human_joints,
        )

        # Scene and robot-self collision have independent switches.
        if self.hard_penetration_constraint:
            penetration_constraints = self._compute_penetration_constraints(q, dqa)
            constraints.extend(penetration_constraints)
        self_collision_objective = self._self_collision_objective(q, dqa, constraints, q_last)

        # Trust region
        if not tangent.enabled(self):
            constraints.append(cp.SOC(self.step_size, dqa))

        # Whole-body objective (the lowest level of the hierarchy).
        mesh_cfg = dict(self.end_effector_config.get("interaction_mesh", {}) or {})
        laplacian_weight = float(mesh_cfg.get("laplacian_weight", 10.0))  # matching original laplacian_weights = 10
        weights = np.ones(len(vertices)) * laplacian_weight
        # The target Laplacian keeps human metric bone lengths at scale 1.0,
        # which a shorter robot cannot reproduce.  Per-joint overrides let the
        # profile relax exactly the limb rows whose infeasible human-length
        # targets would otherwise stretch elbows/knees, while terrain rows and
        # torso rows keep encoding the scene interaction at full weight.
        for joint_name, joint_weight in dict(
            mesh_cfg.get("joint_weight_overrides", {}) or {}
        ).items():
            if joint_name not in joint_names_ordered:
                raise ValueError(
                    f"interaction_mesh.joint_weight_overrides refers to unmapped "
                    f"joint '{joint_name}'."
                )
            weights[joint_names_ordered.index(joint_name)] = float(joint_weight)
        sqrt_w3 = np.sqrt(np.repeat(weights, 3))

        # Minimize: ||lap_var - target_lap_vec||^2
        # where lap_var = lap0_vec + J_L @ dqa (from constraint)
        body_obj = cp.sum_squares(cp.multiply(sqrt_w3, lap_var - target_lap_vec))
        body_obj += self_collision_objective
        if base_position_obj is not None:
            body_obj += float(base_cfg.get("centering_weight", 5.0)) * base_position_obj

        # Joint regularization cost (keep joints near zero/neutral)
        # Matching original: Q_diag cost uses q_a_n_last (last optimized at current time step)
        # In our case, q_a_current = q[self.q_a_indices] which is the current guess
        Qd = np.asarray(self.Q_diag, dtype=float).reshape(-1)

        # Q_diag cost: ||sqrt(Q_diag) * (dqa + q_a_current)||^2
        # This matches original: cp.sum_squares(cp.multiply(np.sqrt(Qd), dqa + q_a_n_last))
        body_obj += cp.sum_squares(cp.multiply(np.sqrt(Qd), dqa + q_a_current))

        # Smoothness cost (matching original implementation exactly)
        # CRITICAL FIX: Use previous frame's velocity, not current guess
        if q_last is not None:
            q_a_current = q[self.q_a_indices]  # Current guess at this SQP iteration
            dqa_smooth = q_last[self.q_a_indices] - q_a_current  # Velocity from prev frame to current guess
            smooth_residual = (tangent.temporal_residual(self, q, q_last, dqa)
                               if tangent.enabled(self) else dqa - dqa_smooth)
            body_obj += self.smooth_weight * cp.sum_squares(smooth_residual)

        # Base orientation tracking cost
        # Keep the base orientation close to the target (estimated from human joints).
        # The residual is the world-frame rotation vector to the target and its
        # linearization is the same qdot->omega map used by every other
        # orientation task.  Penalizing raw quaternion components instead mixes
        # a non-metric distance into the objective and pulls along the radial
        # quaternion direction that the kinematics cannot see.
        base_orientation_residual = None
        if (
            target_base_orientation is not None
            and 3 in self.q_a_indices
            and all(idx in self.q_a_indices for idx in (4, 5, 6))
            and self.robot_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        ):
            base_body_id = int(self.robot_model.jnt_bodyid[0])
            target_wxyz = np.asarray(target_base_orientation, dtype=float).reshape(4)
            target_rotation = Rotation.from_quat(
                target_wxyz, scalar_first=True
            ).as_matrix()
            current_rotation = self.robot_data.xmat[base_body_id].reshape(3, 3)
            base_rotation_error = Rotation.from_matrix(
                target_rotation @ current_rotation.T
            ).as_rotvec()
            base_angular_jacobian = self._body_angular_jacobian(base_body_id)
            # 1.25 rad^-2 keeps the strength of the replaced component cost:
            # its weight was 5.0 and |dq| ~ |dtheta|/2 for unit quaternions.
            orientation_weight = float(
                self.end_effector_config.get("base_orientation_weight", 1.25)
            )
            base_orientation_residual = (
                base_angular_jacobian @ dqa - base_rotation_error
            )
            body_obj += orientation_weight * cp.sum_squares(
                base_orientation_residual
            )

        body_orientation_residual = self._build_body_orientation_residual(
            dqa, source_joint_orientations, target_base_orientation
        )
        if body_orientation_residual is not None:
            body_obj += float(
                self.end_effector_config.get("body_orientation_weight", 1.0)
            ) * cp.sum_squares(body_orientation_residual)

        last_end_effector_rotations: Dict[str, np.ndarray] = {}
        last_pose_directions: Dict[Tuple[str, str], np.ndarray] = {}
        previous_pose_directions: Dict[Tuple[str, str], np.ndarray] = {}
        if q_last is not None and (
            self.end_effector_enabled or self.pose_direction_enabled
        ):
            current_qpos = self.robot_data.qpos.copy()
            self.robot_data.qpos[:] = q_last
            mujoco.mj_forward(self.robot_model, self.robot_data)
            for name, spec in self.end_effectors.items():
                if spec.get("track_orientation", False):
                    last_end_effector_rotations[name] = self.robot_data.xmat[
                        spec["body_id"]
                    ].reshape(3, 3).copy()
            for parent_name, child_name, _ in self.pose_bones:
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
                vector = (
                    self.robot_data.xpos[child_id]
                    - self.robot_data.xpos[parent_id]
                )
                vector_norm = float(np.linalg.norm(vector))
                if vector_norm > 1e-8:
                    last_pose_directions[(parent_name, child_name)] = (
                        vector / vector_norm
                    )
            self.robot_data.qpos[:] = current_qpos
            mujoco.mj_forward(self.robot_model, self.robot_data)

        if q_prev is not None and self.pose_direction_enabled:
            current_qpos = self.robot_data.qpos.copy()
            self.robot_data.qpos[:] = q_prev
            mujoco.mj_forward(self.robot_model, self.robot_data)
            for parent_name, child_name, _ in self.pose_bones:
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
                vector = (
                    self.robot_data.xpos[child_id]
                    - self.robot_data.xpos[parent_id]
                )
                vector_norm = float(np.linalg.norm(vector))
                if vector_norm > 1e-8:
                    previous_pose_directions[(parent_name, child_name)] = (
                        vector / vector_norm
                    )
            self.robot_data.qpos[:] = current_qpos
            mujoco.mj_forward(self.robot_model, self.robot_data)

        (
            contact_residual,
            swing_residual,
            swing_orientation_residual,
            endpoint_contact_constraints,
            num_contacts,
            num_swings,
            num_swing_orientations,
        ) = (
            self._build_end_effector_objectives(
                dqa, end_effector_targets, last_end_effector_rotations
            )
        )
        constraints.extend(endpoint_contact_constraints)
        pose_direction_residual = self._build_pose_direction_residual(
            dqa,
            p_V,
            J_V,
            target_human_joints,
        )
        pose_velocity_residual = self._build_pose_velocity_residual(
            dqa,
            p_V,
            J_V,
            target_human_joints,
            previous_human_joints,
            last_pose_directions,
        )
        if pose_velocity_residual is not None:
            pose_direction_residual = (
                cp.hstack([pose_direction_residual, pose_velocity_residual])
                if pose_direction_residual is not None
                else pose_velocity_residual
            )
        pose_acceleration_residual = self._build_pose_acceleration_residual(
            dqa,
            p_V,
            J_V,
            target_human_joints,
            previous_human_joints,
            previous_previous_human_joints,
            last_pose_directions,
            previous_pose_directions,
        )
        if pose_acceleration_residual is not None:
            pose_direction_residual = (
                cp.hstack(
                    [pose_direction_residual, pose_acceleration_residual]
                )
                if pose_direction_residual is not None
                else pose_acceleration_residual
            )
        temporal_residual = None
        if q_last is not None and q_prev is not None:
            actuated_qa = np.flatnonzero(self.q_a_indices >= 7)
            if len(actuated_qa):
                actuated_qpos = self.q_a_indices[actuated_qa]
                new_step = (
                    q_a_current[actuated_qa]
                    + dqa[actuated_qa]
                    - q_last[actuated_qpos]
                )
                previous_step = (
                    q_last[actuated_qpos] - q_prev[actuated_qpos]
                )
                new_acceleration = new_step - previous_step
                acceleration_sigma = max(
                    float(temporal_cfg.get("acceleration_sigma", 0.015)),
                    1e-8,
                )
                temporal_terms: List[cp.Expression] = [
                    new_acceleration / acceleration_sigma
                ]
                if q_prevprev is not None:
                    previous_acceleration = previous_step - (
                        q_prev[actuated_qpos] - q_prevprev[actuated_qpos]
                    )
                    jerk_sigma = max(
                        float(temporal_cfg.get("jerk_sigma", 0.01)), 1e-8
                    )
                    temporal_terms.append(
                        (new_acceleration - previous_acceleration) / jerk_sigma
                    )
                temporal_residual = cp.hstack(temporal_terms)

        # Acceleration/jerk is a null-space preference, not a motion task.
        # Keeping it in the final whole-body level preserves the previously
        # locked contact, swing and limb-direction optima while selecting the
        # least oscillatory joint solution among their redundant IK solutions.
        temporal_weight = float(
            self.end_effector_config.get("temporal_weight", 0.02)
        )
        if temporal_residual is not None and temporal_weight > 0.0:
            body_obj += temporal_weight * cp.sum_squares(temporal_residual)

        # Same source targets and nv rows are shared with refinement and local
        # candidates. Convert only here to this solver's active qpos increments.
        # This soft term belongs to the existing free-motion body objective;
        # the existing contact optimum and constraints retain their priority.
        if self.body_relation_tasks is not None and relation_frame is not None:
            rows = list(self.body_relation_tasks.rows(q, relation_frame, np.arange(self.robot_model.nv)))
            if rows:
                transform = self._build_transform_qdot_to_qvel_fast()[:, self.q_a_indices]
                for weight, residual, jacobian in rows:
                    body_obj += weight * cp.sum_squares(residual + jacobian @ transform @ dqa)

        self_contact_residual = None
        if self.self_contact_tasks is not None and relation_frame is not None:
            transform = self._build_transform_qdot_to_qvel_fast()[:, self.q_a_indices]
            terms = [np.sqrt(w) * (r + j @ transform @ dqa)
                     for w, r, j in self.self_contact_tasks.rows(q, relation_frame, np.arange(self.robot_model.nv))]
            if terms:
                self_contact_residual = cp.hstack(terms)

        surface_contact_residual = None
        if self.surface_contact_tasks is not None and relation_frame is not None:
            transform = self._build_transform_qdot_to_qvel_fast()[:, self.q_a_indices]
            terms = [np.sqrt(w) * (r + j @ transform @ dqa)
                     for w, r, j in self.surface_contact_tasks.rows(q, relation_frame, np.arange(self.robot_model.nv))]
            if terms:
                surface_contact_residual = cp.hstack(terms)

        return solve_hierarchy(
            retargeter=self,
            base_cfg=base_cfg,
            base_orientation_residual=base_orientation_residual,
            body_obj=body_obj,
            body_orientation_residual=body_orientation_residual,
            constraints=constraints,
            contact_residual=contact_residual,
            self_contact_residual=self_contact_residual,
            surface_contact_residual=surface_contact_residual,
            prioritize_surface_contact=(self.surface_contact_tasks is not None
                                        and self.surface_contact_tasks.cfg["priority"] == "after_end_effectors"),
            dqa=dqa,
            end_effector_targets=end_effector_targets,
            endpoint_contact_constraints=endpoint_contact_constraints,
            num_contacts=num_contacts,
            num_swing_orientations=num_swing_orientations,
            num_swings=num_swings,
            pose_direction_residual=pose_direction_residual,
            previous_target_base_orientation=previous_target_base_orientation,
            q=q,
            q_a_current=q_a_current,
            q_last=q_last,
            swing_orientation_residual=swing_orientation_residual,
            swing_residual=swing_residual,
            target_base_orientation=target_base_orientation,
            target_base_position=target_base_position,
            temporal_cfg=temporal_cfg,
        )
