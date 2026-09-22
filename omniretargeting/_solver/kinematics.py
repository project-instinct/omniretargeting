"""Robot Jacobians and penetration constraints."""

from __future__ import annotations

from typing import Dict, List, Tuple

import cvxpy as cp
import mujoco
import numpy as np


# Cylinder feature changes can need less than 1/32 of the proposed step.
# Keep checking the same nonlinear tolerances down to a bounded small step;
# rejecting all joint motion otherwise lets only the floating base advance.
_COLLISION_BACKTRACK_FRACTIONS = tuple(2.0 ** -i for i in range(13)) + (0.0,)


class RobotKinematicsMixin:
    """Robot Jacobians and penetration constraints.

    Internal methods sharing the owning GenericInteractionRetargeter instance; no separate state.
    """

    def _point_position_and_jacobian(
        self, body_id: int, local_point: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return a body-fixed point position and Jacobian w.r.t. ``dqa``."""
        local_point = np.asarray(local_point, dtype=float).reshape(3)
        rotation = self.robot_data.xmat[body_id].reshape(3, 3)
        position = self.robot_data.xpos[body_id] + rotation @ local_point
        jacobian_full = self._calc_contact_jacobian_from_point(body_id, local_point)
        valid_indices = self.q_a_indices[self.q_a_indices < jacobian_full.shape[1]]
        jacobian = jacobian_full[:, valid_indices]
        if jacobian.shape[1] < self.nq_a:
            padded = np.zeros((3, self.nq_a), dtype=float)
            padded[:, : jacobian.shape[1]] = jacobian
            jacobian = padded
        return position, jacobian


    def _body_angular_jacobian(self, body_id: int) -> np.ndarray:
        """Return world angular-velocity Jacobian w.r.t. optimized qpos."""
        jacobian_position = np.zeros((3, self.robot_model.nv), dtype=float)
        jacobian_rotation = np.zeros((3, self.robot_model.nv), dtype=float)
        mujoco.mj_jacBody(
            self.robot_model,
            self.robot_data,
            jacobian_position,
            jacobian_rotation,
            int(body_id),
        )
        jacobian_full = jacobian_rotation @ self._build_transform_qdot_to_qvel_fast()
        valid_indices = self.q_a_indices[self.q_a_indices < jacobian_full.shape[1]]
        jacobian = jacobian_full[:, valid_indices]
        if jacobian.shape[1] < self.nq_a:
            padded = np.zeros((3, self.nq_a), dtype=float)
            padded[:, : jacobian.shape[1]] = jacobian
            jacobian = padded
        return jacobian


    def _build_transform_qdot_to_qvel_fast(self, use_world_omega=True):
        """
        Return T(q) (nv x nq) such that v = T(q) @ qdot.
        - Free root: qpos=[x,y,z, qw,qx,qy,qz], qvel=[vx,vy,vz, ωx,ωy,ωz]
        where v is world-expressed and ω uses MuJoCo's free-joint convention
        (rotational velocity in the body-local tangent frame).  The historical
        helper names are swapped: the default ``get_e_world`` matrix is the
        one that matches MuJoCo, verified against ``mj_integratePos``.
        - 23 hinge joints: v = qdot.

        If use_world_omega=False, uses the alternate mapping (for debugging).
        """
        nq, nv = self.robot_model.nq, self.robot_model.nv
        T = np.zeros((nv, nq), dtype=float)

        # ---- root free joint (assumed joint 0) ----
        j0 = 0
        if self.robot_model.jnt_type[j0] == mujoco.mjtJoint.mjJNT_FREE:
            qadr = self.robot_model.jnt_qposadr[j0]  # 0
            dadr = self.robot_model.jnt_dofadr[j0]  # 0

            # Linear block: v_lin = xyz_dot
            T[dadr : dadr + 3, qadr : qadr + 3] = np.eye(3)

            # Angular block: ω_* = 2 * E_*(q) * quat_dot
            w, x, y, z = self.robot_data.qpos[qadr + 3 : qadr + 7]

            def get_e_world(qw, qx, qy, qz):
                return np.array(
                    [
                        [-qx, qw, qz, -qy],
                        [-qy, -qz, qw, qx],
                        [-qz, qy, -qx, qw],
                    ]
                )

            def get_e_body(qw, qx, qy, qz):
                return np.array(
                    [
                        [-qx, qw, -qz, qy],
                        [-qy, qz, qw, -qx],
                        [-qz, -qy, qx, qw],
                    ]
                )

            E_fn = get_e_world if use_world_omega else get_e_body
            E1 = 2.0 * E_fn(w, x, y, z)

            # linear-first: v_W = rdot, ω_W = 2E(q) * quat_dot
            # T[dadr + 0 : dadr + 3, qadr + 0 : qadr + 3] = np.eye(3) # Already set
            T[dadr + 3 : dadr + 6, qadr + 3 : qadr + 7] = E1  # ω block

        # ---- remaining hinge/slide joints: v = qdot ----
        for j in range(1 if self.robot_model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE else 0, self.robot_model.njnt):
            jt = self.robot_model.jnt_type[j]
            if jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                qa = self.robot_model.jnt_qposadr[j]
                da = self.robot_model.jnt_dofadr[j]
                T[da, qa] = 1.0
            elif jt == mujoco.mjtJoint.mjJNT_BALL:
                raise NotImplementedError("BALL joint block not implemented.")

        return T


    def _calc_contact_jacobian_from_point(self, body_idx: int, p_body: np.ndarray = None, input_world=False):
        """
        Translational Jacobian J(q) (3 x nq) such that
        v_point_world = J(q) @ qdot.

        Fast analytic version: J_qdot = J_v @ T(q)
        """
        if p_body is None:
            p_body = np.zeros(3)

        p_body = np.asarray(p_body, dtype=float).reshape(3)

        # 1) Make sure kinematics are current once
        # mujoco.mj_forward(self.robot_model, self.robot_data) # Assumed called before

        # 2) World point (3,1) for mj_jac
        R_WB = self.robot_data.xmat[body_idx].reshape(3, 3)
        p_WB = self.robot_data.xpos[body_idx]

        if input_world:
            p_W = p_body.astype(np.float64).reshape(3, 1)
        else:
            p_W = (p_WB + R_WB @ p_body).astype(np.float64).reshape(3, 1)

        # 3) J_v: translational Jacobian wrt generalized velocities (3 x nv)
        Jp = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        Jr = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        mujoco.mj_jac(self.robot_model, self.robot_data, Jp, Jr, p_W, int(body_idx))  # Jp = J_v

        T = self._build_transform_qdot_to_qvel_fast()

        return Jp @ T


    def _compute_robot_jacobians(self, q: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray], None]:
        """Compute Jacobians for robot joints in world frame.

        Args:
            q: Robot configuration

        Returns:
            Tuple of (J_V, p_dict, None):
                - J_V: Stacked Jacobians (3*num_joints, nq_a)
                - p_dict: Dictionary of positions for each joint
                - None: Placeholder for compatibility
        """
        J_dict = {}
        p_dict = {}

        for joint_name, link_name in self.joint_mapping.items():
            try:
                body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)

                # Get position in world frame
                pos = self.robot_data.xpos[body_id].copy()

                # Compute Jacobian in world frame
                J_full = self._calc_contact_jacobian_from_point(body_id)

                # Extract optimized part (J_full is already in qpos coordinates)
                valid_indices = self.q_a_indices[self.q_a_indices < J_full.shape[1]]
                if len(valid_indices) < len(self.q_a_indices):
                    print(
                        f"Warning: Truncating indices for joint {joint_name}. "
                        f"J width: {J_full.shape[1]}, Max idx: {self.q_a_indices.max()}"
                    )

                J_reduced = J_full[:, valid_indices]

                # Pad if needed
                if J_reduced.shape[1] < self.nq_a:
                    J_pad = np.zeros((3, self.nq_a))
                    J_pad[:, :J_reduced.shape[1]] = J_reduced
                    J_reduced = J_pad

                J_dict[joint_name] = J_reduced
                p_dict[joint_name] = pos

            except Exception as e:
                # CRITICAL: All joints should exist (validated in __init__), so this is unexpected
                # Raise error instead of skipping to ensure size consistency
                raise RuntimeError(
                    f"Failed to compute Jacobian for joint '{joint_name}' -> link '{link_name}'. "
                    f"This should not happen if joint_mapping was validated. Error: {e}"
                ) from e

        # Stack Jacobians in the SAME ORDER as valid_joint_names to match human_joints order
        # This is critical for correct Laplacian matching!
        # CRITICAL: Use self.valid_joint_names to ensure consistent ordering
        joint_names_ordered = self.valid_joint_names
        num_joints = len(joint_names_ordered)

        if num_joints > 0:
            J_V = np.zeros((3 * num_joints, self.nq_a))
            for i, joint_name in enumerate(joint_names_ordered):
                if joint_name in J_dict:
                    J = J_dict[joint_name]
                    # Ensure J has the correct shape (3, nq_a)
                    if J.shape != (3, self.nq_a):
                        if J.shape[1] > self.nq_a:
                            J = J[:, :self.nq_a]
                        elif J.shape[1] < self.nq_a:
                            J_pad = np.zeros((3, self.nq_a))
                            J_pad[:, :J.shape[1]] = J
                            J = J_pad
                    J_V[3 * i:3 * (i + 1), :] = J
                else:
                    # CRITICAL: All joints should exist (validated in __init__), so this is unexpected
                    raise RuntimeError(
                        f"Jacobian for joint '{joint_name}' not found in J_dict. "
                        f"This should not happen if joint_mapping was validated. "
                        f"Available joints in J_dict: {list(J_dict.keys())}"
                    )
        else:
            J_V = np.zeros((0, self.nq_a))

        return J_V, p_dict, None


    def _get_self_collision_query(self):
        from ..self_collision import SelfCollisionQuery

        if not hasattr(self, "_self_collision_query"):
            self._self_collision_query = SelfCollisionQuery(
                self.robot_model, getattr(self, "self_collision_config", None)
            )
        return self._self_collision_query

    def _self_collision_objective(self, q, dqa, constraints, q_last=None):
        from ..self_collision import bounded_collision_constraints

        if not getattr(self, "self_collision_enabled", False):
            return cp.Constant(0.0)
        query = self._get_self_collision_query()
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)
        deferred_path = (getattr(self, "_defer_self_collision_path_repair", False)
                         and self.end_effector_config.get("step_coordinates") == "tangent")
        rows = (query.query(data=self.robot_data) if q_last is None or deferred_path
                else query.segment(q_last, q, jacobians=True)["rows"])
        collision_constraints, penalty = bounded_collision_constraints(
            query, rows, self.robot_data, dqa, self.q_a_indices,
            self._build_transform_qdot_to_qvel_fast(),
        )
        constraints.extend(collision_constraints)
        return penalty

    def _guard_self_collision_step(self, before, candidate, q_last=None):
        """Recheck nonlinear distances after clipping and quaternion updates.

        Tangent solves backtrack the entire proposed physical step. Keeping a
        full root update after rejecting its compensating joint motion breaks
        the coupled task linearization and can tilt the trunk every iteration.
        Legacy qpos solves retain their historical internal-joint behavior.
        """
        if not getattr(self, "self_collision_enabled", False):
            return candidate
        if not getattr(self, "_defer_self_collision_path_repair", False):
            before = self._self_collision_initial_guess(before, q_last)
        query = self._get_self_collision_query()
        old = query.depths(before)
        deferred_path = (getattr(self, "_defer_self_collision_path_repair", False)
                         and getattr(self, "end_effector_config", {}).get("step_coordinates") == "tangent")
        # The full-trajectory caller repairs and validates both endpoints of
        # every interval before returning. Do not freeze one endpoint here and
        # prevent that coupled solve from receiving a usable motion proposal.
        check_path = q_last is not None and not deferred_path
        old_segment = query.segment(q_last, before) if check_path else None
        scalar = np.array([
            self.robot_model.jnt_qposadr[j] for j in range(self.robot_model.njnt)
            if self.robot_model.jnt_type[j] in (
                mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE
            )
        ], dtype=int)
        tangent_step = None
        if getattr(self, "end_effector_config", {}).get("step_coordinates") == "tangent":
            tangent_step = np.zeros(self.robot_model.nv)
            mujoco.mj_differentiatePos(self.robot_model, tangent_step, 1., before, candidate)
        for alpha in _COLLISION_BACKTRACK_FRACTIONS:
            if tangent_step is not None:
                trial = before.copy()
                mujoco.mj_integratePos(self.robot_model, trial, tangent_step, alpha)
            else:
                trial = candidate.copy()
                trial[scalar] = before[scalar] + alpha * (candidate[scalar] - before[scalar])
            if not query.acceptable(old, query.depths(trial)):
                continue
            if check_path:
                segment = query.segment(q_last, trial)
                if segment["subdivision_limit"] or not query.acceptable(old_segment["depths"], segment["depths"]):
                    continue
            return trial
        return before.copy()

    def _self_collision_initial_guess(self, initial, previous):
        """Warm starts must be reachable from the actual previous output.

        A separately solved baseline can be clear at both ends yet cross a
        limb on the playback path. It cannot set the allowed penetration for
        the new solve. Retain its root and shorten only the internal warm-start
        displacement; the subsequent SQP still solves the current source tasks.
        """
        if previous is None or not getattr(self, "self_collision_enabled", False):
            return initial.copy()
        query = self._get_self_collision_query()
        allowed = query.depths(previous)
        scalar = np.asarray([
            self.robot_model.jnt_qposadr[j] for j in range(self.robot_model.njnt)
            if self.robot_model.jnt_type[j] in (
                mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE
            )
        ], dtype=int)
        for alpha in _COLLISION_BACKTRACK_FRACTIONS:
            trial = initial.copy()
            trial[scalar] = previous[scalar] + alpha * (initial[scalar] - previous[scalar])
            path = query.segment(previous, trial)
            if not path["subdivision_limit"] and query.acceptable(allowed, path["depths"]):
                return trial
        raise RuntimeError("Cannot validate self-collision warm start within subdivision budget")


    def _compute_penetration_constraints(self, q: np.ndarray, dqa: cp.Variable) -> List[cp.Constraint]:
        """
        Compute robot-environment penetration constraints.

        Scene constraints use the shared signed query and collision-shell
        samples. Only confirmed free, distant geoms may be culled. Reliable
        samples use their full exit gradient; unknown sides are reported and
        never assigned an arbitrary upward hard constraint.

        Robot self-collision is handled independently by _self_collision_objective.

        The terrain mesh is NOT embedded in the MuJoCo model, so MuJoCo's own
        collision pipeline cannot detect robot-terrain contacts.  We handle them
        using the shared scene query and the robot's point Jacobian.
        """
        constraints = []

        # Ensure kinematics are current
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)

        threshold = float(self.collision_detection_threshold)

        # ------------------------------------------------------------------
        # Robot-terrain penetration constraints (via trimesh proximity).
        constraints.extend(
            self._compute_terrain_penetration_constraints(q, dqa, threshold)
        )

        return constraints


    def _compute_terrain_penetration_constraints(
        self, q: np.ndarray, dqa: cp.Variable, threshold: float
    ) -> List[cp.Constraint]:
        """Linearize the same signed geometry used by refinement/acceptance."""
        from ..scene_geometry import ScenePenetrationProbe, collision_geom_ids
        if not hasattr(self, "_scene_probe"):
            self._scene_probe = ScenePenetrationProbe(
                self.robot_model, self.terrain_mesh,
                collision_geom_ids(self.robot_model), threshold,
                query=getattr(self, "scene_query", None),
            )
        frame = self._scene_probe.probe_trajectory(np.asarray(q)[None])[0]
        constraints = []
        active = np.flatnonzero((frame["clearance"] <= threshold) & ~frame["uncertain"])
        if np.any(frame["uncertain"]) and not getattr(self, "_reported_uncertain_scene", False):
            print("Scene query: unknown free side; ambiguous samples remain unresolved (no hard direction).")
            self._reported_uncertain_scene = True
        for k in active:
            jac = self._calc_contact_jacobian_from_point(
                int(frame["body_ids"][k]), frame["points"][k], input_world=True,
            )
            row = frame["directions"][k] @ jac
            constraints.append(row[self.q_a_indices] @ dqa >= -frame["clearance"][k] - self.penetration_tolerance)
        return constraints
