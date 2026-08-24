"""Core retargeting functionality adapted for generic robots and terrains."""

from __future__ import annotations

import numpy as np
import mujoco
import cvxpy as cp
from scipy import sparse as sp
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation
import trimesh
from typing import Any, Dict, List, Mapping, Optional, Tuple
from pathlib import Path
import yourdfpy

from .utils import (
    load_terrain_mesh,
    sample_points_on_mesh,
    scale_mesh,
    compute_mesh_height_at_point,
    validate_smplx_trajectory,
    transform_points_local_to_world,
    get_adjacency_list,
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
)


def _transform_swing_target(
    tracking_position: np.ndarray,
    source_root_position: np.ndarray | None,
    robot_root_z_offset: float,
    morphology_scale: float,
    preserve_world_target: bool = False,
) -> np.ndarray:
    """Express a source end-effector target around the robot root target.

    The root-height offset is a world-space registration correction and must
    therefore be applied even when ``morphology_scale`` is exactly one.  The
    previous implementation skipped the entire transform at unit scale, which
    left swing hands and feet at the human actor's height after the shorter
    robot root had been moved down.
    """
    target = np.asarray(tracking_position, dtype=float).reshape(3)
    if preserve_world_target or source_root_position is None:
        return target.copy()
    source_root = np.asarray(source_root_position, dtype=float).reshape(3)
    robot_root_target = source_root.copy()
    robot_root_target[2] += float(robot_root_z_offset)
    return robot_root_target + float(morphology_scale) * (target - source_root)


def _clamp_target_to_reach(
    target: np.ndarray,
    reach_origin: np.ndarray,
    max_reach: float | None,
) -> np.ndarray:
    """Project an unreachable swing target onto a robot reach sphere.

    Contact targets are intentionally not passed through this helper: they
    describe real scene interactions and must stay in world coordinates.  For
    free-space motion, chasing an impossible target only drives redundant arm
    joints into their limits and creates an unnatural straight or folded arm.
    """
    result = np.asarray(target, dtype=float).reshape(3).copy()
    if max_reach is None or float(max_reach) <= 0.0:
        return result
    origin = np.asarray(reach_origin, dtype=float).reshape(3)
    delta = result - origin
    distance = float(np.linalg.norm(delta))
    if distance <= float(max_reach) or distance <= 1e-10:
        return result
    return origin + delta * (float(max_reach) / distance)


def _apply_configured_joint_limits(
    model: mujoco.MjModel,
    qpos_indices: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    configured_limits: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Intersect mechanical limits with configured task-safe joint bounds."""
    result_lower = np.asarray(lower, dtype=float).copy()
    result_upper = np.asarray(upper, dtype=float).copy()
    coordinate_by_qpos = {
        int(qpos_idx): coordinate_idx
        for coordinate_idx, qpos_idx in enumerate(qpos_indices)
    }
    for joint_name, raw_bounds in dict(configured_limits or {}).items():
        bounds = np.asarray(raw_bounds, dtype=float).reshape(-1)
        if bounds.shape != (2,):
            raise ValueError(
                f"joint_posture_limits['{joint_name}'] must contain [lower, upper]."
            )
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name)
        )
        if joint_id < 0:
            raise ValueError(
                f"joint_posture_limits refers to missing joint '{joint_name}'."
            )
        qpos_idx = int(model.jnt_qposadr[joint_id])
        coordinate_idx = coordinate_by_qpos.get(qpos_idx)
        if coordinate_idx is None:
            continue
        result_lower[coordinate_idx] = max(
            result_lower[coordinate_idx], float(bounds[0])
        )
        result_upper[coordinate_idx] = min(
            result_upper[coordinate_idx], float(bounds[1])
        )
        if result_lower[coordinate_idx] > result_upper[coordinate_idx]:
            raise ValueError(
                f"joint_posture_limits for '{joint_name}' do not overlap its "
                "mechanical range."
            )
    return result_lower, result_upper


class GenericInteractionRetargeter:
    """
    Generic interaction mesh retargeter that works with any robot and terrain.

    This adapts the interaction mesh retargeting approach from holosoma_retargeting
    to work with generic URDF robots and terrain meshes.
    """

    def __init__(
        self,
        robot_model: mujoco.MjModel,
        robot_data: mujoco.MjData,
        terrain_mesh: trimesh.Trimesh,
        joint_mapping: Dict[str, str],
        robot_height: float,
        q_a_init_idx: int = -7,
        step_size: float = 0.2,
        penetration_tolerance: float = 1e-3,
        foot_sticking_tolerance: float = 1e-3,
        collision_detection_threshold: float = 0.1,
        terrain_sample_points: int = 100,
        terrain_sample_seed: Optional[int] = None,
        valid_joint_names: Optional[List[str]] = None,
        replace_cylinders_with_capsules: bool = False,
        hard_penetration_constraint: bool = False,
        preserve_root_z: bool = False,
        end_effector_config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the generic retargeter.

        Args:
            robot_model: MuJoCo model of the robot
            robot_data: MuJoCo data for the robot
            terrain_mesh: Terrain mesh (already scaled if needed)
            joint_mapping: Mapping from SMPLX joint names to robot link names
            robot_height: Height of the robot
            q_a_init_idx: Index where optimization variables start
            step_size: Trust region size for SQP
            penetration_tolerance: Tolerance for penetration constraints
            foot_sticking_tolerance: Tolerance for foot sticking
            collision_detection_threshold: Distance threshold for collision detection
            terrain_sample_points: Number of sampled terrain points for interaction mesh
            terrain_sample_seed: Seed for the terrain surface draw. Leave unset for the
                legacy random draw; set it to compare two runs against each other.
            valid_joint_names: Ordered list of joint names to ensure consistent ordering
            replace_cylinders_with_capsules: If True, replace all cylinder collision geoms
                with capsules before computing penetration constraints. This matches
                IsaacLab/PhysX convention where ``replace_cylinders_with_capsules=True``
                is commonly used, ensuring that the retargeted motion is checked against
                the same collision shapes used in downstream simulation.
            hard_penetration_constraint: If True, enforce penetration
                constraints inside the optimizer. If False, skip them so
                outer post-processing can handle contact correction.
            preserve_root_z: If True, keep qpos[2] out of the SQP variables so
                root height is supplied by the caller and not changed by IK.
            end_effector_config: Optional contact-aware hand/foot task
                configuration.  When enabled, contact, swing end-effector and
                whole-body objectives are solved lexicographically.
        """
        self.robot_model = robot_model
        self.robot_data = robot_data
        self.terrain_mesh = terrain_mesh
        self.joint_mapping = joint_mapping  # This should already be filtered to valid joints only
        self.robot_height = robot_height
        self.hard_penetration_constraint = hard_penetration_constraint
        self.end_effector_config = dict(end_effector_config or {})
        
        # CRITICAL: Store ordered joint names to ensure consistent ordering
        # This ensures human_joints[i] matches robot_points[i] for all i
        if valid_joint_names is not None:
            self.valid_joint_names = valid_joint_names
            # Verify that valid_joint_names matches joint_mapping keys
            if set(self.valid_joint_names) != set(joint_mapping.keys()):
                raise ValueError(
                    f"valid_joint_names ({set(self.valid_joint_names)}) "
                    f"does not match joint_mapping keys ({set(joint_mapping.keys())})"
                )
        else:
            # Fallback: use dictionary insertion order (Python 3.7+)
            self.valid_joint_names = list(joint_mapping.keys())
        
        # Validate that all joints in mapping exist in robot
        # This is a final safety check - fail fast if joints are missing
        self._validate_joint_mapping()

        # Retargeting parameters
        self.q_a_init_idx = q_a_init_idx
        self.step_size = step_size
        self.penetration_tolerance = penetration_tolerance
        self.foot_sticking_tolerance = foot_sticking_tolerance
        self.collision_detection_threshold = collision_detection_threshold
        self.terrain_sample_points = int(terrain_sample_points)
        self.terrain_sample_seed = (
            None if terrain_sample_seed is None else int(terrain_sample_seed)
        )
        self.preserve_root_z = bool(preserve_root_z)

        # Apply cylinder → capsule replacement if requested
        if replace_cylinders_with_capsules:
            self._replace_cylinders_with_capsules()

        # Setup robot configuration
        self._setup_robot_config()

        # Resolve robot palm/sole markers after q_a_indices are available.
        self._setup_end_effectors()

        # Resolve morphology-aware bone-direction tasks.  End-effector IK
        # fixes hand/foot positions but otherwise leaves the elbow/knee branch
        # ambiguous; these tasks select the branch observed in the source.
        self._setup_pose_bones()

        # Setup terrain interaction
        self._setup_terrain_interaction()

    def _replace_cylinders_with_capsules(self):
        """Replace all cylinder collision geoms with capsules in the MuJoCo model.

        A URDF ``<cylinder>`` has flat end-caps, while a capsule adds
        hemispherical caps of the same radius.  MuJoCo keeps the same
        ``size`` layout for both types (``[radius, half_length]``), so
        the only change needed is the ``geom_type`` field.

        This is done **in-place** on ``self.robot_model`` so that all
        subsequent calls to ``mj_collision`` / ``mj_geomDistance`` use
        capsule geometry — matching IsaacLab's
        ``replace_cylinders_with_capsules=True`` convention.
        """
        m = self.robot_model
        n_replaced = 0
        for gi in range(m.ngeom):
            if m.geom_type[gi] == mujoco.mjtGeom.mjGEOM_CYLINDER:
                m.geom_type[gi] = mujoco.mjtGeom.mjGEOM_CAPSULE
                n_replaced += 1
        if n_replaced > 0:
            print(f"Replaced {n_replaced} cylinder geom(s) with capsules for collision.")

    def _setup_robot_config(self):
        """Setup robot configuration parameters."""
        self.nq = self.robot_model.nq
        self.nv = self.robot_model.nv
        # Determine which qpos indices are optimized.
        # q_a_init_idx follows the original convention:
        #   -7: include floating base (0..nq)
        #    0: start at actuated joints (after floating base)
        #   12: start at waist, etc.
        # This assumes standard MuJoCo convention:
        # qpos structure: [floating_base (7), joint1 (1), joint2 (1), ...]
        start_idx = 7 + self.q_a_init_idx
        start_idx = int(np.clip(start_idx, 0, self.nq))
        self.q_a_indices = np.arange(start_idx, self.nq)
        if self.preserve_root_z and 2 in self.q_a_indices:
            self.q_a_indices = self.q_a_indices[self.q_a_indices != 2]
        self.nq_a = len(self.q_a_indices)
        
        print(f"Robot config: nq={self.nq}, nv={self.nv}, nq_a={self.nq_a}")
        if self.nq_a > 0:
            print(f"q_a_indices range: {self.q_a_indices.min()} to {self.q_a_indices.max()}")
        if self.preserve_root_z:
            print("Root z is fixed: qpos[2] is not optimized.")

        # Joint limits
        joint_names = [self.robot_model.joint(i).name for i in range(self.robot_model.njnt)]
        actuated_joints = [(i, name) for i, name in enumerate(joint_names) if name]
        
        large_number = 1e6
        # Construct full limits array matching nq size
        # Start with floating base limits (unbounded)
        full_lower_limits = -large_number * np.ones(self.nq)
        full_upper_limits = large_number * np.ones(self.nq)
        
        # Fill in limits for actuated joints
        # This assumes joint addresses are contiguous after the base
        # Depending on the robot model, we might need to be more careful here
        # But for standard humanoids this usually holds
        
        # Typically self.robot_model.jnt_qposadr gives the index in qpos for each joint
        for i in range(self.robot_model.njnt):
            qpos_adr = self.robot_model.jnt_qposadr[i]
            if qpos_adr >= 7: # Skip root joint(s) if they are part of the base
                # For 1-DOF joints
                full_lower_limits[qpos_adr] = self.robot_model.jnt_range[i, 0]
                full_upper_limits[qpos_adr] = self.robot_model.jnt_range[i, 1]

        self.q_a_lb, self.q_a_ub = _apply_configured_joint_limits(
            self.robot_model,
            self.q_a_indices,
            full_lower_limits[self.q_a_indices],
            full_upper_limits[self.q_a_indices],
            self.end_effector_config.get("joint_posture_limits"),
        )

        # Joint cost weights - small regularization to prevent extreme angles
        # Floating base (first 7 DOF) gets very small weight, joints get moderate weight
        self.Q_diag = np.ones(self.nq_a) * 1e-3  # Small default regularization (matching original)
        
        # Reduce weight for floating base to allow free movement
        base_indices_in_qa = []
        for base_idx in range(7):
            if base_idx in self.q_a_indices:
                idx_in_qa = np.where(self.q_a_indices == base_idx)[0]
                if len(idx_in_qa) > 0:
                    base_indices_in_qa.append(idx_in_qa[0])
        
        if len(base_indices_in_qa) > 0:
            self.Q_diag[base_indices_in_qa] = 0.001  # Very small weight for base
        
        # Store smoothness weight (matching original: 0.2)
        self.smooth_weight = 0.2

    def _setup_end_effectors(self) -> None:
        """Resolve configured robot palm/sole markers.

        Offsets are expressed in the corresponding MuJoCo body frame.  A
        contact task may expose several surface points (for example heel,
        toe, and sole corners); the point closest to the human-derived scene
        anchor is selected when a contact run starts and then kept for the
        whole run, so the active marker cannot chatter between neighbouring
        offsets across SQP iterations or frames.  This avoids incorrectly
        forcing the ankle or wrist joint centre onto the scene surface.
        """
        cfg = self.end_effector_config
        self.end_effector_enabled = bool(
            cfg.get("enabled", False) or cfg.get("world_tracking_only", False)
        )
        self.end_effectors: Dict[str, Dict[str, Any]] = {}
        self._active_contact_offsets: Dict[str, int] = {}
        if not self.end_effector_enabled:
            return

        specs = dict(cfg.get("effectors", {}) or {})
        if bool(cfg.get("world_tracking_only", False)) and not bool(
            cfg.get("enabled", False)
        ):
            selected_names = set(
                cfg.get(
                    "world_tracking_effectors",
                    ["left_foot", "right_foot"],
                )
            )
            specs = {
                name: spec for name, spec in specs.items() if name in selected_names
            }
        for name, raw_spec in specs.items():
            spec = dict(raw_spec or {})
            body_name = spec.get("robot_body")
            if not body_name:
                raise ValueError(
                    f"End-effector '{name}' is missing required 'robot_body'."
                )
            body_id = mujoco.mj_name2id(
                self.robot_model, mujoco.mjtObj.mjOBJ_BODY, str(body_name)
            )
            if body_id < 0:
                raise ValueError(
                    f"End-effector '{name}' refers to missing robot body "
                    f"'{body_name}'."
                )

            reach_reference_body = spec.get("reach_reference_body")
            reach_reference_body_id = -1
            if reach_reference_body:
                reach_reference_body_id = mujoco.mj_name2id(
                    self.robot_model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    str(reach_reference_body),
                )
                if reach_reference_body_id < 0:
                    raise ValueError(
                        f"End-effector '{name}' refers to missing reach reference "
                        f"body '{reach_reference_body}'."
                    )

            tracking_offset = np.asarray(
                spec.get("tracking_offset", [0.0, 0.0, 0.0]), dtype=float
            ).reshape(3)
            contact_offsets = np.asarray(
                spec.get("contact_offsets", [tracking_offset.tolist()]), dtype=float
            )
            if contact_offsets.ndim == 1:
                contact_offsets = contact_offsets[None, :]
            if contact_offsets.ndim != 2 or contact_offsets.shape[1] != 3:
                raise ValueError(
                    f"End-effector '{name}' contact_offsets must have shape (K, 3)."
                )

            self.end_effectors[name] = {
                "body_name": str(body_name),
                "body_id": int(body_id),
                "tracking_offset": tracking_offset,
                "contact_offsets": contact_offsets,
                "track_orientation": bool(spec.get("track_orientation", False)),
                "track_swing_heading": bool(
                    spec.get("track_swing_heading", False)
                ),
                "track_swing_orientation": bool(
                    spec.get("track_swing_orientation", False)
                ),
                "orientation_sigma": float(
                    spec.get(
                        "orientation_sigma",
                        cfg.get("orientation_sigma", np.deg2rad(7.5)),
                    )
                ),
                "max_orientation_step": float(
                    spec.get("max_orientation_step", np.deg2rad(6.0))
                ),
                "contact_sigma": float(
                    spec.get("contact_sigma", cfg.get("contact_sigma", 0.025))
                ),
                "swing_sigma": float(
                    spec.get("swing_sigma", cfg.get("swing_sigma", 0.06))
                ),
                "swing_morphology_scale": float(
                    spec.get(
                        "swing_morphology_scale",
                        cfg.get("swing_morphology_scale", 1.0),
                    )
                ),
                "preserve_world_target": bool(
                    spec.get("preserve_world_target", False)
                ),
                "reach_reference_body_id": int(reach_reference_body_id),
                "reach_reference_offset": np.asarray(
                    spec.get("reach_reference_offset", [0.0, 0.0, 0.0]),
                    dtype=float,
                ).reshape(3),
                "max_swing_reach": (
                    None
                    if spec.get("max_swing_reach") is None
                    else float(spec["max_swing_reach"])
                ),
            }

        if not self.end_effectors:
            raise ValueError(
                "end_effector_priority is enabled but no effectors were configured."
            )
        mode_label = (
            "World-tracking end effectors"
            if bool(cfg.get("world_tracking_only", False))
            and not bool(cfg.get("enabled", False))
            else "Contact-priority end effectors"
        )
        print(
            f"{mode_label}: "
            + ", ".join(
                f"{name}->{spec['body_name']}" for name, spec in self.end_effectors.items()
            )
        )

    def _setup_pose_bones(self) -> None:
        """Cache robot bone lengths for source/robot direction matching."""
        pose_cfg = dict(self.end_effector_config.get("pose_direction", {}) or {})
        self.pose_direction_enabled = bool(pose_cfg.get("enabled", True))
        # The whole-step bisection projection rescales every optimized
        # coordinate toward the previous frame, so it can undo contact and
        # swing progress that the convex solve already secured.  Profiles that
        # rely on the in-problem direction constraints can turn it off.
        self.pose_direction_post_projection = bool(
            pose_cfg.get("post_projection", True)
        )
        # The direction-velocity target transports an ambient difference
        # between different points of the unit sphere, which can fight the
        # position-level direction matching; profiles can keep only the
        # position-level term.
        self.pose_velocity_enabled = bool(pose_cfg.get("velocity_enabled", True))
        self.pose_direction_sigma = max(float(pose_cfg.get("sigma", 0.12)), 1e-6)
        self.pose_arm_weight = max(float(pose_cfg.get("arm_weight", 1.0)), 0.0)
        self.pose_leg_weight = max(float(pose_cfg.get("leg_weight", 1.0)), 0.0)
        self.pose_max_direction_step = max(
            float(pose_cfg.get("max_direction_step", np.deg2rad(6.0))), 1e-6
        )
        self.pose_min_direction_step = max(
            float(pose_cfg.get("min_direction_step", np.deg2rad(1.5))), 1e-6
        )
        self.pose_direction_step_margin = max(
            float(pose_cfg.get("direction_step_margin", np.deg2rad(1.0))), 0.0
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
        raw_edges = pose_cfg.get(
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
        self.pose_bones: List[Tuple[str, str, float]] = []
        if not self.pose_direction_enabled:
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
            is_leg = any(
                token in parent_name or token in child_name
                for token in ("Hip", "Knee", "Ankle")
            )
            direction_weight = (
                self.pose_leg_weight if is_leg else self.pose_arm_weight
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

    def _build_pose_direction_constraints(
        self,
        dqa: cp.Variable,
        robot_positions: Mapping[str, np.ndarray],
        robot_jacobian: np.ndarray,
        last_directions: Mapping[Tuple[str, str], np.ndarray],
        direction_step_limits: Mapping[Tuple[str, str], float],
    ) -> List[cp.Constraint]:
        """Bound visible bone rotation between output frames."""
        if not self.pose_direction_enabled or not last_directions:
            return []
        name_to_index = {
            name: index for index, name in enumerate(self.valid_joint_names)
        }
        constraints: List[cp.Constraint] = []
        for parent_name, child_name, _ in self.pose_bones:
            edge = (parent_name, child_name)
            if edge not in last_directions:
                continue
            parent_idx = name_to_index[parent_name]
            child_idx = name_to_index[child_name]
            vector = robot_positions[child_name] - robot_positions[parent_name]
            length = float(np.linalg.norm(vector))
            if length < 1e-8:
                continue
            direction = vector / length
            vector_jacobian = (
                robot_jacobian[3 * child_idx : 3 * child_idx + 3]
                - robot_jacobian[3 * parent_idx : 3 * parent_idx + 3]
            )
            direction_jacobian = (
                np.eye(3) - np.outer(direction, direction)
            ) @ vector_jacobian / length
            constraints.append(
                cp.norm(
                    direction
                    + direction_jacobian @ dqa
                    - np.asarray(last_directions[edge], dtype=float),
                    2,
                )
                <= float(
                    direction_step_limits.get(
                        edge, self.pose_max_direction_step
                    )
                )
            )
        return constraints

    def _pose_direction_step_limits(
        self,
        target_human_joints: Optional[np.ndarray],
        previous_human_joints: Optional[np.ndarray],
    ) -> Dict[Tuple[str, str], float]:
        """Set bone speed limits from the captured per-frame bone motion."""
        limits: Dict[Tuple[str, str], float] = {}
        if target_human_joints is None or previous_human_joints is None:
            return limits
        name_to_index = {
            name: index for index, name in enumerate(self.valid_joint_names)
        }
        current = np.asarray(target_human_joints, dtype=float)
        previous = np.asarray(previous_human_joints, dtype=float)
        for parent_name, child_name, _ in self.pose_bones:
            parent_idx = name_to_index[parent_name]
            child_idx = name_to_index[child_name]
            current_vector = current[child_idx] - current[parent_idx]
            previous_vector = previous[child_idx] - previous[parent_idx]
            current_norm = float(np.linalg.norm(current_vector))
            previous_norm = float(np.linalg.norm(previous_vector))
            if current_norm < 1e-8 or previous_norm < 1e-8:
                continue
            source_step = float(
                np.arccos(
                    np.clip(
                        np.dot(current_vector, previous_vector)
                        / (current_norm * previous_norm),
                        -1.0,
                        1.0,
                    )
                )
            )
            limits[(parent_name, child_name)] = float(
                np.clip(
                    source_step + self.pose_direction_step_margin,
                    self.pose_min_direction_step,
                    self.pose_max_direction_step,
                )
            )
        return limits

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

    def _build_end_effector_objectives(
        self,
        dqa: cp.Variable,
        targets: Optional[Mapping[str, Mapping[str, Any]]],
        last_rotations: Optional[Mapping[str, np.ndarray]] = None,
    ) -> Tuple[
        Optional[cp.Expression],
        Optional[cp.Expression],
        Optional[cp.Expression],
        List[cp.Constraint],
        int,
        int,
        int,
    ]:
        """Linearise contact and swing tracking objectives for one SQP step.

        Contact and swing expressions are deliberately returned separately so
        the caller can solve them lexicographically.  The normalization by a
        metric tolerance ``sigma`` makes each squared residual dimensionless.
        """
        if not self.end_effector_enabled or not targets:
            return None, None, None, [], 0, 0, 0

        contact_terms: List[cp.Expression] = []
        swing_terms: List[cp.Expression] = []
        swing_orientation_terms: List[cp.Expression] = []
        contact_constraints: List[cp.Constraint] = []
        for name, target_data in targets.items():
            if name not in self.end_effectors:
                continue
            spec = self.end_effectors[name]
            body_id = spec["body_id"]
            confidence = float(np.clip(target_data.get("confidence", 1.0), 0.0, 1.0))
            forward_direction = target_data.get("forward_direction")
            up_direction = target_data.get("up_direction")

            if bool(target_data.get("contact", False)):
                target = np.asarray(target_data["contact_position"], dtype=float).reshape(3)
                offset_index = self._active_contact_offsets.get(name)
                if offset_index is None:
                    offset_index = int(
                        np.argmin(
                            [
                                np.linalg.norm(
                                    self._point_position_and_jacobian(body_id, offset)[0]
                                    - target
                                )
                                for offset in spec["contact_offsets"]
                            ]
                        )
                    )
                    self._active_contact_offsets[name] = offset_index
                position, jacobian = self._point_position_and_jacobian(
                    body_id, spec["contact_offsets"][offset_index]
                )
                sigma = max(float(spec["contact_sigma"]), 1e-6)
                # Keep every detected contact in the primary level. Confidence
                # only ranks contacts within that same priority level.
                weight = max(confidence, 0.25)
                residual = position + jacobian @ dqa - target
                contact_terms.append(np.sqrt(weight) * residual / sigma)
                surface_normal = np.asarray(
                    target_data.get("surface_normal", [0.0, 0.0, 1.0]),
                    dtype=float,
                ).reshape(3)
                normal_norm = float(np.linalg.norm(surface_normal))
                if normal_norm > 1e-8:
                    surface_normal /= normal_norm
                    normal_tolerance = max(
                        float(
                            self.end_effector_config.get(
                                "contact_normal_tolerance", 0.005
                            )
                        ),
                        0.0,
                    )
                    # The inferred normal is oriented from the scene toward
                    # the source SMPL-X surface. Keep the robot marker on that
                    # same side of the contact plane.
                    contact_constraints.append(
                        surface_normal @ residual >= -normal_tolerance
                    )

                    if spec.get("track_orientation", False) and forward_direction is not None:
                        forward = np.asarray(forward_direction, dtype=float).reshape(3)
                        orientation_up = np.asarray(
                            up_direction
                            if up_direction is not None
                            else surface_normal,
                            dtype=float,
                        ).reshape(3)
                        orientation_up /= max(
                            float(np.linalg.norm(orientation_up)), 1e-10
                        )
                        # The contact detector has already verified that the
                        # captured sole is compatible with the surface.  Track
                        # the captured rigid foot frame rather than replacing
                        # its pitch/roll with a noisy mesh-triangle normal.
                        forward -= orientation_up * float(
                            np.dot(orientation_up, forward)
                        )
                        forward_norm = float(np.linalg.norm(forward))
                        if forward_norm > 1e-8:
                            forward /= forward_norm
                            lateral = np.cross(orientation_up, forward)
                            lateral /= max(float(np.linalg.norm(lateral)), 1e-10)
                            forward = np.cross(lateral, orientation_up)
                            target_rotation = np.column_stack(
                                [forward, lateral, orientation_up]
                            )
                            current_rotation = self.robot_data.xmat[body_id].reshape(
                                3, 3
                            )
                            rotation_error = Rotation.from_matrix(
                                target_rotation @ current_rotation.T
                            ).as_rotvec()
                            angular_jacobian = self._body_angular_jacobian(body_id)
                            orientation_sigma = max(
                                float(
                                    spec.get(
                                        "orientation_sigma",
                                        self.end_effector_config.get(
                                            "orientation_sigma", np.deg2rad(7.5)
                                        ),
                                    )
                                ),
                                1e-6,
                            )
                            contact_terms.append(
                                np.sqrt(weight)
                                * (angular_jacobian @ dqa - rotation_error)
                                / orientation_sigma
                            )
            else:
                self._active_contact_offsets.pop(name, None)
                target = np.asarray(target_data["tracking_position"], dtype=float).reshape(3)
                source_root_position = target_data.get("source_root_position")
                swing_morphology_scale = float(spec["swing_morphology_scale"])
                root_z_offset = float(
                    (
                        self.end_effector_config.get(
                            "base_position_constraint", {}
                        )
                        or {}
                    ).get("z_offset", 0.0)
                )
                # Contact positions remain exact world-space constraints.
                # Swing targets follow the registered robot root and may also
                # be scaled root-relatively for morphology.  Root registration
                # still applies when morphology_scale == 1.0.
                target = _transform_swing_target(
                    target,
                    source_root_position,
                    root_z_offset,
                    swing_morphology_scale,
                    bool(spec["preserve_world_target"]),
                )
                reach_body_id = int(spec["reach_reference_body_id"])
                if reach_body_id >= 0:
                    reach_rotation = self.robot_data.xmat[reach_body_id].reshape(3, 3)
                    reach_origin = (
                        self.robot_data.xpos[reach_body_id]
                        + reach_rotation @ spec["reach_reference_offset"]
                    )
                    target = _clamp_target_to_reach(
                        target,
                        reach_origin,
                        spec["max_swing_reach"],
                    )
                position, jacobian = self._point_position_and_jacobian(
                    body_id, spec["tracking_offset"]
                )
                sigma = max(float(spec["swing_sigma"]), 1e-6)
                residual = position + jacobian @ dqa - target
                swing_terms.append(residual / sigma)
                if (
                    spec.get("track_orientation", False)
                    and spec.get("track_swing_orientation", False)
                    and forward_direction is not None
                    and up_direction is not None
                ):
                    target_up = np.asarray(up_direction, dtype=float).reshape(3)
                    target_up /= max(float(np.linalg.norm(target_up)), 1e-10)
                    target_forward = np.asarray(forward_direction, dtype=float).reshape(3)
                    target_forward -= target_up * float(
                        np.dot(target_up, target_forward)
                    )
                    forward_norm = float(np.linalg.norm(target_forward))
                    if forward_norm > 1e-8:
                        target_forward /= forward_norm
                        target_lateral = np.cross(target_up, target_forward)
                        target_lateral /= max(
                            float(np.linalg.norm(target_lateral)), 1e-10
                        )
                        target_forward = np.cross(target_lateral, target_up)
                        target_rotation = np.column_stack(
                            [target_forward, target_lateral, target_up]
                        )
                        current_rotation = self.robot_data.xmat[body_id].reshape(3, 3)
                        rotation_error = Rotation.from_matrix(
                            target_rotation @ current_rotation.T
                        ).as_rotvec()
                        angular_jacobian = self._body_angular_jacobian(body_id)
                        orientation_sigma = max(float(spec["orientation_sigma"]), 1e-6)
                        swing_orientation_terms.append(
                            (angular_jacobian @ dqa - rotation_error)
                            / orientation_sigma
                        )
                elif (
                    spec.get("track_orientation", False)
                    and spec.get("track_swing_heading", False)
                    and forward_direction is not None
                ):
                    target_forward = np.asarray(forward_direction, dtype=float).reshape(3)
                    target_forward[2] = 0.0
                    current_rotation = self.robot_data.xmat[body_id].reshape(3, 3)
                    current_forward = current_rotation[:, 0].copy()
                    current_forward[2] = 0.0
                    target_norm = float(np.linalg.norm(target_forward))
                    current_norm = float(np.linalg.norm(current_forward))
                    if target_norm > 1e-8 and current_norm > 1e-8:
                        target_forward /= target_norm
                        current_forward /= current_norm
                        yaw_error = float(
                            np.arctan2(
                                np.cross(current_forward, target_forward)[2],
                                np.dot(current_forward, target_forward),
                            )
                        )
                        angular_jacobian = self._body_angular_jacobian(body_id)
                        orientation_sigma = max(float(spec["orientation_sigma"]), 1e-6)
                        swing_orientation_terms.append(
                            cp.hstack([angular_jacobian[2] @ dqa - yaw_error])
                            / orientation_sigma
                        )

            # This is a physical angular-velocity bound, not a soft smoothing
            # term.  It prevents a newly activated contact from rotating an
            # already stable foot when simultaneous human anchors are not
            # exactly feasible for the robot's morphology.
            if (
                spec.get("track_orientation", False)
                and last_rotations is not None
                and name in last_rotations
            ):
                current_rotation = self.robot_data.xmat[body_id].reshape(3, 3)
                rotation_since_last = Rotation.from_matrix(
                    current_rotation @ np.asarray(last_rotations[name]).T
                ).as_rotvec()
                angular_jacobian = self._body_angular_jacobian(body_id)
                max_orientation_step = max(
                    float(spec.get("max_orientation_step", np.deg2rad(6.0))),
                    1e-6,
                )
                contact_constraints.append(
                    cp.norm(rotation_since_last + angular_jacobian @ dqa, 2)
                    <= max_orientation_step
                )

        contact_residual = cp.hstack(contact_terms) if contact_terms else None
        swing_residual = cp.hstack(swing_terms) if swing_terms else None
        swing_orientation_residual = (
            cp.hstack(swing_orientation_terms)
            if swing_orientation_terms
            else None
        )
        # A foot pose is one SE(3) task: translating the sole marker and
        # rotating its rigid frame are coupled by the ankle kinematics.  When
        # requested, optimize both in one dimensionless Pareto level using the
        # configured metre/radian sigmas.  Separating them lexicographically
        # makes the orientation level inherit an already saturated ankle and
        # can leave the toe axis tens of degrees from the captured foot.
        if (
            bool(self.end_effector_config.get("combine_swing_foot_pose", False))
            and swing_residual is not None
            and swing_orientation_residual is not None
        ):
            swing_residual = cp.hstack(
                [swing_residual, swing_orientation_residual]
            )
            swing_orientation_residual = None
        return (
            contact_residual,
            swing_residual,
            swing_orientation_residual,
            contact_constraints,
            len(contact_terms),
            len(swing_terms),
            len(swing_orientation_terms),
        )
    
    def _validate_joint_mapping(self):
        """Validate that all joints in mapping exist in robot. Raise error if any are missing."""
        # Get all body names from robot
        robot_bodies = set()
        for i in range(self.robot_model.nbody):
            body_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name:
                robot_bodies.add(body_name)
        
        # Check all mapped bodies exist
        missing_bodies = []
        for joint_name, link_name in self.joint_mapping.items():
            if link_name not in robot_bodies:
                missing_bodies.append((joint_name, link_name))
        
        if missing_bodies:
            raise ValueError(
                f"The following robot links from joint_mapping were not found in URDF: {missing_bodies}. "
                f"Please check your joint_mapping. Available bodies: {sorted(list(robot_bodies))[:10]}..."
            )

    def _setup_terrain_interaction(self):
        """Setup terrain interaction parameters."""
        # Sample points on terrain for interaction mesh
        self.terrain_points = sample_points_on_mesh(
            self.terrain_mesh, self.terrain_sample_points, seed=self.terrain_sample_seed
        )


    def create_interaction_mesh(self, human_joints: np.ndarray, terrain_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create interaction mesh from human joints and terrain points.

        Args:
            human_joints: Human joint positions (N, 3)
            terrain_points: Terrain surface points (M, 3)

        Returns:
            Tuple of (vertices, tetrahedra)
        """
        # Combine human joints and terrain points
        vertices = np.vstack([human_joints, terrain_points])

        # Create Delaunay triangulation
        tri = Delaunay(vertices)

        return vertices, tri.simplices

    def retarget_frame(
        self,
        human_joints: np.ndarray,
        q_init: np.ndarray,
        max_iter: int = 10,
        q_last: Optional[np.ndarray] = None,
        q_prev: Optional[np.ndarray] = None,
        q_prevprev: Optional[np.ndarray] = None,
        previous_human_joints: Optional[np.ndarray] = None,
        previous_previous_human_joints: Optional[np.ndarray] = None,
        target_base_orientation: Optional[np.ndarray] = None,
        target_base_position: Optional[np.ndarray] = None,
        end_effector_targets: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> np.ndarray:
        """
        Retarget a single frame of human motion to robot motion.

        Args:
            human_joints: Human joint positions (N, 3)
            q_init: Initial robot configuration
            max_iter: Maximum optimization iterations
            q_last: Configuration at previous time step (for smoothness)

        Returns:
            Optimized robot configuration
        """
        # Note: self.terrain_points are sampled from self.terrain_mesh
        # In core.py, self.terrain_mesh is already the *scaled* terrain mesh passed to __init__
        # So we don't need to apply terrain_scale again here.
        # We use self.terrain_points directly.
        scaled_terrain_points = self.terrain_points 

        # Create interaction mesh
        vertices, tetrahedra = self.create_interaction_mesh(human_joints, scaled_terrain_points)

        # Create adjacency list
        adj_list = get_adjacency_list(tetrahedra, len(vertices))

        # Calculate target Laplacian coordinates
        # CRITICAL: Use uniform_weight=True to match the matrix computation in optimization
        # This ensures target_laplacian and lap0 use the same weighting scheme
        target_laplacian = calculate_laplacian_coordinates(vertices, adj_list, uniform_weight=True)

        # Perform optimization
        q_opt = self._optimize_configuration(
            q_init.copy(),
            target_laplacian,
            adj_list,
            scaled_terrain_points,
            target_human_joints=human_joints,
            max_iter=max_iter,
            q_last=q_last,
            q_prev=q_prev,
            q_prevprev=q_prevprev,
            previous_human_joints=previous_human_joints,
            previous_previous_human_joints=previous_previous_human_joints,
            target_base_orientation=target_base_orientation,
            target_base_position=target_base_position,
            end_effector_targets=end_effector_targets,
        )
        if self.pose_direction_post_projection:
            direction_step_limits = self._pose_direction_step_limits(
                human_joints, previous_human_joints
            )
            q_opt = self._project_pose_direction_step(
                q_opt, q_last, direction_step_limits
            )
        return self._project_actuated_temporal_step(
            q_opt, q_last, q_prev, q_prevprev
        )

    def _project_actuated_temporal_step(
        self,
        q_candidate: np.ndarray,
        q_last: Optional[np.ndarray],
        q_prev: Optional[np.ndarray],
        q_prevprev: Optional[np.ndarray],
    ) -> np.ndarray:
        """Exactly enforce output velocity, acceleration, and jerk bounds."""
        temporal_cfg = dict(self.end_effector_config.get("temporal", {}) or {})
        if q_last is None or not bool(temporal_cfg.get("enabled", False)):
            return q_candidate
        actuated = np.flatnonzero(np.arange(self.robot_model.nq) >= 7)
        if len(actuated) == 0:
            return q_candidate

        max_step = max(float(temporal_cfg.get("max_joint_step", 0.06)), 1e-8)
        lower = np.full(len(actuated), -max_step)
        upper = np.full(len(actuated), max_step)
        if q_prev is not None:
            previous_step = q_last[actuated] - q_prev[actuated]
            max_acceleration = max(
                float(temporal_cfg.get("max_joint_acceleration_step", 0.015)),
                1e-8,
            )
            lower = np.maximum(lower, previous_step - max_acceleration)
            upper = np.minimum(upper, previous_step + max_acceleration)
            if q_prevprev is not None and bool(
                temporal_cfg.get("hard_jerk_enabled", False)
            ):
                previous_previous_step = (
                    q_prev[actuated] - q_prevprev[actuated]
                )
                previous_acceleration = previous_step - previous_previous_step
                max_jerk = max(
                    float(temporal_cfg.get("max_joint_jerk_step", 0.005)),
                    1e-8,
                )
                lower = np.maximum(
                    lower,
                    previous_step + previous_acceleration - max_jerk,
                )
                upper = np.minimum(
                    upper,
                    previous_step + previous_acceleration + max_jerk,
                )

        proposed_step = q_candidate[actuated] - q_last[actuated]
        projected = np.asarray(q_candidate, dtype=float).copy()
        projected[actuated] = q_last[actuated] + np.clip(
            proposed_step, lower, upper
        )
        return projected

    def _project_pose_direction_step(
        self,
        q_candidate: np.ndarray,
        q_last: Optional[np.ndarray],
        direction_step_limits: Mapping[Tuple[str, str], float],
    ) -> np.ndarray:
        """Enforce the nonlinear bone-angle bound after the SQP solve."""
        if (
            q_last is None
            or not self.pose_direction_enabled
            or not self.pose_bones
        ):
            return q_candidate

        def directions(qpos: np.ndarray) -> np.ndarray:
            self.robot_data.qpos[:] = qpos
            mujoco.mj_forward(self.robot_model, self.robot_data)
            result = []
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
                result.append(vector / max(float(np.linalg.norm(vector)), 1e-10))
            return np.asarray(result)

        last_directions = directions(q_last)

        step_limits = np.asarray(
            [
                direction_step_limits.get(
                    (parent_name, child_name), self.pose_max_direction_step
                )
                for parent_name, child_name, _ in self.pose_bones
            ],
            dtype=float,
        )

        def maximum_ratio(qpos: np.ndarray) -> float:
            candidate_directions = directions(qpos)
            cosines = np.clip(
                np.sum(last_directions * candidate_directions, axis=1),
                -1.0,
                1.0,
            )
            return float(np.max(np.arccos(cosines) / step_limits))

        if maximum_ratio(q_candidate) <= 1.001:
            return q_candidate

        delta = np.asarray(q_candidate, dtype=float) - np.asarray(q_last, dtype=float)
        lower, upper = 0.0, 1.0
        best = np.asarray(q_last, dtype=float).copy()
        for _ in range(24):
            alpha = 0.5 * (lower + upper)
            trial = np.asarray(q_last, dtype=float) + alpha * delta
            trial[3:7] /= max(float(np.linalg.norm(trial[3:7])), 1e-12)
            if np.dot(trial[3:7], q_last[3:7]) < 0.0:
                trial[3:7] *= -1.0
            if maximum_ratio(trial) <= 1.0:
                best = trial
                lower = alpha
            else:
                upper = alpha
        return best

    def _optimize_configuration(
        self,
        q_init: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: List[List[int]],
        terrain_points: np.ndarray,
        target_human_joints: Optional[np.ndarray] = None,
        max_iter: int = 10,
        q_last: Optional[np.ndarray] = None,
        q_prev: Optional[np.ndarray] = None,
        q_prevprev: Optional[np.ndarray] = None,
        previous_human_joints: Optional[np.ndarray] = None,
        previous_previous_human_joints: Optional[np.ndarray] = None,
        target_base_orientation: Optional[np.ndarray] = None,
        target_base_position: Optional[np.ndarray] = None,
        end_effector_targets: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> np.ndarray:
        """
        Optimize robot configuration using SQP with interaction mesh constraints.

        Args:
            q_init: Initial configuration
            target_laplacian: Target Laplacian coordinates
            adj_list: Mesh adjacency list
            terrain_points: Terrain contact points
            max_iter: Maximum iterations
            q_last: Configuration at previous time step (for smoothness)

        Returns:
            Optimized configuration
        """
        q = q_init.copy()
        last_cost = np.inf
        
        import sys
        # Enable debug for first few frames to see the pattern
        debug = not hasattr(sys, '_omni_frame_count')
        if not hasattr(sys, '_omni_frame_count'):
            sys._omni_frame_count = 0
        
        frame_num = sys._omni_frame_count
        sys._omni_frame_count += 1
        
        # Debug first 5 frames to see the pattern
        show_debug = frame_num < 5

        if show_debug:
            print(f"\n=== Frame {frame_num} Optimization ===")
            print(f"q_init[:10]: {q_init[:10]}")
            if q_last is not None:
                print(f"q_last[:10]: {q_last[:10]}")
                print(f"Has q_last: True")
            else:
                print(f"Has q_last: False")

        for iteration in range(max_iter):
            # Single optimization step
            q_new, cost = self._single_optimization_step(
                q,
                target_laplacian,
                adj_list,
                terrain_points,
                target_human_joints,
                q_last,
                q_prev,
                q_prevprev,
                previous_human_joints,
                previous_previous_human_joints,
                target_base_orientation,
                target_base_position,
                end_effector_targets,
            )
            
            if show_debug and iteration < 3:
                print(f"  Iter {iteration}: cost={cost:.4f}, |dq|={np.linalg.norm(q_new - q):.6f}, status={'OK' if cost < np.inf else 'FAIL'}")

            if not np.isfinite(cost):
                if show_debug:
                    print(f"  Stopping SQP at iteration {iteration}: convex subproblem failed.")
                break

            # Check convergence
            if abs(cost - last_cost) < 1e-6:
                if show_debug:
                    print(f"  Converged at iteration {iteration}")
                break

            q = q_new
            last_cost = cost
        
        if show_debug:
            print(f"Final cost: {last_cost:.4f}")
            print(f"Final q[:10]: {q[:10]}")
            print(f"Total change: {np.linalg.norm(q - q_init):.6f}")

        return q

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
        target_base_position: Optional[np.ndarray] = None,
        end_effector_targets: Optional[Mapping[str, Mapping[str, Any]]] = None,
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
        L = calculate_laplacian_matrix(vertices, adj_list, uniform_weight=True)
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
        lap_var = cp.Variable(3 * len(vertices), name="laplacian")

        # Constraints
        constraints = []
        
        # CRITICAL: Linear equality constraint matching original implementation
        # This defines: lap_var = lap0_vec + J_L @ dqa
        # Original uses: J_L[:, self.q_a_indices] @ dqa - lap_var == -lap0_vec
        # Rearranged: lap_var == lap0_vec + J_L @ dqa
        lap0_vec = (L @ vertices).reshape(-1)
        target_lap_vec = target_laplacian.reshape(-1)
        
        # Note: J_L already has columns only for q_a_indices (from J_V construction)
        # But original slices again, so we match that exactly
        constraints.append(cp.Constant(J_L) @ dqa - lap_var == -lap0_vec)

        # Joint limits
        q_a_current = q[self.q_a_indices]
        constraints.extend([
            dqa >= (self.q_a_lb - q_a_current),
            dqa <= (self.q_a_ub - q_a_current),
        ])

        temporal_cfg = dict(self.end_effector_config.get("temporal", {}) or {})
        if q_last is not None and bool(temporal_cfg.get("enabled", False)):
            actuated_qa = np.flatnonzero(self.q_a_indices >= 7)
            if len(actuated_qa):
                actuated_qpos = self.q_a_indices[actuated_qa]
                new_actuated = q_a_current[actuated_qa] + dqa[actuated_qa]
                last_actuated = q_last[actuated_qpos]
                max_joint_step = max(
                    float(temporal_cfg.get("max_joint_step", 0.15)), 1e-6
                )
                constraints.extend(
                    [
                        new_actuated - last_actuated <= max_joint_step,
                        last_actuated - new_actuated <= max_joint_step,
                    ]
                )
                if q_prev is not None:
                    previous_step = last_actuated - q_prev[actuated_qpos]
                    new_step = new_actuated - last_actuated
                    max_acceleration_step = max(
                        float(
                            temporal_cfg.get("max_joint_acceleration_step", 0.12)
                        ),
                        1e-6,
                    )
                    constraints.extend(
                        [
                            new_step - previous_step <= max_acceleration_step,
                            previous_step - new_step <= max_acceleration_step,
                        ]
                    )
                    if q_prevprev is not None and bool(
                        temporal_cfg.get("hard_jerk_enabled", False)
                    ):
                        previous_previous_step = (
                            q_prev[actuated_qpos]
                            - q_prevprev[actuated_qpos]
                        )
                        previous_acceleration = (
                            previous_step - previous_previous_step
                        )
                        new_acceleration = new_step - previous_step
                        max_jerk_step = max(
                            float(
                                temporal_cfg.get("max_joint_jerk_step", 0.005)
                            ),
                            1e-8,
                        )
                        constraints.extend(
                            [
                                new_acceleration - previous_acceleration
                                <= max_jerk_step,
                                previous_acceleration - new_acceleration
                                <= max_jerk_step,
                            ]
                        )

            base_qa = [
                int(np.where(self.q_a_indices == index)[0][0])
                for index in (0, 1, 2)
                if index in self.q_a_indices
            ]
            if len(base_qa) == 3:
                new_base = q_a_current[base_qa] + dqa[base_qa]
                max_base_step = max(
                    float(temporal_cfg.get("max_base_step", 0.05)), 1e-6
                )
                constraints.extend(
                    [
                        new_base - q_last[:3] <= max_base_step,
                        q_last[:3] - new_base <= max_base_step,
                    ]
                )

        # The interaction-mesh Laplacian is nearly translation invariant.  An
        # explicit morphology-aware base envelope prevents infeasible hand/foot
        # targets from dragging the floating base away from the source actor.
        base_position_obj = None
        base_cfg = dict(
            self.end_effector_config.get("base_position_constraint", {}) or {}
        )
        if target_base_position is not None and bool(base_cfg.get("enabled", False)):
            base_target = np.asarray(target_base_position, dtype=float).reshape(3).copy()
            base_target[2] += float(base_cfg.get("z_offset", 0.0))
            max_xy_error = max(float(base_cfg.get("max_xy_error", 0.08)), 0.0)
            max_z_error = max(float(base_cfg.get("max_z_error", 0.12)), 0.0)

            base_new_components = []
            for qpos_idx in (0, 1, 2):
                if qpos_idx in self.q_a_indices:
                    qa_idx = int(np.where(self.q_a_indices == qpos_idx)[0][0])
                    base_new_components.append(q[qpos_idx] + dqa[qa_idx])
                else:
                    base_new_components.append(cp.Constant(float(q[qpos_idx])))
            base_new = cp.hstack(base_new_components)
            if bool(base_cfg.get("hard_constraints", True)):
                constraints.append(
                    cp.norm(base_new[:2] - base_target[:2], 2) <= max_xy_error
                )
                constraints.extend(
                    [
                        base_new[2] - base_target[2] <= max_z_error,
                        base_target[2] - base_new[2] <= max_z_error,
                    ]
                )
            xy_scale = max(max_xy_error, 1e-6)
            z_scale = max(max_z_error, 1e-6)
            base_position_obj = (
                cp.sum_squares((base_new[:2] - base_target[:2]) / xy_scale)
                + cp.square((base_new[2] - base_target[2]) / z_scale)
            )

        # Non-penetration constraints (self-collision + terrain)
        if self.hard_penetration_constraint:
            penetration_constraints = self._compute_penetration_constraints(q, dqa)
            constraints.extend(penetration_constraints)

        # Trust region
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
        if base_position_obj is not None:
            body_obj += float(base_cfg.get("centering_weight", 5.0)) * base_position_obj
        
        # Joint regularization cost (keep joints near zero/neutral)
        # Matching original: Q_diag cost uses q_a_n_last (last optimized at current time step)
        # In our case, q_a_current = q[self.q_a_indices] which is the current guess
        Qd = np.asarray(self.Q_diag, dtype=float).reshape(-1)
        
        # Modify Q_diag for specific joints (matching original MANUAL_COST logic)
        Q_diag_modified = Qd.copy()
        for i in range(self.robot_model.njnt):
            joint_name = self.robot_model.joint(i).name
            if joint_name:
                joint_name_lower = joint_name.lower()
                # Strong regularization for joints prone to 180° flips
                if ('waist' in joint_name_lower or 'torso' in joint_name_lower or 
                    ('hip' in joint_name_lower and 'yaw' in joint_name_lower)):
                    qpos_adr = self.robot_model.jnt_qposadr[i]
                    if qpos_adr in self.q_a_indices:
                        idx_in_qa = np.where(self.q_a_indices == qpos_adr)[0]
                        if len(idx_in_qa) > 0:
                            Q_diag_modified[idx_in_qa[0]] = 0.2  # Strong regularization (matching original MANUAL_COST)
        
        # Q_diag cost: ||sqrt(Q_diag) * (dqa + q_a_current)||^2
        # This matches original: cp.sum_squares(cp.multiply(np.sqrt(Qd), dqa + q_a_n_last))
        body_obj += cp.sum_squares(cp.multiply(np.sqrt(Q_diag_modified), dqa + q_a_current))

        # Smoothness cost (matching original implementation exactly)
        # CRITICAL FIX: Use previous frame's velocity, not current guess
        if q_last is not None:
            q_a_current = q[self.q_a_indices]  # Current guess at this SQP iteration
            dqa_smooth = q_last[self.q_a_indices] - q_a_current  # Velocity from prev frame to current guess
            body_obj += self.smooth_weight * cp.sum_squares(dqa - dqa_smooth)
        
        # Base orientation tracking cost
        # Keep the base orientation close to the target (estimated from human joints).
        # The residual is the world-frame rotation vector to the target and its
        # linearization is the same qdot->omega map used by every other
        # orientation task.  Penalizing raw quaternion components instead mixes
        # a non-metric distance into the objective and pulls along the radial
        # quaternion direction that the kinematics cannot see.
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
            body_obj += orientation_weight * cp.sum_squares(
                base_angular_jacobian @ dqa - base_rotation_error
            )

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
        direction_step_limits = self._pose_direction_step_limits(
            target_human_joints, previous_human_joints
        )
        constraints.extend(
            self._build_pose_direction_constraints(
                dqa,
                p_V,
                J_V,
                last_pose_directions,
                direction_step_limits,
            )
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

        hierarchy_enabled = bool(
            self.end_effector_enabled
            and self.end_effector_config.get("hierarchy", True)
            and (
                contact_residual is not None
                or swing_residual is not None
                or swing_orientation_residual is not None
                or pose_direction_residual is not None
            )
        )
        if not hierarchy_enabled:
            # Weighted fallback is useful for ablations, but the default is the
            # hierarchy below because finite weights cannot guarantee priority.
            endpoint_weight = float(self.end_effector_config.get("weighted_scale", 1.0))
            if contact_residual is not None:
                body_obj += endpoint_weight * cp.sum_squares(contact_residual)
            if swing_residual is not None:
                body_obj += endpoint_weight * cp.sum_squares(swing_residual)
            if swing_orientation_residual is not None:
                body_obj += endpoint_weight * cp.sum_squares(
                    swing_orientation_residual
                )
            if pose_direction_residual is not None:
                body_obj += float(
                    self.end_effector_config.get("pose_direction_weight", 10.0)
                ) * cp.sum_squares(pose_direction_residual)
        
        import sys
        show_solver_debug = hasattr(sys, '_omni_frame_count') and sys._omni_frame_count <= 5
        
        try:
            active_constraints = list(constraints)

            def solve_stage(stage_obj: cp.Expression, stage_name: str) -> cp.Problem:
                nonlocal active_constraints
                stage_problem = cp.Problem(cp.Minimize(stage_obj), active_constraints)

                def run_solver(candidate: cp.Problem) -> None:
                    try:
                        candidate.solve(solver=cp.CLARABEL, verbose=False)
                    except cp.error.SolverError:
                        pass
                    if candidate.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                        candidate.solve(
                            solver=cp.SCS,
                            verbose=False,
                            eps=float(self.end_effector_config.get("solver_tolerance", 1e-4)),
                            max_iters=int(
                                self.end_effector_config.get("solver_max_iterations", 20000)
                            ),
                        )

                run_solver(stage_problem)
                if (
                    stage_problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
                    and not hierarchy_enabled
                    and endpoint_contact_constraints
                ):
                    # The contact-plane halfspaces and orientation-step caps can
                    # be jointly infeasible with the trust region for a single
                    # linearized step.  Relax them before touching the trust
                    # region itself: an unbounded linearized step is worse than
                    # one frame of relaxed contact-side constraints.
                    if show_solver_debug:
                        print(
                            f"    {stage_name} solve failed: {stage_problem.status}, "
                            "retrying without endpoint contact constraints..."
                        )
                    relaxable = {id(c) for c in endpoint_contact_constraints}
                    active_constraints = [
                        constraint
                        for constraint in active_constraints
                        if id(constraint) not in relaxable
                    ]
                    stage_problem = cp.Problem(cp.Minimize(stage_obj), active_constraints)
                    run_solver(stage_problem)
                if (
                    stage_problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
                    and not hierarchy_enabled
                ):
                    if show_solver_debug:
                        print(
                            f"    {stage_name} solve failed: {stage_problem.status}, "
                            "trying without SOC..."
                        )
                    active_constraints = [
                        constraint
                        for constraint in active_constraints
                        if not isinstance(constraint, cp.constraints.second_order.SOC)
                    ]
                    stage_problem = cp.Problem(cp.Minimize(stage_obj), active_constraints)
                    run_solver(stage_problem)
                return stage_problem

            if hierarchy_enabled:
                relative_tolerance = max(
                    float(self.end_effector_config.get("priority_relative_tolerance", 0.01)),
                    0.0,
                )
                absolute_tolerance = max(
                    float(self.end_effector_config.get("priority_absolute_tolerance", 1e-4)),
                    0.0,
                )
                tie_break = float(self.end_effector_config.get("tie_break_regularization", 1e-8))

                if contact_residual is not None:
                    contact_level = cp.norm(contact_residual, 2)
                    contact_problem = solve_stage(
                        contact_level + tie_break * cp.sum_squares(dqa), "contact"
                    )
                    if contact_problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                        return q, np.inf
                    contact_optimum = max(float(contact_level.value), 0.0)
                    contact_bound = (
                        (1.0 + relative_tolerance) * contact_optimum
                        + absolute_tolerance
                    )
                    active_constraints.append(contact_level <= contact_bound)

                if swing_residual is not None:
                    swing_level = cp.norm(swing_residual, 2)
                    swing_problem = solve_stage(
                        swing_level + tie_break * cp.sum_squares(dqa), "swing"
                    )
                    if swing_problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                        return q, np.inf
                    swing_optimum = max(float(swing_level.value), 0.0)
                    swing_relative_tolerance = max(
                        float(
                            self.end_effector_config.get(
                                "swing_priority_relative_tolerance",
                                relative_tolerance,
                            )
                        ),
                        0.0,
                    )
                    swing_absolute_tolerance = max(
                        float(
                            self.end_effector_config.get(
                                "swing_priority_absolute_tolerance",
                                absolute_tolerance,
                            )
                        ),
                        0.0,
                    )
                    swing_bound = (
                        (1.0 + swing_relative_tolerance) * swing_optimum
                        + swing_absolute_tolerance
                    )
                    active_constraints.append(swing_level <= swing_bound)

                if swing_orientation_residual is not None:
                    swing_orientation_level = cp.norm(
                        swing_orientation_residual, 2
                    )
                    swing_orientation_problem = solve_stage(
                        swing_orientation_level
                        + tie_break * cp.sum_squares(dqa),
                        "swing-orientation",
                    )
                    if swing_orientation_problem.status not in (
                        cp.OPTIMAL,
                        cp.OPTIMAL_INACCURATE,
                    ):
                        return q, np.inf
                    swing_orientation_optimum = max(
                        float(swing_orientation_level.value), 0.0
                    )
                    orientation_relative_tolerance = max(
                        float(
                            self.end_effector_config.get(
                                "orientation_priority_relative_tolerance",
                                relative_tolerance,
                            )
                        ),
                        0.0,
                    )
                    orientation_absolute_tolerance = max(
                        float(
                            self.end_effector_config.get(
                                "orientation_priority_absolute_tolerance",
                                absolute_tolerance,
                            )
                        ),
                        0.0,
                    )
                    active_constraints.append(
                        swing_orientation_level
                        <= (1.0 + orientation_relative_tolerance)
                        * swing_orientation_optimum
                        + orientation_absolute_tolerance
                    )

                if pose_direction_residual is not None:
                    pose_level = cp.norm(pose_direction_residual, 2)
                    pose_problem = solve_stage(
                        pose_level + tie_break * cp.sum_squares(dqa),
                        "pose-direction",
                    )
                    if pose_problem.status not in (
                        cp.OPTIMAL,
                        cp.OPTIMAL_INACCURATE,
                    ):
                        return q, np.inf
                    pose_optimum = max(float(pose_level.value), 0.0)
                    pose_relative_tolerance = max(
                        float(
                            self.end_effector_config.get(
                                "pose_priority_relative_tolerance", 0.02
                            )
                        ),
                        0.0,
                    )
                    pose_absolute_tolerance = max(
                        float(
                            self.end_effector_config.get(
                                "pose_priority_absolute_tolerance", 0.04
                            )
                        ),
                        0.0,
                    )
                    active_constraints.append(
                        pose_level
                        <= (1.0 + pose_relative_tolerance) * pose_optimum
                        + pose_absolute_tolerance
                    )

                problem = solve_stage(body_obj, "whole-body")
                if show_solver_debug:
                    print(
                        f"    hierarchy tasks: contact={num_contacts}, "
                        f"swing={num_swings}, "
                        f"swing_orientation={num_swing_orientations}, "
                        f"pose_bones={len(self.pose_bones)}"
                    )
            else:
                problem = solve_stage(body_obj, "weighted")

            if (
                problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
                and show_solver_debug
            ):
                print(f"    Final solve status: {problem.status}")

            if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                dqa_opt = np.asarray(dqa.value, dtype=float).reshape(-1)
                # First-order SQP guarantees only hold inside the trust
                # region.  Conic solvers may return an OPTIMAL_INACCURATE
                # point with constraint violations, so enforce the radius and
                # safety-critical base envelope explicitly before applying it.
                if hierarchy_enabled:
                    base_translation_qa = []
                    if (
                        target_base_position is not None
                        and bool(base_cfg.get("enabled", False))
                    ):
                        base_target = np.asarray(
                            target_base_position, dtype=float
                        ).reshape(3).copy()
                        base_target[2] += float(base_cfg.get("z_offset", 0.0))
                        proposed_base = q[:3].copy()
                        for qpos_idx in (0, 1, 2):
                            if qpos_idx in self.q_a_indices:
                                qa_idx = int(
                                    np.where(self.q_a_indices == qpos_idx)[0][0]
                                )
                                base_translation_qa.append(qa_idx)
                                proposed_base[qpos_idx] += dqa_opt[qa_idx]

                        xy_delta = proposed_base[:2] - base_target[:2]
                        xy_norm = float(np.linalg.norm(xy_delta))
                        max_xy_error = max(
                            float(base_cfg.get("max_xy_error", 0.08)), 0.0
                        )
                        if xy_norm > max_xy_error:
                            proposed_base[:2] = base_target[:2] + (
                                max_xy_error / max(xy_norm, 1e-12)
                            ) * xy_delta
                        max_z_error = max(
                            float(base_cfg.get("max_z_error", 0.12)), 0.0
                        )
                        proposed_base[2] = np.clip(
                            proposed_base[2],
                            base_target[2] - max_z_error,
                            base_target[2] + max_z_error,
                        )
                        for qpos_idx in (0, 1, 2):
                            if qpos_idx in self.q_a_indices:
                                qa_idx = int(
                                    np.where(self.q_a_indices == qpos_idx)[0][0]
                                )
                                dqa_opt[qa_idx] = proposed_base[qpos_idx] - q[qpos_idx]

                    fixed_norm_sq = float(
                        np.sum(np.square(dqa_opt[base_translation_qa]))
                    )
                    free_indices = np.array(
                        [
                            idx
                            for idx in range(len(dqa_opt))
                            if idx not in base_translation_qa
                        ],
                        dtype=int,
                    )
                    remaining_radius = np.sqrt(
                        max(self.step_size**2 - fixed_norm_sq, 0.0)
                    )
                    free_norm = float(np.linalg.norm(dqa_opt[free_indices]))
                    if free_norm > remaining_radius:
                        dqa_opt[free_indices] *= remaining_radius / max(
                            free_norm, 1e-12
                        )
                cost = float(problem.value)

                q_opt = q.copy()
                q_opt[self.q_a_indices] = dqa_opt + q_a_current
                
                # CRITICAL FIX: Normalize quaternion with sign continuity to prevent frame-to-frame jumps
                quat_new = q_opt[3:7]
                quat_new = quat_new / (np.linalg.norm(quat_new) + 1e-12)
                
                # Ensure quaternion sign continuity with previous frame (if available)
                if q_last is not None:
                    quat_prev = q_last[3:7]
                    # If dot product is negative, quaternions are in opposite hemispheres
                    # Flip sign to ensure continuity
                    if np.dot(quat_new, quat_prev) < 0:
                        quat_new = -quat_new
                
                q_opt[3:7] = quat_new

                return q_opt, cost
            else:
                if show_solver_debug:
                    print(f"    SOLVER FAILED: {problem.status}")
                return q, np.inf

        except Exception as e:
            if show_solver_debug:
                print(f"    EXCEPTION: {e}")
            return q, np.inf

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

    def _prefilter_pairs_with_mj_collision(self, threshold: float) -> set:
        """
        Use MuJoCo collision detection to find candidate geometry pairs.
        
        Args:
            threshold: Distance threshold for collision detection
            
        Returns:
            Set of (geom1_id, geom2_id) tuples for candidate collision pairs
        """
        m, d = self.robot_model, self.robot_data
        ngeom = m.ngeom

        # Cache geometry names
        if not hasattr(self, '_geom_names'):
            self._geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(ngeom)]

        # Save original margins
        if not hasattr(self, '_saved_margins'):
            self._saved_margins = np.empty_like(m.geom_margin)
        self._saved_margins[:] = m.geom_margin

        # Temporarily set margins to threshold
        m.geom_margin[:] = threshold

        # Run collision detection
        mujoco.mj_collision(m, d)

        # Collect unique candidate pairs
        candidates = set()
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 < 0 or g2 < 0:
                continue
            candidates.add((min(g1, g2), max(g1, g2)))

        # Restore original margins
        m.geom_margin[:] = self._saved_margins

        return candidates

    def _compute_jacobian_for_contact_relative(
        self, 
        geom1_id: int, 
        geom2_id: int, 
        geom1_name: str,
        geom2_name: str,
        fromto: np.ndarray, 
        dist: float
    ) -> np.ndarray:
        """
        Compute relative contact Jacobian for a geometry pair.
        
        Args:
            geom1_id: First geometry ID
            geom2_id: Second geometry ID
            geom1_name: First geometry name
            geom2_name: Second geometry name
            fromto: Contact points [pos1_x, pos1_y, pos1_z, pos2_x, pos2_y, pos2_z]
            dist: Signed distance between geometries
            
        Returns:
            Contact Jacobian (1D array of length nq)
        """
        # Get closest points from fromto buffer
        pos1 = fromto[:3]  # closest point on geom1
        pos2 = fromto[3:]  # closest point on geom2

        v = pos1 - pos2
        norm_v = np.linalg.norm(v)

        if norm_v > 1e-12:
            nhat_BA_W = np.sign(dist) * (v / norm_v)
        # Degenerate case: points coincide
        elif "ground" in geom2_name.lower():
            nhat_BA_W = np.array([0.0, 0.0, 1.0]) * (1.0 if dist >= 0 else -1.0)
        elif "ground" in geom1_name.lower():
            nhat_BA_W = np.array([0.0, 0.0, -1.0]) * (1.0 if dist >= 0 else -1.0)
        else:
            nhat_BA_W = np.array([0.0, 0.0, 0.0])

        # Get body IDs for the geometries
        body1_id = self.robot_model.geom_bodyid[geom1_id]
        body2_id = self.robot_model.geom_bodyid[geom2_id]

        # Compute Jacobians for both contact points (in world frame)
        J_bodyA = self._calc_contact_jacobian_from_point(body1_id, pos1, input_world=True)
        J_bodyB = self._calc_contact_jacobian_from_point(body2_id, pos2, input_world=True)

        # Compute relative Jacobian
        Jc = J_bodyA - J_bodyB

        # Project onto contact normal
        return nhat_BA_W @ Jc

    def _compute_penetration_constraints(self, q: np.ndarray, dqa: cp.Variable) -> List[cp.Constraint]:
        """
        Compute penetration constraints for robot-robot and robot-terrain contacts.

        Two sources of constraints are combined:
        1. **Self-collision** – MuJoCo's built-in collision detection finds pairs of
           robot geoms that are close to each other and builds linearised
           non-penetration constraints via contact Jacobians.
        2. **Terrain penetration** – for every robot geom whose centre is within
           ``collision_detection_threshold`` of the terrain mesh surface (measured
           via ``trimesh.proximity.closest_point``), a unilateral constraint is
           added that pushes the geom upward (in the terrain-surface-normal
           direction) to avoid terrain penetration.

        The terrain mesh is NOT embedded in the MuJoCo model, so MuJoCo's own
        collision pipeline cannot detect robot-terrain contacts.  We handle them
        analytically using the trimesh proximity query and the robot's
        translational Jacobian.
        """
        constraints = []

        # Ensure kinematics are current
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)

        m, d = self.robot_model, self.robot_data
        threshold = float(self.collision_detection_threshold)

        # ------------------------------------------------------------------
        # 1) Robot self-collision constraints (via MuJoCo collision)
        # ------------------------------------------------------------------
        candidates = self._prefilter_pairs_with_mj_collision(threshold)
        fromto = np.zeros(6, dtype=float)
        contype, conaff = m.geom_contype, m.geom_conaffinity

        for g1, g2 in candidates:
            # Skip geoms with no collision masks
            if contype[g1] == 0 and conaff[g1] == 0:
                continue
            if contype[g2] == 0 and conaff[g2] == 0:
                continue

            fromto[:] = 0.0
            dist = mujoco.mj_geomDistance(m, d, g1, g2, threshold, fromto)
            if dist <= threshold:
                J_rel = self._compute_jacobian_for_contact_relative(
                    g1, g2, self._geom_names[g1], self._geom_names[g2], fromto, dist
                )
                Ja = J_rel[self.q_a_indices]
                rhs = -dist - self.penetration_tolerance
                constraints.append(Ja @ dqa >= rhs)

        # ------------------------------------------------------------------
        # 2) Robot-terrain penetration constraints (via trimesh proximity)
        # ------------------------------------------------------------------
        constraints.extend(
            self._compute_terrain_penetration_constraints(q, dqa, threshold)
        )

        return constraints

    def _compute_terrain_penetration_constraints(
        self, q: np.ndarray, dqa: cp.Variable, threshold: float
    ) -> List[cp.Constraint]:
        """
        Compute non-penetration constraints between robot geoms and the
        terrain trimesh.

        Samples points on the actual surface of each collision geom based on
        its shape, then checks each point for penetration with the terrain.
        This avoids the limitation of only checking the geom center which can
        miss penetration when the geom has large extent.

        **Trade-off**: This approach samples points on robot collision geoms
        for checking against the external terrain trimesh. Only primitive geom
        types (sphere, box, capsule, cylinder) are fully supported with surface
        sampling. Other geom types (mesh, heightfield, etc.) fall back to only
        checking the center point.

        For each sampled point that is close to or inside the terrain, we add
        the linear constraint:

            n^T J_a  dqa  >=  -(d - tol)

        where
        - d   is the signed distance (positive = above terrain),
        - n   is the outward terrain surface normal at the closest point,
        - J_a is the translational Jacobian of the geom's body at the sampled
          point (columns for the actuated DOFs only),
        - tol is ``self.penetration_tolerance``.
        """
        import trimesh as _trimesh

        def sample_geom_surface_points(geom, geom_pos, geom_rot):
            """Sample points on the surface of a MuJoCo geom based on its type.
            Returns an array of shape (N, 3) of world-frame points.

            ## Implementation Notes / Trade-offs
            Currently only supports **primitive-shaped collision geoms**:
            - Sphere (mjGEOM_SPHERE)
            - Box (mjGEOM_BOX)
            - Capsule (mjGEOM_CAPSULE)
            - Cylinder (mjGEOM_CYLINDER)
            - Plane (skipped)

            For other geom types (meshes, heightfields, ellipsoids), this falls back
            to only checking the geom center point.

            ## To add support for a new geom type:
            1. Add a new `elif geom_type == mujoco.mjtGeom.mjGEOM_XXX:` case
            2. Compute the appropriate surface points in the geom's local frame
               based on the `geom.size` parameters
            3. Transform the local points to world frame using:
               `world_pt = geom_pos + geom_rot.apply(local_pt)`
            4. Add all world points to the `points` list and return
            """
            geom_type = geom.type
            size = geom.size
            points = []

            if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                # Sphere: radius = size[0], sample points on surface
                radius = size[0]
                # Sample 6 outward points along major axes
                for dx, dy, dz in [(1, 0, 0), (-1, 0, 0),
                                   (0, 1, 0), (0, -1, 0),
                                   (0, 0, 1), (0, 0, -1)]:
                    local_pt = np.array([dx, dy, dz]) * radius
                    world_pt = geom_pos + geom_rot.apply(local_pt)
                    points.append(world_pt)
                return np.array(points)

            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                # Box: size = half extents, sample center of each face
                half_extents = size[:3]
                # Sample center of each of the 6 faces
                for sx, sy, sz in [(1, 0, 0), (-1, 0, 0),
                                   (0, 1, 0), (0, -1, 0),
                                   (0, 0, 1), (0, 0, -1)]:
                    local_pt = np.array([
                        sx * half_extents[0],
                        sy * half_extents[1],
                        sz * half_extents[2]
                    ])
                    world_pt = geom_pos + geom_rot.apply(local_pt)
                    points.append(world_pt)
                return np.array(points)

            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                # Capsule: size[0] = radius, size[1] = half-length along the
                # local z axis (MuJoCo convention; the previous x-axis layout
                # sampled points that are not on the capsule surface).
                radius = size[0]
                half_len = size[1]
                # Barrel rings at both ends and the middle, plus the two
                # hemispherical cap tips along z.
                for s in [-half_len, 0.0, half_len]:
                    for theta in [0, np.pi/2, np.pi, 3*np.pi/2]:
                        local_pt = np.array([
                            radius * np.cos(theta),
                            radius * np.sin(theta),
                            s,
                        ])
                        points.append(geom_pos + geom_rot.apply(local_pt))
                for tip in [-(half_len + radius), half_len + radius]:
                    points.append(
                        geom_pos + geom_rot.apply(np.array([0.0, 0.0, tip]))
                    )
                return np.array(points)

            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                # Cylinder: size[0] = radius, size[1] = half-length along the
                # local z axis (MuJoCo convention).
                radius = size[0]
                half_len = size[1]
                # Center of each end cap + 4 side points at the midpoint.
                for s in [-half_len, half_len]:
                    points.append(
                        geom_pos + geom_rot.apply(np.array([0.0, 0.0, s]))
                    )
                for theta in [0, np.pi/2, np.pi, 3*np.pi/2]:
                    local_pt = np.array([
                        radius * np.cos(theta),
                        radius * np.sin(theta),
                        0.0,
                    ])
                    points.append(geom_pos + geom_rot.apply(local_pt))
                return np.array(points)

            elif geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
                # Plane is infinite, skip terrain collision checking
                return np.empty((0, 3))

            else:
                # For other geom types (meshes, heightfields, ellipsoid),
                # just return the center point as a fallback
                return np.array([geom_pos])

        constraints: list = []
        m, d = self.robot_model, self.robot_data

        # Collect world-frame sample points on the surface of every collision geom
        # Sample points based on actual geom shape instead of just checking center
        all_points = []
        all_geom_info = []
        for gi in range(m.ngeom):
            # Skip purely visual geoms with no collision flags
            if m.geom_contype[gi] == 0 and m.geom_conaffinity[gi] == 0:
                continue

            # Get current geom pose in world frame
            pos = d.geom_xpos[gi].copy()
            rot_mat = d.geom_xmat[gi].reshape(3, 3).copy()
            rot = Rotation.from_matrix(rot_mat)

            # Sample surface points based on geom type
            geom = m.geom(gi)
            points = sample_geom_surface_points(geom, pos, rot)

            # Always add the center as a fallback even if other sampling failed
            if len(points) == 0:
                points = np.array([pos])

            for pt in points:
                all_points.append(pt)
                all_geom_info.append((gi, pt))

        if len(all_points) == 0:
            return constraints

        all_points = np.array(all_points)  # (N, 3)

        # Query terrain mesh for closest points to each sampled point
        closest_pts, dists, tri_ids = _trimesh.proximity.closest_point(
            self.terrain_mesh, all_points
        )

        for k, (gi, query_pt) in enumerate(all_geom_info):
            if dists[k] > threshold:
                continue

            # Signed distance: positive when above terrain.
            # closest_pts[k] is on the terrain surface; query_pt is the
            # point on the geom surface. We define "above" as the direction
            # of the terrain face normal.
            surface_pt = closest_pts[k]

            # Face normal from terrain mesh
            face_normal = self.terrain_mesh.face_normals[tri_ids[k]]
            # Ensure normal points "outward" (upward for typical terrains)
            if face_normal[2] < 0:
                face_normal = -face_normal

            # Signed distance along the normal
            signed_dist = np.dot(query_pt - surface_pt, face_normal)

            # Only constrain points that are close to or below the surface
            if signed_dist > threshold:
                continue

            # Translational Jacobian for this geom's body at the query point
            body_id = m.geom_bodyid[gi]
            J_full = self._calc_contact_jacobian_from_point(
                body_id, query_pt, input_world=True
            )
            # Project onto terrain normal -> 1-D Jacobian
            J_n = face_normal @ J_full  # (nq,)
            Ja = J_n[self.q_a_indices]

            # Constraint: J_a @ dqa >= -(signed_dist - tolerance)
            rhs = -signed_dist - self.penetration_tolerance
            constraints.append(Ja @ dqa >= rhs)

        return constraints


def retarget_smplx_to_robot(
    smplx_trajectory: np.ndarray,
    robot_urdf_path: Path,
    terrain_mesh_path: Path,
    joint_mapping: Dict[str, str],
    robot_height: Optional[float] = None,
) -> Tuple[float, np.ndarray]:
    """
    High-level function to retarget SMPLX trajectory to any robot on any terrain.

    Args:
        smplx_trajectory: SMPLX joint positions (T, J, 3)
        robot_urdf_path: Path to robot URDF
        terrain_mesh_path: Path to terrain mesh
        joint_mapping: Mapping from SMPLX joints to robot links
        robot_height: Robot height override

    Returns:
        Tuple of (terrain_scale, retargeted_trajectory)
    """
    # Validate inputs
    if not validate_smplx_trajectory(smplx_trajectory):
        raise ValueError("Invalid SMPLX trajectory format")

    # Load robot
    robot_urdf = yourdfpy.URDF.load(str(robot_urdf_path), load_meshes=True)
    robot_model = mujoco.MjModel.from_xml_path(str(robot_urdf_path))
    robot_data = mujoco.MjData(robot_model)

    # Detect robot height if not provided
    if robot_height is None:
        robot_height = 1.6  # Default humanoid height

    # Load terrain
    terrain_mesh = load_terrain_mesh(terrain_mesh_path)

    # Compute terrain scaling
    from .core import OmniRetargeter
    temp_retargeter = OmniRetargeter(
        robot_urdf_path=robot_urdf_path,
        terrain_mesh_path=terrain_mesh_path,
        joint_mapping=joint_mapping,
        robot_height=robot_height
    )
    terrain_scale = temp_retargeter._compute_terrain_scale(smplx_trajectory)

    # Scale terrain
    scaled_terrain = scale_mesh(terrain_mesh, terrain_scale)

    # Initialize retargeter with scaled terrain
    retargeter = GenericInteractionRetargeter(
        robot_model, robot_data, scaled_terrain, joint_mapping, robot_height
    )

    # Retarget each frame
    retargeted_trajectory = []
    q_init = np.zeros(robot_model.nq)
    q_init[3:7] = [1, 0, 0, 0]  # Identity quaternion
    q_init[2] = robot_height * 0.5  # Initial height
    q_last = None

    for frame_idx, human_joints in enumerate(smplx_trajectory):
        # Extract mapped joints
        mapped_indices = []  # TODO: Map joint names to indices
        if len(mapped_indices) == 0:
            # Fallback: use first few joints
            mapped_joints = human_joints[:len(joint_mapping)]
        else:
            mapped_joints = human_joints[mapped_indices]

        # Retarget frame
        q_opt = retargeter.retarget_frame(mapped_joints, q_init, q_last=q_last)
        retargeted_trajectory.append(q_opt)

        # Update initial guess for next frame
        q_last = q_opt.copy()
        q_init = q_opt

    return terrain_scale, np.array(retargeted_trajectory)
