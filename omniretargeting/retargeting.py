"""Core retargeting functionality adapted for generic robots and terrains."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Tuple

import mujoco
import numpy as np
import trimesh
import yourdfpy
from scipy.spatial import Delaunay

from ._solver.end_effectors import EndEffectorMixin
from ._solver.kinematics import RobotKinematicsMixin
from ._solver.initialization import PoseInitializationMixin
from ._solver.orientation import BodyOrientationMixin
from ._solver.pose import PoseDirectionMixin
from ._solver.step import OptimizationStepMixin
from ._solver.targets import (
    _apply_configured_joint_limits as _apply_configured_joint_limits,
    _apply_configured_joint_regularization as _apply_configured_joint_regularization,
    _apply_joint_limit_safety_fraction as _apply_joint_limit_safety_fraction,
    _base_orientation_step_limit as _base_orientation_step_limit,
    _calibrated_orientation_target as _calibrated_orientation_target,
    _clamp_target_to_reach as _clamp_target_to_reach,
    _limit_relative_orientation_step as _limit_relative_orientation_step,
    _relative_orientation_step_magnitude as _relative_orientation_step_magnitude,
    _source_orientation_step as _source_orientation_step,
    _swing_target_z_offset as _swing_target_z_offset,
    _transform_swing_target as _transform_swing_target,
)
from .utils import (
    calculate_laplacian_coordinates,
    calculate_source_laplacian,
    get_adjacency_list,
    load_terrain_mesh,
    sample_points_on_mesh,
    scale_mesh,
    validate_smplx_trajectory,
)


class GenericInteractionRetargeter(
    BodyOrientationMixin,
    PoseInitializationMixin,
    PoseDirectionMixin,
    EndEffectorMixin,
    RobotKinematicsMixin,
    OptimizationStepMixin,
):
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
        self_collision_enabled: bool = True,
        joint_limit_safety_fraction: float = 1.0,
        preserve_root_z: bool = False,
        end_effector_config: Optional[Dict[str, Any]] = None,
        self_collision: Optional[Dict[str, Any]] = None,
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
            self_collision_enabled: Enable collision-body self avoidance,
                independently of the scene penetration mode. Visual-only
                geometry is excluded.
            self_collision: Optional distance and bounded slack configuration.
            joint_limit_safety_fraction: Fraction of each scalar joint's
                neutral-to-mechanical-limit excursion available to the solver.
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
        self.self_collision_enabled = bool(self_collision_enabled)
        self.self_collision_config = dict(self_collision or {})
        self.joint_limit_safety_fraction = float(joint_limit_safety_fraction)
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

        # Full body-frame orientations disambiguate the IK branch in ways
        # segment positions alone cannot (especially shoulder/elbow twist).
        self._setup_body_orientations()

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
        self.lock_root_orientation_to_source = bool(
            self.end_effector_config.get(
                "lock_root_orientation_to_source", False
            )
        )
        if self.lock_root_orientation_to_source:
            self.q_a_indices = self.q_a_indices[
                ~np.isin(self.q_a_indices, np.asarray([3, 4, 5, 6]))
            ]
        self.nq_a = len(self.q_a_indices)

        print(f"Robot config: nq={self.nq}, nv={self.nv}, nq_a={self.nq_a}")
        if self.nq_a > 0:
            print(f"q_a_indices range: {self.q_a_indices.min()} to {self.q_a_indices.max()}")
        if self.preserve_root_z:
            print("Root z is fixed: qpos[2] is not optimized.")
        if self.lock_root_orientation_to_source:
            print("Root orientation is fixed to the source quaternion.")

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

        safe_lower, safe_upper = _apply_joint_limit_safety_fraction(
            self.robot_model,
            self.q_a_indices,
            full_lower_limits[self.q_a_indices],
            full_upper_limits[self.q_a_indices],
            self.joint_limit_safety_fraction,
        )
        self.q_a_lb, self.q_a_ub = _apply_configured_joint_limits(
            self.robot_model,
            self.q_a_indices,
            safe_lower,
            safe_upper,
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

        self.Q_diag = _apply_configured_joint_regularization(
            self.robot_model,
            self.q_a_indices,
            self.Q_diag,
            self.end_effector_config.get("joint_regularization_weights"),
        )

        # Store smoothness weight (matching original: 0.2)
        self.smooth_weight = max(
            float(self.end_effector_config.get("smooth_weight", 0.2)), 0.0
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
        previous_target_base_orientation: Optional[np.ndarray] = None,
        source_joint_orientations: Optional[np.ndarray] = None,
        target_base_position: Optional[np.ndarray] = None,
        end_effector_targets: Optional[Mapping[str, Mapping[str, Any]]] = None,
        relation_frame: int | None = None,
        object_points: np.ndarray | None = None,
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
        if object_points is not None:
            object_points = np.asarray(object_points, dtype=float)
            if (object_points.ndim != 2 or object_points.shape[1] != 3
                    or not np.isfinite(object_points).all()):
                raise ValueError("object_points must have finite shape (N, 3).")
            # The source and robot interaction graphs share these immutable
            # world vertices. Their Jacobians are zero, just like the terrain.
            scaled_terrain_points = np.vstack([scaled_terrain_points, object_points])

        # Create interaction mesh
        vertices, tetrahedra = self.create_interaction_mesh(human_joints, scaled_terrain_points)

        # Create adjacency list
        adj_list = get_adjacency_list(tetrahedra, len(vertices))

        mesh_cfg = self.end_effector_config.get("interaction_mesh", {}) or {}
        weighting = mesh_cfg.get("weighting", "uniform")
        operator_kwargs = {}
        if weighting == "exponential":
            # These are exactly the registered targets used by the old graph.
            # kappa is in 1/metre; no additional body or scene scaling occurs.
            source_laplacian = calculate_source_laplacian(
                vertices, adj_list, kappa=float(mesh_cfg.get("kappa", 3.0))
            )
            target_laplacian = source_laplacian @ vertices
            operator_kwargs["source_laplacian"] = source_laplacian
        elif weighting == "uniform":
            # Preserve the original floating-point path when experiments are off.
            target_laplacian = calculate_laplacian_coordinates(vertices, adj_list, uniform_weight=True)
        else:
            raise ValueError("interaction_mesh.weighting must be 'uniform' or 'exponential'.")

        q_seed = self._initialize_pose(q_init, human_joints, source_joint_orientations)
        started = perf_counter()

        # Perform optimization
        q_opt = self._optimize_configuration(
            q_seed,
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
            previous_target_base_orientation=previous_target_base_orientation,
            source_joint_orientations=source_joint_orientations,
            target_base_position=target_base_position,
            end_effector_targets=end_effector_targets,
            relation_frame=relation_frame,
            **operator_kwargs,
        )
        self.last_frame_solve = dict(
            initialization=dict(self.last_pose_initialization),
            full_solve_seconds=perf_counter() - started,
            optimization=dict(getattr(self, "last_optimization", {})),
            weighting=weighting,
        )
        return q_opt

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
        previous_target_base_orientation: Optional[np.ndarray] = None,
        source_joint_orientations: Optional[np.ndarray] = None,
        target_base_position: Optional[np.ndarray] = None,
        end_effector_targets: Optional[Mapping[str, Mapping[str, Any]]] = None,
        relation_frame: int | None = None,
        source_laplacian: Optional[np.ndarray] = None,
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
        q = (q_init.copy() if getattr(self, "_defer_self_collision_path_repair", False)
             else self._self_collision_initial_guess(q_init, q_last))
        last_cost = np.inf
        self.last_optimization = dict(iterations=0, failed_subproblems=0, status="no_iterations")
        operator_kwargs = {} if source_laplacian is None else {"source_laplacian": source_laplacian}

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
                previous_target_base_orientation,
                source_joint_orientations,
                target_base_position,
                end_effector_targets,
                relation_frame,
                **operator_kwargs,
            )
            self.last_optimization["iterations"] += 1

            if show_debug and iteration < 3:
                print(f"  Iter {iteration}: cost={cost:.4f}, |dq|={np.linalg.norm(q_new - q):.6f}, status={'OK' if cost < np.inf else 'FAIL'}")

            if not np.isfinite(cost):
                self.last_optimization["failed_subproblems"] += 1
                self.last_optimization["status"] = "subproblem_failed"
                if show_debug:
                    print(f"  Stopping SQP at iteration {iteration}: convex subproblem failed.")
                break

            self.last_optimization["status"] = "finite_result"

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
