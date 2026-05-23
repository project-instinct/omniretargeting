"""Core retargeting functionality adapted for generic robots and terrains."""

from __future__ import annotations

import numpy as np
import mujoco
import cvxpy as cp
from scipy import sparse as sp
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation
import trimesh
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from .utils import (
    sample_points_on_mesh,
    compute_mesh_height_at_point,
    transform_points_local_to_world,
    get_adjacency_list,
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
)
from .data_sources.base import validate_motion_positions


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
        source_target_names: Optional[List[str]] = None,
        replace_cylinders_with_capsules: bool = False,
        hard_penetration_constraint: bool = False,
        link_offset_config: Optional[Dict[str, np.ndarray]] = None,
    ):
        """Initialize the generic retargeter.

        Args:
            robot_model: MuJoCo model of the robot
            robot_data: MuJoCo data for the robot
            terrain_mesh: Terrain mesh (already scaled if needed)
            joint_mapping: Mapping from source target names to robot link names
            robot_height: Height of the robot
            q_a_init_idx: Index where optimization variables start
            step_size: Trust region size for SQP
            penetration_tolerance: Tolerance for penetration constraints
            foot_sticking_tolerance: Tolerance for foot sticking
            collision_detection_threshold: Distance threshold for collision detection
            terrain_sample_points: Number of sampled terrain points for interaction mesh
            source_target_names: Ordered source target names to ensure consistent ordering
            replace_cylinders_with_capsules: If True, replace all cylinder collision geoms
                with capsules before computing penetration constraints. This matches
                IsaacLab/PhysX convention where ``replace_cylinders_with_capsules=True``
                is commonly used, ensuring that the retargeted motion is checked against
                the same collision shapes used in downstream simulation.
            hard_penetration_constraint: If True, enforce penetration
                constraints inside the optimizer. If False, skip them so
                outer post-processing can handle contact correction.
            link_offset_config: Optional per-link offset dictionary. Keys are robot
                link names (values in joint_mapping), values are 3-element local-frame
                offset vectors [dx, dy, dz] (meters). The offset represents the
                displacement from the link body origin to the actual target position
                (e.g., a source landmark or keypoint position). The local frame is the link
                coordinate system as expressed in the URDF.
        """
        self.robot_model = robot_model
        self.robot_data = robot_data
        self.terrain_mesh = terrain_mesh
        self.joint_mapping = joint_mapping  # This should already be filtered to valid source targets only.
        self.robot_height = robot_height
        self.hard_penetration_constraint = hard_penetration_constraint

        # ---- link offset configuration ----
        # Normalize offsets to numpy arrays and validate keys
        self.link_offset_config: Dict[str, np.ndarray] = {}
        if link_offset_config:
            robot_link_names = set(joint_mapping.values())
            for link_name, offset in link_offset_config.items():
                if link_name not in robot_link_names:
                    import warnings
                    warnings.warn(
                        f"link_offset_config key {link_name} is not in joint_mapping. "
                        f"Skipping. Available links: {sorted(robot_link_names)}",
                        UserWarning
                    )
                    continue
                self.link_offset_config[link_name] = np.asarray(offset, dtype=float).reshape(3)
            print(f"Loaded link offsets for {len(self.link_offset_config)} link(s): "
                  f"{list(self.link_offset_config.keys())}")

        # CRITICAL: Store ordered source target names to ensure consistent ordering.
        # This ensures source_target_positions[i] matches robot_points[i] for all i.
        if source_target_names is not None:
            self.source_target_names = source_target_names
            # Verify that source_target_names matches joint_mapping keys.
            if set(self.source_target_names) != set(joint_mapping.keys()):
                raise ValueError(
                    f"source_target_names ({set(self.source_target_names)}) "
                    f"does not match joint_mapping keys ({set(joint_mapping.keys())})"
                )
        else:
            # Fallback: use dictionary insertion order (Python 3.7+)
            self.source_target_names = list(joint_mapping.keys())
        
        # Validate that all mapped robot links exist.
        # This is a final safety check - fail fast if links are missing.
        self._validate_joint_mapping()

        # Retargeting parameters
        self.q_a_init_idx = q_a_init_idx
        self.step_size = step_size
        self.penetration_tolerance = penetration_tolerance
        self.foot_sticking_tolerance = foot_sticking_tolerance
        self.collision_detection_threshold = collision_detection_threshold
        self.terrain_sample_points = int(terrain_sample_points)

        # Apply cylinder → capsule replacement if requested
        if replace_cylinders_with_capsules:
            self._replace_cylinders_with_capsules()

        # Setup robot configuration
        self._setup_robot_config()

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
        self.nq_a = len(self.q_a_indices)
        
        print(f"Robot config: nq={self.nq}, nv={self.nv}, nq_a={self.nq_a}")
        print(f"q_a_indices range: {self.q_a_indices.min()} to {self.q_a_indices.max()}")

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

        self.q_a_lb = full_lower_limits[self.q_a_indices]
        self.q_a_ub = full_upper_limits[self.q_a_indices]

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
    
    def _validate_joint_mapping(self):
        """Validate that all mapped robot links exist. Raise error if any are missing.
        
        This method now delegates to the shared utility function in utils.py.
        """
        from .utils import validate_robot_joint_mapping
        validate_robot_joint_mapping(
            self.robot_model,
            self.joint_mapping,
            raise_on_missing=True
        )

    def _setup_terrain_interaction(self):
        """Setup terrain interaction parameters."""
        # Sample points on terrain for interaction mesh
        self.terrain_points = sample_points_on_mesh(self.terrain_mesh, self.terrain_sample_points)


    def create_interaction_mesh(
        self,
        source_target_positions: np.ndarray,
        terrain_points: np.ndarray,
        object_points: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create interaction mesh from source target positions, terrain points, and optional object points.

        Args:
            source_target_positions: Source target positions (N, 3)
            terrain_points: Terrain surface points (M, 3)
            object_points: Optional object surface points (K, 3)

        Returns:
            Tuple of (vertices, tetrahedra)
        """
        # Combine source targets, terrain points, and object points.
        vertices = [source_target_positions, terrain_points]
        if object_points is not None and len(object_points) > 0:
            vertices.append(object_points)
        vertices = np.vstack(vertices)

        # Create Delaunay triangulation
        tri = Delaunay(vertices)

        return vertices, tri.simplices

    def retarget_frame(
        self,
        source_target_positions: np.ndarray,
        q_init: np.ndarray,
        max_iter: int = 10,
        q_last: Optional[np.ndarray] = None,
        target_base_orientation: Optional[np.ndarray] = None,
        object_points: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Retarget a single frame of source target positions to robot motion.

        Args:
            source_target_positions: Mapped source target positions (N, 3)
            q_init: Initial robot configuration
            max_iter: Maximum optimization iterations
            q_last: Configuration at previous time step (for smoothness)
            object_points: Optional object surface points (K, 3)

        Returns:
            Optimized robot configuration
        """
        # self.terrain_points are sampled from the terrain mesh passed to this retargeter.
        # The caller owns any batch scaling before constructing the stream state.
        terrain_points = self.terrain_points

        # Create interaction mesh
        vertices, tetrahedra = self.create_interaction_mesh(
            source_target_positions, terrain_points, object_points
        )

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
            terrain_points,
            max_iter=max_iter,
            q_last=q_last,
            target_base_orientation=target_base_orientation,
            object_points=object_points,
        )

        return q_opt

    def _optimize_configuration(
        self,
        q_init: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: List[List[int]],
        terrain_points: np.ndarray,
        max_iter: int = 10,
        q_last: Optional[np.ndarray] = None,
        target_base_orientation: Optional[np.ndarray] = None,
        object_points: Optional[np.ndarray] = None,
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
                q, target_laplacian, adj_list, terrain_points, q_last, target_base_orientation, object_points
            )
            
            if show_debug and iteration < 3:
                print(f"  Iter {iteration}: cost={cost:.4f}, |dq|={np.linalg.norm(q_new - q):.6f}, status={'OK' if cost < np.inf else 'FAIL'}")

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
        q_last: Optional[np.ndarray] = None,
        target_base_orientation: Optional[np.ndarray] = None,
        object_points: Optional[np.ndarray] = None,
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

        # Compute Jacobians for mapped robot link points.
        J_V, p_V, _ = self._compute_robot_jacobians(q)

        # Create Laplacian matrices
        # CRITICAL: Ensure robot_points are in the SAME ORDER as source_target_positions passed to retarget_frame.
        # The order MUST match: source_target_positions[i] corresponds to robot_points[i] for all i.
        # 
        # Order flow:
        #   1. In core.py: valid_source_target_names is built by iterating joint_mapping.keys() in order.
        #   2. In core.py: mapped_source_targets = source_positions[mapped_source_target_indices]
        #      where mapped_source_target_indices corresponds to valid_source_target_names[i].
        #   3. In core.py: valid_source_to_robot_link_mapping preserves valid_source_target_names order.
        #   4. Here: self.joint_mapping is valid_source_to_robot_link_mapping (passed from core.py).
        #   5. Here: robot_points built by iterating source_target_names.
        #   6. J_V stacked by iterating self.joint_mapping.keys() in the same order
        #
        # So: source_target_positions[i] should match robot_points[i].
        # CRITICAL: Use self.source_target_names to ensure consistent ordering.
        source_target_names_ordered = self.source_target_names
        
        # CRITICAL: Verify that p_V has all source targets in the correct order.
        if set(source_target_names_ordered) != set(p_V.keys()):
            missing = set(source_target_names_ordered) - set(p_V.keys())
            extra = set(p_V.keys()) - set(source_target_names_ordered)
            raise RuntimeError(
                f"Source target mismatch: p_V has different targets than joint_mapping. "
                f"Missing from p_V: {missing}, Extra in p_V: {extra}"
            )
        
        robot_points = []
        for target_name in source_target_names_ordered:
            robot_points.append(p_V[target_name])
        
        robot_points = np.array(robot_points)
        
        # Debug: Print order for first frame to verify
        import sys
        if not hasattr(sys, '_omni_joint_order_printed'):
            print(f"\n=== Source Target Order Verification ===")
            print(f"Source target mapping order: {source_target_names_ordered}")
            print(f"p_V keys order: {list(p_V.keys())}")
            print(f"First 3 robot points correspond to: {source_target_names_ordered[:3]}")
            sys._omni_joint_order_printed = True
        
        # CRITICAL: Validate that sizes match exactly
        expected_num_targets = len(source_target_names_ordered)
        if len(robot_points) != expected_num_targets:
            raise ValueError(
                f"Size mismatch: robot_points has {len(robot_points)} targets, "
                f"but expected {expected_num_targets} targets from joint_mapping.keys()."
            )
        if J_V.shape[0] != 3 * expected_num_targets:
            raise ValueError(
                f"Jacobian dimension mismatch: J_V has {J_V.shape[0]//3} targets, "
                f"but expected {expected_num_targets} targets from joint_mapping. "
                f"J_V shape: {J_V.shape}, expected rows: {3 * expected_num_targets}"
            )
        if len(robot_points) != J_V.shape[0] // 3:
            raise ValueError(
                f"Size mismatch between robot_points ({len(robot_points)}) and J_V ({J_V.shape[0]//3} targets)."
            )
        # Combine all environment points (terrain + objects) as locked vertices
        env_points_list = [terrain_points]
        if object_points is not None and len(object_points) > 0:
            env_points_list.append(object_points)
        all_env_points = np.vstack(env_points_list)

        if len(robot_points) == 0:
            if len(all_env_points) == 0:
                raise ValueError("Both robot_points and environment points are empty")
            vertices = all_env_points
            print("WARNING: No robot points found! Only using environment points.")
        else:
            vertices = np.vstack([robot_points, all_env_points])

        # CRITICAL: Use uniform_weight=True to match target_laplacian computation
        # This ensures consistent Laplacian computation between target and current
        L = calculate_laplacian_matrix(vertices, adj_list, uniform_weight=True)
        if not sp.issparse(L):
            L = sp.csr_matrix(L)

        # Kron shape: (3*num_vertices, 3*num_vertices)
        Kron = sp.kron(L, sp.eye(3, format="csr"), format="csr")
        
        # J_V shape: (3*num_targets, nq_a) - stacked Jacobians for each mapped source target.
        # BUT Kron expects input vector of size 3*num_vertices (where vertices = mapped source targets + terrain_points).
        # J_V maps qpos deltas (dqa) to velocities of mapped robot link points (and 0 for terrain points).
        
        # We need to construct a full Jacobian J_full of shape (3*num_vertices, nq_a)
        # The top part corresponds to robot_points (mapped source targets), bottom part (terrain) is zeros.
        
        num_robot_points = len(robot_points)
        num_env_points = len(all_env_points)
        num_vertices = num_robot_points + num_env_points
        
        # Verify sizes match
        if J_V.shape[0] != 3 * num_robot_points:
             # This can happen if p_V has different number of points than J_V's stack
             # But they come from the same loop, so they should match
             print(f"Warning: J_V rows ({J_V.shape[0]}) != 3 * num_robot_points ({3*num_robot_points})")
        
        # Construct full Jacobian for all vertices
        # Top part: J_V (robot points), Bottom part: 0 (environment points, static)
        J_full_vertices = sp.vstack([
            sp.csr_matrix(J_V),  # Jacobians for robot points
            sp.csr_matrix((3 * num_env_points, self.nq_a)) # Zeros for environment points (terrain + objects)
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

        # Non-penetration constraints (self-collision + terrain)
        if self.hard_penetration_constraint:
            penetration_constraints = self._compute_penetration_constraints(q, dqa)
            constraints.extend(penetration_constraints)

        # Trust region
        constraints.append(cp.SOC(self.step_size, dqa))

        # Objective - matching original implementation exactly
        weights = np.ones(len(vertices)) * 10  # Laplacian weights (matching original laplacian_weights = 10)
        sqrt_w3 = np.sqrt(np.repeat(weights, 3))
        
        # Minimize: ||lap_var - target_lap_vec||^2
        # where lap_var = lap0_vec + J_L @ dqa (from constraint)
        obj = cp.sum_squares(cp.multiply(sqrt_w3, lap_var - target_lap_vec))
        
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
        obj += cp.sum_squares(cp.multiply(np.sqrt(Q_diag_modified), dqa + q_a_current))

        # Smoothness cost (matching original implementation exactly)
        # CRITICAL FIX: Use previous frame's velocity, not current guess
        if q_last is not None:
            q_a_current = q[self.q_a_indices]  # Current guess at this SQP iteration
            dqa_smooth = q_last[self.q_a_indices] - q_a_current  # Velocity from prev frame to current guess
            obj += self.smooth_weight * cp.sum_squares(dqa - dqa_smooth)
        
        # Base orientation tracking cost
        # Keep the base orientation close to the target estimated from source target positions.
        if target_base_orientation is not None and 3 in self.q_a_indices:
            # Find quaternion indices in q_a_indices
            quat_indices_in_qa = []
            for quat_idx in [3, 4, 5, 6]:  # wxyz quaternion
                if quat_idx in self.q_a_indices:
                    idx_in_qa = np.where(self.q_a_indices == quat_idx)[0]
                    if len(idx_in_qa) > 0:
                        quat_indices_in_qa.append(idx_in_qa[0])
            
            if len(quat_indices_in_qa) == 4:
                # Target quaternion in wxyz (MuJoCo convention)
                quat_target_wxyz = target_base_orientation
                quat_current = q[3:7]  # Current quaternion
                
                # Add cost to keep quaternion close to target
                orientation_weight = 5.0  # Strong preference to maintain orientation
                for i, qa_idx in enumerate(quat_indices_in_qa):
                    target_val = quat_target_wxyz[i]
                    current_val = quat_current[i]
                    # Penalize deviation: (q_new - target)^2 = (q_current + dqa - target)^2
                    obj += orientation_weight * cp.square(dqa[qa_idx] + current_val - target_val)

        # Solve
        problem = cp.Problem(cp.Minimize(obj), constraints)
        
        import sys
        show_solver_debug = hasattr(sys, '_omni_frame_count') and sys._omni_frame_count <= 5
        
        try:
            problem.solve(solver=cp.CLARABEL, verbose=False)

            if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                if show_solver_debug:
                    print(f"    First solve failed: {problem.status}, trying without SOC...")
                # Fallback to simpler problem without trust region
                constraints = [c for c in constraints if not isinstance(c, cp.constraints.second_order.SOC)]
                problem = cp.Problem(cp.Minimize(obj), constraints)
                problem.solve(solver=cp.CLARABEL, verbose=False)
                if show_solver_debug:
                    print(f"    Second solve status: {problem.status}")

            if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                dqa_opt = dqa.value
                cost = problem.value

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
        where ω and v are WORLD-expressed in MuJoCo.
        - 23 hinge joints: v = qdot.

        If use_world_omega=False, uses BODY-omega mapping (for debugging).
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

    def _skew(self, v: np.ndarray) -> np.ndarray:
        """Return 3x3 skew-symmetric matrix of vector v."""
        return np.array([
            [0.0, -v[2],  v[1]],
            [v[2],  0.0, -v[0]],
            [-v[1],  v[0],  0.0],
        ], dtype=float)

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
        """Compute Jacobians for mapped robot link points in world frame.
        
        Args:
            q: Robot configuration
            
        Returns:
            Tuple of (J_V, p_dict, None):
                - J_V: Stacked Jacobians (3*num_targets, nq_a)
                - p_dict: Dictionary of robot link point positions keyed by source target name
                - None: Placeholder for compatibility
        """
        J_dict = {}
        p_dict = {}

        for target_name, link_name in self.joint_mapping.items():
            try:
                body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)

                # Get position in world frame
                pos = self.robot_data.xpos[body_id].copy()

                # Compute base Jacobian for body origin (3 x nq)
                J_base = self._calc_contact_jacobian_from_point(body_id)

                # Apply offset if configured
                if link_name in self.link_offset_config:
                    o_local = self.link_offset_config[link_name]
                    R_WB = self.robot_data.xmat[body_id].reshape(3, 3)
                    o_world = R_WB @ o_local
                    pos = pos + o_world  # p_target = p_body + R @ o_local

                    # Rotational Jacobian Jr (3 x nq) for the cross-term correction
                    p_WB = self.robot_data.xpos[body_id]
                    p_W = p_WB.astype(np.float64).reshape(3, 1)
                    Jp = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
                    Jr = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
                    mujoco.mj_jac(self.robot_model, self.robot_data, Jp, Jr, p_W, int(body_id))
                    T = self._build_transform_qdot_to_qvel_fast()
                    Jr_world = Jr @ T  # rotational Jacobian in world frame (3 x nq)
                    # Cross-term: skew(o_world) @ Jr_world
                    J_full = J_base + self._skew(o_world) @ Jr_world
                else:
                    J_full = J_base

                # Extract optimized part (J_full is already in qpos coordinates)
                valid_indices = self.q_a_indices[self.q_a_indices < J_full.shape[1]]
                if len(valid_indices) < len(self.q_a_indices):
                    print(
                        f"Warning: Truncating indices for source target {target_name}. "
                        f"J width: {J_full.shape[1]}, Max idx: {self.q_a_indices.max()}"
                    )

                J_reduced = J_full[:, valid_indices]
                
                # Pad if needed
                if J_reduced.shape[1] < self.nq_a:
                    J_pad = np.zeros((3, self.nq_a))
                    J_pad[:, :J_reduced.shape[1]] = J_reduced
                    J_reduced = J_pad
                    
                J_dict[target_name] = J_reduced
                p_dict[target_name] = pos

            except Exception as e:
                # CRITICAL: All mapped targets should resolve to robot links (validated in __init__).
                # Raise error instead of skipping to ensure size consistency
                build_error_msg = (
                    f"Failed to compute Jacobian for source target '{target_name}' -> link '{link_name}'. "
                    f"This should not happen if joint_mapping was validated. Error: {e}"
                )
                raise RuntimeError(build_error_msg) from e

        # Stack Jacobians in the SAME ORDER as source_target_names to match source_target_positions order.
        # This is critical for correct Laplacian matching!
        source_target_names_ordered = self.source_target_names
        num_targets = len(source_target_names_ordered)
        
        if num_targets > 0:
            J_V = np.zeros((3 * num_targets, self.nq_a))
            for i, target_name in enumerate(source_target_names_ordered):
                if target_name in J_dict:
                    J = J_dict[target_name]
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
                    # CRITICAL: All targets should exist (validated in __init__), so this is unexpected.
                    raise RuntimeError(
                        f"Jacobian for source target '{target_name}' not found in J_dict. "
                        f"This should not happen if joint_mapping was validated. "
                        f"Available targets in J_dict: {list(J_dict.keys())}"
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
                # Capsule: size[0] = radius, size[1] = half-length along x
                radius = size[0]
                half_len = size[1]
                # Sample points on each end hemisphere + mid-body side points
                for s in [-half_len, half_len]:
                    for dx, dy, dz in [(1, 0, 0), (-1, 0, 0),
                                       (0, 1, 0), (0, -1, 0),
                                       (0, 0, 1), (0, 0, -1)]:
                        local_pt = np.array([s, 0, 0])
                        if dx != 0:
                            local_pt[0] += dx * radius
                        elif dy != 0:
                            local_pt[1] += dy * radius
                        else:
                            local_pt[2] += dz * radius
                        world_pt = geom_pos + geom_rot.apply(local_pt)
                        points.append(world_pt)
                # Add mid-body points along the cylinder surface
                for theta in [0, np.pi/2, np.pi, 3*np.pi/2]:
                    local_pt = np.array([
                        0,
                        radius * np.cos(theta),
                        radius * np.sin(theta)
                    ])
                    world_pt = geom_pos + geom_rot.apply(local_pt)
                    points.append(world_pt)
                return np.array(points)

            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                # Cylinder: size[0] = radius, size[1] = half-length along x
                radius = size[0]
                half_len = size[1]
                # Sample center of each end cap + 4 side points at midpoint
                for s in [-half_len, half_len]:
                    local_pt = np.array([s, 0, 0])
                    world_pt = geom_pos + geom_rot.apply(local_pt)
                    points.append(world_pt)
                for theta in [0, np.pi/2, np.pi, 3*np.pi/2]:
                    local_pt = np.array([
                        0,
                        radius * np.cos(theta),
                        radius * np.sin(theta)
                    ])
                    world_pt = geom_pos + geom_rot.apply(local_pt)
                    points.append(world_pt)
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


def retarget_source_to_robot(
    source_positions: np.ndarray,
    robot_urdf_path: Path,
    terrain_mesh_path: Path,
    joint_mapping: Dict[str, str],
    robot_height: Optional[float] = None,
    source_target_names: Optional[List[str]] = None,
) -> Tuple[float, np.ndarray]:
    """
    High-level function to retarget source target positions to any robot on any terrain.

    Args:
        source_positions: Source target positions (T, N, 3)
        robot_urdf_path: Path to robot URDF
        terrain_mesh_path: Path to terrain mesh
        joint_mapping: Mapping from source target names to robot links
        robot_height: Robot height override
        source_target_names: Ordered source target names for source_positions

    Returns:
        Tuple of (source_to_robot_scale, retargeted_trajectory)
    """
    # Validate inputs
    if not validate_motion_positions(source_positions):
        raise ValueError("Invalid source position trajectory format")

    from .core import OmniRetargeter

    retargeter = OmniRetargeter(
        robot_urdf_path=robot_urdf_path,
        terrain_mesh_path=terrain_mesh_path,
        joint_mapping=joint_mapping,
        robot_height=robot_height,
        source_target_names=source_target_names,
    )
    return retargeter.retarget_motion(
        source_positions,
        visualize_trajectory=False,
        enable_terrain_scaling=True,
    )
