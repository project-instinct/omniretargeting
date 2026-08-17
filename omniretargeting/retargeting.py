"""Core retargeting functionality adapted for generic robots and terrains."""

from __future__ import annotations

import numpy as np
import mujoco
import clarabel
from scipy import sparse as sp
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation
import trimesh
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import fnmatch

from .utils import (
    sample_points_on_mesh,
    compute_mesh_height_at_point,
    transform_points_local_to_world,
    get_adjacency_list,
    calculate_exponential_edge_weights,
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
)
from .data_sources.base import validate_motion_positions


def parse_bone_chains(
    chains: List[List[str]],
    source_target_names: List[str],
) -> List[Tuple[int, int, int]]:
    """Convert bone chains of source target names into adjacent bone-pair triples.

    Each chain [t0, t1, t2, ...] defines bones (t0->t1), (t1->t2), ... along the
    chain (TopoRetarget Sec. 3.2: bones follow the same limb/finger). Adjacent
    bone pairs AB are consecutive bones, stored as index triples (a, b, c)
    meaning bone k = (a->b) and bone l = (b->c). Chains of length 2 define a
    bone but contribute no adjacent pair.

    Args:
        chains: List of chains, each a list of source target names.
        source_target_names: Ordered mapped source target names.

    Returns:
        List of (a, b, c) index triples into the mapped target arrays.

    Raises:
        ValueError: If a chain has fewer than 2 targets or names an unmapped target.
    """
    index = {name: i for i, name in enumerate(source_target_names)}
    triples: List[Tuple[int, int, int]] = []
    for chain in chains:
        if len(chain) < 2:
            raise ValueError(f"bone_direction chain must list at least 2 targets, got {chain}")
        for name in chain:
            if name not in index:
                raise ValueError(
                    f"bone_direction chain target '{name}' is not a mapped source target. "
                    f"Available targets: {source_target_names}"
                )
        idx = [index[name] for name in chain]
        for i in range(len(idx) - 2):
            triples.append((idx[i], idx[i + 1], idx[i + 2]))
    return triples


def compute_bone_direction_targets(
    points: np.ndarray,
    triples: List[Tuple[int, int, int]],
    eps: float = 1e-8,
) -> np.ndarray:
    """Source-side relative bone directions (d_k - d_l) per adjacent bone triple.

    d_k is the unit vector from keypoint a to keypoint b (TopoRetarget Eq. 1).
    Directions are expressed in the world frame: source and robot world frames
    are aligned by the retargeting pipeline, so relative directions are directly
    comparable (the paper uses wrist-attached frames for the same purpose).

    Args:
        points: (N, 3) source target positions (mapped order).
        triples: Adjacent bone triples (a, b, c) from parse_bone_chains.
        eps: Lower bound for segment lengths when normalizing.

    Returns:
        (3 * len(triples),) stacked (d_k - d_l) vectors.
    """
    targets = np.zeros(3 * len(triples))
    for p, (a, b, c) in enumerate(triples):
        e_k = points[b] - points[a]
        e_l = points[c] - points[b]
        d_k = e_k / max(np.linalg.norm(e_k), eps)
        d_l = e_l / max(np.linalg.norm(e_l), eps)
        targets[3 * p: 3 * p + 3] = d_k - d_l
    return targets


def compute_bone_direction_residual_and_jacobian(
    robot_points: np.ndarray,
    J_V: np.ndarray,
    triples: List[Tuple[int, int, int]],
    targets: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Robot-side bone-direction residual and its Jacobian w.r.t. dqa.

    For bone (a->b) with segment e = p_b - p_a, L = ||e||, d = e / L:
        dd/dq = (I - d d^T) / L @ (J_b - J_a)
    For an adjacent pair (k, l) the residual is (d_k - d_l) - target and its
    Jacobian is Jd_k - Jd_l, so residual(q + dq) ~= res + J @ dqa.

    Args:
        robot_points: (N, 3) current robot link point positions (mapped order).
        J_V: (3N, nq_a) stacked translational Jacobians (same order).
        triples: Adjacent bone triples (a, b, c).
        targets: (3P,) source-side (d_k - d_l) from compute_bone_direction_targets.
        eps: Lower bound for segment lengths when normalizing.

    Returns:
        (res, J): residual (3P,) and Jacobian (3P, nq_a).
    """
    n_pairs = len(triples)
    res = np.zeros(3 * n_pairs)
    J = np.zeros((3 * n_pairs, J_V.shape[1]))
    eye3 = np.eye(3)
    for p, (a, b, c) in enumerate(triples):
        J_a, J_b, J_c = J_V[3 * a: 3 * a + 3], J_V[3 * b: 3 * b + 3], J_V[3 * c: 3 * c + 3]
        e_k = robot_points[b] - robot_points[a]
        e_l = robot_points[c] - robot_points[b]
        L_k = max(np.linalg.norm(e_k), eps)
        L_l = max(np.linalg.norm(e_l), eps)
        d_k = e_k / L_k
        d_l = e_l / L_l
        Jd_k = (eye3 - np.outer(d_k, d_k)) / L_k @ (J_b - J_a)
        Jd_l = (eye3 - np.outer(d_l, d_l)) / L_l @ (J_c - J_b)
        res[3 * p: 3 * p + 3] = (d_k - d_l) - targets[3 * p: 3 * p + 3]
        J[3 * p: 3 * p + 3] = Jd_k - Jd_l
    return res, J


def _solve_qp_clarabel(
    P: sp.spmatrix,
    c: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    ge_rows: Optional[List[Tuple[np.ndarray, float]]] = None,
    trust_region_dim: Optional[int] = None,
    step_size: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], bool]:
    """Solve a small dense QP directly with CLARABEL (no CVXPY overhead).

        min  0.5 x' P x + c' x
        s.t. lb <= x <= ub
             a' x >= h            for each (a, h) in ge_rows
             ||x[:trust_region_dim]||_2 <= step_size   (if both given)

    This is the exact same KKT system CVXPY sends to CLARABEL for the
    per-iteration SQP subproblem (same solver, same default settings), but
    assembled with numpy/scipy directly: the Laplacian error definition
    ``lap_var = lap0 + J_L @ dqa`` is substituted into the objective, so the
    problem keeps only ``len(c)`` scalar variables instead of carrying the
    3*num_vertices auxiliary ``lap_var`` variables plus equality constraints.

    Returns:
        (x, True) on Solved/AlmostSolved, (None, False) otherwise.
    """
    n = len(c)
    A_parts = [sp.vstack([sp.eye(n, format="csc"), -sp.eye(n, format="csc")], format="csc")]
    b_parts = [np.concatenate([ub, -lb])]
    n_nonneg = 2 * n

    if ge_rows:
        A_parts.append(sp.csr_matrix(np.vstack([-np.asarray(a, dtype=float).ravel() for a, _ in ge_rows])))
        b_parts.append(np.array([-h for _, h in ge_rows], dtype=float))
        n_nonneg += len(ge_rows)

    A = sp.vstack(A_parts, format="csc")
    b = np.concatenate(b_parts)
    cones = [clarabel.NonnegativeConeT(n_nonneg)]

    if trust_region_dim is not None and step_size is not None:
        d = int(trust_region_dim)
        A_soc = sp.vstack(
            [
                sp.csr_matrix((1, n)),
                sp.hstack([-sp.eye(d, format="csc"), sp.csr_matrix((d, n - d))], format="csc"),
            ],
            format="csc",
        )
        A = sp.vstack([A, A_soc], format="csc")
        b = np.concatenate([b, [float(step_size)], np.zeros(d)])
        cones.append(clarabel.SecondOrderConeT(d + 1))

    settings = clarabel.DefaultSettings()
    settings.verbose = False

    try:
        P_sym = sp.triu((P + P.T) * 0.5, format="csc")
        solver = clarabel.DefaultSolver(P_sym, np.asarray(c, dtype=float), A, b, cones, settings)
        sol = solver.solve()
    except Exception as e:
        print(f"WARNING: CLARABEL QP solve raised: {e}")
        return None, False

    if sol.status in (clarabel.SolverStatus.Solved, clarabel.SolverStatus.AlmostSolved):
        return np.asarray(sol.x, dtype=float).ravel(), True
    return None, False


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
        joint_mapping: Dict[str, str | Dict[str, object]],
        robot_height: float,
        q_a_init_idx: int = -7,
        step_size: float = 0.2,
        penetration_tolerance: float = 1e-3,
        foot_sticking_tolerance: float = 1e-3,
        collision_detection_threshold: float = 0.1,
        terrain_sample_points: int = 100,
        terrain_sample_seed: Optional[int] = None,
        source_target_names: Optional[List[str]] = None,
        replace_cylinders_with_capsules: bool = False,
        hard_penetration_constraint: bool = False,
        joint_regularization_boost: Optional[Dict] = None,
        laplacian_edge_weighting: str = "uniform",
        laplacian_distance_decay: float = 30.0,
        bone_direction: Optional[Dict] = None,
        penetration_slack: Optional[Dict] = None,
        base_position_tracking_weight: float = 0.0,
        foot_orientation: Optional[Dict] = None,
    ):
        """Initialize the generic retargeter.

        Args:
            robot_model: MuJoCo model of the robot
            robot_data: MuJoCo data for the robot
            terrain_mesh: Terrain mesh (already scaled if needed)
            joint_mapping: Mapping from source target names to robot link names.
                Each value is either a string (link name, zero offset) or a dict
                with keys ``link`` (robot link name) and ``offset`` (optional
                3-element local-frame offset vector [dx, dy, dz], default [0,0,0]).
            robot_height: Height of the robot
            q_a_init_idx: Index where optimization variables start
            step_size: Trust region size for SQP
            penetration_tolerance: Tolerance for penetration constraints
            foot_sticking_tolerance: Tolerance for foot sticking
            collision_detection_threshold: Distance threshold for collision detection
            terrain_sample_points: Number of sampled terrain points for interaction mesh
            terrain_sample_seed: Seed for the terrain surface draw. Leave unset for the
                historical non-reproducible behaviour; set it to compare runs.
            source_target_names: Ordered source target names to ensure consistent ordering
            replace_cylinders_with_capsules: If True, replace all cylinder collision geoms
                with capsules before computing penetration constraints. This matches
                IsaacLab/PhysX convention where ``replace_cylinders_with_capsules=True``
                is commonly used, ensuring that the retargeted motion is checked against
                the same collision shapes used in downstream simulation.
            hard_penetration_constraint: If True, enforce penetration
                constraints inside the optimizer. If False, skip them so
                outer post-processing can handle contact correction.
            laplacian_edge_weighting: Weighting scheme for the interaction-mesh
                Laplacian edges. "uniform" (default) keeps the original
                OmniRetarget behavior (every neighbor weighted equally).
                "exponential" uses the distance-dependent weights from
                TopoRetarget (arXiv:2606.16272, Eq. 5): w_ij ∝ exp(-kappa * d_ij),
                computed once on the source configuration per frame and reused
                for the robot-side Laplacian.
            laplacian_distance_decay: Spatial decay factor kappa used when
                laplacian_edge_weighting == "exponential" (paper uses 30 for
                meter-scale data).
            bone_direction: Optional dict enabling the TopoRetarget bone-direction
                prior (Eq. 1-2, 8). Keys:
                - enabled (bool, default False)
                - chains (list of source-target-name chains, required when enabled;
                  bones are consecutive pairs along each chain)
                - lambda_warm (float, paper 1.0): warm-init bone weight
                - lambda_smooth (float, paper 2.5): warm-init temporal weight
                - lambda_bone (float, paper 0.1): refinement-stage bone weight
                - warm_init (bool, default True): run the Eq. (2) pre-solve per frame
                - warm_init_iters (int, default 3): warm-stage SQP iterations
            penetration_slack: Optional dict of TopoRetarget slack parameters
                (Eq. 8), used in place of the single hard penetration tolerance.
                Providing this dict enables the slack mode, so it requires
                hard_penetration_constraint=True (i.e. penetration_resolver
                "hard_constraint_slack"). Keys:
                - soft_tolerance (float, paper 0.001): tau, soft penetration margin
                - hard_bound (float, paper 0.03): b, hard penetration backstop
                - slack_penalty (float, paper 1e5): w_s, quadratic slack penalty
        """
        self.robot_model = robot_model
        self.robot_data = robot_data
        self.terrain_mesh = terrain_mesh
        self.robot_height = robot_height
        self.hard_penetration_constraint = hard_penetration_constraint
        self.joint_regularization_boost = joint_regularization_boost
        if laplacian_edge_weighting not in ("uniform", "exponential"):
            raise ValueError(
                f"Unknown laplacian_edge_weighting '{laplacian_edge_weighting}'. "
                "Expected 'uniform' or 'exponential'."
            )
        self.laplacian_edge_weighting = laplacian_edge_weighting
        self.laplacian_distance_decay = float(laplacian_distance_decay)

        # ---- Parse target mapping (supports mixed string/dict values) ----
        # Extract link names and optional local-frame offsets for each source target.
        # Format: {target_name: "link_name"} or {target_name: {"robot_link": "link_name", "offset": [dx,dy,dz]}}
        self.joint_mapping: Dict[str, str] = {}
        self.target_offset_map: Dict[str, np.ndarray] = {}
        for target_name, value in joint_mapping.items():
            if isinstance(value, dict):
                link_name = value["robot_link"]
                offset = np.asarray(value.get("offset", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
                self.joint_mapping[target_name] = link_name
                self.target_offset_map[target_name] = offset
            else:
                self.joint_mapping[target_name] = value
                self.target_offset_map[target_name] = np.zeros(3, dtype=float)

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

        # ---- Bone-direction prior config (TopoRetarget Eq. 1-2, 8; default off) ----
        bd = bone_direction or {}
        self.bone_direction_enabled = bool(bd.get("enabled", False))
        self.bone_triples: List[Tuple[int, int, int]] = []
        self.lambda_warm = float(bd.get("lambda_warm", 1.0))
        self.lambda_smooth = float(bd.get("lambda_smooth", 2.5))
        self.lambda_bone = float(bd.get("lambda_bone", 0.1))
        self.bone_warm_init = bool(bd.get("warm_init", True))
        self.bone_warm_init_iters = int(bd.get("warm_init_iters", 3))
        if self.bone_direction_enabled:
            chains = bd.get("chains")
            if not chains:
                raise ValueError(
                    "bone_direction.enabled=True requires 'chains': a list of "
                    "source-target-name chains defining bones along each limb."
                )
            self.bone_triples = parse_bone_chains(chains, self.source_target_names)
            if not self.bone_triples:
                raise ValueError(
                    "bone_direction chains produced no adjacent bone pairs; "
                    "at least one chain must contain 3 or more targets."
                )

        # ---- Penetration slack parameters (TopoRetarget Eq. 8) ----
        # Enabled by passing this dict (i.e. penetration_resolver ==
        # "hard_constraint_slack"); it modifies how the hard-constraint
        # penetration resolver builds its QP constraints.
        if penetration_slack is not None and not self.hard_penetration_constraint:
            raise ValueError(
                "penetration_slack requires hard_penetration_constraint=True "
                "(penetration_resolver 'hard_constraint_slack')."
            )
        ps = penetration_slack or {}
        self.penetration_slack_enabled = penetration_slack is not None
        self.penetration_soft_tolerance = float(ps.get("soft_tolerance", 1e-3))
        self.penetration_hard_bound = float(ps.get("hard_bound", 0.03))
        self.penetration_slack_penalty = float(ps.get("slack_penalty", 1e5))
        self.base_position_tracking_weight = float(base_position_tracking_weight)
        self._setup_foot_orientation(foot_orientation)
        if self.penetration_slack_enabled and self.penetration_hard_bound <= self.penetration_soft_tolerance:
            raise ValueError(
                f"penetration_slack.hard_bound ({self.penetration_hard_bound}) must be "
                f"greater than soft_tolerance ({self.penetration_soft_tolerance})."
            )

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
        self.terrain_sample_seed = (
            None if terrain_sample_seed is None else int(terrain_sample_seed)
        )

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

        # Joint cost weights - configurable per-robot via joint_regularization_boost
        boost_cfg = self.joint_regularization_boost or {}
        default_weight = float(boost_cfg.get("default", 1e-3))
        self.Q_diag = np.ones(self.nq_a) * default_weight

        # Reduce weight for floating base to allow free movement
        base_weight = float(boost_cfg.get("base", 0.001))
        base_indices_in_qa = []
        for base_idx in range(7):
            if base_idx in self.q_a_indices:
                idx_in_qa = np.where(self.q_a_indices == base_idx)[0]
                if len(idx_in_qa) > 0:
                    base_indices_in_qa.append(idx_in_qa[0])
        
        if len(base_indices_in_qa) > 0:
            self.Q_diag[base_indices_in_qa] = base_weight
        
        # Store smoothness weight (matching original: 0.2)
        self.smooth_weight = 0.2

        # Per-joint regularization boost (constant across SQP iterations).
        # Modify Q_diag for specific joints (matching original MANUAL_COST logic).
        Q_diag_modified = self.Q_diag.copy()
        boost_joints = (self.joint_regularization_boost or {}).get("joints") or {}
        for i in range(self.robot_model.njnt):
            joint_name = self.robot_model.joint(i).name
            if not joint_name:
                continue
            joint_name_lower = joint_name.lower()
            for pattern, boost_value in boost_joints.items():
                if fnmatch.fnmatch(joint_name_lower, pattern.lower()):
                    qpos_adr = self.robot_model.jnt_qposadr[i]
                    if qpos_adr in self.q_a_indices:
                        idx_in_qa = np.where(self.q_a_indices == qpos_adr)[0]
                        if len(idx_in_qa) > 0:
                            Q_diag_modified[idx_in_qa[0]] = max(
                                Q_diag_modified[idx_in_qa[0]], float(boost_value)
                            )
                    break
        self.Q_diag_modified = Q_diag_modified
    
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
        self.terrain_points = sample_points_on_mesh(
            self.terrain_mesh, self.terrain_sample_points, seed=self.terrain_sample_seed
        )

        # Conservative bounding-sphere radius per geom, used to skip terrain
        # proximity queries for geoms that provably cannot be within
        # collision_detection_threshold of the terrain. Computed from the
        # (possibly cylinder->capsule replaced) geom types, so it must not be
        # cached before _replace_cylinders_with_capsules runs.
        self._geom_bounding_radii = np.array(
            [self._geom_bounding_radius(gi) for gi in range(self.robot_model.ngeom)]
        )

    def _geom_bounding_radius(self, geom_id: int) -> float:
        """Conservative world-space bounding-sphere radius of one geom."""
        m = self.robot_model
        geom_type = m.geom_type[geom_id]
        size = m.geom_size[geom_id]
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return float(size[0])
        if geom_type in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            # radius + half-length (conservative for cylinder, exact for capsule)
            return float(size[0] + size[1])
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            return float(np.linalg.norm(size[:3]))
        # Other types (mesh, ...): fall back to MuJoCo's bounding radius; if
        # unavailable, return inf so the geom is never prefiltered away.
        rbound = float(m.geom_rbound[geom_id])
        return rbound if rbound > 0 else np.inf


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

    def _setup_foot_orientation(self, config: Optional[Dict]) -> None:
        """Resolve the foot orientation task from the robot profile.

        Left disabled unless a profile asks for it, so existing profiles keep
        their behaviour. Bodies are resolved once here rather than per frame,
        and a missing body is an error at construction instead of a silently
        skipped objective at frame two hundred.
        """
        from omniretargeting.foot_frames import parse_body_to_sole

        self.foot_orientation_specs: Dict[str, dict] = {}
        self.foot_orientation_weight = 0.0
        self.foot_orientation_enabled = False

        if not config or not config.get("enabled", False):
            return

        weight = float(config.get("weight", 0.0))
        if weight <= 0.0:
            return

        effectors = config.get("effectors") or {}
        if not effectors:
            raise ValueError("foot_orientation is enabled but declares no effectors")

        # The body carrying the free joint is the root the residual is taken
        # relative to. Body 0 is always the world.
        free_joints = [
            j for j in range(self.robot_model.njnt)
            if self.robot_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        ]
        if len(free_joints) != 1:
            raise ValueError(
                f"foot_orientation expects exactly one free joint, found {len(free_joints)}"
            )
        self.root_body_id = int(self.robot_model.jnt_bodyid[free_joints[0]])

        for side, spec in effectors.items():
            body_name = spec.get("body")
            if not body_name:
                raise ValueError(f"foot_orientation effector {side!r} is missing 'body'")
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(
                    f"foot_orientation effector {side!r} references body {body_name!r}, "
                    "which is not present in the compiled model"
                )
            self.foot_orientation_specs[side] = {
                "body_name": body_name,
                "body_id": int(body_id),
                "body_to_sole": parse_body_to_sole(spec.get("body_to_sole"), f"{side} foot"),
            }

        self.foot_orientation_weight = weight
        self.foot_orientation_enabled = True

    def _body_angular_jacobian(self, body_id: int, T: Optional[np.ndarray] = None) -> np.ndarray:
        """World angular Jacobian of a body, in qpos coordinates, actuated columns."""
        if T is None:
            T = self._build_transform_qdot_to_qvel_fast()
        p_W = self.robot_data.xpos[int(body_id)].astype(np.float64).reshape(3, 1)
        Jp = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        Jr = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        mujoco.mj_jac(self.robot_model, self.robot_data, Jp, Jr, p_W, int(body_id))
        return (Jr @ T)[:, self.q_a_indices]

    def _foot_orientation_residuals(
        self,
        targets: Optional[Dict[str, np.ndarray]],
        T: Optional[np.ndarray] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Root-relative sole orientation residuals, one per configured foot.

        The residual is expressed relative to the floating base rather than in
        the world.  The actuated index set includes the base degrees of
        freedom, and the base carries almost no regularization, so a
        world-frame foot residual is cheapest to satisfy by rolling the whole
        robot.  Taking the residual relative to the root removes that
        shortcut: rotating the base moves target and current together and
        cannot reduce the error.  Base attitude stays the job of the existing
        base orientation objective.

        Returns a list of ``(J, e)`` with ``J`` of shape (3, nq_a), such that
        the term to minimise is ``weight * ||J @ dqa - e||^2``.
        """
        if not self.foot_orientation_enabled or not targets:
            return []

        from scipy.spatial.transform import Rotation

        if T is None:
            T = self._build_transform_qdot_to_qvel_fast()

        root_rot = self.robot_data.xmat[self.root_body_id].reshape(3, 3)
        root_jac = self._body_angular_jacobian(self.root_body_id, T=T)

        residuals: List[Tuple[np.ndarray, np.ndarray]] = []
        for side, spec in self.foot_orientation_specs.items():
            target_world = targets.get(side)
            if target_world is None:
                continue
            target_world = np.asarray(target_world, dtype=float)
            if target_world.shape != (3, 3):
                raise ValueError(f"foot orientation target for {side!r} must be a 3x3 rotation")

            current_world = self.robot_data.xmat[spec["body_id"]].reshape(3, 3) @ spec["body_to_sole"]

            current_rel = root_rot.T @ current_world
            target_rel = root_rot.T @ target_world
            error = Rotation.from_matrix(target_rel @ current_rel.T).as_rotvec()

            # Relative angular velocity in the root frame.
            jac = root_rot.T @ (self._body_angular_jacobian(spec["body_id"], T=T) - root_jac)
            residuals.append((jac, error))

        return residuals

    def retarget_frame(
        self,
        source_target_positions: np.ndarray,
        q_init: np.ndarray,
        max_iter: int = 10,
        q_last: Optional[np.ndarray] = None,
        target_base_orientation: Optional[np.ndarray] = None,
        object_points: Optional[np.ndarray] = None,
        root_translation: Optional[np.ndarray] = None,
        foot_orientation_targets: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Retarget a single frame of source target positions to robot motion.

        Args:
            source_target_positions: Mapped source target positions (N, 3)
            q_init: Initial robot configuration
            max_iter: Maximum optimization iterations
            q_last: Configuration at previous time step (for smoothness)
            object_points: Optional object surface points (K, 3)
            root_translation: Optional source root translation (3,) for base
                position tracking.

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

        # Distance-dependent edge weights (TopoRetarget Eq. 5), computed once on
        # the source configuration and reused for the robot-side Laplacian.
        # None means uniform weighting (original OmniRetarget behavior).
        if self.laplacian_edge_weighting == "exponential":
            edge_weights = calculate_exponential_edge_weights(
                vertices, adj_list, kappa=self.laplacian_distance_decay
            )
        else:
            edge_weights = None

        # Calculate target Laplacian coordinates
        # CRITICAL: Use the same weighting scheme as the matrix computation in
        # optimization so target_laplacian and lap0 stay consistent.
        target_laplacian = calculate_laplacian_coordinates(
            vertices, adj_list, uniform_weight=True, edge_weights=edge_weights
        )

        # Laplacian matrix and its 3D Kronecker lift are constant for this
        # frame: with uniform weights they depend only on adj_list, and with
        # exponential weights on the precomputed source-frame edge_weights —
        # never on the robot vertex positions that change per SQP iteration.
        L = calculate_laplacian_matrix(vertices, adj_list, uniform_weight=True, edge_weights=edge_weights)
        L = sp.csr_matrix(L)  # calculate_laplacian_matrix always returns dense
        Kron = sp.kron(L, sp.eye(3, format="csr"), format="csr")

        # Source-side relative bone directions (TopoRetarget Eq. 1), constant
        # across the SQP iterations for this frame.
        bone_targets = None
        if self.bone_direction_enabled:
            bone_targets = compute_bone_direction_targets(source_target_positions, self.bone_triples)

        # Perform optimization
        q_opt = self._optimize_configuration(
            q_init.copy(),
            target_laplacian,
            L,
            Kron,
            terrain_points,
            max_iter=max_iter,
            q_last=q_last,
            target_base_orientation=target_base_orientation,
            object_points=object_points,
            bone_targets=bone_targets,
            root_translation=root_translation,
            foot_orientation_targets=foot_orientation_targets,
        )

        return q_opt

    def _optimize_configuration(
        self,
        q_init: np.ndarray,
        target_laplacian: np.ndarray,
        L: sp.spmatrix,
        Kron: sp.spmatrix,
        terrain_points: np.ndarray,
        max_iter: int = 10,
        q_last: Optional[np.ndarray] = None,
        target_base_orientation: Optional[np.ndarray] = None,
        object_points: Optional[np.ndarray] = None,
        bone_targets: Optional[np.ndarray] = None,
        root_translation: Optional[np.ndarray] = None,
        foot_orientation_targets: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Optimize robot configuration using SQP with interaction mesh constraints.

        Args:
            q_init: Initial configuration
            target_laplacian: Target Laplacian coordinates
            L: Per-frame Laplacian matrix (constant across SQP iterations)
            Kron: Per-frame kron(L, eye(3)) (constant across SQP iterations)
            terrain_points: Terrain contact points
            max_iter: Maximum iterations
            q_last: Configuration at previous time step (for smoothness)
            bone_targets: Optional source-side relative bone directions for the
                bone-direction prior (from compute_bone_direction_targets)
            root_translation: Optional source root translation (3,) for base
                position tracking.

        Returns:
            Optimized configuration
        """
        q = q_init.copy()

        # Warm-init stage (TopoRetarget Eq. 2): refine the initial guess toward
        # the source's relative bone directions before the main refinement.
        if self.bone_direction_enabled and bone_targets is not None and self.bone_warm_init:
            q = self._warm_init_bone_direction(q, bone_targets, q_last)

        last_cost = np.inf

        for iteration in range(max_iter):
            # Single optimization step
            q_new, cost = self._single_optimization_step(
                q,
                target_laplacian,
                L,
                Kron,
                terrain_points,
                q_last=q_last,
                target_base_orientation=target_base_orientation,
                object_points=object_points,
                bone_targets=bone_targets,
                root_translation=root_translation,
                foot_orientation_targets=foot_orientation_targets,
            )

            # Solver failure at this linearization point: an identical retry
            # cannot help, so stop instead of burning the remaining iterations.
            if not np.isfinite(cost):
                break

            # Accept the step before testing convergence.  The step that meets
            # the tolerance is still an improvement, so breaking ahead of the
            # assignment would discard it and return the previous iterate.
            converged = abs(cost - last_cost) < 1e-6
            q = q_new
            last_cost = cost
            if converged:
                break

        return q

    def _warm_init_bone_direction(
        self,
        q_init: np.ndarray,
        bone_targets: np.ndarray,
        q_last: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Warm-init SQP stage from TopoRetarget Eq. (2).

        Refines the initial guess toward the source's relative bone directions:
            min  lambda_warm * E_bone(q) + lambda_smooth * ||q - q_ref||^2
        where q_ref is the previous frame's configuration (or q_init for the
        first frame). Runs a small QP for ``bone_warm_init_iters`` iterations
        and returns the refined configuration (input is not modified).
        """
        q = q_init.copy()
        q_ref = q_last if q_last is not None else q_init
        ref_a = q_ref[self.q_a_indices]

        n = self.nq_a
        for _ in range(self.bone_warm_init_iters):
            self.robot_data.qpos[:] = q
            mujoco.mj_forward(self.robot_model, self.robot_data)
            J_V, p_V, _ = self._compute_robot_jacobians(q)
            robot_points = np.array([p_V[name] for name in self.source_target_names])
            res, J = compute_bone_direction_residual_and_jacobian(
                robot_points, J_V, self.bone_triples, bone_targets
            )

            q_a_current = q[self.q_a_indices]
            # min lambda_warm * ||res + J dqa||^2 + lambda_smooth * ||dqa + q_a - ref_a||^2
            P = 2.0 * self.lambda_warm * (J.T @ J) + 2.0 * self.lambda_smooth * np.eye(n)
            c = 2.0 * self.lambda_warm * (J.T @ res) + 2.0 * self.lambda_smooth * (q_a_current - ref_a)

            dqa_opt, ok = _solve_qp_clarabel(
                sp.csr_matrix(P),
                c,
                lb=self.q_a_lb - q_a_current,
                ub=self.q_a_ub - q_a_current,
                trust_region_dim=n,
                step_size=self.step_size,
            )
            if not ok:
                break

            q[self.q_a_indices] = q_a_current + dqa_opt
            # Keep the base quaternion a valid unit quaternion between iterations
            quat = q[3:7]
            norm = np.linalg.norm(quat)
            if norm > 1e-12:
                q[3:7] = quat / norm

        return q

    def _single_optimization_step(
        self,
        q: np.ndarray,
        target_laplacian: np.ndarray,
        L: sp.spmatrix,
        Kron: sp.spmatrix,
        terrain_points: np.ndarray,
        q_last: Optional[np.ndarray] = None,
        target_base_orientation: Optional[np.ndarray] = None,
        object_points: Optional[np.ndarray] = None,
        bone_targets: Optional[np.ndarray] = None,
        root_translation: Optional[np.ndarray] = None,
        foot_orientation_targets: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Single SQP optimization step.

        The QP is assembled directly and solved with CLARABEL (see
        _solve_qp_clarabel): the auxiliary Laplacian variables are substituted
        out, leaving a small QP in dqa (plus optional penetration slack).

        Args:
            q: Current configuration
            target_laplacian: Target Laplacian coordinates
            L: Per-frame Laplacian matrix (constant across SQP iterations)
            Kron: Per-frame kron(L, eye(3)) (constant across SQP iterations)
            terrain_points: Terrain contact points
            q_last: Configuration at previous time step (for smoothness)
            bone_targets: Optional source-side relative bone directions for the
                bone-direction prior (from compute_bone_direction_targets)
            root_translation: Optional source root translation (3,) for base
                position tracking.

        Returns:
            Tuple of (optimized_config, cost)
        """
        # Update robot state
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)

        # qdot->qvel transform depends only on q; build once per iteration.
        T_qvel = self._build_transform_qdot_to_qvel_fast()

        # Compute Jacobians for mapped robot link points.
        J_V, p_V, _ = self._compute_robot_jacobians(q, T=T_qvel)

        # CRITICAL: Ensure robot_points are in the SAME ORDER as source_target_positions passed to retarget_frame.
        # The order MUST match: source_target_positions[i] corresponds to robot_points[i] for all i.
        source_target_names_ordered = self.source_target_names

        # CRITICAL: Verify that p_V has all source targets in the correct order.
        if set(source_target_names_ordered) != set(p_V.keys()):
            missing = set(source_target_names_ordered) - set(p_V.keys())
            extra = set(p_V.keys()) - set(source_target_names_ordered)
            raise RuntimeError(
                f"Source target mismatch: p_V has different targets than joint_mapping. "
                f"Missing from p_V: {missing}, Extra in p_V: {extra}"
            )

        robot_points = np.array([p_V[target_name] for target_name in source_target_names_ordered])

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

        num_robot_points = len(robot_points)
        num_env_points = len(all_env_points)

        # Construct full Jacobian for all vertices
        # Top part: J_V (robot points), Bottom part: 0 (environment points, static)
        J_full_vertices = sp.vstack([
            sp.csr_matrix(J_V),  # Jacobians for robot points
            sp.csr_matrix((3 * num_env_points, self.nq_a))  # Zeros for environment points (terrain + objects)
        ])

        # J_L maps dqa to Laplacian-coordinate deltas: (3*V, nq_a)
        J_L = Kron @ J_full_vertices

        # ---- Assemble the QP in dqa (+ optional penetration slack vars) ----
        # Objective terms (matching the previous CVXPY formulation exactly):
        #   ||sqrt(W) (lap0 + J_L dqa - target)||^2          (Laplacian)
        #   + lambda_bone ||res + J_bone dqa||^2             (bone prior)
        #   + ||sqrt(Q_diag) (dqa + q_a)||^2                 (joint regularization)
        #   + smooth_weight ||dqa - dqa_smooth||^2           (temporal smoothness)
        #   + sum_i w_orient (dqa[i] + q[i] - target[i])^2   (base orientation)
        #   + (w_slack/2) sum((span * s_unit)^2)             (penetration slack)
        n = self.nq_a
        q_a_current = q[self.q_a_indices]

        lap0_vec = (L @ vertices).reshape(-1)
        target_lap_vec = target_laplacian.reshape(-1)
        r0 = lap0_vec - target_lap_vec

        weights = np.ones(len(vertices)) * 10  # Laplacian weights (matching original laplacian_weights = 10)
        w3 = np.repeat(weights, 3)
        Jw = J_L.multiply(w3[:, None])  # = W @ J_L (W diagonal)

        P = 2.0 * (Jw.T @ J_L)
        c = 2.0 * (Jw.T @ r0)

        # Bone-direction prior (TopoRetarget Eq. 1, refinement-stage weight):
        # keep relative directions of adjacent bones close to the source's.
        bone_res = None
        bone_J = None
        if self.bone_direction_enabled and bone_targets is not None:
            bone_res, bone_J = compute_bone_direction_residual_and_jacobian(
                robot_points, J_V, self.bone_triples, bone_targets
            )
            P = P + 2.0 * self.lambda_bone * sp.csr_matrix(bone_J.T @ bone_J)
            c = c + 2.0 * self.lambda_bone * (bone_J.T @ bone_res)

        # Joint regularization cost (keep joints near zero/neutral).
        # Q_diag_modified is precomputed once in _setup_robot_config.
        # All diagonal Hessian contributions are collected into one vector and
        # added once below (scalar += on sparse matrices is not portable).
        P_diag_extra = 2.0 * self.Q_diag_modified
        c = c + 2.0 * self.Q_diag_modified * q_a_current

        # Smoothness cost: use previous frame's velocity, not current guess
        dqa_smooth = None
        if q_last is not None:
            dqa_smooth = q_last[self.q_a_indices] - q_a_current
            P_diag_extra = P_diag_extra + 2.0 * self.smooth_weight
            c = c - 2.0 * self.smooth_weight * dqa_smooth

        # Base orientation tracking cost: keep the base orientation close to
        # the target estimated from source target positions.
        orientation_weight = 5.0  # Strong preference to maintain orientation
        orient_terms: List[Tuple[int, float]] = []
        base_pos_terms: List[Tuple[int, float]] = []
        if target_base_orientation is not None and 3 in self.q_a_indices:
            # Find quaternion indices in q_a_indices
            quat_indices_in_qa = []
            for quat_idx in [3, 4, 5, 6]:  # wxyz quaternion
                if quat_idx in self.q_a_indices:
                    idx_in_qa = np.where(self.q_a_indices == quat_idx)[0]
                    if len(idx_in_qa) > 0:
                        quat_indices_in_qa.append(idx_in_qa[0])

            if len(quat_indices_in_qa) == 4:
                quat_current = q[3:7]  # Current quaternion
                for i, qa_idx in enumerate(quat_indices_in_qa):
                    diff = float(quat_current[i] - target_base_orientation[i])
                    P_diag_extra[qa_idx] += 2.0 * orientation_weight
                    c[qa_idx] += 2.0 * orientation_weight * diff
                    orient_terms.append((qa_idx, diff))

        # Base position tracking cost: keep the floating-base x-y origin close
        # to the source root translation when it is available. This compensates
        # for the loss of global x-y anchoring that happens with distance-
        # weighted Laplacian edges (TopoRetarget), where static terrain
        # vertices receive very low weights. We intentionally skip Z so the
        # optimizer can still respect the robot's leg kinematics and terrain
        # penetration constraints.
        if root_translation is not None and self.base_position_tracking_weight > 0.0:
            w_pos = self.base_position_tracking_weight
            for pos_idx in [0, 1]:
                if pos_idx in self.q_a_indices:
                    idx_in_qa = int(np.where(self.q_a_indices == pos_idx)[0][0])
                    diff = float(q_a_current[idx_in_qa] - root_translation[pos_idx])
                    # Add 2*w to the diagonal and 2*w*diff to the linear term
                    # so the QP cost is w * (dqa[idx] + diff)^2.
                    P_diag_extra[idx_in_qa] += 2.0 * w_pos
                    c[idx_in_qa] += 2.0 * w_pos * diff
                    base_pos_terms.append((idx_in_qa, diff))

        # Foot sole orientation, root-relative.  Unlike the base orientation
        # term this is not diagonal in the actuated coordinates: rotating a
        # sole couples the whole leg chain, so it contributes a dense rank-3
        # block rather than per-index entries.
        foot_terms: List[Tuple[np.ndarray, np.ndarray]] = []
        if self.foot_orientation_enabled and foot_orientation_targets:
            w_foot = self.foot_orientation_weight
            for jac, error in self._foot_orientation_residuals(foot_orientation_targets, T=T_qvel):
                P = P + sp.csr_matrix(2.0 * w_foot * (jac.T @ jac))
                c = c - 2.0 * w_foot * (jac.T @ error)
                foot_terms.append((jac, error))

        P = P + sp.diags(P_diag_extra)

        # Non-penetration rows (self-collision + terrain), numeric form:
        # hard_rows: (Ja, rhs) meaning Ja @ dqa >= rhs
        # slack_rows: (Ja, rhs_soft, span) adding span * s_unit_i to the row
        hard_rows: List[Tuple[np.ndarray, float]] = []
        slack_rows: List[Tuple[np.ndarray, float, float]] = []
        if self.hard_penetration_constraint:
            hard_rows, slack_rows = self._compute_penetration_constraints(T=T_qvel)

        # Extend variables with unit slacks s_unit in [0, 1] (slack s = span * s_unit)
        m = len(slack_rows)
        if m:
            slack_diag = np.array([
                self.penetration_slack_penalty * span ** 2 for (_, _, span) in slack_rows
            ])
            P = sp.block_diag([P, sp.diags(slack_diag)], format="lil")
            c = np.concatenate([c, np.zeros(m)])

        num_vars = n + m
        ge_rows: List[Tuple[np.ndarray, float]] = []
        for Ja, rhs in hard_rows:
            a = np.zeros(num_vars)
            a[:n] = Ja
            ge_rows.append((a, rhs))
        for i, (Ja, rhs_soft, span) in enumerate(slack_rows):
            a = np.zeros(num_vars)
            a[:n] = Ja
            a[n + i] = span
            ge_rows.append((a, rhs_soft))

        lb = np.concatenate([self.q_a_lb - q_a_current, np.zeros(m)])
        ub = np.concatenate([self.q_a_ub - q_a_current, np.ones(m)])

        # Solve (fallback: retry without the trust region, as before)
        P_csr = sp.csr_matrix(P)
        x, ok = _solve_qp_clarabel(
            P_csr, c, lb, ub, ge_rows, trust_region_dim=n, step_size=self.step_size
        )
        if not ok:
            x, ok = _solve_qp_clarabel(P_csr, c, lb, ub, ge_rows)
        if not ok:
            return q, np.inf

        dqa_opt = x[:n]

        # Objective value in the original residual form (same value CVXPY's
        # problem.value reported), used for the SQP convergence check.
        r = r0 + J_L @ dqa_opt
        cost = float(w3 @ (r ** 2))
        if bone_res is not None:
            rb = bone_res + bone_J @ dqa_opt
            cost += self.lambda_bone * float(rb @ rb)
        cost += float(self.Q_diag_modified @ ((dqa_opt + q_a_current) ** 2))
        if dqa_smooth is not None:
            cost += self.smooth_weight * float(((dqa_opt - dqa_smooth) ** 2).sum())
        for qa_idx, diff in orient_terms:
            cost += orientation_weight * (dqa_opt[qa_idx] + diff) ** 2
        for qa_idx, diff in base_pos_terms:
            cost += self.base_position_tracking_weight * (dqa_opt[qa_idx] + diff) ** 2
        for jac, error in foot_terms:
            residual = jac @ dqa_opt - error
            cost += self.foot_orientation_weight * float(residual @ residual)
        if m:
            s_true = np.array([span for (_, _, span) in slack_rows]) * x[n:]
            cost += (self.penetration_slack_penalty / 2.0) * float((s_true ** 2).sum())

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

    def _calc_contact_jacobian_from_point(self, body_idx: int, p_body: np.ndarray = None, input_world=False, T: Optional[np.ndarray] = None):
        """
        Translational Jacobian J(q) (3 x nq) such that
        v_point_world = J(q) @ qdot.

        Fast analytic version: J_qdot = J_v @ T(q)

        Args:
            T: Optional precomputed qdot->qvel transform. Built here if not
                provided; callers computing many Jacobians at one configuration
                should build it once and pass it in.
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

        if T is None:
            T = self._build_transform_qdot_to_qvel_fast()

        return Jp @ T

    def _compute_robot_jacobians(self, q: np.ndarray, T: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray], None]:
        """Compute Jacobians for mapped robot link points in world frame.

        Args:
            q: Robot configuration
            T: Optional precomputed qdot->qvel transform (built here if None)

        Returns:
            Tuple of (J_V, p_dict, None):
                - J_V: Stacked Jacobians (3*num_targets, nq_a)
                - p_dict: Dictionary of robot link point positions keyed by source target name
                - None: Placeholder for compatibility
        """
        J_dict = {}
        p_dict = {}

        if T is None:
            T = self._build_transform_qdot_to_qvel_fast()

        for target_name, link_name in self.joint_mapping.items():
            try:
                body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link_name)

                # Get position in world frame
                pos = self.robot_data.xpos[body_id].copy()

                # Compute base Jacobian for body origin (3 x nq)
                J_base = self._calc_contact_jacobian_from_point(body_id, T=T)

                # Apply offset from merged target_mapping (if non-zero)
                offset = self.target_offset_map.get(target_name)
                if offset is not None and np.any(offset != 0):
                    o_local = offset
                    R_WB = self.robot_data.xmat[body_id].reshape(3, 3)
                    o_world = R_WB @ o_local
                    pos = pos + o_world  # p_target = p_body + R @ o_local

                    # Rotational Jacobian Jr (3 x nq) for the cross-term correction
                    p_WB = self.robot_data.xpos[body_id]
                    p_W = p_WB.astype(np.float64).reshape(3, 1)
                    Jp = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
                    Jr = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
                    mujoco.mj_jac(self.robot_model, self.robot_data, Jp, Jr, p_W, int(body_id))
                    Jr_world = Jr @ T  # rotational Jacobian in world frame (3 x nq)
                    # Cross-term: -skew(o_world) @ Jr_world
                    # Derivation: d/dt(p_body + R @ o_local)
                    #   = v_body + ω × o_world
                    #   = v_body - o_world × ω
                    #   = v_body - skew(o_world) @ ω
                    # J_full = J_base - skew(o_world) @ Jr_world
                    J_full = J_base - self._skew(o_world) @ Jr_world
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
        try:
            mujoco.mj_collision(m, d)
        except mujoco.FatalError:
            # Box-box or other geom pairs can exceed mjMAXCONPAIR (8 contacts).
            # Fall back to brute-force pairwise distance checks.
            m.geom_margin[:] = self._saved_margins
            return self._brute_force_candidate_pairs(threshold)

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

    def _brute_force_candidate_pairs(self, threshold: float) -> set:
        """Fallback: enumerate all geom pairs via mj_geomDistance."""
        m, d = self.robot_model, self.robot_data
        ngeom = m.ngeom
        fromto = np.zeros(6, dtype=float)
        candidates = set()
        for i in range(ngeom):
            if m.geom_contype[i] == 0 and m.geom_conaffinity[i] == 0:
                continue
            for j in range(i + 1, ngeom):
                if m.geom_contype[j] == 0 and m.geom_conaffinity[j] == 0:
                    continue
                if m.geom_bodyid[i] == m.geom_bodyid[j]:
                    continue
                dist = mujoco.mj_geomDistance(m, d, i, j, threshold, fromto)
                if dist <= threshold:
                    candidates.add((i, j))
        return candidates

    def _compute_jacobian_for_contact_relative(
        self,
        geom1_id: int,
        geom2_id: int,
        geom1_name: str,
        geom2_name: str,
        fromto: np.ndarray,
        dist: float,
        T: Optional[np.ndarray] = None,
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
        J_bodyA = self._calc_contact_jacobian_from_point(body1_id, pos1, input_world=True, T=T)
        J_bodyB = self._calc_contact_jacobian_from_point(body2_id, pos2, input_world=True, T=T)

        # Compute relative Jacobian
        Jc = J_bodyA - J_bodyB

        # Project onto contact normal
        return nhat_BA_W @ Jc

    def _penetration_constraint_terms(
        self,
        Ja: np.ndarray,
        dist: float,
    ) -> Tuple[List[Tuple[np.ndarray, float]], Optional[Tuple[np.ndarray, float, float]]]:
        """Linearized non-penetration row(s) for one queried contact pair.

        Returns ``(hard_rows, slack_row)`` where every row is numeric:
        - ``hard_rows``: list of ``(Ja, rhs)`` meaning ``Ja @ dqa >= rhs``.
          Default (slack disabled): single hard row ``phi >= -penetration_tolerance``.
        - ``slack_row``: TopoRetarget slack mode (Eq. 8) soft row
          ``(Ja, rhs_soft, span)`` meaning ``Ja @ dqa + span * s_unit >= rhs_soft``
          with ``0 <= s_unit <= 1``; the caller adds the ``w_s/2 * s^2`` objective
          term. The hard backstop ``phi >= -b`` is included in ``hard_rows``.

        The slack is parameterized as ``s = span * s_unit`` with
        ``span = b - tau`` and ``s_unit`` in [0, 1] — mathematically identical
        to optimizing ``s`` in meters, but avoids the ill-conditioned
        5e4-scale quadratic that stalls CLARABEL when ``s`` is optimized
        directly.
        """
        if not self.penetration_slack_enabled:
            return [(Ja, -dist - self.penetration_tolerance)], None

        span = self.penetration_hard_bound - self.penetration_soft_tolerance
        hard_row = (Ja, -dist - self.penetration_hard_bound)
        slack_row = (Ja, -dist - self.penetration_soft_tolerance, span)
        return [hard_row], slack_row

    def _compute_penetration_constraints(
        self, T: Optional[np.ndarray] = None
    ) -> Tuple[List[Tuple[np.ndarray, float]], List[Tuple[np.ndarray, float, float]]]:
        """
        Compute penetration constraint rows for robot-robot and robot-terrain contacts.

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

        Precondition: kinematics must already be current (the caller runs
        mj_forward with the current configuration first).

        Returns:
            (hard_rows, slack_rows): numeric rows; see _penetration_constraint_terms.
        """
        hard_rows: List[Tuple[np.ndarray, float]] = []
        slack_rows: List[Tuple[np.ndarray, float, float]] = []

        if T is None:
            T = self._build_transform_qdot_to_qvel_fast()

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
                    g1, g2, self._geom_names[g1], self._geom_names[g2], fromto, dist, T=T
                )
                Ja = J_rel[self.q_a_indices]
                hard, slack = self._penetration_constraint_terms(Ja, dist)
                hard_rows.extend(hard)
                if slack is not None:
                    slack_rows.append(slack)

        # ------------------------------------------------------------------
        # 2) Robot-terrain penetration constraints (via trimesh proximity)
        # ------------------------------------------------------------------
        terrain_hard, terrain_slack = self._compute_terrain_penetration_constraints(
            threshold, T=T
        )
        hard_rows.extend(terrain_hard)
        slack_rows.extend(terrain_slack)

        return hard_rows, slack_rows

    def _compute_terrain_penetration_constraints(
        self, threshold: float, T: Optional[np.ndarray] = None
    ) -> Tuple[List[Tuple[np.ndarray, float]], List[Tuple[np.ndarray, float, float]]]:
        """
        Compute non-penetration rows between robot geoms and the terrain trimesh.

        Samples points on the actual surface of each collision geom based on
        its shape, then checks each point for penetration with the terrain.
        This avoids the limitation of only checking the geom center which can
        miss penetration when the geom has large extent.

        **Prefilter**: a geom is only sampled if its center is within
        ``threshold + bounding_radius`` of the terrain surface. Every surface
        point of a skipped geom is provably farther than ``threshold`` from the
        terrain, so the per-point distance check below would discard it anyway
        — the emitted rows are identical, but the expensive closest-point query
        runs on far fewer points.

        **Trade-off**: Only primitive geom types (sphere, box, capsule,
        cylinder) are fully supported with surface sampling. Other geom types
        (mesh, heightfield, etc.) fall back to only checking the center point.

        For each sampled point that is close to or inside the terrain, we add
        the linear constraint:

            n^T J_a  dqa  >=  -(d - tol)

        where
        - d   is the signed distance (positive = above terrain),
        - n   is the outward terrain surface normal at the closest point,
        - J_a is the translational Jacobian of the geom's body at the sampled
          point (columns for the actuated DOFs only),
        - tol is ``self.penetration_tolerance`` (or the TopoRetarget soft/hard
          slack bounds when ``penetration_slack`` is enabled).

        Precondition: kinematics must already be current (mj_forward done by
        the caller).

        Returns:
            (hard_rows, slack_rows): numeric rows; see _penetration_constraint_terms.
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

        hard_rows: List[Tuple[np.ndarray, float]] = []
        slack_rows: List[Tuple[np.ndarray, float, float]] = []
        m, d = self.robot_model, self.robot_data

        if T is None:
            T = self._build_transform_qdot_to_qvel_fast()

        # Collision-enabled geoms (skip purely visual geoms)
        coll_geoms = [
            gi for gi in range(m.ngeom)
            if not (m.geom_contype[gi] == 0 and m.geom_conaffinity[gi] == 0)
        ]
        if not coll_geoms:
            return hard_rows, slack_rows

        # Prefilter geoms by bounding sphere: a geom can only contribute a
        # constraint if some surface point is within `threshold` of the terrain,
        # i.e. if center_dist - bounding_radius <= threshold.
        centers = np.array([d.geom_xpos[gi] for gi in coll_geoms])
        _, center_dists, _ = _trimesh.proximity.closest_point(self.terrain_mesh, centers)
        kept_geoms = [
            gi
            for gi, center_dist in zip(coll_geoms, center_dists)
            if center_dist - self._geom_bounding_radii[gi] <= threshold
        ]

        # Collect world-frame sample points on the surface of every kept geom
        all_points = []
        all_geom_info = []
        for gi in kept_geoms:
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
                all_geom_info.append(gi)

        if len(all_points) == 0:
            return hard_rows, slack_rows

        all_points = np.array(all_points)  # (N, 3)

        # Query terrain mesh for closest points to each sampled point
        closest_pts, dists, tri_ids = _trimesh.proximity.closest_point(
            self.terrain_mesh, all_points
        )

        for k, gi in enumerate(all_geom_info):
            if dists[k] > threshold:
                continue

            # Signed distance: positive when above terrain.
            # closest_pts[k] is on the terrain surface; query_pt is the
            # point on the geom surface. We define "above" as the direction
            # of the terrain face normal.
            query_pt = all_points[k]
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
                body_id, query_pt, input_world=True, T=T
            )
            # Project onto terrain normal -> 1-D Jacobian
            J_n = face_normal @ J_full  # (nq,)
            Ja = J_n[self.q_a_indices]

            hard, slack = self._penetration_constraint_terms(Ja, signed_dist)
            hard_rows.extend(hard)
            if slack is not None:
                slack_rows.append(slack)

        return hard_rows, slack_rows


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
