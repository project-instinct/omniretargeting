"""Core retargeting functionality adapted for generic robots and terrains."""

from __future__ import annotations

import numpy as np
import mujoco
import clarabel
from scipy import sparse as sp
from scipy.spatial import Delaunay
import trimesh
import open3d as o3d
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import fnmatch
import time

from .utils import (
    sample_points_on_mesh,
    compute_mesh_height_at_point,
    transform_points_local_to_world,
    get_adjacency_list,
    calculate_exponential_edge_weights,
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
    sample_mujoco_geom_local_points,
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
    """Robot-side bone-direction residual and its Jacobian w.r.t. tangent DOFs.

    For bone (a->b) with segment e = p_b - p_a, L = ||e||, d = e / L:
        dd/dq = (I - d d^T) / L @ (J_b - J_a)
    For an adjacent pair (k, l) the residual is (d_k - d_l) - target and its
    Jacobian is Jd_k - Jd_l, so ``res(q ⊕ delta_v) ~= res + J @ delta_v``.

    Args:
        robot_points: (N, 3) current robot link point positions (mapped order).
        J_V: (3N, nv_a) stacked translational Jacobians (same order).
        triples: Adjacent bone triples (a, b, c).
        targets: (3P,) source-side (d_k - d_l) from compute_bone_direction_targets.
        eps: Lower bound for segment lengths when normalizing.

    Returns:
        (res, J): residual (3P,) and Jacobian (3P, nv_a).
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
    ``lap_var = lap0 + J_L @ delta_v`` is substituted into the objective, so the
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
        source_target_names: Optional[List[str]] = None,
        replace_cylinders_with_capsules: bool = False,
        hard_penetration_constraint: bool = False,
        joint_regularization_boost: Optional[Dict] = None,
        laplacian_edge_weighting: str = "uniform",
        laplacian_distance_decay: float = 30.0,
        bone_direction: Optional[Dict] = None,
        penetration_slack: Optional[Dict] = None,
        base_position_tracking_weight: float = 0.0,
        base_position_tracking_weight_z: float = 0.0,
        penetration_correction: Optional[Dict] = None,
        solver_diagnostics: bool = False,
        terrain_deep_penetration_depth: float = 0.5,
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
            penetration_correction: Optional physical tangent-space state metric
                and per-block step limits used by the SQP.
            solver_diagnostics: Retain detailed per-contact and per-DOF-block
                diagnostics on ``last_solve_diagnostics``. Summary success and
                feasibility fields are always retained.
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
        if penetration_slack is not None and not isinstance(penetration_slack, dict):
            raise ValueError("penetration_slack must be a dictionary when provided.")
        ps = penetration_slack or {}
        self.penetration_slack_enabled = penetration_slack is not None
        self.penetration_soft_tolerance = float(ps.get("soft_tolerance", 1e-3))
        self.penetration_hard_bound = float(ps.get("hard_bound", 0.1))
        self.penetration_slack_penalty = float(ps.get("slack_penalty", 1e5))
        self.base_position_tracking_weight = float(base_position_tracking_weight)
        self.base_position_tracking_weight_z = float(base_position_tracking_weight_z)
        self.solver_diagnostics = bool(solver_diagnostics)
        if self.penetration_slack_enabled:
            slack_values = np.array(
                [
                    self.penetration_soft_tolerance,
                    self.penetration_hard_bound,
                    self.penetration_slack_penalty,
                ]
            )
            if not np.isfinite(slack_values).all():
                raise ValueError("penetration_slack values must be finite.")
            if self.penetration_soft_tolerance < 0:
                raise ValueError("penetration_slack.soft_tolerance must be non-negative.")
            if self.penetration_hard_bound <= self.penetration_soft_tolerance:
                raise ValueError(
                    f"penetration_slack.hard_bound ({self.penetration_hard_bound}) must be "
                    f"greater than soft_tolerance ({self.penetration_soft_tolerance})."
                )
            if self.penetration_slack_penalty <= 0:
                raise ValueError("penetration_slack.slack_penalty must be positive.")

        if penetration_correction is not None and not isinstance(penetration_correction, dict):
            raise ValueError("penetration_correction must be a dictionary when provided.")
        correction = penetration_correction or {}
        self.base_translation_weights = np.asarray(
            correction.get("base_translation_weights", [1e-3, 1e-3, 1.0]),
            dtype=float,
        )
        self.base_translation_step = np.asarray(
            correction.get("base_translation_step", [step_size, step_size, min(step_size, 0.05)]),
            dtype=float,
        )
        if self.base_translation_weights.shape != (3,):
            raise ValueError("penetration_correction.base_translation_weights must contain 3 values.")
        if self.base_translation_step.shape != (3,):
            raise ValueError("penetration_correction.base_translation_step must contain 3 values.")
        self.base_rotation_weight = float(correction.get("base_rotation_weight", 5.0))
        self.base_rotation_step = float(correction.get("base_rotation_step", step_size))
        self.joint_weight = float(
            correction.get(
                "joint_weight",
                (joint_regularization_boost or {}).get("default", 1e-3),
            )
        )
        self.joint_range_normalization = bool(
            correction.get("joint_range_normalization", True)
        )
        self.joint_step_fraction = float(correction.get("joint_step_fraction", 0.1))
        self.sqp_step_tolerance = float(correction.get("step_tolerance", 1e-5))
        self.sqp_feasibility_tolerance = float(
            correction.get("feasibility_tolerance", 1e-6)
        )
        self.sqp_max_backtracks = int(correction.get("max_backtracks", 6))
        self.restoration_penalty = float(
            correction.get("restoration_penalty", 1e7)
        )
        correction_values = np.concatenate(
            [
                self.base_translation_weights,
                self.base_translation_step,
                np.array(
                    [
                        self.base_rotation_weight,
                        self.base_rotation_step,
                        self.joint_weight,
                        self.joint_step_fraction,
                        self.sqp_step_tolerance,
                        self.sqp_feasibility_tolerance,
                        self.restoration_penalty,
                    ]
                ),
            ]
        )
        if not np.isfinite(correction_values).all():
            raise ValueError("penetration_correction values must be finite.")
        if np.any(self.base_translation_weights < 0) or self.base_rotation_weight < 0:
            raise ValueError("penetration_correction weights must be non-negative.")
        if self.joint_weight < 0:
            raise ValueError("penetration_correction.joint_weight must be non-negative.")
        if np.any(self.base_translation_step <= 0) or self.base_rotation_step <= 0:
            raise ValueError("penetration_correction base step limits must be positive.")
        if self.joint_step_fraction <= 0:
            raise ValueError("penetration_correction.joint_step_fraction must be positive.")
        if self.sqp_step_tolerance <= 0 or self.sqp_feasibility_tolerance < 0:
            raise ValueError("penetration_correction SQP tolerances are invalid.")
        if self.sqp_max_backtracks < 0:
            raise ValueError("penetration_correction.max_backtracks must be non-negative.")
        if self.restoration_penalty <= 0:
            raise ValueError("penetration_correction.restoration_penalty must be positive.")

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
        """Map the configured optimization slice to physical MuJoCo DOFs."""
        m = self.robot_model
        self.nq = m.nq
        self.nv = m.nv
        self.neutral_qpos = m.qpos0.copy()

        start_qpos = int(np.clip(7 + self.q_a_init_idx, 0, self.nq))
        dof_indices: List[int] = []
        self.root_joint_id: Optional[int] = None
        self.root_qpos_adr: Optional[int] = None
        self.root_dof_adr: Optional[int] = None

        for joint_id in range(m.njnt):
            joint_type = m.jnt_type[joint_id]
            qpos_adr = int(m.jnt_qposadr[joint_id])
            dof_adr = int(m.jnt_dofadr[joint_id])
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                dof_width = 6
                if self.root_joint_id is None:
                    self.root_joint_id = joint_id
                    self.root_qpos_adr = qpos_adr
                    self.root_dof_adr = dof_adr
                if start_qpos <= qpos_adr:
                    dof_indices.extend(range(dof_adr, dof_adr + dof_width))
            elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
                if qpos_adr >= start_qpos:
                    dof_indices.extend(range(dof_adr, dof_adr + 3))
            elif qpos_adr >= start_qpos:
                dof_indices.append(dof_adr)

        self.dof_indices = np.asarray(dof_indices, dtype=int)
        self.nv_a = len(self.dof_indices)
        if self.nv_a == 0:
            raise ValueError("q_a_init_idx selects no physical robot DOFs.")
        self._dof_to_opt = {dof: i for i, dof in enumerate(self.dof_indices)}

        print(f"Robot config: nq={self.nq}, nv={self.nv}, nv_a={self.nv_a}")
        print(f"dof_indices range: {self.dof_indices.min()} to {self.dof_indices.max()}")

        self.base_translation_opt_indices: List[int] = []
        self.base_rotation_opt_indices: List[int] = []
        if self.root_dof_adr is not None and self.root_dof_adr in self._dof_to_opt:
            self.base_translation_opt_indices = [
                self._dof_to_opt[self.root_dof_adr + i] for i in range(3)
            ]
            self.base_rotation_opt_indices = [
                self._dof_to_opt[self.root_dof_adr + 3 + i] for i in range(3)
            ]

        self.joint_regularization_diag = np.zeros(self.nv_a)
        self.step_limits = np.full(self.nv_a, self.step_size, dtype=float)
        self.limited_joint_dofs: List[Tuple[int, int, float, float]] = []

        if self.base_translation_opt_indices:
            self.step_limits[self.base_translation_opt_indices] = self.base_translation_step
            self.step_limits[self.base_rotation_opt_indices] = self.base_rotation_step

        boost_joints = (self.joint_regularization_boost or {}).get("joints") or {}
        self.dof_group_indices: Dict[str, List[int]] = {
            "base_translation": list(self.base_translation_opt_indices),
            "base_rotation": list(self.base_rotation_opt_indices),
            "legs": [],
            "waist": [],
            "arms": [],
            "other_joints": [],
        }
        for joint_id in range(m.njnt):
            joint_type = m.jnt_type[joint_id]
            if joint_type == mujoco.mjtJoint.mjJNT_FREE:
                continue
            dof_adr = int(m.jnt_dofadr[joint_id])
            dof_width = 3 if joint_type == mujoco.mjtJoint.mjJNT_BALL else 1
            opt_indices = [
                self._dof_to_opt[dof_adr + offset]
                for offset in range(dof_width)
                if dof_adr + offset in self._dof_to_opt
            ]
            if not opt_indices:
                continue
            qpos_adr = int(m.jnt_qposadr[joint_id])
            limited = bool(m.jnt_limited[joint_id])
            range_min, range_max = map(float, m.jnt_range[joint_id])
            range_width = (
                range_max - range_min
                if limited and joint_type != mujoco.mjtJoint.mjJNT_BALL
                else 0.0
            )

            joint_weight = self.joint_weight
            joint_name = m.joint(joint_id).name or ""
            for pattern, boost_value in boost_joints.items():
                if fnmatch.fnmatch(joint_name.lower(), pattern.lower()):
                    joint_weight = max(joint_weight, float(boost_value))
                    break
            if self.joint_range_normalization and range_width > 0:
                joint_weight /= range_width**2
            self.joint_regularization_diag[opt_indices] = joint_weight

            joint_name_lower = joint_name.lower()
            if any(token in joint_name_lower for token in ("hip", "knee", "ankle", "leg", "toe")):
                group = "legs"
            elif any(token in joint_name_lower for token in ("waist", "torso", "spine")):
                group = "waist"
            elif any(token in joint_name_lower for token in ("shoulder", "elbow", "wrist", "arm")):
                group = "arms"
            else:
                group = "other_joints"
            self.dof_group_indices[group].extend(opt_indices)

            if range_width > 0:
                opt_idx = opt_indices[0]
                self.step_limits[opt_idx] = self.joint_step_fraction * range_width
                self.limited_joint_dofs.append(
                    (opt_idx, qpos_adr, range_min, range_max)
                )

        # Store smoothness weight (matching the previous solver).
        self.smooth_weight = 0.2

    def _configuration_residual(self, reference: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Return the optimized tangent displacement from ``reference`` to ``q``."""
        residual = np.zeros(self.nv)
        mujoco.mj_differentiatePos(self.robot_model, residual, 1.0, reference, q)
        return residual[self.dof_indices]

    def _step_bounds(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return physical per-DOF step and joint-position bounds at ``q``."""
        lb = -self.step_limits.copy()
        ub = self.step_limits.copy()
        for opt_idx, qpos_adr, range_min, range_max in self.limited_joint_dofs:
            lb[opt_idx] = max(lb[opt_idx], range_min - q[qpos_adr])
            ub[opt_idx] = min(ub[opt_idx], range_max - q[qpos_adr])
        return lb, ub

    def _integrate_optimized_step(
        self,
        q: np.ndarray,
        delta_v: np.ndarray,
        scale: float = 1.0,
    ) -> np.ndarray:
        """Integrate an optimized tangent step and return a valid ``qpos``."""
        full_delta_v = np.zeros(self.nv)
        full_delta_v[self.dof_indices] = np.asarray(delta_v, dtype=float) * scale
        q_new = q.copy()
        mujoco.mj_integratePos(self.robot_model, q_new, full_delta_v, 1.0)
        return q_new

    def _dof_block_norms(self, values: np.ndarray) -> Dict[str, float]:
        """Return Euclidean norms for generic robot DOF groups."""
        values = np.asarray(values, dtype=float)
        return {
            group: float(np.linalg.norm(values[indices])) if indices else 0.0
            for group, indices in self.dof_group_indices.items()
        }
    
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
        self._terrain_face_normals = np.asarray(
            self.terrain_mesh.face_normals, dtype=float
        ).copy()
        terrain_vertices = np.ascontiguousarray(
            self.terrain_mesh.vertices, dtype=np.float32
        )
        terrain_faces = np.ascontiguousarray(
            self.terrain_mesh.faces, dtype=np.uint32
        )
        self._terrain_scene = o3d.t.geometry.RaycastingScene()
        self._terrain_scene.add_triangles(
            o3d.core.Tensor(terrain_vertices),
            o3d.core.Tensor(terrain_faces),
        )
        self._terrain_collision_geoms: List[Tuple[int, np.ndarray]] = []
        for geom_id in range(self.robot_model.ngeom):
            if (
                self.robot_model.geom_contype[geom_id] == 0
                and self.robot_model.geom_conaffinity[geom_id] == 0
            ):
                continue
            points_local = sample_mujoco_geom_local_points(
                int(self.robot_model.geom_type[geom_id]),
                self.robot_model.geom_size[geom_id],
            )
            if len(points_local):
                self._terrain_collision_geoms.append((geom_id, points_local))

    def _closest_points_on_terrain(
        self,
        points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return closest terrain points through Open3D's CPU acceleration structure."""
        points = np.asarray(points, dtype=float)
        if len(points) == 0:
            return points.copy(), np.empty(0), np.empty(0, dtype=int)

        query_points = np.ascontiguousarray(points, dtype=np.float32)
        result = self._terrain_scene.compute_closest_points(
            o3d.core.Tensor(query_points), nthreads=1
        )
        closest = np.asarray(result["points"].numpy(), dtype=float)
        tri_ids = np.asarray(result["primitive_ids"].numpy(), dtype=int)
        dists = np.linalg.norm(points - closest, axis=1)
        return closest, dists, tri_ids

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
        root_translation: Optional[np.ndarray] = None,
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
        solve_started = time.perf_counter()
        self._last_qp_diagnostics = {}
        q = q_init.copy()
        base_position_reference = q_init.copy()
        if root_translation is not None and self.root_qpos_adr is not None:
            base_position_reference[
                self.root_qpos_adr : self.root_qpos_adr + 3
            ] = root_translation

        # Warm-init stage (TopoRetarget Eq. 2): refine the initial guess toward
        # the source's relative bone directions before the main refinement.
        if self.bone_direction_enabled and bone_targets is not None and self.bone_warm_init:
            q = self._warm_init_bone_direction(q, bone_targets, q_last)

        last_cost = np.inf
        current_violation = self._nonlinear_penetration_violation(q)
        converged = False
        solver_failed = False
        backtrack_failed = False
        accepted_any = False
        iterations = 0
        accepted_step = np.zeros(self.nv_a)
        total_backtracks = 0

        for iteration in range(max_iter):
            iterations = iteration + 1
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
                base_position_reference=base_position_reference,
            )

            # Solver failure at this linearization point: an identical retry
            # cannot help, so stop instead of burning the remaining iterations.
            if not np.isfinite(cost):
                solver_failed = True
                break

            accepted_step = self._configuration_residual(q, q_new)
            candidate_violation = self._nonlinear_penetration_violation(q_new)

            if candidate_violation > current_violation + self.sqp_feasibility_tolerance:
                backtrack_accepted = False
                for backtrack in range(1, self.sqp_max_backtracks + 1):
                    total_backtracks += 1
                    scale = 0.5**backtrack
                    q_backtracked = self._integrate_optimized_step(q, accepted_step, scale)
                    backtracked_violation = self._nonlinear_penetration_violation(q_backtracked)
                    if backtracked_violation <= current_violation + self.sqp_feasibility_tolerance:
                        q_new = q_backtracked
                        accepted_step = accepted_step * scale
                        candidate_violation = backtracked_violation
                        backtrack_accepted = True
                        break
                if not backtrack_accepted:
                    # No positive backtracking scale preserves the nonlinear
                    # hard bound. If no usable step has been accepted yet, the
                    # frame has not moved and must be reported as failed instead
                    # of silently accepted. Otherwise, keep the previously
                    # accepted feasible pose and treat the zero step as
                    # convergence.
                    if not accepted_any:
                        backtrack_failed = True
                        q_new = q
                        accepted_step = np.zeros_like(accepted_step)
                        candidate_violation = current_violation
                        last_cost = cost
                        break
                    q_new = q
                    accepted_step = np.zeros_like(accepted_step)
                    candidate_violation = current_violation

            q = q_new
            current_violation = candidate_violation
            if np.linalg.norm(accepted_step) > 0.0:
                accepted_any = True
            scaled_step_norm = float(
                np.linalg.norm(accepted_step / self.step_limits)
            )
            feasible = current_violation <= self.sqp_feasibility_tolerance
            if feasible and scaled_step_norm <= self.sqp_step_tolerance:
                converged = True
                last_cost = cost
                break
            last_cost = cost

        feasible = current_violation <= self.sqp_feasibility_tolerance
        net_delta = self._configuration_residual(q_init, q)
        if solver_failed:
            failure_reason = "qp_solver_failed"
        elif backtrack_failed:
            failure_reason = "nonlinear_step_rejected"
        elif not feasible:
            failure_reason = "nonlinear_hard_bound_violation"
        else:
            failure_reason = None
        self.last_solve_diagnostics = {
            "success": bool(feasible and not solver_failed and not backtrack_failed),
            "converged": converged,
            "solver_failed": solver_failed,
            "backtrack_failed": backtrack_failed,
            "failure_reason": failure_reason,
            "iterations": iterations,
            "cost": float(last_cost),
            "max_hard_violation": float(current_violation),
            "accepted_delta_v": accepted_step.copy(),
            "net_delta_v": net_delta.copy(),
            "runtime_seconds": float(time.perf_counter() - solve_started),
            **getattr(self, "_last_qp_diagnostics", {}),
        }
        if self.solver_diagnostics:
            self.last_solve_diagnostics.update(
                {
                    "backtrack_attempts": total_backtracks,
                    "correction_block_norms": self._dof_block_norms(net_delta),
                }
            )
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

        n = self.nv_a
        for _ in range(self.bone_warm_init_iters):
            self.robot_data.qpos[:] = q
            mujoco.mj_forward(self.robot_model, self.robot_data)
            J_V, p_V, _ = self._compute_robot_jacobians(q)
            robot_points = np.array([p_V[name] for name in self.source_target_names])
            res, J = compute_bone_direction_residual_and_jacobian(
                robot_points, J_V, self.bone_triples, bone_targets
            )

            reference_residual = self._configuration_residual(q_ref, q)
            # min lambda_warm * ||res + J dv||^2
            #   + lambda_smooth * ||reference_residual + dv||^2
            P = 2.0 * self.lambda_warm * (J.T @ J) + 2.0 * self.lambda_smooth * np.eye(n)
            c = 2.0 * self.lambda_warm * (J.T @ res) + 2.0 * self.lambda_smooth * reference_residual
            lb, ub = self._step_bounds(q)

            delta_v, ok = _solve_qp_clarabel(
                sp.csr_matrix(P),
                c,
                lb=lb,
                ub=ub,
            )
            if not ok:
                break

            q = self._integrate_optimized_step(q, delta_v)

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
        base_position_reference: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Single SQP optimization step.

        The QP is assembled directly and solved with CLARABEL (see
        _solve_qp_clarabel): the auxiliary Laplacian variables are substituted
        out, leaving a small QP in ``delta_v`` (plus optional penetration slack).

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

        # Compute Jacobians for mapped robot link points.
        J_V, p_V, _ = self._compute_robot_jacobians(q)

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
            sp.csr_matrix((3 * num_env_points, self.nv_a))  # Zeros for environment points (terrain + objects)
        ])

        # J_L maps delta_v to Laplacian-coordinate deltas: (3*V, nv_a)
        J_L = Kron @ J_full_vertices

        # ---- Assemble the QP in delta_v (+ optional penetration slacks) ----
        n = self.nv_a

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

        # Joint regularization uses a valid neutral qpos and a tangent residual.
        neutral_residual = self._configuration_residual(self.neutral_qpos, q)
        P_diag_extra = 2.0 * self.joint_regularization_diag
        c = c + 2.0 * self.joint_regularization_diag * neutral_residual

        # Temporal smoothness is also expressed in tangent coordinates.
        smooth_residual = None
        if q_last is not None:
            smooth_residual = self._configuration_residual(q_last, q)
            P_diag_extra = P_diag_extra + 2.0 * self.smooth_weight
            c = c + 2.0 * self.smooth_weight * smooth_residual

        # Base orientation tracking uses the three-dimensional free-joint
        # rotational tangent instead of four ambient quaternion components.
        orientation_residual = None
        if (
            target_base_orientation is not None
            and self.root_qpos_adr is not None
            and self.base_rotation_opt_indices
        ):
            orientation_target = q.copy()
            orientation_target[
                self.root_qpos_adr + 3 : self.root_qpos_adr + 7
            ] = target_base_orientation
            orientation_residual_full = self._configuration_residual(
                orientation_target, q
            )
            orientation_residual = orientation_residual_full[
                self.base_rotation_opt_indices
            ]
            for i, opt_idx in enumerate(self.base_rotation_opt_indices):
                P_diag_extra[opt_idx] += 2.0 * self.base_rotation_weight
                c[opt_idx] += 2.0 * self.base_rotation_weight * orientation_residual[i]

        # Base translation has independent XYZ weights.  The legacy scalar
        # keeps precedence for X/Y when an explicit source root is available.
        position_terms: List[Tuple[int, float, float]] = []
        if (
            base_position_reference is not None
            and self.root_qpos_adr is not None
            and self.base_translation_opt_indices
        ):
            position_weights = self.base_translation_weights.copy()
            if root_translation is not None:
                if self.base_position_tracking_weight > 0.0:
                    position_weights[:2] = self.base_position_tracking_weight
                if self.base_position_tracking_weight_z > 0.0:
                    position_weights[2] = self.base_position_tracking_weight_z
            for axis, opt_idx in enumerate(self.base_translation_opt_indices):
                diff = float(
                    q[self.root_qpos_adr + axis]
                    - base_position_reference[self.root_qpos_adr + axis]
                )
                weight = float(position_weights[axis])
                P_diag_extra[opt_idx] += 2.0 * weight
                c[opt_idx] += 2.0 * weight * diff
                position_terms.append((opt_idx, diff, weight))

        P = P + sp.diags(P_diag_extra)

        # Non-penetration rows (self-collision + terrain), numeric form:
        # hard_rows: (Ja, rhs) meaning Ja @ delta_v >= rhs
        # slack_rows: (Ja, rhs_soft, span) adding span * s_unit_i to the row
        hard_rows: List[Tuple[np.ndarray, float]] = []
        slack_rows: List[Tuple[np.ndarray, float, float]] = []
        if self.hard_penetration_constraint:
            hard_rows, slack_rows = self._compute_penetration_constraints()

        # Extend variables with unit slacks s_unit in [0, 1] (slack s = span * s_unit).
        m = len(slack_rows)
        if m:
            slack_diag = np.array([
                self.penetration_slack_penalty * span ** 2 for (_, _, span) in slack_rows
            ])
            P = sp.block_diag([P, sp.diags(slack_diag)], format="lil")
            c = np.concatenate([c, np.zeros(m)])

        # If the current nonlinear pose lies outside the hard bound by more
        # than a physical step can repair, a literal hard row makes the first
        # restoration QP infeasible.  Add a heavily penalized, bounded elastic
        # variable only to rows that are already violated.  Nonlinear
        # acceptance still requires the true hard bound before success; these
        # variables merely let successive SQP iterations make monotone progress
        # into the feasible set.
        restoration_specs = [
            (row_idx, float(rhs))
            for row_idx, (_, rhs) in enumerate(hard_rows)
            if rhs > self.sqp_feasibility_tolerance
        ]
        restoration_index = {
            row_idx: restore_idx
            for restore_idx, (row_idx, _) in enumerate(restoration_specs)
        }
        restoration_spans = np.array(
            [span for _, span in restoration_specs], dtype=float
        )
        r_count = len(restoration_specs)
        if r_count:
            restoration_diag = self.restoration_penalty * restoration_spans**2
            P = sp.block_diag([P, sp.diags(restoration_diag)], format="lil")
            c = np.concatenate([c, np.zeros(r_count)])

        num_vars = n + m + r_count
        ge_rows: List[Tuple[np.ndarray, float]] = []
        for row_idx, (Ja, rhs) in enumerate(hard_rows):
            a = np.zeros(num_vars)
            a[:n] = Ja
            restore_idx = restoration_index.get(row_idx)
            if restore_idx is not None:
                a[n + m + restore_idx] = restoration_spans[restore_idx]
            ge_rows.append((a, rhs))
        for i, (Ja, rhs_soft, span) in enumerate(slack_rows):
            a = np.zeros(num_vars)
            a[:n] = Ja
            a[n + i] = span
            restore_idx = restoration_index.get(i)
            if restore_idx is not None:
                a[n + m + restore_idx] = restoration_spans[restore_idx]
            ge_rows.append((a, rhs_soft))

        delta_lb, delta_ub = self._step_bounds(q)
        lb = np.concatenate([delta_lb, np.zeros(m + r_count)])
        ub = np.concatenate([delta_ub, np.ones(m + r_count)])

        # Physical per-block limits are already included in lb/ub; there is no
        # mixed-unit all-state trust-region SOC.
        P_csr = sp.csr_matrix(P)
        x, ok = _solve_qp_clarabel(P_csr, c, lb, ub, ge_rows)
        if not ok:
            return q, np.inf

        delta_v = x[:n]

        # Objective value in the original residual form (same value CVXPY's
        # problem.value reported), used for the SQP convergence check.
        r = r0 + J_L @ delta_v
        cost = float(w3 @ (r ** 2))
        if bone_res is not None:
            rb = bone_res + bone_J @ delta_v
            cost += self.lambda_bone * float(rb @ rb)
        cost += float(
            self.joint_regularization_diag
            @ ((delta_v + neutral_residual) ** 2)
        )
        if smooth_residual is not None:
            cost += self.smooth_weight * float(
                ((delta_v + smooth_residual) ** 2).sum()
            )
        if orientation_residual is not None:
            orient_delta = delta_v[self.base_rotation_opt_indices]
            cost += self.base_rotation_weight * float(
                ((orient_delta + orientation_residual) ** 2).sum()
            )
        for opt_idx, diff, weight in position_terms:
            cost += weight * (delta_v[opt_idx] + diff) ** 2
        if m:
            s_true = np.array([span for (_, _, span) in slack_rows]) * x[n : n + m]
            cost += (self.penetration_slack_penalty / 2.0) * float((s_true ** 2).sum())
        else:
            s_true = np.empty(0)
        if r_count:
            r_true = restoration_spans * x[n + m : n + m + r_count]
            cost += (self.restoration_penalty / 2.0) * float((r_true**2).sum())
        else:
            r_true = np.empty(0)

        q_opt = self._integrate_optimized_step(q, delta_v)
        self._last_qp_diagnostics = {
            "active_hard_rows": len(hard_rows),
            "active_slack_rows": len(slack_rows),
            "slack_values": s_true.copy(),
            "restoration_values": r_true.copy(),
            "active_restoration_rows": r_count,
            "proposed_delta_v": delta_v.copy(),
            **getattr(self, "_last_constraint_counts", {}),
        }
        if self.solver_diagnostics:
            self._last_qp_diagnostics["constraint_diagnostics"] = list(
                getattr(self, "_active_constraint_diagnostics", [])
            )

        return q_opt, cost

    def _skew(self, v: np.ndarray) -> np.ndarray:
        """Return 3x3 skew-symmetric matrix of vector v."""
        return np.array([
            [0.0, -v[2],  v[1]],
            [v[2],  0.0, -v[0]],
            [-v[1],  v[0],  0.0],
        ], dtype=float)

    def _calc_contact_jacobian_from_point(
        self,
        body_idx: int,
        p_body: Optional[np.ndarray] = None,
        input_world: bool = False,
    ) -> np.ndarray:
        """Return the native ``3 x nv`` point Jacobian in world coordinates."""
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

        return Jp

    def _compute_robot_jacobians(
        self, q: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], None]:
        """Compute Jacobians for mapped robot link points in world frame.

        Args:
            q: Robot configuration
        Returns:
            Tuple of (J_V, p_dict, None):
                - J_V: Stacked Jacobians (3*num_targets, nv_a)
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

                # Compute base Jacobian for body origin (3 x nv)
                J_base = self._calc_contact_jacobian_from_point(body_id)

                # Apply offset from merged target_mapping (if non-zero)
                offset = self.target_offset_map.get(target_name)
                if offset is not None and np.any(offset != 0):
                    o_local = offset
                    R_WB = self.robot_data.xmat[body_id].reshape(3, 3)
                    o_world = R_WB @ o_local
                    pos = pos + o_world  # p_target = p_body + R @ o_local

                    # Rotational Jacobian Jr (3 x nv) for the cross-term correction
                    p_WB = self.robot_data.xpos[body_id]
                    p_W = p_WB.astype(np.float64).reshape(3, 1)
                    Jp = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
                    Jr = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
                    mujoco.mj_jac(self.robot_model, self.robot_data, Jp, Jr, p_W, int(body_id))
                    # Cross-term: -skew(o_world) @ Jr_world
                    # Derivation: d/dt(p_body + R @ o_local)
                    #   = v_body + ω × o_world
                    #   = v_body - o_world × ω
                    #   = v_body - skew(o_world) @ ω
                    # J_full = J_base - skew(o_world) @ Jr_world
                    J_full = J_base - self._skew(o_world) @ Jr
                else:
                    J_full = J_base

                J_reduced = J_full[:, self.dof_indices]
                    
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
            J_V = np.zeros((3 * num_targets, self.nv_a))
            for i, target_name in enumerate(source_target_names_ordered):
                if target_name in J_dict:
                    J = J_dict[target_name]
                    if J.shape != (3, self.nv_a):
                        raise RuntimeError(
                            f"Jacobian for '{target_name}' has shape {J.shape}, "
                            f"expected (3, {self.nv_a})."
                        )
                    J_V[3 * i:3 * (i + 1), :] = J
                else:
                    # CRITICAL: All targets should exist (validated in __init__), so this is unexpected.
                    raise RuntimeError(
                        f"Jacobian for source target '{target_name}' not found in J_dict. "
                        f"This should not happen if joint_mapping was validated. "
                        f"Available targets in J_dict: {list(J_dict.keys())}"
                    )
        else:
            J_V = np.zeros((0, self.nv_a))

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
            Contact Jacobian (1D array of length nv)
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

    def _penetration_constraint_terms(
        self,
        Ja: np.ndarray,
        dist: float,
    ) -> Tuple[List[Tuple[np.ndarray, float]], Optional[Tuple[np.ndarray, float, float]]]:
        """Linearized non-penetration row(s) for one queried contact pair.

        Returns ``(hard_rows, slack_row)`` where every row is numeric:
        - ``hard_rows``: list of ``(Ja, rhs)`` meaning ``Ja @ delta_v >= rhs``.
          Default (slack disabled): single hard row ``phi >= -penetration_tolerance``.
        - ``slack_row``: TopoRetarget slack mode (Eq. 8) soft row
          ``(Ja, rhs_soft, span)`` means ``Ja @ delta_v + span * s_unit >= rhs_soft``
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

    def _record_penetration_diagnostic(
        self,
        source: str,
        jacobian: np.ndarray,
        signed_distance: float,
        hard_rows: List[Tuple[np.ndarray, float]],
        slack_row: Optional[Tuple[np.ndarray, float, float]],
    ) -> None:
        """Record one active contact row when detailed diagnostics are enabled."""
        if not self.solver_diagnostics:
            return
        if not hasattr(self, "_active_constraint_diagnostics"):
            self._active_constraint_diagnostics = []
        self._active_constraint_diagnostics.append(
            {
                "source": source,
                "signed_distance": float(signed_distance),
                "hard_rhs": float(hard_rows[0][1]),
                "soft_rhs": None if slack_row is None else float(slack_row[1]),
                "jacobian_block_norms": self._dof_block_norms(jacobian),
            }
        )

    def _compute_penetration_constraints(
        self,
    ) -> Tuple[List[Tuple[np.ndarray, float]], List[Tuple[np.ndarray, float, float]]]:
        """
        Compute penetration constraint rows for robot-robot and robot-terrain contacts.

        Two sources of constraints are combined:
        1. **Self-collision** – MuJoCo's built-in collision detection finds pairs of
           robot geoms that are close to each other and builds linearised
           non-penetration constraints via contact Jacobians.
        2. **Terrain penetration** – representative geom-surface points are
           queried with ``trimesh.proximity.closest_point``. Nearby surfaces are
           constrained directly; deep negative distances are also retained for
           upward-facing support surfaces so an initially buried geom cannot be
           hidden by an unsigned-distance prefilter.

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
        self._active_constraint_diagnostics = []

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
                Ja = J_rel[self.dof_indices]
                hard, slack = self._penetration_constraint_terms(Ja, dist)
                self._record_penetration_diagnostic(
                    "self_collision", Ja, dist, hard, slack
                )
                hard_rows.extend(hard)
                if slack is not None:
                    slack_rows.append(slack)

        # ------------------------------------------------------------------
        # 2) Robot-terrain penetration constraints (via trimesh proximity)
        # ------------------------------------------------------------------
        self_hard_count = len(hard_rows)
        self_slack_count = len(slack_rows)
        terrain_hard, terrain_slack = self._compute_terrain_penetration_constraints(threshold)
        hard_rows.extend(terrain_hard)
        slack_rows.extend(terrain_slack)
        self._last_constraint_counts = {
            "self_collision_hard_rows": self_hard_count,
            "self_collision_slack_rows": self_slack_count,
            "terrain_hard_rows": len(terrain_hard),
            "terrain_slack_rows": len(terrain_slack),
        }

        return hard_rows, slack_rows

    def _nonlinear_penetration_violation(self, q: np.ndarray) -> float:
        """Return the largest hard-bound violation after forward kinematics."""
        if not self.hard_penetration_constraint:
            return 0.0
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)
        hard_rows, _ = self._compute_penetration_constraints()
        if not hard_rows:
            return 0.0
        return max(0.0, max(float(rhs) for _, rhs in hard_rows))

    def _compute_terrain_penetration_constraints(
        self, threshold: float
    ) -> Tuple[List[Tuple[np.ndarray, float]], List[Tuple[np.ndarray, float, float]]]:
        """
        Compute non-penetration rows between robot geoms and the terrain trimesh.

        Samples points on the actual surface of each collision geom based on
        its shape, then checks each point for penetration with the terrain.
        This avoids the limitation of only checking the geom center which can
        miss penetration when the geom has large extent.

        **Trade-off**: Primitive geom types (sphere, box, capsule, cylinder,
        ellipsoid) use surface samples. Meshes and heightfields fall back to
        checking the geom center.

        For each sampled point that is close to or inside the terrain, we add
        the linear constraint:

            n^T J_a  delta_v  >=  -(d - tol)

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
        hard_rows: List[Tuple[np.ndarray, float]] = []
        slack_rows: List[Tuple[np.ndarray, float, float]] = []
        m, d = self.robot_model, self.robot_data

        if not self._terrain_collision_geoms:
            return hard_rows, slack_rows

        # Transform the static local collision samples into world coordinates.
        all_points = []
        all_geom_info = []
        for gi, points_local in self._terrain_collision_geoms:
            # Get current geom pose in world frame
            pos = d.geom_xpos[gi].copy()
            rot_mat = d.geom_xmat[gi].reshape(3, 3).copy()
            points = points_local @ rot_mat.T + pos
            all_points.append(points)
            all_geom_info.append(np.full(len(points), gi, dtype=int))

        if len(all_points) == 0:
            return hard_rows, slack_rows

        all_points = np.vstack(all_points)
        all_geom_info = np.concatenate(all_geom_info)

        # Query terrain mesh for closest points to each sampled point through
        # Open3D's persistent CPU acceleration structure.
        closest_pts, unsigned_dists, tri_ids = self._closest_points_on_terrain(
            all_points
        )

        for k, gi in enumerate(all_geom_info):
            # Signed distance: positive when the point is on the outside of
            # the terrain surface, negative when it has penetrated the surface.
            # The terrain mesh is consistently wound, so its face normals already
            # point outward; penetration is therefore measured along the surface
            # normal, not forced to +Z.
            query_pt = all_points[k]
            surface_pt = closest_pts[k]

            # Face normal from terrain mesh
            raw_face_normal = self._terrain_face_normals[tri_ids[k]]
            face_normal = raw_face_normal.copy()

            # Signed distance along the outward surface normal.
            signed_dist = np.dot(query_pt - surface_pt, face_normal)

            # Only constrain points that are close to or below the surface.
            # Deep signed penetrations are retained for every face orientation
            # so an initially buried geom cannot be hidden by an
            # unsigned-distance prefilter.
            if signed_dist > threshold:
                continue

            # Translational Jacobian for this geom's body at the query point
            body_id = m.geom_bodyid[gi]
            J_full = self._calc_contact_jacobian_from_point(
                body_id, query_pt, input_world=True
            )
            # Project onto terrain normal -> 1-D Jacobian
            J_n = face_normal @ J_full  # (nv,)
            Ja = J_n[self.dof_indices]

            hard, slack = self._penetration_constraint_terms(Ja, signed_dist)
            self._record_penetration_diagnostic(
                "terrain", Ja, signed_dist, hard, slack
            )
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
