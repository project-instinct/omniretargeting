"""Core OmniRetargeting functionality."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import mujoco
import numpy as np
import trimesh

from ._core.feet import FootStabilizationMixin
from ._core.geometry import RobotGeometryMixin
from ._core.motion import FramewiseMotionMixin
from ._core.postprocessing import MotionPostprocessingMixin
from ._core.visualization import TrajectoryVisualizationMixin

try:
    import yourdfpy
except ImportError:
    yourdfpy = None

from .contact import (
    SceneContactTask,
    build_world_tracking_tasks,
    infer_scene_contact_tasks,
)
from .local_collision_repair import (
    diagnostics_json,
    local_repair_config,
    repair_local_collisions,
    select_safe_postprocessing,
    trajectory_digest,
)
from ._refinement.guard import RefinementGuard
from .trajectory_validation import TrajectoryValidator
from .scene_geometry import SceneQuery
from .trajectory_refinement import (
    measure_scene_penetration as measure_scene_penetration,
    refine_joint_trajectory,
)


def _select_penetration_safe_refinement(
    model: mujoco.MjModel,
    baseline: np.ndarray,
    refined: np.ndarray,
    scene_mesh: trimesh.Trimesh,
    penetration_config: Mapping[str, Any],
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> tuple[np.ndarray, Mapping[str, Any], float]:
    """Compatibility entry point: joint acceptance with local rollback, no blends."""
    result, stats, report = select_safe_postprocessing(
        model, baseline, refined, scene_mesh, {}, {}, penetration_config,
    )
    if not stats:
        stats = dict(before)
    return result, stats, float(report['accepted_fraction'])


def _partition_end_effector_markers(
    markers: Mapping[str, Dict[str, np.ndarray]],
    config: Mapping[str, Any],
    *,
    semantic_contact_enabled: bool,
    world_tracking_enabled: bool,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
    """Split source markers into semantic-contact and direct-tracking groups."""
    all_names = set(markers)
    world_names = (
        set(config.get("world_tracking_effectors", ["left_foot", "right_foot"]))
        if world_tracking_enabled
        else set()
    )
    semantic_names: set[str] = set()
    if semantic_contact_enabled:
        configured_semantic_names = config.get("semantic_contact_effectors")
        semantic_names = (
            set(configured_semantic_names)
            if configured_semantic_names is not None
            else all_names - world_names
        )

    overlap = semantic_names & world_names
    if overlap:
        raise ValueError(
            "End effectors cannot use semantic contact and direct world tracking "
            f"at the same time: {sorted(overlap)}"
        )

    semantic_markers = {
        name: marker for name, marker in markers.items() if name in semantic_names
    }
    world_markers = {
        name: marker for name, marker in markers.items() if name in world_names
    }
    return semantic_markers, world_markers


class OmniRetargeter(
    FramewiseMotionMixin,
    RobotGeometryMixin,
    FootStabilizationMixin,
    MotionPostprocessingMixin,
    TrajectoryVisualizationMixin,
):
    """
    Generic motion retargeting for any humanoid URDF and terrain mesh.

    This class provides functionality to retarget SMPLX trajectories to any humanoid robot
    operating on any terrain mesh by automatically scaling the terrain and computing
    appropriate joint mappings.
    """

    def __init__(
        self,
        robot_urdf_path: Union[str, Path],
        terrain_mesh_path: Union[str, Path],
        joint_mapping: Dict[str, str],
        robot_height: Optional[float] = None,
        smplx_joint_names: Optional[List[str]] = None,
        height_estimation: Optional[Dict[str, Any]] = None,
        base_orientation: Optional[Dict[str, str]] = None,
        retargeting: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the OmniRetargeter.

        Args:
            robot_urdf_path: Path to the humanoid robot URDF file
            terrain_mesh_path: Path to the terrain mesh file (any common format)
            joint_mapping: Dictionary mapping SMPLX joint names to robot link names
            robot_height: Height of the robot in meters (auto-detected if None)
            smplx_joint_names: List of SMPLX joint names in order (required for proper joint mapping)
            height_estimation: Optional settings for human height estimation from SMPLX joints
            base_orientation: Optional joint names used for base orientation estimation
            retargeting: Optional solver/retargeting settings forwarded to interaction retargeter
        """
        self.robot_urdf_path = Path(robot_urdf_path)
        self.terrain_mesh_path = Path(terrain_mesh_path)
        self.joint_mapping = joint_mapping

        # SMPLX joint names (default to standard SMPLX joint ordering)
        if smplx_joint_names is None:
            # Standard SMPLX joint names (first 22 body joints)
            self.smplx_joint_names = [
                "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
                "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
                "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
                "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"
            ]
        else:
            self.smplx_joint_names = smplx_joint_names

        # Optional per-robot configuration with safe defaults.
        self.height_estimation_config = dict(height_estimation or {})
        self.base_orientation_config = dict(base_orientation or {})
        self.retargeting_config = dict(retargeting or {})

        # Create mapping from SMPLX joint names to indices
        self.smplx_joint_indices = {}
        for idx, name in enumerate(self.smplx_joint_names):
            self.smplx_joint_indices[name] = idx

        # Get indices for mapped joints
        # CRITICAL: Only include joints that exist in SMPLX AND will be validated in robot
        # Store the valid joint names to ensure consistent ordering
        self.valid_joint_names = []  # Joint names that exist in both SMPLX and robot
        self.mapped_joint_indices = []

        for smplx_joint_name in joint_mapping.keys():
            if smplx_joint_name in self.smplx_joint_indices:
                self.mapped_joint_indices.append(self.smplx_joint_indices[smplx_joint_name])
                self.valid_joint_names.append(smplx_joint_name)
            else:
                raise ValueError(
                    f"SMPLX joint '{smplx_joint_name}' not found in SMPLX joint names. "
                    f"Available joints: {list(self.smplx_joint_indices.keys())[:10]}..."
                )

        if len(self.mapped_joint_indices) == 0:
            raise ValueError("No valid joint mappings found. Please check your joint_mapping dictionary.")

        # Store the filtered joint mapping (only valid joints) for use in retargeter
        self.valid_joint_mapping = {name: joint_mapping[name] for name in self.valid_joint_names}

        # Load robot model.  URDF files are also parsed by yourdfpy for
        # compatibility with older callers; MJCF files are consumed directly by
        # MuJoCo and do not need a URDF scene graph.
        self.robot_urdf = None
        if self.robot_urdf_path.suffix.lower() == ".urdf":
            if yourdfpy is None:
                raise ImportError("yourdfpy is required when loading URDF robot models.")
            self.robot_urdf = yourdfpy.URDF.load(str(robot_urdf_path), load_meshes=True)
        self.robot_model = mujoco.MjModel.from_xml_path(str(robot_urdf_path))
        self.robot_data = mujoco.MjData(self.robot_model)
        self._infer_end_effector_geometry()

        # Load terrain mesh
        self.terrain_mesh = trimesh.load(str(terrain_mesh_path))

        # Detect robot height if not provided
        if robot_height is None:
            self.robot_height = self._detect_robot_height()
        else:
            self.robot_height = robot_height

        # Initialize retargeting components
        self._setup_retargeting_components()


    def _setup_retargeting_components(self):
        """Setup internal retargeting components."""
        # Validate joint mapping and filter out invalid joints
        # CRITICAL: Only keep joints that exist in BOTH SMPLX and robot URDF
        # This ensures consistent sizes between human_joints and robot_points

        # First, filter out joints missing from robot URDF
        missing_robot_links = self.validate_joint_mapping()
        if missing_robot_links:
            print(f"Warning: The following robot links from joint_mapping were not found in URDF: {missing_robot_links}")
            print("These joints will be removed from the mapping.")
            print(f"\nAvailable robot bodies:")
            for i in range(min(20, self.robot_model.nbody)):
                body_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, i)
                print(f"  {body_name}")
            if self.robot_model.nbody > 20:
                print(f"  ... and {self.robot_model.nbody - 20} more")

            # Remove joints with missing robot links from valid_joint_names
            self.valid_joint_names = [
                name for name in self.valid_joint_names
                if self.valid_joint_mapping[name] not in missing_robot_links
            ]
            # Rebuild mapped_joint_indices and valid_joint_mapping
            self.mapped_joint_indices = [
                self.smplx_joint_indices[name] for name in self.valid_joint_names
            ]
            self.valid_joint_mapping = {name: self.joint_mapping[name] for name in self.valid_joint_names}

        if len(self.valid_joint_names) == 0:
            raise ValueError(
                "No valid joint mappings found after filtering. "
                "Please check that joint_mapping contains joints that exist in both SMPLX and robot URDF."
            )

        print(f"Successfully initialized retargeting with {len(self.valid_joint_names)} mapped joints.")
        print(f"Valid joints: {self.valid_joint_names}")
        print(f"Robot height: {self.robot_height:.3f}m")
        print(f"Robot DOF: {self.get_robot_dof()}")

    def retarget_motion(
        self,
        smplx_trajectory: np.ndarray,
        base_orientations: np.ndarray | None = None,
        base_translations: np.ndarray | None = None,
        source_joint_orientations: np.ndarray | None = None,
        source_reference_joint_orientations: np.ndarray | None = None,
        source_reference_joint_positions: np.ndarray | None = None,
        framerate: float | None = None,
        visualize_trajectory: bool = True,
        enable_terrain_scaling: bool = False,
        end_effector_markers: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        surface_tasks=None,
        body_surface_samples=None,
        scene_objects=None,
        object_points: np.ndarray | None = None,
    ) -> Tuple[float, np.ndarray]:
        """
        Retarget SMPLX motion to the robot on the terrain.

        Args:
            smplx_trajectory: SMPLX joint positions of shape (T, J, 3) where T is frames, J is joints
            base_orientations: Optional source-root orientation with shape
                ``(T, 4)`` as wxyz quaternions (the ``MotionData`` contract),
                or the legacy ``(T, 3)`` axis-angle representation.
            base_translations: Optional source-root world translations.
            object_points: Optional fixed world samples on moving objects,
                with shape (T, N, 3); shared by source and robot interaction meshes.
            source_joint_orientations: Optional source global joint
                orientations with shape ``(T, J, 4)`` in wxyz order.
            source_reference_joint_orientations: Optional source bind/rest
                global orientations with shape ``(J, 4)`` in wxyz order. They
                calibrate source joint frames to robot body frames.
            source_reference_joint_positions: Optional source bind/rest joint
                positions with shape ``(J, 3)``. They are used to solve a
                morphology-matched robot calibration pose before computing
                body-frame offsets.
            framerate: Motion framerate used by optional post-processing.
            enable_terrain_scaling: If True, estimate a terrain scale factor from the
                SMPLX trajectory and retarget against a scaled terrain mesh. If False,
                keep the original terrain mesh and use a scale factor of 1.0.
            end_effector_markers: Optional SMPL-X palm/sole centers and body
                surface samples, as returned by ``load_smplx_trajectory``.
            body_surface_samples: Persistent skin triangles/barycentric samples
                from a source adapter, used for body-to-body contact preservation.
            surface_tasks: Optional verified SurfaceTasks in metric world coordinates.
                Adapters without real surfaces and corresponding reference poses
                leave this disabled; joint markers do not fabricate bindings.

        Returns:
            Tuple of (terrain_scale, retargeted_trajectory)
            - terrain_scale: Scalar factor to scale the terrain mesh
            - retargeted_trajectory: Robot motion of shape (T, 7 + DOF) with [pos, quat, joints]
        """
        # Step 1: Compute terrain scaling factor
        self.last_motion_solve_reports = []
        if object_points is not None:
            object_points = np.asarray(object_points, dtype=float)
            if (object_points.ndim != 3 or object_points.shape[0] != len(smplx_trajectory)
                    or object_points.shape[2] != 3 or not np.isfinite(object_points).all()):
                raise ValueError("object_points must have finite shape (T, N, 3) on the source timeline.")
        if source_joint_orientations is not None:
            source_joint_orientations = np.asarray(
                source_joint_orientations, dtype=float
            )
            expected_shape = (*smplx_trajectory.shape[:2], 4)
            if source_joint_orientations.shape != expected_shape:
                raise ValueError(
                    "source_joint_orientations must have shape "
                    f"{expected_shape}, got {source_joint_orientations.shape}."
                )
            if not np.all(np.isfinite(source_joint_orientations)):
                raise ValueError("source_joint_orientations must be finite.")
        if source_reference_joint_orientations is not None:
            source_reference_joint_orientations = np.asarray(
                source_reference_joint_orientations, dtype=float
            )
            expected_reference_shape = (smplx_trajectory.shape[1], 4)
            if (
                source_reference_joint_orientations.shape
                != expected_reference_shape
            ):
                raise ValueError(
                    "source_reference_joint_orientations must have shape "
                    f"{expected_reference_shape}, got "
                    f"{source_reference_joint_orientations.shape}."
                )
            if not np.all(np.isfinite(source_reference_joint_orientations)):
                raise ValueError(
                    "source_reference_joint_orientations must be finite."
                )
        if source_reference_joint_positions is not None:
            source_reference_joint_positions = np.asarray(
                source_reference_joint_positions, dtype=float
            )
            expected_reference_position_shape = (
                smplx_trajectory.shape[1],
                3,
            )
            if (
                source_reference_joint_positions.shape
                != expected_reference_position_shape
            ):
                raise ValueError(
                    "source_reference_joint_positions must have shape "
                    f"{expected_reference_position_shape}, got "
                    f"{source_reference_joint_positions.shape}."
                )

        if enable_terrain_scaling:
            terrain_scale = self._compute_terrain_scale(smplx_trajectory)
            scaled_terrain = self._scale_terrain_mesh(terrain_scale)
        else:
            terrain_scale = 1.0
            scaled_terrain = self.terrain_mesh.copy()

        # Step 3: Process SMPLX trajectory
        processed_trajectory = self._process_smplx_trajectory(smplx_trajectory, terrain_scale)
        processed_base_translations = None
        if base_translations is not None:
            processed_base_translations = base_translations * terrain_scale
        processed_reference_joint_positions = None
        if source_reference_joint_positions is not None:
            processed_reference_joint_positions = (
                source_reference_joint_positions * terrain_scale
            )

        retargeting_config = getattr(self, "retargeting_config", {})
        end_effector_cfg = dict(
            retargeting_config.get("end_effector_priority", {}) or {}
        )
        semantic_contact_enabled = bool(end_effector_cfg.get("enabled", False))
        world_tracking_enabled = bool(
            end_effector_cfg.get("world_tracking_only", False)
            or end_effector_cfg.get("world_tracking_enabled", False)
        )
        semantic_tasks: Dict[str, SceneContactTask] = {}
        world_tasks: Dict[str, SceneContactTask] = {}
        end_effector_tasks: Dict[str, SceneContactTask] = {}
        if (semantic_contact_enabled or world_tracking_enabled) and end_effector_markers:
            processed_markers = {
                name: {
                    key: (
                        np.asarray(value, dtype=bool)
                        if key == "source_contact_mask"
                        else (
                            np.asarray(value, dtype=float) * terrain_scale
                            if key
                            in ("tracking_positions", "contact_samples")
                            else np.asarray(value, dtype=float)
                        )
                    )
                    for key, value in marker.items()
                }
                for name, marker in end_effector_markers.items()
            }
            semantic_markers, world_markers = _partition_end_effector_markers(
                processed_markers,
                end_effector_cfg,
                semantic_contact_enabled=semantic_contact_enabled,
                world_tracking_enabled=world_tracking_enabled,
            )
            if semantic_contact_enabled:
                semantic_tasks = infer_scene_contact_tasks(
                    semantic_markers,
                    scaled_terrain,
                    framerate=float(framerate or 30.0),
                    config=end_effector_cfg.get("contact_detection", {}),
                )
                contact_summary = ", ".join(
                    f"{name}={int(np.count_nonzero(task.contact_mask))}/{len(task.contact_mask)}"
                    for name, task in semantic_tasks.items()
                )
                print(f"Inferred SMPL-X scene contacts (frames): {contact_summary}")
                end_effector_tasks.update(semantic_tasks)
            if world_tracking_enabled:
                hinted_world_markers = {
                    name: marker
                    for name, marker in world_markers.items()
                    if marker.get("source_contact_mask") is not None
                }
                direct_world_markers = {
                    name: marker
                    for name, marker in world_markers.items()
                    if name not in hinted_world_markers
                }
                if hinted_world_markers and semantic_contact_enabled:
                    world_tasks.update(
                        infer_scene_contact_tasks(
                            hinted_world_markers,
                            scaled_terrain,
                            framerate=float(framerate or 30.0),
                            config=end_effector_cfg.get(
                                "contact_detection", {}
                            ),
                        )
                    )
                else:
                    direct_world_markers.update(hinted_world_markers)
                world_tasks.update(
                    build_world_tracking_tasks(direct_world_markers)
                )
                end_effector_tasks.update(world_tasks)
                if hinted_world_markers and semantic_contact_enabled:
                    hinted_summary = ", ".join(
                        f"{name}={int(np.count_nonzero(task.contact_mask))}/"
                        f"{len(task.contact_mask)}"
                        for name, task in world_tasks.items()
                        if name in hinted_world_markers
                    )
                    print(
                        "Using source contact timing plus scene-projected "
                        "world targets: " + hinted_summary
                    )
                if direct_world_markers:
                    print(
                        "Using per-frame SMPL-X world tracking without contact "
                        "classification or anchors: "
                        + ", ".join(direct_world_markers)
                    )
        elif semantic_contact_enabled or world_tracking_enabled:
            print(
                "End-effector world tracking requested, but no SMPL-X "
                "end-effector markers were supplied; using whole-body retargeting."
            )
        if scene_objects and semantic_contact_enabled and end_effector_markers:
            from .object_contact import merge_object_contacts
            end_effector_tasks = merge_object_contacts(
                end_effector_tasks, processed_markers, scene_objects,
                framerate=float(framerate or 30.0), config=end_effector_cfg.get("contact_detection", {}),
            )
            # The baseline/overlay decision must use the final contact plan.
            # Keeping the old static-only hand plan skips the foot baseline
            # for an object-only grasp, then local-chain constraints freeze
            # the legs and base at the neutral initialization.
            semantic_tasks = {name: end_effector_tasks[name] for name in semantic_tasks}
            print("Moving-object contacts (frames): " + ", ".join(
                f"{name}={int(np.count_nonzero(task.contact_mask & (task.environment_ids > 0)))}"
                for name, task in end_effector_tasks.items() if task.environment_ids is not None
            ))
        if retargeting_config.get("sequence_solver", {}).get("enabled", False):
            end_effector_tasks = {
                name: replace(task, preserve_contact_anchors=True)
                for name, task in end_effector_tasks.items()
            }
        self.last_end_effector_tasks = end_effector_tasks
        self.last_body_relation_tasks = None
        if (body_surface_samples is not None or scene_objects) and terrain_scale != 1.0:
            raise ValueError("Body surface contact requires unscaled metric source and scene")
        self._body_surface_samples = body_surface_samples
        self._scene_objects = tuple(scene_objects or ())
        self._source_framerate = float(framerate or 30.0)
        self.last_surface_contact_tasks = None
        self.last_surface_contact_diagnostics = {"status": "disabled", "reason": "not_built"}
        self._surface_contacts_built = False
        self.last_self_contact_tasks = None
        self.last_self_contact_diagnostics = {"status": "disabled", "reason": "not_built"}
        self.last_body_relation_diagnostics = {"status": "disabled", "reason": "experiment_disabled"}
        local_cfg = local_repair_config(
            retargeting_config.get("local_collision_repair", {}),
            priority=end_effector_cfg,
            joint_limit_safety_fraction=float(retargeting_config.get("joint_limit_safety_fraction", 1.0)),
            self_collision_enabled=bool(retargeting_config.get("self_collision_enabled", True)),
        )
        from .self_collision import self_collision_config
        local_cfg["self_collision"] = self_collision_config(retargeting_config.get("self_collision"))
        self.scene_query = SceneQuery.from_tasks(
            scaled_terrain, end_effector_tasks, full_geometry=local_cfg["full_geometry"]
        )
        effective_config = dict(retargeting_config, local_collision_repair=local_cfg)
        effective_config["active_stages"] = dict(
            robot_scale=1.0, scene_scale=float(terrain_scale),
            framewise_scene_constraints=retargeting_config.get("penetration_resolver", "hard_constraint") == "hard_constraint",
            self_collision=bool(retargeting_config.get("self_collision_enabled", True)),
            moving_object_contacts=bool(scene_objects and semantic_contact_enabled),
            moving_object_collision_constraints=False,
            trajectory_refinement=bool(retargeting_config.get("trajectory_refinement", {}).get("enabled", False) and (end_effector_tasks or local_cfg["self_collision_enabled"])),
            smoothing=bool(retargeting_config.get("trajectory_smoothing", {}).get("enabled", False)),
            local_collision_repair=bool(local_cfg["enabled"]),
            local_bidirectional=bool(local_cfg["enabled"] and local_cfg["bidirectional_enabled"] and local_cfg["iterations"]),
            surface_enhancement=bool(surface_tasks is not None),
            foot_stabilization=False,  # Fixed-base postprocessing excludes root nudges.
            contact_effectors=sorted(name for name, task in end_effector_tasks.items() if np.any(task.contact_mask)),
            tracking_effectors=sorted(end_effector_tasks),
        )
        effective_config["trajectory_refinement"] = dict(
            retargeting_config.get("trajectory_refinement", {}) or {}
        )
        effective_config["trajectory_refinement"].setdefault("joint_limit_safety_fraction", local_cfg["joint_limit_safety_fraction"])
        effective_config["self_collision"] = local_cfg["self_collision"]
        effective_config["trajectory_refinement"]["self_collision_enabled"] = local_cfg["self_collision_enabled"]
        effective_config["trajectory_refinement"]["self_collision"] = local_cfg["self_collision"]
        self.last_effective_config = effective_config
        print("Effective retargeting configuration:\n" + diagnostics_json(effective_config))

        # Option Step:
        if visualize_trajectory:
            # visualize the processed_trajectory
            self._visualize_trajectory(processed_trajectory, scaled_terrain)

        # Step 4: Perform retargeting using the generic retargeting system
        self._coupled_self_collision_projection = True
        self._projection_framerate = float(framerate or 30.0)
        self._self_collision_projection_config = {**retargeting_config, "local_collision_repair": local_cfg}
        self.last_framewise_self_collision_reports = []
        retargeting_kwargs: Dict[str, Any] = dict(
            base_orientations=base_orientations,
            base_translations=processed_base_translations,
        )
        if object_points is not None:
            retargeting_kwargs["object_points"] = object_points * terrain_scale
        if source_joint_orientations is not None:
            retargeting_kwargs["source_joint_orientations"] = (
                source_joint_orientations
            )
        if source_reference_joint_orientations is not None:
            retargeting_kwargs["source_reference_joint_orientations"] = (
                source_reference_joint_orientations
            )
        if processed_reference_joint_positions is not None:
            retargeting_kwargs["source_reference_joint_positions"] = (
                processed_reference_joint_positions
            )
        preserve_noncontact_baseline = bool(
            end_effector_cfg.get("preserve_noncontact_baseline", True)
        )
        sequence_enabled = bool(retargeting_config.get("sequence_solver", {}).get("enabled", False))
        if sequence_enabled:
            preserve_noncontact_baseline = False
        if (
            preserve_noncontact_baseline
            and semantic_tasks
            and world_tasks
            and any(np.any(task.contact_mask) for task in semantic_tasks.values())
        ):
            print(
                "Building foot-tracking baseline before applying hand-contact "
                "overlay. Non-contact frames remain identical to the baseline."
            )
            baseline_motion = self._perform_retargeting(
                processed_trajectory,
                scaled_terrain,
                end_effector_tasks=world_tasks,
                apply_surface_contacts=False,
                **retargeting_kwargs,
            )
            retargeted_motion = self._perform_retargeting(
                processed_trajectory,
                scaled_terrain,
                end_effector_tasks=end_effector_tasks,
                reference_trajectory=baseline_motion,
                overlay_contact_names=set(semantic_tasks),
                **retargeting_kwargs,
            )
        else:
            if end_effector_tasks:
                retargeting_kwargs["end_effector_tasks"] = end_effector_tasks
            retargeted_motion = self._perform_retargeting(
                processed_trajectory, scaled_terrain, **retargeting_kwargs
            )
        framewise_motion = np.asarray(retargeted_motion, dtype=float).copy()
        effective_config["active_stages"]["sequence_solver"] = sequence_enabled
        if sequence_enabled:
            effective_config["active_stages"]["framewise_scene_constraints"] = False
            effective_config["sequence_solver_report"] = self.last_sequence_report
        # Raw full-solve output for objective-independent ablation, before any
        # refinement, smoothing, clamp, acceptance or local collision repair.
        self.last_framewise_motion = getattr(self, "last_unprocessed_framewise_motion", framewise_motion).copy()
        relation_tasks = self.last_body_relation_tasks
        effective_config["active_stages"]["self_contact"] = self.last_self_contact_tasks is not None
        effective_config["active_stages"]["surface_contact"] = self.last_surface_contact_tasks is not None
        relation_stages = {}

        def record_relations(stage, motion):
            if relation_tasks is not None:
                relation_stages[stage] = relation_tasks.metrics(motion)

        # Resolve final mechanical bounds and immutable endpoint references
        # before the second solver, so it sees the same feasible set as acceptance.
        framewise_motion = self._clamp_final_joint_safety_limits(framewise_motion)
        validation_tasks, validation_priority, fallback_feet = self._validation_foot_references(
            framewise_motion, end_effector_tasks, end_effector_cfg,
        )
        effective_config["validation_only_robot_foot_references"] = fallback_feet
        effective_config["validation_end_effector_priority"] = validation_priority
        sequence_choices = {}
        if sequence_enabled:
            sequence_choices["contact_offset_choice"] = {
                name: self.last_sequence_report["contact_offset_choice"].get(
                    name, np.zeros(len(framewise_motion), dtype=int))
                for name in validation_tasks
            }
        if fallback_feet:
            print("Preserving existing robot foot world markers without inventing a contact plan: "
                  + ", ".join(fallback_feet))
        # Geometry/model/evidence remain fixed across all guarded stages. Reuse
        # exact-pose collision results; each stage still checks its own contacts.
        geometry_cache = {}
        retargeted_motion = framewise_motion.copy()
        refinement_guard = None
        record_relations("framewise", framewise_motion)
        refinement_cfg = dict(
            retargeting_config.get("trajectory_refinement", {}) or {}
        )
        refinement_cfg.setdefault(
            "joint_limit_safety_fraction",
            float(retargeting_config.get("joint_limit_safety_fraction", 1.0)),
        )
        refinement_cfg["self_collision_enabled"] = local_cfg["self_collision_enabled"]
        refinement_cfg["self_collision"] = local_cfg["self_collision"]
        if not sequence_enabled and bool(refinement_cfg.get("enabled", False)) and (end_effector_tasks or local_cfg["self_collision_enabled"] or self.last_self_contact_tasks is not None or self.last_surface_contact_tasks is not None):
            refinement_guard = RefinementGuard(TrajectoryValidator(
                self.robot_model, framewise_motion, self.scene_query,
                validation_tasks, validation_priority, local_cfg,
                float(framerate or 30.0), geometry_cache=geometry_cache,
                body_relation_tasks=relation_tasks,
                self_contact_tasks=self.last_self_contact_tasks,
                surface_contact_tasks=self.last_surface_contact_tasks,
            ))
            source_root_idx = self._resolve_source_root_index()
            retargeted_motion = refine_joint_trajectory(
                self.robot_model,
                retargeted_motion,
                end_effector_tasks,
                end_effector_cfg,
                refinement_cfg,
                refinement_guard=refinement_guard,
                scene_mesh=scaled_terrain,
                scene_query=self.scene_query,
                framerate=float(framerate or 30.0),
                source_root_positions=processed_trajectory[:, source_root_idx],
                source_joint_positions=processed_trajectory,
                source_joint_names=self.smplx_joint_names,
                **({"body_relation_tasks": relation_tasks} if relation_tasks is not None else {}),
                **({"self_contact_tasks": self.last_self_contact_tasks} if self.last_self_contact_tasks is not None else {}),
                **({"surface_contact_tasks": self.last_surface_contact_tasks} if self.last_surface_contact_tasks is not None else {}),
            )
        record_relations("refinement", retargeted_motion)
        effective_config["active_stages"]["trajectory_refinement"] = refinement_guard is not None
        before_smoothing = retargeted_motion
        if not sequence_enabled:
            retargeted_motion = self._apply_trajectory_smoothing(
                retargeted_motion,
                framerate=float(framerate or 30.0),
            )
        else:
            effective_config["active_stages"]["smoothing"] = False
        if refinement_guard is not None:
            retargeted_motion = refinement_guard.preserve_safe_smoothing(
                before_smoothing, retargeted_motion,
            )

        foot_stabilization_cfg = dict(retargeting_config.get("foot_stabilization", {}) or {})
        if (
            retargeting_config.get("penetration_resolver", "hard_constraint") == "xyz_nudge"
            or bool(foot_stabilization_cfg.get("enabled", False))
        ):
            print(
                "Foot stabilization skipped: postprocessing preserves the "
                "framewise floating base; ground lock and XYZ nudges change it. "
                "Local joint repair handles remaining scene intersections."
            )
        effective_config["foot_stabilization_runtime"] = {"status": "skipped_fixed_base"}

        retargeted_motion = self._clamp_final_joint_safety_limits(
            retargeted_motion
        )

        record_relations("postprocessing_candidate", retargeted_motion)
        retargeted_motion, postprocess_geometry, postprocess_report = select_safe_postprocessing(
            self.robot_model, framewise_motion, retargeted_motion, scaled_terrain,
            validation_tasks, validation_priority, local_cfg,
            framerate=float(framerate or 30.0), scene_query=self.scene_query,
            geometry_cache=geometry_cache,
            **sequence_choices,
            **({"body_relation_tasks": relation_tasks} if relation_tasks is not None else {}),
            **({"self_contact_tasks": self.last_self_contact_tasks} if self.last_self_contact_tasks is not None else {}),
            **({"surface_contact_tasks": self.last_surface_contact_tasks} if self.last_surface_contact_tasks is not None else {}),
        )
        record_relations("before_local_repair", retargeted_motion)
        retargeted_motion, local_report = repair_local_collisions(
            self.robot_model, retargeted_motion, scaled_terrain,
            validation_tasks, validation_priority, local_cfg,
            framerate=float(framerate or 30.0), scene_query=self.scene_query,
            geometry_cache=geometry_cache,
            **sequence_choices,
            **({"surface_tasks": surface_tasks} if surface_tasks is not None else {}),
            **({"body_relation_tasks": relation_tasks} if relation_tasks is not None else {}),
            **({"self_contact_tasks": self.last_self_contact_tasks} if self.last_self_contact_tasks is not None else {}),
            **({"surface_contact_tasks": self.last_surface_contact_tasks} if self.last_surface_contact_tasks is not None else {}),
        )
        record_relations("after_local_repair", retargeted_motion)
        # Final coupled stage uses the accepted robot motion as its fixed
        # reference. Later smoothing/repair cannot undo object avoidance or
        # select different material contact markers.
        from .whole_body_correction import correct_whole_body_trajectory
        retargeted_motion, whole_body_report = correct_whole_body_trajectory(
            self.robot_model, retargeted_motion, validation_tasks, validation_priority,
            retargeting_config.get("whole_body_correction"), scene_query=self.scene_query,
            scene_objects=self._scene_objects, framerate=float(framerate or 30.0),
            validation_config=local_cfg, body_relation_tasks=relation_tasks,
            self_contact_tasks=self.last_self_contact_tasks,
            surface_contact_tasks=self.last_surface_contact_tasks,
            contact_offset_choice=local_report.get("contact_offset_choice"),
        )
        record_relations("whole_body_correction", retargeted_motion)
        effective_config["active_stages"]["whole_body_correction"] = whole_body_report["status"] not in {"disabled", "no_active_tasks"}
        effective_config["active_stages"]["moving_object_collision_constraints"] = whole_body_report["moving_collision"].get("status") == "active"
        print("Whole-body correction: " + diagnostics_json({
            key: whole_body_report.get(key) for key in (
                "status", "accepted_update_count", "root_invariant", "max_joint_correction", "seconds"
            )
        }))
        effective_config["body_relations_runtime"] = self.last_body_relation_diagnostics
        effective_config["self_contact_runtime"] = self.last_self_contact_diagnostics
        effective_config["surface_contact_runtime"] = self.last_surface_contact_diagnostics
        from .self_collision import SelfCollisionQuery, summarize_self_collisions
        if local_cfg["self_collision_enabled"]:
            self_report = summarize_self_collisions(
                SelfCollisionQuery(self.robot_model, local_cfg["self_collision"]), retargeted_motion,
                include_interframes=True,
            )
        else:
            self_report = {"status": "disabled", "max_depth": None}
        print("Self collision: " + diagnostics_json({
            k: self_report.get(k) for k in ("status", "acceptance_status", "geometry", "pair_count", "max_depth", "interframe_max_depth", "worst", "frames_over_tolerance")
        }))
        self.last_collision_diagnostics = dict(
            framewise_self_collision_projection=self.last_framewise_self_collision_reports,
            motion_quality_experiments=getattr(self, "last_motion_solve_reports", []),
            effective_config=effective_config, postprocessing=postprocess_report,
            trajectory_refinement=(refinement_guard.report if refinement_guard is not None
                                   else {"status": "disabled"}),
            local_repair=local_report, self_collision=self_report,
            whole_body_correction=whole_body_report,
            accepted_qpos_sha256=trajectory_digest(retargeted_motion),
            self_contact={"configuration": self.last_self_contact_diagnostics,
                          "final": self.last_self_contact_tasks.metrics(retargeted_motion)
                          if self.last_self_contact_tasks is not None else None},
            surface_contact={"configuration": self.last_surface_contact_diagnostics,
                             "framewise": self.last_surface_contact_tasks.metrics(framewise_motion)
                             if self.last_surface_contact_tasks is not None else None,
                             "final": self.last_surface_contact_tasks.metrics(retargeted_motion)
                             if self.last_surface_contact_tasks is not None else None},
            body_relations=dict(configuration=self.last_body_relation_diagnostics, stages=relation_stages),
            execution_order=["framewise", "baseline_joint_clamp", "validation_references",
                             "trajectory_refinement", "optional_smoothing", "joint_clamp", "joint_acceptance",
                             "local_collision_repair", "whole_body_correction", "final_verification", "save"],
        )
        print("Local collision repair: " + diagnostics_json({
            "status": local_report["status"],
            "events": [{k: e.get(k) for k in ("window", "method", "reason", "iterations")}
                       for e in local_report["events"]],
            "unresolved": local_report.get("unresolved"),
            "unresolved_reasons": local_report.get("unresolved_reasons", [local_report.get("reason")]),
            "seconds": local_report.get("seconds"),
            "query_points": local_report.get("query_points"),
        }))

        if end_effector_tasks:
            self._report_final_motion_quality(retargeted_motion, end_effector_tasks)

        return terrain_scale, retargeted_motion


    def _compute_terrain_scale(
        self,
        smplx_trajectory: np.ndarray,
    ) -> float:
        """
        Compute the appropriate scaling factor for the terrain.

        This implementation assumes the terrain provided corresponds to the real-world scale
        relative to the robot's size.
        """
        # Default implementation: Scale based on robot vs human height ratio only
        # This assumes the terrain mesh is already at a "human scale" (e.g. captured or designed for humans)
        # and we just need to adjust it for the robot's size.

        # Estimate human height from SMPLX trajectory
        # Height ≈ (head Z - min foot Z) in standing pose
        # Or more robustly: Max height difference in trajectory

        # Use simple heuristic if we don't have betas to calculate exact height
        # Assuming typical human height of 1.7m if estimation fails or is unreliable
        estimated_human_height = 1.7

        if smplx_trajectory is not None and len(smplx_trajectory) > 0:
            # Estimate height from joints
            # SMPLX/SMPL joints: 0: Pelvis, 10/11: Feet, 15: Head (approx)
            # We can find the max vertical extent across all frames

            # Find the frame where the person is most upright (max head height)
            # Assuming Z is up
            head_joint_name = self.height_estimation_config.get("head_joint", "Head")
            foot_joint_names = self.height_estimation_config.get("foot_joints", ["L_Foot", "R_Foot"])
            head_top_offset = float(self.height_estimation_config.get("head_top_offset", 0.12))

            head_idx = self.smplx_joint_indices.get(head_joint_name, 15)
            foot_indices = [
                self.smplx_joint_indices[name]
                for name in foot_joint_names
                if name in self.smplx_joint_indices
            ]
            if not foot_indices:
                foot_indices = [10, 11]  # fallback defaults

            # Check if we have enough joints
            if smplx_trajectory.shape[1] > head_idx and all(idx < smplx_trajectory.shape[1] for idx in foot_indices):
                # Calculate height per frame: Head Z - Average Foot Z
                head_z = smplx_trajectory[:, head_idx, 2]

                # Use min foot Z per frame as ground reference relative to body
                feet_z = np.min(smplx_trajectory[:, foot_indices, 2], axis=1)

                # Height per frame
                heights = head_z - feet_z

                # Use the maximum height observed (standing pose)
                # Add offset for top of head (head joint is in neck/center of head)
                # Approx 10-15cm from head joint to top of head
                estimated_human_height = np.max(heights) + head_top_offset

                # Sanity check: Constrain to reasonable human range [1.4, 2.2]
                estimated_human_height = np.clip(estimated_human_height, 1.4, 2.2)

                print(f"Estimated human height from trajectory: {estimated_human_height:.3f}m")

        scale_factor = self.robot_height / estimated_human_height

        print(f"Computed terrain scale factor: {scale_factor:.4f} (Robot: {self.robot_height}m, Human Est: {estimated_human_height:.3f}m)")

        return float(scale_factor)

    def _extract_foot_positions(self, smplx_trajectory: np.ndarray) -> np.ndarray:
        """Extract foot positions from SMPLX trajectory."""
        # SMPLX joint indices for feet
        # L_Foot: 10, R_Foot: 11 (standard SMPLX ordering)
        foot_indices = [10, 11]

        foot_positions = []
        for frame in smplx_trajectory:
            for foot_idx in foot_indices:
                if foot_idx < len(frame):
                    foot_positions.append(frame[foot_idx])

        return np.array(foot_positions)

    def _scale_terrain_mesh(self, scale_factor: float) -> trimesh.Trimesh:
        """Scale the terrain mesh by the given factor."""
        scaled_mesh = self.terrain_mesh.copy()
        scaled_mesh.apply_scale(scale_factor)
        return scaled_mesh

    def _process_smplx_trajectory(
        self,
        smplx_trajectory: np.ndarray,
        terrain_scale: float
    ) -> np.ndarray:
        """Process SMPLX trajectory for retargeting."""
        # Apply terrain scaling to trajectory coordinates
        scaled_trajectory = smplx_trajectory * terrain_scale

        # Transform coordinate system if needed (SMPLX uses different convention)
        # TODO: Add coordinate system transformation if needed

        return scaled_trajectory


    def get_robot_dof(self) -> int:
        """Get the number of degrees of freedom of the robot."""
        return self.robot_model.nq - 7  # Subtract floating base DOF

    def get_joint_names(self) -> List[str]:
        """Get the names of all robot joints (excluding floating base)."""
        return [self.robot_model.joint(i).name for i in range(self.robot_model.njnt)
                if self.robot_model.joint(i).name and self.robot_model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]

    def validate_joint_mapping(self) -> List[str]:
        """Validate that the joint mapping is compatible with the robot.

        Note: joint_mapping maps SMPLX joint names to robot BODY (link) names, not joint names.
        So we check for body names in the URDF.
        """
        # Get all body names from the robot model
        robot_bodies = set()
        for i in range(self.robot_model.nbody):
            body_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name:
                robot_bodies.add(body_name)

        # Check which mapped body names don't exist
        mapped_bodies = set(self.joint_mapping.values())
        missing_bodies = mapped_bodies - robot_bodies
        return list(missing_bodies)
