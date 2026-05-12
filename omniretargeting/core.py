"""Core OmniRetargeting functionality."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Iterator

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import trimesh
import mujoco
import yourdfpy
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from .data_sources.base import DataSource, MotionData, MotionFrame
from .utils import compute_mesh_height_at_point


@dataclass
class RetargetingStreamState:
    retargeter: Any
    q_init: np.ndarray
    q_last: np.ndarray | None
    last_estimated_quat: np.ndarray | None
    frame_idx: int
    scaled_terrain: trimesh.Trimesh


class OmniRetargeter:
    """
    Generic motion retargeting for any humanoid URDF and terrain mesh.

    This class provides functionality to retarget source trajectories to any humanoid robot
    operating on any terrain mesh by automatically scaling the terrain and computing
    appropriate joint mappings.
    """

    def __init__(
        self,
        robot_urdf_path: Union[str, Path],
        terrain_mesh_path: Union[str, Path],
        joint_mapping: Dict[str, str],
        robot_height: Optional[float] = None,
        source_target_names: Optional[List[str]] = None,
        retargeting: Optional[Dict[str, Any]] = None,
        link_offset_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the OmniRetargeter.

        Args:
            robot_urdf_path: Path to the humanoid robot URDF file
            terrain_mesh_path: Path to the terrain mesh file (any common format)
            joint_mapping: Dictionary mapping source target names to robot link names
            robot_height: Height of the robot in meters (auto-detected if None)
            source_target_names: Ordered source target names used by joint_mapping

            base_orientation: Optional source target names used for base orientation estimation
            retargeting: Optional solver/retargeting settings forwarded to interaction retargeter
            link_offset_config: Optional per-link offset dictionary forwarded to GenericInteractionRetargeter.
        """
        self.robot_urdf_path = Path(robot_urdf_path)
        self.terrain_mesh_path = Path(terrain_mesh_path)
        self.joint_mapping = joint_mapping

        if source_target_names is None:
            source_target_names = list(joint_mapping.keys())
        self.source_target_names = source_target_names

        # Optional per-robot configuration with safe defaults.

        self.retargeting_config = dict(retargeting or {})
        self.link_offset_config = link_offset_config

        # Create mapping from source target names to indices.
        self.source_target_indices = {}
        for idx, name in enumerate(self.source_target_names):
            self.source_target_indices[name] = idx

        # Get indices for mapped source targets.
        # CRITICAL: Only include targets that exist in source AND will be validated in robot.
        # Store the valid target names to ensure consistent ordering.
        self.valid_source_target_names = []
        self.mapped_source_target_indices = []
        
        for source_target_name in joint_mapping.keys():
            if source_target_name in self.source_target_indices:
                self.mapped_source_target_indices.append(self.source_target_indices[source_target_name])
                self.valid_source_target_names.append(source_target_name)
            else:
                raise ValueError(
                    f"Source target '{source_target_name}' not found in source target names. "
                    f"Available targets: {list(self.source_target_indices.keys())[:10]}..."
                )

        if len(self.mapped_source_target_indices) == 0:
            raise ValueError("No valid source target mappings found. Please check your joint_mapping dictionary.")
        
        # Store the filtered mapping (only valid source targets) for use in retargeter.
        self.valid_source_to_robot_link_mapping = {
            name: joint_mapping[name] for name in self.valid_source_target_names
        }

        # Load robot URDF
        self.robot_urdf = yourdfpy.URDF.load(str(robot_urdf_path), load_meshes=True)
        self.robot_model = mujoco.MjModel.from_xml_path(str(robot_urdf_path))
        self.robot_data = mujoco.MjData(self.robot_model)

        # Load terrain mesh
        self.terrain_mesh = trimesh.load(str(terrain_mesh_path))

        # Detect robot height if not provided
        if robot_height is None:
            self.robot_height = self._detect_robot_height()
        else:
            self.robot_height = robot_height

        # Initialize retargeting components
        self._setup_retargeting_components()

    def _detect_robot_height(self) -> float:
        """
        Detect robot height from URDF by calculating the vertical span of the robot in its default configuration.
        
        Since this is a floating-base robot, we assume the default configuration puts the robot
        in a nominal pose (e.g. standing). We calculate the difference between the highest
        and lowest points of all visual meshes to get the full height.
        """
        # Use MuJoCo to get robot height in default configuration
        # This is more reliable than parsing the URDF scene graph
        
        # Set robot to default configuration (zeros)
        mujoco.mj_resetData(self.robot_model, self.robot_data)
        self.robot_data.qpos[3:7] = [1, 0, 0, 0]  # Set base quaternion to identity
        mujoco.mj_forward(self.robot_model, self.robot_data)
        
        min_z = float('inf')
        max_z = float('-inf')
        
        # Iterate through all bodies and get their positions
        for body_idx in range(self.robot_model.nbody):
            body_pos = self.robot_data.xpos[body_idx]
            z = body_pos[2]
            min_z = min(min_z, z)
            max_z = max(max_z, z)
        
        # Also check geometry positions for more accuracy
        for geom_idx in range(self.robot_model.ngeom):
            geom_pos = self.robot_data.geom_xpos[geom_idx]
            z = geom_pos[2]
            
            # Get geometry size to account for extent
            geom_size = self.robot_model.geom_size[geom_idx]
            geom_type = self.robot_model.geom_type[geom_idx]
            
            # Estimate vertical extent based on geometry type
            if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                radius = geom_size[0]
                min_z = min(min_z, z - radius)
                max_z = max(max_z, z + radius)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                radius = geom_size[0]
                half_height = geom_size[1]
                min_z = min(min_z, z - half_height - radius)
                max_z = max(max_z, z + half_height + radius)
            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                half_size = geom_size[2]  # Z dimension
                min_z = min(min_z, z - half_size)
                max_z = max(max_z, z + half_size)
            else:
                # For other types, just use the position
                min_z = min(min_z, z)
                max_z = max(max_z, z)
        
        height = max_z - min_z
        
        # Sanity check
        if height < 0.3 or height > 3.0:
            print(f"Warning: Detected height {height:.3f}m seems unreasonable. Using default 1.6m")
            return 1.6
        
        print(f"Detected robot height: {height:.4f}m (Min Z: {min_z:.4f}, Max Z: {max_z:.4f})")

        return height

    def _setup_retargeting_components(self):
        """Setup internal retargeting components."""
        # Validate source-to-robot mapping and filter out invalid robot links.
        # CRITICAL: Only keep source targets whose mapped robot links exist in the robot URDF.
        # This ensures consistent sizes between source target positions and robot link points.
        
        # First, filter out source targets mapped to missing robot links.
        missing_robot_links = self.validate_joint_mapping()
        if missing_robot_links:
            print(f"Warning: The following robot links from joint_mapping were not found in URDF: {missing_robot_links}")
            print("Source targets mapped to these links will be removed from the mapping.")
            print(f"\nAvailable robot bodies:")
            for i in range(min(20, self.robot_model.nbody)):
                body_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, i)
                print(f"  {body_name}")
            if self.robot_model.nbody > 20:
                print(f"  ... and {self.robot_model.nbody - 20} more")
            
            # Remove source targets with missing robot links from valid_source_target_names.
            self.valid_source_target_names = [
                name for name in self.valid_source_target_names
                if self.valid_source_to_robot_link_mapping[name] not in missing_robot_links
            ]
            # Rebuild mapped_source_target_indices and valid_source_to_robot_link_mapping.
            self.mapped_source_target_indices = [
                self.source_target_indices[name] for name in self.valid_source_target_names
            ]
            self.valid_source_to_robot_link_mapping = {
                name: self.joint_mapping[name] for name in self.valid_source_target_names
            }
        
        if len(self.valid_source_target_names) == 0:
            raise ValueError(
                "No valid source target mappings found after filtering. "
                "Please check that joint_mapping maps source targets to robot links that exist in the robot URDF."
            )
        
        print(f"Successfully initialized retargeting with {len(self.valid_source_target_names)} mapped source targets.")
        print(f"Valid source targets: {self.valid_source_target_names}")
        print(f"Robot height: {self.robot_height:.3f}m")
        print(f"Robot DOF: {self.get_robot_dof()}")

    def retarget_motion(
        self,
        motion: np.ndarray | MotionData | DataSource,
        base_orientations: np.ndarray | None = None,
        base_translations: np.ndarray | None = None,
        framerate: float | None = None,
        visualize_trajectory: bool = True,
        enable_terrain_scaling: bool = False,
    ) -> Tuple[float, np.ndarray]:
        """
        Retarget a complete source motion and return ``(source_to_robot_scale, robot_motion)``.

        ``enable_terrain_scaling`` preserves the existing public API name, but the
        computed scalar is the robot/source height ratio. When enabled, batch mode
        applies it to source positions, optional root translations, and the terrain
        mesh before delegating to ``retarget_stream``.
        """
        motion_data = self._coerce_motion_data(motion)
        if base_orientations is not None or base_translations is not None:
            motion_data = MotionData(
                positions=motion_data.positions,
                target_names=motion_data.target_names,
                root_orientations=base_orientations if base_orientations is not None else motion_data.root_orientations,
                root_translations=base_translations if base_translations is not None else motion_data.root_translations,
                framerate=framerate if framerate is not None else motion_data.framerate,
                source_height=motion_data.source_height,
                metadata=motion_data.metadata,
            )
        elif framerate is not None and motion_data.framerate is None:
            motion_data.framerate = framerate

        source_to_robot_scale = self._resolve_source_to_robot_scale(
            apply_source_to_robot_scaling=enable_terrain_scaling,
            source_height=motion_data.source_height,
        )
        scaled_terrain = self._scale_terrain_mesh(source_to_robot_scale) if enable_terrain_scaling else self.terrain_mesh.copy()

        scaled_motion_data = MotionData(
            positions=motion_data.positions * source_to_robot_scale,
            target_names=motion_data.target_names,
            root_orientations=motion_data.root_orientations,
            root_translations=(
                motion_data.root_translations * source_to_robot_scale
                if motion_data.root_translations is not None
                else None
            ),
            framerate=motion_data.framerate,
            source_height=motion_data.source_height,
            metadata=motion_data.metadata,
        )

        if visualize_trajectory:
            self._visualize_trajectory(scaled_motion_data.positions, scaled_terrain)

        retargeted_motion = np.array(
            list(self.retarget_stream(scaled_motion_data, scaled_terrain=scaled_terrain))
        )

        retargeting_config = getattr(self, "retargeting_config", {})
        if retargeting_config.get("penetration_resolver", "hard_constraint") == "xyz_nudge":
            # This stabilization is intentionally batch-only: it detects contact runs,
            # smooths across windows, and corrects stance drift using temporal context
            # that retarget_stream() does not have while yielding frames one by one.
            retargeted_motion = self._apply_foot_stabilization(
                retargeted_motion,
                scaled_terrain,
                framerate=motion_data.framerate,
            )

        return source_to_robot_scale, retargeted_motion

    def retarget_stream(
        self,
        source: DataSource | MotionData | Iterable[MotionFrame] | np.ndarray,
        scaled_terrain: trimesh.Trimesh | None = None,
    ) -> Iterator[np.ndarray]:
        frames = self._iter_motion_frames(source)
        state = self.create_stream_state(scaled_terrain=scaled_terrain)
        for frame in frames:
            yield self.retarget_frame(frame, state)

    def create_stream_state(
        self,
        scaled_terrain: trimesh.Trimesh | None = None,
    ) -> RetargetingStreamState:
        from .retargeting import GenericInteractionRetargeter

        if scaled_terrain is None:
            scaled_terrain = self.terrain_mesh.copy()

        retargeter = GenericInteractionRetargeter(
            self.robot_model,
            self.robot_data,
            scaled_terrain,
            self.valid_source_to_robot_link_mapping,
            self.robot_height,
            collision_detection_threshold=float(self.retargeting_config.get("collision_detection_threshold", 0.1)),
            terrain_sample_points=int(self.retargeting_config.get("terrain_sample_points", 100)),
            source_target_names=self.valid_source_target_names,
            replace_cylinders_with_capsules=bool(self.retargeting_config.get("replace_cylinders_with_capsules", False)),
            hard_penetration_constraint=self.retargeting_config.get("penetration_resolver", "hard_constraint") == "hard_constraint",
            link_offset_config=self.link_offset_config,
        )

        q_init = np.zeros(self.robot_model.nq)
        q_init[3:7] = [1, 0, 0, 0]
        for i in range(self.robot_model.njnt):
            qpos_adr = self.robot_model.jnt_qposadr[i]
            if qpos_adr >= 7:
                joint_range = self.robot_model.jnt_range[i]
                q_init[qpos_adr] = (joint_range[0] + joint_range[1]) / 2.0
        q_init[2] = self.robot_height * 0.5

        return RetargetingStreamState(
            retargeter=retargeter,
            q_init=q_init,
            q_last=None,
            last_estimated_quat=None,
            frame_idx=0,
            scaled_terrain=scaled_terrain,
        )

    def retarget_frame(self, frame: MotionFrame | np.ndarray, state: RetargetingStreamState) -> np.ndarray:
        positions = frame.positions if isinstance(frame, MotionFrame) else frame
        root_orientation = frame.root_orientation if isinstance(frame, MotionFrame) else None
        root_translation = frame.root_translation if isinstance(frame, MotionFrame) else None
        source_positions = positions
        q_init = state.q_init

        if state.frame_idx == 0:
            if root_translation is not None:
                q_init[:3] = root_translation
            else:
                q_init[:3] = source_positions[0]
            if root_orientation is not None:
                quat_xyzw = Rotation.from_rotvec(root_orientation).as_quat()
                q_init[3:7] = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        mapped_source_targets = self._extract_mapped_source_targets(source_positions)
        
        # Use root_orientation from motion data if available
        target_quat_wxyz = None
        if root_orientation is not None:
            quat_xyzw = Rotation.from_rotvec(root_orientation).as_quat()
            target_quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        q_opt = state.retargeter.retarget_frame(
            mapped_source_targets,
            q_init,
            q_last=state.q_last,
            target_base_orientation=target_quat_wxyz,
        )
        state.q_init = q_opt
        state.q_last = q_opt
        state.frame_idx += 1
        return q_opt

    def _extract_mapped_source_targets(self, source_positions: np.ndarray) -> np.ndarray:
        if len(self.mapped_source_target_indices) != len(self.valid_source_target_names):
            raise ValueError(
                f"Order mismatch: mapped_source_target_indices has {len(self.mapped_source_target_indices)} elements, "
                f"but valid_source_target_names has {len(self.valid_source_target_names)} elements."
            )
        for target_name, source_idx in zip(self.valid_source_target_names, self.mapped_source_target_indices):
            expected_idx = self.source_target_indices.get(target_name)
            if expected_idx != source_idx:
                raise ValueError(
                    f"Order mismatch: source target '{target_name}' has index {source_idx}, "
                    f"but source_target_indices says it should be {expected_idx}."
                )
        mapped_source_targets = source_positions[self.mapped_source_target_indices]
        expected_num_targets = len(self.valid_source_target_names)
        if len(mapped_source_targets) != expected_num_targets:
            raise ValueError(
                f"Size mismatch: extracted {len(mapped_source_targets)} targets from source positions, "
                f"but expected {expected_num_targets} targets from valid_source_target_names."
            )
        return mapped_source_targets

    def _iter_motion_frames(
        self,
        source: DataSource | MotionData | Iterable[MotionFrame] | np.ndarray,
    ) -> Iterator[MotionFrame]:
        if isinstance(source, DataSource):
            return source.iter_frames()
        if isinstance(source, MotionData):
            return source.iter_frames()
        if isinstance(source, np.ndarray):
            return MotionData(positions=source).iter_frames()
        return iter(source)

    def _coerce_motion_data(self, motion: np.ndarray | MotionData | DataSource) -> MotionData:
        if isinstance(motion, MotionData):
            return motion
        if isinstance(motion, DataSource):
            return motion.load()
        return MotionData(positions=motion)

    def _resolve_source_to_robot_scale(
        self,
        apply_source_to_robot_scaling: bool,
        source_height: float | None,
    ) -> float:
        if not apply_source_to_robot_scaling:
            return 1.0
        return self._compute_source_to_robot_scale(source_height=source_height)

    def _compute_source_to_robot_scale(
        self,
        source_height: float | None = None,
    ) -> float:
        """
        Compute the source-to-robot length scale from robot and source heights.

        Batch retargeting uses this factor to put source target positions, optional root
        translations, and optionally the terrain mesh into the robot's scale.
        
        Args:
            source_height: Source height in meters (from motion data source)
        
        Returns:
            Scale factor to convert source coordinates to robot scale
        """
        # Use provided source height, or fallback to default human height
        if source_height is None:
            source_height = 1.7
            print(f"Using default source height: {source_height:.3f}m")
        else:
            print(f"Using source height from data source: {source_height:.3f}m")

        source_to_robot_scale = self.robot_height / source_height

        print(f"Computed source-to-robot scale factor: {source_to_robot_scale:.4f} (Robot: {self.robot_height}m, Source: {source_height:.3f}m)")

        return float(source_to_robot_scale)

    def _scale_terrain_mesh(self, source_to_robot_scale: float) -> trimesh.Trimesh:
        """Scale the terrain mesh by the given factor."""
        scaled_mesh = self.terrain_mesh.copy()
        scaled_mesh.apply_scale(source_to_robot_scale)
        return scaled_mesh

    def _get_foot_stabilization_config(self) -> Dict[str, Any]:
        """Return merged foot stabilization settings."""
        defaults = {
            "enabled": False,
            "clearance": 0.01,
            "surface_clearance": 0.005,
            "contact_clearance": 0.04,
            "contact_vertical_speed": 0.18,
            "min_contact_frames": 3,
            "anchor_frames": 3,
            "xy_correction_gain": 1.0,
            "xy_smoothing_window": 5,
            "z_smoothing_window": 5,
            "contact_point_height_band": 0.01,
            "max_xy_correction": 0.08,
            "max_surface_correction": 0.08,
            "surface_iterations": 4,
            "wall_red_axis_only": True,
            "wall_normal_z_threshold": 0.35,
            "wall_x_dominance_threshold": 0.5,
            "body_names": {},
        }
        cfg = dict(defaults)
        cfg.update(self.retargeting_config.get("foot_stabilization", {}) or {})
        return cfg

    def _apply_foot_stabilization(
        self,
        retargeted_motion: np.ndarray,
        terrain_mesh: trimesh.Trimesh,
        framerate: float | None = None,
    ) -> np.ndarray:
        """Post-process the retargeted motion to reduce foot penetration and stance slip."""
        cfg = self._get_foot_stabilization_config()
        if not cfg.get("enabled", False):
            return retargeted_motion
        if retargeted_motion.size == 0:
            return retargeted_motion

        foot_specs = self._build_foot_stabilization_specs(cfg)
        if not foot_specs:
            print("Foot stabilization skipped: no foot bodies could be resolved.")
            return retargeted_motion

        stabilized = np.array(retargeted_motion, copy=True)
        stabilized, wall_contact_mask = self._apply_surface_collision_corrections(
            stabilized,
            terrain_mesh,
            foot_specs,
            cfg,
        )

        contact_band = float(cfg["contact_point_height_band"])
        clearance_target = float(cfg["clearance"])
        z_window = int(cfg["z_smoothing_window"])
        xy_window = int(cfg["xy_smoothing_window"])
        min_contact_frames = int(cfg["min_contact_frames"])
        anchor_frames = max(int(cfg["anchor_frames"]), 1)
        xy_gain = float(cfg["xy_correction_gain"])
        max_xy_correction = float(cfg["max_xy_correction"])
        vertical_speed_threshold = float(cfg["contact_vertical_speed"])
        contact_clearance = float(cfg["contact_clearance"])

        positions, min_z = self._compute_foot_contact_series(stabilized, foot_specs, contact_band)
        terrain_heights = self._compute_terrain_heights(terrain_mesh, positions[:, :, :2])
        clearances = min_z - terrain_heights

        lift_clearances = np.where(~wall_contact_mask, clearances, np.inf)
        min_clearance = np.min(lift_clearances, axis=1)
        base_lift = np.maximum(clearance_target - min_clearance, 0.0)
        base_lift[~np.isfinite(min_clearance)] = 0.0
        base_lift_smoothed = self._smooth_signal(base_lift, z_window)
        base_lift = np.maximum(base_lift, base_lift_smoothed)
        stabilized[:, 2] += base_lift

        positions, min_z = self._compute_foot_contact_series(stabilized, foot_specs, contact_band)
        terrain_heights = self._compute_terrain_heights(terrain_mesh, positions[:, :, :2])
        clearances = min_z - terrain_heights

        dt = 1.0 / framerate if framerate and framerate > 0 else 1.0
        z_vel = np.zeros_like(min_z)
        if len(stabilized) > 1:
            z_vel[1:] = np.abs(np.diff(min_z, axis=0)) / dt

        contact_mask = (clearances <= contact_clearance) & (z_vel <= vertical_speed_threshold) & (~wall_contact_mask)
        for foot_idx in range(contact_mask.shape[1]):
            contact_mask[:, foot_idx] = self._filter_short_contact_runs(
                contact_mask[:, foot_idx],
                min_contact_frames=min_contact_frames,
            )

        corrections = np.zeros((len(stabilized), 2), dtype=float)
        weights = np.zeros(len(stabilized), dtype=float)

        for foot_idx in range(contact_mask.shape[1]):
            for start, end in self._iter_true_runs(contact_mask[:, foot_idx]):
                anchor_end = min(start + anchor_frames, end)
                anchor_xy = np.median(positions[start:anchor_end, foot_idx, :2], axis=0)
                drift = positions[start:end, foot_idx, :2] - anchor_xy
                corrections[start:end] += -xy_gain * drift
                weights[start:end] += 1.0

        active = weights > 0
        if np.any(active):
            corrections[active] /= weights[active, None]
            corrections = self._smooth_signal(corrections, xy_window)
            norms = np.linalg.norm(corrections, axis=1)
            oversized = norms > max_xy_correction
            if np.any(oversized):
                corrections[oversized] *= (max_xy_correction / norms[oversized])[:, None]
            stabilized[:, :2] += corrections

            positions, min_z = self._compute_foot_contact_series(stabilized, foot_specs, contact_band)
            terrain_heights = self._compute_terrain_heights(terrain_mesh, positions[:, :, :2])
            clearances = min_z - terrain_heights
            lift_clearances = np.where(~wall_contact_mask, clearances, np.inf)
            min_clearance = np.min(lift_clearances, axis=1)
            residual_lift = np.maximum(clearance_target - min_clearance, 0.0)
            residual_lift[~np.isfinite(min_clearance)] = 0.0
            stabilized[:, 2] += residual_lift

        stabilized, _ = self._apply_surface_collision_corrections(
            stabilized,
            terrain_mesh,
            foot_specs,
            cfg,
        )

        if np.any(base_lift > 0) or np.any(active):
            print(
                "Applied foot stabilization: "
                f"max base lift={base_lift.max():.4f}m, "
                f"contact frames={int(np.sum(contact_mask))}"
            )

        return stabilized

    def _build_foot_stabilization_specs(self, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Resolve foot bodies and pre-sample contact candidates in body coordinates."""
        specs = []
        body_ids = self._resolve_foot_body_ids(cfg)
        for side, body_id in body_ids.items():
            if body_id < 0:
                continue
            sample_points = self._collect_body_contact_points(body_id)
            if sample_points.size == 0:
                sample_points = np.zeros((1, 3), dtype=float)
            specs.append({
                "side": side,
                "body_id": body_id,
                "body_name": mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                "sample_points": sample_points,
                "collision_points": self._select_collision_probe_points(sample_points),
            })
        return specs

    def _resolve_foot_body_ids(self, cfg: Dict[str, Any]) -> Dict[str, int]:
        """Resolve left/right foot body ids from explicit config or robot body name search.
        
        Resolution order:
        1. Explicit config: cfg["body_names"]["left"|"right"] specifies robot body name
        2. Keyword search: search robot bodies for side ("left"/"l_") + ("foot"/"ankle")
        
        This is source-agnostic - it does not depend on source target names like
        "L_Foot" or "R_Foot". The user can override via foot_stabilization config:
            retargeting:
              foot_stabilization:
                body_names:
                  left: "left_ankle_roll_link"
                  right: "right_ankle_roll_link"
        """
        resolved = {}
        explicit = dict(cfg.get("body_names", {}) or {})

        for side in ("left", "right"):
            body_id = -1
            
            # Priority 1: Explicit robot body name from config
            explicit_name = explicit.get(side)
            if explicit_name:
                body_id = self._body_name_to_id(explicit_name)

            # Priority 2: Keyword-based search of robot body names
            if body_id < 0:
                body_id = self._search_body_id_by_keywords(side)

            resolved[side] = body_id

        return resolved

    def _body_name_to_id(self, body_name: str) -> int:
        """Convert a body name to a MuJoCo id."""
        try:
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        except Exception:
            return -1
        return int(body_id) if body_id is not None and body_id >= 0 else -1

    def _search_body_id_by_keywords(self, side: str) -> int:
        """Fallback search for foot/ankle body ids when mappings are incomplete."""
        side_tokens = ("left", "l_") if side == "left" else ("right", "r_")
        candidates = []
        for body_idx in range(self.robot_model.nbody):
            body_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_idx)
            if not body_name:
                continue
            name_lower = body_name.lower()
            if not any(token in name_lower for token in side_tokens):
                continue
            if "foot" in name_lower:
                return body_idx
            if "ankle" in name_lower:
                candidates.append(body_idx)
        return candidates[0] if candidates else -1

    def _collect_body_contact_points(self, body_id: int) -> np.ndarray:
        """Collect representative support points for a foot body in body coordinates."""
        geom_ids = [geom_id for geom_id in range(self.robot_model.ngeom) if int(self.robot_model.geom_bodyid[geom_id]) == body_id]
        if not geom_ids:
            return np.zeros((0, 3), dtype=float)

        primitive_geom_ids = [
            geom_id for geom_id in geom_ids
            if int(self.robot_model.geom_type[geom_id]) != mujoco.mjtGeom.mjGEOM_MESH
        ]
        if primitive_geom_ids:
            geom_ids = primitive_geom_ids

        points = [self._sample_geom_points_in_body_frame(geom_id) for geom_id in geom_ids]
        points = [pts for pts in points if pts.size > 0]
        if not points:
            return np.zeros((0, 3), dtype=float)
        return np.vstack(points)

    def _sample_geom_points_in_body_frame(self, geom_id: int) -> np.ndarray:
        """Sample support candidate points for one geom in the owning body frame."""
        geom_type = int(self.robot_model.geom_type[geom_id])
        size = np.asarray(self.robot_model.geom_size[geom_id], dtype=float)

        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            radius = size[0]
            points_local = np.array([
                [0.0, 0.0, -radius],
                [0.0, 0.0, radius],
                [radius, 0.0, 0.0],
                [-radius, 0.0, 0.0],
                [0.0, radius, 0.0],
                [0.0, -radius, 0.0],
            ])
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            hx, hy, hz = size
            corners = []
            for sx in (-hx, hx):
                for sy in (-hy, hy):
                    for sz in (-hz, hz):
                        corners.append([sx, sy, sz])
            corners.extend([[0.0, 0.0, -hz], [0.0, 0.0, hz]])
            points_local = np.asarray(corners, dtype=float)
        elif geom_type in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            radius = size[0]
            half_length = size[1]
            theta = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
            rings = []
            for z in (-half_length, 0.0, half_length):
                ring = np.column_stack([
                    radius * np.cos(theta),
                    radius * np.sin(theta),
                    np.full_like(theta, z),
                ])
                rings.append(ring)
            endpoints = np.array([[0.0, 0.0, -half_length], [0.0, 0.0, half_length]], dtype=float)
            if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                endpoints = np.vstack([endpoints, [[0.0, 0.0, -half_length - radius], [0.0, 0.0, half_length + radius]]])
            points_local = np.vstack(rings + [endpoints])
        elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            rx, ry, rz = size
            points_local = np.array([
                [0.0, 0.0, -rz],
                [0.0, 0.0, rz],
                [rx, 0.0, 0.0],
                [-rx, 0.0, 0.0],
                [0.0, ry, 0.0],
                [0.0, -ry, 0.0],
            ])
        else:
            points_local = np.zeros((1, 3), dtype=float)

        geom_pos = np.asarray(self.robot_model.geom_pos[geom_id], dtype=float)
        geom_quat = np.asarray(self.robot_model.geom_quat[geom_id], dtype=float)
        geom_quat_xyzw = np.array([geom_quat[1], geom_quat[2], geom_quat[3], geom_quat[0]], dtype=float)
        geom_rot = Rotation.from_quat(geom_quat_xyzw).as_matrix()

        return points_local @ geom_rot.T + geom_pos

    def _select_collision_probe_points(self, sample_points: np.ndarray) -> np.ndarray:
        """Reduce dense foot samples to a compact set of boundary probes for collision checks."""
        if sample_points.size == 0:
            return np.zeros((0, 3), dtype=float)
        if len(sample_points) <= 24:
            return np.array(sample_points, copy=True)

        directions = np.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
            [0.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0],
        ], dtype=float)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        chosen = set()
        for direction in directions:
            chosen.add(int(np.argmax(sample_points @ direction)))

        local_min_z = np.min(sample_points[:, 2])
        bottom_band = np.where(sample_points[:, 2] <= local_min_z + 0.01)[0]
        chosen.update(int(idx) for idx in bottom_band)

        chosen_points = sample_points[sorted(chosen)]
        unique_points = np.unique(np.round(chosen_points, decimals=6), axis=0)
        return unique_points.astype(float, copy=False)

    def _compute_foot_contact_series(
        self,
        motion: np.ndarray,
        foot_specs: List[Dict[str, Any]],
        contact_band: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-frame foot support positions and minimum support heights."""
        num_frames = len(motion)
        num_feet = len(foot_specs)
        positions = np.zeros((num_frames, num_feet, 3), dtype=float)
        min_z = np.zeros((num_frames, num_feet), dtype=float)

        for frame_idx, qpos in enumerate(motion):
            self.robot_data.qpos[:] = qpos
            mujoco.mj_forward(self.robot_model, self.robot_data)

            for foot_idx, spec in enumerate(foot_specs):
                body_id = spec["body_id"]
                body_pos = np.asarray(self.robot_data.xpos[body_id], dtype=float)
                body_rot = np.asarray(self.robot_data.xmat[body_id], dtype=float).reshape(3, 3)
                world_points = spec["sample_points"] @ body_rot.T + body_pos
                point_z = world_points[:, 2]
                frame_min_z = float(np.min(point_z))
                support_mask = point_z <= frame_min_z + contact_band
                support_points = world_points[support_mask]
                positions[frame_idx, foot_idx] = np.mean(support_points, axis=0)
                min_z[frame_idx, foot_idx] = frame_min_z

        return positions, min_z

    def _compute_terrain_heights(self, terrain_mesh: trimesh.Trimesh, xy_points: np.ndarray) -> np.ndarray:
        """Compute terrain heights for a batch of XY positions."""
        heights = np.zeros(xy_points.shape[:2], dtype=float)
        for frame_idx in range(xy_points.shape[0]):
            for foot_idx in range(xy_points.shape[1]):
                x, y = xy_points[frame_idx, foot_idx]
                heights[frame_idx, foot_idx] = compute_mesh_height_at_point(terrain_mesh, float(x), float(y))
        return heights

    def _apply_surface_collision_corrections(
        self,
        motion: np.ndarray,
        terrain_mesh: trimesh.Trimesh,
        foot_specs: List[Dict[str, Any]],
        cfg: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project penetrating foot probe points out of nearby terrain surfaces and flag wall-contact feet."""
        if len(motion) == 0 or not foot_specs:
            return motion, np.zeros((len(motion), len(foot_specs)), dtype=bool)

        triangles = np.asarray(terrain_mesh.triangles, dtype=float)
        if len(triangles) == 0:
            return motion, np.zeros((len(motion), len(foot_specs)), dtype=bool)
        face_normals = np.asarray(terrain_mesh.face_normals, dtype=float)

        surface_clearance = float(cfg["surface_clearance"])
        max_surface_correction = float(cfg["max_surface_correction"])
        surface_iterations = max(int(cfg["surface_iterations"]), 1)

        stabilized = np.array(motion, copy=True)
        total_xy_shift = np.zeros(len(stabilized), dtype=float)
        wall_contact_mask = np.zeros((len(stabilized), len(foot_specs)), dtype=bool)

        for frame_idx, qpos in enumerate(stabilized):
            for _ in range(surface_iterations):
                correction_vectors: List[np.ndarray] = []
                wall_x_corrections: List[float] = []
                self.robot_data.qpos[:] = qpos
                mujoco.mj_forward(self.robot_model, self.robot_data)

                for foot_idx, spec in enumerate(foot_specs):
                    collision_points = spec.get("collision_points")
                    if collision_points is None or collision_points.size == 0:
                        continue

                    body_id = spec["body_id"]
                    body_pos = np.asarray(self.robot_data.xpos[body_id], dtype=float)
                    body_rot = np.asarray(self.robot_data.xmat[body_id], dtype=float).reshape(3, 3)
                    world_points = collision_points @ body_rot.T + body_pos

                    for point in world_points:
                        correction, wall_contact = self._compute_surface_point_correction(
                            point,
                            triangles,
                            face_normals,
                            clearance=surface_clearance,
                            cfg=cfg,
                        )
                        wall_contact_mask[frame_idx, foot_idx] |= wall_contact
                        if correction is not None:
                            if wall_contact:
                                wall_x_corrections.append(float(correction[0]))
                            else:
                                correction_vectors.append(correction)

                if not correction_vectors and not wall_x_corrections:
                    break

                correction = np.zeros(3, dtype=float)
                if correction_vectors:
                    correction = np.mean(correction_vectors, axis=0)
                    correction[2] = max(correction[2], 0.0)

                if wall_x_corrections:
                    pos = max((value for value in wall_x_corrections if value > 0.0), default=0.0)
                    neg = min((value for value in wall_x_corrections if value < 0.0), default=0.0)
                    correction[0] = pos if abs(pos) >= abs(neg) else neg

                norm = np.linalg.norm(correction)
                if norm > max_surface_correction and norm > 1e-12:
                    correction *= max_surface_correction / norm

                qpos[:3] += correction
                total_xy_shift[frame_idx] += np.linalg.norm(correction[:2])

            stabilized[frame_idx] = qpos

        if np.any(total_xy_shift > 0):
            print(
                "Applied surface collision correction: "
                f"max xy shift={total_xy_shift.max():.4f}m"
            )

        return stabilized, wall_contact_mask

    def _compute_surface_point_correction(
        self,
        point: np.ndarray,
        triangles: np.ndarray,
        face_normals: np.ndarray,
        clearance: float,
        cfg: Dict[str, Any],
    ) -> Tuple[np.ndarray | None, bool]:
        """Return a correction vector plus whether the point is in wall-contact mode."""
        repeated_points = np.repeat(point[None, :], len(triangles), axis=0)
        closest_points = trimesh.triangles.closest_point(triangles, repeated_points)
        delta = point[None, :] - closest_points
        dist2 = np.einsum("ij,ij->i", delta, delta)
        face_idx = int(np.argmin(dist2))

        closest = closest_points[face_idx]
        normal = face_normals[face_idx]
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-12:
            return None, False
        normal = normal / normal_norm

        signed_offset = float(np.dot(point - closest, normal))
        penetration = clearance - signed_offset
        if penetration <= 0.0:
            return None, False

        wall_contact = False
        if bool(cfg.get("wall_red_axis_only", True)):
            wall_normal_z_threshold = float(cfg.get("wall_normal_z_threshold", 0.35))
            wall_x_dominance_threshold = float(cfg.get("wall_x_dominance_threshold", 0.5))
            if abs(normal[2]) <= wall_normal_z_threshold and abs(normal[0]) >= wall_x_dominance_threshold:
                correction = np.array([penetration * np.sign(normal[0]), 0.0, 0.0], dtype=float)
                wall_contact = True
            else:
                correction = penetration * normal
        else:
            correction = penetration * normal

        if not wall_contact:
            correction[2] = max(correction[2], 0.0)
        if np.linalg.norm(correction) < 1e-9:
            return None, wall_contact
        return correction, wall_contact

    def _smooth_signal(self, values: np.ndarray, window: int) -> np.ndarray:
        """Apply edge-padded moving average smoothing to 1D or 2D arrays."""
        if window <= 1 or values.shape[0] <= 1:
            return np.array(values, copy=True)

        kernel = np.ones(window, dtype=float) / float(window)
        pad_left = window // 2
        pad_right = window - 1 - pad_left

        if values.ndim == 1:
            padded = np.pad(values, (pad_left, pad_right), mode="edge")
            return np.convolve(padded, kernel, mode="valid")

        padded = np.pad(values, ((pad_left, pad_right), (0, 0)), mode="edge")
        smoothed = np.zeros_like(values, dtype=float)
        for col_idx in range(values.shape[1]):
            smoothed[:, col_idx] = np.convolve(padded[:, col_idx], kernel, mode="valid")
        return smoothed

    def _filter_short_contact_runs(self, mask: np.ndarray, min_contact_frames: int) -> np.ndarray:
        """Drop contact runs shorter than the configured minimum."""
        if min_contact_frames <= 1:
            return np.array(mask, copy=True)

        filtered = np.array(mask, copy=True)
        for start, end in self._iter_true_runs(mask):
            if end - start < min_contact_frames:
                filtered[start:end] = False
        return filtered

    def _iter_true_runs(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Return half-open index ranges for all True runs in a boolean mask."""
        runs: List[Tuple[int, int]] = []
        start = None
        for idx, value in enumerate(mask):
            if value and start is None:
                start = idx
            elif not value and start is not None:
                runs.append((start, idx))
                start = None
        if start is not None:
            runs.append((start, len(mask)))
        return runs

    def get_robot_dof(self) -> int:
        """Get the number of degrees of freedom of the robot."""
        return self.robot_model.nq - 7  # Subtract floating base DOF

    def get_joint_names(self) -> List[str]:
        """Get the names of all robot joints (excluding floating base)."""
        return [self.robot_model.joint(i).name for i in range(self.robot_model.njnt)
                if self.robot_model.joint(i).name and self.robot_model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]

    def validate_joint_mapping(self) -> List[str]:
        """Validate that the joint mapping is compatible with the robot.
        
        Note: joint_mapping maps source target names to robot BODY (link) names, not joint names.
        So we check for body names in the URDF.
        
        This method now delegates to the shared utility function in utils.py.
        """
        from .utils import validate_robot_joint_mapping
        return validate_robot_joint_mapping(
            self.robot_model,
            self.joint_mapping,
            raise_on_missing=False
        )

    def _visualize_trajectory(self, trajectory: np.ndarray, scaled_terrain: trimesh.Trimesh):
        """
        Visualize the source trajectory using matplotlib 3D animation with terrain mesh.
        
        Args:
            trajectory: Processed trajectory of shape (T, N, 3) where T is frames, N is source targets
                       Coordinates are assumed to be in +Z up convention (already transformed)
            scaled_terrain: Scaled terrain mesh
        """
        print(f"Visualizing trajectory with shape: {trajectory.shape}")
        print(f"Terrain mesh: {len(scaled_terrain.vertices)} vertices, {len(scaled_terrain.faces)} faces")
        print("Coordinate system: +Z is up")
        
        num_frames, num_targets, _ = trajectory.shape
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Compute axis limits based on trajectory AND terrain bounds
        all_points = trajectory.reshape(-1, 3)
        traj_bounds = np.array([all_points.min(axis=0), all_points.max(axis=0)])
        
        # Get terrain bounds
        terrain_bounds = scaled_terrain.bounds  # Shape: (2, 3) - min and max
        
        # Combine bounds
        x_min = min(traj_bounds[0, 0], terrain_bounds[0, 0])
        y_min = min(traj_bounds[0, 1], terrain_bounds[0, 1])
        z_min = min(traj_bounds[0, 2], terrain_bounds[0, 2])
        x_max = max(traj_bounds[1, 0], terrain_bounds[1, 0])
        y_max = max(traj_bounds[1, 1], terrain_bounds[1, 1])
        z_max = max(traj_bounds[1, 2], terrain_bounds[1, 2])
        
        # Add some margin
        margin = 0.2
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        ax.set_xlim([x_min - margin * x_range, x_max + margin * x_range])
        ax.set_ylim([y_min - margin * y_range, y_max + margin * y_range])
        ax.set_zlim([z_min - margin * z_range, z_max + margin * z_range])
        
        # Set labels (Z is up)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m) - Up')
        ax.set_title('Source Trajectory with Terrain Visualization')
        
        # Set equal aspect ratio
        max_range = max(x_range, y_range, z_range)
        mid_x = (x_max + x_min) / 2
        mid_y = (y_max + y_min) / 2
        mid_z = (z_max + z_min) / 2
        ax.set_xlim([mid_x - max_range/2, mid_x + max_range/2])
        ax.set_ylim([mid_y - max_range/2, mid_y + max_range/2])
        ax.set_zlim([mid_z - max_range/2, mid_z + max_range/2])
        
        # Plot terrain mesh
        print("Rendering terrain mesh...")
        terrain_vertices = scaled_terrain.vertices
        terrain_faces = scaled_terrain.faces
        
        # Create a simplified mesh for visualization if too complex
        if len(terrain_faces) > 10000:
            print(f"Terrain has {len(terrain_faces)} faces, simplifying for visualization...")
            try:
                simplified_terrain = scaled_terrain.simplify_quadric_decimation(10000)
            except ValueError:
                # Fallback for trimesh/fast_simplification version mismatch
                # where target_count is interpreted as target_reduction
                print("Using direct fast_simplification fallback due to trimesh error...")
                import fast_simplification
                vertices, faces = fast_simplification.simplify(
                    scaled_terrain.vertices, 
                    scaled_terrain.faces, 
                    target_count=10000
                )
                simplified_terrain = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            terrain_vertices = simplified_terrain.vertices
            terrain_faces = simplified_terrain.faces
            print(f"Simplified to {len(terrain_faces)} faces")
        
        # Plot terrain as a triangulated surface
        ax.plot_trisurf(
            terrain_vertices[:, 0], 
            terrain_vertices[:, 1], 
            terrain_vertices[:, 2],
            triangles=terrain_faces,
            color='gray',
            alpha=0.3,
            edgecolor='none',
            shade=True,
            linewidth=0
        )
        
        # Initialize scatter plot for source targets.
        scatter = ax.scatter([], [], [], c='blue', marker='o', s=50, alpha=0.9, edgecolors='black', linewidths=0.5)
        
        # Add text for frame counter
        frame_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
        
        # Build skeleton connections dynamically based on available targets
        skeleton_pairs = [
            # Spine
            ("Pelvis", "Spine1"),
            ("Spine1", "Spine2"),
            ("Spine2", "Spine3"),
            ("Spine3", "Neck"),
            ("Neck", "Head"),
            # Left leg
            ("Pelvis", "L_Hip"),
            ("L_Hip", "L_Knee"),
            ("L_Knee", "L_Ankle"),
            ("L_Ankle", "L_Foot"),
            # Right leg
            ("Pelvis", "R_Hip"),
            ("R_Hip", "R_Knee"),
            ("R_Knee", "R_Ankle"),
            ("R_Ankle", "R_Foot"),
            # Left arm
            ("Spine3", "L_Collar"),
            ("L_Collar", "L_Shoulder"),
            ("L_Shoulder", "L_Elbow"),
            ("L_Elbow", "L_Wrist"),
            # Right arm
            ("Spine3", "R_Collar"),
            ("R_Collar", "R_Shoulder"),
            ("R_Shoulder", "R_Elbow"),
            ("R_Elbow", "R_Wrist"),
        ]
        
        # Convert target names to indices using the mapping
        valid_connections = []
        for name1, name2 in skeleton_pairs:
            if name1 in self.source_target_indices and name2 in self.source_target_indices:
                idx1 = self.source_target_indices[name1]
                idx2 = self.source_target_indices[name2]
                valid_connections.append((idx1, idx2))
        
        # Initialize line objects for skeleton
        lines = []
        for _ in valid_connections:
            line, = ax.plot([], [], [], 'r-', linewidth=2.5, alpha=0.8)
            lines.append(line)
        
        # Set viewing angle for better visibility
        ax.view_init(elev=20, azim=45)
        
        def init():
            """Initialize animation."""
            scatter._offsets3d = ([], [], [])
            frame_text.set_text('')
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            return [scatter, frame_text] + lines
        
        def update(frame_idx):
            """Update animation for each frame."""
            # Get source target positions for current frame.
            targets = trajectory[frame_idx]  # Shape: (N, 3)
            
            # Update scatter plot
            xs, ys, zs = targets[:, 0], targets[:, 1], targets[:, 2]
            scatter._offsets3d = (xs, ys, zs)
            
            # Update frame counter
            frame_text.set_text(f'Frame: {frame_idx + 1}/{num_frames}')
            
            # Update skeleton lines
            for line, (i, j) in zip(lines, valid_connections):
                x_data = [targets[i, 0], targets[j, 0]]
                y_data = [targets[i, 1], targets[j, 1]]
                z_data = [targets[i, 2], targets[j, 2]]
                line.set_data(x_data, y_data)
                line.set_3d_properties(z_data)
            
            return [scatter, frame_text] + lines
        
        # Create animation
        print(f"Creating animation for {num_frames} frames...")
        anim = FuncAnimation(
            fig, 
            update, 
            frames=num_frames,
            init_func=init,
            interval=33,  # ~30 FPS
            blit=True,
            repeat=True
        )
        
        print("Displaying animation. Close the window to continue...")
        plt.tight_layout()
        plt.show()
        print("Visualization complete.")
