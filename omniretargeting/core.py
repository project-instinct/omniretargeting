"""Core OmniRetargeting functionality."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import trimesh
import mujoco
try:
    import yourdfpy
except ImportError:
    yourdfpy = None
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from .utils import compute_mesh_height_at_point
from .contact import SceneContactTask, infer_scene_contact_tasks
from .trajectory_refinement import (
    measure_scene_penetration,
    refine_joint_trajectory,
)


class OmniRetargeter:
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
        framerate: float | None = None,
        visualize_trajectory: bool = True,
        enable_terrain_scaling: bool = False,
        end_effector_markers: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    ) -> Tuple[float, np.ndarray]:
        """
        Retarget SMPLX motion to the robot on the terrain.

        Args:
            smplx_trajectory: SMPLX joint positions of shape (T, J, 3) where T is frames, J is joints
            framerate: Motion framerate used by optional post-processing.
            enable_terrain_scaling: If True, estimate a terrain scale factor from the
                SMPLX trajectory and retarget against a scaled terrain mesh. If False,
                keep the original terrain mesh and use a scale factor of 1.0.
            end_effector_markers: Optional SMPL-X palm/sole centers and body
                surface samples, as returned by ``load_smplx_trajectory``.

        Returns:
            Tuple of (terrain_scale, retargeted_trajectory)
            - terrain_scale: Scalar factor to scale the terrain mesh
            - retargeted_trajectory: Robot motion of shape (T, 7 + DOF) with [pos, quat, joints]
        """
        # Step 1: Compute terrain scaling factor
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

        retargeting_config = getattr(self, "retargeting_config", {})
        end_effector_cfg = dict(
            retargeting_config.get("end_effector_priority", {}) or {}
        )
        end_effector_tasks: Dict[str, SceneContactTask] = {}
        if bool(end_effector_cfg.get("enabled", False)) and end_effector_markers:
            processed_markers = {
                name: {
                    key: np.asarray(value, dtype=float) * terrain_scale
                    for key, value in marker.items()
                }
                for name, marker in end_effector_markers.items()
            }
            end_effector_tasks = infer_scene_contact_tasks(
                processed_markers,
                scaled_terrain,
                framerate=float(framerate or 30.0),
                config=end_effector_cfg.get("contact_detection", {}),
            )
            contact_summary = ", ".join(
                f"{name}={int(np.count_nonzero(task.contact_mask))}/{len(task.contact_mask)}"
                for name, task in end_effector_tasks.items()
            )
            print(f"Inferred SMPL-X scene contacts (frames): {contact_summary}")
        elif bool(end_effector_cfg.get("enabled", False)):
            print(
                "Contact-priority optimization requested, but no SMPL-X "
                "end-effector markers were supplied; using whole-body retargeting."
            )
        self.last_end_effector_tasks = end_effector_tasks

        # Option Step:
        if visualize_trajectory:
            # visualize the processed_trajectory
            self._visualize_trajectory(processed_trajectory, scaled_terrain)

        # Step 4: Perform retargeting using the generic retargeting system
        retargeting_kwargs = dict(
            base_orientations=base_orientations,
            base_translations=processed_base_translations,
        )
        if end_effector_tasks:
            retargeting_kwargs["end_effector_tasks"] = end_effector_tasks
        retargeted_motion = self._perform_retargeting(
            processed_trajectory, scaled_terrain, **retargeting_kwargs
        )
        refinement_cfg = dict(
            retargeting_config.get("trajectory_refinement", {}) or {}
        )
        if bool(refinement_cfg.get("enabled", False)) and end_effector_tasks:
            penetration_cfg = dict(refinement_cfg.get("penetration", {}) or {})
            report_penetration = bool(penetration_cfg.get("report", True))
            before = (
                measure_scene_penetration(
                    self.robot_model,
                    retargeted_motion,
                    scaled_terrain,
                    penetration_cfg,
                )
                if report_penetration
                else None
            )
            retargeted_motion = refine_joint_trajectory(
                self.robot_model,
                retargeted_motion,
                end_effector_tasks,
                end_effector_cfg,
                refinement_cfg,
                scene_mesh=scaled_terrain,
                framerate=float(framerate or 30.0),
                source_root_positions=processed_trajectory[
                    :,
                    self.smplx_joint_indices.get(
                        self.base_orientation_config.get(
                            "root_position", "Spine3"
                        ),
                        0,
                    ),
                ],
                source_joint_positions=processed_trajectory,
                source_joint_names=self.smplx_joint_names,
            )
            if before is not None:
                after = measure_scene_penetration(
                    self.robot_model,
                    retargeted_motion,
                    scaled_terrain,
                    penetration_cfg,
                )
                # Jerk is part of the refinement objective, so its improvement
                # is not evidence of a good result.  Penetration is only ever
                # penalized, which makes it the honest acceptance check.
                for label, stats in (("before", before), ("after", after)):
                    print(
                        f"Scene penetration {label} refinement: "
                        f"max={stats['max_depth'] * 100.0:.2f} cm, "
                        f"mean={stats['mean_depth'] * 100.0:.2f} cm, "
                        f"frames={stats['frames_penetrating']}/{stats['num_frames']}"
                        + (
                            f", worst={stats['worst_body']}@{stats['worst_frame']}"
                            if stats["worst_body"]
                            else ""
                        )
                    )
                if after["max_depth"] > before["max_depth"] + 1e-4:
                    print(
                        "Scene penetration warning: refinement increased the "
                        "worst penetration depth. Tighten "
                        "trajectory_refinement.penetration.sigma or relax "
                        "acceleration_sigma/jerk_sigma."
                    )
        retargeted_motion = self._apply_trajectory_smoothing(
            retargeted_motion,
            framerate=float(framerate or 30.0),
        )

        foot_stabilization_cfg = dict(retargeting_config.get("foot_stabilization", {}) or {})
        if end_effector_tasks and bool(foot_stabilization_cfg.get("enabled", False)):
            print(
                "Foot stabilization skipped: contact-priority retargeting already "
                "anchors the feet to the scene, and the post-hoc ground lock would "
                "move the base away from those anchors."
            )
        if not end_effector_tasks and (
            retargeting_config.get("penetration_resolver", "hard_constraint") == "xyz_nudge"
            or bool(foot_stabilization_cfg.get("enabled", False))
        ):
            retargeted_motion = self._apply_foot_stabilization(
                retargeted_motion,
                scaled_terrain,
                framerate=framerate,
            )

        return terrain_scale, retargeted_motion

    def _apply_trajectory_smoothing(
        self,
        trajectory: np.ndarray,
        framerate: float,
    ) -> np.ndarray:
        """Remove zero-mean framewise IK chatter without moving the root.

        A symmetric Gaussian convolution is zero phase: it suppresses the
        high-frequency redundant-joint solution switching without introducing
        a time delay between the human and robot.  Its positive kernel also
        avoids ringing around contact transitions.  Only scalar robot joints
        are filtered.  Floating-base translation and orientation remain
        exactly as solved, so scene alignment and base position cannot drift.
        """
        cfg = dict(
            getattr(self, "retargeting_config", {}).get(
                "trajectory_smoothing", {}
            )
            or {}
        )
        if not bool(cfg.get("enabled", False)) or len(trajectory) < 8:
            return trajectory
        sigma_frames = float(cfg.get("sigma_frames", 1.0))
        if sigma_frames <= 0.0:
            raise ValueError(
                "trajectory_smoothing.sigma_frames must be positive, got "
                f"{sigma_frames:g}."
            )
        joint_qpos_indices = []
        for joint_id in range(self.robot_model.njnt):
            joint_type = self.robot_model.jnt_type[joint_id]
            if joint_type in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                qpos_index = int(self.robot_model.jnt_qposadr[joint_id])
                if qpos_index >= 7:
                    joint_qpos_indices.append(qpos_index)
        if not joint_qpos_indices:
            return trajectory

        result = np.asarray(trajectory, dtype=float).copy()
        result[:, joint_qpos_indices] = gaussian_filter1d(
            result[:, joint_qpos_indices],
            sigma=sigma_frames,
            axis=0,
            mode="nearest",
        )

        # The zero-phase filter can overshoot a saturated joint by a fraction
        # of a degree.  Restore the model's exact scalar joint limits.
        for joint_id in range(self.robot_model.njnt):
            if not self.robot_model.jnt_limited[joint_id]:
                continue
            qpos_index = int(self.robot_model.jnt_qposadr[joint_id])
            if qpos_index not in joint_qpos_indices:
                continue
            lower, upper = self.robot_model.jnt_range[joint_id]
            result[:, qpos_index] = np.clip(
                result[:, qpos_index], lower, upper
            )
        return result

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

    def _perform_retargeting(
        self,
        processed_trajectory: np.ndarray,
        scaled_terrain: trimesh.Trimesh,
        base_orientations: np.ndarray | None = None,
        base_translations: np.ndarray | None = None,
        end_effector_tasks: Optional[Dict[str, SceneContactTask]] = None,
    ) -> np.ndarray:
        """Perform the actual motion retargeting using generic interaction mesh retargeting."""
        from .retargeting import GenericInteractionRetargeter

        # Create retargeter instance
        # Note: scaled_terrain is already scaled by _scale_terrain_mesh
        # CRITICAL: Pass only valid_joint_mapping to ensure consistent sizes
        # This ensures robot_points and human_joints have the same number of joints
        preserve_root_z = bool(self.retargeting_config.get("preserve_root_z", False))
        retargeter_kwargs: Dict[str, Any] = {}
        if end_effector_tasks:
            retargeter_kwargs["end_effector_config"] = self.retargeting_config.get(
                "end_effector_priority", {}
            )
        if preserve_root_z:
            retargeter_kwargs["preserve_root_z"] = True
        retargeter = GenericInteractionRetargeter(
            self.robot_model,
            self.robot_data,
            scaled_terrain,
            self.valid_joint_mapping,  # Use filtered mapping, not full joint_mapping
            self.robot_height,
            collision_detection_threshold=float(self.retargeting_config.get("collision_detection_threshold", 0.1)),
            terrain_sample_points=int(self.retargeting_config.get("terrain_sample_points", 100)),
            valid_joint_names=self.valid_joint_names,  # CRITICAL: Pass ordered joint names for consistency
            replace_cylinders_with_capsules=bool(self.retargeting_config.get("replace_cylinders_with_capsules", False)),
            hard_penetration_constraint=self.retargeting_config.get("penetration_resolver", "hard_constraint") == "hard_constraint",
            **retargeter_kwargs,
        )

        # Retarget each frame
        retargeted_trajectory = []
        
        # Start from the model's nominal configuration.  Joint-range midpoints
        # are not a neutral pose for humanoids (for G1 they bend the knees by
        # about 1.4 rad), which previously forced the floating base downward to
        # compensate for artificially shortened legs.
        try:
            q_init = np.asarray(self.robot_model.qpos0, dtype=float).copy()
            if q_init.shape != (int(self.robot_model.nq),):
                raise ValueError("invalid model qpos0 shape")
        except (TypeError, ValueError):
            # Lightweight unit-test models and third-party wrappers may not
            # expose qpos0.  Preserve the historical zero initialization for
            # those callers while real MuJoCo models use their neutral pose.
            q_init = np.zeros(int(self.robot_model.nq), dtype=float)
        if len(q_init) >= 7 and np.linalg.norm(q_init[3:7]) < 1e-8:
            q_init[3:7] = [1, 0, 0, 0]  # Identity quaternion (wxyz)
        root_position_name = self.base_orientation_config.get("root_position", "Pelvis")
        root_position_idx = getattr(self, "smplx_joint_indices", {}).get(
            root_position_name, 0
        )
        
        # Initial height: Use processed (scaled) trajectory root height or default
        # Assuming joint 0 is root
        if len(processed_trajectory) > 0:
            q_init[:3] = processed_trajectory[0, root_position_idx] # Start at first frame root pos
            # Maintain initial rotation (identity or from trajectory if available)
        else:
            q_init[2] = self.robot_height * 0.5  # Default height (Z coordinate)

        for frame_idx, human_joints in enumerate(processed_trajectory):
            # Only initialize base from SMPLX for the first frame
            # For subsequent frames, the optimization will handle base movement
            if frame_idx == 0:
                # Initialize base position from SMPLX root
                if base_translations is not None:
                    q_init[:3] = base_translations[frame_idx]
                else:
                    # Use the configured root position as initial guess
                    q_init[:3] = human_joints[root_position_idx]

                # Initialize base orientation
                if base_orientations is not None:
                    root_rotvec = base_orientations[frame_idx]
                    quat_xyzw = Rotation.from_rotvec(root_rotvec).as_quat()
                    q_init[3:7] = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
                else:
                    quat_xyzw = self._estimate_base_orientation_from_joints(human_joints)
                    if quat_xyzw is not None:
                        q_init[3:7] = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            # For subsequent frames, q_init already contains the previous optimized configuration
            if preserve_root_z:
                if base_translations is not None:
                    q_init[2] = base_translations[frame_idx, 2]
                else:
                    q_init[2] = human_joints[root_position_idx, 2]
            
            # Extract mapped joints using the proper indices
            # CRITICAL: The order MUST match valid_joint_names exactly
            # mapped_joint_indices[i] corresponds to valid_joint_names[i] for all i
            # human_joints shape: (num_smplx_joints, 3)
            
            # Validate order consistency before extraction
            if len(self.mapped_joint_indices) != len(self.valid_joint_names):
                raise ValueError(
                    f"Order mismatch: mapped_joint_indices has {len(self.mapped_joint_indices)} elements, "
                    f"but valid_joint_names has {len(self.valid_joint_names)} elements."
                )
            
            # Verify that each index corresponds to the correct joint name
            for i, (joint_name, smplx_idx) in enumerate(zip(self.valid_joint_names, self.mapped_joint_indices)):
                expected_idx = self.smplx_joint_indices.get(joint_name)
                if expected_idx != smplx_idx:
                    raise ValueError(
                        f"Order mismatch at position {i}: joint '{joint_name}' has index {smplx_idx} "
                        f"in mapped_joint_indices, but smplx_joint_indices says it should be {expected_idx}."
                    )
            
            mapped_joints = human_joints[self.mapped_joint_indices]
            # mapped_joints shape: (num_mapped_joints, 3)
            # mapped_joints[i] corresponds to valid_joint_names[i] for all i
            
            # CRITICAL: Validate that we extracted the expected number of joints
            expected_num_joints = len(self.valid_joint_names)
            if len(mapped_joints) != expected_num_joints:
                raise ValueError(
                    f"Size mismatch: extracted {len(mapped_joints)} joints from human_joints, "
                    f"but expected {expected_num_joints} joints from valid_joint_names. "
                    f"This indicates an inconsistency in joint mapping."
                )
            
            # Debug: Print order and actual positions for first frame to verify matching
            import sys
            if not hasattr(sys, '_omni_human_joint_order_printed'):
                print(f"\n=== Human Joint Extraction Order ===")
                print(f"valid_joint_names: {self.valid_joint_names}")
                print(f"mapped_joint_indices: {self.mapped_joint_indices}")
                print(f"Extracted {len(mapped_joints)} joints in order: {self.valid_joint_names}")
                print(f"\n=== First Frame Joint Positions (to verify order) ===")
                for i, (name, idx) in enumerate(zip(self.valid_joint_names, self.mapped_joint_indices)):
                    pos = mapped_joints[i] if frame_idx == 0 else mapped_joints[i]
                    if frame_idx == 0:
                        print(f"  {i}: {name} (SMPLX idx {idx}) -> pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                # Specifically check knee vs ankle positions
                if frame_idx == 0:
                    l_knee_idx = self.valid_joint_names.index("L_Knee") if "L_Knee" in self.valid_joint_names else None
                    l_ankle_idx = self.valid_joint_names.index("L_Ankle") if "L_Ankle" in self.valid_joint_names else None
                    if l_knee_idx is not None and l_ankle_idx is not None:
                        l_knee_pos = mapped_joints[l_knee_idx]
                        l_ankle_pos = mapped_joints[l_ankle_idx]
                        print(f"\n=== Knee vs Ankle Position Check ===")
                        print(f"L_Knee (idx {l_knee_idx}, SMPLX {self.mapped_joint_indices[l_knee_idx]}): [{l_knee_pos[0]:.3f}, {l_knee_pos[1]:.3f}, {l_knee_pos[2]:.3f}]")
                        print(f"L_Ankle (idx {l_ankle_idx}, SMPLX {self.mapped_joint_indices[l_ankle_idx]}): [{l_ankle_pos[0]:.3f}, {l_ankle_pos[1]:.3f}, {l_ankle_pos[2]:.3f}]")
                        print(f"Knee should be HIGHER than ankle (knee.z > ankle.z): {l_knee_pos[2] > l_ankle_pos[2]}")
                sys._omni_human_joint_order_printed = True

            # Estimate target base orientation from human joints
            last_target_quat = None
            if frame_idx > 0 and len(retargeted_trajectory) > 0:
                # Get last orientation from previous frame result or previous target estimate
                # We use the result (wxyz) converted back to xyzw for continuity check
                # But wait, retargeted_trajectory has wxyz.
                # _estimate returns xyzw.
                # Better to store the previous estimated xyzw.
                pass
                
            # We need to maintain state of estimated quaternion to ensure continuity
            if not hasattr(self, '_last_estimated_quat'):
                self._last_estimated_quat = None
                
            target_quat_xyzw = self._estimate_base_orientation_from_joints(human_joints, self._last_estimated_quat)
            self._last_estimated_quat = target_quat_xyzw
            
            target_quat_wxyz = None
            if target_quat_xyzw is not None:
                # Convert xyzw to wxyz for MuJoCo
                target_quat_wxyz = np.array([target_quat_xyzw[3], target_quat_xyzw[0], 
                                            target_quat_xyzw[1], target_quat_xyzw[2]])

            # Retarget frame with previous frame as reference for smoothness
            q_last = retargeted_trajectory[-1] if len(retargeted_trajectory) > 0 else None
            frame_kwargs: Dict[str, Any] = dict(
                q_last=q_last,
                target_base_orientation=target_quat_wxyz,
            )
            if end_effector_tasks:
                base_position_cfg = dict(
                    self.retargeting_config.get("end_effector_priority", {}).get(
                        "base_position_constraint", {}
                    )
                    or {}
                )
                if bool(base_position_cfg.get("enabled", False)):
                    frame_kwargs["target_base_position"] = human_joints[
                        root_position_idx
                    ]
                frame_kwargs["end_effector_targets"] = {}
                for name, task in end_effector_tasks.items():
                    target_payload = task.frame(frame_idx)
                    target_payload["source_root_position"] = human_joints[
                        root_position_idx
                    ]
                    frame_kwargs["end_effector_targets"][name] = target_payload
                frame_kwargs["q_prev"] = (
                    retargeted_trajectory[-2]
                    if len(retargeted_trajectory) > 1
                    else None
                )
                frame_kwargs["q_prevprev"] = (
                    retargeted_trajectory[-3]
                    if len(retargeted_trajectory) > 2
                    else None
                )
                frame_kwargs["previous_human_joints"] = (
                    processed_trajectory[frame_idx - 1, self.mapped_joint_indices]
                    if frame_idx > 0
                    else None
                )
                frame_kwargs["previous_previous_human_joints"] = (
                    processed_trajectory[frame_idx - 2, self.mapped_joint_indices]
                    if frame_idx > 1
                    else None
                )
            q_opt = retargeter.retarget_frame(mapped_joints, q_init, **frame_kwargs)
            if preserve_root_z:
                q_opt[2] = q_init[2]
            retargeted_trajectory.append(q_opt)

            # Update initial guess for next frame (smooth trajectory)
            q_init = q_opt.copy()

        return np.array(retargeted_trajectory)

    def _estimate_base_orientation_from_joints(self, joints: np.ndarray, last_quat: np.ndarray | None = None) -> np.ndarray | None:
        """
        Estimate a base orientation from joint positions.

        Uses pelvis/hips/spine to build an approximate body frame:
        - up: pelvis -> spine1
        - right: left_hip -> right_hip
        - forward: right x up
        Returns quaternion in xyzw order.
        
        Args:
            joints: Joint positions array.
            last_quat: Quaternion from previous frame to ensure continuity (xyzw).
        """
        if joints.shape[0] < 4:
            return None

        pelvis_name = self.base_orientation_config.get("pelvis", "Pelvis")
        left_hip_name = self.base_orientation_config.get("left_hip", "L_Hip")
        right_hip_name = self.base_orientation_config.get("right_hip", "R_Hip")
        spine_name = self.base_orientation_config.get("spine", "Spine1")

        pelvis_idx = self.smplx_joint_indices.get(pelvis_name, 0)
        left_hip_idx = self.smplx_joint_indices.get(left_hip_name, 1)
        right_hip_idx = self.smplx_joint_indices.get(right_hip_name, 2)
        spine_idx = self.smplx_joint_indices.get(spine_name, 3)

        max_required_idx = max(pelvis_idx, left_hip_idx, right_hip_idx, spine_idx)
        if joints.shape[0] <= max_required_idx:
            return None

        pelvis = joints[pelvis_idx]
        left_hip = joints[left_hip_idx]
        right_hip = joints[right_hip_idx]
        spine1 = joints[spine_idx]

        up = spine1 - pelvis
        right = right_hip - left_hip

        up_norm = np.linalg.norm(up)
        right_norm = np.linalg.norm(right)
        if up_norm < 1e-8 or right_norm < 1e-8:
            return None

        up = up / up_norm
        right = right / right_norm
        
        # Build forward using right-hand rule
        forward = np.cross(up, right)
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-8:
            return None
        forward = forward / forward_norm

        # Re-orthogonalize right to ensure perfect orthogonality
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)

        # Rotation matrix with body axes in world coordinates.
        # Convention: X=forward, Y=right, Z=up
        rot = np.column_stack([forward, right, up])
        
        # Verify right-handed
        det = np.linalg.det(rot)
        if det < 0:
            forward = -forward
            right = -right
            rot = np.column_stack([forward, right, up])
        
        current_quat = Rotation.from_matrix(rot).as_quat() # xyzw
        
        # Ensure continuity if last quaternion is provided
        if last_quat is not None:
            # Check dot product to see if we need to flip sign
            dot = np.dot(current_quat, last_quat)
            if dot < 0:
                current_quat = -current_quat
                
        return current_quat

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
            "ground_lock": True,
            "ground_lock_clearance": 0.0,
            "ground_lock_body_tokens": ["ankle", "foot"],
            "ground_lock_geom_groups": [2, 3],
            "preserve_root_z": False,
        }
        cfg = dict(defaults)
        cfg.update(self.retargeting_config.get("foot_stabilization", {}) or {})
        if bool(self.retargeting_config.get("preserve_root_z", False)):
            cfg["preserve_root_z"] = True
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

        ground_lock_specs = []
        if bool(cfg.get("ground_lock", True)):
            ground_lock_specs = self._build_foot_ground_lock_specs(cfg)
            if not ground_lock_specs:
                print("Foot ground lock skipped: no foot geoms could be resolved.")

        stabilized = np.array(retargeted_motion, copy=True)
        preserve_root_z = bool(cfg.get("preserve_root_z", False))
        original_root_z = stabilized[:, 2].copy()
        stabilized, wall_contact_mask = self._apply_surface_collision_corrections(
            stabilized,
            terrain_mesh,
            foot_specs,
            cfg,
        )
        if preserve_root_z:
            stabilized[:, 2] = original_root_z

        contact_band = float(cfg["contact_point_height_band"])
        clearance_target = float(cfg.get("ground_lock_clearance", cfg["clearance"])) if ground_lock_specs else float(cfg["clearance"])
        z_window = int(cfg["z_smoothing_window"])
        xy_window = int(cfg["xy_smoothing_window"])
        min_contact_frames = int(cfg["min_contact_frames"])
        anchor_frames = max(int(cfg["anchor_frames"]), 1)
        xy_gain = float(cfg["xy_correction_gain"])
        max_xy_correction = float(cfg["max_xy_correction"])
        vertical_speed_threshold = float(cfg["contact_vertical_speed"])
        contact_clearance = float(cfg["contact_clearance"])

        if ground_lock_specs:
            positions, min_z = self._compute_foot_ground_lock_series(stabilized, ground_lock_specs, contact_band)
            wall_contact_mask = np.zeros_like(min_z, dtype=bool)
        else:
            positions, min_z = self._compute_foot_contact_series(stabilized, foot_specs, contact_band)
        terrain_heights = self._compute_terrain_heights(terrain_mesh, positions[:, :, :2])
        clearances = min_z - terrain_heights

        lift_clearances = np.where(~wall_contact_mask, clearances, np.inf)
        min_clearance = np.min(lift_clearances, axis=1)
        base_lift = np.maximum(clearance_target - min_clearance, 0.0)
        base_lift[~np.isfinite(min_clearance)] = 0.0
        base_lift_smoothed = self._smooth_signal(base_lift, z_window)
        base_lift = np.maximum(base_lift, base_lift_smoothed)
        if preserve_root_z:
            base_lift[:] = 0.0
        else:
            stabilized[:, 2] += base_lift

        if ground_lock_specs:
            positions, min_z = self._compute_foot_ground_lock_series(stabilized, ground_lock_specs, contact_band)
        else:
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

            if ground_lock_specs:
                positions, min_z = self._compute_foot_ground_lock_series(stabilized, ground_lock_specs, contact_band)
            else:
                positions, min_z = self._compute_foot_contact_series(stabilized, foot_specs, contact_band)
            terrain_heights = self._compute_terrain_heights(terrain_mesh, positions[:, :, :2])
            clearances = min_z - terrain_heights
            lift_clearances = np.where(~wall_contact_mask, clearances, np.inf)
            min_clearance = np.min(lift_clearances, axis=1)
            residual_lift = np.maximum(clearance_target - min_clearance, 0.0)
            residual_lift[~np.isfinite(min_clearance)] = 0.0
            if not preserve_root_z:
                stabilized[:, 2] += residual_lift

        stabilized, _ = self._apply_surface_collision_corrections(
            stabilized,
            terrain_mesh,
            foot_specs,
            cfg,
        )
        if preserve_root_z:
            stabilized[:, 2] = original_root_z

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

    def _build_foot_ground_lock_specs(self, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect visible/collision foot geoms used for ground locking."""
        body_tokens = tuple(str(token).lower() for token in cfg.get("ground_lock_body_tokens", ["ankle", "foot"]))
        geom_groups = cfg.get("ground_lock_geom_groups", [2, 3])
        geom_groups = set(int(group) for group in geom_groups) if geom_groups is not None else None

        specs = []
        for side, side_tokens in {
            "left": ("left", "l_"),
            "right": ("right", "r_"),
        }.items():
            geom_specs = []
            for geom_id in range(self.robot_model.ngeom):
                if geom_groups is not None and int(self.robot_model.geom_group[geom_id]) not in geom_groups:
                    continue
                body_id = int(self.robot_model.geom_bodyid[geom_id])
                body_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
                body_name_lower = body_name.lower()
                if not any(token in body_name_lower for token in side_tokens):
                    continue
                if not any(token in body_name_lower for token in body_tokens):
                    continue

                sample_points = self._sample_geom_points_in_body_frame(geom_id)
                if sample_points.size == 0:
                    continue
                geom_specs.append({
                    "geom_id": geom_id,
                    "body_id": body_id,
                    "body_name": body_name,
                    "sample_points": sample_points,
                })

            if geom_specs:
                specs.append({"side": side, "geom_specs": geom_specs})
        return specs

    def _resolve_foot_body_ids(self, cfg: Dict[str, Any]) -> Dict[str, int]:
        """Resolve left/right foot body ids from config, mapping, or robot body names."""
        resolved = {}
        explicit = dict(cfg.get("body_names", {}) or {})

        for side, joint_candidates in {
            "left": ["L_Foot", "L_Ankle"],
            "right": ["R_Foot", "R_Ankle"],
        }.items():
            body_id = -1
            explicit_name = explicit.get(side)
            if explicit_name:
                body_id = self._body_name_to_id(explicit_name)

            if body_id < 0:
                for joint_name in joint_candidates:
                    body_name = self.valid_joint_mapping.get(joint_name)
                    if body_name:
                        body_id = self._body_name_to_id(body_name)
                        if body_id >= 0:
                            break

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

        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(self.robot_model.geom_dataid[geom_id])
            if mesh_id < 0:
                points_local = np.zeros((1, 3), dtype=float)
            else:
                vert_start = int(self.robot_model.mesh_vertadr[mesh_id])
                vert_count = int(self.robot_model.mesh_vertnum[mesh_id])
                points_local = np.asarray(
                    self.robot_model.mesh_vert[vert_start : vert_start + vert_count],
                    dtype=float,
                )
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
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

    def _compute_foot_ground_lock_series(
        self,
        motion: np.ndarray,
        ground_lock_specs: List[Dict[str, Any]],
        contact_band: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute support points from all configured visible/collision foot geoms."""
        num_frames = len(motion)
        num_feet = len(ground_lock_specs)
        positions = np.zeros((num_frames, num_feet, 3), dtype=float)
        min_z = np.zeros((num_frames, num_feet), dtype=float)

        for frame_idx, qpos in enumerate(motion):
            self.robot_data.qpos[:] = qpos
            mujoco.mj_forward(self.robot_model, self.robot_data)

            for foot_idx, spec in enumerate(ground_lock_specs):
                world_points = []
                for geom_spec in spec["geom_specs"]:
                    body_id = geom_spec["body_id"]
                    body_pos = np.asarray(self.robot_data.xpos[body_id], dtype=float)
                    body_rot = np.asarray(self.robot_data.xmat[body_id], dtype=float).reshape(3, 3)
                    world_points.append(geom_spec["sample_points"] @ body_rot.T + body_pos)
                world_points = np.vstack(world_points)
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
        preserve_root_z = bool(cfg.get("preserve_root_z", False))

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

                if preserve_root_z:
                    correction[2] = 0.0

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

    def _estimate_base_orientation_from_joints(self, joints: np.ndarray, last_quat: np.ndarray | None = None) -> np.ndarray | None:
        """
        Estimate a base orientation from joint positions.

        Uses pelvis/hips/spine to build an approximate body frame:
        - up: pelvis -> spine1
        - right: left_hip -> right_hip
        - forward: right x up
        Returns quaternion in xyzw order.
        
        Args:
            joints: Joint positions array.
            last_quat: Quaternion from previous frame to ensure continuity (xyzw).
        """
        if joints.shape[0] < 4:
            return None

        pelvis_name = self.base_orientation_config.get("pelvis", "Pelvis")
        left_hip_name = self.base_orientation_config.get("left_hip", "L_Hip")
        right_hip_name = self.base_orientation_config.get("right_hip", "R_Hip")
        spine_name = self.base_orientation_config.get("spine", "Spine1")

        pelvis_idx = self.smplx_joint_indices.get(pelvis_name, 0)
        left_hip_idx = self.smplx_joint_indices.get(left_hip_name, 1)
        right_hip_idx = self.smplx_joint_indices.get(right_hip_name, 2)
        spine_idx = self.smplx_joint_indices.get(spine_name, 3)

        max_required_idx = max(pelvis_idx, left_hip_idx, right_hip_idx, spine_idx)
        if joints.shape[0] <= max_required_idx:
            return None

        pelvis = joints[pelvis_idx]
        left_hip = joints[left_hip_idx]
        right_hip = joints[right_hip_idx]
        spine1 = joints[spine_idx]

        up = spine1 - pelvis
        right = right_hip - left_hip

        up_norm = np.linalg.norm(up)
        right_norm = np.linalg.norm(right)
        if up_norm < 1e-8 or right_norm < 1e-8:
            return None

        up = up / up_norm
        right = right / right_norm
        
        # Build forward using right-hand rule
        forward = np.cross(up, right)
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-8:
            return None
        forward = forward / forward_norm

        # Re-orthogonalize right to ensure perfect orthogonality
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)

        # Rotation matrix with body axes in world coordinates.
        # Convention: X=forward, Y=right, Z=up
        rot = np.column_stack([forward, right, up])
        
        # Verify right-handed
        det = np.linalg.det(rot)
        if det < 0:
            forward = -forward
            right = -right
            rot = np.column_stack([forward, right, up])
        
        current_quat = Rotation.from_matrix(rot).as_quat() # xyzw
        
        # Ensure continuity if last quaternion is provided
        if last_quat is not None:
            # Check dot product to see if we need to flip sign
            dot = np.dot(current_quat, last_quat)
            if dot < 0:
                current_quat = -current_quat
                
        return current_quat

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

    def _visualize_trajectory(self, trajectory: np.ndarray, scaled_terrain: trimesh.Trimesh):
        """
        Visualize the SMPLX trajectory using matplotlib 3D animation with terrain mesh.
        
        Args:
            trajectory: Processed trajectory of shape (T, J, 3) where T is frames, J is joints
                       Coordinates are assumed to be in +Z up convention (already transformed)
            scaled_terrain: Scaled terrain mesh
        """
        print(f"Visualizing trajectory with shape: {trajectory.shape}")
        print(f"Terrain mesh: {len(scaled_terrain.vertices)} vertices, {len(scaled_terrain.faces)} faces")
        print("Coordinate system: +Z is up")
        
        num_frames, num_joints, _ = trajectory.shape
        
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
        ax.set_title('SMPLX Trajectory with Terrain Visualization')
        
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
        
        # Initialize scatter plot for joints
        scatter = ax.scatter([], [], [], c='blue', marker='o', s=50, alpha=0.9, edgecolors='black', linewidths=0.5)
        
        # Add text for frame counter
        frame_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
        
        # Define SMPLX skeleton connections (approximate body structure)
        # These are indices in the standard SMPLX joint ordering
        skeleton_connections = [
            # Spine
            (0, 3),   # Pelvis -> Spine1
            (3, 6),   # Spine1 -> Spine2
            (6, 9),   # Spine2 -> Spine3
            (9, 12),  # Spine3 -> Neck
            (12, 15), # Neck -> Head
            
            # Left leg
            (0, 1),   # Pelvis -> L_Hip
            (1, 4),   # L_Hip -> L_Knee
            (4, 7),   # L_Knee -> L_Ankle
            (7, 10),  # L_Ankle -> L_Foot
            
            # Right leg
            (0, 2),   # Pelvis -> R_Hip
            (2, 5),   # R_Hip -> R_Knee
            (5, 8),   # R_Knee -> R_Ankle
            (8, 11),  # R_Ankle -> R_Foot
            
            # Left arm
            (9, 13),  # Spine3 -> L_Collar
            (13, 16), # L_Collar -> L_Shoulder
            (16, 18), # L_Shoulder -> L_Elbow
            (18, 20), # L_Elbow -> L_Wrist
            
            # Right arm
            (9, 14),  # Spine3 -> R_Collar
            (14, 17), # R_Collar -> R_Shoulder
            (17, 19), # R_Shoulder -> R_Elbow
            (19, 21), # R_Elbow -> R_Wrist
        ]
        
        # Filter connections to only include joints we have
        valid_connections = [(i, j) for i, j in skeleton_connections if i < num_joints and j < num_joints]
        
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
            # Get joint positions for current frame
            joints = trajectory[frame_idx]  # Shape: (J, 3)
            
            # Update scatter plot
            xs, ys, zs = joints[:, 0], joints[:, 1], joints[:, 2]
            scatter._offsets3d = (xs, ys, zs)
            
            # Update frame counter
            frame_text.set_text(f'Frame: {frame_idx + 1}/{num_frames}')
            
            # Update skeleton lines
            for line, (i, j) in zip(lines, valid_connections):
                x_data = [joints[i, 0], joints[j, 0]]
                y_data = [joints[i, 1], joints[j, 1]]
                z_data = [joints[i, 2], joints[j, 2]]
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
