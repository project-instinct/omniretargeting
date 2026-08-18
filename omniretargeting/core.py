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
from .utils import compute_mesh_height_at_point, detect_robot_height, load_robot_urdf_with_floating_base


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
        base_orientation: Optional[Dict[str, str]] = None,
        retargeting: Optional[Dict[str, Any]] = None,
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
        """
        self.robot_urdf_path = Path(robot_urdf_path)
        self.terrain_mesh_path = Path(terrain_mesh_path)
        self.joint_mapping = joint_mapping

        if source_target_names is None:
            source_target_names = list(joint_mapping.keys())
        self.source_target_names = source_target_names

        # Optional per-robot configuration with safe defaults.

        self.base_orientation_config = dict(base_orientation or {})
        self.retargeting_config = dict(retargeting or {})

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
                    f"Available targets: {list(self.source_target_indices.keys())}"
                )

        if len(self.mapped_source_target_indices) == 0:
            raise ValueError("No valid source target mappings found. Please check your joint_mapping dictionary.")
        
        # Store the filtered mapping (only valid source targets) for use in retargeter.
        self.valid_source_to_robot_link_mapping = {
            name: joint_mapping[name] for name in self.valid_source_target_names
        }

        # Load robot URDF
        self.robot_urdf = yourdfpy.URDF.load(str(robot_urdf_path), load_meshes=True)
        self.robot_model = load_robot_urdf_with_floating_base(str(robot_urdf_path))
        self.robot_data = mujoco.MjData(self.robot_model)

        # Load terrain mesh
        self.terrain_mesh = trimesh.load(str(terrain_mesh_path))

        # Detect robot height if not provided
        if robot_height is None:
            self.robot_height = detect_robot_height(self.robot_model, self.robot_data)
        else:
            self.robot_height = robot_height

        # Initialize retargeting components
        self._setup_retargeting_components()


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
        enable_terrain_scaling: bool | None = None,
        enable_scene_scaling: bool | None = None,
    ) -> Tuple[float, np.ndarray]:
        """
        Retarget a complete source motion and return ``(source_to_robot_scale, robot_motion)``.

        Args:
            enable_scene_scaling: When True, scale terrain, human trajectory, and objects.
            enable_terrain_scaling: Deprecated alias for enable_scene_scaling.

        The computed scalar is the robot/source height ratio. When enabled, batch mode
        applies it to source positions, root translations, object points, and terrain mesh.
        """
        # Handle backward compatibility
        if enable_terrain_scaling is not None and enable_scene_scaling is not None:
            raise ValueError("Cannot specify both enable_terrain_scaling and enable_scene_scaling.")
        
        if enable_terrain_scaling is not None:
            import warnings
            warnings.warn(
                "enable_terrain_scaling is deprecated, use enable_scene_scaling instead",
                DeprecationWarning,
                stacklevel=2
            )
            enable_scene_scaling = enable_terrain_scaling
        
        if enable_scene_scaling is None:
            enable_scene_scaling = False
        
        motion_data = self._coerce_motion_data(motion)
        if base_orientations is not None or base_translations is not None:
            motion_data = MotionData(
                positions=motion_data.positions,
                target_names=motion_data.target_names,
                root_orientations=base_orientations if base_orientations is not None else motion_data.root_orientations,
                root_translations=base_translations if base_translations is not None else motion_data.root_translations,
                framerate=framerate if framerate is not None else motion_data.framerate,
                source_height=motion_data.source_height,
                object_points=motion_data.object_points,
                metadata=dict(motion_data.metadata),
            )
        elif framerate is not None and motion_data.framerate is None:
            motion_data.framerate = framerate

        source_to_robot_scale = self._resolve_source_to_robot_scale(
            apply_source_to_robot_scaling=enable_scene_scaling,
            source_height=motion_data.source_height,
        )
        scaled_terrain = self._scale_terrain_mesh(source_to_robot_scale) if enable_scene_scaling else self.terrain_mesh.copy()

        scaled_motion_data = MotionData(
            positions=motion_data.positions * source_to_robot_scale if enable_scene_scaling else motion_data.positions,
            target_names=motion_data.target_names,
            root_orientations=motion_data.root_orientations,
            root_translations=(
                motion_data.root_translations * source_to_robot_scale
                if enable_scene_scaling and motion_data.root_translations is not None
                else motion_data.root_translations
            ),
            framerate=motion_data.framerate,
            source_height=motion_data.source_height,
            object_points=(
                motion_data.object_points * source_to_robot_scale
                if enable_scene_scaling and motion_data.object_points is not None
                else motion_data.object_points
            ),
            metadata=dict(motion_data.metadata),
        )
        self._attach_contact_targets(
            scaled_motion_data,
            scaled_terrain,
            source_to_robot_scale if enable_scene_scaling else 1.0,
        )
        scaled_motion_data = self._match_skeleton_to_robot(scaled_motion_data)

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

        # Penetration handling mode:
        # - "hard_constraint": non-penetration as QP constraints (single tolerance)
        # - "hard_constraint_slack": QP constraints with TopoRetarget slack
        #   variables (soft tolerance + hard backstop, see penetration_slack)
        # - "xyz_nudge": skip QP constraints; batch post-processing corrects contact
        penetration_resolver = self.retargeting_config.get("penetration_resolver", "hard_constraint")
        valid_resolvers = ("hard_constraint", "hard_constraint_slack", "xyz_nudge")
        if penetration_resolver not in valid_resolvers:
            raise ValueError(
                f"Unknown penetration_resolver '{penetration_resolver}'. "
                f"Expected one of {valid_resolvers}."
            )

        retargeter = GenericInteractionRetargeter(
            self.robot_model,
            self.robot_data,
            scaled_terrain,
            self.valid_source_to_robot_link_mapping,
            self.robot_height,
            collision_detection_threshold=float(self.retargeting_config.get("collision_detection_threshold", 0.1)),
            terrain_sample_points=int(self.retargeting_config.get("terrain_sample_points", 100)),
            terrain_sample_seed=self.retargeting_config.get("terrain_sample_seed"),
            link_orientation=self.retargeting_config.get("link_orientation"),
            scene_contact=self.retargeting_config.get("scene_contact"),
            foot_penetration=self.retargeting_config.get("foot_penetration"),
            smooth_weight=float(self.retargeting_config.get("smooth_weight", 0.2)),
            source_target_names=self.valid_source_target_names,
            replace_cylinders_with_capsules=bool(self.retargeting_config.get("replace_cylinders_with_capsules", False)),
            hard_penetration_constraint=penetration_resolver in ("hard_constraint", "hard_constraint_slack"),
            joint_regularization_boost=self.retargeting_config.get("joint_regularization_boost"),
            laplacian_edge_weighting=self.retargeting_config.get("laplacian_edge_weighting", "uniform"),
            laplacian_distance_decay=float(self.retargeting_config.get("laplacian_distance_decay", 30.0)),
            bone_direction=self.retargeting_config.get("bone_direction"),
            penetration_slack=(
                self.retargeting_config.get("penetration_slack")
                if penetration_resolver == "hard_constraint_slack"
                else None
            ),
            base_position_tracking_weight=float(
                self.retargeting_config.get("base_position_tracking_weight", 0.0)
            ),
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

    def _estimate_base_orientation_from_joints(
        self,
        joints: np.ndarray,
        last_quat: np.ndarray | None = None,
    ) -> np.ndarray | None:
        if joints.shape[0] == 0:
            return None

        pelvis_name = self.base_orientation_config.get("pelvis")
        left_hip_name = self.base_orientation_config.get("left_hip")
        right_hip_name = self.base_orientation_config.get("right_hip")
        spine_name = self.base_orientation_config.get("spine")

        if not all([pelvis_name, left_hip_name, right_hip_name, spine_name]):
            raise ValueError(
                "base_orientation_config must specify 'pelvis', 'left_hip', 'right_hip', and 'spine' joint names. "
                "Example: { 'pelvis': 'Pelvis', 'left_hip': 'L_Hip', 'right_hip': 'R_Hip', 'spine': 'Spine1' }. "
                f"Current config: {self.base_orientation_config}"
            )

        joint_indices = {name: idx for idx, name in enumerate(self.source_target_names)}
        pelvis_idx = joint_indices.get(pelvis_name)
        left_hip_idx = joint_indices.get(left_hip_name)
        right_hip_idx = joint_indices.get(right_hip_name)
        spine_idx = joint_indices.get(spine_name)

        missing = []
        if pelvis_idx is None:
            missing.append(f"pelvis='{pelvis_name}'")
        if left_hip_idx is None:
            missing.append(f"left_hip='{left_hip_name}'")
        if right_hip_idx is None:
            missing.append(f"right_hip='{right_hip_name}'")
        if spine_idx is None:
            missing.append(f"spine='{spine_name}'")
        if missing:
            raise ValueError(
                f"Source target names {self.source_target_names} missing required joints: {', '.join(missing)}. "
                "Update base_orientation_config to match the source skeleton's joint naming."
            )

        max_required_idx = max(pelvis_idx, left_hip_idx, right_hip_idx, spine_idx)
        if joints.shape[0] <= max_required_idx:
            return None

        pelvis = joints[pelvis_idx]
        left_hip = joints[left_hip_idx]
        right_hip = joints[right_hip_idx]
        spine = joints[spine_idx]

        up = spine - pelvis
        right = right_hip - left_hip

        up_norm = np.linalg.norm(up)
        right_norm = np.linalg.norm(right)
        if up_norm < 1e-8 or right_norm < 1e-8:
            return None

        up = up / up_norm
        right = right / right_norm
        forward = np.cross(up, right)
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-8:
            return None
        forward = forward / forward_norm

        right = np.cross(up, forward)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-8:
            return None
        right = right / right_norm

        rot = np.column_stack([forward, right, up])
        if np.linalg.det(rot) < 0:
            forward = -forward
            right = -right
            rot = np.column_stack([forward, right, up])

        quat_wxyz = Rotation.from_matrix(rot).as_quat(scalar_first=True)
        if last_quat is not None and np.dot(quat_wxyz, last_quat) < 0:
            quat_wxyz = -quat_wxyz
        return quat_wxyz

    @staticmethod
    def _source_effector_lengths(motion_data: MotionData, scale: float) -> Dict[str, float]:
        """Longest extent of each effector's own contact surface, in metres.

        Used to express the proximity gate as a fraction of the thing being
        tested rather than as an absolute distance that means different things
        for different subjects and different scene scales.
        """
        ids = motion_data.metadata.get("foot_sole_vertex_ids")
        template = motion_data.metadata.get("foot_sole_template_vertices")
        lengths: Dict[str, float] = {}
        if not ids or template is None:
            return lengths
        template = np.asarray(template, dtype=float)
        for name, vertex_ids in ids.items():
            try:
                band = template[np.asarray(vertex_ids, dtype=int)]
            except (IndexError, TypeError):
                continue
            if band.shape[0] < 2:
                continue
            extent = band.max(axis=0) - band.min(axis=0)
            lengths[name] = float(np.max(extent) * scale)
        return lengths

    def _match_skeleton_to_robot(self, motion_data: MotionData) -> MotionData:
        """Bring the source skeleton to robot proportions about its contacts.

        The Laplacian tracks absolute source joint positions, so it pulls the
        pelvis toward wherever the source's pelvis was.  When the scene is left
        at its captured scale and the robot is shorter than the subject, that
        target is unreachable: the pelvis cannot be at human height and the feet
        on the real floor at the same time.  What happened instead was two
        regimes -- the contact task won on contact frames and dragged the base
        down, the Laplacian won between them and pulled it back up, eighteen
        centimetres apart and switching ten times across one clip.

        Rescaling the source skeleton to the robot's proportions removes the
        conflict rather than balancing it.  The scale is applied about the
        contacting foot rather than about the pelvis or the origin, which is
        what keeps the feet on the real surface while the pelvis descends by the
        leg-length difference.  Between contacts the last anchor is carried
        forward, so the rescaling never jumps.

        Off unless configured, and a no-op when the scene has been scaled: there
        the whole world was already brought to the robot and the source targets
        are reachable as they stand.
        """
        config = self.retargeting_config.get("skeleton_matching") or {}
        if not config.get("enabled", False):
            return motion_data

        names = motion_data.target_names or []
        anchors = motion_data.metadata.get("contact_targets")
        if not anchors:
            print("Warning: skeleton_matching is enabled but no scene contacts were inferred")
            return motion_data

        scale = self._derive_skeleton_scale(motion_data, config)
        if scale is None:
            return motion_data

        positions = np.asarray(motion_data.positions, dtype=float)
        surface, source_point = self._contact_anchor_series(
            positions, names, anchors, motion_data.metadata.get("foot_sole_centres")
        )

        # Rescale about the source's own contacting point, then place that point
        # on the surface anchor.  Rescaling about the surface point alone leaves
        # the foot wherever the source had it, and a source reconstructed from
        # video floats several centimetres above its own scan, so the Laplacian
        # would still be asking for a foot the contact task has to drag down.
        rescaled = surface[:, None, :] + scale * (positions - source_point[:, None, :])
        root_translations = motion_data.root_translations
        if root_translations is not None:
            root_translations = surface + scale * (
                np.asarray(root_translations, dtype=float) - source_point
            )

        print(
            f"[skeleton_matching] source skeleton scaled by {scale:.4f} about its contacts "
            f"(pelvis-to-sole span: source "
            f"{self._source_leg_length(positions, names, motion_data.metadata.get('foot_sole_centres')):.3f} m, "
            f"robot {self._robot_leg_length():.3f} m)"
        )
        return MotionData(
            positions=rescaled,
            target_names=motion_data.target_names,
            root_orientations=motion_data.root_orientations,
            root_translations=root_translations,
            framerate=motion_data.framerate,
            source_height=motion_data.source_height,
            object_points=motion_data.object_points,
            object_mesh=motion_data.object_mesh,
            metadata=dict(motion_data.metadata),
        )

    # The leg span is taken at a high percentile rather than at the median.
    # Stance is where the source leg is most extended, and stance is exactly
    # when the sole is pinned, so a median taken over a clip full of bent-leg
    # swing frames calibrates the scale on the wrong part of the motion and
    # leaves the stance demand outside what the robot can reach.
    _LEG_SPAN_PERCENTILE = 90.0

    @staticmethod
    def _source_leg_length(
        positions: np.ndarray,
        names: List[str],
        sole_centres: Optional[Dict[str, np.ndarray]] = None,
    ) -> float:
        """Pelvis-to-sole distance in the source, in metres.

        Measured to the sole rather than to the ankle joint.  The contact task
        pins the sole, so the sole is the point whose reach decides whether the
        robot can stand where the source stood.  The two differ by more than
        they look: the human's ankle sits about 8 cm above its own sole while
        this robot's sits 3.5 cm above, so a scale derived from ankle spans
        leaves several centimetres the leg cannot make up once the sole is
        planted -- which the base then absorbs by dropping, but only on the
        frames where a contact is active.
        """
        try:
            pelvis = names.index("Pelvis")
        except ValueError:
            return 0.0

        targets = []
        if sole_centres:
            targets = [np.asarray(v, dtype=float) for v in sole_centres.values()]
        if not targets:
            targets = [
                positions[:, names.index(a)] for a in ("L_Ankle", "R_Ankle") if a in names
            ]
        if not targets:
            return 0.0

        lengths = [np.linalg.norm(positions[:, pelvis] - t, axis=1) for t in targets]
        return float(
            np.percentile(np.concatenate(lengths), OmniRetargeter._LEG_SPAN_PERCENTILE)
        )

    def _robot_leg_length(self) -> float:
        """Pelvis-to-ankle distance on the robot at its nominal configuration."""
        data = mujoco.MjData(self.robot_model)
        data.qpos[:] = self.robot_model.qpos0
        mujoco.mj_kinematics(self.robot_model, data)

        # Mapping entries carry a link name alone in the older schema and a
        # dict with an offset in the newer one.
        def link_of(name):
            entry = self.valid_source_to_robot_link_mapping.get(name)
            if isinstance(entry, dict):
                return entry.get("robot_link")
            return entry

        pelvis_link = link_of("Pelvis")
        ankle_links = [link_of(n) for n in ("L_Ankle", "R_Ankle")]
        pelvis_id = (
            mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, pelvis_link)
            if pelvis_link
            else -1
        )
        if pelvis_id < 0:
            return 0.0
        # Measure to the sole, matching the source side and matching the point
        # the contact task pins.  The sole is read off the foot's own collision
        # geoms rather than assumed, so this holds for feet of any shape.
        from omniretargeting.foot_frames import (
            derive_sole_from_points,
            sample_body_collision_points,
        )

        lengths = []
        for link in ankle_links:
            if not link:
                continue
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, link)
            if body_id < 0:
                continue
            sole_offset = np.zeros(3)
            try:
                points = sample_body_collision_points(self.robot_model, body_id)
                sole_offset = derive_sole_from_points(points).reference_point
            except (ValueError, ImportError) as exc:
                print(
                    f"Warning: could not derive the sole of {link!r} ({exc}); "
                    "measuring the leg to the ankle instead"
                )
            sole_world = data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ sole_offset
            lengths.append(float(np.linalg.norm(sole_world - data.xpos[pelvis_id])))
        return float(np.median(lengths)) if lengths else 0.0

    def _derive_skeleton_scale(self, motion_data: MotionData, config: dict) -> Optional[float]:
        """Ratio of robot to source leg length, measured rather than declared.

        Measured from the segment that decides whether the feet reach the
        ground, not from overall height: two robots of the same height can have
        very different leg-to-torso proportions, and it is the leg that has to
        span the gap.
        """
        override = config.get("scale")
        if override is not None:
            return float(override)

        source_leg = self._source_leg_length(
            np.asarray(motion_data.positions, dtype=float),
            motion_data.target_names or [],
            motion_data.metadata.get("foot_sole_centres"),
        )
        robot_leg = self._robot_leg_length()
        if source_leg <= 1e-6 or robot_leg <= 1e-6:
            print(
                "Warning: skeleton_matching could not measure leg lengths "
                f"(source {source_leg:.4f} m, robot {robot_leg:.4f} m); leaving the skeleton alone"
            )
            return None
        return robot_leg / source_leg

    @staticmethod
    def _contact_anchor_series(
        positions: np.ndarray,
        names: List[str],
        anchors: Dict[str, np.ndarray],
        sole_centres: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Per-frame pair of points that pins the rescaling.

        Returns ``(surface, source_point)``: where the contact is on the scene,
        and where the source had the corresponding part of itself.  The
        rescaling maps the second onto the first.

        A contacting foot when there is one, otherwise the last one carried
        forward, and before the first contact the first one carried backward.
        Carrying forward is what keeps the rescaling continuous; recentring on
        the pelvis between contacts would reintroduce the pull toward human
        height that this is meant to remove.
        """
        n_frames = positions.shape[0]
        series = np.full((n_frames, 3), np.nan)

        for name, target in anchors.items():
            target = np.asarray(target, dtype=float)
            if target.shape[0] != n_frames:
                continue
            usable = np.isfinite(target[:, 0])
            fill = usable & ~np.isfinite(series[:, 0])
            series[fill] = target[fill]

        source_series = np.full((n_frames, 3), np.nan)
        if sole_centres:
            for name, target in anchors.items():
                centres = sole_centres.get(name)
                if centres is None:
                    continue
                centres = np.asarray(centres, dtype=float)
                target = np.asarray(target, dtype=float)
                if centres.shape[0] != n_frames or target.shape[0] != n_frames:
                    continue
                usable = np.isfinite(target[:, 0]) & ~np.isfinite(source_series[:, 0])
                source_series[usable] = centres[usable]

        finite = np.flatnonzero(np.isfinite(series[:, 0]) & np.isfinite(source_series[:, 0]))
        if finite.size == 0:
            # Nothing to pin to; rescale about the source's own feet so the body
            # at least keeps them still relative to itself.
            foot_names = [n for n in ("L_Ankle", "R_Ankle") if n in names]
            if foot_names:
                idx = [names.index(n) for n in foot_names]
                own = positions[:, idx, :].mean(axis=1)
            else:
                own = positions.mean(axis=1)
            return own, own

        # Hold the nearest known pair on either side of a gap, so the mapping
        # never jumps when contact starts or ends.
        for axis in range(3):
            series[:, axis] = np.interp(np.arange(n_frames), finite, series[finite, axis])
            source_series[:, axis] = np.interp(
                np.arange(n_frames), finite, source_series[finite, axis]
            )
        return series, source_series

    def _attach_contact_targets(
        self,
        motion_data: MotionData,
        terrain,
        scale: float,
    ) -> None:
        """Detect the source's scene contacts and anchor them, in place.

        Runs once per motion rather than per frame: anchoring a contact needs
        the whole interval, which a frame-at-a-time pass cannot see.
        """
        config = self.retargeting_config.get("scene_contact") or {}
        if not config.get("enabled", False):
            return

        centres = motion_data.metadata.get("foot_sole_centres")
        if not centres:
            print("Warning: scene_contact is enabled but the source exposes no sole centres")
            return

        from omniretargeting.scene_contact import ContactSpec, infer_contact_targets

        defaults = ContactSpec()
        spec = ContactSpec(
            enter_gap_fraction=float(config.get("enter_gap_fraction", defaults.enter_gap_fraction)),
            leave_gap_fraction=float(config.get("leave_gap_fraction", defaults.leave_gap_fraction)),
            gap_floor=float(config.get("gap_floor", defaults.gap_floor)),
            enter_speed_fraction=float(
                config.get("enter_speed_fraction", defaults.enter_speed_fraction)
            ),
            leave_speed_fraction=float(
                config.get("leave_speed_fraction", defaults.leave_speed_fraction)
            ),
            min_frames=int(config.get("min_frames", defaults.min_frames)),
            anchor_frames=int(config.get("anchor_frames", defaults.anchor_frames)),
            blend_frames=int(config.get("blend_frames", defaults.blend_frames)),
            bias_percentile=float(config.get("bias_percentile", defaults.bias_percentile)),
            estimate_bias=bool(config.get("estimate_bias", defaults.estimate_bias)),
        )
        scaled_centres = {side: np.asarray(pts, dtype=float) * scale for side, pts in centres.items()}

        # The gates are fractions of the source's own geometry, so the detector
        # needs to be told what that geometry is rather than carrying absolute
        # numbers that only suit one subject in one scene.
        positions = np.asarray(motion_data.positions, dtype=float) * scale
        names = motion_data.target_names or []
        leg_length = self._source_leg_length(positions, names, scaled_centres)
        effector_lengths = self._source_effector_lengths(motion_data, scale)

        clouds = motion_data.metadata.get("foot_sole_points") or {}
        scaled_clouds = {
            side: np.asarray(pts, dtype=float) * scale
            for side, pts in clouds.items()
            if side in scaled_centres
        }

        targets, normals, weights, stats = infer_contact_targets(
            scaled_centres,
            terrain,
            motion_data.framerate,
            spec,
            effector_lengths=effector_lengths,
            leg_length=leg_length,
            effector_clouds=scaled_clouds or None,
        )
        motion_data.metadata["contact_targets"] = targets
        motion_data.metadata["contact_normals"] = normals
        motion_data.metadata["contact_weights"] = weights
        for name, entry in stats.items():
            print(
                f"[scene_contact] {name}: {entry['contact_frames']} contact frames "
                f"in {entry['runs']} runs; source floats {entry['float_bias'] * 100:.2f} cm "
                f"above the scan, gates {entry['enter_gap'] * 100:.2f}/"
                f"{entry['leave_gap'] * 100:.2f} cm, median confidence "
                f"{entry['median_confidence']:.2f}"
            )

    def retarget_frame(self, frame: MotionFrame | np.ndarray, state: RetargetingStreamState) -> np.ndarray:
        positions = frame.positions if isinstance(frame, MotionFrame) else frame
        root_orientation = frame.root_orientation if isinstance(frame, MotionFrame) else None
        root_translation = frame.root_translation if isinstance(frame, MotionFrame) else None
        source_positions = positions
        q_init = state.q_init

        estimated_quat_wxyz = self._estimate_base_orientation_from_joints(
            source_positions,
            state.last_estimated_quat,
        )

        if state.frame_idx == 0:
            if root_translation is not None:
                q_init[:3] = root_translation
            else:
                q_init[:3] = source_positions[0]
            if root_orientation is not None:
                q_init[3:7] = root_orientation
            elif estimated_quat_wxyz is not None:
                q_init[3:7] = estimated_quat_wxyz

        mapped_source_targets = self._extract_mapped_source_targets(source_positions)

        # Which orientation the base tracks.  The default estimates a frame
        # from four joint positions, which is only as good as that
        # construction: measured against SMPL-X it leaves the pelvis about
        # fourteen degrees out, and the waist then bends to its limit
        # compensating for the error.  Sources that carry a real root
        # orientation can ask for it instead; frame zero already trusts it for
        # initialization, so trusting it as the target too is consistent.
        # Left as the estimate by default because profiles without a reliable
        # root orientation depend on the inferred frame.
        prefer_root = str(
            (getattr(self, "base_orientation_config", None) or {}).get("target", "estimate")
        ).lower() == "root"

        target_quat_wxyz = None
        if prefer_root and root_orientation is not None:
            target_quat_wxyz = np.asarray(root_orientation, dtype=float)
            # Keep the hemisphere continuous with the previous frame, as the
            # estimator does, so the base orientation cost does not see a sign
            # flip as a large error.
            if state.last_estimated_quat is not None and np.dot(
                target_quat_wxyz, state.last_estimated_quat
            ) < 0:
                target_quat_wxyz = -target_quat_wxyz
            state.last_estimated_quat = target_quat_wxyz
        elif estimated_quat_wxyz is not None:
            target_quat_wxyz = estimated_quat_wxyz
            state.last_estimated_quat = estimated_quat_wxyz

        # Extract object points if present
        object_points = frame.object_points if isinstance(frame, MotionFrame) else None

        solver_max_iter = self.retargeting_config.get("solver_max_iter")
        if isinstance(solver_max_iter, dict):
            max_iter = solver_max_iter.get("first_frame", 200) if state.frame_idx == 0 else solver_max_iter.get("subsequent", 50)
        elif solver_max_iter is not None:
            max_iter = int(solver_max_iter)
        else:
            max_iter = 10

        link_orientation_targets = (
            frame.metadata.get("link_orientations") if isinstance(frame, MotionFrame) else None
        )
        contact_targets = (
            frame.metadata.get("contact_targets") if isinstance(frame, MotionFrame) else None
        )
        contact_normals = (
            frame.metadata.get("contact_normals") if isinstance(frame, MotionFrame) else None
        )
        contact_weights = (
            frame.metadata.get("contact_weights") if isinstance(frame, MotionFrame) else None
        )

        q_opt = state.retargeter.retarget_frame(
            mapped_source_targets,
            q_init,
            max_iter=max_iter,
            q_last=state.q_last,
            target_base_orientation=target_quat_wxyz,
            object_points=object_points,
            root_translation=root_translation,
            link_orientation_targets=link_orientation_targets,
            contact_targets=contact_targets,
            contact_normals=contact_normals,
            contact_weights=contact_weights,
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
            # Seconds, not frames: the same motion resampled to a different rate
            # should get the same lift curve.  Falls back to z_smoothing_window
            # when the framerate is not known.
            "lift_smoothing_time": 0.25,
            # Ceiling on the lift, as a fraction of the robot's own leg.  The
            # lift is there to clear a residue of a few millimetres left by the
            # solve; a demand far beyond that is not penetration to be cleared
            # but a frame the solver got wrong, usually because the motion asks
            # the robot to reach somewhere its legs do not go.  Absorbing it
            # silently floats the whole body off the source, and the smoothing
            # spreads that across neighbouring frames.  Better to clamp, say so,
            # and let the clip be judged.
            "max_lift_fraction": 0.15,
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
        if framerate and framerate > 0:
            lift_window = max(2.0, float(cfg["lift_smoothing_time"]) * float(framerate))
        else:
            lift_window = float(z_window)

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

        stabilized, wall_contact_mask = self._apply_surface_collision_corrections(
            stabilized,
            terrain_mesh,
            foot_specs,
            cfg,
        )

        # Lifting the body out of the ground is done once, here, after every
        # correction that could move a sole in x, y or along a surface normal.
        # An earlier pass followed by top-ups had to re-measure and re-add after
        # each of those corrections, and each of those additions was a raw
        # per-frame number: together they put a step of vertical noise into the
        # base that the retargeting itself never produced.
        #
        # Measured once at the final pose, the requirement is exact.  A vertical
        # translation moves every probe by the same amount and a downward ray
        # from the raised probe lands on the same triangle, so clearance after a
        # lift of l is clearance before plus l -- no iteration needed.
        probe_clearances = self._foot_support_clearances(stabilized, terrain_mesh, foot_specs)
        # A wall contact is exempted from being lifted toward its target
        # clearance, since lifting does not move a foot away from a vertical
        # surface.  It is not exempted from being lifted out of the floor:
        # a foot may be against a wall and through the ground at the same time,
        # and leaving it there is what let penetration survive this pass.
        lift_clearances = np.where(~wall_contact_mask, probe_clearances, np.minimum(probe_clearances, 0.0))
        min_clearance = np.min(lift_clearances, axis=1)
        required_lift = np.maximum(clearance_target - min_clearance, 0.0)
        required_lift[~np.isfinite(min_clearance)] = 0.0

        max_lift = float(cfg["max_lift_fraction"]) * self._robot_leg_length()
        beyond = required_lift > max_lift
        if max_lift > 0.0 and np.any(beyond):
            frames = np.flatnonzero(beyond)
            print(
                f"Foot stabilization: {len(frames)} frame(s) asked to be lifted more than "
                f"{max_lift * 100:.1f} cm (up to {required_lift.max() * 100:.1f} cm at frame "
                f"{int(np.argmax(required_lift))}); clamped.  These frames are left penetrating: "
                "the solve did not reach the surface there, and lifting the body far enough to "
                "clear it would move the whole motion off the source instead.  Frames: "
                f"{frames[:20].tolist()}{' ...' if len(frames) > 20 else ''}"
            )
            required_lift = np.minimum(required_lift, max_lift)

        base_lift = self._minimal_smooth_lift(required_lift, lift_window)
        stabilized[:, 2] += base_lift

        if np.any(base_lift > 0) or np.any(active):
            lifted = required_lift > 1e-6
            print(
                "Applied foot stabilization: "
                f"max base lift={base_lift.max():.4f}m "
                f"(required {required_lift.max():.4f}m on "
                f"{int(lifted.sum())}/{len(required_lift)} frames, "
                f"median where needed {np.median(required_lift[lifted]) if lifted.any() else 0.0:.4f}m), "
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
                # The name search can land on a link that carries no collision
                # geometry at all -- on the G1 it matches the ankle pitch link,
                # which has none, while the sole cylinders live one link further
                # down on the ankle roll link.  Falling back to a single point
                # at that link's origin silently reduces the whole stabilisation
                # to one notional point and lets the actual sole pass through
                # the ground.  Walk down the chain to the nearest descendant
                # that has something to stand on.
                descendant = self._nearest_collision_descendant(body_id)
                if descendant is not None and descendant != body_id:
                    resolved = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, descendant)
                    original = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                    print(
                        f"[foot_stabilization] {side}: {original!r} carries no collision geometry; "
                        f"using {resolved!r} instead"
                    )
                    body_id = descendant
                    sample_points = self._collect_body_contact_points(body_id)
            if sample_points.size == 0:
                print(
                    f"[foot_stabilization] {side}: no collision geometry found on the foot chain; "
                    "stabilisation will act on the link origin only"
                )
                sample_points = np.zeros((1, 3), dtype=float)
            specs.append({
                "side": side,
                "body_id": body_id,
                "body_name": mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                "sample_points": sample_points,
                "collision_points": self._select_collision_probe_points(sample_points),
            })
        return specs

    def _declared_effector_bodies(self, role: Optional[str] = None) -> Dict[str, str]:
        """Foot and torso bodies as the profile declares them.

        Reads the shared ``retargeting.effectors`` block first, then falls back
        to whatever the individual end-effector tasks named, so a profile that
        predates the shared block still resolves without a keyword search.
        """
        shared = self.retargeting_config.get("effectors") or {}
        declared: Dict[str, str] = {}
        for side, entry in shared.items():
            if not isinstance(entry, dict):
                continue
            if role is not None and entry.get("role", "foot") != role:
                continue
            if entry.get("body"):
                declared[side] = entry["body"]
        if declared:
            return declared

        for task in ("foot_penetration", "scene_contact", "link_orientation"):
            effectors = (self.retargeting_config.get(task) or {}).get("effectors") or {}
            for side, entry in effectors.items():
                if not isinstance(entry, dict) or not entry.get("body"):
                    continue
                if role == "foot" and side not in ("left", "right"):
                    continue
                declared.setdefault(side, entry["body"])
            if declared:
                return declared
        return declared

    def _validate_foot_sides(self, resolved: Dict[str, int], guessed: bool) -> None:
        """Refuse a foot assignment that is obviously wrong.

        A keyword search over body names is a guess, and a wrong guess here is
        silent and severe: on one robot in this repository searching for "l_"
        matches the "ll_l" inside `r_ankle_roll_link`, so the left foot resolved
        to the right leg and nothing downstream could tell. Better to stop and
        ask for a declaration than to retarget onto the wrong leg.
        """
        names = {
            side: mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            for side, body_id in resolved.items()
            if body_id is not None and body_id >= 0
        }
        if len(names) < 2:
            return
        if names.get("left") == names.get("right"):
            raise ValueError(
                f"both feet resolved to the same body {names.get('left')!r}; "
                "declare retargeting.effectors explicitly"
            )
        # A body carrying the other side's marker is the failure mode worth
        # catching; a name carrying neither is fine and common.
        for side, other in (("left", "right"), ("right", "left")):
            name = (names.get(side) or "").lower()
            claims_other = name.startswith(f"{other}_") or name.startswith(f"{other[0]}_")
            claims_own = name.startswith(f"{side}_") or name.startswith(f"{side[0]}_")
            if claims_other and not claims_own:
                message = (
                    f"the {side} foot resolved to {names[side]!r}, which names the {other} side"
                )
                if guessed:
                    raise ValueError(
                        message + "; declare retargeting.effectors explicitly rather than "
                        "relying on a name search"
                    )
                print(f"Warning: {message}")

    def _nearest_collision_descendant(self, body_id: int) -> Optional[int]:
        """Closest body at or below ``body_id`` that carries collision geometry."""
        def has_collision(bid: int) -> bool:
            return any(
                int(self.robot_model.geom_bodyid[g]) == bid
                and (self.robot_model.geom_contype[g] or self.robot_model.geom_conaffinity[g])
                for g in range(self.robot_model.ngeom)
            )

        if has_collision(body_id):
            return body_id
        frontier = [body_id]
        seen = {body_id}
        while frontier:
            children = [
                b
                for b in range(self.robot_model.nbody)
                if int(self.robot_model.body_parentid[b]) in frontier and b not in seen
            ]
            if not children:
                return None
            for child in children:
                if has_collision(child):
                    return child
            seen.update(children)
            frontier = children
        return None

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
        # Prefer a declaration over a search.  The same foot body is already
        # named by every end-effector task in the profile, and a keyword search
        # is only ever a guess: on this robot "left" plus "ankle" matches the
        # ankle pitch link, which carries no collision geometry at all, so the
        # search silently produced a foot that could not be stood on.  Other
        # robots here name the same part `ankle_link`, `foot_link` and
        # `l_ankle_roll_link`, so no keyword set covers them.
        explicit = dict(cfg.get("body_names", {}) or {})
        if not explicit:
            explicit = self._declared_effector_bodies(role="foot")

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

        self._validate_foot_sides(resolved, guessed=not explicit)
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

        # Prefer primitives when a body carries both: they are the collision
        # proxy the model author intended, and a mesh alongside them is usually
        # the visual shell.  A mesh-only foot still has to be sampled, though,
        # rather than collapsed to its origin.
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
        elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            # A mesh foot is the common case for shelled robots, and returning a
            # single point for it silently reduces the foot to its origin.  The
            # convex hull of a subsample keeps the boundary that matters for
            # support while staying small enough to work with.
            mesh_id = int(self.robot_model.geom_dataid[geom_id])
            if mesh_id < 0:
                points_local = np.zeros((1, 3), dtype=float)
            else:
                start = int(self.robot_model.mesh_vertadr[mesh_id])
                count = int(self.robot_model.mesh_vertnum[mesh_id])
                verts = np.asarray(
                    self.robot_model.mesh_vert[start:start + count], dtype=float
                ).reshape(-1, 3)
                stride = max(1, verts.shape[0] // 512)
                points_local = verts[::stride]
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

    def _foot_support_clearances(
        self,
        motion: np.ndarray,
        terrain_mesh: trimesh.Trimesh,
        foot_specs: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Smallest signed clearance of any sole point above the ground below it.

        Sampling the ground at one representative point per foot and comparing
        it against the lowest sole point mixes two different places: on blocky
        or sloped terrain the lowest point of the sole can sit over a higher
        piece of ground than the representative one, which reads as positive
        clearance while that point is already through the surface.  Each probe
        is therefore tested against the ground directly beneath itself.

        A downward ray is used rather than a signed distance to the nearest
        face because scanned scenes are rarely watertight and their winding is
        often inconsistent, which makes face normals an unreliable guide to
        which side of the surface a point is on.  A ray straight down does not
        depend on that.
        """
        num_frames = len(motion)
        clearances = np.full((num_frames, len(foot_specs)), np.inf)
        down = np.array([0.0, 0.0, -1.0])
        lift = float(terrain_mesh.bounds[1][2] - terrain_mesh.bounds[0][2]) + 1.0

        for frame_idx, qpos in enumerate(motion):
            self.robot_data.qpos[:] = qpos
            mujoco.mj_forward(self.robot_model, self.robot_data)
            for foot_idx, spec in enumerate(foot_specs):
                body_id = spec["body_id"]
                rot = np.asarray(self.robot_data.xmat[body_id], dtype=float).reshape(3, 3)
                points = spec["sample_points"] @ rot.T + np.asarray(
                    self.robot_data.xpos[body_id], dtype=float
                )
                origins = points + np.array([0.0, 0.0, lift])
                try:
                    hits, ray_ids, _ = terrain_mesh.ray.intersects_location(
                        origins, np.tile(down, (len(points), 1)), multiple_hits=False
                    )
                except Exception:  # noqa: BLE001 - ray backend unavailable
                    continue
                if len(hits) == 0:
                    continue
                ray_ids = np.asarray(ray_ids, dtype=int)
                clearances[frame_idx, foot_idx] = float(
                    np.min(points[ray_ids, 2] - hits[:, 2])
                )
        return clearances

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

    @staticmethod
    def _minimal_smooth_lift(required: np.ndarray, window: float) -> np.ndarray:
        """Smoothest non-negative lift that is nowhere below what is required.

        Solves

            min  sum (l[t-1] - 2 l[t] + l[t+1])^2  +  mu * sum l[t]^2
            s.t. l >= required

        The inequality is what makes this usable as a penetration fix: any
        low-pass of the requirement, causal or not, can dip below it and put the
        sole back through the surface, so smoothing has to be constrained rather
        than applied and hoped over.  The mu term pulls the lift to zero wherever
        smoothness does not demand otherwise, which is what keeps the body on the
        source trajectory -- holding a whole contact interval at its maximum, or
        carrying a lift through frames that never needed one, is a deviation the
        source never asked for.

        mu is set from the window because a second-difference smoother responds
        over a length of roughly mu^(-1/4); window is in frames, so callers that
        care about framerate invariance should derive it from a time constant.
        """
        required = np.asarray(required, dtype=float).reshape(-1)
        required = np.maximum(np.nan_to_num(required, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        num_frames = required.shape[0]
        if num_frames < 3 or not np.any(required > 0.0) or window <= 1.0:
            return required

        import clarabel
        from scipy import sparse

        # Second difference operator, rows t = 1 .. T-2.
        rows = num_frames - 2
        diag = np.ones(rows)
        second_difference = sparse.diags(
            [diag, -2.0 * diag, diag], offsets=[0, 1, 2], shape=(rows, num_frames)
        ).tocsc()
        mu = float(window) ** -4.0
        # Objective is (1/2) l' P l with P = 2 (D'D + mu I).
        P = 2.0 * (second_difference.T @ second_difference + mu * sparse.eye(num_frames))
        q = np.zeros(num_frames)
        # -l <= -required, and l <= max(required).  The upper bound is there
        # because a fourth-order smoother rings: fitting a cluster of large
        # requirements makes it overshoot slightly past the largest of them, and
        # lifting the body higher than the motion ever needed is exactly the
        # deviation from the source this formulation is trying to avoid.  The
        # bound costs nothing in feasibility, since l = required satisfies it.
        ceiling = float(required.max())
        A = sparse.vstack(
            [-sparse.eye(num_frames), sparse.eye(num_frames)], format="csc"
        )
        b = np.concatenate([-required, np.full(num_frames, ceiling)])

        settings = clarabel.DefaultSettings()
        settings.verbose = False
        try:
            solver = clarabel.DefaultSolver(
                sparse.triu(P).tocsc(), q, A, b, [clarabel.NonnegativeConeT(2 * num_frames)], settings
            )
            solution = solver.solve()
        except Exception as exc:  # pragma: no cover - solver availability
            print(f"Warning: lift smoothing failed ({exc}); using the raw requirement")
            return required
        if solution.status not in (
            clarabel.SolverStatus.Solved,
            clarabel.SolverStatus.AlmostSolved,
        ):
            print(f"Warning: lift smoothing did not solve ({solution.status}); using the raw requirement")
            return required

        # The solver satisfies the constraint to a tolerance, and a tolerance is
        # a penetration.  Clamp back onto the feasible side.
        return np.clip(np.asarray(solution.x, dtype=float), required, ceiling)

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
