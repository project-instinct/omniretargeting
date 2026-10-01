"""SMPL-X AMASS style motion data source adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from omniretargeting.data_sources.base import DataSource, MotionData, MotionFrame
from omniretargeting.data_sources.registry import register_data_source
from omniretargeting.utils import (
    _densify_smplx_frame_gaps,
    _extract_smplx_end_effector_markers,
    _fallback_end_effector_markers,
    estimate_body_height,
)

# T-pose body-aligned rotation: maps body-frame to SMPL-X T-pose frame.
# SMPL-X canonical pose has body facing approx. +Z, Y-up.
# Columns = [forward, right, up] computed with _estimate_base_orientation_from_joints
# algorithm on the neutral (betas=0) SMPL-X T-pose joint positions.
# Variation across body shapes (betas) is < 2 degrees, negligible in practice.
_SMPLX_ROOT_OFFSET = np.array(
    [[0.0, 1.0, 0.0],
     [0.0, 0.0, 1.0],
     [1.0, 0.0, 0.0]],
    dtype=np.float32,
)


DEFAULT_SMPLX_TARGET_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
]


def _default_target_names(num_targets: int) -> list[str]:
    names = DEFAULT_SMPLX_TARGET_NAMES[:num_targets]
    if num_targets > len(DEFAULT_SMPLX_TARGET_NAMES):
        names.extend(f"SMPLX_Joint_{idx}" for idx in range(len(DEFAULT_SMPLX_TARGET_NAMES), num_targets))
    return names


@dataclass
class SmplxDataSource(DataSource):
    motion_file: Path
    model_directory: str | None = None
    gender: str = "neutral"
    target_names_override: list[str] | None = None
    betas: list[float] | None = None
    use_smplx_base_pose: bool = True
    metadata: dict = field(default_factory=dict)
    # Some exporters retain pre-resampling frame IDs only as provenance. Those
    # IDs must not introduce frames into the already published output cadence.
    frame_id_semantics: str = "timeline"

    def __post_init__(self) -> None:
        self.motion_file = Path(self.motion_file)
        if self.frame_id_semantics not in {"timeline", "provenance"}:
            raise ValueError("frame_id_semantics must be 'timeline' or 'provenance'.")
        self._motion_data: MotionData | None = None

    @property
    def target_names(self) -> list[str] | None:
        if self._motion_data is not None:
            return self._motion_data.target_names
        return self.target_names_override

    @property
    def framerate(self) -> float | None:
        return self._motion_data.framerate if self._motion_data is not None else None

    @property
    def source_height(self) -> float | None:
        return self._motion_data.source_height if self._motion_data is not None else self.compute_human_height()

    @property
    def human_height(self) -> float | None:
        return self.source_height

    def load(self) -> MotionData:
        if self._motion_data is None:
            positions, orientations, root_orient, trans, framerate, metadata = self._load_arrays(self.motion_file)
            names = self.target_names_override or _default_target_names(positions.shape[1])

            # Compute source height: try betas first, then trajectory, then None
            source_height = self.compute_human_height()
            if source_height is None:
                source_height = self.estimate_height_from_trajectory(positions, names)
            
            # Correct root_orient from SMPLX T-pose frame to body-aligned frame
            # and convert from axis-angle to wxyz quaternion at the boundary.
            if self.use_smplx_base_pose and root_orient is not None:
                root_orient_mat = Rotation.from_rotvec(root_orient).as_matrix()
                root_orient = Rotation.from_matrix(root_orient_mat @ _SMPLX_ROOT_OFFSET).as_quat(scalar_first=True)
            combined_metadata = {
                **self.metadata,
                **metadata,
                "source_type": "smplx",
                "joint_orientations": orientations,
                **self._foot_sole_metadata(orientations),
            }
            if self.use_smplx_base_pose and orientations is not None:
                # Calibrated frames for any mapped floating body, not just the
                # SMPL-X pelvis. Keep raw joint rotations intact for other tasks.
                aligned = Rotation.from_quat(
                    np.asarray(orientations).reshape(-1, 4), scalar_first=True
                ).as_matrix() @ _SMPLX_ROOT_OFFSET
                combined_metadata["body_aligned_joint_orientations"] = (
                    Rotation.from_matrix(aligned).as_quat(scalar_first=True)
                    .reshape(np.asarray(orientations).shape)
                )
            # End-effector targets are part of the generic MotionData contract,
            # just like joint orientations.  Never make callers switch back to
            # the legacy SMPL-X loader merely to activate contact tracking.
            combined_metadata.setdefault(
                "end_effector_markers",
                _fallback_end_effector_markers(positions),
            )
            self._motion_data = MotionData(
                positions=positions,
                target_names=names,
                root_orientations=root_orient if self.use_smplx_base_pose else None,
                root_translations=trans if self.use_smplx_base_pose else None,
                framerate=framerate,
                source_height=source_height,
                metadata=combined_metadata,
            )
        return self._motion_data

    # SMPL-X body joint indices for each foot: (ankle, toe base).
    _FOOT_JOINTS = {"left": (7, 10), "right": (8, 11)}
    _SPINE3_INDEX = 9

    def _foot_sole_metadata(self, orientations: np.ndarray | None) -> dict:
        """Per-frame world sole rotations for each foot, when derivable.

        Returns an empty dict rather than raising when the ingredients are
        missing, because positions-only sources are legitimate; consumers
        decide whether to fall back or to disable the orientation task.
        """
        if orientations is None or self.model_directory is None:
            return {}

        try:
            import smplx
            import torch

            from .smplx_feet import smplx_sole_metadata

            result = getattr(self, "_sole_metadata_cache", None)
            if result is None:
                body_model = smplx.create(
                    self.model_directory, "smplx", gender=self.gender, use_pca=False
                )
                kwargs = {} if self.betas is None else {
                    "betas": torch.as_tensor(self.betas[:10], dtype=torch.float32).reshape(1, -1)
                }
                with torch.no_grad():
                    template = body_model(**kwargs)
                result = smplx_sole_metadata(
                    template.vertices.detach().cpu().numpy()[0],
                    template.joints.detach().cpu().numpy()[0],
                    body_model.lbs_weights.detach().cpu().numpy(), orientations,
                    getattr(self, "_posed_vertices_cache", None),
                )
            # Raw loading populates this from the exact model gender and betas
            # used for the source, rather than recreating a neutral subject.
            result = {**result, "link_orientations": dict(result["link_orientations"])}
            rotations = result["link_orientations"]
        except Exception as exc:  # noqa: BLE001 - degrade to positions-only
            print(f"Warning: could not calibrate SMPL-X foot sole frames: {exc}")
            return {}

        # The torso needs an orientation target for the same reason the feet
        # did, and more starkly: on the G1 the torso link origin sits exactly
        # on the waist pitch axis, so a positional marker there has a zero
        # lever arm and cannot constrain that joint at all, no matter its
        # weight.  Every source profile maps its spine marker to the link
        # upstream of waist pitch, which leaves the joint free to sit at its
        # limit.  Unlike a foot there is no sole plane to calibrate against:
        # SMPL-X rest joint rotations are identity, so the same axis remap the
        # root already uses carries the spine frame into the robot's
        # convention.
        rotations["torso"] = np.einsum(
            "tij,jk->tik",
            Rotation.from_quat(orientations[:, self._SPINE3_INDEX], scalar_first=True).as_matrix(),
            _SMPLX_ROOT_OFFSET,
        )

        self._sole_metadata_cache = None
        self._posed_vertices_cache = None
        return result

    @staticmethod
    def _sole_band_vertex_ids(
        template_vertices: np.ndarray, foot_mask: np.ndarray, sole_normal: np.ndarray,
        band_fraction: float = 0.15,
    ) -> np.ndarray:
        """Template vertex ids of the ground-contact band of one foot.

        Fixed ids rather than a per-frame "lowest vertices" query, so the point
        being tracked is the same piece of anatomy on every frame. Re-selecting
        per frame would let the tracked point slide around the sole as the foot
        rotates, which is indistinguishable from the foot itself sliding.
        """
        from .smplx_feet import sole_band_vertex_ids

        return sole_band_vertex_ids(template_vertices, foot_mask, sole_normal, band_fraction)

    def _foot_sole_points(self, sole_vertex_ids: dict) -> dict:
        """World positions of every sole band vertex, per frame.

        The centre answers "where is the foot"; this answers "is any part of the
        foot standing still".  They are not the same question, and using the
        centre for both is wrong in a specific way: a foot planted through a
        heel-strike to toe-off roll pivots about its contact patch, so the sole
        centre translates at a fair speed while the part actually touching the
        ground does not move at all.  On the source clip the sole centre never
        drops below about 10 cm/s on any frame, which makes a plant test built
        on it either vacuous or impossible to satisfy.
        """
        vertices = getattr(self, "_posed_vertices_cache", None)
        if vertices is None:
            return {}
        try:
            return {
                side: np.asarray(vertices[:, ids, :], dtype=float)
                for side, ids in sole_vertex_ids.items()
            }
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not compute SMPL-X sole points: {exc}")
            return {}

    def _foot_sole_centres(self, sole_vertex_ids: dict) -> dict:
        """World position of each foot's sole centre, per frame.

        This is the point a contact task anchors: with sole orientation already
        tracked, pinning one sole point fixes the remaining three degrees of
        freedom, so position and orientation together give the full foot pose
        without over-constraining it.
        """
        vertices = getattr(self, "_posed_vertices_cache", None)
        if vertices is None:
            return {}
        try:
            centres = {
                side: vertices[:, ids, :].mean(axis=1).astype(float)
                for side, ids in sole_vertex_ids.items()
            }
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not compute SMPL-X sole centres: {exc}")
            return {}
        return centres

    def iter_frames(self):
        yield from self.load().iter_frames()

    def load_trajectory(
        self,
        return_meta: bool = False,
        return_end_effectors: bool = False,
        end_effector_samples: int = 32,
    ) -> tuple:
        motion_data = self.load()
        orientations = motion_data.metadata.get("joint_orientations")
        if return_end_effectors:
            # Surface markers are only available when the loader ran the body
            # model. Position-only sources fall back to the mapped hand and foot
            # joints so contact inference still has a marker set to work with.
            markers = motion_data.metadata.get("end_effector_markers")
            if markers is None:
                markers = _fallback_end_effector_markers(motion_data.positions)
            if return_meta:
                return (
                    motion_data.positions,
                    orientations,
                    motion_data.root_orientations,
                    motion_data.root_translations,
                    markers,
                )
            return motion_data.positions, orientations, markers
        if return_meta:
            return motion_data.positions, orientations, motion_data.root_orientations, motion_data.root_translations
        return motion_data.positions, orientations

    def compute_human_height(self) -> float | None:
        if self.betas is None:
            return None
        try:
            import smplx as smplx_lib
            import torch
        except ImportError:
            return None

        import os
        search_paths = [self.model_directory] if self.model_directory else []
        search_paths.extend(["/localhdd/Datasets/smplx", "/localhdd/Datasets/", "data/body_models/smplx"])
        model_path = next((p for p in search_paths if p and os.path.exists(p)), None)
        if model_path is None:
            return None

        try:
            model = smplx_lib.SMPLX(model_path, num_betas=len(self.betas), use_hands=False, use_face=False)
            betas_tensor = torch.tensor([self.betas], dtype=torch.float32)
            with torch.no_grad():
                out = model(betas=betas_tensor)
            joints = out.joints[0, :22].numpy()
            return float(joints[:, 1].max() - joints[:, 1].min())
        except Exception as exc:
            print(f"[SmplxDataSource] Failed to compute height from betas: {exc}")
            return None


    def estimate_height_from_trajectory(self, positions: np.ndarray, target_names: list[str]) -> float | None:
        """Estimate human height from trajectory positions using shared utility.

        Args:
            positions: Motion positions array of shape ``(T, J, 3)``.
            target_names: List of joint names corresponding to the J axis.

        Returns:
            Estimated height in meters, or ``None`` if estimation fails.
        """
        return estimate_body_height(positions, target_names, head_joint="Head", foot_joints=("L_Foot", "R_Foot"))

    def _load_arrays(
        self,
        motion_file: Path,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, float | None, dict]:
        if motion_file.suffix == ".npy":
            joints = np.load(motion_file, allow_pickle=True)
            print("Warning: Cannot compute orientations from .npy file (positions only). Returning None for orientations.")
            return joints, None, None, None, None, {}

        motion = np.load(motion_file, allow_pickle=True)
        framerate = self._detect_framerate(motion) if isinstance(motion, np.lib.npyio.NpzFile) else None

        if isinstance(motion, np.lib.npyio.NpzFile) and "global_joint_positions" in motion:
            return self._load_processed_npz(motion, framerate)

        return self._load_raw_npz(motion, framerate)

    def _load_processed_npz(
        self,
        motion: np.lib.npyio.NpzFile,
        framerate: float | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, float | None, dict]:
        joints = motion["global_joint_positions"]
        root_orient = motion["root_orient"] if "root_orient" in motion else None
        trans = motion["trans"] if "trans" in motion else None
        orientations = None
        if "full_pose" in motion and self.model_directory is not None and root_orient is not None:
            import smplx

            body_model = smplx.create(self.model_directory, "smplx", gender=self.gender, use_pca=False)
            full_pose = motion["full_pose"]
            if isinstance(full_pose, np.ndarray) and full_pose.ndim == 2:
                full_pose = full_pose.reshape(full_pose.shape[0], -1, 3)
            orientations = self.compute_world_joint_orientations(
                root_orient,
                full_pose,
                body_model.parents.cpu().numpy(),
                num_body_joints=22,
            )
        else:
            print("Warning: Cannot compute orientations from .npz file (missing full_pose, root_orient, or model directory). Returning None for orientations.")
        return joints, orientations, root_orient, trans, framerate, {}

    def _prepare_raw_timeline(
        self,
        motion: np.lib.npyio.NpzFile,
        framerate: float | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, dict]:
        """Interpret frame IDs before FK, without modifying the source arrays.

        ``provenance`` is an explicit assertion that rows already follow the
        published cadence; it is never inferred just from gaps in frame IDs.
        Original IDs remain metadata, not replacement timestamps. In particular,
        a sparse person timeline cannot be treated as dense published rows.
        """
        pose_body = np.asarray(motion["pose_body"])
        root_orient = np.asarray(motion["root_orient"])
        trans = np.asarray(motion["trans"])
        pose_hand = np.asarray(motion["pose_hand"]) if "pose_hand" in motion else None
        timeline_metadata = {"frame_id_semantics": self.frame_id_semantics}
        if "frame_ids" in motion:
            timeline_metadata["original_frame_ids"] = np.array(motion["frame_ids"], copy=True)

        if self.frame_id_semantics == "provenance":
            if framerate is None or not np.isfinite(framerate) or framerate <= 0:
                raise ValueError("Provenance frame IDs require a finite, positive published framerate.")
            if bool(np.asarray(motion.get("person_timeline_sparse", False)).item()):
                raise ValueError("Provenance frame IDs cannot describe a sparse person timeline.")
            if "frame_ids" in motion and np.asarray(motion["frame_ids"]).shape != (len(trans),):
                raise ValueError("Provenance frame_ids must contain one ID per published motion row.")
            return trans, root_orient, pose_body, pose_hand, timeline_metadata

        if "frame_ids" in motion:
            original_num_frames = len(trans)
            trans, root_orient, pose_body, pose_hand, _ = _densify_smplx_frame_gaps(
                motion["frame_ids"], trans, root_orient, pose_body, pose_hand
            )
            if len(trans) != original_num_frames:
                print(
                    "Restored "
                    f"{len(trans) - original_num_frames} missing SMPL-X "
                    "timeline frame(s) from frame_ids."
                )
        return trans, root_orient, pose_body, pose_hand, timeline_metadata

    def _load_raw_npz(
        self,
        motion: np.lib.npyio.NpzFile,
        framerate: float | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, float | None, dict]:
        import smplx
        import torch

        trans, root_orient, pose_body, pose_hand, timeline_metadata = self._prepare_raw_timeline(
            motion, framerate
        )
        body_model = smplx.create(
            self.model_directory,
            "smplx",
            gender=self._model_gender(motion),
            use_pca=False,
        )

        if self.betas is None:
            betas_tensor = torch.tensor(motion["betas"]).float().view(1, -1)
        else:
            betas_tensor = torch.tensor([self.betas]).float()
        if betas_tensor.shape[1] > 10:
            betas_tensor = betas_tensor[:, :10]

        num_frames = pose_body.shape[0]
        if pose_hand is not None and pose_hand.ndim == 2 and pose_hand.shape[1] >= 90:
            left_hand_pose = torch.tensor(pose_hand[:, :45]).float()
            right_hand_pose = torch.tensor(pose_hand[:, 45:90]).float()
        else:
            left_hand_pose = torch.zeros(num_frames, 45).float()
            right_hand_pose = torch.zeros(num_frames, 45).float()
        output = body_model(
            betas=betas_tensor,
            global_orient=torch.tensor(root_orient).float(),
            body_pose=torch.tensor(pose_body).float(),
            transl=torch.tensor(trans).float(),
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            jaw_pose=torch.zeros(num_frames, 3).float(),
            leye_pose=torch.zeros(num_frames, 3).float(),
            reye_pose=torch.zeros(num_frames, 3).float(),
            expression=torch.zeros(num_frames, 10).float(),
            return_full_pose=True,
        )

        joints = output.joints.detach().cpu().numpy()[:, :22, :]
        # Kept only long enough to read the sole band out of it; the surface is
        # needed to locate contact points that joints alone cannot express.
        self._posed_vertices_cache = output.vertices.detach().cpu().numpy()
        full_pose = output.full_pose.detach().cpu().numpy().reshape(num_frames, -1, 3)
        orientations = self.compute_world_joint_orientations(
            root_orient,
            full_pose,
            body_model.parents.cpu().numpy(),
            num_body_joints=22,
        )
        # A clip can start mid-flip: its first pose is not a bind pose. Evaluate
        # the same shaped model at rest only for orientation calibration. This
        # separate reference does not transform the captured world trajectory.
        with torch.no_grad():
            reference = body_model(
                betas=betas_tensor,
                global_orient=torch.zeros(1, 3),
                body_pose=torch.zeros(1, 63),
                transl=torch.zeros(1, 3),
            )
        from .smplx_feet import smplx_sole_metadata

        self._sole_metadata_cache = smplx_sole_metadata(
            reference.vertices.detach().cpu().numpy()[0],
            reference.joints.detach().cpu().numpy()[0],
            body_model.lbs_weights.detach().cpu().numpy(), orientations,
            self._posed_vertices_cache,
        )
        end_effector_markers = _extract_smplx_end_effector_markers(
            body_model,
            output,
            max_samples=32,
            world_joint_orientations=orientations,
            shape_betas=betas_tensor.detach().cpu().numpy(),
            sole_metadata=self._sole_metadata_cache,
        )
        from .smplx_surface import sample_smplx_surfaces
        body_surfaces = sample_smplx_surfaces(
            body_model, reference, self._posed_vertices_cache, _SMPLX_ROOT_OFFSET.T,
        )
        return joints, orientations, root_orient, trans, framerate, {
            **timeline_metadata,
            "body_surface_samples": body_surfaces,
            "betas": betas_tensor.detach().cpu().numpy()[0].tolist(),
            "end_effector_markers": end_effector_markers,
            "joint_rest_positions": reference.joints.detach().cpu().numpy()[0, :22] @ _SMPLX_ROOT_OFFSET,
            "joint_rest_orientations": np.tile(
                Rotation.from_matrix(_SMPLX_ROOT_OFFSET.T).as_quat(scalar_first=True), (22, 1)
            ),
            "joint_reference_pose": "SMPL-X shaped zero body pose, calibrated Z-up reference",
            "joint_reference_canonical_to_reference": _SMPLX_ROOT_OFFSET.T.copy(),
        }

    @staticmethod
    def compute_world_joint_orientations(
        global_orient: np.ndarray,
        full_pose: np.ndarray,
        parents: np.ndarray,
        num_body_joints: int = 22,
    ) -> np.ndarray:
        num_frames = global_orient.shape[0]
        num_joints = min(full_pose.shape[1], num_body_joints)
        joint_orientations = np.zeros((num_frames, num_joints, 4))

        for frame_idx in range(num_frames):
            frame_rotations = []
            for joint_idx in range(num_joints):
                if joint_idx == 0:
                    rot = Rotation.from_rotvec(global_orient[frame_idx])
                else:
                    parent_idx = parents[joint_idx]
                    if 0 <= parent_idx < len(frame_rotations):
                        rot = frame_rotations[parent_idx] * Rotation.from_rotvec(full_pose[frame_idx, joint_idx])
                    else:
                        rot = Rotation.from_rotvec(full_pose[frame_idx, joint_idx])
                frame_rotations.append(rot)
                joint_orientations[frame_idx, joint_idx] = rot.as_quat(scalar_first=True)

        return joint_orientations

    @staticmethod
    def _detect_framerate(motion: np.lib.npyio.NpzFile) -> float | None:
        for key in ("framerate", "mocap_framerate", "mocap_frame_rate"):
            if key in motion:
                return float(motion[key])
        return None

    def _model_gender(self, motion: np.lib.npyio.NpzFile) -> str:
        value = motion.get("gender", self.gender)
        if isinstance(value, np.ndarray):
            value = value.item()
        return str(value)


def compute_world_joint_orientations(*args, **kwargs):
    return SmplxDataSource.compute_world_joint_orientations(*args, **kwargs)


def validate_smplx_trajectory(trajectory: np.ndarray) -> bool:
    from omniretargeting.data_sources.base import validate_motion_positions

    return validate_motion_positions(trajectory)


def extract_smplx_joint_positions(trajectory: np.ndarray, joint_indices: list) -> np.ndarray:
    return trajectory[:, joint_indices, :]


def create_smplx_data_source(
    motion_file: Path,
    source_config: dict | None = None,
    runtime_options: dict | None = None,
) -> SmplxDataSource:
    source_config = dict(source_config or {})
    runtime_options = dict(runtime_options or {})
    adapter_options = dict(source_config.get("adapter_options") or {})

    def option(*keys, default=None):
        for container in (runtime_options, adapter_options, source_config):
            for key in keys:
                if key in container and container[key] is not None:
                    return container[key]
        return default

    target_names = option("target_names_override", "target_names", "joint_names")
    model_directory = option("model_directory", "model_dir", "smpl_model_dir", "smplx_model_dir")

    return SmplxDataSource(
        motion_file=motion_file,
        model_directory=model_directory,
        gender=option("gender", default="neutral"),
        target_names_override=target_names,
        betas=option("betas", "smplx_betas"),
        use_smplx_base_pose=option("use_smplx_base_pose", default=False),
        frame_id_semantics=option("frame_id_semantics", default="timeline"),
    )


def load_smplx_motion(
    smplx_file: Path,
    smplx_model_directory: Optional[str] = None,
    gender: str = "neutral",
    target_names: list[str] | None = None,
    betas: list[float] | None = None,
) -> MotionData:
    return SmplxDataSource(
        motion_file=smplx_file,
        model_directory=smplx_model_directory,
        gender=gender,
        target_names_override=target_names,
        betas=betas,
    ).load()


def load_smplx_trajectory(
    smplx_file: Path,
    smplx_model_directory: Optional[str] = None,
    gender: str = "neutral",
    return_meta: bool = False,
    return_end_effectors: bool = False,
    end_effector_samples: int = 32,
) -> tuple:
    return SmplxDataSource(
        motion_file=smplx_file,
        model_directory=smplx_model_directory,
        gender=gender,
    ).load_trajectory(
        return_meta=return_meta,
        return_end_effectors=return_end_effectors,
        end_effector_samples=end_effector_samples,
    )


def retarget_smplx_to_robot(
    smplx_trajectory: np.ndarray,
    robot_urdf_path: Path,
    terrain_mesh_path: Path,
    joint_mapping: Dict[str, str],
    robot_height: Optional[float] = None,
    smplx_joint_names: Optional[List[str]] = None,
) -> Tuple[float, np.ndarray]:
    """Backward-compatible wrapper for older SMPL-X-specific callers."""
    from omniretargeting.retargeting import retarget_source_to_robot

    return retarget_source_to_robot(
        source_positions=smplx_trajectory,
        robot_urdf_path=robot_urdf_path,
        terrain_mesh_path=terrain_mesh_path,
        joint_mapping=joint_mapping,
        robot_height=robot_height,
        source_target_names=smplx_joint_names,
    )


register_data_source("smplx", create_smplx_data_source, extensions=[".npz"])
