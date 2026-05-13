"""SMPL-X motion data source adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from omniretargeting.data_sources.base import DataSource, MotionData, MotionFrame
from omniretargeting.data_sources.registry import register_data_source

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
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.motion_file = Path(self.motion_file)
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
                source_height = self.estimate_height_from_trajectory(positions)
            
            self._motion_data = MotionData(
                positions=positions,
                target_names=names,
                root_orientations=root_orient,
                root_translations=trans,
                framerate=framerate,
                source_height=source_height,
                metadata={**self.metadata, **metadata, "source_type": "smplx", "joint_orientations": orientations},
            )
        return self._motion_data

    def iter_frames(self):
        yield from self.load().iter_frames()

    def load_trajectory(
        self,
        return_meta: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None] | tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        motion_data = self.load()
        orientations = motion_data.metadata.get("joint_orientations")
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


    def estimate_height_from_trajectory(self, positions: np.ndarray) -> float | None:
        """
        Estimate human height from trajectory positions.
        
        Uses head and foot positions across all frames to estimate standing height.
        
        Args:
            positions: Motion positions array of shape (T, J, 3)
            
        Returns:
            Estimated height in meters, or None if estimation fails
        """
        if positions is None or len(positions) == 0:
            return None
        
        # Default SMPL-X joint indices
        head_idx = 15
        foot_indices = [10, 11]
        head_top_offset = 0.12
        
        # Try to use target names if available
        if self.target_names_override:
            try:
                head_idx = self.target_names_override.index("Head")
            except ValueError:
                pass
            
            foot_indices = []
            for foot_name in ["L_Foot", "R_Foot"]:
                try:
                    foot_indices.append(self.target_names_override.index(foot_name))
                except ValueError:
                    pass
            if not foot_indices:
                foot_indices = [10, 11]
        
        # Check if we have enough joints
        if positions.shape[1] <= head_idx or not all(idx < positions.shape[1] for idx in foot_indices):
            return None
        
        try:
            # Calculate height per frame: Head Z - Min Foot Z
            head_z = positions[:, head_idx, 2]
            feet_z = np.min(positions[:, foot_indices, 2], axis=1)
            heights = head_z - feet_z
            
            # Use maximum height (standing pose) + head top offset
            estimated_height = float(np.max(heights) + head_top_offset)
            
            # Sanity check: reasonable human range
            estimated_height = np.clip(estimated_height, 1.4, 2.2)
            
            return estimated_height
        except Exception as e:
            print(f"[SmplxDataSource] Failed to estimate height from trajectory: {e}")
            return None

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

    def _load_raw_npz(
        self,
        motion: np.lib.npyio.NpzFile,
        framerate: float | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, float | None, dict]:
        import smplx
        import torch

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

        num_frames = motion["pose_body"].shape[0]
        root_orient = motion["root_orient"]
        trans = motion["trans"]
        output = body_model(
            betas=betas_tensor,
            global_orient=torch.tensor(root_orient).float(),
            body_pose=torch.tensor(motion["pose_body"]).float(),
            transl=torch.tensor(trans).float(),
            left_hand_pose=torch.zeros(num_frames, 45).float(),
            right_hand_pose=torch.zeros(num_frames, 45).float(),
            jaw_pose=torch.zeros(num_frames, 3).float(),
            leye_pose=torch.zeros(num_frames, 3).float(),
            reye_pose=torch.zeros(num_frames, 3).float(),
            expression=torch.zeros(num_frames, 10).float(),
            return_full_pose=True,
        )

        joints = output.joints.detach().cpu().numpy()[:, :22, :]
        full_pose = output.full_pose.detach().cpu().numpy().reshape(num_frames, -1, 3)
        orientations = self.compute_world_joint_orientations(
            root_orient,
            full_pose,
            body_model.parents.cpu().numpy(),
            num_body_joints=22,
        )
        return joints, orientations, root_orient, trans, framerate, {"betas": betas_tensor.detach().cpu().numpy()[0].tolist()}

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
) -> tuple[np.ndarray, np.ndarray | None] | tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    return SmplxDataSource(
        motion_file=smplx_file,
        model_directory=smplx_model_directory,
        gender=gender,
    ).load_trajectory(return_meta=return_meta)


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


register_data_source("smplx", create_smplx_data_source)
