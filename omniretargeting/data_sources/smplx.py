"""SMPL-X motion data source adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from omniretargeting.data_sources.base import DataSource, MotionData, MotionFrame

DEFAULT_SMPLX_TARGET_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
]


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
            names = self.target_names_override or DEFAULT_SMPLX_TARGET_NAMES[: positions.shape[1]]
            self._motion_data = MotionData(
                positions=positions,
                target_names=names,
                root_orientations=root_orient,
                root_translations=trans,
                framerate=framerate,
                source_height=self.compute_human_height(),
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
