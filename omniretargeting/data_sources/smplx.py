"""SMPL-X AMASS style motion data source adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from omniretargeting.data_sources.base import DataSource, MotionData, MotionFrame
from omniretargeting.data_sources.registry import register_data_source
from omniretargeting.utils import estimate_body_height

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
                source_height = self.estimate_height_from_trajectory(positions, names)
            
            # Correct root_orient from SMPLX T-pose frame to body-aligned frame
            # and convert from axis-angle to wxyz quaternion at the boundary.
            if self.use_smplx_base_pose and root_orient is not None:
                root_orient_mat = Rotation.from_rotvec(root_orient).as_matrix()
                root_orient = Rotation.from_matrix(root_orient_mat @ _SMPLX_ROOT_OFFSET).as_quat(scalar_first=True)
            self._motion_data = MotionData(
                positions=positions,
                target_names=names,
                root_orientations=root_orient if self.use_smplx_base_pose else None,
                root_translations=trans if self.use_smplx_base_pose else None,
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

        if not self.model_directory:
            return None
        model_directory = Path(self.model_directory)
        search_paths = [model_directory / "smplx", model_directory]
        model_path = next((path for path in search_paths if path.exists()), None)
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

        # Raw SMPL-X npz uses ``body_pose``/``global_orient``/``transl``; the
        # SMPL-H/AMASS format uses ``pose_body``/``root_orient``/``trans``.
        # Accept both key sets.
        def _arr(*names: str) -> np.ndarray | None:
            for name in names:
                if name in motion:
                    return np.asarray(motion[name], dtype=np.float32)
            return None

        body_pose_arr = _arr("pose_body", "body_pose")
        root_orient = _arr("root_orient", "global_orient")
        trans = _arr("trans", "transl")
        if body_pose_arr is None or root_orient is None or trans is None:
            missing = [
                name for name, arr in (
                    ("pose_body/body_pose", body_pose_arr),
                    ("root_orient/global_orient", root_orient),
                    ("trans/transl", trans),
                ) if arr is None
            ]
            raise KeyError(f"SMPL-X npz missing pose keys: {missing}")

        if self.betas is None:
            betas_arr = _arr("betas")
            if betas_arr is None:
                betas_arr = np.zeros((1, 10), dtype=np.float32)
            if betas_arr.ndim == 2 and betas_arr.shape[0] > 1:
                betas_arr = betas_arr[0]  # per-frame betas: use the first frame
            betas_tensor = torch.tensor(betas_arr).float().view(1, -1)
        else:
            betas_tensor = torch.tensor([self.betas]).float()
        if betas_tensor.shape[1] > 10:
            betas_tensor = betas_tensor[:, :10]

        num_frames = body_pose_arr.shape[0]

        def _pose(arr: np.ndarray | None, dims: int) -> torch.Tensor:
            if arr is not None and arr.shape[0] == num_frames:
                return torch.tensor(arr).float()
            return torch.zeros(num_frames, dims).float()

        output = body_model(
            betas=betas_tensor,
            global_orient=torch.tensor(root_orient).float(),
            body_pose=torch.tensor(body_pose_arr).float(),
            transl=torch.tensor(trans).float(),
            left_hand_pose=_pose(_arr("left_hand_pose"), 45),
            right_hand_pose=_pose(_arr("right_hand_pose"), 45),
            jaw_pose=_pose(_arr("jaw_pose"), 3),
            leye_pose=_pose(_arr("leye_pose"), 3),
            reye_pose=_pose(_arr("reye_pose"), 3),
            expression=_pose(_arr("expression"), 10),
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
        use_smplx_base_pose=option("use_smplx_base_pose", default=False),
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


register_data_source("smplx", create_smplx_data_source, extensions=[".npz"])
