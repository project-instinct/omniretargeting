"""OMOMO dataset adapter for object manipulation motion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import trimesh

from .base import DataSource, MotionData
from .smplx import DEFAULT_SMPLX_TARGET_NAMES


@dataclass
class OmomoDataSource(DataSource):
    """
    Data source for OMOMO (Object Motion Guided Human Motion Synthesis) dataset.

    OMOMO provides SMPL body parameters with object 6D poses for manipulation tasks.
    """

    sequence_file: Path
    sequence_index: int
    data_root: Path
    n_object_samples: int = 100
    target_names: list[str] | None = None
    model_directory: str | None = None
    framerate: float = 30.0

    def __post_init__(self):
        self.sequence_file = Path(self.sequence_file)
        self.data_root = Path(self.data_root)
        self._motion_data: MotionData | None = None

        if not self.sequence_file.exists():
            raise FileNotFoundError(f"Sequence file not found: {self.sequence_file}")

        data = joblib.load(self.sequence_file)
        if self.sequence_index >= len(data):
            raise ValueError(f"Sequence index {self.sequence_index} out of range (max: {len(data)-1})")

        self.sequence = data[self.sequence_index]
        self.seq_name = self.sequence["seq_name"]

        parts = self.seq_name.split("_")
        self.object_name = "_".join(parts[1:-1])

        mesh_path = self.data_root / "data" / "captured_objects" / f"{self.object_name}_cleaned_simplified.obj"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Object mesh not found: {mesh_path}")

        self.object_mesh = trimesh.load(mesh_path, force="mesh")
        self.object_local_points = self._sample_object_points()

        self.metadata = {
            "seq_name": self.seq_name,
            "object_name": self.object_name,
            "gender": self._sequence_gender(),
            "source": "omomo",
        }

    def _sequence_gender(self) -> str:
        value = self.sequence.get("gender", "neutral")
        if isinstance(value, np.ndarray):
            value = value.item()
        return str(value)

    def _resolve_model_directory(self) -> str:
        search_paths = [
            self.model_directory,
            "/localhdd/Datasets/smplx",
            "/localhdd/Datasets",
            "data/body_models/smplx",
        ]
        for candidate in search_paths:
            if candidate and Path(candidate).exists():
                return str(candidate)
        raise FileNotFoundError(
            "Could not locate SMPL-X model directory for OMOMO. "
            "Provide model_directory or install the models under /localhdd/Datasets/smplx or /localhdd/Datasets."
        )

    def _sample_object_points(self) -> np.ndarray:
        vertices = np.array(self.object_mesh.vertices)
        if len(vertices) <= self.n_object_samples:
            return vertices

        sampled_indices = [0]
        remaining = set(range(len(vertices)))
        remaining.remove(0)

        for _ in range(self.n_object_samples - 1):
            sampled_points = vertices[sampled_indices]
            distances = np.min(
                np.linalg.norm(
                    vertices[list(remaining)][:, None, :] - sampled_points[None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            farthest_idx = list(remaining)[np.argmax(distances)]
            sampled_indices.append(farthest_idx)
            remaining.remove(farthest_idx)

        return vertices[sampled_indices]

    def _object_pose_data(self) -> dict[str, np.ndarray]:
        obj_scale = np.asarray(self.sequence["obj_scale"], dtype=np.float32)
        obj_rot = np.asarray(self.sequence["obj_rot"], dtype=np.float32)
        obj_trans = np.asarray(self.sequence["obj_trans"], dtype=np.float32).reshape(len(obj_scale), 3)
        obj_com = np.asarray(self.sequence["obj_com_pos"], dtype=np.float32).reshape(len(obj_scale), 3)
        return {
            "scale": obj_scale,
            "rotation": obj_rot,
            "translation": obj_trans,
            "centroid_world": obj_com,
        }

    def _transform_object_points(self) -> np.ndarray:
        pose = self._object_pose_data()
        object_points = []
        for frame_idx in range(len(pose["translation"])):
            scaled_points = self.object_local_points * pose["scale"][frame_idx]
            world_points = scaled_points @ pose["rotation"][frame_idx].T + pose["translation"][frame_idx].reshape(1, 3)
            object_points.append(world_points)
        return np.array(object_points, dtype=np.float32)

    def _load_body_positions(self) -> np.ndarray:
        import smplx
        import torch

        model_directory = self._resolve_model_directory()
        raw_trans = np.asarray(self.sequence["trans"], dtype=np.float32)
        root_orient = np.asarray(self.sequence["root_orient"], dtype=np.float32)
        body_pose = np.asarray(self.sequence["pose_body"], dtype=np.float32)
        trans2joint = np.asarray(self.sequence["trans2joint"], dtype=np.float32).reshape(1, 3)
        transl = raw_trans + trans2joint
        betas = np.asarray(self.sequence["betas"], dtype=np.float32).reshape(1, -1)
        if betas.shape[1] > 10:
            betas = betas[:, :10]

        num_frames = body_pose.shape[0]
        body_model = smplx.create(
            model_directory,
            "smplx",
            gender=self._sequence_gender(),
            use_pca=False,
        )
        output = body_model(
            betas=torch.tensor(betas).float(),
            global_orient=torch.tensor(root_orient).float(),
            body_pose=torch.tensor(body_pose).float(),
            transl=torch.tensor(transl).float(),
            left_hand_pose=torch.zeros(num_frames, 45).float(),
            right_hand_pose=torch.zeros(num_frames, 45).float(),
            jaw_pose=torch.zeros(num_frames, 3).float(),
            leye_pose=torch.zeros(num_frames, 3).float(),
            reye_pose=torch.zeros(num_frames, 3).float(),
            expression=torch.zeros(num_frames, 10).float(),
            return_full_pose=True,
        )
        return output.joints.detach().cpu().numpy()[:, :22, :].astype(np.float32)

    def _estimate_height_from_positions(self, positions: np.ndarray) -> float | None:
        if positions is None or len(positions) == 0 or positions.shape[1] < 16:
            return None

        head_idx = 15
        foot_indices = [10, 11]
        head_top_offset = 0.12
        head_z = positions[:, head_idx, 2]
        feet_z = np.min(positions[:, foot_indices, 2], axis=1)
        estimated_height = float(np.max(head_z - feet_z) + head_top_offset)
        return float(np.clip(estimated_height, 1.4, 2.2))

    def load(self) -> MotionData:
        if self._motion_data is None:
            raw_trans = np.asarray(self.sequence["trans"], dtype=np.float32)
            root_orient = np.asarray(self.sequence["root_orient"], dtype=np.float32)
            positions = self._load_body_positions()
            object_points = self._transform_object_points()
            target_names = self.target_names or DEFAULT_SMPLX_TARGET_NAMES[: positions.shape[1]]
            object_pose = self._object_pose_data()
            object_centroid_local = np.asarray(self.object_mesh.vertices, dtype=np.float32).mean(axis=0)

            self._motion_data = MotionData(
                positions=positions,
                target_names=target_names,
                root_orientations=root_orient,
                root_translations=positions[:, 0, :],
                framerate=self.framerate,
                source_height=self._estimate_height_from_positions(positions),
                object_points=object_points,
                object_mesh=self.object_mesh,
                metadata={
                    **self.metadata,
                    "joint_orientations": None,
                    "raw_root_translations": raw_trans,
                    "object_translations": object_pose["translation"],
                    "object_rotations": object_pose["rotation"],
                    "object_scales": object_pose["scale"],
                    "object_centroid_world": object_pose["centroid_world"],
                    "object_centroid_local": object_centroid_local,
                },
            )

        return self._motion_data

    def iter_frames(self):
        yield from self.load().iter_frames()


def create_omomo_data_source(motion_file, source_config, runtime_options):
    sequence_index = runtime_options.get("sequence_index", 0)
    data_root = runtime_options.get("data_root", "/localhdd/Datasets/OMOMO")
    n_object_samples = runtime_options.get("n_object_samples", 100)
    target_names_override = runtime_options.get("target_names_override", None)
    model_directory = runtime_options.get("model_directory", None)

    return OmomoDataSource(
        sequence_file=motion_file,
        sequence_index=sequence_index,
        data_root=data_root,
        n_object_samples=n_object_samples,
        target_names=target_names_override,
        model_directory=model_directory,
    )


from .registry import register_data_source

register_data_source("omomo", create_omomo_data_source)
