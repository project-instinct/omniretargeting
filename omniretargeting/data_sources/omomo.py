"""OMOMO dataset adapter for object manipulation motion."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from omniretargeting.utils import estimate_body_height

from .base import DataSource, MotionData
from .omomo_validity import prepare_object_frames
from .smplx import _SMPLX_ROOT_OFFSET, DEFAULT_SMPLX_TARGET_NAMES


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
    use_smplx_base_pose: bool = True
    source_denoise: str = "off"

    def __post_init__(self):
        self.sequence_file = Path(self.sequence_file)
        self.data_root = Path(self.data_root)
        self._motion_data: MotionData | None = None

        if not self.sequence_file.exists():
            raise FileNotFoundError(f"Sequence file not found: {self.sequence_file}")

        data = joblib.load(self.sequence_file)
        if self.sequence_index < 0 or self.sequence_index >= len(data):
            raise ValueError(f"Sequence index {self.sequence_index} out of range (max: {len(data)-1})")

        self.sequence, self.object_valid, self.object_gap_report = prepare_object_frames(
            data[self.sequence_index], self.framerate,
        )
        if self.object_gap_report["invalid_frame_count"]:
            print("OMOMO object gaps: " + json.dumps(self.object_gap_report), flush=True)
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
            "object_valid": self.object_valid,
            "object_gap_report": self.object_gap_report,
            "source_frame_indices": np.arange(
                self.object_gap_report["retained_start"], self.object_gap_report["retained_stop"],
            ),
        }

    def _sequence_gender(self) -> str:
        value = self.sequence.get("gender", "neutral")
        if isinstance(value, np.ndarray):
            value = value.item()
        return str(value)

    def _resolve_model_directory(self) -> str:
        from .smplx_paths import smplx_model_search_paths
        for candidate in smplx_model_search_paths(self.model_directory):
            if candidate and Path(candidate).exists():
                return str(candidate)
        raise FileNotFoundError(
            "Could not locate SMPL-X model directory for OMOMO. "
            "Provide model_directory, set OMNIRETARGETING_SMPLX_MODELS, or install the models under assets/body_models."
        )

    def _sample_object_points(self) -> np.ndarray:
        vertices = np.array(self.object_mesh.vertices)
        if len(vertices) <= self.n_object_samples:
            return vertices

        from .._core.preprocessing_parallel import can_parallelize
        if can_parallelize():
            # Each pairwise distance uses the same three-coordinate norm.
            # Retain the original set iteration and tie-breaking order.
            sampled_indices = [0]
            remaining = set(range(len(vertices)))
            remaining.remove(0)
            nearest = np.full(len(vertices),np.inf)
            for _ in range(self.n_object_samples-1):
                ids = list(remaining)
                distance = np.linalg.norm(vertices[ids]-vertices[sampled_indices[-1]],axis=1)
                nearest[ids] = np.minimum(nearest[ids],distance)
                farthest_idx = ids[np.argmax(nearest[ids])]
                sampled_indices.append(farthest_idx)
                remaining.remove(farthest_idx)
            return vertices[sampled_indices]

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

    def _load_body_motion(self):
        import torch

        from .smplx import SmplxDataSource

        # OMOMO stores the SMPL-X model translation, not the pelvis position.
        # Its upstream skeleton FK uses pelvis = trans - trans2joint. Adding
        # trans2joint to the model translation applies the rest offset twice.
        # Reuse the SMPL-X adapter so skin, soles and joint frames are generated
        # from exactly the same unmodified pose parameters.
        adapter = SmplxDataSource(
            self.sequence_file, model_directory=self._resolve_model_directory(),
            gender=self._sequence_gender(), use_smplx_base_pose=self.use_smplx_base_pose,
            frame_id_semantics="provenance", source_denoise=self.source_denoise,
        )
        raw = dict(self.sequence)
        raw["pose_body"] = np.asarray(raw["pose_body"]).reshape(-1, 63)
        with torch.no_grad():
            positions, orientations, root, _, _, metadata = adapter._load_raw_npz(raw, self.framerate)
        metadata.update(adapter._foot_sole_metadata(orientations))
        return positions, orientations, root, metadata

    def _load_body_positions(self) -> np.ndarray:
        return self._load_body_motion()[0]

    def _estimate_height_from_positions(self, positions: np.ndarray, target_names: list[str]) -> float | None:
        """Estimate human height using shared utility.

        Args:
            positions: Joint positions array of shape ``(T, J, 3)``.
            target_names: List of joint names corresponding to the J axis.

        Returns:
            Estimated height in meters, or ``None`` if estimation fails.
        """
        return estimate_body_height(positions, target_names, head_joint="Head", foot_joints=("L_Foot", "R_Foot"))

    def load(self) -> MotionData:
        if self._motion_data is None:
            positions, orientations, root_orient, body_metadata = self._load_body_motion()
            object_points = self._transform_object_points()
            target_names = self.target_names or DEFAULT_SMPLX_TARGET_NAMES[: positions.shape[1]]
            object_pose = self._object_pose_data()
            object_centroid_local = np.asarray(self.object_mesh.vertices, dtype=np.float32).mean(axis=0)

            # Correct root_orient from SMPLX T-pose frame to body-aligned frame
            # and convert from axis-angle to wxyz quaternion at the boundary.
            if self.use_smplx_base_pose and root_orient is not None:
                root_orient_mat = Rotation.from_rotvec(root_orient).as_matrix()
                root_orient = Rotation.from_matrix(root_orient_mat @ _SMPLX_ROOT_OFFSET).as_quat(scalar_first=True)

            self._motion_data = MotionData(
                positions=positions,
                target_names=target_names,
                root_orientations=root_orient if self.use_smplx_base_pose else None,
                root_translations=positions[:, 0, :] if self.use_smplx_base_pose else None,
                framerate=self.framerate,
                source_height=self._estimate_height_from_positions(positions, target_names),
                object_points=object_points,
                object_mesh=self.object_mesh,
                metadata={
                    **self.metadata,
                    **body_metadata,
                    "joint_orientations": orientations,
                    "body_aligned_joint_orientations": Rotation.from_matrix(
                        Rotation.from_quat(orientations.reshape(-1, 4), scalar_first=True).as_matrix() @ _SMPLX_ROOT_OFFSET
                    ).as_quat(scalar_first=True).reshape(orientations.shape),
                    "object_mesh_path": str(self.data_root / "data" / "captured_objects" / f"{self.object_name}_cleaned_simplified.obj"),
                    "human_translation_convention": "SMPL-X model translation; no trans2joint addition",
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
    source_config = dict(source_config or {})
    runtime_options = dict(runtime_options or {})
    adapter_options = dict(source_config.get("adapter_options") or {})

    def option(*keys, default=None):
        for container in (runtime_options, adapter_options, source_config):
            for key in keys:
                if key in container and container[key] is not None:
                    return container[key]
        return default

    sequence_index = int(option("sequence_index", default=0))
    # OMOMO sequence files live in <data_root>/data/, next to captured_objects/.
    data_root = option("data_root", default=str(Path(motion_file).resolve().parent.parent))
    n_object_samples = int(option("n_object_samples", default=100))
    target_names_override = option("target_names_override")
    model_directory = option("model_directory")

    return OmomoDataSource(
        sequence_file=motion_file,
        sequence_index=sequence_index,
        data_root=data_root,
        n_object_samples=n_object_samples,
        target_names=target_names_override,
        model_directory=model_directory,
        use_smplx_base_pose=option("use_smplx_base_pose", default=True),
        framerate=float(option("framerate", default=30.0)),
        source_denoise=option("source_denoise", default="off"),
    )


from .registry import register_data_source

register_data_source("omomo", create_omomo_data_source, extensions=[".p", ".joblib"])
