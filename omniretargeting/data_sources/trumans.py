"""TRUMANS recordings as a registered source for the main pipeline, metric Z-up.

The recorded Y-up SMPL-X parameters are expressed in Z-up by the rigid world
rotation of ``trumans_review`` and evaluated by ``SmplxDataSource``, so skin,
soles, markers and joint frames are generated exactly as for every other
SMPL-X source. The released joints remain the immutable skeleton reference
(``trumans_review.reconstruct_source``). The room mesh and every recorded
object keep their original geometry and per-frame poses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from omniretargeting.utils import estimate_body_height

from .base import DataSource, MotionData
from .smplx import _SMPLX_ROOT_OFFSET, DEFAULT_SMPLX_TARGET_NAMES
from .trumans_review import WORLD_ROTATION, load_scene

# The release stores joints with slightly different numeric precision; the
# reconstruction must still agree with them (``reconstruct_source``).
RECONSTRUCTION_TOLERANCE_M = 1e-3


@dataclass
class TrumansDataSource(DataSource):
    data_root: Path
    segment: str
    model_directory: str | None = None
    framerate: float = 30.0
    frame_start: int = 0
    frame_stop: int | None = None

    def __post_init__(self):
        self.data_root = Path(self.data_root)
        self.target_names = list(DEFAULT_SMPLX_TARGET_NAMES)
        self.metadata = {}
        self._motion_data = None

    def _clip(self):
        """Segment rows, room mesh and objects of the requested frame range."""
        indices, scene, objects = load_scene(self.data_root, self.segment)
        clip = slice(self.frame_start, self.frame_stop)
        if not len(indices[clip]):
            raise ValueError(f"Empty TRUMANS frame range {self.frame_start}:{self.frame_stop}")
        objects = [dict(obj, translations=obj["translations"][clip], rotations=obj["rotations"][clip])
                   for obj in objects]
        return indices[clip], scene, objects

    def _parameters(self, indices):
        """Recorded parameters with the world rotation applied to the root.

        SMPL-X rotates the body about its rest pelvis ``j0`` before adding the
        translation, so the rotated translation is ``W (j0 + t) - j0``.
        """
        import smplx
        import torch

        arrays = {name: np.asarray(np.load(self.data_root / f"{name}.npy", mmap_mode="r")[indices], dtype=np.float64)
                  for name in ("human_orient", "human_pose", "human_transl", "left_hand_pose", "right_hand_pose")}
        model = smplx.create(str(self.model_directory), "smplx", gender="male", use_pca=False)
        with torch.no_grad():
            j0 = model(betas=torch.zeros((1, 10))).joints[0, 0].cpu().numpy().astype(np.float64)
        world = Rotation.from_matrix(WORLD_ROTATION)
        return {
            "root_orient": (world * Rotation.from_rotvec(arrays["human_orient"])).as_rotvec(),
            "trans": (j0 + arrays["human_transl"]) @ WORLD_ROTATION.T - j0,
            "pose_body": arrays["human_pose"].reshape(len(indices), 63),
            "pose_hand": np.concatenate([arrays["left_hand_pose"], arrays["right_hand_pose"]], axis=1),
            "betas": np.zeros(10),
            "gender": "male",
        }

    def load(self) -> MotionData:
        if self._motion_data is not None:
            return self._motion_data
        from .smplx import SmplxDataSource

        indices, scene, objects = self._clip()
        adapter = SmplxDataSource(self.data_root / "seg_name.npy", model_directory=self.model_directory,
                                  gender="male", use_smplx_base_pose=True, frame_id_semantics="provenance")
        positions, orientations, root_orient, _, _, body = adapter._load_raw_npz(
            self._parameters(indices), self.framerate)
        body.update(adapter._foot_sole_metadata(orientations))
        recorded = np.asarray(np.load(self.data_root / "human_joints.npy", mmap_mode="r")[indices, :22],
                              dtype=np.float64) @ WORLD_ROTATION.T
        error = float(np.linalg.norm(positions - recorded, axis=-1).max())
        if error > RECONSTRUCTION_TOLERANCE_M:
            raise ValueError(f"SMPL-X reconstruction differs from recorded joints by {error} m")
        root = Rotation.from_matrix(Rotation.from_rotvec(root_orient).as_matrix() @ _SMPLX_ROOT_OFFSET)
        aligned = Rotation.from_quat(orientations.reshape(-1, 4), scalar_first=True).as_matrix() @ _SMPLX_ROOT_OFFSET
        self._motion_data = MotionData(
            positions=recorded,
            target_names=self.target_names,
            root_orientations=root.as_quat(scalar_first=True),
            root_translations=recorded[:, 0].copy(),
            framerate=self.framerate,
            source_height=estimate_body_height(recorded, self.target_names, head_joint="Head",
                                               foot_joints=("L_Foot", "R_Foot")),
            metadata={
                **body,
                "source_type": "trumans",
                "segment": self.segment,
                "source_rows": indices,
                "reconstruction_error_m": error,
                "joint_orientations": orientations,
                "body_aligned_joint_orientations": Rotation.from_matrix(aligned).as_quat(
                    scalar_first=True).reshape(orientations.shape),
                "terrain_mesh": scene,
                "scene_objects": objects,
                # TRUMANS room meshes delimit the navigable interior, with
                # obstacle shells oppositely oriented: a geometry convention.
                "scene_enclosed_free_space": True,
            },
        )
        return self._motion_data

    def iter_frames(self):
        yield from self.load().iter_frames()


def create_trumans_data_source(motion_file, source_config, runtime_options):
    """``motion`` is the release directory (``Data_release``); ``segment`` names the clip."""
    options = {**dict(source_config or {}), **dict(source_config.get("adapter_options") or {}),
               **dict(runtime_options or {})}
    if not options.get("segment"):
        raise ValueError("A TRUMANS source requires 'segment'")
    stop = options.get("frame_stop")
    return TrumansDataSource(
        data_root=Path(options.get("data_root") or motion_file),
        segment=str(options["segment"]),
        model_directory=options.get("model_directory"),
        framerate=float(options.get("framerate", 30.0)),
        frame_start=int(options.get("frame_start", 0)),
        frame_stop=None if stop is None else int(stop),
    )


from .registry import register_data_source  # noqa: E402

register_data_source("trumans", create_trumans_data_source)
