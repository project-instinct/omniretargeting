"""PARC ``MSFileData`` pickle data-source adapter.

The on-disk format is the ``parc_ms`` layout used by human-humanoid-tools:
one pickle containing separately pickled ``motion_data``/``terrain_data``
blobs, plus a sibling ``*_terrain.obj``.  The source files do not contain a
skeleton, so this module bakes the 15-body PARC ``humanoid.xml`` rest rig and
performs the same forward kinematics as the hhtools adapter.

The PARC skeleton constants and FK convention are derived from the
Apache-2.0-licensed human-humanoid-tools implementation:
https://github.com/Roboparty/human-humanoid-tools
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from omniretargeting.data_sources.base import DataSource, MotionData
from omniretargeting.data_sources.registry import register_data_source


PARC_MS_BONE_NAMES: tuple[str, ...] = (
    "pelvis",
    "torso",
    "head",
    "right_upper_arm",
    "right_lower_arm",
    "right_hand",
    "left_upper_arm",
    "left_lower_arm",
    "left_hand",
    "right_thigh",
    "right_shin",
    "right_foot",
    "left_thigh",
    "left_shin",
    "left_foot",
)

# Canonical semantic labels let existing robot profiles map this reduced rig
# without pretending that the input itself is SMPL-X.  PARC's distal foot body
# origin is the ankle joint, hence the L/R_Ankle names.
PARC_MS_TARGET_NAMES: tuple[str, ...] = (
    "Pelvis",
    "Spine1",
    "Head",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
)

PARC_MS_PARENT_INDICES = np.asarray(
    [-1, 0, 1, 1, 3, 4, 1, 6, 7, 0, 9, 10, 0, 12, 13],
    dtype=np.int32,
)

# Rest offsets in the parent body's local frame, in metres.  PARC's body bind
# rotations are all identity.
PARC_MS_LOCAL_TRANSLATIONS = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.236151],
        [0.0, 0.0, 0.223894],
        [-0.02405, -0.18311, 0.24350],
        [0.0, -0.274788, 0.0],
        [0.0, -0.258947, 0.0],
        [-0.02405, 0.18311, 0.24350],
        [0.0, 0.274788, 0.0],
        [0.0, 0.258947, 0.0],
        [0.0, -0.084887, 0.0],
        [0.0, 0.0, -0.421546],
        [0.0, 0.0, -0.409870],
        [0.0, 0.084887, 0.0],
        [0.0, 0.0, -0.421546],
        [0.0, 0.0, -0.409870],
    ],
    dtype=np.float32,
)

PARC_MS_FOOT_CONTACT_OFFSET_M = 0.05

# Source-specific defaults belong to the adapter, not to every clip config.
# PARC's 15-body skeleton deliberately has no Spine2/Spine3 bodies, so using
# an SMPL-X robot-profile mapping directly would reject an otherwise valid
# clip before retargeting starts.
PARC_MS_JOINT_MAPPING = {
    "Pelvis": "pelvis",
    "Spine1": "torso_link",
    "L_Hip": "left_hip_roll_link",
    "R_Hip": "right_hip_roll_link",
    "L_Knee": "left_knee_link",
    "R_Knee": "right_knee_link",
    "L_Ankle": "left_ankle_roll_link",
    "R_Ankle": "right_ankle_roll_link",
    "L_Shoulder": "left_shoulder_roll_link",
    "R_Shoulder": "right_shoulder_roll_link",
    "L_Elbow": "left_elbow_link",
    "R_Elbow": "right_elbow_link",
    "L_Wrist": "left_wrist_roll_link",
    "R_Wrist": "right_wrist_roll_link",
}

PARC_MS_BASE_ORIENTATION = {
    "pelvis": "Pelvis",
    "left_hip": "L_Hip",
    "right_hip": "R_Hip",
    "spine": "Spine1",
    "root_position": "Spine1",
}


def _normalize_quaternions_xyzw(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("PARC motion contains a zero-length quaternion.")
    return values / norms


def _multiply_quaternions_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for arrays stored scalar-last (xyzw)."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    xyz = (
        left[..., 3:4] * right[..., :3]
        + right[..., 3:4] * left[..., :3]
        + np.cross(left[..., :3], right[..., :3])
    )
    scalar = (
        left[..., 3:4] * right[..., 3:4]
        - np.sum(left[..., :3] * right[..., :3], axis=-1, keepdims=True)
    )
    return np.concatenate([xyz, scalar], axis=-1)


def _rotate_vectors_xyzw(quaternions: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    quaternions = _normalize_quaternions_xyzw(quaternions)
    vectors = np.asarray(vectors, dtype=np.float64)
    twice_cross = 2.0 * np.cross(quaternions[..., :3], vectors)
    return (
        vectors
        + quaternions[..., 3:4] * twice_cross
        + np.cross(quaternions[..., :3], twice_cross)
    )


def forward_kinematics_parc_ms(
    root_positions: np.ndarray,
    root_rotations_xyzw: np.ndarray,
    joint_rotations_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct PARC body positions and world rotations at metric scale."""
    root_positions = np.asarray(root_positions, dtype=np.float64)
    root_rotations_xyzw = _normalize_quaternions_xyzw(root_rotations_xyzw)
    joint_rotations_xyzw = np.asarray(joint_rotations_xyzw, dtype=np.float64)

    if root_positions.ndim != 2 or root_positions.shape[1] != 3:
        raise ValueError(f"root_pos must have shape (T, 3), got {root_positions.shape}.")
    num_frames = len(root_positions)
    if root_rotations_xyzw.shape != (num_frames, 4):
        raise ValueError(
            f"root_rot must have shape ({num_frames}, 4), got "
            f"{root_rotations_xyzw.shape}."
        )
    if joint_rotations_xyzw.shape == (num_frames, len(PARC_MS_BONE_NAMES), 4):
        joint_rotations_xyzw = joint_rotations_xyzw[:, :-1]
    expected_joint_shape = (num_frames, len(PARC_MS_BONE_NAMES) - 1, 4)
    if joint_rotations_xyzw.shape != expected_joint_shape:
        raise ValueError(
            f"joint_rot must have shape {expected_joint_shape}, got "
            f"{joint_rotations_xyzw.shape}."
        )
    joint_rotations_xyzw = _normalize_quaternions_xyzw(joint_rotations_xyzw)

    positions = np.zeros((num_frames, len(PARC_MS_BONE_NAMES), 3), dtype=np.float64)
    world_rotations = np.zeros(
        (num_frames, len(PARC_MS_BONE_NAMES), 4), dtype=np.float64
    )
    positions[:, 0] = root_positions
    world_rotations[:, 0] = root_rotations_xyzw
    for joint_index in range(1, len(PARC_MS_BONE_NAMES)):
        parent_index = int(PARC_MS_PARENT_INDICES[joint_index])
        positions[:, joint_index] = positions[:, parent_index] + _rotate_vectors_xyzw(
            world_rotations[:, parent_index],
            np.broadcast_to(
                PARC_MS_LOCAL_TRANSLATIONS[joint_index], (num_frames, 3)
            ),
        )
        world_rotations[:, joint_index] = _normalize_quaternions_xyzw(
            _multiply_quaternions_xyzw(
                world_rotations[:, parent_index],
                joint_rotations_xyzw[:, joint_index - 1],
            )
        )

    return positions.astype(np.float32), world_rotations.astype(np.float32)


def load_parc_ms_pickle(
    motion_file: str | Path,
) -> tuple[dict[str, Any], Any, Any]:
    """Load the top-level MS container and its separately pickled blobs.

    Pickle is executable data.  Callers should only use files from a trusted
    dataset source, just as they would with the upstream hhtools loader.
    """
    path = Path(motion_file)
    with path.open("rb") as file:
        container = pickle.load(file)
    if not isinstance(container, dict):
        raise ValueError(f"{path}: expected an MS container dict, got {type(container)}.")

    def load_blob(key: str) -> Any:
        value = container.get(key)
        if value is None:
            return None
        if not isinstance(value, (bytes, bytearray)):
            raise ValueError(f"{path}: {key!r} must be a pickled byte blob.")
        return pickle.loads(value)

    motion = load_blob("motion_data")
    if not isinstance(motion, dict):
        raise ValueError(f"{path}: missing or invalid motion_data blob.")
    return motion, load_blob("terrain_data"), load_blob("misc_data")


def _leading_bind_frame_count(root_positions: np.ndarray) -> int:
    """Match hhtools' removal of origin-only authoring lead-in frames."""
    for frame_index, position in enumerate(root_positions):
        if float(np.linalg.norm(position[:2])) > 1e-3:
            return frame_index
    return min(1, max(0, len(root_positions) - 1))


def _end_effector_markers(
    positions: np.ndarray,
    world_rotations_xyzw: np.ndarray,
    body_contacts: np.ndarray | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    markers: dict[str, dict[str, np.ndarray]] = {}
    for name, joint_index in (("left_hand", 8), ("right_hand", 5)):
        joint_positions = np.asarray(positions[:, joint_index], dtype=float)
        markers[name] = {
            "tracking_positions": joint_positions,
            "contact_samples": joint_positions[:, None, :],
        }
        # Hand labels reconstructed from video are only advisory and can mark
        # near passes as contact. Keep hands on scene-based inference; PARC's
        # explicit support labels are consumed only for the feet below.

    local_forward = np.asarray([1.0, 0.0, 0.0])
    local_up = np.asarray([0.0, 0.0, 1.0])
    for name, joint_index in (("left_foot", 14), ("right_foot", 11)):
        rotations = world_rotations_xyzw[:, joint_index]
        forward = _rotate_vectors_xyzw(
            rotations, np.broadcast_to(local_forward, (len(rotations), 3))
        )
        up = _rotate_vectors_xyzw(
            rotations, np.broadcast_to(local_up, (len(rotations), 3))
        )
        joint_positions = np.asarray(positions[:, joint_index], dtype=float)
        sole_positions = joint_positions - PARC_MS_FOOT_CONTACT_OFFSET_M * up
        markers[name] = {
            # PARC's distal foot body is the ankle origin; the interaction
            # point is the sole, 5 cm below it.  World tracking must therefore
            # use the sole too, or it would deliberately float the robot foot.
            "tracking_positions": sole_positions,
            "contact_samples": sole_positions[:, None, :],
            "forward_directions": forward,
            "up_directions": up,
        }
        if body_contacts is not None:
            markers[name]["source_contact_mask"] = np.asarray(
                body_contacts[:, joint_index], dtype=bool
            )
    return markers


def _terrain_mesh_path(motion_file: Path) -> Path | None:
    candidate = motion_file.with_name(f"{motion_file.stem}_terrain.obj")
    return candidate if candidate.is_file() else None


@dataclass
class ParcMsDataSource(DataSource):
    motion_file: Path
    trim_leading_bind_frames: bool = True

    def __post_init__(self) -> None:
        self.motion_file = Path(self.motion_file)
        if not self.motion_file.is_file():
            raise FileNotFoundError(self.motion_file)
        self._motion_data: MotionData | None = None

    @property
    def target_names(self) -> list[str]:
        return list(PARC_MS_TARGET_NAMES)

    @property
    def framerate(self) -> float | None:
        return self.load().framerate

    @property
    def source_height(self) -> float | None:
        return self.load().source_height

    @property
    def human_height(self) -> float | None:
        return self.source_height

    def load(self) -> MotionData:
        if self._motion_data is not None:
            return self._motion_data

        motion, terrain, misc = load_parc_ms_pickle(self.motion_file)
        required = ("root_pos", "root_rot", "joint_rot")
        missing = [key for key in required if key not in motion]
        if missing:
            raise ValueError(f"{self.motion_file}: motion_data is missing {missing}.")

        root_positions = np.asarray(motion["root_pos"], dtype=np.float32)
        root_rotations = np.asarray(motion["root_rot"], dtype=np.float32)
        joint_rotations = np.asarray(motion["joint_rot"], dtype=np.float32)
        trim_count = (
            _leading_bind_frame_count(root_positions)
            if self.trim_leading_bind_frames
            else 0
        )
        if trim_count:
            root_positions = root_positions[trim_count:]
            root_rotations = root_rotations[trim_count:]
            joint_rotations = joint_rotations[trim_count:]

        positions, world_rotations_xyzw = forward_kinematics_parc_ms(
            root_positions, root_rotations, joint_rotations
        )
        rest_root_position = np.zeros((1, 3), dtype=np.float32)
        rest_root_rotation = np.asarray(
            [[0.0, 0.0, 0.0, 1.0]], dtype=np.float32
        )
        rest_joint_rotations = np.zeros((1, 14, 4), dtype=np.float32)
        rest_joint_rotations[..., 3] = 1.0
        rest_positions, _ = forward_kinematics_parc_ms(
            rest_root_position,
            rest_root_rotation,
            rest_joint_rotations,
        )
        terrain_path = _terrain_mesh_path(self.motion_file)
        body_contacts = motion.get("body_contacts")
        if body_contacts is not None:
            body_contacts = np.asarray(body_contacts)[trim_count:]
            expected_contact_shape = (
                len(root_positions),
                len(PARC_MS_BONE_NAMES),
            )
            if body_contacts.shape != expected_contact_shape:
                raise ValueError(
                    f"{self.motion_file}: body_contacts must have shape "
                    f"{expected_contact_shape}, got {body_contacts.shape}."
                )

        # The rest skeleton spans ankle to head by 1.291 m.  Add the known
        # ankle-to-sole distance and a conservative head-radius allowance.
        source_height = float(
            np.max(
                positions[:, 2, 2]
                - np.minimum(positions[:, 11, 2], positions[:, 14, 2])
            )
            + PARC_MS_FOOT_CONTACT_OFFSET_M
            + 0.12
        )
        metadata: dict[str, Any] = {
            "source_type": "parc_ms",
            "source_format": "parc_ms_pkl",
            "source_file": str(self.motion_file.resolve()),
            "source_bone_names": list(PARC_MS_BONE_NAMES),
            "source_parent_indices": PARC_MS_PARENT_INDICES.copy(),
            "joint_orientations": world_rotations_xyzw[..., [3, 0, 1, 2]],
            # PARC-MS stores every local joint rotation relative to an
            # identity-orientation bind skeleton.  Exposing that reference is
            # what lets a retargeter calibrate source body frames to a robot's
            # differently oriented link frames instead of assuming their axes
            # already match.
            "joint_rest_orientations": np.tile(
                np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                (len(PARC_MS_TARGET_NAMES), 1),
            ),
            "joint_rest_positions": rest_positions[0],
            "end_effector_markers": _end_effector_markers(
                positions, world_rotations_xyzw, body_contacts
            ),
            "terrain_mesh": str(terrain_path.resolve()) if terrain_path else None,
            "terrain_data_present": terrain is not None,
            "misc_data_present": misc is not None,
            "loop_mode": str(motion.get("loop_mode", "CLAMP")),
            "trimmed_leading_bind_frames": trim_count,
            "joint_mapping": dict(PARC_MS_JOINT_MAPPING),
            "base_orientation": dict(PARC_MS_BASE_ORIENTATION),
            # PARC has one long pelvis-to-torso segment followed by a torso
            # body, while G1's rigid torso represents most of that complete
            # silhouette. A pelvis/shoulder landmark frame is therefore the
            # semantic default; callers can still override it in source config.
            "torso_mapping": "pelvis_shoulders",
        }
        if body_contacts is not None:
            metadata["body_contacts"] = body_contacts

        # MotionData uses scalar-first quaternions at its source-neutral API.
        root_rotations_wxyz = _normalize_quaternions_xyzw(root_rotations)[
            ..., [3, 0, 1, 2]
        ].astype(np.float32)
        self._motion_data = MotionData(
            positions=positions,
            target_names=self.target_names,
            root_orientations=root_rotations_wxyz,
            root_translations=root_positions.copy(),
            framerate=float(np.asarray(motion.get("fps", 30.0)).item()),
            source_height=source_height,
            metadata=metadata,
        )
        return self._motion_data

    def iter_frames(self):
        yield from self.load().iter_frames()


def create_parc_ms_data_source(
    motion_file: Path,
    source_config: Mapping[str, Any] | None = None,
    runtime_options: Mapping[str, Any] | None = None,
) -> ParcMsDataSource:
    source_config = dict(source_config or {})
    runtime_options = dict(runtime_options or {})
    adapter_options = dict(source_config.get("adapter_options") or {})

    trim = runtime_options.get(
        "trim_leading_bind_frames",
        adapter_options.get(
            "trim_leading_bind_frames",
            source_config.get("trim_leading_bind_frames", True),
        ),
    )
    return ParcMsDataSource(
        motion_file=motion_file,
        trim_leading_bind_frames=bool(trim),
    )


register_data_source("parc_ms", create_parc_ms_data_source, extensions=[".pkl"])


__all__ = [
    "PARC_MS_BONE_NAMES",
    "PARC_MS_FOOT_CONTACT_OFFSET_M",
    "PARC_MS_LOCAL_TRANSLATIONS",
    "PARC_MS_PARENT_INDICES",
    "PARC_MS_TARGET_NAMES",
    "ParcMsDataSource",
    "forward_kinematics_parc_ms",
    "load_parc_ms_pickle",
]
