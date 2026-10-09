"""Source-neutral motion data-source contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from omniretargeting.utils import linear_interpolate, slerp_interpolate
from omniretargeting.contacts import (
    Contact,
    EntityPose,
    EntityTrajectory,
    Scene,
    validate_frame_contacts,
)

@dataclass
class MotionFrame:
    positions: np.ndarray
    root_orientation: np.ndarray | None = None
    root_translation: np.ndarray | None = None
    timestamp: float | None = None
    object_points: np.ndarray | None = None
    object_mesh: Any | None = None  # trimesh.Trimesh object for visualization
    metadata: dict[str, Any] = field(default_factory=dict)
    entity_poses: dict[str, EntityPose] | None = None
    contacts: list[Contact] | None = None
    scene: Scene | None = None
    target_names: list[str] | None = None

    def __post_init__(self) -> None:
        if not validate_motion_frame_positions(self.positions):
            raise ValueError("MotionFrame.positions must have finite shape (J, 3) with J greater than zero.")
        if self.object_points is not None and not validate_object_points(self.object_points):
            raise ValueError("MotionFrame.object_points must have finite shape (N, 3) with N >= 0.")
        if self.target_names is not None and (
            len(self.target_names) != len(self.positions)
            or len(set(self.target_names)) != len(self.target_names)
        ):
            raise ValueError(
                "MotionFrame.target_names must uniquely name every position."
            )
        validate_frame_contacts(
            self.target_names, self.scene, self.entity_poses, self.contacts
        )


@dataclass
class MotionData:
    positions: np.ndarray
    target_names: list[str] | None = None
    root_orientations: np.ndarray | None = None
    root_translations: np.ndarray | None = None
    framerate: float | None = None
    source_height: float | None = None
    human_height: float | None = None
    object_points: np.ndarray | None = None
    object_mesh: Any | None = None  # trimesh.Trimesh object for visualization
    metadata: dict[str, Any] = field(default_factory=dict)
    entity_trajectories: dict[str, EntityTrajectory] | None = None
    contact_trajectory: list[list[Contact] | None] | None = None
    scene: Scene | None = None

    def __post_init__(self) -> None:
        if not validate_motion_positions(self.positions):
            raise ValueError("MotionData.positions must have finite shape (T, J, 3) with T and J greater than zero.")
        if self.target_names is not None and len(self.target_names) != self.positions.shape[1]:
            raise ValueError("MotionData.target_names length must match positions.shape[1].")
        if self.target_names is not None and len(set(self.target_names)) != len(
            self.target_names
        ):
            raise ValueError("MotionData.target_names must be unique.")
        if self.root_orientations is not None and (
            self.root_orientations.shape[0] != self.positions.shape[0]
            or self.root_orientations.shape[-1] != 4
        ):
            raise ValueError("MotionData.root_orientations must have shape (T, 4) wxyz quaternion when provided.")
        if self.root_translations is not None and self.root_translations.shape != (self.positions.shape[0], 3):
            raise ValueError("MotionData.root_translations must have shape (T, 3) when provided.")
        if self.object_points is not None:
            if self.object_points.ndim != 3 or self.object_points.shape[2] != 3:
                raise ValueError("MotionData.object_points must have shape (T, N, 3) when provided.")
            if self.object_points.shape[0] != self.positions.shape[0]:
                raise ValueError(
                    f"MotionData.object_points has {self.object_points.shape[0]} frames "
                    f"but positions has {self.positions.shape[0]} frames."
                )
            if not np.isfinite(self.object_points).all():
                raise ValueError("MotionData.object_points must contain finite values.")
        if self.source_height is None:
            self.source_height = self.human_height
        if self.human_height is None:
            self.human_height = self.source_height
        for body_id, track in (self.entity_trajectories or {}).items():
            if not isinstance(track, EntityTrajectory) or len(
                track.translations
            ) != len(self.positions):
                raise ValueError(
                    f"Entity trajectory {body_id!r} must have the same frame count as positions."
                )
        if self.contact_trajectory is not None and (
            not isinstance(self.contact_trajectory, list)
            or len(self.contact_trajectory) != len(self.positions)
        ):
            raise ValueError(
                "MotionData.contact_trajectory must be a list with exactly T entries."
            )
        for t in range(len(self.positions)):
            poses = {
                body_id: track.pose(t)
                for body_id, track in (self.entity_trajectories or {}).items()
            }
            contacts = (
                self.contact_trajectory[t]
                if self.contact_trajectory is not None
                else None
            )
            validate_frame_contacts(self.target_names, self.scene, poses, contacts)

    def copy(self) -> MotionData:
        """Copy aligned motion, contacts, poses, and scene geometry independently."""
        return deepcopy(self)

    def slice_frames(self, frame_slice: slice) -> MotionData:
        """Slice the shared timeline, copying annotations and aligned metadata."""
        if not isinstance(frame_slice, slice) or (
            frame_slice.step is not None and frame_slice.step <= 0
        ):
            raise ValueError("slice_frames requires a slice with a positive step.")
        result = self.copy()
        for name in (
            "positions",
            "root_orientations",
            "root_translations",
            "object_points",
        ):
            value = getattr(result, name)
            if value is not None:
                setattr(result, name, value[frame_slice].copy())
        result.entity_trajectories = (
            None
            if self.entity_trajectories is None
            else {
                body_id: EntityTrajectory(
                    track.translations[frame_slice].copy(),
                    track.orientations[frame_slice].copy(),
                )
                for body_id, track in self.entity_trajectories.items()
            }
        )
        if result.contact_trajectory is not None:
            result.contact_trajectory = result.contact_trajectory[frame_slice]
        result.metadata = {
            key: (
                value[frame_slice].copy()
                if isinstance(value, np.ndarray)
                and value.ndim > 0
                and len(value) == len(self.positions)
                else value
            )
            for key, value in result.metadata.items()
        }
        if result.framerate is not None:
            result.framerate /= frame_slice.step or 1
        result.__post_init__()
        return result

    def scaled(self, factor: float) -> MotionData:
        """Uniformly scale world motion and body-local geometry/anchors once."""
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("Motion scale must be finite and positive.")
        result = self.copy()
        result.positions = self.positions * factor
        if result.root_translations is not None:
            result.root_translations = self.root_translations * factor
        if result.object_points is not None:
            result.object_points = self.object_points * factor  # Already world-space.
        result.entity_trajectories = (
            None
            if self.entity_trajectories is None
            else {
                body_id: EntityTrajectory(
                    track.translations * factor, track.orientations.copy()
                )
                for body_id, track in self.entity_trajectories.items()
            }
        )
        if result.scene is not None:
            for body_id, original in self.scene.entities.items():
                entity = deepcopy(original)
                result.scene.entities[body_id] = entity
                entity.local_samples = original.local_samples * factor
                if entity.geometry is not None:
                    entity.geometry.apply_scale(factor)
                if entity.static_pose is not None:
                    entity.static_pose.translation = (
                        original.static_pose.translation * factor
                    )
        result.contact_trajectory = (
            None
            if self.contact_trajectory is None
            else [
                (
                    None
                    if contacts is None
                    else [deepcopy(contact) for contact in contacts]
                )
                for contacts in self.contact_trajectory
            ]
        )
        for contacts in result.contact_trajectory or []:
            for contact in contacts or []:
                contact.point_local = contact.point_local * factor
        if result.object_mesh is not None:
            # Visualization's legacy mesh and poses have their own scale track.
            # Keep the raw mesh, scale translations and dimensionless scale metadata.
            for key in (
                "object_translations",
                "object_centroid_world",
                "object_scales",
            ):
                if key in result.metadata:
                    result.metadata[key] = result.metadata[key] * factor
        for name in ("source_height", "human_height"):
            if getattr(result, name) is not None:
                setattr(result, name, getattr(result, name) * factor)
        result.__post_init__()
        return result

    def transformed(
        self, orientation: np.ndarray, translation: np.ndarray
    ) -> MotionData:
        """Change world frame while retaining body-local anchors and normals."""
        from scipy.spatial.transform import Rotation

        transform = EntityPose(translation, orientation)
        rotation = Rotation.from_quat(transform.orientation, scalar_first=True)

        def points(array):
            return (
                rotation.apply(array.reshape(-1, 3)).reshape(array.shape)
                + transform.translation
            )

        def orientations(array):
            return (
                (rotation * Rotation.from_quat(array.reshape(-1, 4), scalar_first=True))
                .as_quat(scalar_first=True)
                .reshape(array.shape)
            )

        result = self.copy()
        result.positions = points(result.positions)
        if result.root_translations is not None:
            result.root_translations = points(result.root_translations)
        if result.root_orientations is not None:
            result.root_orientations = orientations(result.root_orientations)
        if result.object_points is not None:
            result.object_points = points(result.object_points)
        result.entity_trajectories = (
            None
            if self.entity_trajectories is None
            else {
                body_id: EntityTrajectory(
                    points(track.translations), orientations(track.orientations)
                )
                for body_id, track in self.entity_trajectories.items()
            }
        )
        if result.scene is not None:
            for body_id, original in self.scene.entities.items():
                entity = deepcopy(original)
                result.scene.entities[body_id] = entity
                if entity.static_pose is not None:
                    entity.static_pose = EntityPose(
                        points(original.static_pose.translation),
                        orientations(original.static_pose.orientation),
                    )
        joint_orientations = result.metadata.get("joint_orientations")
        if joint_orientations is not None:
            result.metadata["joint_orientations"] = orientations(joint_orientations)
        for key in ("object_translations", "object_centroid_world"):
            if key in result.metadata:
                result.metadata[key] = points(result.metadata[key])
        if result.metadata.get("object_rotations") is not None:
            result.metadata["object_rotations"] = (
                rotation * Rotation.from_matrix(result.metadata["object_rotations"])
            ).as_matrix()
        result.__post_init__()
        return result

    def resample(self, target_framerate: float) -> MotionData:
        """Resample motion and poses, preserving rigid samples and discrete contacts."""
        if (
            self.framerate is None
            or not np.isfinite(self.framerate)
            or self.framerate <= 0
        ):
            raise ValueError("Cannot resample: source framerate is unknown.")
        if not np.isfinite(target_framerate) or target_framerate <= 0:
            raise ValueError(f"target_framerate must be positive, got {target_framerate}")
        if abs(target_framerate - self.framerate) < 1e-6:
            return self.copy()

        T = self.positions.shape[0]
        duration = (T - 1) / self.framerate
        T_new = int(np.floor(duration * target_framerate + 1e-9)) + 1
        dst_indices = np.minimum(
            np.arange(T_new) * self.framerate / target_framerate, T - 1
        )

        new_positions = linear_interpolate(self.positions, dst_indices, axis=0)
        new_root_orientations = slerp_interpolate(self.root_orientations, dst_indices, axis=0) if self.root_orientations is not None else None
        new_root_translations = linear_interpolate(self.root_translations, dst_indices, axis=0) if self.root_translations is not None else None
        new_object_points = linear_interpolate(self.object_points, dst_indices, axis=0) if self.object_points is not None else None

        new_metadata = dict(self.metadata)
        jo = new_metadata.get("joint_orientations")
        if jo is not None and isinstance(jo, np.ndarray) and jo.shape[0] == T:
            new_metadata["joint_orientations"] = slerp_interpolate(jo, dst_indices, axis=0)

        for key in (
            "object_translations",
            "object_scales",
            "recorded_object_scales",
            "object_centroid_world",
        ):
            val = new_metadata.get(key)
            if val is not None and isinstance(val, np.ndarray) and val.shape[0] == T:
                new_metadata[key] = linear_interpolate(val, dst_indices, axis=0)

        obj_rot = new_metadata.get("object_rotations")
        if obj_rot is not None and isinstance(obj_rot, np.ndarray) and obj_rot.shape[0] == T:
            from scipy.spatial.transform import Rotation, Slerp
            src_times = np.arange(T, dtype=np.float64)
            dst_times = dst_indices
            if obj_rot.ndim == 3 and obj_rot.shape[1:] == (3, 3) and T > 1:
                rotations = Rotation.from_matrix(obj_rot)
                slerp_fn = Slerp(src_times, rotations)
                new_metadata["object_rotations"] = slerp_fn(dst_times).as_matrix()
            else:
                new_metadata["object_rotations"] = linear_interpolate(obj_rot, dst_indices, axis=0)

        # Earlier source frame wins midpoint ties; entire lists are copied.
        nearest = np.clip(np.ceil(dst_indices - 0.5).astype(int), 0, T - 1)
        new_contacts = (
            None
            if self.contact_trajectory is None
            else [deepcopy(self.contact_trajectory[t]) for t in nearest]
        )
        new_tracks = (
            None
            if self.entity_trajectories is None
            else {
                body_id: EntityTrajectory(
                    linear_interpolate(track.translations, dst_indices),
                    slerp_interpolate(track.orientations, dst_indices),
                )
                for body_id, track in self.entity_trajectories.items()
            }
        )
        legacy_body_id = self.metadata.get("object_points_body_id")
        if (
            new_object_points is not None
            and self.scene is not None
            and legacy_body_id in self.scene.entities
        ):
            # Rigid samples must follow the same interpolated pose as contacts.
            # Interpolating world points instead cuts chords through rotations.
            entity = self.scene.entities[legacy_body_id]
            track = (new_tracks or {}).get(legacy_body_id)
            if len(entity.local_samples):
                local_points = np.broadcast_to(
                    entity.local_samples, (T_new, *entity.local_samples.shape)
                )
            else:
                from scipy.spatial.transform import Rotation

                # Geometry-only entities can still own legacy world samples.
                # Recover their local positions without changing the scene.
                original_track = (self.entity_trajectories or {}).get(legacy_body_id)
                original_local_points = []
                for t, points in enumerate(self.object_points):
                    pose = (
                        original_track.pose(t)
                        if original_track is not None
                        else self.scene.pose(legacy_body_id, {})
                    )
                    original_local_points.append(
                        Rotation.from_quat(pose.orientation, scalar_first=True)
                        .inv()
                        .apply(points - pose.translation)
                    )
                local_points = linear_interpolate(
                    np.stack(original_local_points), dst_indices
                )
            new_object_points = np.stack(
                [
                    (
                        track.pose(t)
                        if track is not None
                        else self.scene.pose(legacy_body_id, {})
                    ).to_world(local_points[t])
                    for t in range(T_new)
                ]
            )
        return MotionData(
            positions=new_positions,
            target_names=self.target_names,
            root_orientations=new_root_orientations,
            root_translations=new_root_translations,
            framerate=target_framerate,
            source_height=self.source_height,
            human_height=self.human_height,
            object_points=new_object_points,
            object_mesh=self.object_mesh,
            metadata=new_metadata,
            entity_trajectories=new_tracks,
            contact_trajectory=new_contacts,
            scene=deepcopy(self.scene),
        )

    def iter_frames(self) -> Iterator[MotionFrame]:
        for frame_idx, positions in enumerate(self.positions):
            root_orientation = self.root_orientations[frame_idx] if self.root_orientations is not None else None
            root_translation = self.root_translations[frame_idx] if self.root_translations is not None else None
            object_points_frame = self.object_points[frame_idx] if self.object_points is not None else None
            yield MotionFrame(
                positions=positions,
                root_orientation=root_orientation,
                root_translation=root_translation,
                timestamp=(frame_idx / self.framerate) if self.framerate else None,
                object_points=object_points_frame,
                object_mesh=self.object_mesh,
                metadata={"frame_index": frame_idx, **self.metadata},
                entity_poses=(
                    None
                    if self.entity_trajectories is None
                    else {
                        body_id: track.pose(frame_idx)
                        for body_id, track in self.entity_trajectories.items()
                    }
                ),
                contacts=(
                    None
                    if self.contact_trajectory is None
                    else self.contact_trajectory[frame_idx]
                ),
                scene=self.scene,
                target_names=self.target_names,
            )


class DataSource(ABC):
    target_names: list[str] | None = None
    framerate: float | None = None
    source_height: float | None = None
    human_height: float | None = None
    metadata: dict[str, Any]

    @abstractmethod
    def iter_frames(self) -> Iterator[MotionFrame]:
        raise NotImplementedError

    def load(self) -> MotionData:
        frames = list(self.iter_frames())
        if not frames:
            raise ValueError("DataSource produced no frames.")
        positions = np.stack([frame.positions for frame in frames], axis=0)
        root_orientations = _stack_optional([frame.root_orientation for frame in frames])
        root_translations = _stack_optional([frame.root_translation for frame in frames])
        object_points = _stack_optional([frame.object_points for frame in frames])
        pose_keys = set(frames[0].entity_poses or {})
        if any(set(frame.entity_poses or {}) != pose_keys for frame in frames):
            raise ValueError(
                "DataSource entity pose IDs must be consistent across frames."
            )
        entity_trajectories = (
            None
            if not pose_keys
            else {
                body_id: EntityTrajectory(
                    np.stack(
                        [frame.entity_poses[body_id].translation for frame in frames]
                    ),
                    np.stack(
                        [frame.entity_poses[body_id].orientation for frame in frames]
                    ),
                )
                for body_id in pose_keys
            }
        )
        contacts = [deepcopy(frame.contacts) for frame in frames]
        source_height = getattr(self, "source_height", None)
        if source_height is None:
            source_height = getattr(self, "human_height", None)
        return MotionData(
            positions=positions,
            target_names=self.target_names,
            root_orientations=root_orientations,
            root_translations=root_translations,
            framerate=self.framerate,
            source_height=source_height,
            object_points=object_points,
            object_mesh=frames[0].object_mesh,
            metadata=dict(getattr(self, "metadata", {})),
            entity_trajectories=entity_trajectories,
            contact_trajectory=contacts,
            scene=frames[0].scene,
        )


def validate_motion_frame_positions(positions: np.ndarray) -> bool:
    if not isinstance(positions, np.ndarray):
        return False
    if positions.ndim != 2:
        return False
    num_targets, num_coords = positions.shape
    if num_coords != 3:
        return False
    if num_targets == 0:
        return False
    return bool(np.isfinite(positions).all())


def validate_motion_positions(positions: np.ndarray) -> bool:
    if not isinstance(positions, np.ndarray):
        return False
    if positions.ndim != 3:
        return False
    num_frames, num_targets, num_coords = positions.shape
    if num_coords != 3:
        return False
    if num_frames == 0 or num_targets == 0:
        return False
    return bool(np.isfinite(positions).all())


def validate_object_points(points: np.ndarray) -> bool:
    """Validate object points array (N, 3) for a single frame."""
    if not isinstance(points, np.ndarray):
        return False
    if points.ndim != 2 or points.shape[1] != 3:
        return False
    return bool(np.isfinite(points).all())


def _stack_optional(values: list[np.ndarray | None]) -> np.ndarray | None:
    if any(value is None for value in values):
        return None
    return np.stack(values, axis=0)
