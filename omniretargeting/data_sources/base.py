"""Source-neutral motion data-source contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from omniretargeting.utils import linear_interpolate, slerp_interpolate


@dataclass
class MotionFrame:
    positions: np.ndarray
    root_orientation: np.ndarray | None = None
    root_translation: np.ndarray | None = None
    timestamp: float | None = None
    object_points: np.ndarray | None = None
    object_mesh: Any | None = None  # trimesh.Trimesh object for visualization
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not validate_motion_frame_positions(self.positions):
            raise ValueError("MotionFrame.positions must have finite shape (J, 3) with J greater than zero.")
        if self.object_points is not None and not validate_object_points(self.object_points):
            raise ValueError("MotionFrame.object_points must have finite shape (N, 3) with N >= 0.")


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

    def __post_init__(self) -> None:
        if not validate_motion_positions(self.positions):
            raise ValueError("MotionData.positions must have finite shape (T, J, 3) with T and J greater than zero.")
        if self.target_names is not None and len(self.target_names) != self.positions.shape[1]:
            raise ValueError("MotionData.target_names length must match positions.shape[1].")
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

    def resample(self, target_framerate: float) -> MotionData:
        """Return a new MotionData resampled to *target_framerate* via linear interpolation."""
        if self.framerate is None or self.framerate <= 0:
            raise ValueError("Cannot resample: source framerate is unknown.")
        if target_framerate <= 0:
            raise ValueError(f"target_framerate must be positive, got {target_framerate}")
        if abs(target_framerate - self.framerate) < 1e-6:
            return MotionData(
                positions=self.positions.copy(),
                target_names=self.target_names,
                root_orientations=self.root_orientations.copy() if self.root_orientations is not None else None,
                root_translations=self.root_translations.copy() if self.root_translations is not None else None,
                framerate=self.framerate,
                source_height=self.source_height,
                human_height=self.human_height,
                object_points=self.object_points.copy() if self.object_points is not None else None,
                object_mesh=self.object_mesh,
                metadata=dict(self.metadata),
            )

        T = self.positions.shape[0]
        duration = (T - 1) / self.framerate
        T_new = max(int(round(duration * target_framerate)) + 1, 2)

        dst_indices = np.linspace(0, T - 1, T_new)

        new_positions = linear_interpolate(self.positions, dst_indices, axis=0)
        new_root_orientations = slerp_interpolate(self.root_orientations, dst_indices, axis=0) if self.root_orientations is not None else None
        new_root_translations = linear_interpolate(self.root_translations, dst_indices, axis=0) if self.root_translations is not None else None
        new_object_points = linear_interpolate(self.object_points, dst_indices, axis=0) if self.object_points is not None else None

        new_metadata = dict(self.metadata)
        jo = new_metadata.get("joint_orientations")
        if jo is not None and isinstance(jo, np.ndarray) and jo.shape[0] == T:
            new_metadata["joint_orientations"] = slerp_interpolate(jo, dst_indices, axis=0)

        for key in ("object_translations", "object_scales"):
            val = new_metadata.get(key)
            if val is not None and isinstance(val, np.ndarray) and val.shape[0] == T:
                new_metadata[key] = linear_interpolate(val, dst_indices, axis=0)

        # Semantic per-frame metadata has to follow the resampling too. A
        # trajectory that keeps its original length while positions are
        # resampled is silently misaligned in time, and nothing downstream can
        # detect that: the arrays are still well formed, just describing
        # different instants.
        centres = new_metadata.get("foot_sole_centres")
        if isinstance(centres, dict):
            new_metadata["foot_sole_centres"] = {
                name: linear_interpolate(np.asarray(series), dst_indices, axis=0)
                if getattr(series, "shape", (0,))[0] == T else series
                for name, series in centres.items()
            }

        rotations = new_metadata.get("link_orientations")
        if isinstance(rotations, dict):
            from scipy.spatial.transform import Rotation, Slerp
            src_times = np.arange(T, dtype=np.float64)
            resampled = {}
            for name, series in rotations.items():
                series = np.asarray(series)
                if series.ndim == 3 and series.shape[1:] == (3, 3) and series.shape[0] == T:
                    # Rotations interpolate on the sphere; component-wise linear
                    # interpolation of a rotation matrix leaves SO(3).
                    resampled[name] = Slerp(src_times, Rotation.from_matrix(series))(
                        dst_indices
                    ).as_matrix()
                else:
                    resampled[name] = series
            new_metadata["link_orientations"] = resampled

        obj_rot = new_metadata.get("object_rotations")
        if obj_rot is not None and isinstance(obj_rot, np.ndarray) and obj_rot.shape[0] == T:
            from scipy.spatial.transform import Rotation, Slerp
            src_times = np.arange(T, dtype=np.float64)
            dst_times = dst_indices
            if obj_rot.ndim == 3 and obj_rot.shape[1:] == (3, 3):
                rotations = Rotation.from_matrix(obj_rot)
                slerp_fn = Slerp(src_times, rotations)
                new_metadata["object_rotations"] = slerp_fn(dst_times).as_matrix()
            else:
                new_metadata["object_rotations"] = linear_interpolate(obj_rot, dst_indices, axis=0)

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
        )

    def iter_frames(self) -> Iterator[MotionFrame]:
        sole_rotations = self.metadata.get("link_orientations")
        contact_targets = self.metadata.get("contact_targets")
        contact_normals = self.metadata.get("contact_normals")
        contact_weights = self.metadata.get("contact_weights")
        for frame_idx, positions in enumerate(self.positions):
            root_orientation = self.root_orientations[frame_idx] if self.root_orientations is not None else None
            root_translation = self.root_translations[frame_idx] if self.root_translations is not None else None
            object_points_frame = self.object_points[frame_idx] if self.object_points is not None else None
            frame_metadata = {"frame_index": frame_idx, **self.metadata}
            if sole_rotations:
                # Hand the frame its own slice rather than the whole
                # trajectory, so a per-frame consumer cannot accidentally read
                # another frame's orientation.
                frame_metadata["link_orientations"] = {
                    side: rotations[frame_idx]
                    for side, rotations in sole_rotations.items()
                    if frame_idx < len(rotations)
                }
            if contact_targets:
                frame_contacts = {}
                for name, series in contact_targets.items():
                    if frame_idx < len(series) and np.all(np.isfinite(series[frame_idx])):
                        frame_contacts[name] = series[frame_idx]
                if frame_contacts:
                    frame_metadata["contact_targets"] = frame_contacts
                    if contact_weights:
                        frame_metadata["contact_weights"] = {
                            name: float(contact_weights[name][frame_idx])
                            for name in frame_contacts
                            if name in contact_weights
                            and frame_idx < len(contact_weights[name])
                        }
                    if contact_normals:
                        frame_metadata["contact_normals"] = {
                            name: contact_normals[name][frame_idx]
                            for name in frame_contacts
                            if name in contact_normals
                            and frame_idx < len(contact_normals[name])
                        }
            yield MotionFrame(
                positions=positions,
                root_orientation=root_orientation,
                root_translation=root_translation,
                timestamp=(frame_idx / self.framerate) if self.framerate else None,
                object_points=object_points_frame,
                metadata=frame_metadata,
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
            metadata=dict(getattr(self, "metadata", {})),
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
