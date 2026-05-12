"""Source-neutral motion data-source contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np


@dataclass
class MotionFrame:
    positions: np.ndarray
    root_orientation: np.ndarray | None = None
    root_translation: np.ndarray | None = None
    timestamp: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not validate_motion_frame_positions(self.positions):
            raise ValueError("MotionFrame.positions must have finite shape (J, 3) with J greater than zero.")


@dataclass
class MotionData:
    positions: np.ndarray
    target_names: list[str] | None = None
    root_orientations: np.ndarray | None = None
    root_translations: np.ndarray | None = None
    framerate: float | None = None
    source_height: float | None = None
    human_height: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not validate_motion_positions(self.positions):
            raise ValueError("MotionData.positions must have finite shape (T, J, 3) with T and J greater than zero.")
        if self.target_names is not None and len(self.target_names) != self.positions.shape[1]:
            raise ValueError("MotionData.target_names length must match positions.shape[1].")
        if self.root_orientations is not None and (
            self.root_orientations.shape[0] != self.positions.shape[0]
            or self.root_orientations.shape[-1] != 3
        ):
            raise ValueError("MotionData.root_orientations must have shape (T, 3) when provided.")
        if self.root_translations is not None and self.root_translations.shape != (self.positions.shape[0], 3):
            raise ValueError("MotionData.root_translations must have shape (T, 3) when provided.")
        if self.source_height is None:
            self.source_height = self.human_height
        if self.human_height is None:
            self.human_height = self.source_height

    def iter_frames(self) -> Iterator[MotionFrame]:
        for frame_idx, positions in enumerate(self.positions):
            root_orientation = self.root_orientations[frame_idx] if self.root_orientations is not None else None
            root_translation = self.root_translations[frame_idx] if self.root_translations is not None else None
            yield MotionFrame(
                positions=positions,
                root_orientation=root_orientation,
                root_translation=root_translation,
                timestamp=(frame_idx / self.framerate) if self.framerate else None,
                metadata={"frame_index": frame_idx, **self.metadata},
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


def _stack_optional(values: list[np.ndarray | None]) -> np.ndarray | None:
    if any(value is None for value in values):
        return None
    return np.stack(values, axis=0)
