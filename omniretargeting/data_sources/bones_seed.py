"""Bones-Seed (SOMA uniform) BVH data source adapter.

Bones-Seed BVH files use a full-body skeleton with 78 joints including fingers,
eyes, and jaw. Channel layout:
  - Root joint: 6 channels (Xposition Yposition Zposition Zrotation Yrotation Xrotation)
  - Hips joint: 6 channels (same layout)
  - All other joints: 3 channels (Zrotation Yrotation Xrotation)

Units are centimeters, Y-up coordinate system, Euler order ZYX.
Dataset is organized as date-coded subdirectories with optional ``_M`` mirrored variants.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from omniretargeting.data_sources.base import DataSource, MotionData
from omniretargeting.data_sources.registry import register_data_source
from omniretargeting.utils import estimate_body_height
from omniretargeting.data_sources.lafan1 import (
    _euler_to_quat,
    _quat_fk,
    _remove_quat_discontinuities,
)

# ---------------------------------------------------------------------------
# BVH parsing constants
# ---------------------------------------------------------------------------

_CHANNELMAP = {"Xrotation": "x", "Yrotation": "y", "Zrotation": "z"}

# ---------------------------------------------------------------------------
# Coordinate transform: BVH (Y-up, cm) to omniretargeting (Z-up, m)
# ---------------------------------------------------------------------------

_ROTATION_Y_TO_Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
_ROTATION_FORWARD = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
_ROTATION_MATRIX = (_ROTATION_FORWARD @ _ROTATION_Y_TO_Z).astype(np.float32)

# ---------------------------------------------------------------------------
# Bone-Seed BVH parser
# ---------------------------------------------------------------------------


def _read_bvh(filename: str | Path) -> dict:
    """Parse a Bone-Seed BVH file with mixed channel counts per joint."""
    with open(filename, "r") as f:
        lines = f.readlines()

    names: list[str] = []
    offsets: list[np.ndarray] = []
    parents: list[int] = []
    channels_per_joint: list[int] = []
    active = -1
    end_site = False
    order: str | None = None
    fnum: int = 0
    frametime: float = 0.0

    for line in lines:
        if "HIERARCHY" in line or "MOTION" in line:
            continue

        rmatch = re.match(r"ROOT (\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets.append(np.array([0, 0, 0], dtype=np.float32))
            parents.append(active)
            channels_per_joint.append(0)
            active = len(parents) - 1
            continue

        if "{" in line:
            continue

        if "}" in line:
            if end_site:
                end_site = False
            else:
                active = parents[active]
            continue

        offmatch = re.match(
            r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line
        )
        if offmatch:
            if not end_site:
                offsets[active] = np.array(
                    [float(x) for x in offmatch.groups()], dtype=np.float32
                )
            continue

        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            nc = int(chanmatch.group(1))
            channels_per_joint[active] = nc
            if order is None:
                parts = line.split()
                rot_parts = parts[-3:]
                if all(p in _CHANNELMAP for p in rot_parts):
                    order = "".join([_CHANNELMAP[p] for p in rot_parts])
            continue

        jmatch = re.match(r"\s*JOINT\s+(\w+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets.append(np.array([0, 0, 0], dtype=np.float32))
            parents.append(active)
            channels_per_joint.append(0)
            active = len(parents) - 1
            continue

        if "End Site" in line:
            end_site = True
            continue

        fmatch = re.match(r"\s*Frames:\s+(\d+)", line)
        if fmatch:
            fnum = int(fmatch.group(1))
            continue

        fmatch = re.match(r"\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

    if order is None:
        order = "zyx"

    parents_arr = np.array(parents, dtype=np.int32)
    offsets_arr = np.array(offsets, dtype=np.float32)
    num_joints = len(names)

    positions = np.tile(offsets_arr[np.newaxis], (fnum, 1, 1)).astype(np.float32)
    rotations = np.zeros((fnum, num_joints, 3), dtype=np.float32)

    i = 0
    in_motion = False
    for line in lines:
        if "MOTION" in line:
            in_motion = True
            continue
        if not in_motion:
            continue
        if "Frames:" in line or "Frame Time:" in line:
            continue

        dmatch = line.strip().split()
        if not dmatch:
            continue

        data_block = np.array([float(x) for x in dmatch], dtype=np.float32)
        offset = 0
        for j in range(num_joints):
            nc = channels_per_joint[j]
            if nc == 6:
                positions[i, j] = data_block[offset : offset + 3]
                rotations[i, j] = data_block[offset + 3 : offset + 6]
            elif nc == 3:
                rotations[i, j] = data_block[offset : offset + 3]
            offset += nc
        i += 1

    quats = _euler_to_quat(np.radians(rotations), order=order)
    quats = _remove_quat_discontinuities(quats)

    return {
        "names": names,
        "parents": parents_arr,
        "offsets": offsets_arr,
        "quats": quats,
        "positions": positions,
        "frametime": frametime,
    }


# ---------------------------------------------------------------------------
# Bone-Seed DataSource
# ---------------------------------------------------------------------------


@dataclass
class BonesSeedDataSource(DataSource):
    motion_file: Path
    start_frame: int = 0
    metadata: dict = field(default_factory=dict)
    height_estimation: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.motion_file = Path(self.motion_file)
        self.start_frame = max(0, int(self.start_frame))
        self._motion_data: MotionData | None = None

    @property
    def target_names(self) -> list[str] | None:
        if self._motion_data is not None:
            return self._motion_data.target_names
        return None

    @property
    def framerate(self) -> float | None:
        if self._motion_data is not None:
            return self._motion_data.framerate
        return None

    @property
    def source_height(self) -> float | None:
        if self._motion_data is not None:
            return self._motion_data.source_height
        return None

    @property
    def human_height(self) -> float | None:
        return self.source_height

    def load(self) -> MotionData:
        if self._motion_data is not None:
            return self._motion_data

        bvh = _read_bvh(self.motion_file)
        quats = bvh["quats"]
        positions = bvh["positions"]
        parents = bvh["parents"]
        names = bvh["names"]
        frametime = bvh["frametime"]

        if self.start_frame > 0:
            quats = quats[self.start_frame:]
            positions = positions[self.start_frame:]

        _, global_positions = _quat_fk(quats, positions, parents)

        transformed_positions = global_positions @ _ROTATION_MATRIX.T / 100.0

        # Root joint is a virtual node fixed at origin; Hips carries actual position.
        hips_idx = names.index("Hips") if "Hips" in names else 0
        root_translations = transformed_positions[:, hips_idx, :].copy()

        source_height = estimate_body_height(
            transformed_positions, names,
            **{
                "head_joint": "Head",
                "foot_joints": ("LeftFoot", "RightFoot"),
                **self.height_estimation,
            },
        )

        framerate = 1.0 / frametime if frametime > 0 else 120.0

        self._motion_data = MotionData(
            positions=transformed_positions,
            target_names=list(names),
            root_orientations=None,
            root_translations=root_translations,
            framerate=framerate,
            source_height=source_height,
            metadata={
                **self.metadata,
                "source_type": "bones_seed",
                "bone_names": list(names),
                "bone_parents": parents.tolist(),
            },
        )
        return self._motion_data

    def iter_frames(self):
        yield from self.load().iter_frames()


def create_bones_seed_data_source(
    motion_file: Path,
    source_config: dict | None = None,
    runtime_options: dict | None = None,
) -> BonesSeedDataSource:
    source_config = dict(source_config or {})
    runtime_options = dict(runtime_options or {})
    adapter_options = dict(source_config.get("adapter_options") or {})

    return BonesSeedDataSource(
        motion_file=motion_file,
        start_frame=int(runtime_options.get("start_frame", 0)),
        metadata=runtime_options.get("metadata", {}),
        height_estimation=runtime_options.get(
            "height_estimation",
            adapter_options.get(
                "height_estimation", source_config.get("height_estimation", {})
            ),
        ),
    )


register_data_source("bones_seed", create_bones_seed_data_source, extensions=[".bvh"])
