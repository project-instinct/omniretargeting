"""Nokov motion capture BVH data source adapter.

Nokov BVH files use the standard BVH format with a 6-channel root joint
(Xposition Yposition Zposition + Zrotation Xrotation Yrotation) and
3-channel child joints (Zrotation Xrotation Yrotation).
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
# Step 1: Rotate -90deg around X to map BVH Y-up to Z-up.
#   BVH: X=right, Y=up, Z=forward
#   After step 1: X=right, Z=up, -Y=forward  (character faces -Y)
# Step 2: Rotate +90deg around Z to align character forward (-Y) with robot forward (+X).
#   After step 2: X=forward, Y=right, Z=up   (character faces +X)

_ROTATION_Y_TO_Z = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)  # -90deg X
_ROTATION_FORWARD = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)  # +90deg Z
_ROTATION_MATRIX = (_ROTATION_FORWARD @ _ROTATION_Y_TO_Z).astype(np.float32)

# ---------------------------------------------------------------------------
# Nokov BVH parser
# ---------------------------------------------------------------------------


def _read_bvh(filename: str | Path) -> dict:
    """Parse a Nokov-style BVH file.

    Handles the common Nokov channel layout:
      - Root joint:  6 channels (Xposition Yposition Zposition + Zrotation Xrotation Yrotation)
      - Child joints: 3 channels (Zrotation Xrotation Yrotation)

    Returns a dict with keys: names, parents, offsets, quats, positions, frametime.
    """
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

    # --- hierarchy pass ---
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
                # Extract rotation order from the last 3 channel names
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
        order = "zxy"

    parents_arr = np.array(parents, dtype=np.int32)
    offsets_arr = np.array(offsets, dtype=np.float32)
    num_joints = len(names)

    # --- motion pass ---
    # Initialize positions with joint offsets (children keep their local offset;
    # root position is overwritten from motion data)
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
                # 6-channel joint: 3 position + 3 rotation values
                positions[i, j] = data_block[offset : offset + 3]
                rotations[i, j] = data_block[offset + 3 : offset + 6]
            elif nc == 3:
                # 3-channel joint: 3 rotation values (position = local offset)
                rotations[i, j] = data_block[offset : offset + 3]
            offset += nc
        i += 1

    # Convert Euler angles to quaternions and smooth frame-to-frame discontinuities
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
# Nokov DataSource
# ---------------------------------------------------------------------------


@dataclass
class NokovDataSource(DataSource):
    motion_file: Path
    start_frame: int = 0
    metadata: dict = field(default_factory=dict)

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

        # Forward kinematics to global positions (still in BVH coords, cm)
        _, global_positions = _quat_fk(quats, positions, parents)

        # Apply coordinate transform and cm → m
        transformed_positions = global_positions @ _ROTATION_MATRIX.T / 100.0

        root_translations = transformed_positions[:, 0, :].copy()

        # Estimate human height across all frames
        source_height = self._estimate_height(names, transformed_positions)

        framerate = 1.0 / frametime if frametime > 0 else 30.0

        self._motion_data = MotionData(
            positions=transformed_positions,
            target_names=list(names),
            root_orientations=None,  # estimated from joint positions by the pipeline
            root_translations=root_translations,
            framerate=framerate,
            source_height=source_height,
            metadata={
                **self.metadata,
                "source_type": "nokov",
                "bone_names": list(names),
                "bone_parents": parents.tolist(),
            },
        )
        return self._motion_data

    @staticmethod
    def _estimate_height(
        names: list[str],
        positions: np.ndarray,
    ) -> float:
        """Estimate human height across all frames using shared utility.

        Args:
            names: List of joint names.
            positions: Transformed joint positions (T, J, 3) in meters.

        Returns:
            Estimated height in meters, or None if required landmarks are unavailable.
        """
        return estimate_body_height(
            positions, names,
            head_joint="Head",
            foot_joints=("LeftFoot", "RightFoot"),
        )

    def iter_frames(self):
        yield from self.load().iter_frames()


def create_nokov_data_source(
    motion_file: Path,
    source_config: dict | None = None,
    runtime_options: dict | None = None,
) -> NokovDataSource:
    source_config = dict(source_config or {})
    runtime_options = dict(runtime_options or {})

    return NokovDataSource(
        motion_file=motion_file,
        start_frame=int(runtime_options.get("start_frame", 0)),
        metadata=runtime_options.get("metadata", {}),
    )


register_data_source("nokov", create_nokov_data_source, extensions=[".bvh"])
