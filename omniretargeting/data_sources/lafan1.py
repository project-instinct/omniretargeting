"""LAFAN1 BVH motion data source adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from omniretargeting.data_sources.base import DataSource, MotionData
from omniretargeting.data_sources.registry import register_data_source


# ---------------------------------------------------------------------------
# Minimal BVH parser (adapted from GMR lafan_vendor)
# ---------------------------------------------------------------------------

_CHANNELMAP = {"Xrotation": "x", "Yrotation": "y", "Zrotation": "z"}


def _euler_to_quat(e: np.ndarray, order: str = "zyx") -> np.ndarray:
    axis = {
        "x": np.asarray([1, 0, 0], dtype=np.float32),
        "y": np.asarray([0, 1, 0], dtype=np.float32),
        "z": np.asarray([0, 0, 1], dtype=np.float32),
    }

    def _angle_axis_to_quat(angle: np.ndarray, ax: np.ndarray) -> np.ndarray:
        c = np.cos(angle / 2.0)[..., np.newaxis]
        s = np.sin(angle / 2.0)[..., np.newaxis]
        return np.concatenate([c, s * ax], axis=-1)

    q0 = _angle_axis_to_quat(e[..., 0], axis[order[0]])
    q1 = _angle_axis_to_quat(e[..., 1], axis[order[1]])
    q2 = _angle_axis_to_quat(e[..., 2], axis[order[2]])
    return _quat_mul(q0, _quat_mul(q1, q2))


def _quat_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]
    return np.concatenate([
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0,
    ], axis=-1)


def _quat_inv(q: np.ndarray) -> np.ndarray:
    return np.asarray([1, -1, -1, -1], dtype=np.float32) * q


def _quat_mul_vec(q: np.ndarray, x: np.ndarray) -> np.ndarray:
    t = 2.0 * np.cross(q[..., 1:], x)
    return x + q[..., 0][..., np.newaxis] * t + np.cross(q[..., 1:], t)


def _quat_fk(lrot: np.ndarray, lpos: np.ndarray, parents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gp, gr = [lpos[..., :1, :]], [lrot[..., :1, :]]
    for i in range(1, len(parents)):
        gp.append(_quat_mul_vec(gr[parents[i]], lpos[..., i:i + 1, :]) + gp[parents[i]])
        gr.append(_quat_mul(gr[parents[i]], lrot[..., i:i + 1, :]))
    return np.concatenate(gr, axis=-2), np.concatenate(gp, axis=-2)


def _remove_quat_discontinuities(rotations: np.ndarray) -> np.ndarray:
    rots_inv = -rotations
    for i in range(1, rotations.shape[0]):
        replace_mask = np.sum(rotations[i - 1:i] * rotations[i:i + 1], axis=-1) < np.sum(
            rotations[i - 1:i] * rots_inv[i:i + 1], axis=-1)
        rotations[i] = replace_mask[..., np.newaxis] * rots_inv[i] + (1.0 - replace_mask[..., np.newaxis]) * rotations[i]
    return rotations


def _read_bvh(filename: str | Path) -> dict:
    """Parse a BVH file and return bone names, parents, offsets, quats, positions, and framerate."""
    with open(filename, "r") as f:
        lines = f.readlines()

    names: list[str] = []
    offsets: list[np.ndarray] = []
    parents: list[int] = []
    active = -1
    end_site = False
    order: str | None = None
    channels: int = 0
    fnum: int = 0
    frametime: float = 0.0

    import re

    # --- hierarchy pass ---
    for line in lines:
        if "HIERARCHY" in line or "MOTION" in line:
            continue

        rmatch = re.match(r"ROOT (\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets.append(np.array([0, 0, 0], dtype=np.float32))
            parents.append(active)
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

        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            if not end_site:
                offsets[active] = np.array([float(x) for x in offmatch.groups()], dtype=np.float32)
            continue

        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis:2 + channelie]
                if all(p in _CHANNELMAP for p in parts):
                    order = "".join([_CHANNELMAP[p] for p in parts])
            continue

        jmatch = re.match(r"\s*JOINT\s+(\w+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets.append(np.array([0, 0, 0], dtype=np.float32))
            parents.append(active)
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

    # --- motion pass ---
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
        if channels == 3:
            positions[i, 0:1] = data_block[0:3]
            rotations[i, :] = data_block[3:].reshape(num_joints, 3)
        elif channels == 6:
            data_block = data_block.reshape(num_joints, 6)
            positions[i, :] = data_block[:, 0:3]
            rotations[i, :] = data_block[:, 3:6]
        elif channels == 9:
            positions[i, 0] = data_block[0:3]
            data_block = data_block[3:].reshape(num_joints - 1, 9)
            rotations[i, 1:] = data_block[:, 3:6]
            positions[i, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
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
# Coordinate transform: BVH (Y-up, cm) to omniretargeting (Z-up, m)
# ---------------------------------------------------------------------------

_ROTATION_MATRIX = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
_ROTATION_QUAT = Rotation.from_matrix(_ROTATION_MATRIX).as_quat(scalar_first=True).astype(np.float32)


# ---------------------------------------------------------------------------
# LAFAN1 DataSource
# ---------------------------------------------------------------------------

@dataclass
class Lafan1DataSource(DataSource):
    motion_file: Path
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.motion_file = Path(self.motion_file)
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

        # Forward kinematics to global quats and positions (still in BVH coords, cm)
        global_quats, global_positions = _quat_fk(quats, positions, parents)

        num_frames = global_positions.shape[0]

        # Apply coordinate transform and cm to m conversion
        transformed_positions = global_positions @ _ROTATION_MATRIX.T / 100.0

        # Transform root quaternion for each frame
        root_quats = np.zeros((num_frames, 4), dtype=np.float32)
        for f in range(num_frames):
            root_quats[f] = _quat_mul(_ROTATION_QUAT, global_quats[f, 0])

        root_orientations = Rotation.from_quat(
            root_quats[:, [1, 2, 3, 0]]  # scalar-first to scalar-last for scipy
        ).as_rotvec().astype(np.float32)

        root_translations = transformed_positions[:, 0, :].copy()

        # Estimate human height from the first frame
        head_name_candidates = ["Head", "head", "Neck", "Neck1"]
        foot_name_candidates = [
            ["LeftFoot", "RightFoot"],
            ["LeftToe", "RightToe"],
            ["LeftFootMod", "RightFootMod"],
        ]
        head_idx = None
        for hn in head_name_candidates:
            try:
                head_idx = names.index(hn)
                break
            except ValueError:
                continue

        foot_indices = None
        for pair in foot_name_candidates:
            try:
                foot_indices = [names.index(pair[0]), names.index(pair[1])]
                break
            except ValueError:
                continue

        source_height = 1.75  # default fallback
        if head_idx is not None and foot_indices is not None:
            head_z = transformed_positions[0, head_idx, 2]
            feet_z = min(transformed_positions[0, foot_indices[0], 2],
                         transformed_positions[0, foot_indices[1], 2])
            source_height = float(np.clip(head_z - feet_z + 0.12, 1.4, 2.2))

        framerate = 1.0 / frametime if frametime > 0 else 30.0

        self._motion_data = MotionData(
            positions=transformed_positions,
            target_names=list(names),
            root_orientations=root_orientations,
            root_translations=root_translations,
            framerate=framerate,
            source_height=source_height,
            metadata={
                **self.metadata,
                "source_type": "lafan1",
                "bone_names": list(names),
                "bone_parents": parents.tolist(),
            },
        )
        return self._motion_data

    def iter_frames(self):
        yield from self.load().iter_frames()


def create_lafan1_data_source(
    motion_file: Path,
    source_config: dict | None = None,
    runtime_options: dict | None = None,
) -> Lafan1DataSource:
    source_config = dict(source_config or {})
    runtime_options = dict(runtime_options or {})

    return Lafan1DataSource(
        motion_file=motion_file,
        metadata=runtime_options.get("metadata", {}),
    )


register_data_source("lafan1", create_lafan1_data_source)
