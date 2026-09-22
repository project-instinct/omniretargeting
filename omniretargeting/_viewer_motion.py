"""Robot motion loading, source skeleton metadata and object mesh helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import viser

SUPPORTED_MOTION_SUFFIXES = (".npz", ".npy")


SMPLX_BODY_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=int,
)


PARC_MS_BODY_PARENTS = np.array(
    [-1, 0, 1, 1, 3, 4, 1, 6, 7, 0, 9, 10, 0, 12, 13],
    dtype=int,
)


def _parc_ms_body_forward(joints: np.ndarray) -> np.ndarray:
    """Recover PARC's +X body axis from pelvis, torso and hip positions."""
    joints = np.asarray(joints, dtype=float)
    up = joints[1] - joints[0]
    right = joints[9] - joints[12]
    up /= max(float(np.linalg.norm(up)), 1e-10)
    right /= max(float(np.linalg.norm(right)), 1e-10)
    forward = np.cross(up, right)
    return forward / max(float(np.linalg.norm(forward)), 1e-10)


OMOMO_OBJECT_SCALES = {
    "floorlamp": (1.55 * 0.3793, 1.55 * 0.3793, 1.55 * 0.3793),
    "largebox": (1.55 * 0.3486, 1.55 * 0.3486, 1.55 * 0.3486),
    "whitechair": (1.55 * 0.3129, 1.55 * 0.3129, 1.55 * 0.3129),
    "trashcan": (1.55 * 0.2326, 1.55 * 0.2326, 1.55 * 0.2326),
    "smalltable": (1.55 * 0.0162, 1.55 * 0.0162, 1.55 * 0.0162),
    "suitcase": (1.55 * 0.3672, 1.55 * 0.3672, 1.55 * 0.3672),
}


def _parse_scale(text: str) -> float | tuple[float, float, float]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if len(values) == 1:
        return values[0]
    if len(values) == 3:
        return values[0], values[1], values[2]
    raise argparse.ArgumentTypeError(f"Invalid scale '{text}'. Expected SCALE or X,Y,Z.")


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _resolve_motion_file(path: Path) -> Path:
    if path.is_file():
        return path

    patterns = ("*retargeted*.npz", "*retargetted*.npz", "*.npz", "*.npy")
    for pattern in patterns:
        files = sorted(path.rglob(pattern))
        if files:
            return files[0]
    raise FileNotFoundError(f"No supported motion files found under {path}.")


def _continuous_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.array(quat, dtype=np.float64, copy=True)
    quat /= np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-12)
    for frame_idx in range(1, len(quat)):
        if np.dot(quat[frame_idx - 1], quat[frame_idx]) < 0.0:
            quat[frame_idx] *= -1.0
    return quat


def _model_scalar_joint_names(model: mujoco.MjModel) -> list[str]:
    joint_names: list[str] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if model.jnt_type[joint_id] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            raise ValueError(f"Unsupported non-scalar joint '{name}' in model.")
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_names.append(joint_name)
    return joint_names


def _free_joint_qpos_adr(model: mujoco.MjModel) -> int:
    free_joint_ids = [
        joint_id for joint_id in range(model.njnt) if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if len(free_joint_ids) != 1:
        raise ValueError(f"Expected exactly one free joint in the model, found {len(free_joint_ids)}.")
    return int(model.jnt_qposadr[free_joint_ids[0]])


def _payload_joint_names(payload: np.lib.npyio.NpzFile, joint_pos: np.ndarray, path: Path) -> tuple[list[str], np.ndarray]:
    if "joint_names" not in payload:
        return [], joint_pos

    raw_joint_names = payload["joint_names"].tolist()
    joint_names = [raw_joint_names] if isinstance(raw_joint_names, str) else list(raw_joint_names)
    joint_names = [name.decode() if isinstance(name, bytes) else str(name) for name in joint_names]

    if joint_names and joint_names[0] == "floating_base_joint":
        if len(joint_names) == joint_pos.shape[1] + 1:
            joint_names = joint_names[1:]
        elif len(joint_names) == joint_pos.shape[1]:
            joint_names = joint_names[1:]
            joint_pos = joint_pos[:, 1:]

    if len(joint_names) != joint_pos.shape[1]:
        raise ValueError(f"{path}: joint_names has {len(joint_names)} entries, joint_pos has {joint_pos.shape[1]} columns.")
    return joint_names, joint_pos


def _build_qpos_from_components(
    model: mujoco.MjModel,
    motion_file: Path,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    joint_pos: np.ndarray,
    joint_names: list[str],
    root_z_offset: float,
) -> np.ndarray:
    if base_pos.shape[0] != base_quat.shape[0] or base_pos.shape[0] != joint_pos.shape[0]:
        raise ValueError(f"{motion_file}: base and joint trajectories have different frame counts.")

    model_joint_names = _model_scalar_joint_names(model)
    if joint_names:
        missing = [name for name in model_joint_names if name not in joint_names]
        if missing:
            raise ValueError(f"{motion_file}: missing joints required by the model: {missing}")
        joint_order = [joint_names.index(name) for name in model_joint_names]
    else:
        if joint_pos.shape[1] != len(model_joint_names):
            raise ValueError(
                f"{motion_file}: joint_names is missing and joint_pos has {joint_pos.shape[1]} columns; "
                f"model expects {len(model_joint_names)} scalar joints."
            )
        joint_order = list(range(len(model_joint_names)))

    base_pos = np.array(base_pos, dtype=np.float64, copy=True)
    base_pos[:, 2] += root_z_offset
    base_quat = _continuous_quat(base_quat)

    qpos = np.tile(model.qpos0, (base_pos.shape[0], 1))
    free_adr = _free_joint_qpos_adr(model)
    qpos[:, free_adr : free_adr + 3] = base_pos
    qpos[:, free_adr + 3 : free_adr + 7] = base_quat
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        qpos_adr = int(model.jnt_qposadr[joint_id])
        qpos[:, qpos_adr] = joint_pos[:, joint_order[model_joint_names.index(joint_name)]]
    return qpos


def _load_motion(
    model: mujoco.MjModel,
    motion_file: Path,
    root_z_offset: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float]:
    if motion_file.suffix == ".npy":
        qpos = np.asarray(np.load(motion_file, allow_pickle=True), dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != model.nq:
            raise ValueError(f"{motion_file}: .npy playback expects shape (T, {model.nq}), got {qpos.shape}.")
        return qpos, None, None, 30.0

    with np.load(motion_file, allow_pickle=True) as payload:
        framerate = float(np.asarray(payload["framerate"]).item()) if "framerate" in payload else 30.0
        object_pos = np.asarray(payload["object_pos_w"], dtype=np.float64) if "object_pos_w" in payload else None
        object_quat = (
            _continuous_quat(np.asarray(payload["object_quat_w"], dtype=np.float64))
            if "object_quat_w" in payload
            else None
        )

        if "qpos" in payload:
            qpos = np.asarray(payload["qpos"], dtype=np.float64)
            if qpos.ndim != 2 or qpos.shape[1] != model.nq:
                raise ValueError(f"{motion_file}: qpos expects shape (T, {model.nq}), got {qpos.shape}.")
            return qpos, object_pos, object_quat, framerate

        joint_pos = np.asarray(payload["joint_pos"], dtype=np.float64)
        joint_names, joint_pos = _payload_joint_names(payload, joint_pos, motion_file)
        qpos = _build_qpos_from_components(
            model=model,
            motion_file=motion_file,
            base_pos=np.asarray(payload["base_pos_w"], dtype=np.float64),
            base_quat=np.asarray(payload["base_quat_w"], dtype=np.float64),
            joint_pos=joint_pos,
            joint_names=joint_names,
            root_z_offset=root_z_offset,
        )

    if object_pos is not None and object_pos.shape[0] != qpos.shape[0]:
        raise ValueError(f"{motion_file}: object_pos_w frame count does not match qpos.")
    if object_quat is not None and object_quat.shape[0] != qpos.shape[0]:
        raise ValueError(f"{motion_file}: object_quat_w frame count does not match qpos.")
    return qpos, object_pos, object_quat, framerate


def _infer_omomo_object_mesh(
    motion_file: Path,
    object_dir: Path,
) -> tuple[Path | None, float | tuple[float, float, float]]:
    if not object_dir.exists():
        return None, 1.0

    tokens = motion_file.stem.split("_")
    for token in tokens:
        exact = object_dir / f"{token}_cleaned_simplified.obj"
        if exact.exists():
            return exact, OMOMO_OBJECT_SCALES.get(token, 1.0)
        matches = sorted(object_dir.glob(f"{token}_cleaned_simplified*.obj"))
        if matches:
            return matches[0], OMOMO_OBJECT_SCALES.get(token, 1.0)
    return None, 1.0


def _add_object(
    server: viser.ViserServer,
    object_mesh: Path | None,
    object_scale: float | tuple[float, float, float],
):
    if object_mesh is None:
        return None
    if not object_mesh.exists():
        print(f"[WARN] Object mesh missing: {object_mesh}")
        return None
    mesh = trimesh.load(object_mesh, force="mesh")
    return server.scene.add_mesh_trimesh(
        "/omomo_object",
        mesh,
        scale=object_scale,
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
    )


def _resolve_object_mesh(
    args: argparse.Namespace,
    motion_file: Path,
) -> tuple[Path | None, float | tuple[float, float, float]]:
    if args.object_mesh:
        return _resolve_path(args.object_mesh), 1.0 if args.object_scale is None else args.object_scale

    object_mesh, object_scale = _infer_omomo_object_mesh(motion_file, _resolve_path(args.omomo_object_dir))
    if args.object_scale is not None:
        object_scale = args.object_scale
    return object_mesh, object_scale
