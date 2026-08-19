from __future__ import annotations

import argparse
import contextlib
import zlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import tempfile
import traceback

import mujoco
import numpy as np

from omniretargeting import OmniRetargeter
from omniretargeting.main import create_flat_terrain, extract_omomo_object_pose
from omniretargeting.robot_config import load_robot_config
from omniretargeting.data_sources.smplx import load_smplx_trajectory
from omniretargeting.utils import normalize_retargeted_output_path


_WORKER: dict[str, object] = {}


def _moving_average_reflect(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    if window % 2 == 0:
        raise ValueError("--smooth-window must be odd.")
    if len(values) < 2:
        return values

    radius = window // 2
    padded = np.pad(values, [(radius, radius), *[(0, 0) for _ in range(values.ndim - 1)]], mode="reflect")
    kernel = np.ones(window, dtype=values.dtype) / window
    smoothed = np.empty_like(values)
    flat_padded = padded.reshape(padded.shape[0], -1)
    flat_smoothed = smoothed.reshape(smoothed.shape[0], -1)
    for dim in range(flat_padded.shape[1]):
        flat_smoothed[:, dim] = np.convolve(flat_padded[:, dim], kernel, mode="valid")
    return smoothed


def _smooth_quaternions(quats: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return quats
    continuous = np.array(quats, copy=True)
    for frame_idx in range(1, len(continuous)):
        if np.dot(continuous[frame_idx - 1], continuous[frame_idx]) < 0.0:
            continuous[frame_idx] *= -1.0
    smoothed = _moving_average_reflect(continuous, window)
    norms = np.linalg.norm(smoothed, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return smoothed / norms


def _smooth_payload(output_payload: dict[str, np.ndarray], window: int) -> dict[str, np.ndarray]:
    if window <= 1:
        return output_payload
    output_payload = dict(output_payload)
    output_payload["joint_pos"] = _moving_average_reflect(output_payload["joint_pos"], window)
    base_pos_w = np.array(output_payload["base_pos_w"], copy=True)
    base_pos_w[:, :2] = _moving_average_reflect(base_pos_w[:, :2], window)
    output_payload["base_pos_w"] = base_pos_w
    output_payload["base_quat_w"] = _smooth_quaternions(output_payload["base_quat_w"], window)
    return output_payload


def _read_framerate(source_file: Path) -> float:
    motion_data = np.load(source_file, allow_pickle=True)
    for key in ("framerate", "mocap_framerate", "mocap_frame_rate", "fps"):
        if key in motion_data:
            return float(motion_data[key])
    return 30.0


def _init_worker(robot_config_path: str, smplx_model_dir: str, output_base_link: str) -> None:
    robot_config = load_robot_config(robot_config_path)
    joint_mapping = robot_config["joint_mapping"]
    robot_urdf_path = robot_config["urdf_path"]

    fd, terrain_path = tempfile.mkstemp(suffix=".obj")
    os.close(fd)
    create_flat_terrain().export(terrain_path)

    retargeting = dict(robot_config.get("retargeting") or {})
    retargeter = OmniRetargeter(
        robot_urdf_path=robot_urdf_path,
        terrain_mesh_path=terrain_path,
        joint_mapping=joint_mapping,
        robot_height=robot_config.get("robot_height"),
        smplx_joint_names=robot_config.get("smplx_joint_names"),
        height_estimation=robot_config.get("height_estimation"),
        base_orientation=robot_config.get("base_orientation"),
        retargeting=retargeting,
    )

    base_model = mujoco.MjModel.from_xml_path(str(robot_urdf_path))
    base_data = mujoco.MjData(base_model)
    base_body_id = mujoco.mj_name2id(base_model, mujoco.mjtObj.mjOBJ_BODY, output_base_link)
    if base_body_id < 0:
        raise ValueError(f"Body '{output_base_link}' not found in robot model: {robot_urdf_path}")
    root_body_id = -1
    for joint_id in range(base_model.njnt):
        if base_model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            root_body_id = int(base_model.jnt_bodyid[joint_id])
            break
    model_joint_names = [
        mujoco.mj_id2name(base_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(base_model.njnt)
        if base_model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
    ]
    smplx_joint_names = list(robot_config.get("smplx_joint_names") or [])
    smplx_joint_indices = {joint_name: joint_idx for joint_idx, joint_name in enumerate(smplx_joint_names)}

    _WORKER["retargeter"] = retargeter
    _WORKER["robot_urdf_path"] = robot_urdf_path
    _WORKER["smplx_model_dir"] = smplx_model_dir
    _WORKER["joint_mapping"] = joint_mapping
    _WORKER["model_joint_names"] = model_joint_names
    _WORKER["smplx_joint_indices"] = smplx_joint_indices
    _WORKER["base_model"] = base_model
    _WORKER["base_data"] = base_data
    _WORKER["base_body_id"] = base_body_id
    _WORKER["base_is_root"] = base_body_id == root_body_id


def _extract_body_base_pose(trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = _WORKER["base_model"]
    data = _WORKER["base_data"]
    body_id = _WORKER["base_body_id"]

    base_pos = np.zeros((len(trajectory), 3), dtype=trajectory.dtype)
    base_quat = np.zeros((len(trajectory), 4), dtype=trajectory.dtype)
    for frame_idx, qpos in enumerate(trajectory):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        base_pos[frame_idx] = data.xpos[body_id]
        base_quat[frame_idx] = data.xquat[body_id]

    return base_pos, base_quat


def _normalize_quaternions(quats: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return quats / norms


def _multiply_quaternions_wxyz(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        axis=-1,
    )


def _yaw_quaternions_wxyz(yaws: np.ndarray) -> np.ndarray:
    half_yaws = 0.5 * yaws
    quats = np.zeros((len(yaws), 4), dtype=np.float64)
    quats[:, 0] = np.cos(half_yaws)
    quats[:, 3] = np.sin(half_yaws)
    return quats


def _smooth_angles(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    sin_values = _moving_average_reflect(np.sin(values)[:, None], window)[:, 0]
    cos_values = _moving_average_reflect(np.cos(values)[:, None], window)[:, 0]
    return np.arctan2(sin_values, cos_values)


def _fit_yaw_xy_transform(source_xy: np.ndarray, target_xy: np.ndarray) -> tuple[float, np.ndarray]:
    source_center = source_xy.mean(axis=0)
    target_center = target_xy.mean(axis=0)
    source_delta = source_xy - source_center
    target_delta = target_xy - target_center
    u_matrix, _, vt_matrix = np.linalg.svd(source_delta.T @ target_delta)
    rotation = vt_matrix.T @ u_matrix.T
    if np.linalg.det(rotation) < 0.0:
        vt_matrix[-1] *= -1.0
        rotation = vt_matrix.T @ u_matrix.T
    yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    translation = target_center - rotation @ source_center
    return yaw, translation


def _payload_to_model_qpos(output_payload: dict[str, np.ndarray]) -> np.ndarray:
    model = _WORKER["base_model"]
    model_joint_names = _WORKER["model_joint_names"]
    payload_joint_names = output_payload["joint_names"]
    if not isinstance(payload_joint_names, list):
        payload_joint_names = payload_joint_names.tolist()
    payload_joint_names = [str(joint_name) for joint_name in payload_joint_names]
    joint_order = [payload_joint_names.index(joint_name) for joint_name in model_joint_names]

    base_pos = np.asarray(output_payload["base_pos_w"], dtype=np.float64)
    base_quat = np.asarray(output_payload["base_quat_w"], dtype=np.float64)
    joint_pos = np.asarray(output_payload["joint_pos"], dtype=np.float64)

    qpos = np.zeros((len(base_pos), model.nq), dtype=np.float64)
    qpos[:, :3] = base_pos
    qpos[:, 3:7] = base_quat
    qpos[:, 7:] = joint_pos[:, joint_order]
    return qpos


def _extract_payload_body_positions(
    output_payload: dict[str, np.ndarray],
    reference_joints: list[str],
) -> dict[str, np.ndarray]:
    model = _WORKER["base_model"]
    data = _WORKER["base_data"]
    joint_mapping = _WORKER["joint_mapping"]
    qpos = _payload_to_model_qpos(output_payload)

    body_ids = {}
    for joint_name in reference_joints:
        body_name = joint_mapping[joint_name]
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' for joint '{joint_name}' not found in robot model.")
        body_ids[joint_name] = body_id

    body_positions = {joint_name: np.zeros((len(qpos), 3), dtype=np.float64) for joint_name in reference_joints}
    for frame_idx, frame_qpos in enumerate(qpos):
        data.qpos[:] = frame_qpos
        mujoco.mj_forward(model, data)
        for joint_name, body_id in body_ids.items():
            body_positions[joint_name][frame_idx] = data.xpos[body_id]
    return body_positions


def _apply_post_smooth_foot_stabilization(
    output_payload: dict[str, np.ndarray],
    framerate: float,
) -> dict[str, np.ndarray]:
    retargeter = _WORKER["retargeter"]
    retargeting_config = getattr(retargeter, "retargeting_config", {})
    foot_stabilization_cfg = dict(retargeting_config.get("foot_stabilization", {}) or {})
    if not foot_stabilization_cfg.get("enabled", False):
        return output_payload
    if not _WORKER["base_is_root"]:
        return output_payload

    smoothed_motion = np.concatenate(
        [
            np.asarray(output_payload["base_pos_w"], dtype=np.float64),
            np.asarray(output_payload["base_quat_w"], dtype=np.float64),
            np.asarray(output_payload["joint_pos"], dtype=np.float64),
        ],
        axis=1,
    )
    stabilized_motion = retargeter._apply_foot_stabilization(
        smoothed_motion,
        retargeter.terrain_mesh,
        framerate=framerate,
    )

    output_payload = dict(output_payload)
    output_payload["base_pos_w"] = stabilized_motion[:, :3]
    output_payload["base_quat_w"] = stabilized_motion[:, 3:7]
    output_payload["joint_pos"] = stabilized_motion[:, 7:]
    return output_payload


def _apply_object_alignment(
    output_payload: dict[str, np.ndarray],
    human_joints: np.ndarray,
) -> dict[str, np.ndarray]:
    if "object_pos_w" not in output_payload or "object_quat_w" not in output_payload:
        return output_payload

    retargeter = _WORKER["retargeter"]
    retargeting_config = getattr(retargeter, "retargeting_config", {})
    object_alignment_cfg = dict(retargeting_config.get("object_alignment", {}) or {})
    if not object_alignment_cfg.get("enabled", False):
        return output_payload
    if not _WORKER["base_is_root"]:
        return output_payload

    method = object_alignment_cfg.get("method", "frame_yaw_xy")
    if method not in ("frame_yaw_xy", "sequence_yaw_xy"):
        raise ValueError(f"Unsupported object_alignment method: {method}")

    smooth_window = int(object_alignment_cfg.get("smooth_window", 1))
    if smooth_window < 1 or smooth_window % 2 == 0:
        raise ValueError("object_alignment.smooth_window must be a positive odd integer.")

    joint_mapping = _WORKER["joint_mapping"]
    smplx_joint_indices = _WORKER["smplx_joint_indices"]
    reference_joints = list(object_alignment_cfg.get("reference_joints") or joint_mapping.keys())
    missing_joints = [joint_name for joint_name in reference_joints if joint_name not in joint_mapping]
    if missing_joints:
        raise ValueError(f"object_alignment.reference_joints missing from joint_mapping: {missing_joints}")
    missing_smplx_joints = [joint_name for joint_name in reference_joints if joint_name not in smplx_joint_indices]
    if missing_smplx_joints:
        raise ValueError(f"object_alignment.reference_joints missing from smplx_joint_names: {missing_smplx_joints}")

    object_pos = np.array(output_payload["object_pos_w"], dtype=np.float64, copy=True)
    object_quat = np.array(output_payload["object_quat_w"], dtype=np.float64, copy=True)
    num_frames = min(len(object_pos), len(human_joints), len(output_payload["base_pos_w"]))
    object_pos = object_pos[:num_frames]
    object_quat = object_quat[:num_frames]
    human_joints = np.asarray(human_joints[:num_frames], dtype=np.float64)
    robot_body_positions = _extract_payload_body_positions(output_payload, reference_joints)

    if method == "sequence_yaw_xy":
        source_xy = np.concatenate(
            [human_joints[:, smplx_joint_indices[joint_name], :2] for joint_name in reference_joints],
            axis=0,
        )
        target_xy = np.concatenate([robot_body_positions[joint_name][:, :2] for joint_name in reference_joints], axis=0)
        yaw, translation = _fit_yaw_xy_transform(source_xy, target_xy)
        yaws = np.full(num_frames, yaw, dtype=np.float64)
        translations = np.repeat(translation[None], num_frames, axis=0)
    else:
        yaws = np.zeros(num_frames, dtype=np.float64)
        translations = np.zeros((num_frames, 2), dtype=np.float64)
        for frame_idx in range(num_frames):
            source_xy = np.stack(
                [human_joints[frame_idx, smplx_joint_indices[joint_name], :2] for joint_name in reference_joints],
                axis=0,
            )
            target_xy = np.stack(
                [robot_body_positions[joint_name][frame_idx, :2] for joint_name in reference_joints],
                axis=0,
            )
            yaws[frame_idx], translations[frame_idx] = _fit_yaw_xy_transform(source_xy, target_xy)

        if smooth_window > 1:
            yaws = _smooth_angles(yaws, smooth_window)
            translations = _moving_average_reflect(translations, smooth_window)

    aligned_pos = np.array(object_pos, copy=True)
    for frame_idx, (yaw, translation) in enumerate(zip(yaws, translations)):
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation_xy = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)
        aligned_pos[frame_idx, :2] = rotation_xy @ object_pos[frame_idx, :2] + translation

    aligned_quat = object_quat
    if object_alignment_cfg.get("apply_yaw_to_orientation", True):
        aligned_quat = _normalize_quaternions(_multiply_quaternions_wxyz(_yaw_quaternions_wxyz(yaws), object_quat))
        for frame_idx in range(1, len(aligned_quat)):
            if np.dot(aligned_quat[frame_idx - 1], aligned_quat[frame_idx]) < 0.0:
                aligned_quat[frame_idx] *= -1.0

    output_payload = dict(output_payload)
    if len(output_payload["object_pos_w"]) == num_frames:
        output_payload["object_pos_w"] = aligned_pos
        output_payload["object_quat_w"] = aligned_quat
    else:
        full_pos = np.array(output_payload["object_pos_w"], dtype=np.float64, copy=True)
        full_quat = np.array(output_payload["object_quat_w"], dtype=np.float64, copy=True)
        full_pos[:num_frames] = aligned_pos
        full_quat[:num_frames] = aligned_quat
        output_payload["object_pos_w"] = full_pos
        output_payload["object_quat_w"] = full_quat

    mean_shift = aligned_pos[:, :2] - object_pos[:, :2]
    print(
        "Applied object xy alignment: "
        f"method={method}, mean_xy_shift=[{mean_shift[:, 0].mean():.4f}, {mean_shift[:, 1].mean():.4f}]m, "
        f"yaw_range=[{np.rad2deg(yaws.min()):.3f}, {np.rad2deg(yaws.max()):.3f}]deg"
    )
    return output_payload


def _process_file(task: tuple[str, str, str, int]) -> tuple[str, bool, str]:
    source_file = Path(task[0])
    output_file = Path(task[1])
    log_file = Path(task[2])
    smooth_window = int(task[3])

    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            smplx_trajectory, _ = load_smplx_trajectory(
                smplx_file=source_file,
                smplx_model_directory=str(_WORKER["smplx_model_dir"]),
            )
            framerate = _read_framerate(source_file)
            np.random.seed(zlib.crc32(source_file.name.encode("utf-8")))

            retargeter = _WORKER["retargeter"]
            if hasattr(retargeter, "_last_estimated_quat"):
                retargeter._last_estimated_quat = None

            _, retargeted_motion = retargeter.retarget_motion(
                smplx_trajectory,
                framerate=framerate,
                visualize_trajectory=False,
                enable_terrain_scaling=False,
            )

            joint_names = retargeter.get_joint_names()
            base_pos, base_quat = _extract_body_base_pose(retargeted_motion)
            object_pose = extract_omomo_object_pose(source_file, len(retargeted_motion))

            output_payload = dict(
                framerate=framerate,
                joint_names=np.array(joint_names),
                joint_pos=retargeted_motion[:, 7:],
                base_pos_w=base_pos,
                base_quat_w=base_quat,
            )
            if object_pose is not None:
                object_pos, object_quat = object_pose
                output_payload["object_pos_w"] = object_pos
                output_payload["object_quat_w"] = object_quat

            output_payload = _smooth_payload(output_payload, smooth_window)
            output_payload = _apply_post_smooth_foot_stabilization(output_payload, framerate)
            output_payload = _apply_object_alignment(output_payload, smplx_trajectory)

            output_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez(output_file, **output_payload)
        except Exception:
            traceback.print_exc()
            return str(source_file), False, str(log_file)

    return str(source_file), True, str(log_file)


def _make_tasks(
    input_dir: Path,
    output_dir: Path,
    log_dir: Path,
    overwrite: bool,
    limit: int | None,
    smooth_window: int,
) -> list[tuple[str, str, str, int]]:
    source_files = sorted(input_dir.glob("*.npz"))
    if limit is not None:
        source_files = source_files[:limit]

    tasks = []
    for source_file in source_files:
        output_file = Path(normalize_retargeted_output_path(str(output_dir / source_file.name)))
        if output_file.exists() and not overwrite:
            continue
        log_file = log_dir / f"{output_file.stem}.log"
        tasks.append((str(source_file), str(output_file), str(log_file), smooth_window))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch retarget OMOMO robot-object motions with OmniRetargeting.")
    parser.add_argument("--input-dir", default="/home/duanxin/Datasets/OMOMO/omniretarget_robot_object/robot-object")
    parser.add_argument("--output-dir", default="/home/duanxin/Datasets/OMOMO/retargeted_omniretarget_torso")
    parser.add_argument("--log-dir", default="/home/duanxin/Datasets/OMOMO/retargeted_omniretarget_torso_logs")
    parser.add_argument("--robot-config", default="robot_models/unitree_g1/unitree_g1.json")
    parser.add_argument("--smplx-model-dir", default="/home/duanxin/Xyk/GMR/assets/body_models")
    parser.add_argument("--output-base-link", default="torso_link")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smooth-window", type=int, default=1)
    args = parser.parse_args()

    if args.smooth_window < 1 or args.smooth_window % 2 == 0:
        raise ValueError("--smooth-window must be a positive odd integer.")

    tasks = _make_tasks(
        Path(args.input_dir),
        Path(args.output_dir),
        Path(args.log_dir),
        overwrite=args.overwrite,
        limit=args.limit,
        smooth_window=args.smooth_window,
    )
    total = len(tasks)
    print(f"[retarget_omomo_batch] queued {total} file(s)")
    if total == 0:
        return

    completed = 0
    failed: list[tuple[str, str]] = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(args.robot_config, args.smplx_model_dir, args.output_base_link),
    ) as executor:
        futures = [executor.submit(_process_file, task) for task in tasks]
        for future in as_completed(futures):
            source_file, ok, log_file = future.result()
            completed += 1
            if not ok:
                failed.append((source_file, log_file))
                print(f"[{completed}/{total}] FAIL {source_file} (log: {log_file})", flush=True)
            elif completed == 1 or completed % 10 == 0 or completed == total:
                print(f"[{completed}/{total}] done", flush=True)

    if failed:
        print(f"[retarget_omomo_batch] failed {len(failed)} file(s):")
        for source_file, log_file in failed[:20]:
            print(f"  {source_file} (log: {log_file})")
        raise SystemExit(1)

    print(f"[retarget_omomo_batch] completed {total} file(s)")


if __name__ == "__main__":
    main()
