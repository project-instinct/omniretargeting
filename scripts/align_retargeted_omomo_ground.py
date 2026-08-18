from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


def _moving_average_reflect(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    if window % 2 == 0:
        raise ValueError("--smooth-window must be odd.")
    if len(values) < 2:
        return values

    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="reflect")
    kernel = np.ones(window, dtype=values.dtype) / window
    return np.convolve(padded, kernel, mode="valid")


def _model_joint_names(model: mujoco.MjModel) -> list[str]:
    return [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
    ]


def _load_qpos(model: mujoco.MjModel, motion: np.lib.npyio.NpzFile) -> np.ndarray:
    joint_names = motion["joint_names"] if isinstance(motion["joint_names"], list) else motion["joint_names"].tolist()
    joint_order = [joint_names.index(name) for name in _model_joint_names(model)]

    qpos = np.zeros((len(motion["base_pos_w"]), model.nq), dtype=np.float64)
    qpos[:, :3] = np.asarray(motion["base_pos_w"], dtype=np.float64)
    qpos[:, 3:7] = np.asarray(motion["base_quat_w"], dtype=np.float64)
    qpos[:, 7:] = np.asarray(motion["joint_pos"], dtype=np.float64)[:, joint_order]
    return qpos


def _foot_geom_ids(model: mujoco.MjModel, body_name_tokens: tuple[str, ...]) -> list[int]:
    geom_ids: list[int] = []
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom_id]) or ""
        if any(token in body_name for token in body_name_tokens):
            geom_ids.append(geom_id)
    if not geom_ids:
        raise ValueError(f"No foot geoms matched body-name tokens: {body_name_tokens}")
    return geom_ids


def _geom_min_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    geom_type = model.geom_type[geom_id]
    geom_pos = data.geom_xpos[geom_id]
    geom_xmat = data.geom_xmat[geom_id].reshape(3, 3)

    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = model.geom_dataid[geom_id]
        vert_start = model.mesh_vertadr[mesh_id]
        vert_count = model.mesh_vertnum[mesh_id]
        vertices = model.mesh_vert[vert_start : vert_start + vert_count]
        return float((geom_pos + vertices @ geom_xmat.T)[:, 2].min())
    if geom_type in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        return float(geom_pos[2] - model.geom_size[geom_id, 0] - model.geom_size[geom_id, 1])
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(geom_pos[2] - model.geom_size[geom_id, 0])
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        half_extents = model.geom_size[geom_id]
        corners = np.array(
            [
                [x, y, z]
                for x in (-half_extents[0], half_extents[0])
                for y in (-half_extents[1], half_extents[1])
                for z in (-half_extents[2], half_extents[2])
            ],
            dtype=np.float64,
        )
        return float((geom_pos + corners @ geom_xmat.T)[:, 2].min())
    return float(geom_pos[2])


def _foot_min_z(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray, foot_geom_ids: list[int]) -> np.ndarray:
    min_z = np.zeros(len(qpos), dtype=np.float64)
    for frame_idx in range(len(qpos)):
        data.qpos[:] = qpos[frame_idx]
        mujoco.mj_forward(model, data)
        min_z[frame_idx] = min(_geom_min_z(model, data, geom_id) for geom_id in foot_geom_ids)
    return min_z


def _process_file(
    motion_file: Path,
    output_dir: Path,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    foot_geom_ids: list[int],
    target_min_z: float,
    smooth_window: int,
    preserve_root_z: bool,
    overwrite: bool,
) -> bool:
    output_file = output_dir / motion_file.name
    if output_file.exists() and not overwrite:
        return False

    motion = np.load(motion_file, allow_pickle=True)
    qpos = _load_qpos(model, motion)
    min_z = _foot_min_z(model, data, qpos, foot_geom_ids)
    z_delta = _moving_average_reflect(target_min_z - min_z, smooth_window)

    payload = {key: motion[key] for key in motion.files}
    base_pos = np.asarray(motion["base_pos_w"], dtype=np.float64).copy()
    if not preserve_root_z:
        base_pos[:, 2] += z_delta
    payload["base_pos_w"] = base_pos

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_file, **payload)
    print(
        f"{motion_file.name}: foot_min_z mean={min_z.mean():.5f}, "
        f"z_delta mean={z_delta.mean():.5f}"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Align retargeted OMOMO robot foot geometry to the ground plane.")
    parser.add_argument("--retargeted-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--robot-config", default="robot_models/unitree_g1/unitree_g1_instinctmj.json")
    parser.add_argument("--foot-body-tokens", nargs="*", default=["ankle", "foot"])
    parser.add_argument("--target-min-z", type=float, default=0.0)
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--preserve-root-z", action="store_true", help="Do not apply the computed z alignment to base_pos_w.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.smooth_window < 1 or args.smooth_window % 2 == 0:
        raise ValueError("--smooth-window must be a positive odd integer.")

    with open(args.robot_config, "r", encoding="utf-8") as f:
        robot_config = json.load(f)

    model = mujoco.MjModel.from_xml_path(robot_config["urdf_path"])
    data = mujoco.MjData(model)
    foot_geom_ids = _foot_geom_ids(model, tuple(args.foot_body_tokens))

    motion_files = sorted(Path(args.retargeted_dir).glob("*.npz"))
    if args.limit is not None:
        motion_files = motion_files[: args.limit]

    processed = 0
    for motion_file in motion_files:
        if _process_file(
            motion_file,
            Path(args.output_dir),
            model,
            data,
            foot_geom_ids,
            args.target_min_z,
            args.smooth_window,
            args.preserve_root_z,
            args.overwrite,
        ):
            processed += 1
    print(f"[align_retargeted_omomo_ground] processed {processed} file(s)")


if __name__ == "__main__":
    main()
