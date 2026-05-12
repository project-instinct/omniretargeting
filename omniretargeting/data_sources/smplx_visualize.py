"""Visualize default SMPL-X joints against robot links from a robot config."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mujoco
import numpy as np

from .robot_config import load_robot_config


SMPLX_JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
    "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
]

SMPLX_BONES = [
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    (0, 1), (1, 4), (4, 7), (7, 10),
    (0, 2), (2, 5), (5, 8), (8, 11),
    (9, 13), (13, 16), (16, 18), (18, 20),
    (9, 14), (14, 17), (17, 19), (19, 21),
]

DEFAULT_SMPLX_OFFSETS = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.1, -0.1],
    [0.0, -0.1, -0.1],
    [0.0, 0.0, 0.2],
    [0.0, 0.1, -0.5],
    [0.0, -0.1, -0.5],
    [0.0, 0.0, 0.4],
    [0.0, 0.1, -0.9],
    [0.0, -0.1, -0.9],
    [0.0, 0.0, 0.6],
    [0.05, 0.1, -0.95],
    [0.05, -0.1, -0.95],
    [0.0, 0.0, 0.8],
    [0.0, 0.15, 0.75],
    [0.0, -0.15, 0.75],
    [0.0, 0.0, 0.95],
    [0.0, 0.3, 0.75],
    [0.0, -0.3, 0.75],
    [0.0, 0.55, 0.75],
    [0.0, -0.55, 0.75],
    [0.0, 0.75, 0.75],
    [0.0, -0.75, 0.75],
], dtype=float)

DEFAULT_SMPLX_HEIGHT = float(DEFAULT_SMPLX_OFFSETS[:, 2].max() - DEFAULT_SMPLX_OFFSETS[:, 2].min())


def _joint_color(name: str) -> tuple[float, float, float]:
    if name.startswith("L_"):
        return (0.20, 0.55, 0.90)
    if name.startswith("R_"):
        return (0.90, 0.40, 0.25)
    if name in {"Head", "Neck"}:
        return (0.80, 0.70, 0.25)
    return (0.65, 0.65, 0.65)


def _detect_robot_height(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    min_z = float("inf")
    max_z = float("-inf")

    for body_idx in range(model.nbody):
        z = float(data.xpos[body_idx][2])
        min_z = min(min_z, z)
        max_z = max(max_z, z)

    for geom_idx in range(model.ngeom):
        z = float(data.geom_xpos[geom_idx][2])
        geom_size = model.geom_size[geom_idx]
        geom_type = model.geom_type[geom_idx]
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            radius = float(geom_size[0])
            min_z = min(min_z, z - radius)
            max_z = max(max_z, z + radius)
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            radius = float(geom_size[0])
            half_height = float(geom_size[1])
            min_z = min(min_z, z - half_height - radius)
            max_z = max(max_z, z + half_height + radius)
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            half_size = float(geom_size[2])
            min_z = min(min_z, z - half_size)
            max_z = max(max_z, z + half_size)
        else:
            min_z = min(min_z, z)
            max_z = max(max_z, z)

    height = max_z - min_z
    if 0.3 <= height <= 3.0:
        return float(height)
    return 1.6


def _apply_joint_pos_fitting_smplx(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_pos_fitting_smplx: dict[str, float] | None,
) -> None:
    if not joint_pos_fitting_smplx:
        return

    for joint_name, joint_value in joint_pos_fitting_smplx.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Robot joint '{joint_name}' from joint_pos_fitting_smplx was not found in the URDF.")
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Robot joint '{joint_name}' in joint_pos_fitting_smplx cannot be a free joint.")
        qpos_adr = int(model.jnt_qposadr[joint_id])
        next_qpos_adr = model.nq
        for next_joint_id in range(joint_id + 1, model.njnt):
            candidate = int(model.jnt_qposadr[next_joint_id])
            if candidate > qpos_adr:
                next_qpos_adr = candidate
                break
        qpos_width = next_qpos_adr - qpos_adr
        if qpos_width != 1:
            raise ValueError(
                f"Robot joint '{joint_name}' in joint_pos_fitting_smplx must map to exactly one qpos entry, got {qpos_width}."
            )
        data.qpos[qpos_adr] = float(joint_value)


def _load_robot_default_pose(
    urdf_path: str | Path,
    joint_pos_fitting_smplx: dict[str, float] | None = None,
) -> tuple[mujoco.MjModel, mujoco.MjData, dict[str, int], np.ndarray, np.ndarray]:
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    if model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and model.nq >= 7:
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    _apply_joint_pos_fitting_smplx(model, data, joint_pos_fitting_smplx)
    mujoco.mj_forward(model, data)

    body_ids = {}
    body_positions = np.zeros((model.nbody, 3), dtype=float)
    body_rotations = np.zeros((model.nbody, 3, 3), dtype=float)
    for body_idx in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_idx)
        if body_name:
            body_ids[body_name] = body_idx
        body_positions[body_idx] = data.xpos[body_idx].copy()
        body_rotations[body_idx] = data.xmat[body_idx].reshape(3, 3).copy()

    return model, data, body_ids, body_positions, body_rotations


def _build_default_smplx_pose(pelvis_position: np.ndarray, robot_height: float | None) -> np.ndarray:
    scale = 1.0
    if robot_height is not None and robot_height > 0:
        scale = float(robot_height) / DEFAULT_SMPLX_HEIGHT
    return pelvis_position[None, :] + DEFAULT_SMPLX_OFFSETS * scale


SMPLX_MODEL_SEARCH_PATHS = [
    "/localhdd/Datasets/smplx",
    "/localhdd/Datasets/",
    "data/body_models/smplx",
]


def _load_smplx_joints_from_betas(
    betas: list[float],
    smplx_model_dir: str | None = None,
) -> np.ndarray | None:
    try:
        import smplx as smplx_lib
        import torch
    except ImportError:
        print("[visualize_offsets] smplx/torch not available, falling back to hardcoded template")
        return None

    import os
    search_paths = [smplx_model_dir] if smplx_model_dir else []
    search_paths.extend(SMPLX_MODEL_SEARCH_PATHS)
    model_path = next((p for p in search_paths if p and os.path.exists(p)), None)
    if model_path is None:
        print("[visualize_offsets] SMPL-X model files not found, falling back to hardcoded template")
        return None

    num_betas = len(betas)
    model = smplx_lib.SMPLX(model_path, num_betas=num_betas, use_hands=False, use_face=False)
    betas_tensor = torch.tensor([betas], dtype=torch.float32)
    with torch.no_grad():
        out = model(betas=betas_tensor)
    joints_smplx = out.joints[0, :22].numpy()

    # Transform from SMPL-X convention (+X left, +Y up, +Z forward)
    # to robot convention (+X forward, +Y left, +Z up)
    joints_robot = np.zeros_like(joints_smplx)
    joints_robot[:, 0] = joints_smplx[:, 2]   # X_robot = Z_smplx (forward)
    joints_robot[:, 1] = joints_smplx[:, 0]   # Y_robot = X_smplx (left)
    joints_robot[:, 2] = joints_smplx[:, 1]   # Z_robot = Y_smplx (up)

    return joints_robot


def _set_equal_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = max(float((maxs - mins).max()) / 2.0, 0.5)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _plot_visualization(
    smplx_joints: np.ndarray,
    model: mujoco.MjModel,
    body_positions: np.ndarray,
    body_rotations: np.ndarray,
    joint_mapping: dict[str, str],
    link_offset_config: dict[str, object] | None,
    output_path: Path | None = None,
) -> None:
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection="3d")

    mapped_link_legend = Line2D(
        [0],
        [0],
        linestyle="None",
        marker="X",
        markersize=10,
        markerfacecolor="#8a2be2",
        markeredgecolor="black",
        markeredgewidth=0.9,
        label="Mapped robot links",
    )
    offset_target_legend = Line2D(
        [0],
        [0],
        linestyle="None",
        marker="o",
        markersize=8,
        markerfacecolor="#00a896",
        markeredgecolor="black",
        markeredgewidth=0.8,
        label="Offset target positions",
    )

    if model.nbody > 1:
        ax.scatter(
            body_positions[1:, 0],
            body_positions[1:, 1],
            body_positions[1:, 2],
            facecolors="none",
            edgecolors="#111111",
            s=95,
            linewidths=1.4,
            alpha=0.95,
            label="Robot links",
        )

    for body_idx in range(1, model.nbody):
        parent_idx = int(model.body_parentid[body_idx])
        p0 = body_positions[parent_idx]
        p1 = body_positions[body_idx]
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color=(0.75, 0.75, 0.75),
            linewidth=1.0,
            alpha=0.5,
        )

    for parent_idx, child_idx in SMPLX_BONES:
        p0 = smplx_joints[parent_idx]
        p1 = smplx_joints[child_idx]
        ax.plot(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            [p0[2], p1[2]],
            color=(0.15, 0.15, 0.15),
            linewidth=2.0,
            alpha=0.9,
        )

    joint_colors = [_joint_color(name) for name in SMPLX_JOINT_NAMES]
    ax.scatter(
        smplx_joints[:, 0],
        smplx_joints[:, 1],
        smplx_joints[:, 2],
        c=joint_colors,
        s=90,
        depthshade=True,
        edgecolors="white",
        linewidths=0.9,
        alpha=0.98,
        label="Default SMPL-X joints",
    )

    smplx_name_to_index = {name: idx for idx, name in enumerate(SMPLX_JOINT_NAMES)}
    offset_target_points = []
    for smplx_name, body_name in joint_mapping.items():
        joint_idx = smplx_name_to_index.get(smplx_name)
        if joint_idx is None:
            continue
        body_idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_idx < 0:
            continue
        smplx_point = smplx_joints[joint_idx]
        robot_point = body_positions[body_idx]
        color = _joint_color(smplx_name)
        ax.plot(
            [smplx_point[0], robot_point[0]],
            [smplx_point[1], robot_point[1]],
            [smplx_point[2], robot_point[2]],
            linestyle="--",
            linewidth=1.6,
            color=color,
            alpha=0.9,
        )
        ax.scatter(
            [robot_point[0]],
            [robot_point[1]],
            [robot_point[2]],
            c=[color],
            s=135,
            marker="X",
            edgecolors="black",
            linewidths=0.9,
            zorder=12,
        )
        ax.text(
            robot_point[0],
            robot_point[1],
            robot_point[2] + 0.03,
            body_name,
            fontsize=7,
            color="black",
        )
        if link_offset_config and body_name in link_offset_config:
            offset_local = np.asarray(link_offset_config[body_name], dtype=float).reshape(3)
            offset_world = body_rotations[body_idx] @ offset_local
            offset_target = robot_point + offset_world
            offset_target_points.append(offset_target)
            ax.plot(
                [robot_point[0], offset_target[0]],
                [robot_point[1], offset_target[1]],
                [robot_point[2], offset_target[2]],
                linestyle="-",
                linewidth=2.0,
                color="#00a896",
                alpha=0.85,
            )
            ax.scatter(
                [offset_target[0]],
                [offset_target[1]],
                [offset_target[2]],
                c=["#00a896"],
                s=85,
                marker="o",
                edgecolors="black",
                linewidths=0.8,
                zorder=13,
            )

    for joint_name in ["Pelvis", "Head", "L_Wrist", "R_Wrist", "L_Ankle", "R_Ankle"]:
        point = smplx_joints[smplx_name_to_index[joint_name]]
        ax.text(point[0], point[1], point[2] + 0.03, joint_name, fontsize=8)

    all_points = [smplx_joints, body_positions]
    if offset_target_points:
        all_points.append(np.asarray(offset_target_points, dtype=float))
    all_points = np.vstack(all_points)
    _set_equal_axes(ax, all_points)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Default SMPL-X joints vs robot links")
    ax.grid(True, alpha=0.3)
    ax.view_init(elev=20, azim=35)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mapped_link_legend)
    labels.append("Mapped robot links")
    if offset_target_points:
        handles.append(offset_target_legend)
        labels.append("Offset target positions")
    ax.legend(handles, labels, loc="upper right")
    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize the default SMPL-X pose against robot links from a robot config."
    )
    parser.add_argument(
        "--robot_config",
        "--robot-config",
        dest="robot_config",
        required=True,
        help="Path to robot config JSON.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output PNG path.",
    )
    parser.add_argument(
        "--smplx_model_dir",
        "--smplx-model-dir",
        dest="smplx_model_dir",
        type=str,
        default=None,
        help="Path to SMPL-X model directory (overrides search paths).",
    )
    args = parser.parse_args()

    if args.output:
        matplotlib.use("Agg", force=True)

    config_path = Path(args.robot_config).expanduser().resolve()
    robot_config = load_robot_config(config_path)
    robot_urdf_path = robot_config.get("urdf_path")
    if not robot_urdf_path:
        raise ValueError("Robot config must define 'urdf_path'.")

    joint_mapping = robot_config.get("joint_mapping")
    if not isinstance(joint_mapping, dict) or not joint_mapping:
        raise ValueError("Robot config must define a non-empty 'joint_mapping'.")

    joint_pos_fitting_smplx = robot_config.get("joint_pos_fitting_smplx")
    model, data, body_ids, body_positions, body_rotations = _load_robot_default_pose(
        robot_urdf_path,
        joint_pos_fitting_smplx=joint_pos_fitting_smplx,
    )
    missing_bodies = sorted({body_name for body_name in joint_mapping.values() if body_name not in body_ids})
    if missing_bodies:
        raise ValueError(f"Mapped robot bodies were not found in the URDF: {missing_bodies}")

    link_offset_config = robot_config.get("link_offset_config")
    if link_offset_config is not None and not isinstance(link_offset_config, dict):
        raise ValueError("Robot config 'link_offset_config' must be a JSON object when provided.")

    robot_height = robot_config.get("robot_height")
    if robot_height is None:
        robot_height = _detect_robot_height(model, data)

    pelvis_body_name = joint_mapping.get("Pelvis")
    pelvis_position = body_positions[body_ids[pelvis_body_name]] if pelvis_body_name else np.zeros(3, dtype=float)

    smplx_betas = robot_config.get("smplx_betas")
    smplx_joints = None
    if smplx_betas is not None:
        raw_joints = _load_smplx_joints_from_betas(smplx_betas, smplx_model_dir=args.smplx_model_dir)
        if raw_joints is not None:
            smplx_height = float(raw_joints[:, 2].max() - raw_joints[:, 2].min())
            scale = robot_height / smplx_height if smplx_height > 0 else 1.0
            smplx_joints = pelvis_position[None, :] + (raw_joints - raw_joints[0:1]) * scale
            print(f"[visualize_offsets] using SMPL-X model with {len(smplx_betas)} betas, smplx_height={smplx_height:.3f} m, scale={scale:.3f}")

    if smplx_joints is None:
        smplx_joints = _build_default_smplx_pose(pelvis_position=pelvis_position, robot_height=robot_height)
        print("[visualize_offsets] using hardcoded SMPL-X template")

    output_path = Path(args.output) if args.output else None
    _plot_visualization(
        smplx_joints=smplx_joints,
        model=model,
        body_positions=body_positions,
        body_rotations=body_rotations,
        joint_mapping=joint_mapping,
        link_offset_config=link_offset_config,
        output_path=output_path,
    )

    print(f"[visualize_offsets] robot_config={config_path}")
    print(f"[visualize_offsets] robot_urdf={robot_urdf_path}")
    print(f"[visualize_offsets] joint_pos_fitting_smplx={0 if not joint_pos_fitting_smplx else len(joint_pos_fitting_smplx)}")
    print(f"[visualize_offsets] smplx_betas={'none' if not smplx_betas else len(smplx_betas)}")
    print(f"[visualize_offsets] robot_height={robot_height:.3f} m")
    print(f"[visualize_offsets] mapped_links={len(joint_mapping)}")
    print(f"[visualize_offsets] link_offsets={0 if not link_offset_config else len(link_offset_config)}")
    if output_path is not None:
        print(f"[visualize_offsets] output={output_path}")


if __name__ == "__main__":
    main()
