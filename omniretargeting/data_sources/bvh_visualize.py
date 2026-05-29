"""Visualize BVH default pose against robot links from a robot config.

Usage:
  python -m omniretargeting.data_sources.bvh_visualize \
    --robot-config robot_models/unitree_g1/unitree_g1.json \
    --source-type lafan1 \
    --output /tmp/bvh_mapping.png

  # Interactive mode (omit --output):
  python -m omniretargeting.data_sources.bvh_visualize \
    --robot-config robot_models/unitree_g1/unitree_g1.json \
    --source-type lafan1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from omniretargeting.data_sources.lafan1 import _read_bvh, _quat_fk, _ROTATION_MATRIX
from omniretargeting.robot_config import load_robot_config
from omniretargeting.utils import detect_robot_height


def _set_equal_axes(ax, points):
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max() + x.min()) * 0.5
    mid_y = (y.max() + y.min()) * 0.5
    mid_z = (z.max() + z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def _apply_default_joint_positions(model, data, default_joint_positions):
    if not default_joint_positions:
        return
    for joint_name, joint_value in default_joint_positions.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint '{joint_name}' not found in URDF.")
        qpos_adr = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_adr] = float(joint_value)


def _load_robot_pose(urdf_path, default_joint_positions=None):
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    if model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and model.nq >= 7:
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    _apply_default_joint_positions(model, data, default_joint_positions)
    mujoco.mj_forward(model, data)

    body_ids = {}
    body_positions = np.zeros((model.nbody, 3))
    body_rotations = np.zeros((model.nbody, 3, 3))
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name:
            body_ids[name] = i
        body_positions[i] = data.xpos[i].copy()
        body_rotations[i] = data.xmat[i].reshape(3, 3).copy()
    return model, data, body_ids, body_positions, body_rotations




def _resolve_robot_pose(robot_config, source_entry):
    all_poses = robot_config.get("default_joint_positions", {})
    if not all_poses:
        return {}
    first_val = next(iter(all_poses.values()), None)
    if not isinstance(first_val, dict):
        return all_poses
    pose_name = source_entry.get("default_pose_on_robot", list(all_poses.keys())[0])
    if pose_name in all_poses:
        return all_poses[pose_name]
    first_name = next(iter(all_poses.keys()))
    print(f"[bvh_viz] pose {pose_name!r} not found, using {first_name!r}")
    return all_poses[first_name]


def _select_source(robot_config, source_type):
    sources = robot_config.get("source", [])
    for s in sources:
        if s.get("type") == source_type or s.get("name") == source_type:
            return s
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Visualize BVH default pose against robot links."
    )
    parser.add_argument(
        "--robot-config", "--robot_config", dest="robot_config",
        required=True, help="Path to robot config JSON.",
    )
    parser.add_argument(
        "--source-type", "--source", dest="source_type",
        default="lafan1", help="Source type in the robot config (default: lafan1).",
    )
    parser.add_argument(
        "--bvh-file", "--bvh", dest="bvh_file",
        default="/home/leo/Projects/ubisoft-laforge-animation-dataset/lafan1/lafan1/fallAndGetUp1_subject1.bvh",
        help="Path to BVH file for default pose skeleton (any BVH works).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output PNG path. Omit for interactive display.",
    )
    parser.add_argument(
        "--yaw", type=float, default=-90.0,
        help="Yaw rotation (degrees) to apply to the robot default pose around Z axis. Default: -90.",
    )
    parser.add_argument(
        "--scale-with-robot", dest="scale_with_robot", action="store_true", default=False,
        help="Scale source positions to match robot height.",
    )
    args = parser.parse_args()

    if args.output:
        matplotlib.use("Agg", force=True)

    # Load robot config and select source
    config_path = Path(args.robot_config).expanduser().resolve()
    robot_config = load_robot_config(config_path)

    source_entry = _select_source(robot_config, args.source_type)
    if source_entry is None:
        avail = [s.get("type") for s in robot_config.get("source", [])]
        print(f"Source type '{args.source_type}' not found. Available: {avail}")
        sys.exit(1)

    joint_mapping = source_entry.get("target_mapping", {})
    link_offset_config = source_entry.get("link_offset_config",
                                          robot_config.get("link_offset_config"))
    pose_name = source_entry.get("default_pose_on_robot", "T-Pose")
    print(f"[bvh_viz] source={args.source_type}, robot_pose={pose_name}, "
          f"mapped_joints={len(joint_mapping)}")

    # Load LAFAN1 default pose
    bvh = _read_bvh(args.bvh_file)
    n = len(bvh["names"])
    quats = bvh["quats"][:1]  # first frame actual rotations
    pos = bvh["positions"][:1]  # first frame local positions
    pos[0, 0] = [0, 0, 0]
    _, gp = _quat_fk(quats, pos, bvh["parents"])
    src_positions = gp[0] @ _ROTATION_MATRIX.T / 100.0
    src_names = bvh["names"]
    src_parents = bvh["parents"].tolist()
    src_name_to_idx = {name: i for i, name in enumerate(src_names)}

    # Load robot default pose
    robot_urdf_path = robot_config["urdf_path"]
    if not Path(robot_urdf_path).is_absolute():
        robot_urdf_path = str(config_path.parent / robot_urdf_path)
    default_pose_dict = _resolve_robot_pose(robot_config, source_entry)
    model, data, body_ids, body_positions, body_rotations = _load_robot_pose(
        robot_urdf_path, default_pose_dict if default_pose_dict else None,
    )

    # Apply yaw rotation around Z axis to robot body positions
    if args.yaw != 0.0:
        yaw_rad = np.radians(args.yaw)
        cos_a, sin_a = np.cos(yaw_rad), np.sin(yaw_rad)
        rot_z = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
        for i in range(len(body_positions)):
            body_positions[i] = body_positions[i] @ rot_z.T
        print(f"[bvh_viz] applied yaw rotation: {args.yaw} degrees")

    # Scale source to robot height (anchor at foot level)
    if args.scale_with_robot:
        from omniretargeting.data_sources.registry import create_data_source
        data_source = create_data_source(args.source_type, args.bvh_file)
        src_height = data_source.source_height or 1.75

        robot_height = detect_robot_height(model, data)
        robot_foot_z = min(
            body_positions[body_ids.get("left_ankle_roll_link", 0), 2],
            body_positions[body_ids.get("right_ankle_roll_link", 0), 2],
        )
        foot_idxs = [src_name_to_idx.get(n) for n in ["LeftFoot", "RightFoot"] if n in src_name_to_idx]
        src_foot_z = min(src_positions[i, 2] for i in foot_idxs) if foot_idxs else -0.85
        scale = robot_height / src_height
        src_positions[:, 2] = (src_positions[:, 2] - src_foot_z) * scale + robot_foot_z
        src_positions[:, :2] = src_positions[:, :2] * scale
        print(f"[bvh_viz] scaled: robot_h={robot_height:.3f}m, src_h={src_height:.3f}m, scale={scale:.4f}")

    missing = sorted({b for b in joint_mapping.values() if b not in body_ids})
    if missing:
        print(f"[bvh_viz] WARNING: mapped bodies not in URDF: {missing}")

    # Print mapping table
    print(f"\n=== {args.source_type} mapping ({pose_name}) ===")
    mapped_set = set(joint_mapping.keys())
    for src_name, link_name in sorted(joint_mapping.items()):
        si = src_name_to_idx.get(src_name, -1)
        sp = src_positions[si] if si >= 0 else None
        rp = body_positions[body_ids[link_name]] if link_name in body_ids else None
        if sp is not None and rp is not None:
            print(f"  {src_name:20s} -> {link_name:30s}  "
                  f"src=({sp[0]:.3f},{sp[1]:.3f},{sp[2]:.3f})  "
                  f"robot=({rp[0]:.3f},{rp[1]:.3f},{rp[2]:.3f})")
        elif sp is not None:
            print(f"  {src_name:20s} -> {link_name:30s}  "
                  f"src=({sp[0]:.3f},{sp[1]:.3f},{sp[2]:.3f})  robot=(MISSING)")

    # ---- Plot ----
    fig = plt.figure(figsize=(21, 10))

    # Left: LAFAN1 skeleton
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_title("LAFAN1 Default Pose (Rest-Pose)\n(green=mapped, gray=unmapped)",
                  fontsize=12, fontweight="bold")
    for i, (name, parent) in enumerate(zip(src_names, src_parents)):
        if parent >= 0:
            p1, p2 = src_positions[parent], src_positions[i]
            is_mapped = name in mapped_set
            color = "green" if is_mapped else "gray"
            ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                     color=color, linewidth=2 if is_mapped else 1,
                     alpha=0.8 if is_mapped else 0.3)
            ax1.scatter(*p2, c=color, s=15 if is_mapped else 8)
        else:
            ax1.scatter(*src_positions[i], c="darkgreen", s=50)
    for name in src_names:
        idx = src_name_to_idx.get(name)
        if idx is not None:
            pos = src_positions[idx]
            c = "darkgreen" if name in mapped_set else "darkred"
            ax1.text(pos[0], pos[1], pos[2], name, fontsize=6, color=c, ha="right")
    _set_equal_axes(ax1, src_positions)
    ax1.set_xlabel("X (forward)"); ax1.set_ylabel("Y (left)"); ax1.set_zlabel("Z (up)")

    # Right: both + mapping
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_title(f"LAFAN1 (blue) <-> Robot {pose_name} (red)\n"
                  f"{len(joint_mapping)} mapped joints",
                  fontsize=12, fontweight="bold")

    # Robot skeleton
    for body_idx in range(1, model.nbody):
        parent_idx = int(model.body_parentid[body_idx])
        p0, p1 = body_positions[parent_idx], body_positions[body_idx]
        ax2.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                 "r-", linewidth=2, alpha=0.6)
    for body_idx in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_idx)
        if name and body_idx > 0:
            ax2.scatter(*body_positions[body_idx], c="red", s=15)

    # LAFAN1 skeleton offset
    offset = np.array([-0.4, 0.0, 0.0])
    for i, (name, parent) in enumerate(zip(src_names, src_parents)):
        if parent >= 0:
            p1, p2 = src_positions[parent] + offset, src_positions[i] + offset
            c = "blue" if name in mapped_set else "lightgray"
            ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                     color=c, linewidth=1.5, alpha=0.5)
    for name in src_names:
        idx = src_name_to_idx.get(name)
        if idx is not None:
            pos = src_positions[idx] + offset
            c = "darkblue" if name in mapped_set else "gray"
            ax2.scatter(*pos, c=c, s=8, alpha=0.6)

    # Mapping lines
    import matplotlib.cm as cm
    colors = cm.tab20(np.linspace(0, 1, len(joint_mapping)))
    for ci, (src_name, link_name) in enumerate(sorted(joint_mapping.items())):
        si = src_name_to_idx.get(src_name)
        if si is None or link_name not in body_ids:
            continue
        sp = src_positions[si] + offset
        rp = body_positions[body_ids[link_name]]
        ax2.plot([sp[0], rp[0]], [sp[1], rp[1]], [sp[2], rp[2]],
                 "-", linewidth=2, color=colors[ci], alpha=0.7)
        ax2.scatter([rp[0]], [rp[1]], [rp[2]], c=[colors[ci]], s=80,
                    marker="X", edgecolors="black", linewidths=0.8)
        mid = (sp + rp) / 2
        ax2.text(mid[0], mid[1], mid[2], src_name, fontsize=5, color="darkred")

        if link_offset_config and link_name in link_offset_config:
            o_local = np.asarray(link_offset_config[link_name], dtype=float).reshape(3)
            o_world = body_rotations[body_ids[link_name]] @ o_local
            ot = rp + o_world
            ax2.plot([rp[0], ot[0]], [rp[1], ot[1]], [rp[2], ot[2]],
                     "-", linewidth=1.5, color="#00a896", alpha=0.7)
            ax2.scatter([ot[0]], [ot[1]], [ot[2]], c="#00a896", s=30, marker="o")

    all_pts = np.vstack([body_positions, src_positions + offset])
    _set_equal_axes(ax2, all_pts)
    ax2.set_xlabel("X (forward)"); ax2.set_ylabel("Y (left)"); ax2.set_zlabel("Z (up)")
    ax2.view_init(elev=25, azim=-60)
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[bvh_viz] saved: {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
