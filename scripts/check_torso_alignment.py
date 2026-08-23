#!/usr/bin/env python3
"""Measure torso-frame and body-axis fidelity for a retargeted motion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from omniretargeting.data_sources.smplx import _SMPLX_ROOT_OFFSET
from omniretargeting.utils import load_smplx_trajectory


SOURCE_JOINTS = {
    "Pelvis": 0,
    "L_Hip": 1,
    "R_Hip": 2,
    "Spine3": 9,
    "L_Shoulder": 16,
    "R_Shoulder": 17,
}


def _profile_model(profile_path: Path) -> mujoco.MjModel:
    profile = json.loads(profile_path.read_text())
    candidates = profile.get("urdf_path", [])
    if isinstance(candidates, str):
        candidates = [candidates]
    for candidate in candidates:
        resolved = Path(os.path.expandvars(os.path.expanduser(candidate)))
        if resolved.exists():
            return mujoco.MjModel.from_xml_path(str(resolved))
    raise FileNotFoundError(
        f"None of {profile_path}'s robot model paths exists: {candidates}"
    )


def _proper_polar_factor(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    sign = np.linalg.det(u @ vt)
    return u @ np.diag([1.0, 1.0, sign]) @ vt


def _quarter_means(values: np.ndarray) -> list[float]:
    return [float(part.mean()) for part in np.array_split(values, 4)]


def _direction_errors(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    source_joints: np.ndarray,
    robot_bodies: tuple[str, str],
    source_names: tuple[str, str],
) -> np.ndarray:
    data = mujoco.MjData(model)
    first_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, robot_bodies[0]
    )
    second_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, robot_bodies[1]
    )
    if first_id < 0 or second_id < 0:
        raise ValueError(f"Robot bodies not found: {robot_bodies}")
    source_first = SOURCE_JOINTS[source_names[0]]
    source_second = SOURCE_JOINTS[source_names[1]]
    errors = np.zeros(len(qpos), dtype=float)
    for frame_idx, frame in enumerate(qpos):
        data.qpos[:] = frame
        mujoco.mj_forward(model, data)
        robot_axis = data.xpos[second_id] - data.xpos[first_id]
        source_axis = (
            source_joints[frame_idx, source_second]
            - source_joints[frame_idx, source_first]
        )
        robot_axis /= max(float(np.linalg.norm(robot_axis)), 1e-12)
        source_axis /= max(float(np.linalg.norm(source_axis)), 1e-12)
        errors[frame_idx] = np.arccos(
            np.clip(float(np.dot(robot_axis, source_axis)), -1.0, 1.0)
        )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Retargeted *_retargeted.npz")
    parser.add_argument("--smplx-motion", type=Path, required=True)
    parser.add_argument("--smplx-model-dir", required=True)
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=Path("robot_models/unitree_g1/unitree_g1_instinctmj.json"),
    )
    args = parser.parse_args()

    output = np.load(args.output, allow_pickle=True)
    qpos = np.asarray(output["qpos"], dtype=float)
    source_joints = np.asarray(output["smplx_joints_w"], dtype=float)
    loaded = load_smplx_trajectory(
        args.smplx_motion,
        args.smplx_model_dir,
        return_meta=True,
        return_end_effectors=False,
    )
    source_orientations = np.asarray(loaded[1], dtype=float)
    if len(qpos) != len(source_orientations):
        raise ValueError(
            f"Frame mismatch: output={len(qpos)}, source={len(source_orientations)}"
        )

    robot_rotations = Rotation.from_quat(
        qpos[:, 3:7], scalar_first=True
    ).as_matrix()
    source_chest_rotations = (
        Rotation.from_quat(
            source_orientations[:, SOURCE_JOINTS["Spine3"]],
            scalar_first=True,
        ).as_matrix()
        @ _SMPLX_ROOT_OFFSET
    )
    cross_covariance = np.einsum(
        "tji,tjk->ik", robot_rotations, source_chest_rotations
    )
    constant_alignment = _proper_polar_factor(cross_covariance)
    aligned_robot = robot_rotations @ constant_alignment
    torso_errors = Rotation.from_matrix(
        source_chest_rotations @ np.transpose(aligned_robot, (0, 2, 1))
    ).magnitude()

    model = _profile_model(args.robot_config)
    axis_specs = {
        "shoulder_axis": (
            ("left_shoulder_roll_link", "right_shoulder_roll_link"),
            ("L_Shoulder", "R_Shoulder"),
        ),
        "hip_axis": (
            ("left_hip_roll_link", "right_hip_roll_link"),
            ("L_Hip", "R_Hip"),
        ),
        "torso_axis": (
            ("pelvis", "torso_link"),
            ("Pelvis", "Spine3"),
        ),
    }
    axis_errors = {
        name: _direction_errors(
            model, qpos, source_joints, robot_bodies, source_names
        )
        for name, (robot_bodies, source_names) in axis_specs.items()
    }

    print(f"output: {args.output} ({len(qpos)} frames)")
    print(
        "torso_vs_source_spine3_deg: "
        f"mean={np.degrees(torso_errors.mean()):.3f} "
        f"max={np.degrees(torso_errors.max()):.3f} "
        "quarters="
        + "/".join(
            f"{value:.3f}"
            for value in _quarter_means(np.degrees(torso_errors))
        )
    )
    for name, errors in axis_errors.items():
        degrees = np.degrees(errors)
        print(
            f"{name}_deg: mean={degrees.mean():.3f} "
            f"max={degrees.max():.3f} quarters="
            + "/".join(f"{value:.3f}" for value in _quarter_means(degrees))
        )


if __name__ == "__main__":
    main()
