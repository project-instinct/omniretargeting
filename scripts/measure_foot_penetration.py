#!/usr/bin/env python3
"""Measure how far the foot collision geometry is through the scene.

The check that matters for downstream RL is not whether the visual mesh looks
right but whether the collision points the simulator will use are on the correct
side of the surface. This measures exactly those points: every collision geom on
each foot body, sampled in body coordinates and carried through the retargeted
motion by forward kinematics.

Two criteria, because neither alone is trustworthy on a scanned scene:

  A. Signed distance along the contact normal of the anchored contact plane.
     Defined only on frames the detector calls contact, but it is the criterion
     that matches what a contact solver will see.

  B. Height above the ground directly below, by vertical ray cast. Defined on
     every frame, and independent of contact detection.

A signed-distance query against the mesh itself is deliberately not used: these
scans are not consistently wound, and on the scene this was developed against
only 84 percent of points known to be outside were judged outside, which is not
a basis for claiming anything about the remaining 16.

Usage:
    measure_foot_penetration.py MOTION.npz --scene scene.obj --source smplx.npz \\
        --urdf robot.urdf [--model-dir DIR] [--bodies left=NAME right=NAME]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from omniretargeting.core import OmniRetargeter
from omniretargeting.data_sources.smplx import SmplxDataSource
from omniretargeting.foot_frames import sample_body_collision_points
from omniretargeting.scene_contact import ContactSpec, infer_contact_targets
from omniretargeting.utils import load_robot_urdf_with_floating_base

DEFAULT_BODIES = {"left": "left_ankle_roll_link", "right": "right_ankle_roll_link"}


def load_model(urdf: Path) -> mujoco.MjModel:
    """Compile the URDF from its own directory, so its mesh paths resolve."""
    cwd = os.getcwd()
    try:
        os.chdir(urdf.parent)
        return load_robot_urdf_with_floating_base(Path(urdf.name))
    finally:
        os.chdir(cwd)


def measure(
    motion_file: Path,
    scene_file: Path,
    source_file: Path,
    urdf: Path,
    model_dir: Path,
    bodies: dict[str, str],
    max_points: int = 128,
) -> dict:
    terrain = trimesh.load(str(scene_file), force="mesh")

    source = SmplxDataSource(
        motion_file=str(source_file), model_directory=str(model_dir), gender="neutral"
    )
    motion = source.load()
    centres = motion.metadata["foot_sole_centres"]
    clouds = motion.metadata.get("foot_sole_points")
    targets, normals, _, _ = infer_contact_targets(
        centres,
        terrain,
        motion.framerate,
        ContactSpec(),
        effector_lengths=OmniRetargeter._source_effector_lengths(motion, 1.0),
        leg_length=OmniRetargeter._source_leg_length(
            motion.positions, motion.target_names, centres
        ),
        effector_clouds=clouds,
    )

    model = load_model(urdf)
    data = mujoco.MjData(model)
    body_ids = {
        side: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for side, name in bodies.items()
    }
    missing = [side for side, bid in body_ids.items() if bid < 0]
    if missing:
        raise SystemExit(f"body not found in {urdf.name} for: {', '.join(missing)}")
    probes = {
        side: sample_body_collision_points(model, bid, max_points=max_points)
        for side, bid in body_ids.items()
    }
    for side, points in probes.items():
        if len(points) <= 1:
            print(
                f"Warning: {bodies[side]!r} yielded {len(points)} collision point(s); "
                "this body probably carries no collision geometry"
            )

    caster = trimesh.ray.ray_triangle.RayMeshIntersector(terrain)
    saved = np.load(motion_file)
    joint_names = [str(name) for name in saved["joint_names"]]
    address = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i): int(model.jnt_qposadr[i])
        for i in range(model.njnt)
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
        and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    }

    plane_gaps: list[float] = []
    ray_gaps: list[float] = []
    for frame in range(len(saved["joint_pos"])):
        data.qpos[:3] = saved["base_pos_w"][frame]
        data.qpos[3:7] = saved["base_quat_w"][frame]
        for name, value in zip(joint_names, saved["joint_pos"][frame]):
            if name in address:
                data.qpos[address[name]] = value
        mujoco.mj_kinematics(model, data)

        for side, body_id in body_ids.items():
            world = data.xpos[body_id] + probes[side] @ data.xmat[body_id].reshape(3, 3).T
            if side in targets and np.isfinite(targets[side][frame, 0]):
                plane_gaps.append(
                    float(np.einsum("ij,j->i", world - targets[side][frame], normals[side][frame]).min())
                )
            origins = world.copy()
            origins[:, 2] += 2.0
            hits, hit_index, _ = caster.intersects_location(
                origins, np.tile([0.0, 0.0, -1.0], (len(world), 1)), multiple_hits=False
            )
            if len(hits):
                ground = np.full(len(world), -np.inf)
                ground[hit_index] = hits[:, 2]
                found = np.isfinite(ground)
                if found.any():
                    ray_gaps.append(float((world[found, 2] - ground[found]).min()))

    return {"plane": np.asarray(plane_gaps), "ray": np.asarray(ray_gaps)}


def summarise(name: str, gaps: np.ndarray, tolerance: float = 0.001) -> str:
    if not len(gaps):
        return f"{name}: no frames"
    return (
        f"{name} median {np.median(gaps) * 1000:+7.2f} mm  worst {gaps.min() * 1000:+8.2f} mm  "
        f"through {(gaps < -tolerance).mean() * 100:5.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("motion", type=Path, help="retargeted .npz")
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True, help="the SMPL-X motion it came from")
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True, help="SMPL-X body model directory")
    parser.add_argument(
        "--bodies",
        nargs="*",
        default=[],
        metavar="SIDE=BODY",
        help=f"foot bodies to probe (default: {DEFAULT_BODIES})",
    )
    parser.add_argument("--max-points", type=int, default=128)
    args = parser.parse_args()

    bodies = dict(DEFAULT_BODIES)
    for entry in args.bodies:
        side, _, name = entry.partition("=")
        if not name:
            raise SystemExit(f"--bodies expects SIDE=BODY, got {entry!r}")
        bodies[side] = name

    result = measure(
        args.motion, args.scene, args.source, args.urdf, args.model_dir, bodies, args.max_points
    )
    print(args.motion.name)
    print("  " + summarise("A contact plane", result["plane"]))
    print("  " + summarise("B ground ray   ", result["ray"]))


if __name__ == "__main__":
    main()
