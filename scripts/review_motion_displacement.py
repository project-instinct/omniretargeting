"""Report metric displacement of unchanged robot material samples before/after."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from omniretargeting.scene_geometry import collision_geom_ids, geom_surface_offsets


def measure(model, before, after, spacing):
    if before.shape != after.shape or before.shape[1] != model.nq:
        raise ValueError(
            "Displacement comparisons need one model and an aligned timeline"
        )
    a, b = mujoco.MjData(model), mujoco.MjData(model)
    geoms = collision_geom_ids(model)
    samples = {int(g): geom_surface_offsets(model, int(g), spacing) for g in geoms}
    bodies = np.unique(model.geom_bodyid[geoms])
    orientation = {"radians": 0.0, "frame": None, "body": None}
    result = {
        k: {"metres": 0.0, "frame": None, "body": None}
        for k in ("horizontal", "vertical", "vector")
    }
    for f, (first, second) in enumerate(zip(before, after)):
        for data, pose in ((a, first), (b, second)):
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
        first_rotations = a.xmat[bodies].reshape(-1, 3, 3)
        second_rotations = b.xmat[bodies].reshape(-1, 3, 3)
        angles = Rotation.from_matrix(
            second_rotations @ np.swapaxes(first_rotations, 1, 2)
        ).magnitude()
        index = int(np.argmax(angles))
        if angles[index] > orientation["radians"]:
            orientation = {
                "radians": float(angles[index]),
                "frame": f,
                "body": model.body(int(bodies[index])).name,
            }
        for g, offsets in samples.items():
            first_points = a.geom_xpos[g] + offsets @ a.geom_xmat[g].reshape(3, 3).T
            second_points = b.geom_xpos[g] + offsets @ b.geom_xmat[g].reshape(3, 3).T
            delta = second_points - first_points
            for name, values in (
                ("horizontal", np.linalg.norm(delta[:, :2], axis=1)),
                ("vertical", abs(delta[:, 2])),
                ("vector", np.linalg.norm(delta, axis=1)),
            ):
                maximum = float(values.max(initial=0))
                if maximum > result[name]["metres"]:
                    result[name] = {
                        "metres": maximum,
                        "frame": f,
                        "body": model.body(int(model.geom_bodyid[g])).name,
                    }
    roots = {}
    for j in np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
        q = model.jnt_qposadr[j]
        delta = after[:, q : q + 3] - before[:, q : q + 3]
        roots[model.body(int(model.jnt_bodyid[j])).name] = {
            "horizontal_max_m": float(np.linalg.norm(delta[:, :2], axis=1).max()),
            "vertical_max_m": float(abs(delta[:, 2]).max()),
            "vertical_signed_mean_m": float(delta[:, 2].mean()),
            "translation_max_m": float(np.linalg.norm(delta, axis=1).max()),
        }
    hinges = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    joint_delta = abs(
        after[:, model.jnt_qposadr[hinges]] - before[:, model.jnt_qposadr[hinges]]
    )
    joint = None
    if joint_delta.size:
        f, j = np.unravel_index(joint_delta.argmax(), joint_delta.shape)
        joint = {
            "radians": float(joint_delta[f, j]),
            "frame": int(f),
            "joint": model.joint(int(hinges[j])).name,
        }
    return {
        "sample_spacing_m": spacing,
        "surface_displacement": result,
        "roots": roots,
        "maximum_body_orientation_change": orientation,
        "maximum_joint_change": joint,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacing", type=float, default=0.02)
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(args.model)
    report = measure(
        model, np.load(args.before)["qpos"], np.load(args.after)["qpos"], args.spacing
    )
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
