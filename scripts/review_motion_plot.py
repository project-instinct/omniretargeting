"""Plot complete trajectories and contact residuals against immutable observations."""

import argparse
from pathlib import Path

import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepared = np.load(args.prepared, allow_pickle=True).item()
    before, after = np.load(args.before), np.load(args.after)
    trajectories = [before["qpos"], after["qpos"]]
    if any(len(q) != len(prepared["roots"]) for q in trajectories):
        raise ValueError("Compare complete, aligned trajectories")
    fps = float(after["framerate"])
    time = np.arange(len(trajectories[0])) / fps
    surface = prepared.get("surface", {})
    dense_names = (
        sorted(
            set(
                np.asarray(surface["surface_contact_parts"])[
                    surface["surface_contact_mask"].any(0)
                ]
            )
        )
        if surface
        else []
    )
    endpoints = [
        name for name, task in prepared["tasks"].items()
        if not task.managed_frames.all()
    ]
    names = dense_names + endpoints
    fig, axes = plt.subplots(
        5 + len(names), 1, figsize=(14, 2 * (5 + len(names))), sharex=True
    )
    model = mujoco.MjModel.from_xml_path(prepared["model_path"])
    data = mujoco.MjData(model)
    from omniretargeting._refinement.math import _lock_contact_offsets

    specs = {
        name: dict(
            prepared["effectors"]["effectors"][name],
            body_id=model.body(
                prepared["effectors"]["effectors"][name]["robot_body"]
            ).id,
            task=prepared["tasks"][name],
        )
        for name in endpoints
    }
    choices = _lock_contact_offsets(model, prepared["initial"], specs) if specs else {}
    anchor_name = prepared["effectors"].get("source_root_robot_body")
    for q, label, color in zip(
        trajectories, ["Before", "After"], ["#a75040", "#236db2"]
    ):
        anchors = q[:, :3].copy()
        if anchor_name is not None:
            for f, pose in enumerate(q):
                data.qpos[:] = pose
                mujoco.mj_forward(model, data)
                anchors[f] = data.xpos[model.body(anchor_name).id]
        offset = anchors - prepared["roots"]
        for k in range(3):
            axes[k].plot(
                time, offset[:, k] * 100, color=color, label=label, linewidth=0.8
            )
            axes[k].set_ylabel(f"Mapped root {'XYZ'[k]} offset (cm)")
        acceleration = np.linalg.norm(np.diff(offset, n=2, axis=0), axis=1)
        axes[3].plot(
            time[2:], acceleration * 100, color=color, label=label, linewidth=0.8
        )
        axes[3].set_ylabel("Root correction\nsecond difference (cm)")
        joint = np.sqrt(np.mean(np.diff(q[:, 7:], n=2, axis=0) ** 2, axis=1))
        axes[4].plot(time[2:], joint, color=color, label=label, linewidth=0.8)
        axes[4].set_ylabel("Joint second difference\nRMS (rad)")
        errors = np.full((len(q), len(names)), np.nan)
        for f, pose in enumerate(q):
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            for k, name in enumerate(names):
                if k < len(dense_names):
                    active = surface["surface_contact_mask"][f] & (
                        np.asarray(surface["surface_contact_parts"]) == name
                    )
                    residuals = []
                    for index in np.flatnonzero(active):
                        gid = int(surface["surface_contact_robot_geoms"][index])
                        point = (
                            data.geom_xpos[gid]
                            + data.geom_xmat[gid].reshape(3, 3)
                            @ surface["surface_contact_robot_local_points"][index]
                        )
                        residuals.append(
                            np.linalg.norm(
                                point - surface["surface_contact_targets_w"][f, index]
                            )
                        )
                    if residuals:
                        errors[f, k] = max(residuals)
                    continue
                task = prepared["tasks"][name]
                if not task.contact_mask[f]:
                    continue
                spec = prepared["effectors"]["effectors"][name]
                body = model.body(spec["robot_body"]).id
                point = (
                    data.xpos[body]
                    + data.xmat[body].reshape(3, 3)
                    @ spec["contact_offsets"][choices[name][f]]
                )
                errors[f, k] = np.linalg.norm(point - task.contact_positions[f])
        for k, name in enumerate(names):
            axes[5 + k].plot(
                time, errors[:, k] * 100, color=color, label=label, linewidth=0.8
            )
            axes[5 + k].set_ylabel(f"{name}\nmax contact error (cm)")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Source time (s); unchanged sampling and contact timing")
    fig.suptitle("Whole-clip comparison using one shared contact plan", fontsize=15)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)


if __name__ == "__main__":
    main()
