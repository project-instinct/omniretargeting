#!/usr/bin/env python3
"""Verify that limb-scaled hand targets use the solved live shoulders."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mujoco
import numpy as np


SOURCE_SHOULDER_INDEX = {"left_hand": 16, "right_hand": 17}


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Retargeted *_retargeted.npz")
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=Path("robot_models/unitree_g1/unitree_g1_instinctmj.json"),
    )
    args = parser.parse_args()

    output = np.load(args.output, allow_pickle=True)
    required = {
        "qpos",
        "smplx_joints_w",
        "contact_names",
        "end_effector_reference_w",
        "swing_limb_offsets_w",
        "swing_anchor_body",
        "swing_anchor_offset",
        "swing_morphology_scale",
    }
    missing = sorted(required.difference(output.files))
    if missing:
        raise KeyError(
            f"{args.output} predates live limb-anchor export; missing {missing}"
        )

    qpos = np.asarray(output["qpos"], dtype=float)
    source_joints = np.asarray(output["smplx_joints_w"], dtype=float)
    source_markers = np.asarray(output["end_effector_reference_w"], dtype=float)
    limb_offsets = np.asarray(output["swing_limb_offsets_w"], dtype=float)
    names = [str(value) for value in output["contact_names"]]
    anchor_bodies = [str(value) for value in output["swing_anchor_body"]]
    anchor_offsets = np.asarray(output["swing_anchor_offset"], dtype=float)
    scales = np.asarray(output["swing_morphology_scale"], dtype=float)

    model = _profile_model(args.robot_config)
    data = mujoco.MjData(model)
    checked = 0
    for effector_name, shoulder_index in SOURCE_SHOULDER_INDEX.items():
        if effector_name not in names:
            continue
        effector_index = names.index(effector_name)
        anchor_name = anchor_bodies[effector_index]
        if not anchor_name:
            raise ValueError(
                f"{effector_name} has no live swing anchor in {args.output}"
            )
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, anchor_name
        )
        if body_id < 0:
            raise ValueError(f"Robot anchor body not found: {anchor_name}")

        live_anchors = np.zeros((len(qpos), 3), dtype=float)
        for frame_idx, frame in enumerate(qpos):
            data.qpos[:] = frame
            mujoco.mj_forward(model, data)
            rotation = data.xmat[body_id].reshape(3, 3)
            live_anchors[frame_idx] = (
                data.xpos[body_id]
                + rotation @ anchor_offsets[effector_index]
            )

        intended_offsets = scales[effector_index] * (
            source_markers[:, effector_index]
            - source_joints[:, shoulder_index]
        )
        exported_offsets = limb_offsets[:, effector_index]
        live_swing_targets = live_anchors + exported_offsets
        anchor_errors = np.linalg.norm(
            (live_swing_targets - intended_offsets) - live_anchors,
            axis=1,
        )
        offset_errors = np.linalg.norm(
            exported_offsets - intended_offsets, axis=1
        )
        print(
            f"{effector_name}: scale={scales[effector_index]:.6f} "
            f"shoulder_anchor_error_m mean={anchor_errors.mean():.9f} "
            f"p95={np.percentile(anchor_errors, 95):.9f} "
            f"max={anchor_errors.max():.9f} "
            f"source_offset_error_max={offset_errors.max():.9f}"
        )
        checked += 1

    if checked != len(SOURCE_SHOULDER_INDEX):
        raise RuntimeError(
            f"Expected both hand anchors, checked {checked}/{len(SOURCE_SHOULDER_INDEX)}"
        )


if __name__ == "__main__":
    main()
