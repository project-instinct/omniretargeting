"""Replay prepared source/contact data through the generic sequence solver.

This is a data/diagnostic wrapper. Optimization lives in
omniretargeting.sequence_retargeting and is also available in the normal CLI
with --solver sequence; it requires no trial caches or upstream worktree.
"""

import argparse
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from omniretargeting.contact import SceneContactTask
from omniretargeting.scene_geometry import SceneQuery
from omniretargeting.sequence_retargeting import retarget_sequence
from omniretargeting._sequence.coordinates import Coordinates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--iterations", type=int, default=35)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--source-cache", type=Path)
    parser.add_argument("--seed", type=Path)
    parser.add_argument("--lock-root-xy", action="store_true", help="Run a locked-path ablation")
    args = parser.parse_args()
    root, clip = args.trial_dir.resolve(), args.clip
    out = args.output_dir or root / "general_sequence" / clip
    out.mkdir(parents=True, exist_ok=True)
    prior = root / "continuity_fix" / clip
    z = dict(np.load(prior / f"{clip}_retargeted.npz"))
    diagnostic = json.loads((prior / f"{clip}_retargeted.diagnostics.json").read_text())
    priority = diagnostic["effective_config"]["validation_end_effector_priority"]
    source = None
    candidates = (
        [args.source_cache]
        if args.source_cache
        else sorted(root.glob("*source_cache.pkl"))
    )
    for path in candidates:
        with path.open("rb") as stream:
            candidate = pickle.load(stream)
        if candidate.metadata.get("seq_name") == clip:
            source = candidate
            break
    if source is None:
        raise ValueError(
            f"No original source cache found for {clip}; pass --source-cache"
        )
    profile = json.loads((root / "upstream_sequence/robot.json").read_text())
    model = mujoco.MjModel.from_xml_path(profile["urdf_path"])
    coords = Coordinates(model)
    seed = args.seed or root / "upstream_reference" / clip / "framewise_checkpoint.npz"
    initial = np.load(seed)["qpos"].copy()
    anchor = profile.get("base_orientation", {}).get(
        "root_position", profile.get("base_orientation", {}).get("pelvis")
    )
    if anchor not in source.target_names:
        anchor = next(
            name
            for name, body in profile["joint_mapping"].items()
            if model.body(body).id == model.jnt_bodyid[0]
        )
    root_positions = source.positions[:, source.target_names.index(anchor)]
    if coords.root_q is not None:
        initial[:, coords.root_q : coords.root_q + 2] = root_positions[:, :2]
    tasks = {}
    for i, name in enumerate(z["contact_names"]):
        name = str(name)
        tasks[name] = SceneContactTask(
            name=name,
            tracking_positions=z["end_effector_reference_w"][:, i],
            contact_positions=z["contact_targets_w"][:, i],
            surface_points=z["contact_targets_w"][:, i],
            surface_normals=z["contact_surface_normals_w"][:, i],
            distances=np.full(len(initial), np.inf),
            relative_speeds=np.zeros(len(initial)),
            contact_confidence=z["contact_confidence"][:, i],
            contact_mask=z["contact_mask"][:, i],
            contact_modes=z["contact_modes"][:, i],
            forward_directions=z["end_effector_forward_w"][:, i],
            up_directions=z["end_effector_up_w"][:, i],
            environment_ids=z["contact_environment_ids"][:, i],
        )
    mapping = dict(profile["joint_mapping"])
    shoulders = []
    for spec in priority["effectors"].values():
        arm = spec.get("arm_posture", {})
        name = arm.get("source_shoulder")
        if name in mapping and name not in shoulders:
            shoulders.append(name)
            mapping[name] = arm["shoulder_body"]
    axes = []
    if len(shoulders) == 2:
        axes = [
            dict(
                weights={shoulders[0]: 0.5, shoulders[1]: 0.5, anchor: -1}, sigma=0.04
            ),
            dict(weights={shoulders[0]: 1, shoulders[1]: -1}, sigma=0.06),
        ]
    ground = trimesh.creation.box(extents=[10.0, 10.0, 0.1])
    ground.apply_translation([0, 0, -0.05])
    corners = np.asarray(ground.vertices)[np.isclose(ground.vertices[:, 2], 0)]
    environment = np.concatenate(
        [
            np.broadcast_to(corners, (len(initial), *corners.shape)),
            source.object_points,
        ],
        axis=1,
    )
    history = []

    def checkpoint(qpos, record):
        history.append(record)
        print(json.dumps(record), flush=True)
        np.savez(out / "framewise_checkpoint.npz", qpos=qpos)
        (out / "optimization.json").write_text(json.dumps(history, indent=2))

    result, report = retarget_sequence(
        model,
        initial,
        source.positions,
        source.target_names,
        mapping,
        root_positions,
        tasks,
        priority,
        framerate=source.framerate,
        config=dict(iterations=args.iterations, pose_axes=axes, lock_root_xy=args.lock_root_xy),
        root_orientation_targets=source.root_orientations,
        environment_points=environment,
        scene_query=SceneQuery(ground),
        contact_offset_choice=diagnostic["local_repair"]["contact_offset_choice"],
        callback=checkpoint,
    )
    z.update(qpos=result, joint_pos=result[:, coords.qi])
    if coords.root_q is not None:
        z.update(
            base_pos_w=result[:, coords.root_q : coords.root_q + 3],
            base_quat_w=result[:, coords.root_q + 3 : coords.root_q + 7],
        )
    np.savez(out / f"{clip}_retargeted.npz", **z)
    (out / "sequence_report.json").write_text(json.dumps(report, indent=2))
    print("SEQUENCE_DONE", clip, flush=True)


if __name__ == "__main__":
    main()
