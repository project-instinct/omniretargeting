"""Bounded near-pass trial, preserving existing pose and continuity budgets."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import trimesh
from review_contact_metrics import evaluate
from review_contact_passes import scene_query
from review_motion_displacement import measure
from review_scene_metrics import audit
from review_source_pose import evaluate as source_pose
from trumans_review_inputs import restore_surface

from omniretargeting._sequence.continuity_guard import SequenceContinuityGuard
from omniretargeting._sequence.pose_guard import SequencePoseGuard
from omniretargeting._sequence.proximity import SurfaceProximityTasks
from omniretargeting._sequence.scene import PrescribedSceneClearance
from omniretargeting._sequence.surface_guard import SurfaceSequenceGuard
from omniretargeting.contact import stack_contact_tasks
from omniretargeting.sequence_retargeting import prepare_sequence, retarget_sequence


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = np.load(args.prepared, allow_pickle=True).item()
    model = mujoco.MjModel.from_xml_path(inputs["model_path"])
    saved = np.load(args.reference)
    initial = saved["qpos"]
    fps = float(saved["framerate"])
    choices = dict(np.load(args.reference.parent / "contact_choices.npz"))
    config = json.loads((args.reference.parent / "stage_configs.json").read_text())[
        "final"
    ]
    objects = np.load(args.prepared.parent / "objects.npy", allow_pickle=True).tolist()
    surface = restore_surface(model, inputs["surface"], inputs["surface_config"])
    proximity = (
        None
        if inputs["proximity"] is None
        else SurfaceProximityTasks(model, **inputs["proximity"])
    )
    extras = [t for t in (surface, proximity) if t is not None]
    if objects:
        extras.append(
            PrescribedSceneClearance(
                model, objects, len(initial), clearance=config["scene_clearance_m"]
            )
        )
    query = scene_query(
        trimesh.load_mesh(args.prepared.parent / "static_scene.obj", process=False),
        enclosed_free_space=inputs["scene_enclosed_free_space"],
    )
    positional = (
        inputs["positions"],
        inputs["source_names"],
        inputs["mapping"],
        inputs["roots"],
        inputs["tasks"],
        inputs["effectors"],
    )
    common = {
        "environment_points": inputs["environment_points"],
        "root_orientation_targets": inputs["root_orientation_targets"],
        "body_orientation_targets": inputs["body_orientation_targets"],
        "contact_offset_choice": choices,
        "extra_tasks": extras,
        "scene_query": query,
    }

    def progress(poses, report):
        print(json.dumps(report), flush=True)

    result, report = retarget_sequence(
        model,
        initial,
        *positional,
        framerate=fps,
        config=config,
        endpoint_reference_trajectory=initial,
        callback=progress,
        **common,
    )
    output = args.output / "trial.npz"
    np.savez(
        output,
        qpos=result,
        source_joints_w=inputs["positions"],
        framerate=fps,
        frame_start=0,
        terrain_scale=1.0,
        **stack_contact_tasks(inputs["tasks"]),
        **{f"contact_choice_{n}": v for n, v in choices.items()},
    )
    coords, _, tasks, rotations = prepare_sequence(
        model, initial, *positional, config, **common
    )
    checks = {
        "retained_contacts": SurfaceSequenceGuard(
            tasks, initial, config["endpoint_regression_m"]
        ).accepts(result),
        "continuity": SequenceContinuityGuard(tasks, initial, rotations).accepts(
            result
        ),
        "cumulative_pose": SequencePoseGuard(tasks, initial).accepts(result),
        "finite": bool(np.isfinite(result).all()),
        "limits": bool(
            np.all(result[:, coords.qi] >= coords.lower - 1e-7)
            and np.all(result[:, coords.qi] <= coords.upper + 1e-7)
        ),
    }

    def write(name, value):
        (args.output / f"{name}.json").write_text(json.dumps(value, indent=2))

    write("solve", report)
    write("config", config)
    write("checks", checks)
    write("displacement", measure(model, initial, result, 0.02))
    write("metrics", evaluate(args.prepared, output, choices))
    write("source_pose", source_pose(args.prepared, output))
    audit(
        SimpleNamespace(
            prepared=args.prepared,
            motion=output,
            objects=args.prepared.parent / "objects.npy" if objects else None,
            output=args.output / "scene.json",
            spacing=0.02,
            batch=64,
        )
    )
    print("COMPLETE", checks, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
