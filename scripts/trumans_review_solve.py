"""Solve a review interval with one full-timeline contact and material-point plan."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from trumans_review_inputs import restore_surface

from omniretargeting._sequence.proximity import SurfaceProximityTasks
from omniretargeting._sequence.scene import PrescribedSceneClearance
from omniretargeting.contact import stack_contact_tasks
from omniretargeting.contact_windows import slice_contact_task
from omniretargeting.scene_geometry import SceneQuery
from omniretargeting.sequence_retargeting import (
    prepare_sequence,
    retarget_sequence,
    sequence_config,
)


def run(args):
    staged = getattr(args, "staged_contact_repair", False)
    if staged and (args.contact_reference is None or args.correction_reference is None):
        raise ValueError("Staged repair requires --contact-reference and --correction-reference")
    inputs = np.load(args.prepared, allow_pickle=True).item()
    model = mujoco.MjModel.from_xml_path(inputs["model_path"])
    initial = inputs["initial"].copy()
    if args.warm is not None:
        initial = np.load(args.warm)["qpos"].copy()
    stop = len(initial) if args.stop is None else args.stop
    sl = slice(args.start, stop)
    cfg = sequence_config(
        {
            **({} if args.config is None else json.loads(args.config.read_text())),
            "iterations": args.iterations,
        }
    )
    # Acquire one persistent material point per uninterrupted contact, once on
    # the complete clip. Never select a fresh sole point independently per frame.
    choice_file = args.prepared.parent / "contact_choices.npz"
    if not choice_file.exists():
        _, _, tasks, _ = prepare_sequence(
            model,
            initial,
            inputs["positions"],
            inputs["source_names"],
            inputs["mapping"],
            inputs["roots"],
            inputs["tasks"],
            inputs["effectors"],
            cfg,
            root_orientation_targets=inputs["root_orientation_targets"],
        )
        np.savez(choice_file, **tasks.choices)
    choices = np.load(choice_file)
    tasks = {
        name: slice_contact_task(task, args.start, stop)
        for name, task in inputs["tasks"].items()
    }
    surface = inputs["surface"]
    surface = {
        key: (value[sl] if value.ndim > 1 and len(value) == len(initial) else value)
        for key, value in surface.items()
    }
    extra = restore_surface(model, surface, inputs["surface_config"])
    extras = [extra] if extra is not None else []
    if staged and extra is not None:
        from omniretargeting.contact_windows import restore_contact_roles

        restore_contact_roles(extra, tasks, inputs["effectors"], surface)
    if inputs.get("proximity") is not None:
        proximity = inputs["proximity"]
        values = {
            key: (
                value[sl]
                if key in ("environment", "source_vectors", "normals", "valid")
                else value
            )
            for key, value in proximity.items()
        }
        extras.append(SurfaceProximityTasks(model, **values))
    if args.objects is not None:
        objects = np.load(args.objects, allow_pickle=True).tolist()
        objects = [
            {
                **obj,
                "translations": obj["translations"][sl],
                "rotations": obj["rotations"][sl],
            }
            for obj in objects
        ]
        extras.append(
            PrescribedSceneClearance(
                model,
                objects,
                stop - args.start,
                clearance=cfg["scene_clearance_m"],
                query_config=cfg.get("moving_scene"),
            )
        )
    scene = SceneQuery(
        trimesh.load_mesh(args.prepared.parent / "static_scene.obj", process=False),
        enclosed_free_space=inputs.get("scene_enclosed_free_space", False),
        closed_sign_method=args.closed_sign_method,
    )
    orientations = [
        {**entry, "rotations": entry["rotations"][sl]}
        for entry in inputs["body_orientation_targets"]
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def progress(poses, report):
        print(json.dumps(report), flush=True)
        np.savez(args.output.with_suffix(".partial.npz"), qpos=poses)

    solve = retarget_sequence
    if staged:
        from omniretargeting.staged_contact_repair import repair_contact_stages

        solve = repair_contact_stages
    solved, report = solve(
        model,
        initial[sl],
        inputs["positions"][sl],
        inputs["source_names"],
        inputs["mapping"],
        inputs["roots"][sl],
        tasks,
        inputs["effectors"],
        framerate=30.0,
        config=cfg,
        root_orientation_targets=inputs["root_orientation_targets"][sl],
        body_orientation_targets=orientations,
        environment_points=inputs["environment_points"][sl],
        scene_query=scene if args.scene else None,
        contact_offset_choice={name: choices[name][sl] for name in tasks},
        extra_tasks=extras,
        callback=progress,
        endpoint_reference_trajectory=(
            None
            if args.contact_reference is None
            else np.load(args.contact_reference)["qpos"][sl]
        ),
        correction_reference_trajectory=(
            None
            if args.correction_reference is None
            else np.load(args.correction_reference)["qpos"][sl]
        ),
    )
    np.savez(
        args.output,
        qpos=solved,
        source_joints_w=inputs["positions"][sl],
        framerate=30.0,
        terrain_scale=1.0,
        frame_start=args.start,
        **stack_contact_tasks(tasks),
        **({} if extra is None else extra.export()),
        **{f"contact_choice_{name}": choices[name][sl] for name in tasks},
    )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print("SAVED", args.output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warm", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--contact-reference", type=Path)
    parser.add_argument("--correction-reference", type=Path)
    parser.add_argument("--objects", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--scene", action="store_true")
    parser.add_argument("--staged-contact-repair", action="store_true")
    parser.add_argument(
        "--closed-sign-method", choices=("winding", "pseudonormal"), default="winding"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
