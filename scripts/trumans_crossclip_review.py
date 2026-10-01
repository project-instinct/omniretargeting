"""Run the same cold-start stages and independent audits on a prepared clip.

Clip/action labels never enter optimization. Preparation captures the model's
reference initialization; no previously fitted recording is used as a seed.
The stage iteration budgets are shared across the entire benchmark.
"""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import trimesh
from review_contact_metrics import evaluate as contact_metrics
from review_motion_displacement import measure
from review_scene_metrics import audit as scene_metrics
from review_source_pose import evaluate as source_pose
from trumans_review_inputs import restore_surface

from omniretargeting._sequence.continuity_guard import SequenceContinuityGuard
from omniretargeting._sequence.pose_guard import SequencePoseGuard
from omniretargeting._sequence.proximity import SurfaceProximityTasks
from omniretargeting._sequence.scene import PrescribedSceneClearance
from omniretargeting._sequence.surface_guard import SurfaceSequenceGuard
from omniretargeting.contact import stack_contact_tasks
from omniretargeting.scene_geometry import SceneQuery
from omniretargeting.sequence_retargeting import (
    SEQUENCE_DEFAULTS,
    prepare_sequence,
    retarget_sequence,
    sequence_config,
)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2))


def run(args):
    root = args.prepared.parent
    inputs = np.load(args.prepared, allow_pickle=True).item()
    if inputs.get("initialization") != "model_reference":
        raise ValueError("Cross-clip validation requires a fresh model initialization")
    model = mujoco.MjModel.from_xml_path(inputs["model_path"])
    initial = inputs["initial"].copy()
    objects = np.load(root / "objects.npy", allow_pickle=True).tolist()
    surface = restore_surface(model, inputs["surface"], inputs["surface_config"])
    proximity = (
        None
        if inputs.get("proximity") is None
        else SurfaceProximityTasks(model, **inputs["proximity"])
    )
    config = sequence_config(inputs["captured_solver_config"])
    for key in ("contact_sigma_m", "contact_normal_sigma_m", "contact_tangent_sigma_m"):
        config[key] = SEQUENCE_DEFAULTS[key]
    config.pop("source_side_support", None)
    config["iterations"] = args.iterations
    bootstrap = sequence_config(inputs["captured_solver_config"])
    bootstrap["iterations"] = args.iterations
    geometry = {**config, "iterations": args.geometry_iterations}
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
    }
    _, _, task_set, _ = prepare_sequence(model, initial, *positional, config, **common)
    choices = task_set.choices
    np.savez(root / "contact_choices.npz", **choices)
    common["contact_offset_choice"] = choices
    payload = {
        "source_joints_w": inputs["positions"],
        "framerate": 30.0,
        "terrain_scale": 1.0,
        "frame_start": 0,
        **stack_contact_tasks(inputs["tasks"]),
        **({} if surface is None else surface.export()),
        **{f"contact_choice_{name}": value for name, value in choices.items()},
    }
    save_json(
        root / "stage_configs.json",
        {
            "bootstrap": bootstrap,
            "acquired": config,
            "final": geometry,
            "cold_start": True,
            "source_scale": 1.0,
            "stage_policy": "source pose, continuous contact acquisition, protected geometry",
        },
    )

    def solve(stage, seed, cfg, **extra):
        output = root / f"{stage}.npz"

        def progress(poses, report):
            print(stage, json.dumps(report), flush=True)
            np.savez(root / f"{stage}.partial.npz", qpos=poses)

        tick = time.monotonic()
        if output.exists() and output.with_suffix(".json").exists():
            print("RESUME", stage, flush=True)
            return np.load(output)["qpos"]
        motion, report = retarget_sequence(
            model,
            seed,
            *positional,
            framerate=30.0,
            config=cfg,
            callback=progress,
            **common,
            **extra,
        )
        report["wall_seconds"] = time.monotonic() - tick
        np.savez(output, qpos=motion, **payload)
        save_json(output.with_suffix(".json"), report)
        print("SAVED", stage, report["wall_seconds"], flush=True)
        return motion

    pose = solve("bootstrap", initial, bootstrap)
    acquired = solve(
        "acquired",
        pose,
        config,
        extra_tasks=(() if proximity is None else (proximity,)),
    )
    mesh = trimesh.load_mesh(root / "static_scene.obj", process=False)
    query_report = {"method": "pseudonormal", "fallback_reason": None}
    try:
        scene = SceneQuery(
            mesh,
            enclosed_free_space=inputs["scene_enclosed_free_space"],
            closed_sign_method="pseudonormal",
        )
    except ValueError as error:
        # The accelerated sign query explicitly excludes degenerate triangles.
        # Keep the original mesh and use the existing winding backend instead.
        # Other geometry errors must still fail, not become presumed free space.
        if str(error) != "Pseudonormals require nondegenerate triangles":
            raise
        query_report = {"method": "winding", "fallback_reason": str(error)}
        scene = SceneQuery(
            mesh,
            enclosed_free_space=inputs["scene_enclosed_free_space"],
        )
        print("SCENE_QUERY_FALLBACK", json.dumps(query_report), flush=True)
    save_json(root / "scene_query.json", query_report)
    extras = [task for task in (surface, proximity) if task is not None]
    if objects:
        extras.append(
            PrescribedSceneClearance(
                model,
                objects,
                len(initial),
                clearance=geometry["scene_clearance_m"],
            )
        )
    final = solve(
        "final",
        acquired,
        geometry,
        scene_query=scene,
        extra_tasks=extras,
        endpoint_reference_trajectory=acquired,
    )

    # Recheck nonlinear budgets independently from the optimizer's return status.
    coords, _, task_set, rotations = prepare_sequence(
        model,
        acquired,
        *positional,
        geometry,
        extra_tasks=(() if surface is None else (surface,)),
        **common,
    )
    guards = {
        "endpoints_and_surfaces": SurfaceSequenceGuard(
            task_set,
            acquired,
            geometry["endpoint_regression_m"],
        ),
        "continuity": SequenceContinuityGuard(task_set, acquired, rotations),
        "pose": SequencePoseGuard(task_set, acquired),
    }
    checks = {name: guard.accepts(final) for name, guard in guards.items()}
    saved = np.load(root / "final.npz")
    checks.update(
        finite=bool(np.isfinite(final).all()),
        source_unchanged=bool(
            np.array_equal(saved["source_joints_w"], inputs["positions"])
        ),
        limits=bool(
            np.all(final[:, coords.qi] >= coords.lower - 1e-7)
            and np.all(final[:, coords.qi] <= coords.upper + 1e-7)
        ),
    )
    save_json(root / "acceptance.json", checks)
    save_json(root / "displacement.json", measure(model, acquired, final, 0.02))
    for stage in ("bootstrap", "acquired", "final"):
        motion = root / f"{stage}.npz"
        save_json(
            root / f"{stage}_metrics.json",
            contact_metrics(args.prepared, motion, choices),
        )
        save_json(root / f"{stage}_pose.json", source_pose(args.prepared, motion))
    for stage in ("acquired", "final"):
        scene_metrics(
            SimpleNamespace(
                prepared=args.prepared,
                motion=root / f"{stage}.npz",
                objects=(root / "objects.npy" if objects else None),
                output=root / f"{stage}_scene.json",
                spacing=0.02,
                batch=64,
            )
        )
    save_json(
        root / "completed.json",
        {
            "frames": len(final),
            "seconds": len(final) / 30,
            "checks": checks,
            "all_guard_checks_pass": all(checks.values()),
            "dynamics_verified": False,
            "collision_free_certificate": False,
        },
    )
    print("COMPLETE", root, checks, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--geometry-iterations", type=int, default=2)
    run(parser.parse_args())
