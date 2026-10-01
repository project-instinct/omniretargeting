"""Replay bounded, fixed-root joint repair against the original review scene."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import trimesh
from trumans_review_inputs import restore_surface

from omniretargeting.scene_geometry import SceneQuery
from omniretargeting.whole_body_correction import correct_whole_body_trajectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    inputs = np.load(args.prepared, allow_pickle=True).item()
    payload = dict(np.load(args.motion))
    model = mujoco.MjModel.from_xml_path(inputs["model_path"])
    objects = np.load(args.objects, allow_pickle=True).tolist()
    scene = trimesh.load_mesh(args.prepared.parent / "static_scene.obj", process=False)
    query = SceneQuery(
        scene,
        enclosed_free_space=inputs.get("scene_enclosed_free_space", False),
        full_geometry=False,
        closed_sign_method="pseudonormal",
    )
    surface = restore_surface(model, inputs["surface"], inputs["surface_config"])
    choices = dict(np.load(args.prepared.parent / "contact_choices.npz"))
    config = {} if args.config is None else json.loads(args.config.read_text())
    print(
        "Starting fixed-root repair; sampled original scene and moving meshes",
        flush=True,
    )
    poses, report = correct_whole_body_trajectory(
        model,
        payload["qpos"],
        inputs["tasks"],
        inputs["effectors"],
        config,
        scene_query=query,
        scene_objects=objects,
        surface_contact_tasks=surface,
        framerate=float(payload["framerate"]),
        validation_config={"full_geometry": False, "max_query_points": None},
        contact_offset_choice=choices,
    )
    report["collision_free_certificate"] = False
    report["static_query"] = "Original triangles; feature pseudonormals; sampled shell"
    np.savez(args.output, **{**payload, "qpos": poses})
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print("SAVED", args.output, report["status"], flush=True)


if __name__ == "__main__":
    main()
