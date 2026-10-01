"""Prepare a reproducible full-clip review from original TRUMANS observations."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from omniretargeting import OmniRetargeter
from omniretargeting.contact import stack_contact_tasks
from omniretargeting.data_sources.trumans_review import (
    WORLD_ROTATION,
    load_scene,
    reconstruct_source,
)
from omniretargeting.robot_config import load_robot_config
from omniretargeting.surface_contact import SurfaceContactTasks
from omniretargeting.surface_tasks import SurfaceBinding


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BODY_MODELS = REPO_ROOT / "assets/body_models"
DEFAULT_ROBOT_PROFILE = (
    REPO_ROOT / "robot_models/unitree_g1/unitree_g1_instinctmj.json"
)


class Prepared(Exception):
    """Stop before optimization/postprocessing in a preparation-only run."""


def restore_surface(model, payload, config):
    if not payload:
        return None
    bindings = [
        SurfaceBinding(
            str(part), int(face), tuple(bary), int(geom), tuple(point), tuple(normal)
        )
        for part, face, bary, geom, point, normal in zip(
            payload["surface_contact_parts"],
            payload["surface_contact_face_ids"],
            payload["surface_contact_barycentric"],
            payload["surface_contact_robot_geoms"],
            payload["surface_contact_robot_local_points"],
            payload["surface_contact_robot_local_normals"],
        )
    ]
    task = SurfaceContactTasks(
        model,
        bindings,
        payload["surface_contact_environment_w"],
        payload["surface_contact_source_vectors_w"],
        payload["surface_contact_mask"],
        payload["surface_contact_normals_w"],
        config,
        confidence=payload["surface_contact_confidence"],
        modes=payload["surface_contact_modes"],
        surface_ids=payload["surface_contact_surface_ids"],
        pose_normal_targets=payload.get("surface_contact_pose_normal_targets_w"),
        pose_normal_valid=payload.get("surface_contact_pose_normal_valid"),
        contact_targets=payload.get("surface_contact_targets_w"),
        effector_names=payload.get("surface_contact_effector_names"),
    )
    task.static_surface_id = 0
    return task


def prepare(args):
    from omniretargeting.sequence_retargeting import retarget_sequence as solve_sequence

    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "source.npy"
    if not cache.exists():
        np.save(
            cache,
            reconstruct_source(args.data, args.segment, args.body_models),
            allow_pickle=True,
        )
    source = np.load(cache, allow_pickle=True).item()
    support_path = getattr(args, "support_evidence", None)
    support = (
        None
        if support_path is None
        else np.load(support_path, allow_pickle=True).item()
    )
    indices, scene, objects = load_scene(args.data, args.segment)
    assert np.array_equal(indices, source["indices"])
    np.save(
        args.output / "objects.npy", np.array(objects, dtype=object), allow_pickle=True
    )
    scene_path = args.output / "static_scene.obj"
    scene.export(scene_path)
    profile = load_robot_config(args.robot)
    config = profile["retargeting"]
    config["sequence_solver"] = {"enabled": True}
    # TRUMANS room meshes delimit the navigable interior, with obstacle shells
    # oppositely oriented. This is a geometry convention, not a motion setting.
    config["scene_enclosed_free_space"] = True
    config["surface_pose"] = {"enabled": False}
    config["self_contact"] = {"enabled": False}
    # Endpoint and surface observations are prepared once on the full timeline.
    priority = config["end_effector_priority"]
    priority["semantic_contact_effectors"] = list(priority["effectors"])
    priority["world_tracking_effectors"] = []
    priority["contact_detection"]["surface_clearance"] = 0.0
    baseline = None if args.baseline is None else np.load(args.baseline)
    owner = OmniRetargeter(
        robot_urdf_path=profile["urdf_path"],
        terrain_mesh_path=scene_path,
        joint_mapping=profile["joint_mapping"],
        robot_height=profile.get("robot_height"),
        smplx_joint_names=profile.get("smplx_joint_names"),
        height_estimation=profile.get("height_estimation"),
        base_orientation=profile.get("base_orientation"),
        retargeting=config,
    )
    perform = owner._perform_retargeting
    stage_reports = []

    def warm(*a, **kw):
        kw["reference_trajectory"] = baseline["qpos"].copy()
        return perform(*a, **kw)

    def capture(
        model, initial, positions, names, mapping, roots, tasks, effectors, **kw
    ):
        surface = owner.last_surface_contact_tasks
        saved = {
            "model_path": profile["urdf_path"],
            "initial": initial,
            "positions": positions,
            "source_names": names,
            "mapping": mapping,
            "roots": roots,
            "effectors": effectors,
            "tasks": {n: replace(t, contact_phase=None) for n, t in tasks.items()},
            "root_orientation_targets": kw["root_orientation_targets"],
            "body_orientation_targets": kw["body_orientation_targets"],
            "environment_points": kw["environment_points"],
            "surface": {} if surface is None else surface.export(),
            "surface_config": {} if surface is None else surface.cfg,
            "proximity": (
                None
                if owner.last_surface_proximity_tasks is None
                else owner.last_surface_proximity_tasks.export()
            ),
            "reconstruction_error_m": source["reconstruction_error_m"],
            "captured_solver_config": kw["config"],
            "initialization": "model_reference"
            if baseline is None
            else "saved_robot_motion",
        }
        saved["scene_enclosed_free_space"] = True
        if stage_reports:
            first = np.load(args.output / "prepared.npy", allow_pickle=True).item()
            saved["initial"] = first["initial"]
            saved["captured_solver_config"] = first["captured_solver_config"]
        np.save(args.output / "prepared.npy", saved, allow_pickle=True)
        (args.output / "contact_choices.npz").unlink(missing_ok=True)
        np.savez(
            args.output / "contact_plan.npz",
            **stack_contact_tasks(tasks),
            **saved["surface"],
        )
        diagnostics = {
            "reconstruction_error_m": source["reconstruction_error_m"],
            "source_frames": len(initial),
            "contact_frames": {n: int(t.contact_mask.sum()) for n, t in tasks.items()},
            "object_contact_frames": {
                n: (
                    0
                    if t.environment_ids is None
                    else int(np.count_nonzero(t.contact_mask & (t.environment_ids > 0)))
                )
                for n, t in tasks.items()
            },
            "surface": owner.last_surface_contact_diagnostics,
        }
        (args.output / "preparation.json").write_text(
            json.dumps(diagnostics, indent=2, default=str)
        )
        print(
            "PREPARED",
            diagnostics["contact_frames"],
            diagnostics["object_contact_frames"],
            flush=True,
        )
        if not getattr(args, "solve", False):
            raise Prepared
        stage = len(stage_reports)
        previous_callback = kw.get("callback")

        def progress(poses, record):
            if previous_callback is not None:
                previous_callback(poses, record)
            np.savez(
                args.output / f"stage_{stage}.partial.npz",
                qpos=poses,
                framerate=30.0,
                source_joints_w=positions,
            )

        kw["callback"] = progress
        resume_through = max(
            0 if getattr(args, "resume_source_pose", False) else -1,
            getattr(args, "resume_through_stage", -1),
        )
        if stage <= resume_through:
            saved_path = args.output / f"stage_{stage}.npz"
            report_path = args.output / f"stage_{stage}.json"
            if not saved_path.exists() or not report_path.exists():
                raise FileNotFoundError(f"Completed stage_{stage} files are required")
            with np.load(saved_path) as saved_motion:
                solved = saved_motion["qpos"].copy()
                saved_source = saved_motion["source_joints_w"]
            if (
                solved.shape != initial.shape
                or not np.isfinite(solved).all()
                or not np.array_equal(saved_source, positions)
            ):
                raise ValueError(f"Saved stage_{stage} does not match this model and source timeline")
            report = json.loads(report_path.read_text())
            if (
                report.get("status") != "solved"
                or report.get("frames") != len(positions)
                or report.get("model_nq") != model.nq
            ):
                raise ValueError(f"Saved stage_{stage} was not a completed solve")
            stage_reports.append(report)
            print(f"RESUMED completed stage_{stage}", saved_path, flush=True)
            return solved, report
        solved, report = solve_sequence(
            model, initial, positions, names, mapping, roots, tasks, effectors, **kw
        )
        stage_reports.append(report)
        np.savez(
            args.output / f"stage_{stage}.npz",
            qpos=solved,
            framerate=30.0,
            source_joints_w=positions,
        )
        (args.output / f"stage_{stage}.json").write_text(
            json.dumps(report, indent=2, default=str)
        )
        return solved, report

    if baseline is not None:
        owner._perform_retargeting = warm
    with patch("omniretargeting._core.sequence.retarget_sequence", capture):
        try:
            scale, motion = owner.retarget_motion(
                source["positions"],
                source_joint_orientations=source["orientations"],
                source_reference_joint_positions=source["surface"]["full_body"][
                    "reference_joint_positions"
                ],
                source_reference_joint_orientations=np.tile(
                    Rotation.from_matrix(WORLD_ROTATION).as_quat(scalar_first=True),
                    (source["positions"].shape[1], 1),
                ),
                end_effector_markers=source["markers"],
                body_surface_samples=source["surface"],
                scene_objects=objects,
                source_support_evidence=support,
                framerate=30.0,
                visualize_trajectory=False,
                enable_terrain_scaling=False,
            )
            surface = owner.last_surface_contact_tasks
            np.savez(
                args.output / "final.npz",
                qpos=motion,
                framerate=30.0,
                terrain_scale=scale,
                source_joints_w=source["positions"],
                frame_start=0,
                **stack_contact_tasks(owner.last_end_effector_tasks),
                **({} if surface is None else surface.export()),
                **{
                    f"contact_choice_{name}": values
                    for name, values in owner.last_sequence_report.get(
                        "contact_offset_choice", {}
                    ).items()
                },
            )
            (args.output / "final.json").write_text(
                json.dumps(
                    {
                        "stages": stage_reports,
                        "main_pipeline": True,
                        "support_evidence": None
                        if support is None
                        else support.diagnostics,
                        "sequence_acceptance": getattr(
                            owner, "last_sequence_acceptance", None
                        ),
                        "effective_config": owner.retargeting_config,
                    },
                    indent=2,
                    default=str,
                )
            )
            print("SAVED main pipeline final", args.output / "final.npz", flush=True)
        except Prepared:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--segment", required=True)
    parser.add_argument(
        "--body-models", type=Path, default=DEFAULT_BODY_MODELS,
        help=f"SMPL-X model directory (default: {DEFAULT_BODY_MODELS})",
    )
    parser.add_argument(
        "--robot", type=Path, default=DEFAULT_ROBOT_PROFILE,
        help=f"Robot profile (default: {DEFAULT_ROBOT_PROFILE})",
    )
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--support-evidence", type=Path)
    parser.add_argument(
        "--resume-source-pose", action="store_true",
        help="Reuse a completed stage_0 result in the output directory",
    )
    parser.add_argument(
        "--resume-through-stage", type=int, choices=(0, 1, 2), default=-1,
        help="Reuse completed, source-matched stages through this index",
    )
    parser.add_argument(
        "--solve",
        action="store_true",
        help="Run the real core stages and bounded postprocessing",
    )
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
