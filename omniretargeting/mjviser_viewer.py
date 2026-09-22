"""mjviser playback for OmniRetargeting outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import trimesh

from omniretargeting._viewer_motion import (
    OMOMO_OBJECT_SCALES as OMOMO_OBJECT_SCALES,
    PARC_MS_BODY_PARENTS as PARC_MS_BODY_PARENTS,
    SMPLX_BODY_PARENTS as SMPLX_BODY_PARENTS,
    SUPPORTED_MOTION_SUFFIXES as SUPPORTED_MOTION_SUFFIXES,
    _add_object as _add_object,
    _build_qpos_from_components as _build_qpos_from_components,
    _continuous_quat as _continuous_quat,
    _free_joint_qpos_adr as _free_joint_qpos_adr,
    _infer_omomo_object_mesh as _infer_omomo_object_mesh,
    _load_motion as _load_motion,
    _model_scalar_joint_names as _model_scalar_joint_names,
    _parc_ms_body_forward as _parc_ms_body_forward,
    _parse_scale as _parse_scale,
    _payload_joint_names as _payload_joint_names,
    _resolve_motion_file as _resolve_motion_file,
    _resolve_object_mesh as _resolve_object_mesh,
    _resolve_path as _resolve_path,
)
from omniretargeting._viewer_scene import (
    _run_viewer as _run_viewer,
)
from omniretargeting.main import DEFAULT_ROBOT_CONFIG_PATH
from omniretargeting.robot_config import load_robot_config


def _model_path_from_args(args: argparse.Namespace) -> str:
    if args.model:
        return str(_resolve_path(args.model))

    robot_config = load_robot_config(args.robot_config)
    urdf_path = robot_config.get("urdf_path")
    if not urdf_path:
        raise ValueError(f"Robot config does not define urdf_path: {args.robot_config}")
    return str(_resolve_path(urdf_path))


def run(args: argparse.Namespace) -> None:
    motion_file = _resolve_motion_file(_resolve_path(args.motion_path))
    terrain_mesh = None if args.no_terrain or args.terrain is None else trimesh.load(_resolve_path(args.terrain), force="mesh")
    model_path = _model_path_from_args(args)
    model = mujoco.MjModel.from_xml_path(model_path)
    qpos, object_pos, object_quat, framerate = _load_motion(
        model, motion_file, args.root_z_offset
    )

    contact_names = None
    contact_mask = None
    contact_targets = None
    contact_normals = None
    contact_forwards = None
    effector_references = None
    effector_ups = None
    source_joints = None
    source_parents = None
    source_label = "source"
    if motion_file.suffix == ".npz":
        with np.load(motion_file, allow_pickle=True) as payload:
            required = {"contact_names", "contact_mask", "contact_targets_w"}
            if required.issubset(payload.files):
                contact_names = [str(name) for name in payload["contact_names"]]
                contact_mask = np.asarray(payload["contact_mask"], dtype=bool)
                contact_targets = np.asarray(payload["contact_targets_w"], dtype=float)
                if "contact_surface_normals_w" in payload:
                    contact_normals = np.asarray(
                        payload["contact_surface_normals_w"], dtype=float
                    )
                if "end_effector_forward_w" in payload:
                    contact_forwards = np.asarray(
                        payload["end_effector_forward_w"], dtype=float
                    )
                if "end_effector_reference_w" in payload:
                    effector_references = np.asarray(
                        payload["end_effector_reference_w"], dtype=float
                    )
                if "end_effector_up_w" in payload:
                    effector_ups = np.asarray(
                        payload["end_effector_up_w"], dtype=float
                    )
            if "source_joints_w" in payload:
                source_joints = np.asarray(payload["source_joints_w"], dtype=float)
            elif "smplx_joints_w" in payload:
                source_joints = np.asarray(payload["smplx_joints_w"], dtype=float)

            source_type = (
                str(np.asarray(payload["source_type"]).item())
                if "source_type" in payload
                else "smplx"
            ).casefold()
            if source_type == "parc_ms":
                source_parents = PARC_MS_BODY_PARENTS
                source_label = "PARC-MS"
            elif source_joints is not None and source_joints.shape[1] == len(
                SMPLX_BODY_PARENTS
            ):
                source_parents = SMPLX_BODY_PARENTS
                source_label = "SMPL-X"

    _run_viewer(
        model,
        qpos,
        object_pos,
        object_quat,
        framerate,
        args,
        motion_file,
        terrain_mesh=terrain_mesh,
        contact_names=contact_names,
        contact_mask=contact_mask,
        contact_targets=contact_targets,
        contact_normals=contact_normals,
        contact_forwards=contact_forwards,
        effector_references=effector_references,
        effector_ups=effector_ups,
        source_joints=source_joints,
        source_parents=source_parents,
        source_label=source_label,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion_path", help="Retargeted .npz/.npy file or directory.")
    parser.add_argument(
        "--robot-config",
        default=DEFAULT_ROBOT_CONFIG_PATH,
        help=f"Robot profile JSON used to resolve the model path (default: {DEFAULT_ROBOT_CONFIG_PATH}).",
    )
    parser.add_argument("--model", default=None, help="MuJoCo XML/URDF path. Overrides --robot-config urdf_path.")
    parser.add_argument("--terrain", default=None, help="Optional terrain mesh to include in the viewer scene.")
    parser.add_argument("--no-terrain", action="store_true", help="Do not load a terrain mesh.")
    parser.add_argument(
        "--show-collision",
        action="store_true",
        help="Also draw collision geoms; by default only visual mesh groups are shown.",
    )
    parser.add_argument("--terrain-target-faces", type=int, default=5000)
    parser.add_argument(
        "--terrain-opacity",
        type=float,
        default=1.0,
        help="Viewer-only terrain opacity; lower values reveal the robot inside room meshes.",
    )
    parser.add_argument("--object-mesh", default=None, help="Optional object mesh for object_pos_w/object_quat_w playback.")
    parser.add_argument(
        "--omomo-object-dir",
        default="~/Datasets/OMOMO/data/captured_objects",
        help="Directory used to infer OMOMO object meshes from motion filenames.",
    )
    parser.add_argument("--object-scale", type=_parse_scale, default=None, help="Object mesh scale: SCALE or X,Y,Z.")
    parser.add_argument("--no-object", action="store_true", help="Disable object mesh loading.")
    parser.add_argument(
        "--no-human",
        action="store_true",
        help="Hide the blue SMPL-X reference skeleton when present in the output.",
    )
    parser.add_argument("--root-z-offset", type=float, default=0.0, help="Visualization-only offset added to root z.")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address; use 127.0.0.1 for local-only playback.")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--refresh-hz", type=float, default=60.0,
                        help="Display refresh target; independent of source fps.")
    parser.add_argument("--shadows", action="store_true", help="Enable dynamic shadows (higher GPU cost).")
    parser.add_argument("--playback-stats", action="store_true", help="Log server submission timing every 5 seconds.")
    parser.add_argument("--playback-dir", type=Path, help="Export a portable browser-cached 3D playback bundle.")
    parser.add_argument("--playback-port", type=int, help="Also serve the cached bundle on this HTTP port.")
    parser.add_argument("--export-only", action="store_true", help="Write --playback-dir and exit.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--track-robot",
        action="store_true",
        help="Keep G1 centered by shifting its scene nodes (off by default for world alignment).",
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--loop", dest="loop", action="store_true", default=True)
    parser.add_argument("--no-loop", dest="loop", action="store_false")
    args = parser.parse_args()
    if (args.playback_port is not None or args.export_only) and args.playback_dir is None:
        parser.error("--playback-port and --export-only require --playback-dir")
    if not np.isfinite(args.refresh_hz) or not 1 <= args.refresh_hz <= 120:
        parser.error("--refresh-hz must be between 1 and 120")
    if args.playback_port == args.port:
        parser.error("--playback-port must differ from the live --port")
    if args.export_only and args.playback_port is not None:
        parser.error("--export-only cannot be combined with --playback-port")
    run(args)


if __name__ == "__main__":
    main()
