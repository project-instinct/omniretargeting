import argparse
import cProfile
from contextlib import contextmanager
import numpy as np
import trimesh
from pathlib import Path
import tempfile
import os
import json
import yaml

from omniretargeting import OmniRetargeter
from omniretargeting.robot_config import load_robot_config
from omniretargeting.data_sources.registry import create_data_source
from omniretargeting.utils import normalize_retargeted_output_path
from omniretargeting.utils import create_flat_terrain
from omniretargeting.visualizer import (
    build_object_tracks,
    save_trajectory_video,
    visualize_trajectory,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOT_CONFIG_PATH = REPO_ROOT / "robot_models" / "unitree_g1" / "unitree_g1.json"


@contextmanager
def cprofile_context(profile_path: str | None):
    """Profile the enclosed CLI work when an output path is supplied."""
    if profile_path is None:
        yield
        return

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        yield
    finally:
        profiler.disable()
        output_path = Path(profile_path).expanduser()
        profiler.dump_stats(str(output_path))
        print(f"Saved cProfile statistics to {output_path}")


def load_source_config(yaml_path: Path) -> dict:
    """Load source configuration from YAML file."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Source config file not found: {yaml_path}")
    
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    
    if not isinstance(config, dict):
        raise ValueError(f"Source config must be a YAML object/dict, got {type(config)}")
    
    if "type" not in config:
        raise ValueError("Source config must specify 'type' field (e.g., 'omomo', 'smplx')")
    if "motion" not in config:
        raise ValueError("Source config must specify 'motion' field (path to motion file)")
    
    return config


def select_robot_source(robot_config: dict, source_type: str) -> dict:
    matches = [
        source
        for source in robot_config.get("source", [])
        if source.get("name") == source_type or source.get("type") == source_type
    ]
    if len(matches) != 1:
        raise ValueError(f"Robot profile must contain exactly one source entry for {source_type!r}.")
    return matches[0]


def export_scaled_objects(
    motion_data,
    scaled_objects_dir: Path,
    source_to_robot_scale: float,
    apply_scene_scaling: bool,
):
    if motion_data is None or getattr(motion_data, "object_mesh", None) is None:
        return None

    scaled_objects_dir = Path(scaled_objects_dir)
    scaled_objects_dir.mkdir(parents=True, exist_ok=True)

    object_name = motion_data.metadata.get("object_name", "object")
    centroid_local = motion_data.metadata.get("object_centroid_local")
    if centroid_local is None:
        centroid_local = np.asarray(motion_data.object_mesh.vertices, dtype=float).mean(axis=0)

    scene_scale = float(source_to_robot_scale) if apply_scene_scaling else 1.0

    # Save centered object mesh so per-frame transforms carry the motion explicitly.
    scaled_mesh = motion_data.object_mesh.copy()
    scaled_mesh.apply_translation(-centroid_local)
    if apply_scene_scaling:
        scaled_mesh.apply_scale(scene_scale)
    mesh_path = scaled_objects_dir / f"{object_name}.obj"
    scaled_mesh.export(mesh_path)
    print(f"Saved scaled object mesh to {mesh_path}")

    translations = motion_data.metadata.get("object_translations")
    rotations = motion_data.metadata.get("object_rotations")
    scales = motion_data.metadata.get("object_scales")
    if translations is None or rotations is None or scales is None:
        return mesh_path, None

    poses = []
    for t in range(len(translations)):
        poses.append(
            {
                "frame": t,
                "translation": (np.asarray(translations[t], dtype=float) * scene_scale).tolist(),
                "rotation_matrix": np.asarray(rotations[t]).tolist(),
                "scale": float(scales[t]) * scene_scale,
            }
        )

    pose_path = scaled_objects_dir / f"{object_name}_poses.json"
    with open(pose_path, "w") as f:
        json.dump(poses, f, indent=2)
    print(f"Saved object pose trajectory to {pose_path}")

    return mesh_path, pose_path


def _run(args, parser: argparse.ArgumentParser):
    if args.scale_factor is not None and (not np.isfinite(args.scale_factor) or args.scale_factor <= 0):
        parser.error("--scale-factor must be a finite positive number")

    args.output = normalize_retargeted_output_path(args.output)

    # Load robot profile config (default profile path can be overridden by --robot-config).
    robot_config = {}
    if args.robot_config:
        robot_config_path = Path(args.robot_config).expanduser()
        if robot_config_path.exists():
            robot_config = load_robot_config(robot_config_path)
            profile_name = robot_config.get("name", robot_config_path.stem)
            print(f"Loaded robot config profile: {profile_name}")
        elif robot_config_path == DEFAULT_ROBOT_CONFIG_PATH:
            print(f"Default robot config not found at {DEFAULT_ROBOT_CONFIG_PATH}, continuing without profile.")
        else:
            raise FileNotFoundError(f"Robot config not found: {args.robot_config}")


    robot_urdf_path = robot_config.get("urdf_path")
    if not robot_urdf_path:
        raise ValueError(
            "Robot URDF is required. Set 'urdf_path' in the robot profile JSON (--robot-config)."
        )

    print(f"Loading source config from {args.source_config}...")
    source_config_dict = load_source_config(args.source_config)
    source_type = source_config_dict["type"]
    source_motion_path = source_config_dict["motion"]
    runtime_source_options = {
        key: value
        for key, value in source_config_dict.items()
        if key not in ["type", "motion"]
    }
    selected_source = select_robot_source(robot_config, source_type)
    data_source_source_config = dict(selected_source)
    print(f"Source type: {source_type}")
    print(f"Motion file: {source_motion_path}")

    joint_mapping = selected_source.get("target_mapping")

    if not isinstance(joint_mapping, dict) or not joint_mapping:
        raise ValueError("Joint mapping must be a non-empty JSON object.")

    robot_height = robot_config.get("robot_height")
    retargeting = robot_config.get("retargeting")

    # Handle terrain
    temp_terrain_paths = []
    if "terrain" in runtime_source_options:
        terrain_path = runtime_source_options.pop("terrain")
        print(f"Using terrain from source config: {terrain_path}")
    else:
        print("No terrain provided, creating default flat terrain.")
        flat_terrain = create_flat_terrain()
        fd, temp_terrain_path = tempfile.mkstemp(suffix=".obj")
        os.close(fd)
        flat_terrain.export(temp_terrain_path)
        temp_terrain_paths.append(temp_terrain_path)
        terrain_path = temp_terrain_path

    try:
        print(f"Loading {source_type} motion from {source_motion_path}...")
        data_source = create_data_source(
            source_type=source_type,
            motion_file=source_motion_path,
            source_config=data_source_source_config,
            runtime_options=runtime_source_options,
        )
        motion_data = data_source.load()
        source_positions = motion_data.positions
        source_orientations = motion_data.metadata.get("joint_orientations")
        framerate = args.framerate or motion_data.framerate
        if framerate is None:
            framerate = 30.0
            print(f"Using default framerate: {framerate}")
        else:
            print(f"Using framerate: {framerate}")
        motion_data.framerate = framerate

        if args.output_framerate is not None:
            print(f"Resampling from {framerate}fps to {args.output_framerate}fps...")
            motion_data = motion_data.resample(args.output_framerate)
            source_positions = motion_data.positions
            source_orientations = motion_data.metadata.get("joint_orientations")
            framerate = args.output_framerate
            print(f"Resampled: {motion_data.positions.shape[0]} frames at {framerate}fps")

        print(f"Loaded trajectory with shape: {source_positions.shape}")
        if source_orientations is not None:
            print(f"Loaded orientations with shape: {source_orientations.shape}")
        else:
            print("Warning: Orientations not available for this file format.")

        if args.scale_factor is not None:
            motion_data.positions = motion_data.positions * args.scale_factor
            if motion_data.root_translations is not None:
                motion_data.root_translations = motion_data.root_translations * args.scale_factor
            if motion_data.object_points is not None:
                motion_data.object_points = motion_data.object_points * args.scale_factor
            if motion_data.source_height is not None:
                motion_data.source_height *= args.scale_factor
            if motion_data.human_height is not None:
                motion_data.human_height *= args.scale_factor
            source_positions = motion_data.positions

            scaled_terrain = trimesh.load(terrain_path, force="mesh")
            scaled_terrain.apply_scale(args.scale_factor)
            fd, scaled_terrain_path = tempfile.mkstemp(suffix=".obj")
            os.close(fd)
            scaled_terrain.export(scaled_terrain_path)
            temp_terrain_paths.append(scaled_terrain_path)
            terrain_path = scaled_terrain_path

        # Initialize Retargeter
        print("Initializing OmniRetargeter...")
        retargeter = OmniRetargeter(
            robot_urdf_path=robot_urdf_path,
            terrain_mesh_path=terrain_path,
            joint_mapping=joint_mapping,
            robot_height=robot_height,
            source_target_names=motion_data.target_names,
            base_orientation=selected_source.get("base_orientation", robot_config.get("base_orientation")),
            retargeting=retargeting,
        )

        # Perform retargeting
        print("Retargeting motion...")
        source_to_robot_scale, retargeted_motion = retargeter.retarget_motion(
            motion_data,
            framerate=framerate,
            visualize_trajectory=args.vis,
            enable_scene_scaling=args.enable_scene_scaling,
            show_progress=args.progress,
        )
        if args.scale_factor is not None:
            source_to_robot_scale = args.scale_factor

        if args.enable_scene_scaling:
            scene_output_dir = Path(args.output).with_suffix("")
            scene_output_dir.mkdir(parents=True, exist_ok=True)

            scaled_terrain = trimesh.load(terrain_path, force="mesh")
            scaled_terrain.apply_scale(source_to_robot_scale)
            output_scaled_terrain_path = scene_output_dir / "scaled_terrain.obj"
            scaled_terrain.export(output_scaled_terrain_path)
            print(f"Saved scaled terrain mesh to {output_scaled_terrain_path}")

            if motion_data.object_mesh is not None:
                export_scaled_objects(
                    motion_data,
                    scene_output_dir,
                    source_to_robot_scale,
                    apply_scene_scaling=True,
                )

        # Save output
        print(f"Saving output to {args.output}...")
        
        # Extract data for saving
        # retargeted_motion shape: (T, 7 + DOF) -> [pos(3), quat(4), joints(DOF)]
        
        # Get joint names from robot model
        joint_names = retargeter.get_joint_names()
        
        # Extract components
        base_pos = retargeted_motion[:, :3]
        base_quat = retargeted_motion[:, 3:7]  # wxyz (MuJoCo convention, consistent with entire pipeline)
        joint_pos = retargeted_motion[:, 7:]

        # Save as .npz with specific keys
        np.savez(
            args.output,
            framerate=framerate,
            joint_names=np.array(joint_names),
            joint_pos=joint_pos,
            base_pos_w=base_pos,
            base_quat_w=base_quat,  # wxyz quaternion
        )
        
        print(f"Done! Source-to-robot scale used: {source_to_robot_scale}")

        # Load terrain for visualization/video if needed
        vis_terrain = None
        if (args.vis or args.save_video) and terrain_path and os.path.exists(terrain_path):
            try:
                vis_terrain = trimesh.load(terrain_path, force='mesh')
                if args.enable_scene_scaling:
                    vis_terrain.apply_scale(source_to_robot_scale)
            except Exception as e:
                print(f"Could not load terrain for visualization: {e}")

        # Extract per-frame object tracks for visualization if available
        vis_object_meshes = None
        if args.vis or args.save_video:
            vis_object_meshes = build_object_tracks(
                motion_data,
                source_to_robot_scale=source_to_robot_scale,
                apply_scene_scaling=args.enable_scene_scaling or args.scale_factor is not None,
            )
            if vis_object_meshes:
                print(f"Loaded object track for visualization: {vis_object_meshes[0].name}")

        scaled_source_positions = source_positions * source_to_robot_scale if args.enable_scene_scaling else source_positions

        if args.save_video:
            save_trajectory_video(
                robot_urdf_path,
                retargeted_motion,
                args.save_video,
                source_trajectory=scaled_source_positions,
                terrain_mesh=vis_terrain,
                object_tracks=vis_object_meshes,
                source_target_names=motion_data.target_names,
                target_mapping=joint_mapping,
                fps=framerate,
            )

        if args.vis:
            visualize_trajectory(
                robot_urdf_path,
                retargeted_motion,
                scaled_source_positions,
                terrain_mesh=vis_terrain,
                object_tracks=vis_object_meshes,
                source_target_names=motion_data.target_names,
                target_mapping=joint_mapping,
                fps=framerate,
            )

    finally:
        for temp_terrain_path in temp_terrain_paths:
            if os.path.exists(temp_terrain_path):
                os.remove(temp_terrain_path)


def main():
    parser = argparse.ArgumentParser(description="OmniRetargeting CLI")
    parser.add_argument(
        "--robot-config",
        default=DEFAULT_ROBOT_CONFIG_PATH,
        help=f"Path to robot configuration JSON file (default: {DEFAULT_ROBOT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--source-config",
        required=True,
        help="Path to YAML source configuration file (see config_templates/ for examples)",
    )
    parser.add_argument("--output", required=True, help="Path to save output motion (.npy)")
    scaling_group = parser.add_mutually_exclusive_group()
    scaling_group.add_argument(
        "--enable-scene-scaling",
        action="store_true",
        help="Scale the source motion, terrain, and objects, and export the scaled scene beside the output motion.",
    )
    scaling_group.add_argument(
        "--scale-factor",
        type=float,
        default=None,
        help="Scale the source motion, terrain, and objects by this factor without exporting a scaled scene.",
    )
    parser.add_argument("--vis", action="store_true", help="Visualize the retargeted motion")
    parser.add_argument("--save-video", dest="save_video", default=None, help="Save retargeted motion video to file (e.g. /tmp/out.mp4). Uses offscreen rendering (set MUJOCO_GL=egl for headless).")
    parser.add_argument("--framerate", type=float, default=None, help="Framerate of the motion (optional, defaults to 30.0 or auto-detected)")
    parser.add_argument("--output-framerate", dest="output_framerate", type=float, default=None,
                        help="Resample motion to this framerate before retargeting (e.g. 30 to downsample 120fps data)")
    parser.add_argument("--progress", action="store_true",
                        help="Show a progress bar while retargeting frames")
    parser.add_argument(
        "--cprofile",
        metavar="PROFILE_FILE_PATH",
        help="Write cProfile statistics for the complete retargeting run to this file.",
    )

    args = parser.parse_args()
    with cprofile_context(args.cprofile):
        _run(args, parser)


if __name__ == "__main__":
    main()
