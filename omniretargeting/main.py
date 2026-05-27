import argparse
import warnings
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
from omniretargeting.visualizer import (
    build_object_tracks,
    create_flat_terrain,
    save_trajectory_video,
    visualize_trajectory,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOT_CONFIG_PATH = REPO_ROOT / "robot_models" / "unitree_g1" / "unitree_g1.json"


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


def main():
    parser = argparse.ArgumentParser(description="OmniRetargeting CLI")
    parser.add_argument(
        "--robot-config",
        default=DEFAULT_ROBOT_CONFIG_PATH,
        help=f"Path to robot configuration JSON file (default: {DEFAULT_ROBOT_CONFIG_PATH})",
    )
    parser.add_argument("--source-config", default=None, help="Path to YAML source configuration file (see config_templates/ for examples)")
    parser.add_argument("--source", default=None, help="Legacy source entry name or source type from the robot profile (default: first source entry)")
    parser.add_argument("--motion", default=None, help="Legacy path to source motion file")
    parser.add_argument("--source-options", default=None, help="Legacy JSON object with adapter-specific source options")
    parser.add_argument("--model-dir", default=None, help="Legacy adapter model directory, when required by the source type")
    parser.add_argument("--smplx_model_dir", default=None, help=argparse.SUPPRESS) # Legacy flag for backward compatibility with old source configs that expect 'model_directory' under the root of the config dict.
    parser.add_argument("--smplx_motion", default=None, help=argparse.SUPPRESS) # Legacy flag for backward compatibility with old source configs that expect 'motion' under the root of the config dict.
    parser.add_argument("--scaled-objects", default=None, help="Directory to save scaled object meshes and pose trajectories (optional)")
    parser.add_argument("--output", required=True, help="Path to save output motion (.npy)")
    parser.add_argument("--terrain", help="Legacy path to terrain mesh file (optional, defaults to flat ground)")
    parser.add_argument(
        "--output-scaled-terrain",
        dest="output_scaled_terrain",
        default=None,
        help="Path to save the scaled terrain mesh. When provided, terrain scaling is enabled.",
    )
    parser.add_argument("--vis", action="store_true", help="Visualize the retargeted motion")
    parser.add_argument("--save-video", dest="save_video", default=None, help="Save retargeted motion video to file (e.g. /tmp/out.mp4). Uses offscreen rendering (set MUJOCO_GL=egl for headless).")
    parser.add_argument("--framerate", type=float, default=None, help="Framerate of the motion (optional, defaults to 30.0 or auto-detected)")
    parser.add_argument("--replace-cylinders-with-capsules", dest="replace_cylinders_with_capsules", action="store_true", default=False,
                        help="Legacy flag to replace cylinder collision geoms with capsules to match IsaacLab/PhysX convention.")
    parser.add_argument("--penetration-resolver", choices=["hard_constraint", "xyz_nudge"], default=None,
                        help="Legacy override the contact handling mode for retargeting.")

    args = parser.parse_args()

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

    selected_source = robot_config.get("selected_source", {})
    legacy_motion_path = args.motion or args.smplx_motion
    legacy_model_dir = args.model_dir or args.smplx_model_dir
    runtime_source_options = {}
    data_source_source_config = {}

    if args.smplx_motion is not None:
        warnings.warn(
            "--smplx_motion is deprecated; use --motion or --source-config instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    if args.smplx_model_dir is not None:
        warnings.warn(
            "--smplx_model_dir is deprecated; use --model-dir or define model_directory in --source-config.",
            DeprecationWarning,
            stacklevel=2,
        )

    if args.source_config:
        print(f"Loading source config from {args.source_config}...")
        source_config_dict = load_source_config(args.source_config)
        source_type = source_config_dict["type"]
        source_motion_path = source_config_dict["motion"]
        runtime_source_options = {
            key: value
            for key, value in source_config_dict.items()
            if key not in ["type", "motion"]
        }
        if legacy_model_dir is not None and "model_directory" not in runtime_source_options and "smplx_model_dir" not in runtime_source_options:
            warnings.warn(
                "--model-dir is deprecated with --source-config; prefer model_directory in the YAML file.",
                DeprecationWarning,
                stacklevel=2,
            )
            runtime_source_options["model_directory"] = legacy_model_dir
        if args.source or legacy_motion_path is not None or args.source_options is not None:
            warnings.warn(
                "Ignoring legacy source arguments because --source-config was provided.",
                DeprecationWarning,
                stacklevel=2,
            )
        selected_source = select_robot_source(robot_config, source_type)
        data_source_source_config = dict(selected_source)
        print(f"Source type: {source_type}")
        print(f"Motion file: {source_motion_path}")
    else:
        warnings.warn(
            "Legacy CLI source arguments are deprecated; prefer --source-config.",
            DeprecationWarning,
            stacklevel=2,
        )
        if args.source:
            source_entries = robot_config.get("source", [])
            matches = [s for s in source_entries if s.get("name") == args.source or s.get("type") == args.source]
            if len(matches) != 1:
                raise ValueError(f"--source {args.source!r} must match exactly one source entry by name or type.")
            selected_source = matches[0]

        source_type = selected_source.get("type", args.source)
        if not source_type:
            raise ValueError(
                "Source type is required. Provide --source-config, set a source entry in the robot profile, or pass --source."
            )

        source_motion_path = legacy_motion_path
        if source_motion_path is None:
            raise ValueError("Motion input is required. Provide --source-config or use legacy --motion.")

        if args.source_options:
            runtime_source_options = json.loads(args.source_options)
            if not isinstance(runtime_source_options, dict):
                raise ValueError("--source-options must be a JSON object.")

        if legacy_model_dir is not None:
            runtime_source_options["model_directory"] = legacy_model_dir

        data_source_source_config = dict(selected_source)
        print(f"Using legacy CLI source resolution for type: {source_type}")
        print(f"Motion file: {source_motion_path}")

    joint_mapping = selected_source.get("target_mapping")

    if not isinstance(joint_mapping, dict) or not joint_mapping:
        raise ValueError("Joint mapping must be a non-empty JSON object.")

    robot_height = robot_config.get("robot_height")
    retargeting = robot_config.get("retargeting")
    link_offset_config = selected_source.get("link_offset_config", robot_config.get("link_offset_config"))

    # Merge CLI flag into retargeting config
    if retargeting is None:
        retargeting = {}
    if args.replace_cylinders_with_capsules:
        retargeting["replace_cylinders_with_capsules"] = True
    if args.penetration_resolver is not None:
        retargeting["penetration_resolver"] = args.penetration_resolver

    # Handle terrain
    temp_terrain_path = None
    # Check if terrain is in source config first
    if "terrain" in runtime_source_options:
        terrain_path = runtime_source_options.pop("terrain")
        print(f"Using terrain from source config: {terrain_path}")
    elif args.terrain:
        terrain_path = args.terrain
    else:
        print("No terrain provided, creating default flat terrain.")
        flat_terrain = create_flat_terrain()
        fd, temp_terrain_path = tempfile.mkstemp(suffix=".obj")
        os.close(fd)
        flat_terrain.export(temp_terrain_path)
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

        print(f"Loaded trajectory with shape: {source_positions.shape}")
        if source_orientations is not None:
            print(f"Loaded orientations with shape: {source_orientations.shape}")
        else:
            print("Warning: Orientations not available for this file format.")

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
            link_offset_config=link_offset_config,
        )

        # Perform retargeting
        print("Retargeting motion...")
        enable_terrain_scaling = bool(args.output_scaled_terrain)
        source_to_robot_scale, retargeted_motion = retargeter.retarget_motion(
            motion_data,
            framerate=framerate,
            visualize_trajectory=args.vis,
            enable_terrain_scaling=enable_terrain_scaling,
        )

        if args.output_scaled_terrain:
            scaled_terrain = trimesh.load(terrain_path, force="mesh")
            scaled_terrain.apply_scale(source_to_robot_scale)
            output_scaled_terrain_path = Path(args.output_scaled_terrain)
            output_scaled_terrain_path.parent.mkdir(parents=True, exist_ok=True)
            scaled_terrain.export(output_scaled_terrain_path)
            print(f"Saved scaled terrain mesh to {output_scaled_terrain_path}")
        
        # Export scaled objects if requested
        if args.scaled_objects and hasattr(motion_data, 'object_mesh') and motion_data.object_mesh is not None:
            export_scaled_objects(
                motion_data,
                Path(args.scaled_objects),
                source_to_robot_scale,
                apply_scene_scaling=bool(args.output_scaled_terrain),
            )

        # Save output
        print(f"Saving output to {args.output}...")
        
        # Extract data for saving
        # retargeted_motion shape: (T, 7 + DOF) -> [pos(3), quat(4), joints(DOF)]
        
        # Get joint names from robot model
        joint_names = retargeter.get_joint_names()
        
        # Extract components
        base_pos = retargeted_motion[:, :3]
        base_quat = retargeted_motion[:, 3:7] # wxyz
        joint_pos = retargeted_motion[:, 7:]
        
        # Convert quaternion to xyzw if needed (standard for many tools)
        # MuJoCo uses wxyz, but many other tools use xyzw.
        # The example file has 'base_quat_w' which implies world frame.
        # Let's assume the example file uses xyzw convention as it's common in ROS/scipy
        # But wait, MuJoCo uses wxyz. Let's check the example file values if possible.
        # For now, let's stick to wxyz as it is what MuJoCo uses and what we have.
        # If the user wants xyzw, we can convert.
        # Actually, let's look at the example file keys again:
        # ['framerate', 'joint_names', 'joint_pos', 'base_pos_w', 'base_quat_w']
        
        # Save as .npz with specific keys
        np.savez(
            args.output,
            framerate=framerate,
            joint_names=np.array(joint_names),
            joint_pos=joint_pos,
            base_pos_w=base_pos,
            base_quat_w=base_quat # Saving as wxyz (MuJoCo convention)
        )
        
        print(f"Done! Source-to-robot scale used: {source_to_robot_scale}")

        # Load terrain for visualization/video if needed
        vis_terrain = None
        if (args.vis or args.save_video) and terrain_path and os.path.exists(terrain_path):
            try:
                vis_terrain = trimesh.load(terrain_path, force='mesh')
                if args.output_scaled_terrain:
                    vis_terrain.apply_scale(source_to_robot_scale)
            except Exception as e:
                print(f"Could not load terrain for visualization: {e}")

        # Extract per-frame object tracks for visualization if available
        vis_object_meshes = None
        if args.vis or args.save_video:
            vis_object_meshes = build_object_tracks(
                motion_data,
                source_to_robot_scale=source_to_robot_scale,
                apply_scene_scaling=bool(args.output_scaled_terrain),
            )
            if vis_object_meshes:
                print(f"Loaded object track for visualization: {vis_object_meshes[0].name}")

        scaled_source_positions = source_positions * source_to_robot_scale if args.output_scaled_terrain else source_positions

        if args.save_video:
            save_trajectory_video(
                robot_urdf_path,
                retargeted_motion,
                args.save_video,
                source_trajectory=scaled_source_positions,
                terrain_mesh=vis_terrain,
                object_tracks=vis_object_meshes,
                fps=framerate,
            )

        if args.vis:
            visualize_trajectory(
                robot_urdf_path,
                retargeted_motion,
                scaled_source_positions,
                terrain_mesh=vis_terrain,
                object_tracks=vis_object_meshes,
                fps=framerate,
            )

    finally:
        # Cleanup temp file
        if temp_terrain_path and os.path.exists(temp_terrain_path):
            os.remove(temp_terrain_path)

if __name__ == "__main__":
    main()
