import argparse
import warnings
import numpy as np
import trimesh
from pathlib import Path
import tempfile
import os
import json
import yaml
import time

from omniretargeting import OmniRetargeter
from omniretargeting.robot_config import load_robot_config
from omniretargeting.data_sources.registry import create_data_source
from omniretargeting.utils import normalize_retargeted_output_path

import contextlib
import shutil
import re
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROBOT_CONFIG_PATH = REPO_ROOT / "robot_models" / "unitree_g1" / "unitree_g1.json"


def _object_track_matrix(rotation_matrix: np.ndarray, translation: np.ndarray, scale: float) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation_matrix, dtype=float) * float(scale)
    transform[:3, 3] = np.asarray(translation, dtype=float)
    return transform


def build_object_tracks(motion_data, source_to_robot_scale: float, apply_scene_scaling: bool):
    if motion_data is None or getattr(motion_data, "object_mesh", None) is None:
        return None

    translations = motion_data.metadata.get("object_translations")
    rotations = motion_data.metadata.get("object_rotations")
    scales = motion_data.metadata.get("object_scales")
    if translations is None or rotations is None or scales is None:
        return None

    scene_scale = float(source_to_robot_scale) if apply_scene_scaling else 1.0
    object_name = motion_data.metadata.get("object_name", "object")
    base_mesh = motion_data.object_mesh.copy()
    centroid_local = motion_data.metadata.get("object_centroid_local")
    if centroid_local is None:
        centroid_local = np.asarray(base_mesh.vertices, dtype=float).mean(axis=0)
    centroid_local = np.asarray(centroid_local, dtype=float)
    base_mesh.apply_translation(-centroid_local)

    frame_transforms = []
    for translation, rotation, scale in zip(translations, rotations, scales):
        translation = np.asarray(translation, dtype=float) * scene_scale
        scale = float(scale) * scene_scale
        frame_transforms.append(_object_track_matrix(rotation, translation, scale))

    return [{
        "name": object_name,
        "mesh": base_mesh,
        "frame_transforms": frame_transforms,
    }]


@contextlib.contextmanager
def temporary_visualization_scene(urdf_path, terrain_mesh, object_meshes=None, target_faces=5000):
    """
    Context manager that creates a temporary MJCF scene with the robot and terrain.
    Yields the path to the temporary XML file.
    """
    if terrain_mesh is None:
        yield str(urdf_path)
        return

    files_to_remove = []
    dirs_to_remove = []
    
    try:
        # Simplify mesh if needed
        simplified_terrain = terrain_mesh
        if hasattr(terrain_mesh, 'faces') and len(terrain_mesh.faces) > target_faces:
            print(f"Simplifying terrain from {len(terrain_mesh.faces)} to {target_faces} faces for visualization...")
            try:
                simplified_terrain = terrain_mesh.simplify_quadric_decimation(target_faces)
            except Exception as e:
                print(f"Trimesh simplification failed ({e}), trying fast_simplification directly...")
                try:
                    import fast_simplification
                    vertices, faces = fast_simplification.simplify(
                        terrain_mesh.vertices, 
                        terrain_mesh.faces, 
                        target_count=target_faces
                    )
                    simplified_terrain = trimesh.Trimesh(vertices=vertices, faces=faces)
                except ImportError:
                    print("fast_simplification not found. Using original mesh.")
                    simplified_terrain = terrain_mesh

        abs_urdf_path = os.path.abspath(urdf_path)
        is_urdf = str(urdf_path).lower().endswith('.urdf')
        
        if is_urdf:
            # Create temp files in the SAME DIRECTORY as the original URDF
            # This ensures relative paths work correctly and avoids MuJoCo path resolution issues
            urdf_dir = os.path.dirname(abs_urdf_path)
            
            # 2. Create temp URDF in URDF directory
            fd_urdf, temp_urdf_path = tempfile.mkstemp(suffix="_with_terrain.urdf", dir=urdf_dir)
            os.close(fd_urdf)
            files_to_remove.append(temp_urdf_path)

            # 3. Inject terrain into URDF
            with open(urdf_path, 'r') as f:
                urdf_content = f.read()
            
            if "</robot>" in urdf_content:
                # Check for meshdir in compiler tag
                # MuJoCo URDF extension: <compiler meshdir="..."/>
                # If present, MuJoCo looks for meshes relative to this dir and strips paths from filenames
                meshdir_match = re.search(r'<compiler[^>]*meshdir=["\']([^"\']*)["\']', urdf_content)
                
                mesh_save_dir = urdf_dir
                
                if meshdir_match:
                    meshdir_rel = meshdir_match.group(1)
                    print(f"Debug: Found meshdir in URDF: {meshdir_rel}")
                    mesh_save_dir = os.path.normpath(os.path.join(urdf_dir, meshdir_rel))
                    if not os.path.exists(mesh_save_dir):
                        print(f"Debug: meshdir {mesh_save_dir} does not exist! Using URDF dir as fallback.")
                        mesh_save_dir = urdf_dir
                
                # 1. Save mesh to the correct directory
                # We do this here instead of earlier to ensure we use the correct directory
                try:
                    fd_mesh, temp_mesh_path = tempfile.mkstemp(suffix="_terrain_vis.obj", dir=mesh_save_dir)
                    os.close(fd_mesh)
                    files_to_remove.append(temp_mesh_path)
                    
                    simplified_terrain.export(temp_mesh_path)
                    print(f"Debug: Saved temp mesh to {temp_mesh_path}")
                    
                    # For URDF injection, we use just the filename if meshdir is present,
                    # or the relative filename if not.
                    # Since we saved it in the expected directory, basename should work if meshdir is set.
                    # If meshdir is NOT set, we saved it in urdf_dir, so basename works too (relative to URDF).
                    mesh_filename_in_urdf = os.path.basename(temp_mesh_path)
                    
                except Exception as e:
                    print(f"Debug: Failed to save mesh to {mesh_save_dir}: {e}")
                    raise e

                # Add a disconnected link for the terrain
                terrain_link = f"""
  <link name="terrain_vis_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_filename_in_urdf}" scale="1 1 1"/>
      </geometry>
      <material name="terrain_mat">
        <color rgba="0.6 0.6 0.6 1"/>
      </material>
    </visual>
  </link>
"""
                
                # Add object meshes if provided
                object_links = ""
                if object_meshes:
                    for obj_idx, obj_mesh_info in enumerate(object_meshes):
                        obj_mesh = obj_mesh_info["mesh"]
                        obj_name = obj_mesh_info.get("name", f"object_{obj_idx}")
                        
                        try:
                            fd_obj, temp_obj_path = tempfile.mkstemp(suffix=f"_{obj_name}.obj", dir=mesh_save_dir)
                            os.close(fd_obj)
                            files_to_remove.append(temp_obj_path)
                            obj_mesh.export(temp_obj_path)
                            obj_filename = os.path.basename(temp_obj_path)
                            
                            object_links += f"""
  <link name="{obj_name}_vis_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <mesh filename="{obj_filename}" scale="1 1 1"/>
      </geometry>
      <material name="{obj_name}_mat">
        <color rgba="0.8 0.6 0.4 1"/>
      </material>
    </visual>
  </link>
"""
                        except Exception as e:
                            print(f"Warning: Could not add object mesh {obj_name}: {e}")
                
                new_content = urdf_content.replace("</robot>", terrain_link + object_links + "\n</robot>")
                
                with open(temp_urdf_path, "w") as f:
                    f.write(new_content)
                
                print(f"Created temporary visualization scene at {temp_urdf_path}")
                yield temp_urdf_path
            else:
                print("Could not find </robot> tag in URDF. Falling back to original.")
                yield str(urdf_path)
        else:
            # MJCF case - use temp dir
            temp_dir = tempfile.mkdtemp()
            dirs_to_remove.append(temp_dir)
            
            mesh_filename = "terrain_vis.obj"
            mesh_path = os.path.join(temp_dir, mesh_filename)
            simplified_terrain.export(mesh_path)
            abs_mesh_path = os.path.abspath(mesh_path)
            
            mjcf_content = f"""<mujoco>
  <include file="{abs_urdf_path}"/>
  <asset>
    <mesh name="terrain_vis_mesh" file="{abs_mesh_path}"/>
    <texture name="terrain_tex" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .2 .3" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
    <material name="terrain_mat" texture="terrain_tex" texrepeat="10 10" reflectance="0.5"/>
  </asset>
  <worldbody>
    <geom name="terrain_geom" type="mesh" mesh="terrain_vis_mesh" material="terrain_mat" pos="0 0 0"/>
  </worldbody>
</mujoco>"""
            mjcf_path = os.path.join(temp_dir, "scene.xml")
            with open(mjcf_path, "w") as f:
                f.write(mjcf_content)
            
            print(f"Created temporary visualization scene at {mjcf_path}")
            yield mjcf_path
        
    except Exception as e:
        print(f"Failed to setup terrain visualization: {e}")
        import traceback
        traceback.print_exc()
        yield str(urdf_path)
    finally:
        # Cleanup
        for f in files_to_remove:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except OSError:
                    pass
        for d in dirs_to_remove:
            if os.path.exists(d):
                try:
                    shutil.rmtree(d)
                except OSError:
                    pass

def save_trajectory_video(urdf_path, trajectory, output_path, source_trajectory=None, terrain_mesh=None, object_meshes=None, fps=30, width=640, height=480):
    """Render the retargeted trajectory to a video file using MuJoCo offscreen renderer.

    Requires MUJOCO_GL=egl (or osmesa) for headless rendering.
    Requires imageio[ffmpeg]: pip install imageio[ffmpeg]
    """
    import mujoco
    try:
        import imageio
    except ImportError:
        print("Error: imageio not found. Install with: pip install imageio[ffmpeg]")
        return

    print(f"Saving video to {output_path} ({len(trajectory)} frames @ {fps} fps)...")

    with temporary_visualization_scene(urdf_path, terrain_mesh, object_meshes) as model_path:
        try:
            model = mujoco.MjModel.from_xml_path(model_path)
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            if model_path != str(urdf_path):
                print("Falling back to original URDF...")
                model = mujoco.MjModel.from_xml_path(str(urdf_path))
            else:
                return

        data = mujoco.MjData(model)
        from mujoco.rendering.classic.renderer import Renderer

        renderer = None
        gl_context = None
        try:
            gl_context = mujoco.GLContext(width, height)
            gl_context.make_current()
            renderer = Renderer(model, height, width)

            model.vis.headlight.ambient[:] = [0.7, 0.7, 0.7]
            model.vis.headlight.diffuse[:] = [0.7, 0.7, 0.7]
            model.vis.headlight.specular[:] = [0.4, 0.4, 0.4]
            model.vis.map.znear = 0.001
            model.vis.map.zfar = 50.0

            scene = None
            if hasattr(renderer, 'scene'):
                scene = renderer.scene
            elif hasattr(renderer, '_scene'):
                scene = renderer._scene
            else:
                print("Note: Could not access renderer scene for background customization")

            if scene is not None:
                try:
                    scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
                    scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
                    if hasattr(scene, 'rgba_background'):
                        scene.rgba_background[:] = [0.9, 0.9, 0.95, 1.0]
                    print("Video scene lighting customized successfully")
                except (AttributeError, TypeError) as e:
                    print(f"Could not customize renderer scene: {e}")

            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.distance = 3.0
            cam.azimuth = 120.0
            cam.elevation = -20.0

            base_body_id = 1 if model.nbody > 1 else 0
            if model.nbody > 0:
                root_body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 0)
                for body_id in range(1, model.nbody):
                    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                    if body_name and body_name != root_body_name and body_name != 'terrain_vis_link':
                        base_body_id = body_id
                        break

            dynamic_object_specs = []
            if object_meshes and scene is not None:
                if model.nmesh <= 0:
                    raise ValueError("Dynamic object rendering requires mesh assets in the visualization scene.")
                for obj_idx, obj_mesh_info in enumerate(object_meshes):
                    frame_transforms = obj_mesh_info.get("frame_transforms") or []
                    if not frame_transforms:
                        continue
                    geom_idx = scene.ngeom
                    geom = scene.geoms[geom_idx]
                    mujoco.mjv_initGeom(
                        geom,
                        type=mujoco.mjtGeom.mjGEOM_MESH,
                        size=np.array([1.0, 1.0, 1.0]),
                        pos=np.zeros(3),
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0.8, 0.6, 0.4, 1.0]),
                    )
                    geom.dataid = obj_idx
                    dynamic_object_specs.append((geom_idx, frame_transforms))
                    scene.ngeom += 1

            num_frames = len(trajectory)
            with imageio.get_writer(output_path, fps=int(fps), codec="libx264", quality=8, macro_block_size=1) as writer:
                for i in range(num_frames):
                    data.qpos[:] = trajectory[i]
                    mujoco.mj_forward(model, data)
                    cam.lookat[:] = data.xpos[base_body_id]
                    renderer.update_scene(data, camera=cam)
                    if scene is not None:
                        for geom_idx, frame_transforms in dynamic_object_specs:
                            transform = frame_transforms[min(i, len(frame_transforms) - 1)]
                            scene.geoms[geom_idx].pos = transform[:3, 3]
                            scene.geoms[geom_idx].mat[:] = transform[:3, :3]
                    frame = renderer.render()
                    writer.append_data(frame)
                    if (i + 1) % 100 == 0:
                        print(f"  {i+1}/{num_frames}")

            size_mb = os.path.getsize(output_path) / 1024 / 1024
            print(f"Video saved: {output_path} ({size_mb:.1f} MB)")
        finally:
            if renderer is not None:
                renderer.close()
            if gl_context is not None:
                try:
                    gl_context.free()
                except Exception:
                    pass



def visualize_trajectory(urdf_path, trajectory, source_trajectory=None, terrain_mesh=None, object_meshes=None):
    """Visualize the retargeted trajectory and optional source targets in MuJoCo viewer."""
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        print("Error: mujoco package not found. Cannot visualize.")
        return

    print("Launching viewer...")
    print("Controls: Space to pause/resume, [ and ] to step frames.")
    
    with temporary_visualization_scene(urdf_path, terrain_mesh, object_meshes) as model_path:
        # Load model
        try:
            model = mujoco.MjModel.from_xml_path(model_path)
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            if model_path != str(urdf_path):
                print("Falling back to original URDF...")
                model = mujoco.MjModel.from_xml_path(str(urdf_path))
            else:
                return

        data = mujoco.MjData(model)
        
        # Brighten the scene by increasing ambient light
        model.vis.headlight.ambient[:] = [0.6, 0.6, 0.6]  # Increase ambient light
        model.vis.headlight.diffuse[:] = [0.6, 0.6, 0.6]  # Increase diffuse light
        model.vis.headlight.specular[:] = [0.3, 0.3, 0.3]  # Add specular highlights
        
        # Set map values for better visibility
        model.vis.map.znear = 0.001  # Better near clipping
        model.vis.map.zfar = 50.0    # Better far clipping
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Configure viewer for better visibility
            # Try to access scene for background color and rendering settings
            scene = None
            if hasattr(viewer, 'user_scn'):
                scene = viewer.user_scn
                print("Using viewer.user_scn")
            elif hasattr(viewer, 'scn'):
                scene = viewer.scn
                print("Using viewer.scn")
            else:
                print(f"Viewer attributes: {dir(viewer)}")
                print("Note: Could not find scene object (scn or user_scn)")
            
            if scene is not None:
                try:
                    # Disable skybox and fog - just use solid background
                    scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
                    scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0  # Fog was making it darker!
                    
                    # Set background color to bright white/light gray (RGBA)
                    if hasattr(scene, 'rgba_background'):
                        scene.rgba_background[:] = [0.9, 0.9, 0.95, 1.0]
                        print(f"Background color set to: {scene.rgba_background}")
                        
                    print("Scene rendering customized successfully")
                except (AttributeError, TypeError) as e:
                    print(f"Could not customize scene: {e}")
            
            # Enable coordinate frame visualization
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD  # Show world frame
            
            # Show visual geometries (meshes) instead of collision shapes
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 0  # Hide convex hulls
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1      # Show static bodies
            
            num_frames = len(trajectory)
            frame_idx = 0
            source_frame_idx = 0
            
            # Playback speed control
            fps = 30.0
            dt = 1.0 / fps
            
            source_geoms_base = None
            source_num_targets = 0
            if source_trajectory is not None and scene is not None:
                source_num_targets = source_trajectory.shape[1]
                source_geoms_base = scene.ngeom
                for i in range(source_num_targets):
                    geom = scene.geoms[source_geoms_base + i]
                    mujoco.mjv_initGeom(
                        geom,
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=np.array([0.02, 0.0, 0.0]),
                        pos=np.zeros(3),
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0.1, 0.9, 0.1, 0.9]),
                    )
                scene.ngeom = source_geoms_base + source_num_targets

            dynamic_object_specs = []
            if object_meshes and scene is not None:
                if model.nmesh <= 0:
                    raise ValueError("Dynamic object rendering requires mesh assets in the visualization scene.")
                for obj_idx, obj_mesh_info in enumerate(object_meshes):
                    frame_transforms = obj_mesh_info.get("frame_transforms") or []
                    if not frame_transforms:
                        continue
                    geom_idx = scene.ngeom
                    geom = scene.geoms[geom_idx]
                    mujoco.mjv_initGeom(
                        geom,
                        type=mujoco.mjtGeom.mjGEOM_MESH,
                        size=np.array([1.0, 1.0, 1.0]),
                        pos=np.zeros(3),
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0.8, 0.6, 0.4, 1.0]),
                    )
                    geom.dataid = obj_idx
                    dynamic_object_specs.append((geom_idx, frame_transforms))
                    scene.ngeom += 1

            while viewer.is_running():
                step_start = time.time()
                
                # Update background color every frame (some viewers need this)
                if scene is not None and hasattr(scene, 'rgba_background'):
                    scene.rgba_background[:] = [0.9, 0.9, 0.95, 1.0]
                
                # Update state
                data.qpos[:] = trajectory[frame_idx]
                mujoco.mj_forward(model, data)

                if source_geoms_base is not None:
                    source_targets = source_trajectory[source_frame_idx]
                    for i in range(source_num_targets):
                        scene.geoms[source_geoms_base + i].pos = source_targets[i]
                
                # Advance frame
                frame_idx = (frame_idx + 1) % num_frames
                if source_trajectory is not None:
                    source_frame_idx = (source_frame_idx + 1) % len(source_trajectory)
                
                # Sync viewer
                viewer.sync()
                
                # Sleep to maintain frame rate
                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        
        # Cleanup handled by context manager

def create_flat_terrain(size=10.0):
    """Create a simple flat terrain mesh."""
    mesh = trimesh.creation.box(extents=[size, size, 0.1])
    mesh.apply_translation([0, 0, -0.05])
    return mesh


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


def main():
    parser = argparse.ArgumentParser(description="OmniRetargeting CLI")
    parser.add_argument(
        "--robot-config",
        default=DEFAULT_ROBOT_CONFIG_PATH,
        help=f"Path to robot configuration JSON file (default: {DEFAULT_ROBOT_CONFIG_PATH})",
    )
    parser.add_argument("--source-config", default=None, help="Path to YAML source configuration file (see config_templates/ for examples)")
    parser.add_argument("--source", default=None, help="Legacy source entry name or source type from the robot profile (default: active_source)")
    parser.add_argument("--motion", default=None, help="Legacy path to source motion file")
    parser.add_argument("--source-options", default=None, help="Legacy JSON object with adapter-specific source options")
    parser.add_argument("--model-dir", default=None, help="Legacy adapter model directory, when required by the source type")
    parser.add_argument("--smplx_model_dir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--smplx_motion", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--scaled-objects", default=None, help="Directory to save scaled object meshes and pose trajectories (optional)")
    parser.add_argument("--output", required=True, help="Path to save output motion (.npy)")
    parser.add_argument("--terrain", help="Path to terrain mesh file (optional, defaults to flat ground)")
    parser.add_argument(
        "--output-scaled-terrain",
        dest="output_scaled_terrain",
        default=None,
        help="Path to save the scaled terrain mesh. When provided, terrain scaling is enabled.",
    )
    parser.add_argument("--mapping", help="Path to joint mapping JSON file (optional, overrides robot profile mapping)")
    parser.add_argument("--vis", action="store_true", help="Visualize the retargeted motion")
    parser.add_argument("--save-video", dest="save_video", default=None, help="Save retargeted motion video to file (e.g. /tmp/out.mp4). Uses offscreen rendering (set MUJOCO_GL=egl for headless).")
    parser.add_argument("--framerate", type=float, default=None, help="Framerate of the motion (optional, defaults to 30.0 or auto-detected)")
    parser.add_argument("--replace-cylinders-with-capsules", dest="replace_cylinders_with_capsules", action="store_true", default=False,
                        help="Replace cylinder collision geoms with capsules to match IsaacLab/PhysX convention.")
    parser.add_argument("--penetration-resolver", choices=["hard_constraint", "xyz_nudge"], default="xyz_nudge",
                        help="Override the contact handling mode for retargeting.")

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

    # Load joint mapping
    if args.mapping:
        with open(args.mapping, 'r') as f:
            joint_mapping = json.load(f)
    elif "joint_mapping" in robot_config:
        joint_mapping = robot_config["joint_mapping"]
    else:
        raise ValueError(
            "No joint mapping available. Provide --mapping or use a robot profile with 'joint_mapping'."
        )

    if not isinstance(joint_mapping, dict) or not joint_mapping:
        raise ValueError("Joint mapping must be a non-empty JSON object.")

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

    robot_height = robot_config.get("robot_height")
    retargeting = robot_config.get("retargeting")
    link_offset_config = robot_config.get("link_offset_config")

    # Merge CLI flag into retargeting config
    if retargeting is None:
        retargeting = {}
    if args.replace_cylinders_with_capsules:
        retargeting["replace_cylinders_with_capsules"] = True
    if args.penetration_resolver is not None:
        retargeting["penetration_resolver"] = args.penetration_resolver

    # Handle terrain
    temp_terrain_path = None
    if args.terrain:
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
            scaled_objects_dir = Path(args.scaled_objects)
            scaled_objects_dir.mkdir(parents=True, exist_ok=True)

            object_name = motion_data.metadata.get("object_name", "object")
            centroid_local = motion_data.metadata.get("object_centroid_local")
            if centroid_local is None:
                centroid_local = np.asarray(motion_data.object_mesh.vertices, dtype=float).mean(axis=0)
            centroid_local = np.asarray(centroid_local, dtype=float)

            # Save centered object mesh so per-frame transforms carry the motion explicitly.
            scaled_mesh = motion_data.object_mesh.copy()
            scaled_mesh.apply_translation(-centroid_local)
            if args.output_scaled_terrain:
                scaled_mesh.apply_scale(source_to_robot_scale)
            mesh_path = scaled_objects_dir / f"{object_name}.obj"
            scaled_mesh.export(mesh_path)
            print(f"Saved scaled object mesh to {mesh_path}")

            translations = motion_data.metadata.get("object_translations")
            rotations = motion_data.metadata.get("object_rotations")
            scales = motion_data.metadata.get("object_scales")
            if translations is not None and rotations is not None and scales is not None:
                scene_scale = source_to_robot_scale if args.output_scaled_terrain else 1.0
                poses = []
                for t in range(len(translations)):
                    poses.append({
                        "frame": t,
                        "translation": np.asarray(translations[t]).tolist(),
                        "rotation_matrix": np.asarray(rotations[t]).tolist(),
                        "scale": float(scales[t]) * scene_scale
                    })

                pose_path = scaled_objects_dir / f"{object_name}_poses.json"
                with open(pose_path, "w") as f:
                    json.dump(poses, f, indent=2)
                print(f"Saved object pose trajectory to {pose_path}")

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
                print(f"Loaded object track for visualization: {vis_object_meshes[0]['name']}")

        if args.save_video:
            save_trajectory_video(
                robot_urdf_path, retargeted_motion, args.save_video,
                source_trajectory=source_positions * source_to_robot_scale,
                terrain_mesh=vis_terrain, object_meshes=vis_object_meshes, fps=framerate,
            )

        if args.vis:
            visualize_trajectory(robot_urdf_path, retargeted_motion, source_positions * source_to_robot_scale, terrain_mesh=vis_terrain, object_meshes=vis_object_meshes)

    finally:
        # Cleanup temp file
        if temp_terrain_path and os.path.exists(temp_terrain_path):
            os.remove(temp_terrain_path)

if __name__ == "__main__":
    main()
