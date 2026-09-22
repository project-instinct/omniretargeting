"""Scene creation, interactive playback and video export for the CLI."""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import tempfile
import time

import mujoco
import numpy as np
import trimesh


@contextlib.contextmanager
def temporary_visualization_scene(urdf_path, terrain_mesh, target_faces=5000):
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
                new_content = urdf_content.replace("</robot>", terrain_link + "\n</robot>")

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


def save_trajectory_video(urdf_path, trajectory, output_path, smplx_trajectory=None, terrain_mesh=None, fps=30, width=640, height=480):
    """Render the retargeted trajectory to a video file using MuJoCo offscreen renderer.

    Requires MUJOCO_GL=egl (or osmesa) for headless rendering.
    Requires imageio[ffmpeg]: pip install imageio[ffmpeg]
    """
    try:
        import imageio
    except ImportError:
        print("Error: imageio not found. Install with: pip install imageio[ffmpeg]")
        return

    print(f"Saving video to {output_path} ({len(trajectory)} frames @ {fps} fps)...")

    with temporary_visualization_scene(urdf_path, terrain_mesh) as model_path:
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
        renderer = mujoco.Renderer(model, height, width)

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = 5.0
        cam.azimuth = 120.0
        cam.elevation = -20.0
        cam.lookat[:] = trajectory[0, :3]

        num_frames = len(trajectory)
        try:
            with imageio.get_writer(output_path, fps=int(fps), codec="libx264",
                                    quality=8, macro_block_size=1) as writer:
                for i in range(num_frames):
                    data.qpos[:] = trajectory[i]
                    mujoco.mj_forward(model, data)
                    # Follow motions with a non-trivial root path.  A fixed
                    # origin camera loses traversal/parkour clips after only a
                    # few frames and produces a technically valid but empty
                    # preview video.
                    cam.lookat[:] = trajectory[i, :3]
                    renderer.update_scene(data, camera=cam)
                    frame = renderer.render()
                    writer.append_data(frame)
                    if (i + 1) % 100 == 0:
                        print(f"  {i+1}/{num_frames}")

            size_mb = os.path.getsize(output_path) / 1024 / 1024
            print(f"Video saved: {output_path} ({size_mb:.1f} MB)")
        finally:
            renderer.close()


def visualize_trajectory(urdf_path, trajectory, smplx_trajectory=None, terrain_mesh=None):
    """Visualize the retargeted trajectory and optional SMPLX joints in MuJoCo viewer."""
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        print("Error: mujoco package not found. Cannot visualize.")
        return

    print("Launching viewer...")
    print("Controls: Space to pause/resume, [ and ] to step frames.")

    with temporary_visualization_scene(urdf_path, terrain_mesh) as model_path:
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
            smplx_frame_idx = 0

            # Playback speed control
            fps = 30.0
            dt = 1.0 / fps

            # Setup SMPLX joint visualization (spheres) if provided
            smplx_geoms_base = None
            smplx_num_joints = 0
            if smplx_trajectory is not None and scene is not None:
                smplx_num_joints = smplx_trajectory.shape[1]
                smplx_geoms_base = scene.ngeom
                for i in range(smplx_num_joints):
                    geom = scene.geoms[smplx_geoms_base + i]
                    mujoco.mjv_initGeom(
                        geom,
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=np.array([0.02, 0.0, 0.0]),
                        pos=np.zeros(3),
                        mat=np.eye(3).flatten(),
                        rgba=np.array([0.1, 0.9, 0.1, 0.9]),
                    )
                scene.ngeom = smplx_geoms_base + smplx_num_joints

            while viewer.is_running():
                step_start = time.time()

                # Update background color every frame (some viewers need this)
                if scene is not None and hasattr(scene, 'rgba_background'):
                    scene.rgba_background[:] = [0.9, 0.9, 0.95, 1.0]

                # Update state
                data.qpos[:] = trajectory[frame_idx]
                mujoco.mj_forward(model, data)

                # Update SMPLX joint markers
                if smplx_geoms_base is not None:
                    smplx_joints = smplx_trajectory[smplx_frame_idx]
                    for i in range(smplx_num_joints):
                        scene.geoms[smplx_geoms_base + i].pos = smplx_joints[i]

                # Advance frame
                frame_idx = (frame_idx + 1) % num_frames
                if smplx_trajectory is not None:
                    smplx_frame_idx = (smplx_frame_idx + 1) % len(smplx_trajectory)

                # Sync viewer
                viewer.sync()

                # Sleep to maintain frame rate
                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
