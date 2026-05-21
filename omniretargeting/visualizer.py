from __future__ import annotations

import contextlib
import os
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


OBJECT_RGBA = np.array([0.8, 0.6, 0.4, 1.0], dtype=float)
SOURCE_RGBA = np.array([0.1, 0.9, 0.1, 0.9], dtype=float)
IDENTITY_MAT = np.eye(3, dtype=float).reshape(-1)
IDENTITY_SIZE = np.array([1.0, 1.0, 1.0], dtype=float)
SOURCE_SPHERE_SIZE = np.array([0.02, 0.0, 0.0], dtype=float)


@dataclass
class ObjectTrack:
    name: str
    mesh: trimesh.Trimesh
    transforms: list[np.ndarray]


@dataclass
class SceneFile:
    model_path: str
    object_mesh_ids: list[int]



def _transform_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    transform[:3, 3] = np.asarray(translation, dtype=float)
    return transform



def build_object_tracks(motion_data, source_to_robot_scale: float, apply_scene_scaling: bool) -> list[ObjectTrack] | None:
    if motion_data is None or getattr(motion_data, "object_mesh", None) is None:
        return None

    translations = motion_data.metadata.get("object_translations")
    rotations = motion_data.metadata.get("object_rotations")
    scales = motion_data.metadata.get("object_scales")
    if translations is None or rotations is None or scales is None:
        return None

    scene_scale = float(source_to_robot_scale) if apply_scene_scaling else 1.0
    mesh = motion_data.object_mesh.copy()
    first_scale = float(scales[0]) * scene_scale
    mesh.apply_scale(first_scale)

    centroid = motion_data.metadata.get("object_centroid_local")
    if centroid is None:
        centroid = np.asarray(mesh.vertices, dtype=float).mean(axis=0)
    else:
        centroid = np.asarray(centroid, dtype=float) * first_scale
    mesh.apply_translation(-centroid)

    transforms = []
    for translation, rotation in zip(translations, rotations):
        translation = np.asarray(translation, dtype=float) * scene_scale
        rotation = np.asarray(rotation, dtype=float).T
        transforms.append(_transform_matrix(rotation, translation))

    object_name = motion_data.metadata.get("object_name", "object")
    return [ObjectTrack(name=object_name, mesh=mesh, transforms=transforms)]



def create_flat_terrain(size: float = 10.0) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=[size, size, 0.1])
    mesh.apply_translation([0.0, 0.0, -0.05])
    return mesh


@contextlib.contextmanager
def temporary_visualization_scene(
    urdf_path,
    terrain_mesh: trimesh.Trimesh | None,
    object_tracks: list[ObjectTrack] | None = None,
    target_faces: int = 5000,
):
    import mujoco

    object_tracks = object_tracks or []
    if terrain_mesh is None and not object_tracks:
        yield SceneFile(model_path=str(urdf_path), object_mesh_ids=[])
        return

    temp_dir = tempfile.mkdtemp()
    try:
        base_model = mujoco.MjModel.from_xml_path(str(urdf_path))
        base_xml_path = os.path.join(temp_dir, "robot.xml")
        mujoco.mj_saveLastXML(base_xml_path, base_model)

        tree = ET.parse(base_xml_path)
        root = tree.getroot()
        _rewrite_mesh_paths(root, os.path.dirname(os.path.abspath(urdf_path)))

        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")
        worldbody = root.find("worldbody")
        if worldbody is None:
            worldbody = ET.SubElement(root, "worldbody")

        if terrain_mesh is not None:
            terrain_mesh = _simplify_terrain(terrain_mesh, target_faces)
            terrain_path = os.path.join(temp_dir, "terrain.obj")
            terrain_mesh.export(terrain_path)
            asset.append(ET.fromstring(f'<mesh name="terrain_vis_mesh" file="{terrain_path}"/>'))
            asset.append(ET.fromstring('<texture name="terrain_tex" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 .2 .3" width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>'))
            asset.append(ET.fromstring('<material name="terrain_mat" texture="terrain_tex" texrepeat="10 10" reflectance="0.5"/>'))
            worldbody.append(ET.fromstring('<geom name="terrain_geom" type="mesh" mesh="terrain_vis_mesh" material="terrain_mat" pos="0 0 0"/>'))

        for idx, track in enumerate(object_tracks):
            mesh_path = os.path.join(temp_dir, f"object_{idx}.obj")
            track.mesh.export(mesh_path)
            mesh_name = f"dynamic_object_mesh_{idx}"
            body_name = f"dynamic_object_body_{idx}"
            geom_name = f"dynamic_object_geom_{idx}"
            asset.append(ET.fromstring(f'<mesh name="{mesh_name}" file="{mesh_path}"/>'))
            worldbody.append(ET.fromstring(
                f'<body name="{body_name}" pos="0 0 0">'
                f'<geom name="{geom_name}" type="mesh" mesh="{mesh_name}" rgba="0.8 0.6 0.4 0"/>'
                f'</body>'
            ))

        scene_path = os.path.join(temp_dir, "scene.xml")
        tree.write(scene_path, encoding="unicode")
        print(f"Created temporary visualization scene at {scene_path}")

        model = mujoco.MjModel.from_xml_path(scene_path)
        object_mesh_ids = []
        for idx in range(len(object_tracks)):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"dynamic_object_geom_{idx}")
            if geom_id < 0:
                raise ValueError(f"Dynamic object geom {idx} missing from visualization scene")
            object_mesh_ids.append(int(model.geom_dataid[geom_id]))

        yield SceneFile(model_path=scene_path, object_mesh_ids=object_mesh_ids)
    finally:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)



def save_trajectory_video(
    urdf_path,
    trajectory,
    output_path,
    source_trajectory=None,
    terrain_mesh=None,
    object_tracks: list[ObjectTrack] | None = None,
    fps: float = 30,
    width: int = 640,
    height: int = 480,
):
    import mujoco

    try:
        import imageio
    except ImportError:
        print("Error: imageio not found. Install with: pip install imageio[ffmpeg]")
        return

    trajectory = np.asarray(trajectory)
    source_trajectory = None if source_trajectory is None else np.asarray(source_trajectory)
    object_tracks = object_tracks or []

    print(f"Saving video to {output_path} ({len(trajectory)} frames @ {fps} fps)...")
    with temporary_visualization_scene(urdf_path, terrain_mesh, object_tracks) as scene_file:
        model = mujoco.MjModel.from_xml_path(scene_file.model_path)
        data = mujoco.MjData(model)
        _configure_model_visuals(model, ambient=0.7, diffuse=0.7, specular=0.4)

        from mujoco.rendering.classic.renderer import Renderer

        renderer = None
        gl_context = None
        try:
            gl_context = mujoco.GLContext(width, height)
            gl_context.make_current()
            renderer = Renderer(model, height, width)
            scene = _renderer_scene(renderer)
            if scene is not None:
                _configure_scene(scene, mujoco)

            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.distance = 3.0
            camera.azimuth = 120.0
            camera.elevation = -20.0
            base_body_id = _primary_body_id(model, mujoco)

            with imageio.get_writer(output_path, fps=int(fps), codec="libx264", quality=8, macro_block_size=1) as writer:
                for frame_idx, qpos in enumerate(trajectory):
                    data.qpos[:] = qpos
                    mujoco.mj_forward(model, data)
                    camera.lookat[:] = data.xpos[base_body_id]
                    renderer.update_scene(data, camera=camera)
                    if scene is not None:
                        _append_dynamic_geoms(scene, scene_file.object_mesh_ids, object_tracks, frame_idx, mujoco)
                    writer.append_data(renderer.render())
                    if (frame_idx + 1) % 100 == 0:
                        print(f"  {frame_idx + 1}/{len(trajectory)}")

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



def visualize_trajectory(
    urdf_path,
    trajectory,
    source_trajectory=None,
    terrain_mesh=None,
    object_tracks: list[ObjectTrack] | None = None,
    fps: float = 30.0,
):
    try:
        import mujoco
        import mujoco.viewer
    except ImportError:
        print("Error: mujoco package not found. Cannot visualize.")
        return

    trajectory = np.asarray(trajectory)
    source_trajectory = None if source_trajectory is None else np.asarray(source_trajectory)
    object_tracks = object_tracks or []

    print("Launching viewer...")
    print("Controls: Space to pause/resume, [ and ] to step frames.")

    with temporary_visualization_scene(urdf_path, terrain_mesh, object_tracks) as scene_file:
        model = mujoco.MjModel.from_xml_path(scene_file.model_path)
        data = mujoco.MjData(model)
        _configure_model_visuals(model, ambient=0.6, diffuse=0.6, specular=0.3)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            scene = _viewer_scene(viewer)
            if scene is not None:
                _configure_scene(scene, mujoco)
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 0
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 1

            source_base = None
            source_count = 0
            if scene is not None and source_trajectory is not None:
                source_count = source_trajectory.shape[1]
                source_base = scene.ngeom
                for idx in range(source_count):
                    geom = scene.geoms[source_base + idx]
                    mujoco.mjv_initGeom(
                        geom,
                        type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=SOURCE_SPHERE_SIZE,
                        pos=np.zeros(3),
                        mat=IDENTITY_MAT,
                        rgba=SOURCE_RGBA,
                    )
                scene.ngeom = source_base + source_count

            object_geom_indices = []
            if scene is not None:
                for mesh_id in scene_file.object_mesh_ids:
                    geom_idx = scene.ngeom
                    geom = scene.geoms[geom_idx]
                    mujoco.mjv_initGeom(
                        geom,
                        type=mujoco.mjtGeom.mjGEOM_MESH,
                        size=IDENTITY_SIZE,
                        pos=np.zeros(3),
                        mat=IDENTITY_MAT,
                        rgba=OBJECT_RGBA,
                    )
                    geom.dataid = mesh_id
                    scene.ngeom += 1
                    object_geom_indices.append(geom_idx)

            frame_idx = 0
            source_frame_idx = 0
            dt = 1.0 / float(fps)
            while viewer.is_running():
                start = time.time()
                if scene is not None and hasattr(scene, "rgba_background"):
                    scene.rgba_background[:] = [0.9, 0.9, 0.95, 1.0]

                data.qpos[:] = trajectory[frame_idx]
                mujoco.mj_forward(model, data)

                if scene is not None and source_base is not None and source_trajectory is not None:
                    source_points = source_trajectory[source_frame_idx]
                    for idx in range(source_count):
                        scene.geoms[source_base + idx].pos = source_points[idx]

                if scene is not None:
                    for geom_idx, track in zip(object_geom_indices, object_tracks):
                        transform = track.transforms[min(frame_idx, len(track.transforms) - 1)]
                        scene.geoms[geom_idx].pos = transform[:3, 3]
                        scene.geoms[geom_idx].mat = transform[:3, :3].reshape(-1)

                frame_idx = (frame_idx + 1) % len(trajectory)
                if source_trajectory is not None:
                    source_frame_idx = (source_frame_idx + 1) % len(source_trajectory)
                viewer.sync()

                remain = dt - (time.time() - start)
                if remain > 0:
                    time.sleep(remain)



def _simplify_terrain(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if not hasattr(mesh, "faces") or len(mesh.faces) <= target_faces:
        return mesh
    print(f"Simplifying terrain from {len(mesh.faces)} to {target_faces} faces for visualization...")
    try:
        return mesh.simplify_quadric_decimation(target_faces)
    except Exception:
        return mesh



def _rewrite_mesh_paths(root: ET.Element, base_dir: str):
    asset = root.find("asset")
    if asset is None:
        return
    for mesh in asset.findall("mesh"):
        filename = mesh.get("file")
        if not filename or os.path.isabs(filename):
            continue
        mesh.set("file", os.path.abspath(os.path.join(base_dir, filename)))



def _configure_model_visuals(model, ambient: float, diffuse: float, specular: float):
    model.vis.headlight.ambient[:] = [ambient, ambient, ambient]
    model.vis.headlight.diffuse[:] = [diffuse, diffuse, diffuse]
    model.vis.headlight.specular[:] = [specular, specular, specular]
    model.vis.map.znear = 0.001
    model.vis.map.zfar = 50.0



def _renderer_scene(renderer):
    if hasattr(renderer, "scene"):
        return renderer.scene
    if hasattr(renderer, "_scene"):
        return renderer._scene
    return None



def _viewer_scene(viewer):
    if hasattr(viewer, "user_scn"):
        return viewer.user_scn
    if hasattr(viewer, "scn"):
        return viewer.scn
    return None



def _configure_scene(scene, mujoco):
    scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
    scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
    if hasattr(scene, "rgba_background"):
        scene.rgba_background[:] = [0.9, 0.9, 0.95, 1.0]



def _primary_body_id(model, mujoco) -> int:
    if model.nbody <= 1:
        return 0
    root_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, 0)
    for body_id in range(1, model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name and name != root_name:
            return body_id
    return 0



def _append_dynamic_geoms(scene, object_mesh_ids: list[int], object_tracks: list[ObjectTrack], frame_idx: int, mujoco):
    for mesh_id, track in zip(object_mesh_ids, object_tracks):
        transform = track.transforms[min(frame_idx, len(track.transforms) - 1)]
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_MESH,
            size=IDENTITY_SIZE,
            pos=transform[:3, 3],
            mat=transform[:3, :3].reshape(-1),
            rgba=OBJECT_RGBA,
        )
        geom.dataid = mesh_id
        scene.ngeom += 1
