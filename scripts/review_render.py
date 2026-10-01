"""Render a saved review motion and unchanged scene with its source skeleton."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from omniretargeting._viewer_motion import SMPLX_BODY_PARENTS


def add_mesh(spec, body, name, mesh, opacity):
    spec.add_mesh(
        name=name,
        uservert=np.asarray(mesh.vertices).ravel(),
        userface=np.asarray(mesh.faces).ravel(),
    )
    body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=name,
        rgba=[0.65, 0.71, 0.76, opacity],
        group=0,
        contype=0,
        conaffinity=0,
    )


def draw_source(scene, joints):
    for child, parent in enumerate(SMPLX_BODY_PARENTS):
        if parent < 0 or child >= len(joints) or scene.ngeom >= scene.maxgeom:
            continue
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.zeros(3),
            np.zeros(3),
            np.eye(3).ravel(),
            np.array([0.1, 0.5, 1.0, 0.65]),
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            0.009,
            joints[parent],
            joints[child],
        )
        scene.ngeom += 1


def render(args):
    inputs = np.load(args.prepared, allow_pickle=True).item()
    payload = np.load(args.motion)
    poses, source = payload["qpos"], payload["source_joints_w"]
    objects = np.load(args.prepared.parent / "objects.npy", allow_pickle=True).tolist()
    spec = mujoco.MjSpec.from_file(inputs["model_path"])
    mesh = trimesh.load_mesh(args.prepared.parent / "static_scene.obj", process=False)
    add_mesh(spec, spec.worldbody, "review_scene", mesh, 0.2)
    for k, obj in enumerate(objects):
        body = spec.worldbody.add_body(name=f"review_object_{k}", mocap=True)
        add_mesh(spec, body, f"review_object_mesh_{k}", obj["mesh"], 0.65)
    model = spec.compile()
    if model.nq != poses.shape[1]:
        raise ValueError("Visualization changed the robot coordinates")
    model.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]
    model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
    renderer = mujoco.Renderer(model, height=480, width=640)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.elevation = 2.5, -20
    frames = args.frames or (
        list(range(len(poses)))
        if args.video is not None
        else np.linspace(0, len(poses) - 1, 6).astype(int).tolist()
    )
    args.output.mkdir(parents=True, exist_ok=True)
    if args.video is not None:
        args.video.parent.mkdir(parents=True, exist_ok=True)
    writer = (
        None
        if args.video is None
        else imageio.get_writer(
            args.video, fps=float(payload["framerate"]), codec="libx264"
        )
    )
    try:
        for frame in frames:
            data.qpos[:] = poses[frame]
            for k, obj in enumerate(objects):
                mocap = model.body_mocapid[model.body(f"review_object_{k}").id]
                data.mocap_pos[mocap] = obj["translations"][frame]
                data.mocap_quat[mocap] = Rotation.from_matrix(
                    obj["rotations"][frame]
                ).as_quat(
                    scalar_first=True,
                )
            mujoco.mj_forward(model, data)
            camera.lookat[:] = poses[frame, :3] + [0, 0, -0.15]
            views = []
            for azimuth in (-50, 40):
                camera.azimuth = azimuth
                renderer.update_scene(data, camera=camera)
                draw_source(renderer.scene, source[frame])
                pixels = renderer.render().copy()
                if writer is None or frame in (0, len(poses) // 2, len(poses) - 1):
                    path = args.output / f"frame_{frame:04d}_{azimuth}.png"
                    imageio.imwrite(path, pixels)
                    print(path, flush=True)
                views.append(pixels)
            if writer is not None:
                writer.append_data(np.concatenate(views, axis=1))
    finally:
        if writer is not None:
            writer.close()
    renderer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", nargs="+", type=int)
    parser.add_argument("--video", type=Path)
    render(parser.parse_args())
