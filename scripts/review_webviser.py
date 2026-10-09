"""Record an audited motion with its original scene, object poses and contacts."""

import argparse
import threading
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import viser
from mjviser import ViserMujocoScene
from scipy.spatial.transform import Rotation

from omniretargeting._viewer_motion import SMPLX_BODY_PARENTS, source_body_parents
from omniretargeting.viser_playback import (
    serve_recording,
    use_rigid_instance_updates,
    write_recording,
)


def run(args):
    prepared = np.load(args.prepared, allow_pickle=True).item()
    payload = np.load(args.motion)
    poses, source = payload["qpos"], payload["source_joints_w"]
    fps, frames = float(payload["framerate"]), len(poses)
    start = int(payload.get("frame_start", 0))
    choices = {}
    choice_path = args.motion.parent / "contact_choices.npz"
    saved_choices = np.load(choice_path) if choice_path.exists() else {}
    for name, task in prepared["tasks"].items():
        active = task.contact_mask[start:start + frames] & ~task.managed_frames[start:start + frames]
        if not active.any():
            continue
        key = f"contact_choice_{name}"
        if key in payload:
            choices[name] = payload[key]
        elif name in saved_choices:
            choices[name] = saved_choices[name][start:start + frames]
        else:
            raise ValueError(f"Missing solved contact-offset choices for {name}")
        if len(choices[name]) != frames:
            raise ValueError(f"Contact-offset choices do not cover the motion: {name}")
    model = mujoco.MjModel.from_xml_path(prepared["model_path"])
    data = mujoco.MjData(model)
    if args.record_only:
        from omniretargeting.viser_recording import RecordingScene
        server = RecordingScene(label="Contact motion review")
    else:
        server = viser.ViserServer(host="0.0.0.0", port=args.viser_port, label="Contact motion review")
    server.scene.configure_default_lights(cast_shadow=False)
    server.initial_camera.look_at = tuple(poses[0, :3])
    server.initial_camera.position = tuple(poses[0, :3] + [1.8, -1.8, 1.2])
    robot = ViserMujocoScene(server, model, num_envs=1)
    use_rigid_instance_updates(robot)
    robot.camera_tracking_enabled = False
    mesh_groups = {
        int(model.geom_group[g])
        for g in range(model.ngeom)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
    }
    visual_groups = {
        int(model.geom_group[g])
        for g in range(model.ngeom)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        and not (model.geom_contype[g] or model.geom_conaffinity[g])
    } or mesh_groups
    if visual_groups:
        for group in range(len(robot.geom_groups_visible)):
            robot.geom_groups_visible[group] = group in visual_groups
        robot._sync_visibilities()
    scene = trimesh.load_mesh(args.prepared.parent / "static_scene.obj", process=False)
    server.scene.add_mesh_simple(
        "/scene/static",
        vertices=np.asarray(scene.vertices),
        faces=np.asarray(scene.faces),
        color=(175, 185, 195),
        opacity=0.25,
        side="double",
    )
    objects = np.load(args.objects, allow_pickle=True).tolist() if args.objects else []
    handles = []
    for i, obj in enumerate(objects):
        mesh = obj["mesh"]
        handles.append(
            server.scene.add_mesh_simple(
                f"/scene/object_{i}",
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.faces),
                color=(205, 215, 225),
                opacity=0.8,
                side="double",
            )
        )
    source_points = server.scene.add_point_cloud(
        "/source/joints", points=source[0], colors=(35, 110, 255), point_size=0.018
    )
    parents = (
        source_body_parents(payload["source_joint_names"])
        if "source_joint_names" in payload
        else None
    )
    if parents is None:
        parents = SMPLX_BODY_PARENTS[: source.shape[1]]
    bones = np.array(
        [
            (parent, child)
            for child, parent in enumerate(parents)
            if 0 <= parent < source.shape[1]
        ]
    )
    source_bones = server.scene.add_line_segments(
        "/source/bones", points=source[0, bones], colors=(35, 110, 255), line_width=3.0
    )
    body = None
    surface = None if args.source_surface is None else np.load(args.source_surface)
    if surface is not None:
        if len(surface["vertices"]) < start + frames:
            raise ValueError("Source surface does not cover the motion timeline")
        body = server.scene.add_mesh_simple(
            "/source/body",
            vertices=surface["vertices"][start],
            faces=surface["faces"],
            color=(35, 110, 255),
            opacity=0.23,
            side="double",
        )
    targets = server.scene.add_point_cloud(
        "/contact/source_targets",
        points=np.zeros((0, 3)),
        colors=(240, 110, 40),
        point_size=0.018,
    )
    errors = server.scene.add_line_segments(
        "/contact/error",
        points=np.zeros((0, 2, 3)),
        colors=(240, 110, 40),
        line_width=2.0,
    )
    source_description = (
        "半透明蓝色：原始人体" if body is not None else "蓝色骨架：原始人体"
    )
    server.gui.add_markdown(
        f"{frames} 帧 · {frames / fps:.1f} 秒 · 原始尺寸与时序\n\n"
        f"灰色：机器人；{source_description}；橙点：当前接触目标。"
        "橙线连接机器人接触点与目标，越短表示贴合越准确。"
    )

    def render(frame):
        f = frame + start
        with server.atomic():
            data.qpos[:] = poses[frame]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            robot.update_from_mjdata(data)
            for handle, obj in zip(handles, objects):
                handle.position = obj["translations"][f]
                handle.wxyz = Rotation.from_matrix(obj["rotations"][f]).as_quat(
                    scalar_first=True
                )
                if obj.get("scales") is not None:
                    handle.scale = float(np.asarray(obj["scales"]).reshape(-1)[f])
            source_points.points = source[frame]
            source_bones.points = source[frame, bones]
            if body is not None:
                body.vertices = surface["vertices"][f]
            wanted, pairs = [], []
            for name, task in prepared["tasks"].items():
                if not task.contact_mask[f] or task.managed_frames[f]:
                    continue
                spec = prepared["effectors"]["effectors"][name]
                bid = model.body(spec["robot_body"]).id
                choice = int(choices[name][frame])
                point = (
                    data.xpos[bid]
                    + data.xmat[bid].reshape(3, 3) @ spec["contact_offsets"][choice]
                )
                wanted.append(task.contact_positions[f])
                pairs.append([point, task.contact_positions[f]])
            dense = prepared.get("surface", {})
            if dense:
                for k in np.flatnonzero(dense["surface_contact_mask"][f]):
                    gid = int(dense["surface_contact_robot_geoms"][k])
                    point = (
                        data.geom_xpos[gid]
                        + data.geom_xmat[gid].reshape(3, 3)
                        @ dense["surface_contact_robot_local_points"][k]
                    )
                    target = dense["surface_contact_targets_w"][f, k]
                    wanted.append(target)
                    pairs.append([point, target])
            targets.points = np.asarray(wanted).reshape(-1, 3)
            errors.points = np.asarray(pairs).reshape(-1, 2, 3)
        server.flush()

    render(0)
    serializer = server.get_scene_serializer()
    for frame in range(1, frames):
        serializer.insert_sleep(1 / fps)
        render(frame)
        if frame % 300 == 0:
            print(f"recorded {frame}/{frames}", flush=True)
    serializer.insert_sleep(1 / fps)
    manifest = write_recording(
        args.output,
        serializer.serialize(),
        camera_position=server.initial_camera.position,
        camera_look_at=server.initial_camera.look_at,
        metadata={
            "source_frames": frames,
            "source_fps": fps,
            "display_hz": fps,
            "motion_duration_s": (frames - 1) / fps,
            "loop_duration_s": frames / fps,
            "motion": str(args.motion),
            "scene": "original metric triangles",
            "moving_objects": len(objects),
            "contact_targets": "fixed prepared world observations",
        },
    )
    print(f"recording {manifest['bytes'] / 1e6:.1f} MB", flush=True)
    if args.record_only:
        server.stop()
        return
    serve_recording(args.output, args.port, host="0.0.0.0")
    print(f"READY port {args.port}", flush=True)
    threading.Event().wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--objects", type=Path)
    parser.add_argument("--source-surface", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--viser-port", type=int, default=8183)
    parser.add_argument("--port", type=int, default=8183)
    parser.add_argument("--record-only", action="store_true")
    run(parser.parse_args())
