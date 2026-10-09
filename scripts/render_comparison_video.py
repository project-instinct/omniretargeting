"""Side-by-side review video of saved motions with per-frame penetration colouring.

Each run directory is a benchmark clip output (prepared.npy, objects.npy,
static_scene.obj, final_retargeted.npz, scene_metrics.npz). Collision geoms
penetrating the static scene or a moving object are drawn orange (> 5 mm) or
red (> 10 mm) in the frames where the audit measured that depth; the source
skeleton is drawn in blue. The video only reads audited outputs.

Usage:
  .venv/bin/python scripts/render_comparison_video.py \
      --run old=out/.../bench/A/normal/repeat_1/omomo_254 \
      --run new=out/.../al_25b/1s/omomo_254 --output out/.../videos/omomo_254.mp4
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402
from scipy.ndimage import uniform_filter1d  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from omniretargeting._viewer_motion import source_body_parents  # noqa: E402
from omniretargeting.scene_geometry import collision_geom_ids  # noqa: E402

ORANGE_M, RED_M = 0.005, 0.010


def scaled_object_mesh(obj):
    """Object mesh at its recorded size: world = R (s v) + t (moving_scene convention).

    OMOMO meshes are authored about 3x too large and carry per-frame scales;
    the scale varies by well under 1% within a clip, so its median is used.
    """
    mesh = obj["mesh"]
    scales = obj.get("scales")
    if scales is None:
        return mesh
    scales = np.asarray(scales, dtype=float).reshape(-1)
    scale = float(np.median(scales))
    if scales.size and float(np.ptp(scales)) > 0.01 * scale:
        print(f"warning: object scale varies {scales.min():.4f}-{scales.max():.4f}; drawing median {scale:.4f}",
              flush=True)
    mesh = mesh.copy()
    mesh.apply_scale(scale)
    return mesh


def load_run(directory):
    directory = Path(directory)
    prepared = np.load(directory / "prepared.npy", allow_pickle=True).item()
    motion = np.load(directory / "final_retargeted.npz")
    metrics_path = directory / "scene_metrics.npz"
    metrics = np.load(metrics_path) if metrics_path.exists() else None
    depths = None
    if metrics is not None:
        depths = np.maximum(metrics["static_depths"], metrics["moving_depths"])
    return dict(
        directory=directory,
        model_path=prepared["model_path"],
        qpos=np.asarray(motion["qpos"]),
        source=np.asarray(motion["source_joints_w"]),
        source_names=motion["source_joint_names"] if "source_joint_names" in motion else None,
        framerate=float(motion["framerate"]),
        objects=np.load(directory / "objects.npy", allow_pickle=True).tolist(),
        scene=trimesh.load_mesh(directory / "static_scene.obj", process=False),
        depths=depths,
        static=None if metrics is None else metrics["static_depths"],
        moving=None if metrics is None else metrics["moving_depths"],
    )


def build_model(run):
    spec = mujoco.MjSpec.from_file(run["model_path"])

    def add(body, name, mesh, rgba):
        spec.add_mesh(name=name, uservert=np.asarray(mesh.vertices).ravel(),
                      userface=np.asarray(mesh.faces).ravel())
        body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, rgba=rgba,
                      group=0, contype=0, conaffinity=0)

    add(spec.worldbody, "video_scene", run["scene"], [.82, .84, .86, .35])
    for index, obj in enumerate(run["objects"]):
        body = spec.worldbody.add_body(name=f"video_object_{index}", mocap=True)
        add(body, f"video_object_{index}", scaled_object_mesh(obj), [.95, .88, .72, .55])
    model = spec.compile()
    for geom in range(model.ngeom):
        if int(model.geom_group[geom]) == 2:
            model.geom_rgba[geom] = [.62, .70, .80, .6]
        elif int(model.geom_group[geom]) == 3:
            model.geom_rgba[geom] = [1.0, .45, .1, 0.0]
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 960
    model.vis.headlight.ambient[:] = [.45, .45, .45]
    model.vis.headlight.diffuse[:] = [.75, .75, .75]
    return model


def draw_source(scene, joints, parents):
    for child, parent in enumerate(parents):
        if parent < 0 or child >= len(joints) or scene.ngeom >= scene.maxgeom:
            continue
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.eye(3).ravel(), np.array([.1, .45, 1.0, .65]))
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .008, joints[parent], joints[child])
        scene.ngeom += 1


def caption(image, lines):
    from PIL import Image, ImageDraw

    from PIL import ImageFont

    canvas = Image.fromarray(image).convert("RGB")
    height = 12 + 20 * len(lines)
    framed = Image.new("RGB", (canvas.width, canvas.height + height), "white")
    framed.paste(canvas, (0, height))
    draw = ImageDraw.Draw(framed)
    font = None
    for path in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                 "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                 "/usr/share/fonts/truetype/arphic/uming.ttc"):
        try:
            font = ImageFont.truetype(path, 14)
            break
        except OSError:
            continue
    for row, (text, color) in enumerate(lines):
        draw.text((10, 4 + 20 * row), text, fill=color, font=font)
    return np.asarray(framed)


def default_azimuth(qpos):
    """Side view of the clip: perpendicular to the mean horizontal root heading."""
    headings = Rotation.from_quat(qpos[:, 3:7], scalar_first=True).apply([1.0, 0.0, 0.0])
    mean = headings[:, :2].mean(axis=0)
    yaw = np.degrees(np.arctan2(mean[1], mean[0])) if np.linalg.norm(mean) > 1e-6 else 0.0
    return yaw + 90.0


def render(runs, labels, output, *, azimuth=None, distance=2.6, width=480, height=360,
           frames=None, elevation=-18.0):
    reference = runs[0]
    model = build_model(reference)
    if any(run["qpos"].shape[1] != model.nq for run in runs):
        raise ValueError("All runs must use the same robot coordinates")
    # Audit depths are indexed by the robot model's geoms; the added scene and
    # object geoms shift the compiled ids, so map each robot geom by its body
    # and its order within that body.
    robot = mujoco.MjModel.from_xml_path(reference["model_path"])

    def by_body(m):
        table = {}
        for geom in range(m.ngeom):
            table.setdefault(m.body(int(m.geom_bodyid[geom])).name, []).append(geom)
        return table

    robot_table, video_table = by_body(robot), by_body(model)
    collision = {}
    for geom in collision_geom_ids(robot):
        body = robot.body(int(robot.geom_bodyid[geom])).name
        collision[int(geom)] = video_table[body][robot_table[body].index(int(geom))]
    renderer = mujoco.Renderer(model, height=height, width=width)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.elevation = distance, elevation
    base_azimuth = default_azimuth(reference["qpos"]) if azimuth is None else azimuth
    count = min(len(run["qpos"]) for run in runs)
    frames = range(count) if frames is None else frames
    # A smoothed root target keeps the view steady without hiding body motion.
    lookat = uniform_filter1d(reference["qpos"][:count, :3], size=15, axis=0, mode="nearest")
    lookat[:, 2] -= 0.15
    names = reference["source_names"]
    parents = source_body_parents([str(n) for n in names]) if names is not None else None
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(output, fps=reference["framerate"], codec="libx264", quality=7,
                                macro_block_size=8)
    try:
        for frame in frames:
            tiles = []
            for run, label in zip(runs, labels):
                data.qpos[:] = run["qpos"][frame]
                for k, obj in enumerate(run["objects"]):
                    mocap = model.body_mocapid[model.body(f"video_object_{k}").id]
                    data.mocap_pos[mocap] = obj["translations"][frame]
                    data.mocap_quat[mocap] = Rotation.from_matrix(obj["rotations"][frame]).as_quat(
                        scalar_first=True)
                mujoco.mj_forward(model, data)
                deep = []
                for source_geom, geom in collision.items():
                    depth = 0.0 if run["depths"] is None else float(run["depths"][frame, source_geom])
                    if depth > RED_M:
                        model.geom_rgba[geom] = [.95, .1, .1, .9]
                        deep.append(model.body(int(model.geom_bodyid[geom])).name)
                    elif depth > ORANGE_M:
                        model.geom_rgba[geom] = [1.0, .55, .05, .85]
                    else:
                        model.geom_rgba[geom] = [1.0, .45, .1, 0.0]
                camera.lookat[:] = lookat[min(frame, count - 1)]
                views = []
                # Two opposite sides, so neither arm hides behind the body.
                for turn in (0.0, 180.0):
                    camera.azimuth = base_azimuth + turn
                    renderer.update_scene(data, camera=camera)
                    if parents is not None:
                        draw_source(renderer.scene, run["source"][frame], parents)
                    views.append(renderer.render().copy())
                pixels = np.concatenate(views, axis=1)
                static = 0.0 if run["static"] is None else float(run["static"][frame].max()) * 1000
                moving = 0.0 if run["moving"] is None else float(run["moving"][frame].max()) * 1000
                alarm = (190, 20, 20) if max(static, moving) > RED_M * 1000 else (40, 40, 40)
                tiles.append(caption(pixels, [
                    (f"{label}   frame {frame}  t={frame / run['framerate']:.2f}s", (10, 10, 10)),
                    (f"max penetration  static {static:.1f} mm   object {moving:.1f} mm", alarm),
                    ("> 1 cm: " + (", ".join(sorted(set(deep)))[:70] if deep else "none"), alarm),
                ]))
            writer.append_data(np.concatenate(tiles, axis=0))
    finally:
        writer.close()
        renderer.close()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="label=clip_directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--azimuth", type=float)
    parser.add_argument("--distance", type=float, default=2.6)
    parser.add_argument("--elevation", type=float, default=-18.0)
    parser.add_argument("--frames", help="start:stop")
    args = parser.parse_args()
    labels, runs = [], []
    for item in args.run:
        label, path = item.split("=", 1)
        labels.append(label)
        runs.append(load_run(path))
    frames = None
    if args.frames:
        start, stop = (int(v) for v in args.frames.split(":"))
        frames = range(start, stop)
    path = render(runs, labels, args.output, azimuth=args.azimuth, distance=args.distance,
                  elevation=args.elevation, frames=frames)
    print(path, flush=True)


if __name__ == "__main__":
    main()
