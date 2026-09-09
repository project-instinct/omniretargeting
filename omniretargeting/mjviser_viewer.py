"""mjviser playback for OmniRetargeting outputs."""

from __future__ import annotations

import argparse
import threading
import time
from functools import wraps
from pathlib import Path

import mujoco
import numpy as np
import trimesh
import viser
from mjviser import ViserMujocoScene

from omniretargeting.main import DEFAULT_ROBOT_CONFIG_PATH
from omniretargeting.robot_config import load_robot_config
from omniretargeting.viser_playback import (
    FramePacer,
    PlaybackClock,
    recording_times,
    serve_recording,
    use_rigid_instance_updates,
    write_recording,
)


SUPPORTED_MOTION_SUFFIXES = (".npz", ".npy")

SMPLX_BODY_PARENTS = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    dtype=int,
)

PARC_MS_BODY_PARENTS = np.array(
    [-1, 0, 1, 1, 3, 4, 1, 6, 7, 0, 9, 10, 0, 12, 13],
    dtype=int,
)


def _parc_ms_body_forward(joints: np.ndarray) -> np.ndarray:
    """Recover PARC's +X body axis from pelvis, torso and hip positions."""
    joints = np.asarray(joints, dtype=float)
    up = joints[1] - joints[0]
    right = joints[9] - joints[12]
    up /= max(float(np.linalg.norm(up)), 1e-10)
    right /= max(float(np.linalg.norm(right)), 1e-10)
    forward = np.cross(up, right)
    return forward / max(float(np.linalg.norm(forward)), 1e-10)

OMOMO_OBJECT_SCALES = {
    "floorlamp": (1.55 * 0.3793, 1.55 * 0.3793, 1.55 * 0.3793),
    "largebox": (1.55 * 0.3486, 1.55 * 0.3486, 1.55 * 0.3486),
    "whitechair": (1.55 * 0.3129, 1.55 * 0.3129, 1.55 * 0.3129),
    "trashcan": (1.55 * 0.2326, 1.55 * 0.2326, 1.55 * 0.2326),
    "smalltable": (1.55 * 0.0162, 1.55 * 0.0162, 1.55 * 0.0162),
    "suitcase": (1.55 * 0.3672, 1.55 * 0.3672, 1.55 * 0.3672),
}


def _parse_scale(text: str) -> float | tuple[float, float, float]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if len(values) == 1:
        return values[0]
    if len(values) == 3:
        return values[0], values[1], values[2]
    raise argparse.ArgumentTypeError(f"Invalid scale '{text}'. Expected SCALE or X,Y,Z.")


def _resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _resolve_motion_file(path: Path) -> Path:
    if path.is_file():
        return path

    patterns = ("*retargeted*.npz", "*retargetted*.npz", "*.npz", "*.npy")
    for pattern in patterns:
        files = sorted(path.rglob(pattern))
        if files:
            return files[0]
    raise FileNotFoundError(f"No supported motion files found under {path}.")


def _continuous_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.array(quat, dtype=np.float64, copy=True)
    quat /= np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-12)
    for frame_idx in range(1, len(quat)):
        if np.dot(quat[frame_idx - 1], quat[frame_idx]) < 0.0:
            quat[frame_idx] *= -1.0
    return quat


def _model_scalar_joint_names(model: mujoco.MjModel) -> list[str]:
    joint_names: list[str] = []
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        if model.jnt_type[joint_id] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            raise ValueError(f"Unsupported non-scalar joint '{name}' in model.")
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_names.append(joint_name)
    return joint_names


def _free_joint_qpos_adr(model: mujoco.MjModel) -> int:
    free_joint_ids = [
        joint_id for joint_id in range(model.njnt) if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if len(free_joint_ids) != 1:
        raise ValueError(f"Expected exactly one free joint in the model, found {len(free_joint_ids)}.")
    return int(model.jnt_qposadr[free_joint_ids[0]])


def _payload_joint_names(payload: np.lib.npyio.NpzFile, joint_pos: np.ndarray, path: Path) -> tuple[list[str], np.ndarray]:
    if "joint_names" not in payload:
        return [], joint_pos

    raw_joint_names = payload["joint_names"].tolist()
    joint_names = [raw_joint_names] if isinstance(raw_joint_names, str) else list(raw_joint_names)
    joint_names = [name.decode() if isinstance(name, bytes) else str(name) for name in joint_names]

    if joint_names and joint_names[0] == "floating_base_joint":
        if len(joint_names) == joint_pos.shape[1] + 1:
            joint_names = joint_names[1:]
        elif len(joint_names) == joint_pos.shape[1]:
            joint_names = joint_names[1:]
            joint_pos = joint_pos[:, 1:]

    if len(joint_names) != joint_pos.shape[1]:
        raise ValueError(f"{path}: joint_names has {len(joint_names)} entries, joint_pos has {joint_pos.shape[1]} columns.")
    return joint_names, joint_pos


def _build_qpos_from_components(
    model: mujoco.MjModel,
    motion_file: Path,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
    joint_pos: np.ndarray,
    joint_names: list[str],
    root_z_offset: float,
) -> np.ndarray:
    if base_pos.shape[0] != base_quat.shape[0] or base_pos.shape[0] != joint_pos.shape[0]:
        raise ValueError(f"{motion_file}: base and joint trajectories have different frame counts.")

    model_joint_names = _model_scalar_joint_names(model)
    if joint_names:
        missing = [name for name in model_joint_names if name not in joint_names]
        if missing:
            raise ValueError(f"{motion_file}: missing joints required by the model: {missing}")
        joint_order = [joint_names.index(name) for name in model_joint_names]
    else:
        if joint_pos.shape[1] != len(model_joint_names):
            raise ValueError(
                f"{motion_file}: joint_names is missing and joint_pos has {joint_pos.shape[1]} columns; "
                f"model expects {len(model_joint_names)} scalar joints."
            )
        joint_order = list(range(len(model_joint_names)))

    base_pos = np.array(base_pos, dtype=np.float64, copy=True)
    base_pos[:, 2] += root_z_offset
    base_quat = _continuous_quat(base_quat)

    qpos = np.tile(model.qpos0, (base_pos.shape[0], 1))
    free_adr = _free_joint_qpos_adr(model)
    qpos[:, free_adr : free_adr + 3] = base_pos
    qpos[:, free_adr + 3 : free_adr + 7] = base_quat
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        qpos_adr = int(model.jnt_qposadr[joint_id])
        qpos[:, qpos_adr] = joint_pos[:, joint_order[model_joint_names.index(joint_name)]]
    return qpos


def _load_motion(
    model: mujoco.MjModel,
    motion_file: Path,
    root_z_offset: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float]:
    if motion_file.suffix == ".npy":
        qpos = np.asarray(np.load(motion_file, allow_pickle=True), dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] != model.nq:
            raise ValueError(f"{motion_file}: .npy playback expects shape (T, {model.nq}), got {qpos.shape}.")
        return qpos, None, None, 30.0

    with np.load(motion_file, allow_pickle=True) as payload:
        framerate = float(np.asarray(payload["framerate"]).item()) if "framerate" in payload else 30.0
        object_pos = np.asarray(payload["object_pos_w"], dtype=np.float64) if "object_pos_w" in payload else None
        object_quat = (
            _continuous_quat(np.asarray(payload["object_quat_w"], dtype=np.float64))
            if "object_quat_w" in payload
            else None
        )

        if "qpos" in payload:
            qpos = np.asarray(payload["qpos"], dtype=np.float64)
            if qpos.ndim != 2 or qpos.shape[1] != model.nq:
                raise ValueError(f"{motion_file}: qpos expects shape (T, {model.nq}), got {qpos.shape}.")
            return qpos, object_pos, object_quat, framerate

        joint_pos = np.asarray(payload["joint_pos"], dtype=np.float64)
        joint_names, joint_pos = _payload_joint_names(payload, joint_pos, motion_file)
        qpos = _build_qpos_from_components(
            model=model,
            motion_file=motion_file,
            base_pos=np.asarray(payload["base_pos_w"], dtype=np.float64),
            base_quat=np.asarray(payload["base_quat_w"], dtype=np.float64),
            joint_pos=joint_pos,
            joint_names=joint_names,
            root_z_offset=root_z_offset,
        )

    if object_pos is not None and object_pos.shape[0] != qpos.shape[0]:
        raise ValueError(f"{motion_file}: object_pos_w frame count does not match qpos.")
    if object_quat is not None and object_quat.shape[0] != qpos.shape[0]:
        raise ValueError(f"{motion_file}: object_quat_w frame count does not match qpos.")
    return qpos, object_pos, object_quat, framerate


def _infer_omomo_object_mesh(
    motion_file: Path,
    object_dir: Path,
) -> tuple[Path | None, float | tuple[float, float, float]]:
    if not object_dir.exists():
        return None, 1.0

    tokens = motion_file.stem.split("_")
    for token in tokens:
        exact = object_dir / f"{token}_cleaned_simplified.obj"
        if exact.exists():
            return exact, OMOMO_OBJECT_SCALES.get(token, 1.0)
        matches = sorted(object_dir.glob(f"{token}_cleaned_simplified*.obj"))
        if matches:
            return matches[0], OMOMO_OBJECT_SCALES.get(token, 1.0)
    return None, 1.0


def _add_object(
    server: viser.ViserServer,
    object_mesh: Path | None,
    object_scale: float | tuple[float, float, float],
):
    if object_mesh is None:
        return None
    if not object_mesh.exists():
        print(f"[WARN] Object mesh missing: {object_mesh}")
        return None
    mesh = trimesh.load(object_mesh, force="mesh")
    return server.scene.add_mesh_trimesh(
        "/omomo_object",
        mesh,
        scale=object_scale,
        wxyz=(1.0, 0.0, 0.0, 0.0),
        position=(0.0, 0.0, 0.0),
    )


def _resolve_object_mesh(
    args: argparse.Namespace,
    motion_file: Path,
) -> tuple[Path | None, float | tuple[float, float, float]]:
    if args.object_mesh:
        return _resolve_path(args.object_mesh), 1.0 if args.object_scale is None else args.object_scale

    object_mesh, object_scale = _infer_omomo_object_mesh(motion_file, _resolve_path(args.omomo_object_dir))
    if args.object_scale is not None:
        object_scale = args.object_scale
    return object_mesh, object_scale


def _run_viewer(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    object_pos: np.ndarray | None,
    object_quat: np.ndarray | None,
    framerate: float,
    args: argparse.Namespace,
    motion_file: Path,
    terrain_mesh: trimesh.Trimesh | None = None,
    contact_names: list[str] | None = None,
    contact_mask: np.ndarray | None = None,
    contact_targets: np.ndarray | None = None,
    contact_normals: np.ndarray | None = None,
    contact_forwards: np.ndarray | None = None,
    effector_references: np.ndarray | None = None,
    effector_ups: np.ndarray | None = None,
    source_joints: np.ndarray | None = None,
    source_parents: np.ndarray | None = None,
    source_label: str = "source",
) -> None:
    data = mujoco.MjData(model)
    clock = PlaybackClock(
        len(qpos), framerate, now=time.perf_counter(), frame=args.start_frame,
        playing=not args.paused, speed=args.speed, loop=args.loop,
    )
    state_lock = threading.RLock()
    wake = threading.Event()

    def locked(callback):
        @wraps(callback)
        def call(*a, **kw):
            with state_lock:
                result = callback(*a, **kw)
            wake.set()
            return result
        return call

    duration = max(qpos.shape[0] - 1, 0) / framerate
    server = viser.ViserServer(port=args.port, label="OmniRetargeting mjviser playback")
    server.scene.configure_default_lights(cast_shadow=args.shadows)
    # World-space motions are often several metres away from the origin.  A
    # Viser client otherwise starts at the origin as well, which can place the
    # camera inside an opaque terrain mesh and make a valid scene look empty.
    # Aim new clients at the first robot pose while keeping a useful oblique
    # overview; users can still orbit or enable camera tracking afterwards.
    initial_root = np.asarray(qpos[0, :3], dtype=float)
    server.initial_camera.look_at = tuple(initial_root + np.array([0.0, 0.0, -0.25]))
    server.initial_camera.position = tuple(initial_root + np.array([2.5, -2.5, 1.5]))
    scene = ViserMujocoScene(server, model, num_envs=1)
    rigid_groups = use_rigid_instance_updates(scene)
    print(f"[playback] {rigid_groups} singleton mesh groups use direct Viser node poses", flush=True)
    # mjviser defaults to shifting every robot body by -root_position so the
    # tracked body stays at the origin.  Custom world-space overlays (terrain,
    # SMPL-X, contact targets) are not shifted, which makes the robot appear
    # frozen and spatially misregistered.  World-coordinate playback must keep
    # this disabled; the browser camera can still be moved independently.
    scene.camera_tracking_enabled = bool(args.track_robot)

    # MuJoCo puts URDF <collision> primitives and <visual> meshes in separate
    # geom groups, and mjviser shows groups 0-2 by default, so the collision
    # capsules/cylinders are drawn on top of the robot's visual shells.  Hide
    # the collision-only groups whenever visual meshes are available; the
    # "Groups" GUI tab can toggle them back on per group.
    if not args.show_collision:
        mesh_groups = {
            int(model.geom_group[geom_id])
            for geom_id in range(model.ngeom)
            if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
        }
        if mesh_groups:
            for group_id in range(len(scene.geom_groups_visible)):
                scene.geom_groups_visible[group_id] = group_id in mesh_groups
            scene._sync_visibilities()

    terrain_handle = None
    if terrain_mesh is not None:
        terrain_handle = server.scene.add_mesh_simple(
            "/terrain",
            vertices=np.asarray(terrain_mesh.vertices),
            faces=np.asarray(terrain_mesh.faces),
            color=(155, 165, 175),
            opacity=args.terrain_opacity,
            side="double",
            cast_shadow=args.shadows,
            receive_shadow=args.shadows,
        )

    contact_colors = {
        "left_hand": (255, 196, 0),
        "right_hand": (255, 92, 64),
        "left_foot": (0, 210, 255),
        "right_foot": (64, 112, 255),
    }
    contact_handles = []
    contact_axis_handles = []
    if (
        contact_names is not None
        and contact_mask is not None
        and contact_targets is not None
    ):
        for effector_idx, name in enumerate(contact_names):
            handle = server.scene.add_icosphere(
                f"/contact_targets/{name}",
                radius=0.035,
                color=contact_colors.get(name, (255, 0, 255)),
                position=contact_targets[0, effector_idx],
                visible=bool(contact_mask[0, effector_idx]),
            )
            contact_handles.append(handle)
            axis_handles = (None, None)
            orientation_ups = (
                effector_ups if effector_ups is not None else contact_normals
            )
            if orientation_ups is not None and contact_forwards is not None:
                normal = orientation_ups[0, effector_idx]
                forward = contact_forwards[0, effector_idx]
                normal_norm = float(np.linalg.norm(normal))
                if normal_norm > 1e-8 and np.all(np.isfinite(forward)):
                    normal = normal / normal_norm
                    forward = forward - normal * float(np.dot(normal, forward))
                    forward_norm = float(np.linalg.norm(forward))
                    if forward_norm > 1e-8:
                        forward = forward / forward_norm
                        axis_origin = (
                            effector_references[0, effector_idx]
                            if effector_references is not None
                            else contact_targets[0, effector_idx]
                        )
                        axis_visible = name.endswith("_hand") or bool(
                            contact_mask[0, effector_idx]
                        )
                        forward_handle = server.scene.add_line_segments(
                            f"/smplx_effector_frame/{name}_forward",
                            points=np.array(
                                [[axis_origin, axis_origin + 0.16 * forward]]
                            ),
                            colors=(30, 255, 80),
                            line_width=5.0,
                            visible=axis_visible,
                        )
                        normal_handle = server.scene.add_line_segments(
                            f"/smplx_effector_frame/{name}_palm_normal",
                            points=np.array(
                                [[axis_origin, axis_origin + 0.12 * normal]]
                            ),
                            colors=(255, 235, 40),
                            line_width=5.0,
                            visible=axis_visible,
                        )
                        axis_handles = (forward_handle, normal_handle)
            contact_axis_handles.append(axis_handles)

    human_points_handle = None
    human_bones_handle = None
    human_forward_handle = None
    human_forward_tip_handle = None
    human_edges = np.empty((0, 2), dtype=int)
    if source_parents is not None:
        human_edges = np.array(
            [
                [parent, child]
                for child, parent in enumerate(source_parents)
                if parent >= 0
            ],
            dtype=int,
        ).reshape(-1, 2)
    if source_joints is not None and not args.no_human:
        human_points_handle = server.scene.add_point_cloud(
            "/source_motion/joints",
            points=source_joints[0],
            colors=(40, 110, 255),
            point_size=0.035,
            point_shape="circle",
        )
        if len(human_edges):
            human_bones_handle = server.scene.add_line_segments(
                "/source_motion/bones",
                points=source_joints[0][human_edges],
                colors=(40, 110, 255),
                line_width=4.0,
            )
        if source_label == "PARC-MS" and source_joints.shape[1] >= 13:
            forward_origin = source_joints[0, 0]
            forward_tip = forward_origin + 0.30 * _parc_ms_body_forward(
                source_joints[0]
            )
            human_forward_handle = server.scene.add_line_segments(
                "/source_motion/body_forward",
                points=np.array([[forward_origin, forward_tip]]),
                colors=(255, 70, 40),
                line_width=7.0,
            )
            human_forward_tip_handle = server.scene.add_icosphere(
                "/source_motion/body_forward_tip",
                radius=0.045,
                color=(255, 70, 40),
                position=forward_tip,
            )

    object_mesh, object_scale = _resolve_object_mesh(args, motion_file)
    object_handle = _add_object(server, object_mesh, object_scale) if not args.no_object else None

    print(f"[omniretargeting-mjviser] motion: {motion_file}")
    print(f"[omniretargeting-mjviser] frames={len(qpos)}, framerate={framerate:.3f}, duration={duration:.3f}s")
    if contact_handles:
        print(
            "[omniretargeting-mjviser] contact targets: "
            + ", ".join(contact_names or [])
        )
    if human_points_handle is not None:
        print(
            "[omniretargeting-mjviser] "
            f"original {source_label} skeleton: blue"
        )
    if object_handle is not None:
        print(f"[omniretargeting-mjviser] object mesh: {object_mesh}, scale={object_scale}")
    elif object_pos is not None or object_quat is not None:
        print("[WARN] Motion contains object poses but no object mesh was resolved. Pass --object-mesh to show it.")

    frame_idx = [min(max(args.start_frame, 0), len(qpos) - 1)]
    needs_render = [True]
    show_contact_targets = [True]
    show_contact_axes = [True]

    tabs = scene.create_visualization_gui()
    with tabs.add_tab("Playback", icon=viser.Icon.PLAYER_PLAY):
        time_label = server.gui.add_html("")
        timeline = server.gui.add_slider(
            "Timeline",
            min=0,
            max=len(qpos) - 1,
            step=1,
            initial_value=frame_idx[0],
            hint="Pause, then drag to inspect an exact source frame.",
        )
        frame_input = server.gui.add_number(
            "Frame number",
            initial_value=frame_idx[0],
            min=0,
            max=len(qpos) - 1,
            step=1,
            hint=f"Enter an integer frame from 0 to {len(qpos) - 1}.",
        )
        navigation_btns = server.gui.add_button_group(
            "Single-frame navigation",
            options=["First", "Prev", "Next", "Last"],
            hint="Prev/Next pauses playback and moves by exactly one frame.",
        )
        step_input = server.gui.add_number(
            "Jump step",
            initial_value=10,
            min=1,
            max=max(len(qpos) - 1, 1),
            step=1,
            hint="Number of frames used by Back/Forward and PageUp/PageDown.",
        )
        jump_btns = server.gui.add_button_group(
            "Jump navigation",
            options=["Back", "Forward"],
        )
        play_btn = server.gui.add_button(
            "Pause" if clock.playing else "Play",
            icon=viser.Icon.PLAYER_PAUSE if clock.playing else viser.Icon.PLAYER_PLAY,
            hint="Play or pause (Space).",
        )
        speed_btns = server.gui.add_button_group(
            "Speed", options=["0.25x", "0.5x", "1x", "2x", "4x"]
        )
        loop_cb = server.gui.add_checkbox("Loop", initial_value=clock.loop)
        contact_event_btns = None
        contact_event_frames = np.empty(0, dtype=int)
        if contact_mask is not None and len(contact_mask):
            any_contact = np.any(contact_mask, axis=1)
            contact_changes = np.flatnonzero(
                np.r_[any_contact[0], any_contact[1:] != any_contact[:-1]]
            )
            if len(contact_changes):
                contact_event_frames = contact_changes
                contact_event_btns = server.gui.add_button_group(
                    "Contact events",
                    options=["Previous event", "Next event"],
                    hint="Jump to the previous/next frame where contact starts or ends.",
                )
        with server.gui.add_folder("Visible overlays", expand_by_default=False):
            human_cb = (
                server.gui.add_checkbox(
                    f"Original {source_label} skeleton",
                    initial_value=True,
                )
                if human_points_handle is not None
                else None
            )
            contacts_cb = (
                server.gui.add_checkbox(
                    "Contact targets",
                    initial_value=True,
                )
                if contact_handles
                else None
            )
            contact_axes_cb = (
                server.gui.add_checkbox(
                    "Hand/foot reference axes",
                    initial_value=True,
                )
                if any(handle is not None for pair in contact_axis_handles for handle in pair)
                else None
            )
            terrain_cb = (
                server.gui.add_checkbox("Terrain", initial_value=True)
                if terrain_handle is not None
                else None
            )
            object_cb = (
                server.gui.add_checkbox("Object", initial_value=True)
                if object_handle is not None
                else None
            )
        server.gui.add_markdown(
            "**Keyboard:** `Space` play/pause · `←`/`→` one frame · "
            "`PageUp`/`PageDown` jump · `Home`/`End` first/last frame"
        )

    # Precompute tangent-space displacements, preserving source sample poses.
    interpolation_steps = np.zeros((max(len(qpos) - 1, 0), model.nv))
    for idx in range(len(interpolation_steps)):
        mujoco.mj_differentiatePos(model, interpolation_steps[idx], 1.0, qpos[idx], qpos[idx + 1])

    def render_frame(frame: float, *, update_ui: bool = True) -> None:
        # Include overlays in mjviser's nested atomic context, so the robot
        # and source skeleton for this timestamp are submitted together.
        with server.atomic():
            _render_frame(frame, update_ui=update_ui)
        server.flush()

    def _render_frame(frame: float, *, update_ui: bool) -> None:
        frame = float(np.clip(frame, 0.0, len(qpos) - 1))
        idx = int(np.floor(frame))
        next_idx = min(idx + 1, len(qpos) - 1)
        alpha = frame - idx

        # Playback runs at the browser/update rate rather than holding every
        # 30 Hz motion sample for two 60 Hz refreshes.  MuJoCo's manifold
        # interpolation handles free/ball-joint quaternions as well as scalar
        # joints, avoiding the visible stop-go "jitter" of frame stepping.
        interpolated_qpos = qpos[idx].copy()
        if alpha > 1e-8 and next_idx != idx:
            mujoco.mj_integratePos(
                model, interpolated_qpos, interpolation_steps[idx], alpha
            )
        data.qpos[:] = interpolated_qpos
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        scene.update_from_mjdata(data)

        if object_handle is not None:
            if object_pos is not None:
                object_handle.position = (
                    (1.0 - alpha) * object_pos[idx]
                    + alpha * object_pos[next_idx]
                )
            if object_quat is not None:
                quat0 = np.asarray(object_quat[idx], dtype=float)
                quat1 = np.asarray(object_quat[next_idx], dtype=float)
                if float(np.dot(quat0, quat1)) < 0.0:
                    quat1 = -quat1
                quat = (1.0 - alpha) * quat0 + alpha * quat1
                object_handle.wxyz = quat / max(float(np.linalg.norm(quat)), 1e-10)

        for effector_idx, handle in enumerate(contact_handles):
            handle.visible = show_contact_targets[0] and bool(contact_mask[idx, effector_idx])
            origin = (
                (1.0 - alpha) * contact_targets[idx, effector_idx]
                + alpha * contact_targets[next_idx, effector_idx]
            )
            if handle.visible:
                handle.position = origin
            forward_handle, normal_handle = contact_axis_handles[effector_idx]
            if forward_handle is not None and normal_handle is not None:
                axis_visible = show_contact_axes[0] and (
                    contact_names[effector_idx].endswith("_hand")
                    or bool(contact_mask[idx, effector_idx])
                )
                forward_handle.visible = normal_handle.visible = axis_visible
                if not axis_visible:
                    continue
                orientation_ups = (
                    effector_ups if effector_ups is not None else contact_normals
                )
                normal = (
                    (1.0 - alpha) * orientation_ups[idx, effector_idx]
                    + alpha * orientation_ups[next_idx, effector_idx]
                )
                normal = normal / max(float(np.linalg.norm(normal)), 1e-10)
                forward = (
                    (1.0 - alpha) * contact_forwards[idx, effector_idx]
                    + alpha * contact_forwards[next_idx, effector_idx]
                )
                forward = forward - normal * float(np.dot(normal, forward))
                forward = forward / max(float(np.linalg.norm(forward)), 1e-10)
                axis_origin = origin
                if effector_references is not None:
                    axis_origin = (
                        (1.0 - alpha) * effector_references[idx, effector_idx]
                        + alpha * effector_references[next_idx, effector_idx]
                    )
                forward_handle.points = np.array(
                    [[axis_origin, axis_origin + 0.16 * forward]]
                )
                normal_handle.points = np.array(
                    [[axis_origin, axis_origin + 0.12 * normal]]
                )

        if human_points_handle is not None and human_points_handle.visible:
            human_points = (
                (1.0 - alpha) * source_joints[idx]
                + alpha * source_joints[next_idx]
            )
            human_points_handle.points = human_points
            if human_bones_handle is not None:
                human_bones_handle.points = human_points[human_edges]
            if (
                human_forward_handle is not None
                and human_forward_tip_handle is not None
            ):
                forward_origin = human_points[0]
                forward_tip = forward_origin + 0.30 * _parc_ms_body_forward(
                    human_points
                )
                human_forward_handle.points = np.array(
                    [[forward_origin, forward_tip]]
                )
                human_forward_tip_handle.position = forward_tip

        if scene.needs_update:
            scene.refresh_visualization()

        if not update_ui:
            return
        timeline.value = int(np.floor(frame))
        frame_input.value = int(np.floor(frame))
        t = frame / framerate
        time_label.content = (
            "<div style='font-family: ui-monospace, SFMono-Regular, Menlo, "
            "monospace; font-variant-numeric: tabular-nums;'>"
            f"<strong>Frame {int(round(frame)):04d}</strong> / "
            f"{len(qpos) - 1:04d}&nbsp;&nbsp;·&nbsp;&nbsp;"
            f"{t:06.2f}s / {duration:06.2f}s"
            "</div>"
        )

    def update_play_button() -> None:
        play_btn.label = "Pause" if clock.playing else "Play"
        play_btn.icon = (
            viser.Icon.PLAYER_PAUSE if clock.playing else viser.Icon.PLAYER_PLAY
        )

    @locked
    def seek_frame(target: float) -> None:
        """Seek to an exact integer frame and request one render."""
        target_idx = int(np.clip(round(float(target)), 0, len(qpos) - 1))
        clock.seek(target_idx, time.perf_counter())
        update_play_button()
        frame_idx[0] = target_idx
        timeline.value = target_idx
        frame_input.value = target_idx
        needs_render[0] = True

    @locked
    def toggle_playback() -> None:
        clock.toggle(time.perf_counter())
        needs_render[0] = True
        update_play_button()

    @play_btn.on_click
    def _(_) -> None:
        toggle_playback()

    @navigation_btns.on_click
    def _(event) -> None:
        target = {
            "First": 0,
            "Prev": frame_idx[0] - 1,
            "Next": frame_idx[0] + 1,
            "Last": len(qpos) - 1,
        }[event.target.value]
        seek_frame(target)

    @jump_btns.on_click
    def _(event) -> None:
        direction = -1 if event.target.value == "Back" else 1
        seek_frame(frame_idx[0] + direction * int(step_input.value))

    @speed_btns.on_click
    @locked
    def _(event) -> None:
        clock.set_speed(float(event.target.value.replace("x", "")), time.perf_counter())

    @loop_cb.on_update
    @locked
    def _(_) -> None:
        clock.sample(time.perf_counter())
        clock.loop = loop_cb.value

    @timeline.on_update
    def _(event) -> None:
        # Viser invokes callbacks for both browser edits and server-side
        # ``.value = ...`` synchronization.  Playback updates the slider every
        # frame, so only a real client event is allowed to seek/pause.
        if event.client is None:
            return
        seek_frame(timeline.value)

    @frame_input.on_update
    def _(event) -> None:
        # Ignore the frame number written by the playback loop itself.  Without
        # this guard the synchronized number field calls ``seek_frame()``,
        # which immediately pauses a newly started playback.
        if event.client is None:
            return
        seek_frame(frame_input.value)

    if contact_event_btns is not None:

        @contact_event_btns.on_click
        def _(event) -> None:
            if event.target.value == "Previous event":
                candidates = contact_event_frames[
                    contact_event_frames < frame_idx[0]
                ]
                target = candidates[-1] if len(candidates) else contact_event_frames[0]
            else:
                candidates = contact_event_frames[
                    contact_event_frames > frame_idx[0]
                ]
                target = candidates[0] if len(candidates) else contact_event_frames[-1]
            seek_frame(target)

    if human_cb is not None:

        @human_cb.on_update
        @locked
        def _(_) -> None:
            visible = bool(human_cb.value)
            human_points_handle.visible = visible
            if human_bones_handle is not None:
                human_bones_handle.visible = visible
            if human_forward_handle is not None:
                human_forward_handle.visible = visible
            if human_forward_tip_handle is not None:
                human_forward_tip_handle.visible = visible
            needs_render[0] = True

    if contacts_cb is not None:

        @contacts_cb.on_update
        @locked
        def _(_) -> None:
            show_contact_targets[0] = bool(contacts_cb.value)
            needs_render[0] = True

    if contact_axes_cb is not None:

        @contact_axes_cb.on_update
        @locked
        def _(_) -> None:
            show_contact_axes[0] = bool(contact_axes_cb.value)
            needs_render[0] = True

    if terrain_cb is not None:

        @terrain_cb.on_update
        def _(_) -> None:
            terrain_handle.visible = bool(terrain_cb.value)

    if object_cb is not None:

        @object_cb.on_update
        def _(_) -> None:
            object_handle.visible = bool(object_cb.value)

    play_command = server.gui.add_command(
        "Play / pause",
        description="Toggle motion playback.",
        hotkey="space",
        icon=viser.Icon.PLAYER_PLAY,
    )
    previous_command = server.gui.add_command(
        "Previous frame",
        description="Pause and move backward by one exact frame.",
        hotkey="arrowleft",
        icon=viser.Icon.PLAYER_TRACK_PREV,
    )
    next_command = server.gui.add_command(
        "Next frame",
        description="Pause and move forward by one exact frame.",
        hotkey="arrowright",
        icon=viser.Icon.PLAYER_TRACK_NEXT,
    )
    first_command = server.gui.add_command(
        "First frame",
        hotkey="home",
        icon=viser.Icon.PLAYER_SKIP_BACK,
    )
    last_command = server.gui.add_command(
        "Last frame",
        hotkey="end",
        icon=viser.Icon.PLAYER_SKIP_FORWARD,
    )
    jump_back_command = server.gui.add_command(
        "Jump backward",
        description="Move backward by the configured jump step.",
        hotkey="pageup",
        icon=viser.Icon.CHEVRONS_LEFT,
    )
    jump_forward_command = server.gui.add_command(
        "Jump forward",
        description="Move forward by the configured jump step.",
        hotkey="pagedown",
        icon=viser.Icon.CHEVRONS_RIGHT,
    )

    play_command.on_trigger(lambda _: toggle_playback())
    previous_command.on_trigger(lambda _: seek_frame(frame_idx[0] - 1))
    next_command.on_trigger(lambda _: seek_frame(frame_idx[0] + 1))
    first_command.on_trigger(lambda _: seek_frame(0))
    last_command.on_trigger(lambda _: seek_frame(len(qpos) - 1))
    jump_back_command.on_trigger(
        lambda _: seek_frame(frame_idx[0] - int(step_input.value))
    )
    jump_forward_command.on_trigger(
        lambda _: seek_frame(frame_idx[0] + int(step_input.value))
    )

    cached_server = None
    render_frame(0 if args.playback_dir else frame_idx[0])
    try:
        if args.playback_dir:
            started = time.perf_counter()
            serializer = server.get_scene_serializer()
            times = recording_times(len(qpos), framerate, args.refresh_hz)
            previous = 0.0
            for timestamp in times[1:]:
                serializer.insert_sleep(float(timestamp - previous))
                render_frame(float(timestamp * framerate), update_ui=False)
                previous = timestamp
            if len(qpos) > 1:
                serializer.insert_sleep(1.0 / framerate)
            payload = serializer.serialize()
            manifest = write_recording(
                Path(args.playback_dir), payload,
                camera_position=server.initial_camera.position,
                camera_look_at=server.initial_camera.look_at,
                metadata={"source": str(motion_file), "source_frames": len(qpos),
                          "source_fps": framerate, "display_hz": args.refresh_hz,
                          "samples": len(times), "motion_duration_s": duration,
                          "loop_duration_s": len(qpos) / framerate if len(qpos) > 1 else 0,
                          "shadows": args.shadows},
            )
            print(f"[cached-playback] {len(times)} samples, {manifest['bytes'] / 1e6:.2f} MB, "
                  f"prepared in {time.perf_counter() - started:.2f}s", flush=True)
            if args.playback_port is not None:
                cached_server = serve_recording(Path(args.playback_dir), args.playback_port)
                server.gui.add_markdown(
                    f"**Smooth cached playback:** open this server's address on port "
                    f"**{args.playback_port}**. Download once, then play locally in your browser. "
                    "Use this live viewer for exact-frame and overlay inspection."
                )
                print(f"[cached-playback] http://localhost:{args.playback_port}/", flush=True)
            if args.export_only:
                return
            render_frame(frame_idx[0])

        clock.rebase(time.perf_counter())
        pacer = FramePacer(args.refresh_hz, time.perf_counter())
        last_ui = 0.0
        last_rendered = None
        stats_start = time.perf_counter()
        render_costs = []

        @server.on_client_connect
        def _(_client):
            wake.set()

        while True:
            now = time.perf_counter()
            with state_lock:
                connected = bool(server.get_clients())
                if connected:
                    frame = clock.sample(now)
                    frame_idx[0] = int(np.floor(frame))
                    if frame != last_rendered or needs_render[0] or scene.needs_update:
                        update_ui = now - last_ui >= 0.1 or needs_render[0] or not clock.playing
                        render_frame(frame, update_ui=update_ui)
                        if update_ui:
                            last_ui = now
                            update_play_button()
                        last_rendered = frame
                        needs_render[0] = False
                        render_costs.append(time.perf_counter() - now)
                else:
                    clock.rebase(now)
                active = connected and clock.playing and len(qpos) > 1
            if now - stats_start >= 5.0:
                if args.playback_stats and render_costs:
                    print(f"[playback] submitted_hz={len(render_costs) / (now - stats_start):.1f} "
                          f"render_ms_p95={np.percentile(render_costs, 95) * 1000:.2f} "
                          f"clients={len(server.get_clients())}", flush=True)
                render_costs.clear()
                stats_start = now
            delay = pacer.delay(time.perf_counter()) if active else 0.1
            wake.wait(delay)
            wake.clear()
    except KeyboardInterrupt:
        print("\n[omniretargeting-mjviser] stopped.")
    finally:
        if cached_server is not None:
            cached_server.shutdown()
            cached_server.server_close()
        server.stop()


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
