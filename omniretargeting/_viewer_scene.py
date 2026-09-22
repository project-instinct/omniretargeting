"""Interactive mjviser scene, GUI callbacks and recording loop."""

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

from omniretargeting.viser_playback import (
    FramePacer,
    PlaybackClock,
    recording_times,
    serve_recording,
    use_rigid_instance_updates,
    write_recording,
)

from ._viewer_motion import _add_object, _parc_ms_body_forward, _resolve_object_mesh


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
    server = viser.ViserServer(host=args.host, port=args.port,
                               label="OmniRetargeting mjviser playback")
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
                cached_server = serve_recording(Path(args.playback_dir), args.playback_port, host=args.host)
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
