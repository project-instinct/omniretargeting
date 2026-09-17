"""Visualize an SMPL-X body mesh together with a terrain mesh.

Reconstructs per-frame SMPL-X mesh vertices from a raw AMASS-style ``.npz``
motion file and renders the body surface in the same 3D axes as a terrain
mesh. The script is intentionally standalone: it does not depend on the robot
config or the retargeting pipeline.

Example
-------
.. code-block:: bash

    python -m omniretargeting.data_sources.smplx_visualize \\
        --smplx-model-dir /home/ziwen/Datasets/smplx \\
        --motion motion.npz \\
        --terrain terrain.obj \\
        --frame 0 \\
        --output /tmp/smplx_terrain.png
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# SMPL-X body frame: +X left, +Y up, +Z forward.
# Repo/world frame used by the terrain meshes: +X forward, +Y left, +Z up.
# This permutation maps (X_world, Y_world, Z_world) = (Z_smplx, X_smplx, Y_smplx).
SMPLX_TO_WORLD_AXES = (2, 0, 1)

DEFAULT_AZIMUTH = -60.0
DEFAULT_ELEVATION = 20.0
DEFAULT_SOURCE_COLOR = (0.80, 0.66, 0.55)  # skin-like
DEFAULT_TERRAIN_COLOR = (0.62, 0.62, 0.66)  # light gray


def _require_optional_libs():
    """Import heavy dependencies lazily so ``--help`` works without them."""
    try:
        import smplx
        import torch
        import trimesh
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise SystemExit(
            "This visualization requires the 'smplx', 'torch', and 'trimesh' "
            f"packages (see the omniretargeting conda environment). Missing: {exc}"
        ) from exc
    return smplx, torch, trimesh


def resolve_smplx_model_dir(model_dir: Path) -> Path:
    """Resolve *model_dir* to the directory containing ``SMPLX_*.npz`` files."""
    model_dir = model_dir.expanduser().resolve()
    candidates = [model_dir, model_dir / "smplx"]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("SMPLX_*.npz")):
            return candidate
    raise FileNotFoundError(
        f"No SMPL-X model files (SMPLX_*.npz) found under {model_dir}. "
        "Pass the directory that contains them (or its parent)."
    )


def _detect_framerate(motion: np.lib.npyio.NpzFile) -> float:
    for key in ("mocap_frame_rate", "framerate", "mocap_framerate"):
        if key in motion:
            value = np.asarray(motion[key]).reshape(-1)
            if value.size:
                return float(value[0])
    return 30.0


def _load_motion(motion_path: Path) -> dict:
    motion = np.load(motion_path, allow_pickle=True)

    def arr(*names):
        for name in names:
            if name in motion:
                return np.asarray(motion[name], dtype=np.float32)
        return None

    body_pose = arr("pose_body", "body_pose")
    root_orient = arr("root_orient", "global_orient")
    transl = arr("trans", "transl")
    if body_pose is None or root_orient is None or transl is None:
        raise ValueError(
            "Motion file must contain raw SMPL-X pose keys "
            "('pose_body'/'body_pose', 'root_orient'/'global_orient', "
            "'trans'/'transl'). Processed joint-only npz files cannot be "
            "used to reconstruct a body mesh."
        )

    num_frames = body_pose.shape[0]

    betas = arr("betas")
    if betas is None:
        betas = np.zeros((1, 10), dtype=np.float32)
    betas = betas.reshape(1, -1) if betas.ndim == 1 else betas[:1]

    hand = arr("pose_hand")
    if hand is not None and hand.ndim == 2 and hand.shape[1] >= 90:
        left_hand = hand[:, :45]
        right_hand = hand[:, 45:90]
    else:
        left_hand = arr("left_hand_pose")
        right_hand = arr("right_hand_pose")

    eye = arr("pose_eye")
    leye = eye[:, :3] if eye is not None and eye.shape[1] >= 6 else arr("leye_pose")
    reye = eye[:, 3:6] if eye is not None and eye.shape[1] >= 6 else arr("reye_pose")

    gender = str(motion["gender"]) if "gender" in motion else "neutral"

    return {
        "body_pose": body_pose,
        "root_orient": root_orient,
        "transl": transl,
        "betas": betas,
        "left_hand": left_hand,
        "right_hand": right_hand,
        "jaw": arr("pose_jaw", "jaw_pose"),
        "leye": leye,
        "reye": reye,
        "expression": arr("expression"),
        "num_frames": num_frames,
        "framerate": _detect_framerate(motion),
        "gender": gender,
    }


def _pose_tensor(arr, num_frames: int, dims: int):
    torch = _require_optional_libs()[1]
    if arr is not None and arr.shape[0] == num_frames:
        return torch.as_tensor(arr, dtype=torch.float32)
    return torch.zeros(num_frames, dims, dtype=torch.float32)


def _forward_frames(body_model, motion: dict, frame_indices, chunk_size: int = 64):
    """Return ``(vertices, faces)`` for the requested frames.

    Vertices are in SMPL-X world coordinates (Y-up), shape ``(N, V, 3)``.
    """
    _, torch, _ = _require_optional_libs()

    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    if frame_indices.size == 0:
        raise ValueError("No frames selected for visualization.")

    num_frames = motion["num_frames"]
    betas = torch.as_tensor(motion["betas"], dtype=torch.float32)
    body_pose = torch.as_tensor(motion["body_pose"], dtype=torch.float32)
    root_orient = torch.as_tensor(motion["root_orient"], dtype=torch.float32)
    transl = torch.as_tensor(motion["transl"], dtype=torch.float32)

    vertices_chunks = []
    with torch.no_grad():
        for start in range(0, frame_indices.size, chunk_size):
            idx = torch.as_tensor(frame_indices[start:start + chunk_size], dtype=torch.long)
            batch = idx.numel()
            out = body_model(
                betas=betas.expand(batch, -1),
                global_orient=root_orient[idx],
                body_pose=body_pose[idx],
                transl=transl[idx],
                left_hand_pose=_pose_tensor(motion["left_hand"], num_frames, 45)[idx],
                right_hand_pose=_pose_tensor(motion["right_hand"], num_frames, 45)[idx],
                jaw_pose=_pose_tensor(motion["jaw"], num_frames, 3)[idx],
                leye_pose=_pose_tensor(motion["leye"], num_frames, 3)[idx],
                reye_pose=_pose_tensor(motion["reye"], num_frames, 3)[idx],
                expression=_pose_tensor(motion["expression"], num_frames, 10)[idx],
                return_verts=True,
            )
            vertices_chunks.append(out.vertices.detach().cpu().numpy())

    faces = body_model.faces
    if hasattr(faces, "detach"):
        faces = faces.detach().cpu().numpy()
    faces = np.asarray(faces, dtype=np.int64)
    return np.concatenate(vertices_chunks, axis=0), faces


def _load_terrain(terrain_path: Path, max_faces: int | None):
    _, _, trimesh = _require_optional_libs()
    mesh = trimesh.load(str(terrain_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if hasattr(g, "faces") and len(g.faces) > 0]
        if not geoms:
            raise ValueError(f"Terrain file {terrain_path} contains no mesh geometry.")
        mesh = trimesh.util.concatenate(geoms)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ValueError(f"Terrain file {terrain_path} did not load as a mesh.")
    if max_faces and len(mesh.faces) > max_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(max_faces)
        except Exception:
            pass
    return mesh


def _convert_vertices(vertices: np.ndarray, coords: str) -> np.ndarray:
    if coords == "smplx":
        return vertices
    return vertices[..., SMPLX_TO_WORLD_AXES]


def _simplify_source(vertices: np.ndarray, faces: np.ndarray, max_faces: int | None):
    _, _, trimesh = _require_optional_libs()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    if max_faces and len(mesh.faces) > max_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(max_faces)
        except Exception:
            pass
    return mesh


def _set_axes_bounds(ax, points: list[np.ndarray]) -> None:
    mins = np.min([np.asarray(p).reshape(-1, 3).min(axis=0) for p in points], axis=0)
    maxs = np.max([np.asarray(p).reshape(-1, 3).max(axis=0) for p in points], axis=0)
    center = (mins + maxs) / 2.0
    radius = float((maxs - mins).max() / 2.0) or 0.5
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def _add_mesh_collection(ax, vertices: np.ndarray, faces: np.ndarray, color, alpha: float):
    _, _, trimesh = _require_optional_libs()
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    collection = Poly3DCollection(
        mesh.triangles,
        facecolor=color,
        edgecolor="none",
        alpha=alpha,
    )
    ax.add_collection3d(collection)
    return collection


def _render_static(vertices: np.ndarray, faces: np.ndarray, terrain, args) -> None:
    body = _simplify_source(vertices, faces, args.source_max_faces)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    _add_mesh_collection(ax, np.asarray(terrain.vertices), np.asarray(terrain.faces), DEFAULT_TERRAIN_COLOR, args.terrain_alpha)
    _add_mesh_collection(ax, np.asarray(body.vertices), np.asarray(body.faces), DEFAULT_SOURCE_COLOR, args.source_alpha)

    _set_axes_bounds(ax, [terrain.vertices, body.vertices])
    ax.view_init(elev=args.elevation, azim=args.azimuth)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"SMPL-X mesh + terrain (frame {args.frame})")

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved static visualization to {args.output}")
    else:
        plt.show()


def _render_animation(vertices_all: np.ndarray, faces: np.ndarray, terrain, args) -> None:
    from matplotlib.animation import FuncAnimation

    body = _simplify_source(vertices_all[0], faces, args.source_max_faces)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    _add_mesh_collection(ax, np.asarray(terrain.vertices), np.asarray(terrain.faces), DEFAULT_TERRAIN_COLOR, args.terrain_alpha)

    collection = Poly3DCollection([], facecolor=DEFAULT_SOURCE_COLOR, edgecolor="none", alpha=args.source_alpha)
    ax.add_collection3d(collection)

    # Use the shared body topology; only vertex positions change per frame.
    body_topology = body.copy()
    all_points = [terrain.vertices]
    for frame_vertices in vertices_all:
        all_points.append(frame_vertices)
    _set_axes_bounds(ax, all_points)
    ax.view_init(elev=args.elevation, azim=args.azimuth)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    def update(frame_idx: int):
        body_topology.vertices = vertices_all[frame_idx]
        collection.set_verts(body_topology.triangles)
        return (collection,)

    ani = FuncAnimation(fig, update, frames=len(vertices_all), interval=1000.0 / args.fps, blit=False)

    if args.output:
        suffix = Path(args.output).suffix.lower()
        if suffix == ".gif":
            ani.save(args.output, writer="pillow", fps=args.fps, dpi=100)
        elif suffix in (".mp4", ".mov"):
            ani.save(args.output, writer="ffmpeg", fps=args.fps, dpi=150)
        else:
            raise SystemExit("Animation output must end in .gif, .mp4, or .mov.")
        print(f"Saved animation ({len(vertices_all)} frames) to {args.output}")
    else:
        plt.show()


def _pv_faces(faces: np.ndarray) -> np.ndarray:
    """Convert an (N, 3) face array to PyVista's padded face format."""
    faces = np.asarray(faces, dtype=np.int64)
    return np.hstack((np.full((len(faces), 1), 3, dtype=np.int64), faces)).astype(np.int64, copy=False)


def _ask_open_motion() -> str | None:
    """Open the OS-native file dialog and return the selected motion path.

    On Linux this uses Zenity (GNOME) or KDialog (KDE) so the standard desktop
    file picker is shown. Tkinter is kept only as a fallback.
    """
    initial_dir = os.path.expanduser("~/Datasets")
    if not os.path.isdir(initial_dir):
        initial_dir = os.path.expanduser("~")

    if shutil.which("zenity"):
        result = subprocess.run(
            [
                "zenity",
                "--file-selection",
                "--title=Select SMPL-X motion file",
                "--file-filter=SMPL-X motion (*.npz) | *.npz",
                f"--filename={initial_dir}",
            ],
            capture_output=True,
            text=True,
        )
        path = result.stdout.strip()
        return path or None

    if shutil.which("kdialog"):
        result = subprocess.run(
            [
                "kdialog",
                "--getopenfilename",
                initial_dir,
                "*.npz",
                "--title",
                "Select SMPL-X motion file",
            ],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None

    # Fallback for systems without Zenity/KDialog (e.g. Windows/macOS).
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
        return filedialog.askopenfilename(
            title="Select SMPL-X motion file",
            initialdir=initial_dir,
            filetypes=[("SMPL-X motion (*.npz)", "*.npz"), ("All files", "*.*")],
        ) or None
    finally:
        root.destroy()


def _rest_pose_template(args):
    """Return ``(template_vertices, faces)`` for a neutral rest-pose placeholder."""
    model_dir = resolve_smplx_model_dir(args.smplx_model_dir)
    smplx, _, _ = _require_optional_libs()
    gender = args.gender or "neutral"
    model = smplx.SMPLX(str(model_dir), gender=gender, num_betas=10, use_pca=False)

    template = model.v_template
    if hasattr(template, "detach"):
        template = template.detach().cpu().numpy()
    template = _convert_vertices(np.asarray(template, dtype=np.float32), args.coords)

    faces = model.faces
    if hasattr(faces, "detach"):
        faces = faces.detach().cpu().numpy()
    faces = np.asarray(faces, dtype=np.int64)
    return template, faces


def _prepare_body(motion_path: Path, args):
    """Load a motion file and reconstruct body vertices in the chosen frame.

    Returns ``(vertices, faces, motion)``; ``vertices`` has shape ``(N, V, 3)``.
    """
    model_dir = resolve_smplx_model_dir(args.smplx_model_dir)
    motion = _load_motion(motion_path)
    smplx, _, _ = _require_optional_libs()
    gender = args.gender or motion["gender"]
    body_model = smplx.SMPLX(
        str(model_dir),
        gender=gender,
        num_betas=int(motion["betas"].shape[1]),
        use_pca=False,
    )
    frame_indices = _select_frames(motion, args)
    vertices_smplx, faces = _forward_frames(body_model, motion, frame_indices)

    vertices = _convert_vertices(vertices_smplx, args.coords)
    if args.scale != 1.0:
        vertices = vertices * args.scale
    if args.coords == "zup" and args.z_offset:
        vertices = vertices + np.array([0.0, 0.0, args.z_offset], dtype=vertices.dtype)
    return vertices, faces, motion


def _add_motion_picker(plotter, state: dict, body, slider, args) -> None:
    """Add a bottom-left button that opens a file dialog to load a motion."""
    holder = {}

    def reset_button() -> None:
        holder["busy"] = True
        holder["widget"].GetRepresentation().SetState(0)
        holder["busy"] = False

    def on_click(checked: bool) -> None:
        if holder.get("busy"):
            return
        if not checked:
            return

        path = _ask_open_motion()
        if not path:
            reset_button()
            return

        try:
            vertices, _faces, _motion = _prepare_body(Path(path), args)
        except Exception as exc:
            print(f"[select motion] failed to load {path}: {exc}")
            reset_button()
            return

        state["vertices"] = vertices
        state["frame"] = int(args.frame % len(vertices))
        state["playing"] = True
        body.points = np.ascontiguousarray(vertices[state["frame"]])
        body.compute_normals(inplace=True)

        rep = slider.GetRepresentation()
        rep.SetMinimumValue(0)
        rep.SetMaximumValue(len(vertices) - 1)
        rep.SetValue(state["frame"])

        print(f"[select motion] loaded {path} ({len(vertices)} frames)")
        plotter.render()
        reset_button()

    holder["widget"] = plotter.add_checkbox_button_widget(
        on_click,
        value=False,
        position=(10, 10),
        size=50,
        color_on="blue",
        color_off="grey",
        background_color="white",
    )
    plotter.add_text("Select motion file", position=(70, 18), font_size=10, viewport=False)


def _render_interactive(
    vertices_all: np.ndarray | None,
    faces: np.ndarray,
    terrain,
    args,
    initial_vertices: np.ndarray | None = None,
) -> None:
    """Open an interactive PyVista window: drag to orbit, scrub/play frames."""
    import pyvista as pv

    if initial_vertices is None:
        if vertices_all is not None:
            initial_vertices = vertices_all[0]
        else:
            initial_vertices = np.zeros((int(faces.max()) + 1, 3), dtype=np.float32)

    body = pv.PolyData(np.ascontiguousarray(initial_vertices), _pv_faces(faces))
    # Skip normals for an all-zero placeholder (degenerate triangles).
    if np.ptp(np.asarray(initial_vertices)) > 0:
        body.compute_normals(inplace=True)

    terrain_pd = pv.PolyData(
        np.ascontiguousarray(np.asarray(terrain.vertices, dtype=np.float64)),
        _pv_faces(np.asarray(terrain.faces)),
    )
    terrain_pd.compute_normals(inplace=True)

    plotter = pv.Plotter()
    plotter.add_mesh(
        terrain_pd,
        color=DEFAULT_TERRAIN_COLOR,
        opacity=args.terrain_alpha,
        name="terrain",
        show_edges=False,
        specular=0.1,
        diffuse=0.85,
    )
    plotter.add_mesh(
        body,
        color=DEFAULT_SOURCE_COLOR,
        opacity=args.source_alpha,
        name="body",
        smooth_shading=True,
        specular=0.2,
        diffuse=0.85,
    )
    plotter.add_text(
        "Drag: rotate | Shift+drag: pan | Scroll: zoom\n"
        "Space: play/pause | Slider: scrub | Q: quit",
        position="upper_left",
        font_size=10,
    )

    state = {
        "vertices": vertices_all,
        "frame": int(args.frame % len(vertices_all)) if vertices_all is not None else 0,
        "playing": True,
    }

    def show_frame(frame_idx: int) -> None:
        v = state["vertices"]
        if v is None or len(v) == 0:
            return
        frame_idx = int(round(frame_idx)) % len(v)
        state["frame"] = frame_idx
        body.points = np.ascontiguousarray(v[frame_idx])
        body.compute_normals(inplace=True)

    slider = plotter.add_slider_widget(
        show_frame,
        [0, max((len(state["vertices"]) - 1) if state["vertices"] is not None else 0, 0)],
        value=state["frame"],
        title="frame",
        pointa=(0.25, 0.94),
        pointb=(0.75, 0.94),
        interaction_event="always",
        fmt="%.0f",
    )

    def toggle_play():
        state["playing"] = not state["playing"]

    plotter.add_key_event("space", toggle_play)

    interval_ms = max(int(round(1000.0 / args.fps)), 16)

    def timer_cb(_step: int) -> None:
        v = state["vertices"]
        if state["playing"] and v is not None and len(v) > 0:
            next_frame = (state["frame"] + 1) % len(v)
            show_frame(next_frame)
            slider.GetRepresentation().SetValue(float(next_frame))
        plotter.render()

    plotter.add_timer_event(max_steps=10**7, duration=interval_ms, callback=timer_cb)

    if vertices_all is None:
        _add_motion_picker(plotter, state, body, slider, args)

    print("Interactive window opened: close it (or press Q) to finish.")
    plotter.show()


def _select_frames(motion: dict, args) -> list[int]:
    """Select frame indices for the requested render mode."""
    num_frames = motion["num_frames"]

    # The interactive PyVista viewer always scrubs/plays a frame range.
    multi_frame = bool(args.animate) or (args.output is None and args.viewer == "pyvista")
    if not multi_frame:
        return [args.frame % num_frames]

    step = max(int(args.frame_step), 1)
    indices = list(range(0, num_frames, step))
    if args.max_frames is not None and len(indices) > args.max_frames:
        indices = indices[: args.max_frames]
    return indices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize an SMPL-X body mesh together with a terrain mesh.",
    )
    parser.add_argument("--smplx-model-dir", "--smplx_model_dir", dest="smplx_model_dir",
                        required=True, type=Path,
                        help="Directory containing SMPLX_*.npz model files (or its parent).")
    parser.add_argument("--motion", type=Path, default=None,
                        help="Raw SMPL-X motion .npz (AMASS-style pose keys). "
                             "Optional in the interactive PyVista viewer: a button is shown "
                             "to select the file from a dialog.")
    parser.add_argument("--terrain", required=True, type=Path,
                        help="Terrain mesh file (.obj/.stl/.ply/.glb).")
    parser.add_argument("--frame", type=int, default=0,
                        help="Motion frame to show in static mode (default: 0).")
    parser.add_argument("--animate", action="store_true",
                        help="Animate all selected frames instead of showing one frame.")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="Frame stride for multi-frame rendering "
                             "(--animate or the interactive viewer, default: 1).")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum frames to load for --animate or the interactive "
                             "viewer (default: load all frames).")
    parser.add_argument("--fps", type=float, default=None,
                        help="Animation FPS (default: motion framerate, else 30).")
    parser.add_argument("--gender", default=None,
                        help="SMPL-X gender (default: read from the motion file, else neutral).")
    parser.add_argument("--coords", choices=("zup", "smplx"), default="zup",
                        help="SMPL-X mesh coordinate frame: 'zup' converts to +Z-up "
                             "world frame to match terrain (default), 'smplx' keeps native Y-up.")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Uniform scale applied to the SMPL-X mesh (default: 1.0).")
    parser.add_argument("--z-offset", type=float, default=0.0,
                        help="Vertical offset (meters) added to the SMPL-X mesh in Z-up mode.")
    parser.add_argument("--source-alpha", type=float, default=1.0,
                        help="Opacity of the SMPL-X body mesh (default: 1.0, fully opaque).")
    parser.add_argument("--terrain-alpha", type=float, default=0.75)
    parser.add_argument("--source-max-faces", type=int, default=None,
                        help="Decimate the SMPL-X mesh to this many faces for faster rendering.")
    parser.add_argument("--terrain-max-faces", type=int, default=None,
                        help="Decimate the terrain mesh to this many faces for faster rendering.")
    parser.add_argument("--azimuth", type=float, default=DEFAULT_AZIMUTH)
    parser.add_argument("--elevation", type=float, default=DEFAULT_ELEVATION)
    parser.add_argument("--viewer", choices=("pyvista", "matplotlib"), default="pyvista",
                        help="Interactive viewer used when --output is omitted. "
                             "'pyvista' (default) opens a window you can drag and scrub; "
                             "'matplotlib' uses the static 3D axes.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path: .png for a static image, .gif/.mp4/.mov for animation.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.output:
        matplotlib.use("Agg", force=True)

    if args.motion is None and (args.output is not None or args.viewer == "matplotlib"):
        raise SystemExit("--motion is required for --output and --viewer matplotlib.")

    terrain = _load_terrain(args.terrain, args.terrain_max_faces)

    # Interactive PyVista mode may start without a motion and load it on demand.
    if args.viewer == "pyvista" and args.output is None:
        if args.motion is None:
            initial_vertices, faces = _rest_pose_template(args)
            vertices = None
            args.fps = float(args.fps if args.fps else 30.0)
        else:
            vertices, faces, motion = _prepare_body(args.motion, args)
            initial_vertices = vertices[0]
            args.fps = float(args.fps if args.fps else motion["framerate"])
            print(f"Motion: {args.motion} ({motion['num_frames']} frames, {motion['framerate']:.2f} fps, gender={motion['gender']})")
        print(f"Terrain: {args.terrain} ({len(terrain.vertices)} vertices, {len(terrain.faces)} faces)")
        print(f"Rendering interactive viewer, coords={args.coords}, scale={args.scale}, z_offset={args.z_offset}")
        _render_interactive(vertices, faces, terrain, args, initial_vertices)
        return

    # Non-interactive paths require a motion file.
    vertices, faces, motion = _prepare_body(args.motion, args)
    args.fps = float(args.fps if args.fps else motion["framerate"])
    print(f"Model: {resolve_smplx_model_dir(args.smplx_model_dir)}")
    print(f"Motion: {args.motion} ({motion['num_frames']} frames, {motion['framerate']:.2f} fps, gender={motion['gender']})")
    print(f"Terrain: {args.terrain} ({len(terrain.vertices)} vertices, {len(terrain.faces)} faces)")
    print(f"Rendering {len(vertices)} frame(s), coords={args.coords}, scale={args.scale}, z_offset={args.z_offset}")

    if args.output:
        if args.animate:
            _render_animation(vertices, faces, terrain, args)
        else:
            _render_static(vertices[0], faces, terrain, args)
    elif args.animate:
        _render_animation(vertices, faces, terrain, args)
    else:
        _render_static(vertices[0], faces, terrain, args)


if __name__ == "__main__":
    main()
