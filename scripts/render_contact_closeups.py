#!/usr/bin/env python3
"""Off-screen closeups of hand/object contacts across benchmark runs."""

from __future__ import annotations

import argparse
import html
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


HAND_TOKENS = ("hand", "forearm")
BACK_THRESHOLD_M = -0.01


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def hand_binding_groups(plan: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Group object-contact bindings by effector and source body part."""
    parts = np.asarray(plan.get("surface_contact_parts", []), dtype=str)
    effectors = np.asarray(
        plan.get("surface_contact_effector_names", parts), dtype=str
    )
    # Surface-only forearm contacts have no endpoint owner. Keep their side
    # from the source part instead of silently selecting the right arm.
    effectors = np.where(effectors == "", parts, effectors)
    surface_ids = np.asarray(plan.get("surface_contact_surface_ids", []), dtype=int)
    result: list[dict[str, Any]] = []
    for effector in dict.fromkeys(effectors.tolist()):
        for part in dict.fromkeys(parts[effectors == effector].tolist()):
            low = part.lower()
            if not any(token in low for token in HAND_TOKENS):
                continue
            columns = np.flatnonzero(
                (effectors == effector) & (parts == part)
            ).astype(int)
            columns = columns[np.any(surface_ids[:, columns] >= 1, axis=0)]
            if len(columns):
                result.append({
                    "effector": str(effector), "part": str(part), "columns": columns,
                })
    return result


def material_points(
    model: Any, qpos: np.ndarray, geom_ids: np.ndarray, local: np.ndarray
) -> np.ndarray:
    """World material points for immutable bindings on one robot model."""
    import mujoco

    data = mujoco.MjData(model)
    qpos = np.asarray(qpos, dtype=float)
    local = np.asarray(local, dtype=float)
    geom_ids = np.asarray(geom_ids, dtype=int)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError("qpos must have shape (frames, model.nq)")
    if geom_ids.shape != local.shape[:1] or local.shape[-1] != 3:
        raise ValueError("geom/local binding arrays do not match")
    result = np.empty((len(qpos), len(geom_ids), 3), dtype=float)
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        result[frame] = data.geom_xpos[geom_ids] + np.einsum(
            "nij,nj->ni", data.geom_xmat[geom_ids].reshape(-1, 3, 3), local
        )
    return result


def _normal_values(
    plan: dict[str, np.ndarray], points: np.ndarray, columns: np.ndarray
) -> np.ndarray:
    targets = plan["surface_contact_environment_w"][:, columns]
    normals = plan["surface_contact_normals_w"][:, columns]
    return np.einsum("fsi,fsi->fs", points[:, columns] - targets, normals)


def binding_material_points(prepared: dict, motion: dict, plan: dict) -> np.ndarray:
    """Resolve saved geom IDs in their original robot, before adding scene geoms."""
    import mujoco

    robot = mujoco.MjModel.from_xml_path(prepared["model_path"])
    return material_points(robot, motion["qpos"], plan["surface_contact_robot_geoms"],
                           plan["surface_contact_robot_local_points"])


def _active(plan: dict[str, np.ndarray], columns: np.ndarray) -> np.ndarray:
    return np.any(
        plan["surface_contact_mask"][:, columns]
        & (plan["surface_contact_surface_ids"][:, columns] >= 1),
        axis=1,
    )


def select_frames(
    plans: list[dict[str, np.ndarray]], points_by_run: list[np.ndarray],
    *, max_frames: int = 8,
) -> list[dict[str, Any]]:
    """Select contact starts/middles and normal-error extremes across runs."""
    candidates: list[dict[str, Any]] = []
    groups = [hand_binding_groups(plan) for plan in plans]
    for run, (plan, points, run_groups) in enumerate(zip(plans, points_by_run, groups)):
        for group in run_groups:
            columns = group["columns"]
            active = np.flatnonzero(_active(plan, columns))
            if not len(active):
                continue
            values = _normal_values(plan, points, columns)
            worst = int(np.nanargmax(np.abs(values).max(axis=1)))
            back = int(np.nanargmin(values.min(axis=1)))
            items = [("worst_normal", worst, float(np.max(np.abs(values[worst]))))]
            if values[back].min() < BACK_THRESHOLD_M:
                items.append(("backside", back, -float(values[back].min())))
            if run == 0:
                start, stop = int(active[0]), int(active[-1])
                items.extend((
                    ("contact_start", start, 0.0),
                    ("contact_middle", (start + stop) // 2, 0.0),
                ))
            for reason, frame, severity in items:
                candidates.append({
                    "frame": frame, "reason": reason, "run": run,
                    "effector": group["effector"], "part": group["part"],
                    "columns": columns, "severity": severity,
                })
    priority = {"contact_start": 0, "backside": 1, "worst_normal": 2, "contact_middle": 3}
    candidates.sort(key=lambda row: (
        priority[row["reason"]], -row["severity"], row["run"], row["frame"],
        row["effector"], row["part"],
    ))
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in candidates:
        if row["frame"] in seen:
            continue
        selected.append(row)
        seen.add(row["frame"])
        if len(selected) >= max_frames:
            break
    return sorted(selected, key=lambda row: row["frame"])


def frame_statistics(
    plan: dict[str, np.ndarray], points: np.ndarray, columns: np.ndarray,
    frame: int, moving_depths: np.ndarray | None = None,
) -> dict[str, Any]:
    """Closeup caption statistics for one hand group and frame."""
    active = bool(np.any(
        plan["surface_contact_mask"][frame, columns]
        & (plan["surface_contact_surface_ids"][frame, columns] >= 1)
    ))
    selected = columns if active else np.array([], dtype=int)
    if len(selected):
        vectors = points[frame, selected] - plan["surface_contact_environment_w"][frame, selected]
        normal_rows = plan["surface_contact_normals_w"][frame, selected]
        values = np.einsum("si,si->s", vectors, normal_rows)
        tangents = vectors - values[:, None] * normal_rows
        geom_ids = np.unique(plan["surface_contact_robot_geoms"][selected]).astype(int)
        depth = (
            float(np.max(moving_depths[frame, geom_ids]))
            if moving_depths is not None and len(geom_ids)
            and frame < len(moving_depths)
            and np.all(geom_ids < moving_depths.shape[1]) else None
        )
        return {
            "active": True,
            "normal_median_mm": float(np.median(np.abs(values)) * 1000),
            "normal_worst_mm": float(np.max(np.abs(values)) * 1000),
            "backside_samples": int(np.count_nonzero(values < BACK_THRESHOLD_M)),
            "moving_depth_mm": None if depth is None else depth * 1000,
            "tangent_median_mm": float(np.median(np.linalg.norm(tangents, axis=1)) * 1000),
        }
    return {"active": False, "normal_median_mm": None, "normal_worst_mm": None,
            "backside_samples": 0, "moving_depth_mm": None, "tangent_median_mm": None}


def camera_azimuth(normal: np.ndarray) -> float:
    """Camera azimuth (degrees) looking along a contact normal's horizontal part."""
    normal = np.asarray(normal, dtype=float)
    length = math.hypot(float(normal[0]), float(normal[1]))
    return -50.0 if length < 1e-9 else math.degrees(math.atan2(
        float(normal[1]), float(normal[0])
    ))


def view_offsets(center: np.ndarray, pelvis: np.ndarray,
                 normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Free-side close-up direction, its 90-degree raised direction, and azimuth."""
    center, pelvis = np.asarray(center, float), np.asarray(pelvis, float)
    normal = np.asarray(normal, float)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    horizontal = normal[:2].copy()
    if float(np.linalg.norm(horizontal)) < 1e-6:
        horizontal = (center - pelvis)[:2]
    if float(np.linalg.norm(horizontal)) < 1e-6:
        horizontal[:] = [1.0, 0.0]
    horizontal /= np.linalg.norm(horizontal)
    side = np.array([horizontal[0], horizontal[1], .35])
    side /= np.linalg.norm(side)
    rotated = np.array([-horizontal[1], horizontal[0], math.sin(math.pi / 4)])
    rotated /= np.linalg.norm(rotated)
    azimuth = math.degrees(math.atan2(-side[1], -side[0]))
    return side, rotated, azimuth


def _choose_backend() -> str:
    import mujoco

    original = os.environ.get("MUJOCO_GL")
    for backend in ("egl", "osmesa"):
        os.environ["MUJOCO_GL"] = backend
        try:
            model = mujoco.MjModel.from_xml_string(
                "<mujoco><worldbody><geom type='sphere' size='0.01'/></worldbody></mujoco>"
            )
            renderer = mujoco.Renderer(model, height=16, width=16)
            renderer.close()
            return backend
        except Exception as error:
            print(f"MUJOCO_GL={backend} unavailable: {error}", flush=True)
            continue
        finally:
            if original is None:
                os.environ.pop("MUJOCO_GL", None)
            else:
                os.environ["MUJOCO_GL"] = original
    return "egl"


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


def _add_mesh(spec: Any, body: Any, name: str, mesh: Any, rgba: list[float]) -> None:
    import mujoco

    spec.add_mesh(
        name=name, uservert=np.asarray(mesh.vertices).ravel(),
        userface=np.asarray(mesh.faces).ravel(),
    )
    body.add_geom(
        name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
        rgba=rgba, group=0, contype=0, conaffinity=0,
    )


def _build_model(model_path: str, scene_mesh: Any, objects: list[dict[str, Any]]):
    import mujoco
    import trimesh

    spec = mujoco.MjSpec.from_file(model_path)
    _add_mesh(spec, spec.worldbody, "closeup_scene", scene_mesh, [0.86, 0.88, 0.9, 0.08])
    for index, obj in enumerate(objects):
        body = spec.worldbody.add_body(name=f"closeup_object_{index}", mocap=True)
        _add_mesh(spec, body, f"closeup_object_{index}", scaled_object_mesh(obj), [.95, .91, .84, .4])
    model = spec.compile()
    for geom in range(model.ngeom):
        if int(model.geom_group[geom]) == 2:
            model.geom_rgba[geom] = [.62, .70, .80, .12]
        elif int(model.geom_group[geom]) == 3:
            model.geom_rgba[geom] = [1.0, .42, .08, 0.0]
    model.vis.global_.offwidth = 720
    model.vis.global_.offheight = 540
    return model


def arm_body_ids(model: Any, effector: str) -> set[int]:
    """Bodies in one robot arm chain, from shoulder links through the hand."""
    side = "left" if "left" in effector.lower() else "right"
    result = set()
    for body in range(1, model.nbody):
        name = str(model.body(body).name).lower()
        if side in name and any(token in name for token in (
                "shoulder", "elbow", "wrist", "hand", "rubber")):
            result.add(int(model.body_weldid[body]))
    return result


def focus_robot(model: Any, arm_bodies: set[int]) -> None:
    """Dim the rest of the robot and show only the selected arm collision hull."""
    for geom in range(model.ngeom):
        body = int(model.body_weldid[int(model.geom_bodyid[geom])])
        group = int(model.geom_group[geom])
        if group == 2:
            model.geom_rgba[geom] = [.62, .70, .80, .8 if body in arm_bodies else .12]
        elif group == 3:
            model.geom_rgba[geom] = [1.0, .42, .08, .45 if body in arm_bodies else 0.0]


def _basis_x(direction: np.ndarray) -> np.ndarray:
    x = direction / max(float(np.linalg.norm(direction)), 1e-12)
    candidate = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < .9 else np.array([1.0, 0.0, 0.0])
    y = np.cross(x, candidate)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return np.column_stack((x, y, z))


def _add_scene_geom(scene: Any, geom_type: int, pos: np.ndarray, mat: np.ndarray,
                    size: np.ndarray, rgba: np.ndarray) -> None:
    import mujoco

    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, geom_type, size, pos, mat.ravel(), rgba)
    scene.ngeom += 1


def draw_contact_overlay(scene: Any, targets: np.ndarray, points: np.ndarray,
                         normals: np.ndarray) -> None:
    import mujoco

    center_target = np.mean(targets, axis=0)
    center_point = np.mean(points, axis=0)
    normal = np.mean(normals, axis=0)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    marker_target, marker_point = center_target, center_point
    for target, point, _sample_normal in zip(targets, points, normals):
        _add_scene_geom(
            scene, mujoco.mjtGeom.mjGEOM_SPHERE, target, np.eye(3),
            np.array([.003, .003, .003]), np.array([.35, .9, .35, .5]),
        )
        _add_scene_geom(
            scene, mujoco.mjtGeom.mjGEOM_SPHERE, point, np.eye(3),
            np.array([.003, .003, .003]), np.array([.95, .4, .4, .5]),
        )

    _add_scene_geom(
        scene, mujoco.mjtGeom.mjGEOM_SPHERE, marker_target, np.eye(3),
        np.array([.012, .012, .012]), np.array([.03, .95, .12, 1.0]),
    )
    _add_scene_geom(
        scene, mujoco.mjtGeom.mjGEOM_SPHERE, marker_point, np.eye(3),
        np.array([.010, .010, .010]), np.array([.98, .06, .06, 1.0]),
    )
    connector = scene.geoms[scene.ngeom] if scene.ngeom < scene.maxgeom else None
    if connector is not None:
        mujoco.mjv_initGeom(
            connector, mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array([.002, .002, .002]), (marker_target + marker_point) / 2,
            np.eye(3).ravel(), np.array([1.0, .87, .1, .95]),
        )
        mujoco.mjv_connector(
            connector, mujoco.mjtGeom.mjGEOM_CAPSULE, .002, marker_target, marker_point,
        )
        scene.ngeom += 1
    _add_scene_geom(
        scene, mujoco.mjtGeom.mjGEOM_ARROW, marker_target, _basis_x(normal),
        np.array([.08, .006, .006]), np.array([.06, .28, 1.0, 1.0]),
    )


def _draw_source(scene: Any, joints: np.ndarray, parents: np.ndarray,
                 keep: set[int] | None = None) -> None:
    import mujoco
    from omniretargeting._viewer_motion import source_body_parents

    if parents is None:
        parents = source_body_parents([str(i) for i in range(len(joints))])
    for child, parent in enumerate(parents):
        if keep is not None and child not in keep:
            continue
        if parent < 0 or child >= len(joints) or scene.ngeom >= scene.maxgeom:
            continue
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
            np.eye(3).ravel(), np.array([.08, .48, 1.0, .7]),
        )
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .007,
            joints[parent], joints[child],
        )
        scene.ngeom += 1


def _source_parents(names: np.ndarray) -> np.ndarray:
    from omniretargeting._viewer_motion import source_body_parents

    return source_body_parents([str(name) for name in names])


def _source_keep(names: np.ndarray, effector: str) -> set[int]:
    prefix = "L_" if "left" in effector.lower() else "R_"
    result = set()
    for index, name in enumerate(names):
        low = str(name).lower()
        if ("torso" in low or "spine" in low or "pelvis" in low or (
                str(name).startswith(prefix) and any(token in low for token in (
                    "shoulder", "elbow", "wrist", "hand")))):
            result.add(index)
    return result


def _caption_array(image: np.ndarray, lines: list[str]) -> np.ndarray:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        height = 28 + 22 * max(1, len(lines))
        caption = np.full((image.shape[0] + height, image.shape[1], 3), 255, dtype=np.uint8)
        caption[height:] = image
        for row in range(len(lines)):
            caption[8 + row * 22:24 + row * 22, 12:12 + min(360, image.shape[1])] = 35
        return caption

    canvas = Image.fromarray(image).convert("RGB")
    height = 28 + 22 * max(1, len(lines))
    bordered = Image.new("RGB", (canvas.width, canvas.height + height), "white")
    bordered.paste(canvas, (0, height))
    draw = ImageDraw.Draw(bordered)
    for row, line in enumerate(lines):
        color = (15, 15, 15) if row == 0 else (55, 55, 55)
        draw.text((12, 8 + row * 20), line, fill=color, spacing=4)
    return np.asarray(bordered)


def _save_tile(images: list[np.ndarray], lines: list[str], path: Path, *,
               vertical: bool = False) -> None:
    import imageio.v2 as imageio

    images = [np.asarray(image, dtype=np.uint8) for image in images]
    tile = np.concatenate(images, axis=0 if vertical else 1)
    imageio.imwrite(path, _caption_array(tile, lines))


def _render_with_overlay(renderer: Any, draw: Any) -> np.ndarray:
    """Composite contact marker geoms over the depth-tested close-up base."""
    base = renderer.render().astype(float)
    renderer.scene.ngeom = 0
    draw()
    overlay = renderer.render().astype(float)
    alpha = np.clip(overlay.max(axis=2) / 255.0, 0.0, 1.0)[..., None]
    return np.rint(base * (1.0 - alpha) + overlay * alpha).astype(np.uint8)


def _render_run_frame(model: Any, renderer: Any, data: Any, objects: list[dict[str, Any]],
                      frame: int, pose: np.ndarray, source: np.ndarray,
                      source_parents: np.ndarray, targets: np.ndarray,
                      points: np.ndarray, normals: np.ndarray, arm_bodies: set[int],
                      source_keep: set[int]) -> list[np.ndarray]:
    import mujoco

    data.qpos[:] = pose
    for index, obj in enumerate(objects):
        mocap = model.body_mocapid[model.body(f"closeup_object_{index}").id]
        data.mocap_pos[mocap] = obj["translations"][frame]
        data.mocap_quat[mocap] = Rotation.from_matrix(
            obj["rotations"][frame]
        ).as_quat(scalar_first=True)
    mujoco.mj_forward(model, data)
    pelvis_body = next((body for body in range(model.nbody)
                        if "pelvis" in str(model.body(body).name).lower()), 1)
    pelvis = data.xpos[pelvis_body].copy()
    center = np.mean(targets, axis=0)
    side, rotated, azimuth = view_offsets(
        center, pelvis, np.mean(normals, axis=0)
    )
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    options = mujoco.MjvOption()
    options.geomgroup[2] = True
    options.geomgroup[3] = True
    focus_robot(model, arm_bodies)
    result = []
    views = ((center, side, 1.0), (center, rotated, 1.1))
    for lookat, offset, distance in views:
        offset = np.asarray(offset) / max(float(np.linalg.norm(offset)), 1e-12)
        camera.lookat[:] = lookat
        camera.distance = distance
        camera.elevation = -math.degrees(math.asin(np.clip(offset[2], -1.0, 1.0)))
        camera.azimuth = math.degrees(math.atan2(-offset[1], -offset[0]))
        renderer.update_scene(data, camera=camera, scene_option=options)
        _draw_source(renderer.scene, source, source_parents, source_keep)
        result.append(_render_with_overlay(
            renderer, lambda: draw_contact_overlay(
                renderer.scene, targets, points, normals
            )
        ))
    camera.lookat[:] = pelvis
    camera.distance = 2.2
    camera.azimuth = azimuth
    camera.elevation = -15.0
    renderer.update_scene(data, camera=camera, scene_option=options)
    _draw_source(renderer.scene, source, source_parents, source_keep)
    result.append(_render_with_overlay(
        renderer, lambda: draw_contact_overlay(
            renderer.scene, targets, points, normals
        )
    ))
    return result


def _load_run(clip_dir: Path) -> dict[str, Any]:
    prepared = np.load(clip_dir / "prepared.npy", allow_pickle=True).item()
    motion = _load_npz(clip_dir / "final_retargeted.npz")
    plan = _load_npz(clip_dir / "contact_plan.npz")
    scene_metrics = (
        _load_npz(clip_dir / "scene_metrics.npz")
        if (clip_dir / "scene_metrics.npz").is_file() else {}
    )
    fixed_reference = (
        _load_npz(clip_dir / "fixed_reference.npz")
        if (clip_dir / "fixed_reference.npz").is_file() else {}
    )
    return {
        "directory": clip_dir, "prepared": prepared, "motion": motion, "plan": plan,
        "objects": np.load(clip_dir / "objects.npy", allow_pickle=True).tolist(),
        "scene_metrics": scene_metrics,
        "fixed_reference": fixed_reference,
    }


def render_clip(label_runs: list[tuple[str, Path]], clip: str, output: Path,
                requested_frames: list[int] | None, backend: str) -> list[Path]:
    os.environ["MUJOCO_GL"] = backend
    import mujoco
    import trimesh

    runs = [_load_run(directory) for _, directory in label_runs]
    models, renderers, datas, all_points = [], [], [], []
    for run, (label, directory) in zip(runs, label_runs):
        model = _build_model(
            run["prepared"]["model_path"],
            trimesh.load_mesh(directory / "static_scene.obj", process=False), run["objects"],
        )
        models.append(model)
        datas.append(mujoco.MjData(model))
        renderers.append(mujoco.Renderer(model, height=540, width=720))
        all_points.append(binding_material_points(run["prepared"], run["motion"], run["plan"]))
    if requested_frames is not None:
        frames = sorted(set(max(0, min(int(frame), len(runs[0]["motion"]["qpos"]) - 1))
                             for frame in requested_frames))
    else:
        frames = [row["frame"] for row in select_frames(
            [run["plan"] for run in runs], all_points,
        )]
    clip_output = output / clip
    clip_output.mkdir(parents=True, exist_ok=True)
    paths = []
    try:
        for frame in frames:
            anchor_groups = hand_binding_groups(runs[0]["plan"])
            active_anchor = [group for group in anchor_groups if bool(np.any(
                runs[0]["plan"]["surface_contact_mask"][frame, group["columns"]]
                & (runs[0]["plan"]["surface_contact_surface_ids"][frame, group["columns"]] >= 1)
            ))]
            if active_anchor:
                anchor_group = max(active_anchor, key=lambda group: float(np.max(np.abs(
                    _normal_values(runs[0]["plan"], all_points[0], group["columns"])[frame]
                ))))
            else:
                anchor_group = anchor_groups[0] if anchor_groups else None
            run_rows_images, run_rows_text = [], []
            for (label, _), run, model, renderer, data, points in zip(
                    label_runs, runs, models, renderers, datas, all_points):
                groups = hand_binding_groups(run["plan"])
                matching = [group for group in groups if (
                    group["effector"] == anchor_group["effector"]
                    and group["part"] == anchor_group["part"]
                )] if anchor_group is not None else []
                chosen = matching or groups[:1]
                active_groups = [group for group in chosen if bool(np.any(
                    run["plan"]["surface_contact_mask"][frame, group["columns"]]
                    & (run["plan"]["surface_contact_surface_ids"][frame, group["columns"]] >= 1)
                ))]
                selected_group = active_groups[0] if active_groups else chosen[0]
                columns = selected_group["columns"]
                if len(columns):
                    mask = run["plan"]["surface_contact_mask"][frame, columns] & (
                        run["plan"]["surface_contact_surface_ids"][frame, columns] >= 1
                    )
                    columns = columns[mask] if np.any(mask) else columns
                    targets = run["plan"]["surface_contact_environment_w"][frame, columns]
                    normals = run["plan"]["surface_contact_normals_w"][frame, columns]
                    material = points[frame, columns]
                    stats = frame_statistics(
                        run["plan"], points, columns, frame,
                        run["scene_metrics"].get("moving_depths"),
                    )
                else:
                    targets = np.zeros((1, 3))
                    normals = np.array([[0.0, 0.0, 1.0]])
                    material = targets.copy()
                    stats = {"normal_median_mm": None, "normal_worst_mm": None,
                             "tangent_median_mm": None, "backside_samples": 0,
                             "moving_depth_mm": None}
                parents = _source_parents(run["motion"].get(
                    "source_joint_names", np.asarray([], dtype=str)
                )) if "source_joint_names" in run["motion"] else None
                source_keep = _source_keep(
                    run["motion"].get("source_joint_names", np.asarray([], dtype=str)),
                    selected_group["effector"],
                )
                reference = run["fixed_reference"].get("qpos")
                root_offset = (
                    float(np.linalg.norm(
                        run["motion"]["qpos"][frame, :3] - reference[frame, :3]
                    )) * 1000 if reference is not None and frame < len(reference) else None
                )
                pixels = _render_run_frame(
                    model, renderer, data, run["objects"], frame,
                    run["motion"]["qpos"][frame], run["motion"]["source_joints_w"][frame],
                    parents, targets, material, normals,
                    arm_body_ids(model, selected_group["effector"]), source_keep,
                )
                run_rows_images.append(np.concatenate(pixels, axis=1))
                def value(item):
                    return "缺失" if item is None else f"{item:.1f}"
                run_rows_text.append(
                    f"{label} | normal med/max {value(stats['normal_median_mm'])}/"
                    f"{value(stats['normal_worst_mm'])} mm | tangent med "
                    f"{value(stats['tangent_median_mm'])} mm | back "
                    f"{stats['backside_samples']} | depth "
                    f"{value(stats['moving_depth_mm'])} mm | root "
                    f"{value(root_offset)} mm"
                )
            lines = [
                f"{clip} frame {frame}; hand: "
                f"{anchor_group['effector'] if anchor_group else '缺失'}",
            ]
            lines.extend(run_rows_text)
            path = clip_output / f"frame_{frame:04d}.png"
            _save_tile(run_rows_images, lines, path, vertical=True)
            paths.append(path)
    finally:
        for renderer in renderers:
            renderer.close()
    return paths


def write_index(output: Path, sections: dict[str, list[Path]]) -> None:
    parts = ["<!doctype html><html><head><meta charset='utf-8'>",
             "<title>Contact closeups</title></head><body>",
             "<h1>Contact closeups</h1>"]
    for clip, paths in sections.items():
        parts.append(f"<h2>{html.escape(clip)}</h2>")
        for path in paths:
            relative = path.relative_to(output).as_posix()
            parts.append(
                f"<figure><a href='{html.escape(relative)}'>"
                f"<img src='{html.escape(relative)}' loading='lazy' width='720'></a>"
                f"<figcaption>{html.escape(path.stem)}</figcaption></figure>"
            )
    parts.append("</body></html>")
    (output / "index.html").write_text("\n".join(parts), encoding="utf-8")


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--run must be LABEL=DIRECTORY")
    label, directory = value.split("=", 1)
    if not label or not directory:
        raise argparse.ArgumentTypeError("--run must have a nonempty label and directory")
    return label, Path(directory)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, action="append", type=_parse_run,
                        metavar="LABEL=CLIP_DIR")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", nargs="+", type=int)
    args = parser.parse_args(argv)
    backend = _choose_backend()
    print(f"MUJOCO_GL={backend}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    sections: dict[str, list[Path]] = {}
    for _, directory in args.run:
        clip = directory.name
        if clip in sections:
            continue
        sections[clip] = render_clip(
            args.run, clip, args.output, args.frames, backend,
        )
    write_index(args.output, sections)
    print(f"SAVED {args.output / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
