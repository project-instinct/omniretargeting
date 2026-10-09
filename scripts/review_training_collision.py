"""Geometry-only audit of compiled collision geoms, without repair.
Uses original scene/object queries; training interpolation and reset require
explicit sourced configuration, otherwise their status remains unverified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import mujoco
import numpy as np
import trimesh
from omniretargeting.collision_manifest import collision_manifest
from omniretargeting.cpu_budget import available_cpus
from omniretargeting.moving_scene import MovingSceneQuery
from omniretargeting.scene_geometry import (
    SceneQuery,
    collision_geom_ids,
    geom_surface_offsets,
)
from omniretargeting.self_collision import SelfCollisionQuery
from omniretargeting.training_collision_match import training_collision_match
from omniretargeting.training_collision_parallel import query_self_frames
SCHEMA_VERSION = 1
STATIC_SAMPLING_M = 0.02
STATIC_RISK_CLEARANCE_M = 0.008
RESET_STATIC_FAILURE_M = 0.005
RESET_DEFECT_M = 0.010
DEPTH_EPSILON_M = 1e-6
MAX_SUBSTEPS = 256
SUPPORTED_PREPROCESS = {"instinct_mj_motion_interpolate_bilinear"}
SUPPORTED_OBJECT_PREPROCESS = {"instinct_mj_pose_interpolate_bilinear"}
SUPPORTED_RUNTIME_SAMPLING = {"round_nearest_frame"}
@dataclass
class TrainingTrajectory:
    qpos: np.ndarray
    source_times: np.ndarray
    labels: np.ndarray
    preprocessing: dict[str, Any]
def _scalar(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    array = np.asarray(value)
    if array.dtype == object and array.size == 1:
        return array.item()
    if array.ndim == 0:
        return array.item()
    return value
def _finite_float(value: Any, name: str, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return result
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
def _xml_documents(path: Path) -> list[tuple[Path, ET.Element]]:
    documents: list[tuple[Path, ET.Element]] = []
    visited: set[Path] = set()
    def visit(current: Path) -> None:
        resolved = current.resolve()
        if resolved in visited:
            return
        visited.add(resolved)
        root = ET.parse(current).getroot()
        documents.append((current, root))
        for include in root.iter("include"):
            name = include.get("file")
            if name:
                visit((current.parent / name).resolve())
    visit(path.resolve())
    return documents
def _mesh_assets(path: Path) -> list[Path]:
    assets: set[Path] = set()
    for document_path, root in _xml_documents(path):
        meshdir = root.find("compiler")
        base = Path(meshdir.get("meshdir", "")) if meshdir is not None else Path()
        for mesh in root.iter("mesh"):
            filename = mesh.get("file")
            if not filename:
                continue
            candidate = Path(filename)
            if candidate.is_absolute():
                resolved = candidate.resolve()
            else:
                resolved = (document_path.parent / base / candidate).resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"robot mesh referenced by {path}: {resolved}")
            assets.add(resolved)
    return sorted(assets)
def model_description(model_path: Path, motion_payload: dict[str, Any], model=None, training_model_path=None):
    """Hash the XML plus every referenced mesh, not only the small XML file."""
    resolved = model_path.resolve()
    xml_hash = _sha256(resolved)
    meshes = _mesh_assets(resolved)
    mesh_hashes = [{"path": str(path), "sha256": _sha256(path)} for path in meshes]
    composite = hashlib.sha256()
    composite.update(b"omniretargeting-training-collision-model-v1\0")
    composite.update(xml_hash.encode("ascii"))
    for item in mesh_hashes:
        composite.update(item["sha256"].encode("ascii") + b"\0")
    if model is None:
        model = mujoco.MjModel.from_xml_path(str(resolved))
    manifest = collision_manifest(model, source_path=resolved)
    ids = collision_geom_ids(model)
    geom_types: dict[str, int] = {}
    for geom_type in np.unique(model.geom_type[ids]):
        geom_types[str(mujoco.mjtGeom(int(geom_type))).split(".")[-1]] = int(
            np.sum(model.geom_type[ids] == geom_type)
        )
    match = training_collision_match(
        manifest,
        motion_payload,
        training_model_path,
        legacy_asset_sha256=composite.hexdigest(),
    )
    return {
        "path": str(resolved),
        "xml_sha256": xml_hash,
        "meshes": mesh_hashes,
        "asset_bundle_sha256": composite.hexdigest(),
        "collision_model_sha256": manifest["sha256"],
        "collision_manifest": manifest,
        "collision_geom_ids": ids.tolist(),
        "collision_geom_count": int(len(ids)),
        "collision_geom_types": geom_types,
        "collision_bodies": sorted(
            {
                model.body(int(body_id)).name
                for body_id in np.unique(model.geom_bodyid[ids])
            }
        ),
        "filter": "scene_geometry.collision_geom_ids: non-world contype/conaffinity geoms",
        "training_match": match,
    }
def _reference_sources(profile: dict[str, Any] | None) -> list[dict[str, str]]:
    if not isinstance(profile, dict):
        return []
    source = profile.get("provenance")
    if not isinstance(source, dict) or not source.get("source"):
        return []
    path = Path(str(source["source"])).expanduser()
    result = {
        "source": str(source["source"]),
        "definition": source.get("definition", ""),
        "lines": str(source.get("lines", "")),
    }
    if path.is_file():
        result["sha256"] = _sha256(path)
    return [result]
def interpolation_profile(payload: dict[str, Any]) -> dict[str, Any]:
    """Read an explicit loader profile; absent or partial data stays unverified."""
    raw = _scalar(payload.get("training_interpolation"))
    result: dict[str, Any] = {
        "status": "unverified",
        "reason": "motion payload has no training_interpolation configuration",
        "provenance": _reference_sources(raw if isinstance(raw, dict) else None),
        "preprocess": None,
        "object_preprocess": None,
        "runtime_sampling": None,
    }
    if not isinstance(raw, dict):
        return result
    result["raw"] = raw
    result["provenance"] = _reference_sources(raw)
    missing: list[str] = []
    preprocess = raw.get("preprocess")
    object_preprocess = raw.get("object_preprocess")
    runtime = raw.get("runtime_sampling")
    if preprocess not in SUPPORTED_PREPROCESS:
        missing.append("preprocess")
    if object_preprocess not in SUPPORTED_OBJECT_PREPROCESS:
        missing.append("object_preprocess")
    if runtime not in SUPPORTED_RUNTIME_SAMPLING:
        missing.append("runtime_sampling")
    try:
        source_fps = _finite_float(
            raw.get("source_framerate", payload.get("framerate")), "source_framerate", 1.0
        )
        target_fps = _finite_float(raw.get("target_framerate"), "target_framerate", 1.0)
    except ValueError:
        missing.append("framerates")
        source_fps = target_fps = 0.0
    if not result["provenance"]:
        missing.append("provenance.source")
    if missing:
        result.update(
            status="unverified",
            reason="unsupported or missing interpolation fields: " + ", ".join(missing),
        )
        return result
    result.update(
        status="explicit_configuration",
        source_framerate_hz=source_fps,
        target_framerate_hz=target_fps,
        preprocess=preprocess,
        object_preprocess=object_preprocess,
        runtime_sampling=runtime,
        update_period_s=(
            None
            if raw.get("update_period_s") is None
            else _finite_float(raw.get("update_period_s"), "update_period_s", 0.0)
        ),
        frame_interval_s=(
            None
            if raw.get("frame_interval_s") is None
            else _finite_float(raw.get("frame_interval_s"), "frame_interval_s", 0.0)
        ),
    )
    return result
def _interpolate_pose(model: mujoco.MjModel, first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    tangent = np.zeros(model.nv)
    mujoco.mj_differentiatePos(model, tangent, 1.0, first, second)
    result = np.asarray(first, dtype=float).copy()
    mujoco.mj_integratePos(model, result, tangent, float(alpha))
    return result
def training_trajectory(
    model: mujoco.MjModel, poses: np.ndarray, framerate: float, profile: dict[str, Any],
    frame_start: int = 0,
) -> TrainingTrajectory:
    poses = np.asarray(poses, dtype=float)
    if profile.get("status") != "explicit_configuration":
        times = np.arange(len(poses), dtype=float)
        return TrainingTrajectory(
            poses,
            times,
            np.arange(len(poses), dtype=int) + frame_start,
            {"status": "none_input_frames_only"},
        )
    source_fps = float(profile["source_framerate_hz"])
    target_fps = float(profile["target_framerate_hz"])
    if len(poses) < 2:
        return TrainingTrajectory(
            poses.copy(),
            np.arange(len(poses), dtype=float),
            np.arange(len(poses), dtype=int) + frame_start,
            {"status": "single_frame_no_interpolation", "source_framerate_hz": source_fps,
             "target_framerate_hz": target_fps},
        )
    count = int(np.ceil((len(poses) - 1) * target_fps / source_fps))
    frame_index = np.arange(count, dtype=float) * source_fps / target_fps
    frame_index = frame_index[frame_index <= len(poses) - 1]
    lower = np.floor(frame_index).astype(int)
    upper = np.ceil(frame_index).astype(int)
    alpha = frame_index - lower
    output = np.empty((len(frame_index), model.nq), dtype=float)
    for row, (lo, hi, ratio) in enumerate(zip(lower, upper, alpha)):
        output[row] = _interpolate_pose(model, poses[lo], poses[hi], ratio)
    details = {
        "status": "resampled",
        "source_framerate_hz": source_fps,
        "target_framerate_hz": target_fps,
        "input_frames": int(len(poses)),
        "output_frames": int(len(output)),
        "runtime_sampling": profile["runtime_sampling"],
    }
    return TrainingTrajectory(
        output, frame_index, np.arange(len(output), dtype=int) + frame_start, details
    )
def _body_name(model: mujoco.MjModel, geom_id: int) -> str:
    return model.body(int(model.geom_bodyid[int(geom_id)])).name
def _depth_summary(
    depths: np.ndarray, model: mujoco.MjModel, geom_ids: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    depths = np.asarray(depths, dtype=float).reshape(len(labels), -1)
    peak = depths.max(axis=1) if depths.size else np.zeros(len(labels))
    max_index = int(np.argmax(peak)) if len(peak) else 0
    geom_index = int(np.argmax(depths[max_index])) if depths.size else 0
    return {
        "mean_m": float(np.mean(peak)),
        "p95_m": float(np.quantile(peak, 0.95)),
        "max_m": float(np.max(peak, initial=0.0)),
        "worst_frame": int(labels[max_index]) if len(labels) else None,
        "worst_body": _body_name(model, geom_ids[geom_index]) if len(geom_ids) else None,
        "frames_over_5mm": np.flatnonzero(peak > RESET_STATIC_FAILURE_M + DEPTH_EPSILON_M).tolist(),
        "frames_over_1cm": np.flatnonzero(peak > RESET_DEFECT_M + DEPTH_EPSILON_M).tolist(),
    }
def _static_shell_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pose: np.ndarray,
    geom_ids: np.ndarray,
    offsets: list[np.ndarray],
) -> np.ndarray:
    data.qpos[:] = pose
    mujoco.mj_forward(model, data)
    return np.concatenate(
        [
            data.geom_xpos[int(gid)] + points @ data.geom_xmat[int(gid)].reshape(3, 3).T
            for gid, points in zip(geom_ids, offsets)
        ]
    )
def _query_static_poses(
    model: mujoco.MjModel,
    poses: np.ndarray,
    geom_ids: np.ndarray,
    query: SceneQuery,
    batch: int = 64,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    offsets = [
        np.unique(
            np.vstack([geom_surface_offsets(model, int(gid), STATIC_SAMPLING_M), np.zeros((1, 3))]),
            axis=0,
        )
        for gid in geom_ids
    ]
    depths = np.zeros((len(poses), model.ngeom), dtype=float)
    clearance = np.full(len(poses), np.inf, dtype=float)
    unknown = unresolved = 0
    data = mujoco.MjData(model)
    for begin in range(0, len(poses), batch):
        chunk = poses[begin : begin + batch]
        points = np.concatenate(
            [_static_shell_points(model, data, pose, geom_ids, offsets) for pose in chunk]
        )
        measured = query.query(points)
        unknown += int(np.count_nonzero(measured["uncertain"]))
        finite = np.isfinite(measured["clearance"])
        unresolved += int(np.count_nonzero(~finite))
        safe = np.where(finite, measured["clearance"], 0.0)
        safe = safe.reshape(len(chunk), -1)
        clearance[begin : begin + len(chunk)] = np.min(safe, axis=1)
        starts = np.concatenate([[0], np.cumsum([len(points) for points in offsets])])
        for row, gid in enumerate(geom_ids):
            block = safe[:, int(starts[row]) : int(starts[row + 1])]
            depths[begin : begin + len(chunk), int(gid)] = np.maximum(-block, 0).max(axis=1)
    return depths, clearance, unknown, unresolved
def audit_static(
    model: mujoco.MjModel,
    trajectory: TrainingTrajectory,
    prepared_path: Path,
    geom_ids: np.ndarray,
) -> dict[str, Any]:
    scene_path = prepared_path.parent / "static_scene.obj"
    result: dict[str, Any] = {
        "status": "unknown",
        "scene": str(scene_path),
        "scene_sha256": _sha256(scene_path) if scene_path.is_file() else None,
        "collision_free_certificate": False,
    }
    if not scene_path.is_file():
        result["reason"] = "static_scene.obj missing"
        result.update(**_depth_summary(np.zeros((len(trajectory.qpos), len(geom_ids))), model, geom_ids, trajectory.labels))
        return result
    try:
        inputs = np.load(prepared_path, allow_pickle=True).item()
        enclosed = bool(_scalar(inputs.get("scene_enclosed_free_space"), False))
        scene = trimesh.load_mesh(scene_path, process=False)
        query = SceneQuery(scene, enclosed_free_space=enclosed, full_geometry=False)
        if query.has_unsigned_only_components:
            from scripts.review_scene_metrics import open_surface_depths
            depths, unknown, _exit, backend = open_surface_depths(
                model, trajectory.qpos, geom_ids, query
            )
            clearance = np.full(len(trajectory.qpos), np.inf)
            result.update(
                status="measured_open_surface",
                method="exact geom/finite-triangle crossing; no solid side for open scans",
                unknown_query_count=int(unknown),
                unresolved_query_count=0,
                backend={key: backend[key] for key in (
                    "open_surface_test", "pair_queries", "cylinder_surface_faces",
                    "ambiguous_surface_intersections", "uncertified_cylinder_witnesses",
                )},
                frame_clearance_m=None,
            )
        else:
            depths, clearance, unknown, unresolved = _query_static_poses(
                model, trajectory.qpos, geom_ids, query
            )
            result.update(
                status="unknown" if unknown or unresolved else "measured",
                method="original triangles; winding containment; deterministic sampled shell",
                spacing_m=STATIC_SAMPLING_M,
                unknown_query_count=int(unknown),
                unresolved_query_count=int(unresolved),
                frame_clearance_m=clearance.tolist(),
            )
        result.update(
            **_depth_summary(depths, model, geom_ids, trajectory.labels),
            frame_depths_max_m=depths.max(axis=1).tolist(),
        )
    except Exception as error:
        result.update(status="unknown", reason=f"{type(error).__name__}: {error}")
    return result
def consume_scene_metrics(path, motion, prepared, objects, model, trajectory, geom_ids):
    """Consume a frozen scene audit only after path/frame/qpos/model checks."""
    record_path = path.resolve()
    try:
        record = json.loads(record_path.read_text())
        arrays_path = record_path.with_suffix(".npz")
        arrays = np.load(arrays_path, allow_pickle=True)
        receipt = json.loads((record_path.parent / "receipt.json").read_text())
        scene_receipt = next(x for x in receipt["acceptance"]["receipts"] if x["name"] == "scene_metrics")
        args = scene_receipt["args"]
        argmap = {str(key): args[index + 1] for index, key in enumerate(args[:-1]) if str(key).startswith("--")}
        qpos_hash = hashlib.sha256(np.ascontiguousarray(np.load(motion)["qpos"]).tobytes()).hexdigest()
        diagnostics = json.loads((record_path.parent / "final_retargeted.diagnostics.json").read_text())
        expected_motion, expected_prepared, expected_objects = (
            Path(argmap[k]).resolve() for k in ("--motion", "--prepared", "--objects")
        )
        checks = {
            "record_motion_path": Path(record["motion"]).resolve() == motion.resolve(),
            "receipt_motion_path": expected_motion == motion.resolve(),
            "receipt_prepared_path": expected_prepared == prepared.resolve(),
            "receipt_objects_path": expected_objects == objects.resolve(),
            "start": int(record["start"]) == int(trajectory.labels[0]),
            "frames": int(record["frames"]) == len(trajectory.qpos),
            "qpos_sha256": qpos_hash == diagnostics["accepted_qpos_sha256"],
            "array_shapes": (
                arrays["static_depths"].shape == (len(trajectory.qpos), model.ngeom)
                and arrays["moving_depths"].shape == (len(trajectory.qpos), model.ngeom)
            ),
            "static_summary": bool(np.isclose(arrays["static_depths"].max(), record["static"]["max_m"])),
            "moving_summary": bool(np.isclose(arrays["moving_depths"].max(), record["moving"]["max_m"])),
        }
        if not all(checks.values()):
            raise ValueError("failed checks: " + ", ".join(k for k, ok in checks.items() if not ok))
        static, moving = dict(record["static"]), dict(record["moving"])
        static_depths, moving_depths = arrays["static_depths"].max(axis=1), arrays["moving_depths"].max(axis=1)
        static.update(
            status="unknown" if static.get("uncertain_samples", 0) else "measured",
            frame_depths_max_m=static_depths.tolist(),
            frames_over_5mm=np.flatnonzero(static_depths > RESET_STATIC_FAILURE_M + DEPTH_EPSILON_M).tolist(),
            frames_over_1cm=np.flatnonzero(static_depths > RESET_DEFECT_M + DEPTH_EPSILON_M).tolist(),
            frame_clearance_m=None,
            frame_unknown_query_count=[int(bool(static.get("uncertain_samples", 0)))] * len(trajectory.qpos),
        )
        moving_unknown = bool(moving.get("unknown_components", 0) or moving.get("failures"))
        moving.update(
            status="unknown" if moving_unknown else "measured",
            frame_depths_max_m=moving_depths.tolist(),
            frames_over_5mm=np.flatnonzero(moving_depths > RESET_STATIC_FAILURE_M + DEPTH_EPSILON_M).tolist(),
            frames_over_1cm=np.flatnonzero(moving_depths > RESET_DEFECT_M + DEPTH_EPSILON_M).tolist(),
            frame_min_clearance_m=None,
            frame_unknown_query_count=[int(moving_unknown)] * len(trajectory.qpos),
        )
        provenance = {
            "status": "consumed", "record": str(record_path), "record_sha256": _sha256(record_path),
            "depths_sha256": _sha256(arrays_path), "motion_qpos_sha256": qpos_hash,
            "prepared_sha256": _sha256(prepared), "receipt_scene_metrics_seconds": scene_receipt["seconds"],
            "objects_sha256": _sha256(objects), "static_scene_sha256": _sha256(record_path.parent / "static_scene.obj"),
            "risk_selection": "endpoint-positive only: consumed report has no per-frame clearance",
        }
        return static, moving, provenance
    except Exception as error:
        detail = f"{type(error).__name__}: {error}" if str(error) else type(error).__name__
        empty = _depth_summary(np.zeros((len(trajectory.qpos), len(geom_ids))), model, geom_ids, trajectory.labels)
        return {**empty, "status": "unknown", "reason": detail, "frame_depths_max_m": [0.0] * len(trajectory.qpos)}, {
            **empty, "status": "unknown", "reason": detail, "moving_object_count": 0,
            "frame_depths_max_m": [0.0] * len(trajectory.qpos), "failures": [],
        }, {"status": "unverified", "reason": detail}
def _slice_objects(objects: list[dict[str, Any]], start: int, frames: int) -> list[dict[str, Any]]:
    sliced: list[dict[str, Any]] = []
    for obj in objects:
        item = dict(obj)
        for key in ("translations", "rotations", "scales"):
            if key in item:
                values = np.asarray(item[key])
                if len(values) < start + frames:
                    raise ValueError(
                        f"object {item.get('name')} has {len(values)} {key}, need {start + frames}"
                    )
                item[key] = values[start : start + frames]
        sliced.append(item)
    return sliced
def audit_moving(
    model: mujoco.MjModel,
    trajectory: TrainingTrajectory,
    objects: list[dict[str, Any]],
    geom_ids: np.ndarray,
) -> tuple[dict[str, Any], MovingSceneQuery | None]:
    if not objects:
        summary = _depth_summary(
            np.zeros((len(trajectory.qpos), len(geom_ids))), model, geom_ids, trajectory.labels
        )
        return {
            **summary,
            "status": "not_applicable",
            "moving_object_count": 0,
            "unknown_query_count": 0,
            "failures": [],
            "collision_free_certificate": True,
        }, None
    object_frames = max(
        len(np.asarray(obj[key]))
        for obj in objects
        for key in ("translations", "rotations", "scales")
        if key in obj
    )
    moving = MovingSceneQuery(model, objects, object_frames)
    depths = np.zeros((len(trajectory.qpos), model.ngeom), dtype=float)
    unknown = np.zeros(len(trajectory.qpos), dtype=int)
    failures: list[dict[str, Any]] = []
    min_clearance = np.full(len(trajectory.qpos), np.inf)
    for frame, (pose, source_time) in enumerate(zip(trajectory.qpos, trajectory.source_times)):
        answer = moving.query(pose, source_time)
        depths[frame] = np.maximum(answer["forward_depth"], answer["reverse_depth"]).max(axis=0)
        unknown[frame] = int(answer["unknown_open_samples"]) + len(answer["failures"])
        min_clearance[frame] = min(
            [float(row["clearance"]) for row in answer["rows"]], default=np.inf
        )
        failures.extend(
            {"frame": int(trajectory.labels[frame]), "message": str(message)}
            for message in answer["failures"]
        )
    summary = _depth_summary(depths, model, geom_ids, trajectory.labels)
    return {
        **summary,
        "status": "unknown" if unknown.any() or failures else "measured",
        "moving_object_count": len(objects),
        "unknown_query_count": int(unknown.sum()),
        "frame_unknown_query_count": unknown.tolist(),
        "failures": failures,
        "frame_min_clearance_m": [None if not math.isfinite(x) else x for x in min_clearance],
        "frame_depths_max_m": depths.max(axis=1).tolist(),
        **moving.describe(),
    }, moving
from scripts.training_self_audit import audit_self

def _required_motion_steps(
    model: mujoco.MjModel,
    first: np.ndarray,
    second: np.ndarray,
    geom_ids: np.ndarray,
    moving: MovingSceneQuery | None,
    frame: int,
) -> tuple[int, int]:
    motion_bound = 0.0
    if moving is not None:
        motion_bound = float(
            np.max(moving.motion_bounds.displacements(first, second), initial=0.0)
        )
    else:
        from omniretargeting.collision_motion import CollisionMotionBounds
        motion_bound = float(
            np.max(CollisionMotionBounds(model, geom_ids).displacements(first, second), initial=0)
        )
    self_steps = max(1, int(np.ceil(motion_bound / 0.005)))
    moving_steps = 1
    if moving is not None and len(moving.objects):
        try:
            moving_steps = moving.interval_steps(first, second, frame, spacing=0.01, max_substeps=MAX_SUBSTEPS)
        except Exception:
            moving_steps = MAX_SUBSTEPS
    required = max(self_steps, int(moving_steps))
    return min(required, MAX_SUBSTEPS), required
def _risk_intervals(
    trajectory: TrainingTrajectory,
    static: dict[str, Any],
    moving: dict[str, Any],
    self_collision: dict[str, Any],
) -> list[int]:
    selected: list[int] = []
    static_depth = np.asarray(static.get("frame_depths_max_m", []), dtype=float)
    static_values = static.get("frame_clearance_m", [np.inf])
    static_clearance = np.asarray([np.inf] * len(static_depth) if static_values is None else static_values, dtype=float)
    moving_depth = np.asarray(moving.get("frame_depths_max_m", []), dtype=float)
    moving_values = moving.get("frame_min_clearance_m", [np.inf])
    moving_clearance = np.asarray([np.inf] * len(moving_depth) if moving_values is None else moving_values, dtype=float)
    self_depth = np.asarray(self_collision.get("frame_depths_max_m", []), dtype=float)
    self_clearance = np.asarray(
        [np.inf if x is None else x for x in self_collision.get("frame_min_clearance_m", [np.inf])], dtype=float
    )
    for interval in range(max(0, len(trajectory.qpos) - 1)):
        values = (
            static_depth[interval : interval + 2],
            moving_depth[interval : interval + 2],
            self_depth[interval : interval + 2],
        )
        near = (
            static_clearance[interval : interval + 2],
            moving_clearance[interval : interval + 2],
            self_clearance[interval : interval + 2],
        )
        if any(np.any(block > DEPTH_EPSILON_M) for block in values) or any(
            np.any(block < STATIC_RISK_CLEARANCE_M) for block in near
        ):
            selected.append(interval)
    return selected
def audit_interframe(
    model: mujoco.MjModel,
    saved: TrainingTrajectory,
    trajectory: TrainingTrajectory,
    profile: dict[str, Any],
    static: dict[str, Any],
    moving_report: dict[str, Any],
    self_report: dict[str, Any],
    moving: MovingSceneQuery | None,
    self_query: SelfCollisionQuery | None,
    geom_ids: np.ndarray,
    prepared_path: Path,
) -> dict[str, Any]:
    common = {
        "status": "unverified",
        "reason": "actual training interpolation is not uniquely determined",
        "risk_intervals": [],
        "collision_free_certificate": False,
    }
    if profile.get("status") != "explicit_configuration":
        return common
    source_risk = set(_risk_intervals(saved, static, moving_report, self_report))
    risk = [
        interval
        for interval in range(max(0, len(trajectory.qpos) - 1))
        if int(np.floor(trajectory.source_times[interval])) in source_risk
    ]
    common.update(
        status="risk_regions_measured",
        reason=None,
        risk_intervals=risk,
        interpolation=trajectory.preprocessing,
        method="MuJoCo tangent interpolation: linear scalars, shortest-arc root quaternion",
        motion_spacing_m=0.005,
        max_substeps=MAX_SUBSTEPS,
        mean_m=0.0,
        p95_m=0.0,
        max_m=0.0,
        worst=None,
    )
    if not risk:
        return common
    try:
        scene_path = prepared_path.parent / "static_scene.obj"
        scene_query = None
        if scene_path.is_file():
            inputs = np.load(prepared_path, allow_pickle=True).item()
            scene_query = SceneQuery(
                trimesh.load_mesh(scene_path, process=False),
                enclosed_free_space=bool(_scalar(inputs.get("scene_enclosed_free_space"), False)),
                full_geometry=False,
            )
    except Exception as error:
        common.update(status="unknown", reason=f"static substep query unavailable: {error}")
        return common
    if static.get("method", "").startswith("exact geom"):
        common.update(status="unknown", reason="open-surface interframe exact backend is not cached here")
        return common
    interval_peaks: dict[int, float] = {}
    worst: dict[str, Any] | None = None
    incomplete: list[int] = []
    unknown = 0
    for interval in risk:
        first, second = trajectory.qpos[interval], trajectory.qpos[interval + 1]
        steps, required = _required_motion_steps(
            model, first, second, geom_ids, moving,
            int(np.floor(trajectory.source_times[interval]))
        )
        if required > steps:
            incomplete.append(interval)
        for part in range(1, steps + 1):
            alpha = part / steps
            pose = _interpolate_pose(model, first, second, alpha)
            source_time = (1.0 - alpha) * trajectory.source_times[interval] + alpha * trajectory.source_times[interval + 1]
            category = "self"
            depth = 0.0
            body = None
            if scene_query is not None and not static.get("method", "").startswith("exact geom"):
                sub_depths, _clear, sub_unknown, _unresolved = _query_static_poses(
                    model, pose[None], geom_ids, scene_query, batch=1
                )
                unknown += int(sub_unknown)
                if sub_depths.max() > depth:
                    category, depth = "static", float(sub_depths.max())
                    body = _body_name(model, int(np.argmax(sub_depths)))
            if moving is not None:
                answer = moving.query(pose, source_time)
                unknown += int(answer["unknown_open_samples"]) + len(answer["failures"])
                moving_depth = float(
                    np.maximum(answer["forward_depth"], answer["reverse_depth"]).max(initial=0.0)
                )
                if moving_depth > depth:
                    category, depth = "moving", moving_depth
                    gid = int(np.argmax(np.maximum(answer["forward_depth"], answer["reverse_depth"]).max(axis=0)))
                    body = _body_name(model, gid)
            if self_query is not None:
                try:
                    rows = [r for r in self_query.query(pose) if tuple(sorted(r["pair"]))
                            not in {pair for pair, evidence in self_query.structural_evidence.items()
                                    if evidence["status"] == "certified"}]
                    self_depth = max((max(-row["clearance"], 0.0) for row in rows), default=0.0)
                    if self_depth > depth:
                        category, depth = "self", self_depth
                        row = min(rows, key=lambda item: item["clearance"])
                        body = " | ".join(_body_name(model, gid) for gid in row["pair"])
                except Exception:
                    unknown += 1
            interval_peaks[interval] = max(interval_peaks.get(interval, 0.0), depth)
            if worst is None or depth > worst["max_m"]:
                worst = {
                    "interval": interval,
                    "alpha": alpha,
                    "category": category,
                    "max_m": depth,
                    "body": body,
                }
    peaks = np.asarray(list(interval_peaks.values()), dtype=float)
    common.update(
        mean_m=float(np.mean(peaks)) if len(peaks) else 0.0,
        p95_m=float(np.quantile(peaks, 0.95)) if len(peaks) else 0.0,
        max_m=float(peaks.max(initial=0.0)),
        worst=worst,
        subdivision_limit_intervals=incomplete,
        unknown_query_count=int(unknown),
    )
    return common
def reset_profile(payload: dict[str, Any]) -> dict[str, Any]:
    raw = _scalar(payload.get("training_reset"))
    if not isinstance(raw, dict):
        return {
            "status": "unverified",
            "reason": "motion payload has no training_reset configuration",
            "provenance": [],
        }
    provenance = _reference_sources(raw)
    required = {"candidate_frames", "ensure_link_below_zero_ground", "motion_start_height_offset", "sampling"}
    missing = sorted(required - set(raw))
    if missing or not provenance:
        return {
            "status": "unverified",
            "reason": "missing reset fields or provenance: " + ", ".join(missing or ["provenance.source"]),
            "raw": raw,
            "provenance": provenance,
        }
    return {
        "status": "explicit_configuration",
        "candidate_frames": raw["candidate_frames"],
        "ensure_link_below_zero_ground": bool(raw["ensure_link_below_zero_ground"]),
        "motion_start_height_offset_m": _finite_float(
            raw["motion_start_height_offset"], "motion_start_height_offset"
        ),
        "sampling": raw["sampling"],
        "raw": raw,
        "provenance": provenance,
    }
def _reset_geometry(
    profile: dict[str, Any],
    static: dict[str, Any],
    moving: dict[str, Any],
    self_report: dict[str, Any],
    labels: np.ndarray,
) -> dict[str, Any]:
    static_depth = np.asarray(static.get("frame_depths_max_m", np.zeros(len(labels))), dtype=float)
    moving_depth = np.asarray(moving.get("frame_depths_max_m", np.zeros(len(labels))), dtype=float)
    self_depth = np.asarray(self_report.get("frame_depths_max_m", np.zeros(len(labels))), dtype=float)
    static_unknown = np.asarray(
        static.get("frame_unknown_query_count", np.zeros(len(labels))), dtype=int
    )
    if static.get("status") == "unknown":
        static_unknown = np.ones(len(labels), dtype=int)
    moving_unknown = np.asarray(moving.get("frame_unknown_query_count", np.zeros(len(labels))), dtype=int)
    self_error_labels = {int(row.get("frame", -1)) for row in self_report.get("errors", [])}
    self_unknown = np.asarray(
        [int(label in self_error_labels) for label in labels], dtype=int
    )
    failed = np.flatnonzero(
        (static_depth > RESET_STATIC_FAILURE_M + DEPTH_EPSILON_M)
        | (moving_depth > RESET_STATIC_FAILURE_M + DEPTH_EPSILON_M)
        | (self_depth > float(self_report.get("config", {}).get("penetration_tolerance", 0.001)) + DEPTH_EPSILON_M)
    )
    unknown = np.flatnonzero((static_unknown > 0) | (moving_unknown > 0) | (self_unknown > 0))
    residual = np.flatnonzero(
        ((static_depth > DEPTH_EPSILON_M) | (moving_depth > DEPTH_EPSILON_M) | (self_depth > DEPTH_EPSILON_M))
        & ~np.isin(np.arange(len(labels)), failed)
    )
    return {
        "failed_frames": labels[failed].tolist(),
        "unknown_geometry_frames": labels[unknown].tolist(),
        "positive_residual_frames": labels[residual].tolist(),
        "physics_unverified_frames": "all candidate frames",
        "failure_thresholds": {
            "static_or_moving_m": RESET_STATIC_FAILURE_M,
            "self_m": self_report.get("config", {}).get("penetration_tolerance", 0.001),
        },
    }
def audit_reset(
    payload: dict[str, Any],
    profile: dict[str, Any],
    trajectory: TrainingTrajectory,
    static: dict[str, Any],
    moving: dict[str, Any],
    self_report: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "status": "unverified",
        "profile": profile,
        "physics_validation": {
            "status": "unverified",
            "reason": "no physics reset/settling harness is part of this audit",
        },
    }
    geometry = _reset_geometry(profile, static, moving, self_report, trajectory.labels)
    result.update(
        geometry_status="failed" if geometry["failed_frames"] else "unverified",
        **geometry,
    )
    if profile.get("status") != "explicit_configuration":
        result["reason"] = profile.get("reason", "reset configuration is unknown")
        return result
    if profile.get("ensure_link_below_zero_ground"):
        result["reason"] = (
            "ensure_link_below_zero_ground lift cannot be reconstructed from this payload "
            "without the training loader's link_pos_w and link-of-interest mapping"
        )
    elif profile.get("motion_start_height_offset_m", 0.0) != 0.0:
        result["reason"] = "nonzero reset height offset is not applied to the saved geometry"
        result["unknown_geometry_frames"] = trajectory.labels.tolist()
    return result
def _global_status(
    static: dict[str, Any], moving: dict[str, Any], self_report: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any]:
    unknown = any(item.get("status") == "unknown" for item in (static, moving, self_report))
    failed = bool(
        static.get("frames_over_1cm") or moving.get("frames_over_1cm") or self_report.get("frames_over_1cm")
    )
    residual = bool(
        static.get("frames_over_5mm") or moving.get("frames_over_5mm") or self_report.get("frames_over_5mm")
    )
    geometry = "unknown" if unknown else "failed" if failed else "residual" if residual else "within_5mm_unverified"
    return {
        "geometry_status": geometry,
        "acceptance_status": "unverified",
        "reason": "training model/interpolation/reset provenance and physics reset harness are not all verified",
        "collision_free_certificate": False,
    }
def audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    prepared = args.prepared.resolve()
    motion_path = args.motion.resolve()
    objects_path = args.objects.resolve()
    inputs = dict(np.load(prepared, allow_pickle=True).item())
    model_path_value = _scalar(inputs.get("model_path"))
    if not model_path_value:
        raise ValueError("prepared.npy has no model_path")
    model_path = Path(str(model_path_value)).resolve()
    payload = dict(np.load(motion_path, allow_pickle=True))
    if "qpos" not in payload:
        raise ValueError("motion npz must contain qpos")
    poses = np.asarray(payload["qpos"], dtype=float)
    if poses.ndim != 2 or poses.shape[1] != mujoco.MjModel.from_xml_path(str(model_path)).nq:
        raise ValueError(f"motion qpos has shape {poses.shape}; expected (frames, model.nq)")
    frame_start = int(_scalar(payload.get("frame_start"), 0))
    framerate = _finite_float(_scalar(payload.get("framerate"), 30.0), "framerate", 1.0)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    geom_ids = collision_geom_ids(model)
    if not len(geom_ids):
        raise ValueError("robot model has no non-world collision geoms")
    model_report = model_description(model_path, payload, model, getattr(args, "training_model", None))
    source_profile_path = Path(__file__).resolve().parents[1] / "robot_models/unitree_g1/unitree_g1_mesh.json"
    if source_profile_path.is_file():
        from omniretargeting.robot_config import load_robot_config
        profile_model = Path(load_robot_config(source_profile_path)["urdf_path"])
        model_report["source_profile"] = {
            "path": str(source_profile_path), "sha256": _sha256(source_profile_path),
            "model_path": str(profile_model),
            "match": profile_model.resolve() == model_path or _sha256(profile_model) == _sha256(model_path),
            "interpretation": "declared mesh collision contract; runtime loading requires separate provenance",
        }
    interp = interpolation_profile(payload)
    saved = TrainingTrajectory(
        poses.copy(),
        np.arange(len(poses), dtype=float),
        np.arange(len(poses), dtype=int) + frame_start,
        {"status": "input_frames"},
    )
    training = training_trajectory(model, poses, framerate, interp, frame_start)
    raw_objects = np.load(objects_path, allow_pickle=True).tolist() if objects_path.is_file() else []
    objects = _slice_objects(raw_objects, frame_start, len(poses))
    if args.scene_metrics:
        static, moving, scene_reuse = consume_scene_metrics(
            args.scene_metrics, motion_path, prepared, objects_path, model, saved, geom_ids
        )
        moving_query = MovingSceneQuery(model, objects, len(poses)) if objects else None
    else:
        scene_reuse = {"status": "not_requested"}
        static = audit_static(model, saved, prepared, geom_ids)
        moving, moving_query = audit_moving(model, saved, objects, geom_ids)
    self_report, self_query = audit_self(model, saved, geom_ids, getattr(args, "workers", 1))
    interframe = audit_interframe(
        model, saved, training, interp, static, moving, self_report,
        moving_query, self_query, geom_ids, prepared
    )
    if scene_reuse.get("status") == "consumed":
        interframe["risk_selection"] = "endpoint-positive intervals only; consumed report lacks clearance"
    reset = audit_reset(payload, reset_profile(payload), saved, static, moving, self_report)
    elapsed = time.perf_counter() - started
    report = {
        "schema_version": SCHEMA_VERSION,
        "audit_mode": "measure_only_no_optimization",
        "inputs": {
            "prepared": str(prepared),
            "motion": str(motion_path),
            "objects": str(objects_path),
            "frame_start": frame_start,
            "motion_payload_framerate_hz": framerate,
            "input_frames": int(len(poses)),
            "audited_frames": int(len(saved.qpos)),
            "interpolation_output_frames": int(len(training.qpos)),
        },
        "model": model_report,
        "training_interpolation": interp,
        "scene_metrics_reuse": scene_reuse,
        "static_scene": static,
        "moving_objects": moving,
        "self_collision": self_report,
        "interframe_risk_audit": interframe,
        "reset": reset,
        **_global_status(static, moving, self_report, model_report),
        "timing": {
            "wall_seconds": elapsed,
            "prior_scene_metrics_seconds": float(scene_reuse.get("receipt_scene_metrics_seconds", 0.0)),
            "total_including_prior_scene_seconds": elapsed + float(scene_reuse.get("receipt_scene_metrics_seconds", 0.0)),
            "seconds_per_audited_frame": elapsed / max(1, len(saved.qpos)),
        },
    }
    from scripts.training_collision_output import write_report
    write_report(args.output, report)
    return report
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--objects", type=Path, required=True)
    parser.add_argument("--scene-metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-model", type=Path)
    parser.add_argument("--workers", type=int, default=min(8, available_cpus()))
    report = audit(parser.parse_args())
    print("SAVED", report["inputs"]["motion"], report["geometry_status"], report["acceptance_status"])
if __name__ == "__main__":
    main()
