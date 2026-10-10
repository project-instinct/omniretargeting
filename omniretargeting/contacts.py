"""Source-independent scene poses, sparse contacts, and geometric detection.

Lengths are in the motion's units; quaternions are local-to-world, wxyz.
Geometry and contact anchors share the contacted rigid body's local frame.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

if TYPE_CHECKING:
    from .data_sources.base import MotionData, MotionFrame


def _vector(value, shape, name):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must have finite shape {shape}.")
    return value


def _identifier(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")


def _unit_quaternions(value, name):
    if not np.allclose(np.linalg.norm(value, axis=-1), 1.0, atol=1e-5):
        raise ValueError(f"{name} must contain unit wxyz quaternions.")


@dataclass
class Contact:
    source_point_id: str
    scene_body_id: str
    point_local: np.ndarray
    confidence: float | None = None
    normal_local: np.ndarray | None = None

    def __post_init__(self):
        _identifier(self.source_point_id, "Contact.source_point_id")
        _identifier(self.scene_body_id, "Contact.scene_body_id")
        self.point_local = _vector(self.point_local, (3,), "Contact.point_local")
        if self.confidence is not None and (
            not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1
        ):
            raise ValueError("Contact.confidence must be finite and in [0, 1].")
        if self.normal_local is not None:
            self.normal_local = _vector(self.normal_local, (3,), "Contact.normal_local")
            if not np.isclose(np.linalg.norm(self.normal_local), 1.0, atol=1e-5):
                raise ValueError("Contact.normal_local must be a unit vector.")


@dataclass
class EntityPose:
    translation: np.ndarray
    orientation: np.ndarray

    def __post_init__(self):
        self.translation = _vector(self.translation, (3,), "EntityPose.translation")
        self.orientation = _vector(self.orientation, (4,), "EntityPose.orientation")
        _unit_quaternions(self.orientation, "EntityPose.orientation")

    def to_world(self, points):
        return (
            Rotation.from_quat(self.orientation, scalar_first=True).apply(points)
            + self.translation
        )


@dataclass
class EntityTrajectory:
    translations: np.ndarray
    orientations: np.ndarray

    def __post_init__(self):
        self.translations = np.asarray(self.translations, dtype=float)
        if (
            self.translations.ndim != 2
            or self.translations.shape[1] != 3
            or not len(self.translations)
        ):
            raise ValueError(
                "EntityTrajectory.translations must have shape (T, 3), T > 0."
            )
        self.translations = _vector(
            self.translations, self.translations.shape, "EntityTrajectory.translations"
        )
        self.orientations = _vector(
            self.orientations,
            (len(self.translations), 4),
            "EntityTrajectory.orientations",
        )
        _unit_quaternions(self.orientations, "EntityTrajectory.orientations")

    def pose(self, frame_index):
        return EntityPose(
            self.translations[frame_index], self.orientations[frame_index]
        )


@dataclass
class SceneEntity:
    """Body-local triangle geometry and samples; static pose must be explicit."""

    geometry: trimesh.Trimesh | None = None
    local_samples: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    static_pose: EntityPose | None = None

    def __post_init__(self):
        self.local_samples = np.asarray(self.local_samples, dtype=float)
        if (
            self.local_samples.ndim != 2
            or self.local_samples.shape[1] != 3
            or not np.isfinite(self.local_samples).all()
        ):
            raise ValueError("SceneEntity.local_samples must have finite shape (N, 3).")


@dataclass
class Scene:
    entities: dict[str, SceneEntity]

    def __post_init__(self):
        for body_id, entity in self.entities.items():
            _identifier(body_id, "Scene entity ID")
            if not isinstance(entity, SceneEntity):
                raise ValueError(f"Scene entity {body_id!r} must be a SceneEntity.")

    def pose(self, body_id, entity_poses):
        if body_id not in self.entities:
            raise ValueError(f"Unknown scene body {body_id!r}.")
        pose = (entity_poses or {}).get(body_id, self.entities[body_id].static_pose)
        if pose is None:
            raise ValueError(
                f"Scene body {body_id!r} has no trajectory or explicit static pose."
            )
        return pose


def validate_frame_contacts(target_names, scene, entity_poses, contacts):
    """Validate endpoints even for contacts that will not be used by the solver."""
    for body_id, pose in (entity_poses or {}).items():
        if scene is None or body_id not in scene.entities:
            raise ValueError(f"Entity pose references unknown scene body {body_id!r}.")
        if not isinstance(pose, EntityPose):
            raise ValueError(f"Entity pose {body_id!r} must be an EntityPose.")
    if scene is not None:
        for body_id in scene.entities:
            scene.pose(body_id, entity_poses)
    if contacts is not None:
        if not isinstance(contacts, list):
            raise ValueError(
                "A contact frame must be a list of Contact records or None."
            )
        for contact in contacts:
            if not isinstance(contact, Contact):
                raise ValueError("Contact frames must contain Contact records.")
            if target_names is None or contact.source_point_id not in target_names:
                raise ValueError(
                    f"Unknown contact source point {contact.source_point_id!r}."
                )
            if scene is None:
                raise ValueError(
                    f"Contact body {contact.scene_body_id!r} requires a scene."
                )
            scene.pose(contact.scene_body_id, entity_poses)


def detect_contacts(
    motion: MotionData, config: dict
) -> list[list[Contact] | None] | None:
    """Infer binary contact from mesh distance and optional body-relative speed.

    One nearest surface anchor per candidate point/body/frame is emitted. Filtering
    removes short runs per pair; anchors stay per-frame, so sliding is preserved.
    Existing available annotations (including []) survive unless overwrite=True.
    """
    import open3d as o3d

    allowed = {
        "enabled",
        "pairs",
        "distance_threshold",
        "velocity_threshold",
        "min_contact_frames",
        "overwrite",
    }
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown contact_detection options: {sorted(unknown)}")
    if not config.get("enabled", True):
        return deepcopy(motion.contact_trajectory)
    if motion.scene is None or motion.target_names is None:
        raise ValueError(
            "Contact detection requires scene geometry and named human targets."
        )
    distance_threshold = float(config.get("distance_threshold", 0.10))
    velocity_threshold = config.get("velocity_threshold", 0.30)
    min_frames = config.get("min_contact_frames", 3)
    if not np.isfinite(distance_threshold) or distance_threshold <= 0:
        raise ValueError(
            "contact_detection.distance_threshold must be finite and positive."
        )
    if velocity_threshold is not None:
        velocity_threshold = float(velocity_threshold)
        if not np.isfinite(velocity_threshold) or velocity_threshold < 0:
            raise ValueError(
                "contact_detection.velocity_threshold must be finite and non-negative or null."
            )
        if (
            motion.framerate is None
            or not np.isfinite(motion.framerate)
            or motion.framerate <= 0
        ):
            raise ValueError(
                "Velocity-based contact detection requires a positive framerate."
            )
    if not isinstance(min_frames, int) or min_frames < 1:
        raise ValueError(
            "contact_detection.min_contact_frames must be a positive integer."
        )
    pairs = config.get("pairs")
    if pairs is None:
        pairs = {body_id: motion.target_names for body_id in motion.scene.entities}
    if not isinstance(pairs, dict):
        raise ValueError(
            "contact_detection.pairs must map scene body IDs to source point lists."
        )
    result = [[] for _ in motion.positions]
    trajectories = motion.entity_trajectories or {}
    for body_id, point_names in pairs.items():
        if body_id not in motion.scene.entities:
            raise ValueError(
                f"Contact detector references unknown scene body {body_id!r}."
            )
        if (
            not isinstance(point_names, list)
            or not point_names
            or len(set(point_names)) != len(point_names)
        ):
            raise ValueError(
                f"Detector points for {body_id!r} must be a non-empty list of unique target names."
            )
        for name in point_names:
            if name not in motion.target_names:
                raise ValueError(
                    f"Contact detector references unknown source point {name!r}."
                )
        mesh = motion.scene.entities[body_id].geometry
        if mesh is None or len(mesh.faces) == 0:
            raise ValueError(
                f"Contact detection requires triangle geometry for scene body {body_id!r}."
            )
        points = motion.positions[
            :, [motion.target_names.index(name) for name in point_names]
        ]
        trajectory = trajectories.get(body_id)
        if trajectory is not None:
            rotations = Rotation.from_quat(
                trajectory.orientations, scalar_first=True
            ).as_matrix()
            translations = trajectory.translations
        else:
            pose = motion.scene.pose(body_id, {})
            rotations = np.broadcast_to(
                Rotation.from_quat(pose.orientation, scalar_first=True).as_matrix(),
                (len(points), 3, 3),
            )
            translations = np.broadcast_to(pose.translation, (len(points), 3))
        # World -> body-local; a sticking point on a moving body has zero speed.
        local_points = np.einsum(
            "tji,tkj->tki", rotations, points - translations[:, None]
        )
        query_scene = o3d.t.geometry.RaycastingScene()
        query_scene.add_triangles(
            o3d.core.Tensor(np.ascontiguousarray(mesh.vertices, dtype=np.float32)),
            o3d.core.Tensor(np.ascontiguousarray(mesh.faces, dtype=np.uint32)),
        )
        nearest = query_scene.compute_closest_points(
            o3d.core.Tensor(
                np.ascontiguousarray(local_points.reshape(-1, 3), dtype=np.float32)
            ),
            nthreads=1,
        )
        anchors = nearest["points"].numpy().reshape(local_points.shape).astype(float)
        face_ids = nearest["primitive_ids"].numpy().reshape(local_points.shape[:2])
        normals = np.asarray(mesh.face_normals)[face_ids]
        mask = np.linalg.norm(local_points - anchors, axis=-1) <= distance_threshold
        if velocity_threshold is not None and len(points) > 1:
            speed = np.linalg.norm(
                np.gradient(local_points, 1.0 / motion.framerate, axis=0), axis=-1
            )
            mask &= speed <= velocity_threshold
        for point_index, name in enumerate(point_names):
            changes = np.diff(np.r_[False, mask[:, point_index], False].astype(int))
            for start, end in zip(
                np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
            ):
                if end - start < min_frames:
                    continue
                for t in range(start, end):
                    normal = normals[t, point_index]
                    result[t].append(
                        Contact(
                            name,
                            body_id,
                            anchors[t, point_index],
                            normal_local=normal if np.linalg.norm(normal) > 0 else None,
                        )
                    )
    if motion.contact_trajectory is not None and not config.get("overwrite", False):
        result = [
            deepcopy(old) if old is not None else new
            for old, new in zip(motion.contact_trajectory, result)
        ]
    return result


def resolve_contact_anchors(frame: MotionFrame, mapped_names: list[str]):
    """Resolve sparse contacts to world anchors and mapped point/anchor edges.

    Duplicate anchors per pair use maximum confidence. Distinct active anchors
    become one confidence-weighted centroid with their mean confidence, so patch
    density adds neither edge budget nor reverse Laplacian residual rows. Stored
    surface annotations remain unchanged. Anchors are shared only within one body.
    """
    grouped = {}
    for contact in frame.contacts or []:
        if contact.source_point_id not in mapped_names:
            raise ValueError(
                f"Contact source point {contact.source_point_id!r} has no robot link-point mapping."
            )
        pair = (mapped_names.index(contact.source_point_id), contact.scene_body_id)
        key = tuple(contact.point_local)
        confidence = 1.0 if contact.confidence is None else float(contact.confidence)
        points = grouped.setdefault(pair, {})
        points[key] = max(points.get(key, 0.0), confidence)
    anchors, edges, anchor_indices = [], [], {}
    for (source_index, body_id), points in grouped.items():
        active = {
            point: confidence for point, confidence in points.items() if confidence > 0
        }
        if not active:
            continue
        confidences = np.array(list(active.values()))
        point = np.average(np.array(list(active)), axis=0, weights=confidences)
        key = (body_id, tuple(point))
        if key not in anchor_indices:
            anchor_indices[key] = len(anchors)
            anchors.append(
                frame.scene.pose(body_id, frame.entity_poses).to_world(point)
            )
        edges.append((source_index, anchor_indices[key], float(confidences.mean())))
    return np.asarray(anchors, dtype=float).reshape(-1, 3), edges


def contact_to_dict(contact):
    result = {
        "source_point_id": contact.source_point_id,
        "scene_body_id": contact.scene_body_id,
        "point_local": contact.point_local.tolist(),
    }
    if contact.confidence is not None:
        result["confidence"] = float(contact.confidence)
    if contact.normal_local is not None:
        result["normal_local"] = contact.normal_local.tolist()
    return result


def save_hsoi_annotations(
    motion: MotionData, path: str | Path, source_to_robot_scale: float = 1.0
):
    """Write reviewable annotations and poses; scene geometry remains separate."""
    result = {
        "framerate": motion.framerate,
        "source_to_robot_scale": source_to_robot_scale,
        "quaternion_convention": "wxyz",
        "target_names": motion.target_names,
        "contact_trajectory": (
            None
            if motion.contact_trajectory is None
            else [
                (
                    None
                    if contacts is None
                    else [contact_to_dict(contact) for contact in contacts]
                )
                for contacts in motion.contact_trajectory
            ]
        ),
        "entity_trajectories": {
            body_id: {
                "translations": track.translations.tolist(),
                "orientations": track.orientations.tolist(),
            }
            for body_id, track in (motion.entity_trajectories or {}).items()
        },
        "static_poses": {
            body_id: {
                "translation": entity.static_pose.translation.tolist(),
                "orientation": entity.static_pose.orientation.tolist(),
            }
            for body_id, entity in (
                motion.scene.entities if motion.scene is not None else {}
            ).items()
            if entity.static_pose is not None
        },
    }
    Path(path).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def load_contact_trajectory(path: str | Path, source_coordinates: bool = False):
    """Read the contact field from an HSOI JSON file for editing/reuse."""
    payload = json.loads(Path(path).expanduser().read_text())
    result = payload["contact_trajectory"]
    contacts = (
        None
        if result is None
        else [
            None if contacts is None else [Contact(**contact) for contact in contacts]
            for contacts in result
        ]
    )
    if source_coordinates:
        scale = float(payload.get("source_to_robot_scale", 1.0))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                "Annotation source_to_robot_scale must be finite and positive."
            )
        for frame in contacts or []:
            for contact in frame or []:
                contact.point_local = contact.point_local / scale
    return contacts
