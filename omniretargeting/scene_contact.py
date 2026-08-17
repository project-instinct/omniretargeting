"""Infer where the source touches the scene, and anchor those contacts.

Preserving an interaction means the robot has to touch what the source touched,
where the source touched it.  A Laplacian interaction mesh preserves relative
shape, not contact: it is free to place a whole limb a few centimetres off the
surface at no cost, because every edge around it stretches by the same small
amount.  Non-penetration constraints do not close that gap either, since
hovering above a surface violates nothing.

So contact has to be an explicit objective.  This module answers two questions
from the source motion alone:

* when is an end effector in contact?  Proximity by itself is not enough: a
  foot passing an inch above the floor is close but not planted.  Contact
  additionally requires the effector to be moving slowly, and to stay that way
  for a few consecutive frames, which is what separates a plant from a
  fly-by.
* where is that contact?  Each contact interval is anchored to a single point
  on the scene surface, taken from the opening frames of the interval, and
  every frame in the interval targets that same point.  Anchoring per interval
  rather than per frame is what stops the effector sliding: a per-frame
  projection would happily track a foot skating across the floor, since each
  individual frame is close to the surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = ["ContactSpec", "infer_contact_targets", "contact_runs"]


@dataclass(frozen=True)
class ContactSpec:
    """Thresholds for calling a frame "in contact"."""

    distance_threshold: float = 0.05
    speed_threshold: float = 0.6
    min_frames: int = 3
    anchor_frames: int = 3

    def __post_init__(self) -> None:
        if self.distance_threshold <= 0.0:
            raise ValueError("distance_threshold must be positive")
        if self.speed_threshold <= 0.0:
            raise ValueError("speed_threshold must be positive")
        if self.min_frames < 1:
            raise ValueError("min_frames must be at least 1")
        if self.anchor_frames < 1:
            raise ValueError("anchor_frames must be at least 1")


def contact_runs(flags: np.ndarray, min_frames: int) -> List[Tuple[int, int]]:
    """Half-open ``[start, end)`` runs of consecutive True at least ``min_frames`` long."""
    padded = np.concatenate(([0], np.asarray(flags, dtype=np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded) != 0).reshape(-1, 2)
    return [(int(a), int(b)) for a, b in edges if b - a >= min_frames]


def infer_contact_targets(
    effector_points: Dict[str, np.ndarray],
    terrain,
    framerate: Optional[float],
    spec: Optional[ContactSpec] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], dict]:
    """Anchor every contact interval of every effector to the scene.

    Args:
        effector_points: ``{name: (T, 3)}`` world positions of the point that
            would touch the scene, in the same frame as ``terrain``.
        terrain: a trimesh surface.
        framerate: used to convert per-frame displacement into speed. When it
            is unknown the speed gate is skipped and proximity alone decides.

    Returns:
        ``(targets, normals, stats)``. ``targets[name]`` is ``(T, 3)`` with NaN
        on frames that are not in contact; ``normals[name]`` is the anchored
        surface normal, also NaN outside contact.
    """
    spec = spec or ContactSpec()
    targets: Dict[str, np.ndarray] = {}
    normals: Dict[str, np.ndarray] = {}
    stats: dict = {}

    for name, points in effector_points.items():
        points = np.asarray(points, dtype=float)
        n_frames = len(points)
        target = np.full((n_frames, 3), np.nan)
        normal = np.full((n_frames, 3), np.nan)

        closest, distance, face_ids = terrain.nearest.on_surface(points)
        distance = np.asarray(distance, dtype=float)

        if framerate:
            step = np.linalg.norm(np.diff(points, axis=0), axis=1) * float(framerate)
            speed = np.concatenate(([step[0] if len(step) else 0.0], step))
            close_and_slow = (distance < spec.distance_threshold) & (speed < spec.speed_threshold)
        else:
            close_and_slow = distance < spec.distance_threshold

        runs = contact_runs(close_and_slow, spec.min_frames)
        face_normals = terrain.face_normals[np.asarray(face_ids, dtype=int)]

        for start, end in runs:
            window = slice(start, min(start + spec.anchor_frames, end))
            anchor = np.median(closest[window], axis=0)
            anchor_normal = np.median(face_normals[window], axis=0)
            norm = np.linalg.norm(anchor_normal)
            anchor_normal = anchor_normal / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0])
            target[start:end] = anchor
            normal[start:end] = anchor_normal

        targets[name] = target
        normals[name] = normal
        stats[name] = {
            "contact_frames": int(np.isfinite(target[:, 0]).sum()),
            "runs": len(runs),
            "median_distance": float(np.median(distance)),
        }

    return targets, normals, stats
