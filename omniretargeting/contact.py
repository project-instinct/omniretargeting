"""Scene-contact inference for contact-prioritized motion retargeting.

The source motion provides body-surface samples for the four end effectors.
This module turns those samples into temporally filtered contact schedules and
world-frame contact anchors on a static scene mesh.  The resulting schedule is
consumed by the kinematic optimizer and can also be exported for downstream
motion-tracking policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Mapping

import numpy as np
import trimesh


class ContactMode(IntEnum):
    """Semantic state of one end effector at one source frame."""

    NONE = 0
    TOUCH = 1
    SLIDE = 2
    SUPPORT = 3


def _parse_contact_mode(value: Any) -> ContactMode:
    if isinstance(value, ContactMode):
        return value
    if isinstance(value, (int, np.integer)):
        return ContactMode(int(value))
    name = str(value).strip().upper()
    if name == "AUTO":
        raise ValueError("AUTO must be resolved from a complete contact segment.")
    try:
        return ContactMode[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown contact mode {value!r}; expected NONE, TOUCH, SLIDE, "
            "SUPPORT, or AUTO."
        ) from exc


@dataclass(frozen=True)
class SceneContactTask:
    """Per-frame reference for one human/robot end-effector pair."""

    name: str
    tracking_positions: np.ndarray
    contact_positions: np.ndarray
    surface_points: np.ndarray
    surface_normals: np.ndarray
    distances: np.ndarray
    relative_speeds: np.ndarray
    contact_confidence: np.ndarray
    contact_modes: np.ndarray
    contact_segment_ids: np.ndarray
    transition_weights: np.ndarray
    normal_speeds: np.ndarray | None = None
    tangential_speeds: np.ndarray | None = None
    forward_directions: np.ndarray | None = None
    up_directions: np.ndarray | None = None
    # ``effective_swing_positions`` is an export/visualisation preview.  For a
    # limb-relative task it is evaluated at the reference base pose; the
    # optimizer must instead use the live anchor fields below.
    effective_swing_positions: np.ndarray | None = None
    swing_limb_offsets: np.ndarray | None = None
    swing_anchor_body: str | None = None
    swing_anchor_offset: np.ndarray | None = None
    contact_bridge_alpha: np.ndarray | None = None
    contact_bridge_positions: np.ndarray | None = None
    # Finished world-frame orientation targets.  Both optimizers consume this
    # matrix track directly instead of independently reconstructing a rotation
    # from the diagnostic source forward/up axes.
    orientation_target_rotations: np.ndarray | None = None
    orientation_bridge_alpha: np.ndarray | None = None
    position_sigma_scale: np.ndarray | None = None
    orientation_sigma_scale: np.ndarray | None = None
    morphology_scale: float | None = None
    # Robot reach is measured from the configured same-side reference body to
    # this effector.  The source chain carries semantic left/right identity and
    # lets refinement preserve the source leg-extension fraction without ever
    # choosing a target by nearest distance.
    robot_max_reach: float | None = None
    source_limb_chain: tuple[str, str, str] | None = None
    active_contact_offset_ids: np.ndarray | None = None
    robot_base_reference_w: np.ndarray | None = None

    def __post_init__(self) -> None:
        num_frames = len(self.tracking_positions)
        for field_name in (
            "tracking_positions",
            "contact_positions",
            "surface_points",
            "surface_normals",
        ):
            value = np.asarray(getattr(self, field_name), dtype=float)
            if value.shape != (num_frames, 3):
                raise ValueError(
                    f"{self.name}: {field_name} must have shape ({num_frames}, 3)."
                )
            object.__setattr__(self, field_name, value)
        for field_name in (
            "distances",
            "relative_speeds",
            "contact_confidence",
        ):
            value = np.asarray(getattr(self, field_name), dtype=float)
            if value.shape != (num_frames,):
                raise ValueError(
                    f"{self.name}: {field_name} must have shape ({num_frames},)."
                )
            object.__setattr__(self, field_name, value)
        for field_name in ("normal_speeds", "tangential_speeds"):
            value = getattr(self, field_name)
            if value is None:
                continue
            value = np.asarray(value, dtype=float)
            if value.shape != (num_frames,):
                raise ValueError(
                    f"{self.name}: {field_name} must have shape ({num_frames},)."
                )
            object.__setattr__(self, field_name, value)

        modes = np.asarray(self.contact_modes, dtype=np.uint8)
        if modes.shape != (num_frames,):
            raise ValueError(
                f"{self.name}: contact_modes must have shape ({num_frames},)."
            )
        if np.any(modes > int(ContactMode.SUPPORT)):
            raise ValueError(f"{self.name}: contact_modes contains an invalid mode id.")
        object.__setattr__(self, "contact_modes", modes)

        segment_ids = np.asarray(self.contact_segment_ids, dtype=np.int32)
        if segment_ids.shape != (num_frames,):
            raise ValueError(
                f"{self.name}: contact_segment_ids must have shape ({num_frames},)."
            )
        in_contact = modes != int(ContactMode.NONE)
        if np.any(segment_ids[in_contact] < 0) or np.any(segment_ids[~in_contact] != -1):
            raise ValueError(
                f"{self.name}: contact frames require a non-negative segment id "
                "and NONE frames require segment id -1."
            )
        object.__setattr__(self, "contact_segment_ids", segment_ids)

        transition_weights = np.asarray(
            self.transition_weights, dtype=float
        ).copy()
        if transition_weights.shape != (num_frames,):
            raise ValueError(
                f"{self.name}: transition_weights must have shape ({num_frames},)."
            )
        if np.any((transition_weights < 0.0) | (transition_weights > 1.0)):
            raise ValueError(f"{self.name}: transition_weights must lie in [0, 1].")
        if np.any(
            transition_weights[modes == int(ContactMode.SUPPORT)] != 1.0
        ):
            raise ValueError(
                f"{self.name}: SUPPORT transition_weights must equal 1."
            )
        object.__setattr__(self, "transition_weights", transition_weights)

        for field_name in (
            "forward_directions",
            "up_directions",
            "effective_swing_positions",
            "swing_limb_offsets",
            "contact_bridge_positions",
            "robot_base_reference_w",
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            value = np.asarray(value, dtype=float)
            if value.shape != (num_frames, 3):
                raise ValueError(
                    f"{self.name}: {field_name} must have shape ({num_frames}, 3)."
                )
            object.__setattr__(self, field_name, value)
        if self.contact_bridge_alpha is not None:
            alpha = np.asarray(self.contact_bridge_alpha, dtype=float)
            if alpha.shape != (num_frames,):
                raise ValueError(
                    f"{self.name}: contact_bridge_alpha must have shape "
                    f"({num_frames},)."
                )
            if np.any((alpha < 0.0) | (alpha > 1.0)):
                raise ValueError(
                    f"{self.name}: contact_bridge_alpha must lie in [0, 1]."
                )
            object.__setattr__(self, "contact_bridge_alpha", alpha)
        if self.orientation_bridge_alpha is not None:
            orientation_alpha = np.asarray(
                self.orientation_bridge_alpha, dtype=float
            )
            if orientation_alpha.shape != (num_frames,):
                raise ValueError(
                    f"{self.name}: orientation_bridge_alpha must have shape "
                    f"({num_frames},)."
                )
            if np.any((orientation_alpha < 0.0) | (orientation_alpha > 1.0)):
                raise ValueError(
                    f"{self.name}: orientation_bridge_alpha must lie in [0, 1]."
                )
            object.__setattr__(
                self, "orientation_bridge_alpha", orientation_alpha
            )
        if self.orientation_target_rotations is not None:
            rotations = np.asarray(self.orientation_target_rotations, dtype=float)
            if rotations.shape != (num_frames, 3, 3):
                raise ValueError(
                    f"{self.name}: orientation_target_rotations must have shape "
                    f"({num_frames}, 3, 3)."
                )
            finite = np.isfinite(rotations)
            valid = np.all(finite, axis=(1, 2))
            if np.any(np.any(finite, axis=(1, 2)) != valid):
                raise ValueError(
                    f"{self.name}: unavailable orientation matrices must be all NaN."
                )
            if np.any(valid):
                gram = np.swapaxes(rotations[valid], 1, 2) @ rotations[valid]
                determinants = np.linalg.det(rotations[valid])
                if not np.allclose(gram, np.eye(3), atol=1e-7) or not np.allclose(
                    determinants, 1.0, atol=1e-7
                ):
                    raise ValueError(
                        f"{self.name}: orientation_target_rotations must be "
                        "right-handed orthonormal matrices."
                    )
            object.__setattr__(self, "orientation_target_rotations", rotations)
        for field_name in ("position_sigma_scale", "orientation_sigma_scale"):
            value = getattr(self, field_name)
            if value is None:
                continue
            scale = np.asarray(value, dtype=float)
            if scale.shape != (num_frames,):
                raise ValueError(
                    f"{self.name}: {field_name} must have shape ({num_frames},)."
                )
            if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
                raise ValueError(f"{self.name}: {field_name} must be positive.")
            object.__setattr__(self, field_name, scale)
        if self.swing_anchor_offset is not None:
            anchor_offset = np.asarray(self.swing_anchor_offset, dtype=float)
            if anchor_offset.shape != (3,):
                raise ValueError(
                    f"{self.name}: swing_anchor_offset must have shape (3,)."
                )
            object.__setattr__(self, "swing_anchor_offset", anchor_offset)
        if (self.contact_bridge_alpha is None) != (
            self.contact_bridge_positions is None
        ):
            raise ValueError(
                f"{self.name}: contact_bridge_alpha and "
                "contact_bridge_positions must be provided together."
            )
        relative_fields = (self.swing_limb_offsets, self.swing_anchor_body)
        if any(value is not None for value in relative_fields) and not all(
            value is not None for value in relative_fields
        ):
            raise ValueError(
                f"{self.name}: limb-relative swing tasks require "
                "swing_limb_offsets and swing_anchor_body together."
            )
        if self.swing_limb_offsets is not None and (
            self.contact_bridge_alpha is None
            or self.contact_bridge_positions is None
        ):
            raise ValueError(
                f"{self.name}: limb-relative swing tasks also require the "
                "contact bridge alpha and world target tracks."
            )
        if self.active_contact_offset_ids is not None:
            offset_ids = np.asarray(self.active_contact_offset_ids, dtype=np.int32)
            if offset_ids.shape != (num_frames,):
                raise ValueError(
                    f"{self.name}: active_contact_offset_ids must have shape "
                    f"({num_frames},)."
                )
            if np.any(offset_ids[~in_contact] != -1) or np.any(
                offset_ids[in_contact] < 0
            ):
                raise ValueError(
                    f"{self.name}: active contact offsets require a non-negative "
                    "id on contact frames and -1 on NONE frames."
                )
            object.__setattr__(self, "active_contact_offset_ids", offset_ids)
        if self.morphology_scale is not None and float(self.morphology_scale) <= 0.0:
            raise ValueError(f"{self.name}: morphology_scale must be positive.")
        if self.robot_max_reach is not None and (
            not np.isfinite(float(self.robot_max_reach))
            or float(self.robot_max_reach) <= 0.0
        ):
            raise ValueError(f"{self.name}: robot_max_reach must be positive.")
        if self.source_limb_chain is not None:
            chain = tuple(str(joint) for joint in self.source_limb_chain)
            if len(chain) != 3:
                raise ValueError(
                    f"{self.name}: source_limb_chain must contain three joints."
                )
            object.__setattr__(self, "source_limb_chain", chain)

    def frame(self, frame_idx: int) -> Dict[str, Any]:
        """Return the optimizer payload for one frame."""
        mode = ContactMode(int(self.contact_modes[frame_idx]))
        payload = {
            "name": self.name,
            "tracking_position": self.tracking_positions[frame_idx],
            "contact_position": self.contact_positions[frame_idx],
            "surface_normal": self.surface_normals[frame_idx],
            "distance": float(self.distances[frame_idx]),
            "relative_speed": float(self.relative_speeds[frame_idx]),
            "confidence": float(self.contact_confidence[frame_idx]),
            "contact_mode": mode.name,
            "contact_segment_id": int(self.contact_segment_ids[frame_idx]),
            "transition_weight": float(self.transition_weights[frame_idx]),
        }
        if self.normal_speeds is not None:
            payload["normal_speed"] = float(self.normal_speeds[frame_idx])
        if self.tangential_speeds is not None:
            payload["tangential_speed"] = float(
                self.tangential_speeds[frame_idx]
            )
        if mode == ContactMode.NONE:
            if self.effective_swing_positions is None:
                raise RuntimeError(
                    f"{self.name}: effective swing targets were not prepared."
                )
            payload["effective_swing_position"] = self.effective_swing_positions[
                frame_idx
            ]
            if self.swing_limb_offsets is not None:
                payload.update(
                    {
                        "swing_limb_offset": self.swing_limb_offsets[frame_idx],
                        "swing_anchor_body": self.swing_anchor_body,
                        "swing_anchor_offset": (
                            np.zeros(3, dtype=float)
                            if self.swing_anchor_offset is None
                            else self.swing_anchor_offset
                        ),
                        "contact_bridge_alpha": float(
                            self.contact_bridge_alpha[frame_idx]
                        ),
                        "contact_bridge_position": self.contact_bridge_positions[
                            frame_idx
                        ],
                    }
                )
        if self.robot_base_reference_w is not None:
            payload["robot_base_reference_w"] = self.robot_base_reference_w[
                frame_idx
            ]
        if self.forward_directions is not None:
            payload["forward_direction"] = self.forward_directions[frame_idx]
        if self.up_directions is not None:
            payload["up_direction"] = self.up_directions[frame_idx]
        if self.orientation_target_rotations is not None:
            payload["orientation_target_rotation"] = (
                self.orientation_target_rotations[frame_idx]
            )
        if self.orientation_bridge_alpha is not None:
            payload["orientation_bridge_alpha"] = float(
                self.orientation_bridge_alpha[frame_idx]
            )
        if self.position_sigma_scale is not None:
            payload["position_sigma_scale"] = float(
                self.position_sigma_scale[frame_idx]
            )
        if self.orientation_sigma_scale is not None:
            payload["orientation_sigma_scale"] = float(
                self.orientation_sigma_scale[frame_idx]
            )
        return payload


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-value))


def _iter_true_runs(mask: np.ndarray):
    start = None
    for idx, value in enumerate(np.r_[mask, False]):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            yield start, idx
            start = None


def _close_short_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    if max_gap_frames <= 0:
        return result
    false_runs = list(_iter_true_runs(~result))
    for start, end in false_runs:
        if start == 0 or end == len(result):
            continue
        if end - start <= max_gap_frames:
            result[start:end] = True
    return result


def _remove_short_runs(mask: np.ndarray, min_contact_frames: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    if min_contact_frames <= 1:
        return result
    for start, end in list(_iter_true_runs(result)):
        if end - start < min_contact_frames:
            result[start:end] = False
    return result


def _hysteresis_contact_mask(
    distances: np.ndarray,
    speeds: np.ndarray,
    *,
    enter_distance: float,
    exit_distance: float,
    enter_speed: float,
    exit_speed: float,
) -> np.ndarray:
    """Apply Schmitt-trigger thresholds to a contact observation track."""
    distances = np.asarray(distances, dtype=float)
    speeds = np.asarray(speeds, dtype=float)
    if distances.shape != speeds.shape or distances.ndim != 1:
        raise ValueError("distances and speeds must share shape (T,).")
    active = False
    result = np.zeros(len(distances), dtype=bool)
    for frame_idx, (distance, speed) in enumerate(zip(distances, speeds)):
        if active:
            active = bool(distance <= exit_distance and speed <= exit_speed)
        else:
            active = bool(distance <= enter_distance and speed <= enter_speed)
        result[frame_idx] = active
    return result


def _classify_auto_contact_modes(
    candidate_mask: np.ndarray,
    surface_points: np.ndarray,
    surface_normals: np.ndarray,
    tangential_speeds: np.ndarray,
    *,
    touch_max_frames: int,
    support_speed_threshold: float,
    support_max_tangential_travel: float,
    support_min_frames: int,
    support_max_gap_frames: int,
) -> np.ndarray:
    """Split proximity contact into transient, sliding, and sticking spans.

    Contact continuity is governed by distance and normal velocity before this
    function is called. Tangential motion is deliberately used only for the
    semantic split: stationary sub-runs become no-slip SUPPORT anchors while
    the remaining contact frames retain tangential freedom as SLIDE. This is
    important for a heel-to-toe or hand-on-surface trajectory where one long
    proximity run may legitimately contain both modes.
    """
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    surface_points = np.asarray(surface_points, dtype=float)
    surface_normals = np.asarray(surface_normals, dtype=float)
    tangential_speeds = np.asarray(tangential_speeds, dtype=float)
    num_frames = len(candidate_mask)
    if surface_points.shape != (num_frames, 3):
        raise ValueError("surface_points must have shape (T, 3).")
    if surface_normals.shape != (num_frames, 3):
        raise ValueError("surface_normals must have shape (T, 3).")
    if tangential_speeds.shape != (num_frames,):
        raise ValueError("tangential_speeds must have shape (T,).")

    modes = np.full(num_frames, int(ContactMode.NONE), dtype=np.uint8)
    for contact_start, contact_end in _iter_true_runs(candidate_mask):
        if contact_end - contact_start <= touch_max_frames:
            modes[contact_start:contact_end] = int(ContactMode.TOUCH)
            continue

        # Proximity and low normal separation already establish contact.
        # Default to sliding, then promote only verified low-tangential-motion
        # sub-runs to no-slip support.
        modes[contact_start:contact_end] = int(ContactMode.SLIDE)
        support_mask = (
            tangential_speeds[contact_start:contact_end]
            <= support_speed_threshold
        )
        support_mask = _close_short_gaps(
            support_mask, support_max_gap_frames
        )
        support_mask = _remove_short_runs(support_mask, support_min_frames)

        for local_start, local_end in _iter_true_runs(support_mask):
            start = contact_start + local_start
            end = contact_start + local_end
            normal = np.median(surface_normals[start:end], axis=0)
            normal_norm = float(np.linalg.norm(normal))
            if normal_norm < 1e-10:
                normal = np.array([0.0, 0.0, 1.0])
            else:
                normal /= normal_norm
            deltas = surface_points[start:end] - surface_points[start]
            tangential_deltas = deltas - np.outer(deltas @ normal, normal)
            tangential_travel = float(
                np.max(np.linalg.norm(tangential_deltas, axis=1), initial=0.0)
            )
            if tangential_travel <= support_max_tangential_travel:
                modes[start:end] = int(ContactMode.SUPPORT)

    return modes


def _closest_scene_samples(
    scene_mesh: trimesh.Trimesh,
    contact_samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Find the closest source sample and scene point at every frame."""
    samples = np.asarray(contact_samples, dtype=float)
    if samples.ndim == 2:
        samples = samples[:, None, :]
    if samples.ndim != 3 or samples.shape[-1] != 3:
        raise ValueError(
            "contact_samples must have shape (T, K, 3) or (T, 3), "
            f"got {samples.shape}."
        )

    num_frames, num_samples, _ = samples.shape
    flat_samples = samples.reshape(-1, 3)
    try:
        closest, flat_distances, triangle_ids = trimesh.proximity.closest_point(
            scene_mesh,
            flat_samples,
        )
    except (ImportError, ModuleNotFoundError):
        closest, flat_distances, triangle_ids = trimesh.proximity.closest_point_naive(
            scene_mesh,
            flat_samples,
        )

    distances = np.asarray(flat_distances, dtype=float).reshape(num_frames, num_samples)
    closest = np.asarray(closest, dtype=float).reshape(num_frames, num_samples, 3)
    triangle_ids = np.asarray(triangle_ids, dtype=int).reshape(num_frames, num_samples)
    sample_ids = np.argmin(distances, axis=1)
    frame_ids = np.arange(num_frames)

    selected_samples = samples[frame_ids, sample_ids]
    surface_points = closest[frame_ids, sample_ids]
    selected_distances = distances[frame_ids, sample_ids]
    selected_triangles = triangle_ids[frame_ids, sample_ids]

    normals = np.zeros((num_frames, 3), dtype=float)
    valid = (selected_triangles >= 0) & (selected_triangles < len(scene_mesh.face_normals))
    normals[valid] = np.asarray(scene_mesh.face_normals, dtype=float)[selected_triangles[valid]]
    missing = np.linalg.norm(normals, axis=1) < 1e-10
    normals[missing] = selected_samples[missing] - surface_points[missing]
    # Exact on-surface queries can have both an invalid triangle id and a zero
    # point-to-surface offset.  Fall back to the nearest mesh vertex normal so
    # a valid contact never silently loses its sole-orientation constraint.
    missing = np.linalg.norm(normals, axis=1) < 1e-10
    if np.any(missing) and len(scene_mesh.vertices):
        _, vertex_ids = scene_mesh.kdtree.query(selected_samples[missing])
        normals[missing] = np.asarray(scene_mesh.vertex_normals, dtype=float)[
            np.asarray(vertex_ids, dtype=int)
        ]
    missing = np.linalg.norm(normals, axis=1) < 1e-10
    normals[missing] = np.array([0.0, 0.0, 1.0])
    normal_norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(normal_norms, 1e-10)

    # Mesh winding is not guaranteed for reconstructed TSDF scenes.  Orient the
    # normal toward the source body point whenever the unsigned distance gives
    # us an unambiguous direction.
    offsets = selected_samples - surface_points
    flip = np.sum(normals * offsets, axis=1) < 0.0
    normals[flip] *= -1.0

    return selected_samples, surface_points, normals, selected_distances


def infer_scene_contact_tasks(
    marker_trajectories: Mapping[str, Mapping[str, np.ndarray]],
    scene_mesh: trimesh.Trimesh,
    framerate: float,
    config: Mapping[str, Any] | None = None,
) -> Dict[str, SceneContactTask]:
    """Infer contact schedules and scene anchors for the four end effectors.

    ``marker_trajectories`` contains a semantic center used during swing and a
    compact set of coherent SMPL-X surface samples used for contact detection.
    A contact requires both proximity and low source-surface relative speed.
    Contacts are anchored per contiguous run to suppress source reconstruction
    jitter and produce a no-slip kinematic target.
    """
    cfg = dict(config or {})
    distance_threshold = float(cfg.get("distance_threshold", 0.06))
    speed_threshold = float(cfg.get("relative_speed_threshold", 0.35))
    distance_softness = max(float(cfg.get("distance_softness", 0.01)), 1e-6)
    speed_softness = max(float(cfg.get("speed_softness", 0.08)), 1e-6)
    min_contact_frames = max(int(cfg.get("min_contact_frames", 3)), 1)
    max_gap_frames = max(int(cfg.get("max_gap_frames", 2)), 0)
    transition_frames = max(int(cfg.get("transition_frames", 0)), 0)
    anchor_contact_orientation = bool(cfg.get("anchor_contact_orientation", True))
    anchor_window_frames = max(int(cfg.get("anchor_window_frames", 3)), 1)
    anchor_drift_tolerance = max(float(cfg.get("anchor_drift_tolerance", 0.0)), 0.0)
    clearance = float(cfg.get("surface_clearance", 0.0))
    fps = float(framerate) if framerate and framerate > 0 else 30.0

    per_effector_cfg = dict(cfg.get("effectors", {}) or {})
    tasks: Dict[str, SceneContactTask] = {}

    for name, marker_data in marker_trajectories.items():
        tracking_positions = np.asarray(marker_data["tracking_positions"], dtype=float)
        contact_samples = np.asarray(
            marker_data.get("contact_samples", tracking_positions[:, None, :]),
            dtype=float,
        )
        forward_directions = marker_data.get("forward_directions")
        if forward_directions is not None:
            forward_directions = np.asarray(forward_directions, dtype=float)
            if forward_directions.shape != tracking_positions.shape:
                raise ValueError(
                    f"{name}: forward_directions must have shape "
                    f"{tracking_positions.shape}, got {forward_directions.shape}."
                )
            norms = np.linalg.norm(forward_directions, axis=1, keepdims=True)
            forward_directions = forward_directions / np.maximum(norms, 1e-10)
        up_directions = marker_data.get("up_directions")
        if up_directions is not None:
            up_directions = np.asarray(up_directions, dtype=float)
            if up_directions.shape != tracking_positions.shape:
                raise ValueError(
                    f"{name}: up_directions must have shape "
                    f"{tracking_positions.shape}, got {up_directions.shape}."
                )
            norms = np.linalg.norm(up_directions, axis=1, keepdims=True)
            up_directions = up_directions / np.maximum(norms, 1e-10)
        selected_samples, surface_points, surface_normals, distances = _closest_scene_samples(
            scene_mesh,
            contact_samples,
        )

        eff_cfg = dict(per_effector_cfg.get(name, {}) or {})
        eff_distance_threshold = float(eff_cfg.get("distance_threshold", distance_threshold))
        eff_speed_threshold = float(eff_cfg.get("relative_speed_threshold", speed_threshold))
        eff_exit_distance_threshold = max(
            float(
                eff_cfg.get(
                    "exit_distance_threshold",
                    cfg.get("exit_distance_threshold", eff_distance_threshold),
                )
            ),
            eff_distance_threshold,
        )
        eff_exit_speed_threshold = max(
            float(
                eff_cfg.get(
                    "exit_relative_speed_threshold",
                    cfg.get("exit_relative_speed_threshold", eff_speed_threshold),
                )
            ),
            eff_speed_threshold,
        )
        eff_min_contact_frames = max(
            int(eff_cfg.get("min_contact_frames", min_contact_frames)), 1
        )
        eff_max_gap_frames = max(
            int(eff_cfg.get("max_gap_frames", max_gap_frames)), 0
        )
        eff_transition_frames = max(
            int(eff_cfg.get("transition_frames", transition_frames)), 0
        )
        eff_anchor_contact_orientation = bool(
            eff_cfg.get(
                "anchor_contact_orientation", anchor_contact_orientation
            )
        )
        eff_anchor_window_frames = max(
            int(eff_cfg.get("anchor_window_frames", anchor_window_frames)), 1
        )
        eff_anchor_drift_tolerance = max(
            float(eff_cfg.get("anchor_drift_tolerance", anchor_drift_tolerance)),
            0.0,
        )
        eff_anchor_tangent_to_tracking = bool(
            eff_cfg.get("anchor_tangent_to_tracking", False)
        )
        eff_stabilize_contact_normal_signs = bool(
            eff_cfg.get(
                "stabilize_contact_normal_signs",
                cfg.get("stabilize_contact_normal_signs", False),
            )
        )

        # Selected samples refer to fixed SMPL-X vertex ids, but the nearest id
        # may switch.  The semantic-center velocity is more robust to those
        # switches and still distinguishes stance/support from swing motion.
        velocity = np.zeros_like(tracking_positions)
        if len(tracking_positions) > 1:
            velocity[1:] = np.diff(tracking_positions, axis=0) * fps
            velocity[0] = velocity[1]
        relative_speeds = np.linalg.norm(velocity, axis=1)
        signed_normal_velocity = np.sum(velocity * surface_normals, axis=1)
        normal_speeds = np.abs(signed_normal_velocity)
        tangential_velocity = velocity - (
            signed_normal_velocity[:, None] * surface_normals
        )
        tangential_speeds = np.linalg.norm(tangential_velocity, axis=1)

        # Sliding contact is detected from separation speed, not total speed.
        # A hand or sole moving quickly along a surface must remain contact-
        # active as long as it is not moving away from that surface.
        allow_sliding = bool(eff_cfg.get("allow_sliding", False))
        normal_speed_threshold = float(
            eff_cfg.get("normal_speed_threshold", eff_speed_threshold)
        )
        exit_normal_speed_threshold = max(
            float(
                eff_cfg.get(
                    "exit_normal_speed_threshold",
                    normal_speed_threshold,
                )
            ),
            normal_speed_threshold,
        )
        gate_speeds = normal_speeds if allow_sliding else relative_speeds
        enter_gate_speed = (
            normal_speed_threshold if allow_sliding else eff_speed_threshold
        )
        exit_gate_speed = (
            exit_normal_speed_threshold
            if allow_sliding
            else eff_exit_speed_threshold
        )

        distance_score = _sigmoid(
            (eff_distance_threshold - distances) / distance_softness
        )
        speed_score = _sigmoid(
            (enter_gate_speed - gate_speeds) / speed_softness
        )
        confidence = distance_score * speed_score
        candidate_mask = _hysteresis_contact_mask(
            distances,
            gate_speeds,
            enter_distance=eff_distance_threshold,
            exit_distance=eff_exit_distance_threshold,
            enter_speed=enter_gate_speed,
            exit_speed=exit_gate_speed,
        )
        require_normal_alignment = bool(
            eff_cfg.get("require_normal_alignment", True)
        )
        if up_directions is not None and require_normal_alignment:
            # A sole can only contact a surface whose outward normal agrees
            # with the captured sole-up direction.  Without this test, a
            # nearby wall/step triangle can be selected for a ground foot and
            # make its orientation jump when the false contact starts/ends.
            max_normal_angle = float(
                eff_cfg.get(
                    "normal_alignment_threshold",
                    cfg.get("normal_alignment_threshold", np.deg2rad(30.0)),
                )
            )
            normal_alignment = np.sum(
                up_directions * surface_normals, axis=1
            ) >= np.cos(max_normal_angle)
            candidate_mask &= normal_alignment
            confidence *= np.clip(
                (np.sum(up_directions * surface_normals, axis=1) + 1.0)
                * 0.5,
                0.0,
                1.0,
            )
        candidate_mask = _close_short_gaps(candidate_mask, eff_max_gap_frames)
        candidate_mask = _remove_short_runs(candidate_mask, eff_min_contact_frames)
        confidence = np.where(candidate_mask, np.maximum(confidence, 0.5), confidence)
        # Resolve semantic modes within every detected segment. A single
        # proximity run may alternate between sticking and sliding; classifying
        # the whole run from its mean speed would erase stance-foot anchors.
        configured_mode = str(
            eff_cfg.get("contact_mode", "AUTO")
        ).upper()
        touch_max_frames = max(
            int(eff_cfg.get("touch_max_frames", eff_min_contact_frames)), 0
        )
        support_speed_threshold = max(
            float(eff_cfg.get("support_speed_threshold", 0.25)), 0.0
        )
        support_max_tangential_travel = max(
            float(eff_cfg.get("support_max_tangential_travel", 0.035)), 0.0
        )
        support_min_frames = max(
            int(eff_cfg.get("support_min_frames", eff_min_contact_frames)), 1
        )
        support_max_gap_frames = max(
            int(eff_cfg.get("support_max_gap_frames", 1)), 0
        )
        if configured_mode == "AUTO":
            contact_modes = _classify_auto_contact_modes(
                candidate_mask,
                surface_points,
                surface_normals,
                tangential_speeds,
                touch_max_frames=touch_max_frames,
                support_speed_threshold=support_speed_threshold,
                support_max_tangential_travel=support_max_tangential_travel,
                support_min_frames=support_min_frames,
                support_max_gap_frames=support_max_gap_frames,
            )
        else:
            resolved_mode = _parse_contact_mode(configured_mode)
            contact_modes = np.where(
                candidate_mask,
                int(resolved_mode),
                int(ContactMode.NONE),
            ).astype(np.uint8)
        contact_segment_ids = np.full(len(candidate_mask), -1, dtype=np.int32)
        contact_positions = selected_samples.copy()
        transition_weights = np.ones(len(candidate_mask), dtype=float)
        next_segment_id = 0

        def _window_anchor(anchor_slice: slice) -> tuple[np.ndarray, np.ndarray]:
            anchor = np.median(surface_points[anchor_slice], axis=0)
            run_normals = surface_normals[anchor_slice].copy()
            # Mesh winding around reconstructed TSDF seams can alternate.
            run_offsets = selected_samples[anchor_slice] - surface_points[anchor_slice]
            reference_idx = int(np.argmax(np.linalg.norm(run_offsets, axis=1)))
            reference_normal = run_normals[reference_idx]
            reference_norm = float(np.linalg.norm(reference_normal))
            if reference_norm < 1e-10:
                reference_normal = np.array([0.0, 0.0, 1.0])
            else:
                reference_normal /= reference_norm
            opposite = (run_normals @ reference_normal) < 0.0
            run_normals[opposite] *= -1.0
            normal = np.median(run_normals, axis=0)
            normal_norm = float(np.linalg.norm(normal))
            normal = normal / normal_norm if normal_norm > 1e-10 else reference_normal
            toward_effector = np.median(
                tracking_positions[anchor_slice] - surface_points[anchor_slice],
                axis=0,
            )
            if float(np.dot(normal, toward_effector)) < 0.0:
                normal *= -1.0
            if eff_anchor_tangent_to_tracking:
                # The closest source sole sample can migrate heel-to-toe as
                # the foot rolls, even on one perfectly flat tread.  Terrain
                # contact only determines the normal coordinate; preserve the
                # semantic foot marker's tangential coordinate so the anchor
                # cannot walk along the sole with that sample-id switch.
                tracking_anchor = np.median(
                    tracking_positions[anchor_slice], axis=0
                )
                anchor = tracking_anchor + normal * float(
                    np.dot(anchor - tracking_anchor, normal)
                )
            return anchor, normal

        semantic_runs = []
        for start in range(len(contact_modes)):
            mode = ContactMode(int(contact_modes[start]))
            if mode == ContactMode.NONE:
                continue
            if start > 0 and contact_modes[start - 1] == contact_modes[start]:
                continue
            end = start + 1
            while (
                end < len(contact_modes)
                and contact_modes[end] == contact_modes[start]
            ):
                end += 1
            semantic_runs.append((start, end, mode))

        for start, end, run_mode in semantic_runs:
            if run_mode == ContactMode.SUPPORT:
                # A SUPPORT segment is a true no-slip world anchor.  If a
                # source reconstruction drifts beyond the configured tolerance,
                # start a new segment explicitly rather than silently moving an
                # existing anchor.
                segment_start = start
                while segment_start < end:
                    anchor_slice = slice(
                        segment_start,
                        min(segment_start + eff_anchor_window_frames, end),
                    )
                    anchor, normal = _window_anchor(anchor_slice)
                    segment_end = end
                    if eff_anchor_drift_tolerance > 0.0:
                        consecutive_drift = 0
                        for frame_idx in range(segment_start, end):
                            if (
                                float(np.linalg.norm(surface_points[frame_idx] - anchor))
                                > eff_anchor_drift_tolerance
                            ):
                                consecutive_drift += 1
                                if consecutive_drift >= 2:
                                    segment_end = max(frame_idx - 1, segment_start + 1)
                                    break
                            else:
                                consecutive_drift = 0
                    contact_modes[segment_start:segment_end] = int(run_mode)
                    contact_segment_ids[segment_start:segment_end] = next_segment_id
                    contact_positions[segment_start:segment_end] = anchor + clearance * normal
                    surface_normals[segment_start:segment_end] = normal
                    if forward_directions is not None and eff_anchor_contact_orientation:
                        orientation_slice = slice(
                            segment_start,
                            min(segment_start + eff_anchor_window_frames, segment_end),
                        )
                        forward = np.median(forward_directions[orientation_slice], axis=0)
                        forward_norm = float(np.linalg.norm(forward))
                        if forward_norm > 1e-10:
                            forward_directions[segment_start:segment_end] = (
                                forward / forward_norm
                            )
                        if up_directions is not None:
                            up = np.median(up_directions[orientation_slice], axis=0)
                            up_norm = float(np.linalg.norm(up))
                            if up_norm > 1e-10:
                                up_directions[segment_start:segment_end] = up / up_norm
                    next_segment_id += 1
                    segment_start = segment_end
            else:
                # TOUCH and SLIDE retain the measured scene point each frame.
                # SLIDE therefore constrains the normal coordinate without
                # freezing the tangential trajectory.
                if eff_stabilize_contact_normal_signs:
                    orientation_slice = slice(
                        start, min(start + eff_anchor_window_frames, end)
                    )
                    _, reference_normal = _window_anchor(orientation_slice)
                    run_normals = surface_normals[start:end].copy()
                    opposite = (run_normals @ reference_normal) < 0.0
                    run_normals[opposite] *= -1.0
                    surface_normals[start:end] = run_normals
                contact_modes[start:end] = int(run_mode)
                contact_segment_ids[start:end] = next_segment_id
                if eff_anchor_tangent_to_tracking:
                    # Preserve the source foot/palm coordinate *within* the
                    # contact manifold while taking only the normal coordinate
                    # from the scene.  This is the same single-sole-point
                    # contract used by the earlier OmniRetarget-style solver:
                    # the nearest heel/toe detection sample must not translate
                    # the whole robot foot by one sample spacing.
                    tracking_to_surface = (
                        surface_points[start:end]
                        - tracking_positions[start:end]
                    )
                    normal_distance = np.sum(
                        tracking_to_surface * surface_normals[start:end],
                        axis=1,
                        keepdims=True,
                    )
                    manifold_points = (
                        tracking_positions[start:end]
                        + normal_distance * surface_normals[start:end]
                    )
                else:
                    manifold_points = surface_points[start:end]
                contact_positions[start:end] = (
                    manifold_points
                    + clearance * surface_normals[start:end]
                )
                if eff_transition_frames > 0:
                    for k in range(min(eff_transition_frames, end - start)):
                        fade = (k + 1.0) / (eff_transition_frames + 1.0)
                        transition_weights[start + k] = min(
                            transition_weights[start + k], fade
                        )
                        transition_weights[end - 1 - k] = min(
                            transition_weights[end - 1 - k], fade
                        )
                next_segment_id += 1

        # Store the same valid SO(3) foot frame that the optimizer receives.
        # The contact-normal consistency test above guarantees that this
        # captured frame is physically compatible with the scene; retaining
        # it here avoids a reference discontinuity at contact transitions.
        if forward_directions is not None and up_directions is not None:
            target_up = up_directions.copy()
            target_up /= np.maximum(
                np.linalg.norm(target_up, axis=1, keepdims=True), 1e-10
            )
            tangent_forward = forward_directions - target_up * np.sum(
                forward_directions * target_up, axis=1, keepdims=True
            )
            tangent_norm = np.linalg.norm(
                tangent_forward, axis=1, keepdims=True
            )
            valid_forward = tangent_norm[:, 0] > 1e-10
            tangent_forward[valid_forward] /= tangent_norm[valid_forward]
            forward_directions[valid_forward] = tangent_forward[valid_forward]
            up_directions = target_up

        tasks[name] = SceneContactTask(
            name=name,
            tracking_positions=tracking_positions,
            contact_positions=contact_positions,
            surface_points=surface_points,
            surface_normals=surface_normals,
            distances=distances,
            relative_speeds=relative_speeds,
            contact_confidence=confidence,
            contact_modes=contact_modes,
            contact_segment_ids=contact_segment_ids,
            transition_weights=transition_weights,
            normal_speeds=normal_speeds,
            tangential_speeds=tangential_speeds,
            forward_directions=forward_directions,
            up_directions=up_directions,
        )

    return tasks


def stack_contact_tasks(
    tasks: Mapping[str, SceneContactTask],
) -> Dict[str, np.ndarray]:
    """Convert contact tasks to arrays suitable for ``numpy.savez``."""
    names = list(tasks)
    if not names:
        return {}
    if any(tasks[name].effective_swing_positions is None for name in names):
        raise RuntimeError(
            "Contact tasks must be morphology-adapted before NPZ export."
        )
    if any(tasks[name].robot_base_reference_w is None for name in names):
        raise RuntimeError(
            "Contact tasks must include robot_base_reference_w before NPZ export."
        )
    if any(tasks[name].morphology_scale is None for name in names):
        raise RuntimeError("Contact tasks must include morphology_scale before export.")
    base_reference = np.asarray(tasks[names[0]].robot_base_reference_w, dtype=float)
    if any(
        not np.allclose(
            np.asarray(tasks[name].robot_base_reference_w, dtype=float),
            base_reference,
        )
        for name in names[1:]
    ):
        raise ValueError("All contact tasks must share robot_base_reference_w.")
    return {
        "contact_names": np.asarray(names),
        "contact_mode": np.stack(
            [tasks[name].contact_modes for name in names], axis=1
        ).astype(np.uint8),
        "contact_mode_names": np.asarray(
            [mode.name for mode in ContactMode]
        ),
        "contact_segment_id": np.stack(
            [tasks[name].contact_segment_ids for name in names], axis=1
        ).astype(np.int32),
        "contact_confidence": np.stack(
            [tasks[name].contact_confidence for name in names], axis=1
        ),
        "contact_normal_speed": np.stack(
            [
                tasks[name].normal_speeds
                if tasks[name].normal_speeds is not None
                else tasks[name].relative_speeds
                for name in names
            ],
            axis=1,
        ),
        "contact_tangential_speed": np.stack(
            [
                tasks[name].tangential_speeds
                if tasks[name].tangential_speeds is not None
                else np.zeros_like(tasks[name].relative_speeds)
                for name in names
            ],
            axis=1,
        ),
        "contact_transition_weight": np.stack(
            [
                tasks[name].transition_weights
                for name in names
            ],
            axis=1,
        ),
        "contact_targets_w": np.stack(
            [tasks[name].contact_positions for name in names], axis=1
        ),
        "end_effector_reference_w": np.stack(
            [tasks[name].tracking_positions for name in names], axis=1
        ),
        "effective_swing_targets_w": np.stack(
            [tasks[name].effective_swing_positions for name in names],
            axis=1,
        ),
        "swing_limb_offsets_w": np.stack(
            [
                tasks[name].swing_limb_offsets
                if tasks[name].swing_limb_offsets is not None
                else np.full_like(tasks[name].tracking_positions, np.nan)
                for name in names
            ],
            axis=1,
        ),
        "swing_anchor_body": np.asarray(
            [tasks[name].swing_anchor_body or "" for name in names]
        ),
        "swing_anchor_offset": np.stack(
            [
                tasks[name].swing_anchor_offset
                if tasks[name].swing_anchor_offset is not None
                else np.full(3, np.nan, dtype=float)
                for name in names
            ],
            axis=0,
        ),
        "contact_bridge_alpha": np.stack(
            [
                tasks[name].contact_bridge_alpha
                if tasks[name].contact_bridge_alpha is not None
                else (
                    tasks[name].contact_modes != int(ContactMode.NONE)
                ).astype(float)
                for name in names
            ],
            axis=1,
        ),
        "contact_bridge_targets_w": np.stack(
            [
                tasks[name].contact_bridge_positions
                if tasks[name].contact_bridge_positions is not None
                else tasks[name].contact_positions
                for name in names
            ],
            axis=1,
        ),
        "contact_orientation_bridge_alpha": np.stack(
            [
                tasks[name].orientation_bridge_alpha
                if tasks[name].orientation_bridge_alpha is not None
                else (
                    tasks[name].contact_modes != int(ContactMode.NONE)
                ).astype(float)
                for name in names
            ],
            axis=1,
        ),
        "end_effector_bridge_forward_w": np.stack(
            [
                tasks[name].orientation_target_rotations[:, :, 0]
                if tasks[name].orientation_target_rotations is not None
                else (
                    tasks[name].forward_directions
                    if tasks[name].forward_directions is not None
                    else np.full_like(tasks[name].tracking_positions, np.nan)
                )
                for name in names
            ],
            axis=1,
        ),
        "end_effector_bridge_up_w": np.stack(
            [
                tasks[name].orientation_target_rotations[:, :, 2]
                if tasks[name].orientation_target_rotations is not None
                else (
                    tasks[name].up_directions
                    if tasks[name].up_directions is not None
                    else np.full_like(tasks[name].tracking_positions, np.nan)
                )
                for name in names
            ],
            axis=1,
        ),
        "swing_morphology_scale": np.asarray(
            [tasks[name].morphology_scale for name in names], dtype=float
        ),
        "end_effector_max_reach": np.asarray(
            [
                np.nan
                if tasks[name].robot_max_reach is None
                else tasks[name].robot_max_reach
                for name in names
            ],
            dtype=float,
        ),
        "end_effector_source_limb_chain": np.asarray(
            [
                ""
                if tasks[name].source_limb_chain is None
                else "/".join(tasks[name].source_limb_chain)
                for name in names
            ]
        ),
        "active_contact_offset_id": np.stack(
            [
                tasks[name].active_contact_offset_ids
                if tasks[name].active_contact_offset_ids is not None
                else np.full(len(tasks[name].contact_modes), -1, dtype=np.int32)
                for name in names
            ],
            axis=1,
        ).astype(np.int32),
        "contact_surface_normals_w": np.stack(
            [tasks[name].surface_normals for name in names], axis=1
        ),
        "end_effector_forward_w": np.stack(
            [
                tasks[name].forward_directions
                if tasks[name].forward_directions is not None
                else np.full_like(tasks[name].tracking_positions, np.nan)
                for name in names
            ],
            axis=1,
        ),
        "end_effector_up_w": np.stack(
            [
                tasks[name].up_directions
                if tasks[name].up_directions is not None
                else np.full_like(tasks[name].tracking_positions, np.nan)
                for name in names
            ],
            axis=1,
        ),
        "robot_base_reference_w": base_reference,
    }
