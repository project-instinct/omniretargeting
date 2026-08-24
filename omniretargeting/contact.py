"""Scene-contact inference for contact-prioritized motion retargeting.

The source motion provides body-surface samples for the four end effectors.
This module turns those samples into temporally filtered contact schedules and
world-frame contact anchors on a static scene mesh.  The resulting schedule is
consumed by the kinematic optimizer and can also be exported for downstream
motion-tracking policies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import numpy as np
import trimesh


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
    contact_mask: np.ndarray
    forward_directions: np.ndarray | None = None
    up_directions: np.ndarray | None = None
    transition_weights: np.ndarray | None = None
    contact_modes: np.ndarray | None = None

    def frame(self, frame_idx: int) -> Dict[str, Any]:
        """Return the optimizer payload for one frame."""
        transition_weight = (
            float(self.transition_weights[frame_idx])
            if self.transition_weights is not None
            else 1.0
        )
        contact_mode = (
            str(self.contact_modes[frame_idx])
            if self.contact_modes is not None
            else ("sticking" if bool(self.contact_mask[frame_idx]) else "none")
        )
        contact_position = self.contact_positions[frame_idx]
        if contact_mode == "sticking" and transition_weight < 1.0:
            contact_position = (
                (1.0 - transition_weight) * self.tracking_positions[frame_idx]
                + transition_weight * contact_position
            )
        payload = {
            "name": self.name,
            "tracking_position": self.tracking_positions[frame_idx],
            "contact_position": contact_position,
            "surface_normal": self.surface_normals[frame_idx],
            "distance": float(self.distances[frame_idx]),
            "relative_speed": float(self.relative_speeds[frame_idx]),
            "confidence": float(self.contact_confidence[frame_idx]),
            "contact": bool(self.contact_mask[frame_idx]),
            "contact_mode": contact_mode,
            "transition_weight": transition_weight,
        }
        if self.forward_directions is not None:
            payload["forward_direction"] = self.forward_directions[frame_idx]
        if self.up_directions is not None:
            payload["up_direction"] = self.up_directions[frame_idx]
        return payload


def build_world_tracking_tasks(
    marker_trajectories: Mapping[str, Mapping[str, np.ndarray]],
) -> Dict[str, SceneContactTask]:
    """Build per-frame world targets without inferring contact semantics.

    This is the lightweight path used when hands/soles should follow the
    source motion but SUPPORT/SLIDE classification and scene anchoring are
    deliberately disabled.  Every frame is therefore a swing-style tracking
    frame; root height remains free for the robot kinematics to resolve.
    """
    tasks: Dict[str, SceneContactTask] = {}
    for name, marker_data in marker_trajectories.items():
        tracking_positions = np.asarray(
            marker_data["tracking_positions"], dtype=float
        )
        if tracking_positions.ndim != 2 or tracking_positions.shape[1] != 3:
            raise ValueError(
                f"{name}: tracking_positions must have shape (T, 3), "
                f"got {tracking_positions.shape}."
            )
        num_frames = len(tracking_positions)
        speeds = np.zeros(num_frames, dtype=float)
        if num_frames > 1:
            speeds[1:] = np.linalg.norm(
                np.diff(tracking_positions, axis=0), axis=1
            )
        forward_directions = marker_data.get("forward_directions")
        up_directions = marker_data.get("up_directions")
        tasks[name] = SceneContactTask(
            name=name,
            tracking_positions=tracking_positions,
            contact_positions=tracking_positions.copy(),
            surface_points=tracking_positions.copy(),
            surface_normals=np.tile([0.0, 0.0, 1.0], (num_frames, 1)),
            distances=np.full(num_frames, np.inf, dtype=float),
            relative_speeds=speeds,
            contact_confidence=np.ones(num_frames, dtype=float),
            contact_mask=np.zeros(num_frames, dtype=bool),
            forward_directions=(
                None
                if forward_directions is None
                else np.asarray(forward_directions, dtype=float)
            ),
            up_directions=(
                None
                if up_directions is None
                else np.asarray(up_directions, dtype=float)
            ),
            transition_weights=np.ones(num_frames, dtype=float),
            contact_modes=np.full(num_frames, "none", dtype="<U8"),
        )
    return tasks


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
    anchor_contacts = bool(cfg.get("anchor_contacts", True))
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
        eff_min_contact_frames = max(
            int(eff_cfg.get("min_contact_frames", min_contact_frames)), 1
        )
        eff_max_gap_frames = max(
            int(eff_cfg.get("max_gap_frames", max_gap_frames)), 0
        )
        eff_transition_frames = max(
            int(eff_cfg.get("transition_frames", transition_frames)), 0
        )
        eff_anchor_contacts = bool(
            eff_cfg.get("anchor_contacts", anchor_contacts)
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
        eff_sticking_tangential_speed_threshold = max(
            float(
                eff_cfg.get(
                    "sticking_tangential_speed_threshold",
                    cfg.get("sticking_tangential_speed_threshold", 0.12),
                )
            ),
            0.0,
        )
        eff_sticking_normal_speed_threshold = max(
            float(
                eff_cfg.get(
                    "sticking_normal_speed_threshold",
                    cfg.get("sticking_normal_speed_threshold", 0.12),
                )
            ),
            0.0,
        )
        eff_min_sticking_frames = max(
            int(
                eff_cfg.get(
                    "min_sticking_frames",
                    cfg.get("min_sticking_frames", min_contact_frames),
                )
            ),
            1,
        )

        # Selected samples refer to fixed SMPL-X vertex ids, but the nearest id
        # may switch.  The semantic-center velocity is more robust to those
        # switches and still distinguishes stance/support from swing motion.
        velocity = np.zeros_like(tracking_positions)
        if len(tracking_positions) > 1:
            velocity[1:] = np.diff(tracking_positions, axis=0) * fps
            velocity[0] = velocity[1]
        relative_speeds = np.linalg.norm(velocity, axis=1)

        distance_score = _sigmoid(
            (eff_distance_threshold - distances) / distance_softness
        )
        speed_score = _sigmoid(
            (eff_speed_threshold - relative_speeds) / speed_softness
        )
        confidence = distance_score * speed_score
        contact_mask = (distances <= eff_distance_threshold) & (
            relative_speeds <= eff_speed_threshold
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
            contact_mask &= normal_alignment
            confidence *= np.clip(
                (np.sum(up_directions * surface_normals, axis=1) + 1.0)
                * 0.5,
                0.0,
                1.0,
            )
        contact_mask = _close_short_gaps(contact_mask, eff_max_gap_frames)
        contact_mask = _remove_short_runs(contact_mask, eff_min_contact_frames)
        confidence = np.where(contact_mask, np.maximum(confidence, 0.5), confidence)

        normal_speeds = np.abs(np.sum(velocity * surface_normals, axis=1))
        tangential_velocity = velocity - surface_normals * np.sum(
            velocity * surface_normals, axis=1, keepdims=True
        )
        tangential_speeds = np.linalg.norm(tangential_velocity, axis=1)
        sticking_mask = (
            contact_mask
            & eff_anchor_contacts
            & (tangential_speeds <= eff_sticking_tangential_speed_threshold)
            & (normal_speeds <= eff_sticking_normal_speed_threshold)
        )
        sticking_mask = _remove_short_runs(
            sticking_mask, eff_min_sticking_frames
        )
        contact_modes = np.full(len(contact_mask), "none", dtype="<U8")
        contact_modes[contact_mask] = "sliding"
        contact_modes[sticking_mask] = "sticking"
        # Fade the contact weight in and out over a few frames, like a
        # foot-lock blend.  A binary weight switch makes the optimizer jump
        # between the swing target and the scene anchor within one frame; the
        # ramp spreads that transient without changing the detected contact
        # interval or the anchor itself.  It is stored separately so the
        # optimizers can apply it after their own confidence floor, which
        # would otherwise clip the fade away.
        transition_weights = np.ones(len(contact_mask), dtype=float)
        if eff_transition_frames > 0:
            for start, end in _iter_true_runs(contact_mask):
                for k in range(min(eff_transition_frames, end - start)):
                    fade = (k + 1.0) / (eff_transition_frames + 1.0)
                    transition_weights[start + k] = min(
                        transition_weights[start + k], fade
                    )
                    transition_weights[end - 1 - k] = min(
                        transition_weights[end - 1 - k], fade
                    )

        contact_positions = selected_samples.copy()
        contact_positions[contact_mask] = (
            surface_points[contact_mask]
            + clearance * surface_normals[contact_mask]
        )
        if eff_anchor_contacts:

            def _window_anchor(anchor_slice: slice) -> tuple[np.ndarray, np.ndarray]:
                anchor = np.median(surface_points[anchor_slice], axis=0)
                run_normals = surface_normals[anchor_slice].copy()
                # Mesh winding around reconstructed TSDF seams can alternate.
                # Put all normals in one hemisphere before taking a robust
                # median; otherwise opposite unit normals cancel to zero.
                run_offsets = (
                    selected_samples[anchor_slice] - surface_points[anchor_slice]
                )
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
                normal = (
                    normal / normal_norm
                    if normal_norm > 1e-10
                    else reference_normal
                )
                # Pick the outward hemisphere using the semantic hand/foot
                # center, which is more reliable than a sampled surface
                # vertex that may lie exactly on a two-sided TSDF triangle.
                toward_effector = np.median(
                    tracking_positions[anchor_slice] - surface_points[anchor_slice],
                    axis=0,
                )
                if float(np.dot(normal, toward_effector)) < 0.0:
                    normal *= -1.0
                return anchor, normal

            for start, end in _iter_true_runs(sticking_mask):
                # Estimate the no-slip anchor at contact onset.  Using the
                # median of an entire run is non-causal and can place the
                # target far ahead of the foot on the first stance frame.
                # A detected "stance" from video-derived data can also contain
                # sustained slow drift; holding one anchor across such a run
                # drags the target far from the source foot, so the run is
                # re-anchored whenever the source contact point durably leaves
                # the current anchor by more than the drift tolerance.
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
                                float(
                                    np.linalg.norm(
                                        surface_points[frame_idx] - anchor
                                    )
                                )
                                > eff_anchor_drift_tolerance
                            ):
                                consecutive_drift += 1
                                if consecutive_drift >= 2:
                                    segment_end = max(
                                        frame_idx - 1, segment_start + 1
                                    )
                                    break
                            else:
                                consecutive_drift = 0
                    contact_positions[segment_start:segment_end] = (
                        anchor + clearance * normal
                    )
                    surface_normals[segment_start:segment_end] = normal
                    segment_start = segment_end
                if (
                    forward_directions is not None
                    and eff_anchor_contact_orientation
                ):
                    # A stance foot must have one orientation throughout the
                    # no-slip contact run.  Taking the median rejects mocap
                    # noise without introducing a future-dependent filter in
                    # the optimizer itself.
                    onset_slice = slice(
                        start, min(start + eff_anchor_window_frames, end)
                    )
                    forward = np.median(forward_directions[onset_slice], axis=0)
                    forward_norm = float(np.linalg.norm(forward))
                    if forward_norm > 1e-10:
                        forward_directions[start:end] = forward / forward_norm
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
            contact_mask=contact_mask,
            forward_directions=forward_directions,
            up_directions=up_directions,
            transition_weights=transition_weights,
            contact_modes=contact_modes,
        )

    return tasks


def stack_contact_tasks(
    tasks: Mapping[str, SceneContactTask],
) -> Dict[str, np.ndarray]:
    """Convert contact tasks to arrays suitable for ``numpy.savez``."""
    names = list(tasks)
    if not names:
        return {}
    return {
        "contact_names": np.asarray(names),
        "contact_mask": np.stack([tasks[name].contact_mask for name in names], axis=1),
        "contact_modes": np.stack(
            [
                tasks[name].contact_modes
                if tasks[name].contact_modes is not None
                else np.where(tasks[name].contact_mask, "sticking", "none")
                for name in names
            ],
            axis=1,
        ),
        "contact_confidence": np.stack(
            [tasks[name].contact_confidence for name in names], axis=1
        ),
        "contact_targets_w": np.stack(
            [tasks[name].contact_positions for name in names], axis=1
        ),
        "end_effector_reference_w": np.stack(
            [tasks[name].tracking_positions for name in names], axis=1
        ),
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
    }
