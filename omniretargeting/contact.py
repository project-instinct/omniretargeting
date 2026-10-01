"""Scene-contact inference for contact-prioritized motion retargeting.

The source motion provides body-surface samples for the four end effectors.
This module turns those samples into temporally filtered contact schedules and
world-frame contact anchors on a static scene mesh.  The resulting schedule is
consumed by the kinematic optimizer and can also be exported for downstream
motion-tracking policies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Dict, Mapping

import numpy as np
import trimesh

from .contact_correspondence import coherent_sample_indices
from .contact_scene_normals import (
    _DEFAULT_NORMAL_ORIENTATION_DEADZONE as _DEFAULT_NORMAL_ORIENTATION_DEADZONE,
    _stabilize_contact_normals as _stabilize_contact_normals,
)
from .contact_scene_samples import (
    _closest_scene_samples as _closest_scene_samples,
    _contact_patch_surface_points as _contact_patch_surface_points,
    _query_scene_samples as _query_scene_samples,
    _select_closest_scene_samples as _select_closest_scene_samples,
)
from .contact_schedule import (
    _apply_contact_hysteresis as _apply_contact_hysteresis,
    _close_short_gaps as _close_short_gaps,
    _iter_true_runs as _iter_true_runs,
    _remove_short_runs as _remove_short_runs,
    _sigmoid as _sigmoid,
    approach_blend,
)

_SURFACE_ONLY_CONTACT_MODES = frozenset({"sliding", "transition"})


def _is_surface_only_contact_mode(contact_mode: str) -> bool:
    """Whether a contact constrains the scene normal but not all three axes."""
    return str(contact_mode) in _SURFACE_ONLY_CONTACT_MODES


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
    environment_ids: np.ndarray | None = None  # 0: static scene; 1..N: moving objects
    preserve_contact_anchors: bool = False  # anchors are always held; kept for callers
    contact_phase: Any = field(default=None, repr=False, compare=False)
    approach_reference: tuple[np.ndarray, np.ndarray] | None = field(
        default=None, repr=False, compare=False
    )
    surface_managed: bool = False
    swing_world_mask: np.ndarray | None = None
    surface_managed_mask: np.ndarray | None = None
    preserve_world_tracking: bool = False

    @cached_property
    def managed_frames(self):
        """Only suppress an endpoint where its surface replacement exists."""
        mask = np.full(len(self.contact_mask), self.surface_managed, dtype=bool)
        if self.surface_managed_mask is not None:
            coverage = np.asarray(self.surface_managed_mask, dtype=bool)
            if coverage.shape != mask.shape:
                raise ValueError("Surface coverage must share the contact timeline")
            mask &= coverage
        mask.setflags(write=False)
        return mask

    @cached_property
    def approach(self):
        """Source-progress hand-over between free targets and contact anchors."""
        if self.approach_reference is not None:
            weights, positions = self.approach_reference
            if (np.shape(weights) != (len(self.contact_mask),)
                    or np.shape(positions) != np.shape(self.tracking_positions)
                    or not np.isfinite(weights).all() or not np.isfinite(positions).all()
                    or np.any(np.asarray(weights) < 0) or np.any(np.asarray(weights) > 1)):
                raise ValueError("Approach references must share the contact timeline")
            return weights, positions
        return approach_blend(
            self.tracking_positions, self.contact_positions, self.contact_mask
        )

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
        # The anchor holds from the first contact frame: the free target was
        # already handed over outside the interval (``approach``), so the
        # contact target is not pulled back toward the swing path inside it.
        contact_position = self.contact_positions[frame_idx]
        approach_weights, approach_positions = self.approach
        payload = {
            "name": self.name,
            "tracking_position": self.tracking_positions[frame_idx],
            "contact_position": contact_position,
            "surface_normal": self.surface_normals[frame_idx],
            "distance": float(self.distances[frame_idx]),
            "relative_speed": float(self.relative_speeds[frame_idx]),
            "confidence": float(self.contact_confidence[frame_idx]),
            "contact": bool(self.contact_mask[frame_idx]),
            "surface_managed": bool(self.managed_frames[frame_idx]),
            "contact_mode": contact_mode,
            "transition_weight": transition_weight,
            "approach_weight": float(approach_weights[frame_idx]),
            "approach_position": approach_positions[frame_idx],
            "environment_id": int(self.environment_ids[frame_idx]) if self.environment_ids is not None else 0,
        }
        if self.swing_world_mask is not None:
            payload["target_world"] = bool(self.swing_world_mask[frame_idx])
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
    normal_orientation_deadzone = max(
        float(
            cfg.get(
                "normal_orientation_deadzone",
                _DEFAULT_NORMAL_ORIENTATION_DEADZONE,
            )
        ),
        0.0,
    )
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
        source_contact_mask = marker_data.get("source_contact_mask")
        if source_contact_mask is not None:
            source_contact_mask = np.asarray(
                source_contact_mask, dtype=bool
            )
            if source_contact_mask.shape != (len(tracking_positions),):
                raise ValueError(
                    f"{name}: source_contact_mask must have shape "
                    f"({len(tracking_positions)},), got "
                    f"{source_contact_mask.shape}."
                )
        (
            all_samples,
            all_surface_points,
            all_distances,
            all_triangle_ids,
        ) = _query_scene_samples(scene_mesh, contact_samples)
        eff_cfg = dict(per_effector_cfg.get(name, {}) or {})
        sample_ids = coherent_sample_indices(
            all_distances, all_surface_points, all_samples,
            max_distance=eff_cfg.get('distance_exit_threshold',
                                    cfg.get('distance_exit_threshold', distance_threshold)),
            switch_weight=eff_cfg.get('contact_sample_switch_weight',
                                      cfg.get('contact_sample_switch_weight', 1.)))
        selected_samples, surface_points, surface_normals, distances = (
            _select_closest_scene_samples(
                scene_mesh,
                all_samples,
                all_surface_points,
                all_distances,
                all_triangle_ids,
                normal_orientation_deadzone=normal_orientation_deadzone,
                enclosed_free_space=bool(cfg.get('scene_enclosed_free_space', False)),
                sample_ids=sample_ids,
            )
        )

        eff_cfg = dict(per_effector_cfg.get(name, {}) or {})
        eff_distance_threshold = float(eff_cfg.get("distance_threshold", distance_threshold))
        eff_speed_threshold = float(eff_cfg.get("relative_speed_threshold", speed_threshold))
        eff_distance_exit_threshold = max(
            float(
                eff_cfg.get(
                    "distance_exit_threshold",
                    cfg.get(
                        "distance_exit_threshold",
                        max(
                            eff_distance_threshold + distance_softness,
                            1.25 * eff_distance_threshold,
                        ),
                    ),
                )
            ),
            eff_distance_threshold,
        )
        eff_speed_exit_threshold = max(
            float(
                eff_cfg.get(
                    "relative_speed_exit_threshold",
                    cfg.get(
                        "relative_speed_exit_threshold",
                        max(
                            eff_speed_threshold + speed_softness,
                            1.25 * eff_speed_threshold,
                        ),
                    ),
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
        eff_sliding_normal_speed_threshold = max(
            float(
                eff_cfg.get(
                    "sliding_normal_speed_threshold",
                    cfg.get(
                        "sliding_normal_speed_threshold",
                        max(eff_sticking_normal_speed_threshold, 0.25),
                    ),
                )
            ),
            eff_sticking_normal_speed_threshold,
        )
        eff_min_sliding_frames = max(
            int(
                eff_cfg.get(
                    "min_sliding_frames",
                    cfg.get("min_sliding_frames", 2),
                )
            ),
            1,
        )
        eff_stabilize_contact_normals = bool(
            eff_cfg.get(
                "stabilize_contact_normals",
                cfg.get("stabilize_contact_normals", True),
            )
        )
        eff_contact_normal_smoothing_frames = max(
            int(
                eff_cfg.get(
                    "contact_normal_smoothing_frames",
                    cfg.get("contact_normal_smoothing_frames", 3),
                )
            ),
            1,
        )
        eff_contact_patch_distance = max(
            float(
                eff_cfg.get(
                    "contact_patch_distance",
                    cfg.get(
                        "contact_patch_distance",
                        min(eff_distance_threshold, 0.015),
                    ),
                )
            ),
            0.0,
        )
        eff_contact_patch_spatial_radius = max(
            float(
                eff_cfg.get(
                    "contact_patch_spatial_radius",
                    cfg.get("contact_patch_spatial_radius", 0.12),
                )
            ),
            0.0,
        )
        eff_contact_patch_normal_angle = max(
            float(
                eff_cfg.get(
                    "contact_patch_normal_angle",
                    cfg.get("contact_patch_normal_angle", np.deg2rad(45.0)),
                )
            ),
            0.0,
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
        enter_mask = (distances <= eff_distance_threshold) & (
            relative_speeds <= eff_speed_threshold
        )
        stay_mask = (distances <= eff_distance_exit_threshold) & (
            relative_speeds <= eff_speed_exit_threshold
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
            enter_mask &= normal_alignment
            stay_mask &= normal_alignment
            confidence *= np.clip(
                (np.sum(up_directions * surface_normals, axis=1) + 1.0)
                * 0.5,
                0.0,
                1.0,
            )
        if source_contact_mask is None:
            contact_mask = _apply_contact_hysteresis(enter_mask, stay_mask)
            contact_mask = _close_short_gaps(contact_mask, eff_max_gap_frames)
            contact_mask = _remove_short_runs(
                contact_mask, eff_min_contact_frames
            )
        else:
            # Some source formats carry authored/simulated contact timing.
            # Preserve that timing exactly and use the scene query only to
            # recover a robot-side world target and reliable exit normal.
            # Re-running distance/speed thresholding here would discard the
            # useful label precisely when video reconstruction is offset.
            contact_mask = source_contact_mask.copy()
        confidence = np.where(contact_mask, np.maximum(confidence, 0.5), confidence)
        if source_contact_mask is not None:
            confidence = np.where(contact_mask, 1.0, confidence)

        if eff_stabilize_contact_normals and not cfg.get('scene_enclosed_free_space', False):
            surface_normals = _stabilize_contact_normals(
                surface_normals,
                tracking_positions,
                surface_points,
                distances,
                contact_mask,
                onset_window_frames=eff_anchor_window_frames,
                smoothing_frames=eff_contact_normal_smoothing_frames,
                normal_orientation_deadzone=normal_orientation_deadzone,
            )
        if up_directions is not None and name.endswith("_foot"):
            # A reconstructed sole can lie just below a thin floor surface.
            # Its approach offset then points at the floor underside even
            # though the sole frame still identifies the contact side.
            opposite = contact_mask & (
                np.sum(surface_normals * up_directions, axis=1) < 0.0
            )
            surface_normals[opposite] *= -1.0
        if eff_contact_patch_distance > 0.0:
            surface_points = _contact_patch_surface_points(
                scene_mesh,
                all_surface_points,
                all_distances,
                all_triangle_ids,
                surface_points,
                surface_normals,
                contact_mask,
                max_patch_distance=eff_contact_patch_distance,
                spatial_radius=eff_contact_patch_spatial_radius,
                normal_angle=eff_contact_patch_normal_angle,
            )

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
        sliding_mask = (
            contact_mask
            & (normal_speeds <= eff_sliding_normal_speed_threshold)
            & (tangential_speeds > eff_sticking_tangential_speed_threshold)
        )
        sliding_mask = _remove_short_runs(
            sliding_mask, eff_min_sliding_frames
        )
        contact_modes = np.full(len(contact_mask), "none", dtype="<U10")
        contact_modes[contact_mask] = "transition"
        contact_modes[sliding_mask] = "sliding"
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
                run_offset_norms = np.linalg.norm(run_offsets, axis=1)
                if float(np.max(run_offset_norms)) > normal_orientation_deadzone:
                    reference_idx = int(np.argmax(run_offset_norms))
                    reference_normal = run_normals[reference_idx]
                else:
                    # With all samples effectively on-surface, offset signs are
                    # numerical noise.  Trust the robust direction of the mesh
                    # normals instead of letting the largest tiny offset choose
                    # the hemisphere for the whole contact run.
                    reference_normal = np.median(run_normals, axis=0)
                reference_norm = float(np.linalg.norm(reference_normal))
                if reference_norm < 1e-10:
                    reference_normal = run_normals[0]
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
                # The contact-run stabilization has already chosen the
                # approach side.  Do not reorient this normal toward the
                # current source center: video reconstruction can put that
                # center inside the scene during an otherwise valid contact.
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
        "contact_environment_ids": np.stack([
            tasks[name].environment_ids if tasks[name].environment_ids is not None
            else np.zeros(len(tasks[name].contact_mask), dtype=int) for name in names
        ], axis=1),
        "contact_mask": np.stack([tasks[name].contact_mask for name in names], axis=1),
        "contact_surface_managed": np.stack([tasks[name].managed_frames for name in names], axis=1),
        "end_effector_world_tracking": np.array([tasks[name].preserve_world_tracking for name in names]),
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
