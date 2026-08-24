"""Core OmniRetargeting functionality."""

from __future__ import annotations

from dataclasses import dataclass, replace
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
import trimesh
import mujoco
try:
    import yourdfpy
except ImportError:
    yourdfpy = None
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from .affordance import ContactRegion, build_contact_region
from .utils import compute_mesh_height_at_point
from .contact import ContactMode, SceneContactTask, infer_scene_contact_tasks
from .feasibility import (
    DROPPED_OPTIONAL,
    INFEASIBLE,
    FeasibilityOptions,
    InteractionFeasibilityResult,
    solve_interaction_feasibility,
)
from .retargeting import (
    MORPHOLOGY_SWING_TARGET_SPACES,
    _bridge_contact_orientation,
    _bridge_swing_to_contact,
    _clamp_target_to_reach,
    _contact_bridge_windows,
    _contact_segment_spans,
    _minimum_jerk_blend,
    _numeric_morphology_scale,
    _local_semantic_rotation,
    _validate_swing_target_space,
)
from .trajectory_refinement import (
    collision_geom_ids,
    geom_surface_points_in_body_frame,
    measure_scene_penetration,
    refine_joint_trajectory,
    resolve_scene_normal_orientation,
    scene_signed_clearance,
)
from .evaluation import evaluate_retargeted_motion
from .morphology import (
    analyze_robot_morphology,
    resolve_end_effector_chain_specs,
)


def _build_per_frame_sigma_scales(
    contact_modes: np.ndarray,
    contact_segment_ids: np.ndarray,
    contact_bridge_alpha: np.ndarray,
    config: Mapping[str, Any],
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    """Derive optional tolerance scales from contact semantics and bridge alpha.

    The static framewise/refinement sigmas remain the locked tolerances.  When
    enabled, free motion starts at a configurable loose multiple and follows
    the position bridge monotonically toward the owning segment's scale.
    SUPPORT reaches exactly 1.0, TOUCH stays loose, and SLIDE keeps orientation
    loose so its tangential heading is never promoted to a full SE(3) lock.
    """
    raw_policy = config.get("time_varying_tolerance", {})
    if isinstance(raw_policy, Mapping):
        policy = dict(raw_policy)
        enabled = bool(policy.get("enabled", False))
    else:
        policy = {}
        enabled = bool(raw_policy)
    if not enabled:
        return None, None

    modes = np.asarray(contact_modes, dtype=np.uint8)
    segment_ids = np.asarray(contact_segment_ids, dtype=np.int32)
    alpha = np.asarray(contact_bridge_alpha, dtype=float)
    num_frames = len(modes)
    if segment_ids.shape != (num_frames,) or alpha.shape != (num_frames,):
        raise ValueError(
            "contact_segment_ids and contact_bridge_alpha must match contact_modes."
        )
    if np.any((alpha < 0.0) | (alpha > 1.0)):
        raise ValueError("contact_bridge_alpha must lie in [0, 1].")

    loose_position = max(float(policy.get("loose_position_scale", 2.0)), 1.0)
    loose_orientation = max(
        float(policy.get("loose_orientation_scale", 2.0)), 1.0
    )
    slide_orientation = max(
        float(policy.get("slide_orientation_scale", loose_orientation)),
        1.0,
    )
    position_scale = np.full(num_frames, loose_position, dtype=float)
    orientation_scale = np.full(num_frames, loose_orientation, dtype=float)

    def locked_scales(mode_value: int) -> Tuple[float, float]:
        mode = ContactMode(int(mode_value))
        if mode == ContactMode.SUPPORT:
            return 1.0, 1.0
        if mode == ContactMode.SLIDE:
            return 1.0, slide_orientation
        # TOUCH remains a transient point contact; it is not an SE(3) lock.
        return loose_position, loose_orientation

    spans = _contact_segment_spans(segment_ids)
    bridge_frames = max(
        int(
            config.get(
                "contact_bridge_frames",
                (config.get("contact_detection", {}) or {}).get(
                    "transition_frames", 0
                ),
            )
        ),
        0,
    )
    windows = _contact_bridge_windows(spans, num_frames, bridge_frames)
    for (start, end), (approach_zero, release_zero) in zip(spans, windows):
        for frame_idx in range(start, end):
            position_scale[frame_idx], orientation_scale[frame_idx] = locked_scales(
                int(modes[frame_idx])
            )
        for frame_idx in range(approach_zero, start):
            target_position, target_orientation = locked_scales(int(modes[start]))
            blend = float(alpha[frame_idx])
            position_scale[frame_idx] = (
                loose_position + blend * (target_position - loose_position)
            )
            orientation_scale[frame_idx] = (
                loose_orientation
                + blend * (target_orientation - loose_orientation)
            )
        for frame_idx in range(end, release_zero + 1):
            target_position, target_orientation = locked_scales(
                int(modes[end - 1])
            )
            blend = float(alpha[frame_idx])
            position_scale[frame_idx] = (
                loose_position + blend * (target_position - loose_position)
            )
            orientation_scale[frame_idx] = (
                loose_orientation
                + blend * (target_orientation - loose_orientation)
            )
    return position_scale, orientation_scale


def _source_limb_chain(
    name: str,
    spec: Mapping[str, Any] | None = None,
) -> Tuple[str, str, str]:
    """Return the source limb chain (base, middle, tip) for one effector.

    The chain is the kinematic path whose summed segment lengths define the
    limb's reach, and whose first joint is the limb base used as the anchor of
    ``limb_morphology_scaled``.  A profile may override it explicitly for a
    source skeleton that does not use the SMPL-X joint names.
    """
    configured = dict(spec or {}).get("source_limb_chain")
    if configured is not None:
        chain = tuple(str(joint) for joint in configured)
        if len(chain) != 3:
            raise ValueError(
                f"End-effector '{name}' source_limb_chain must list exactly "
                f"[base, middle, tip] joints, got {list(configured)!r}."
            )
        return chain
    side = "L" if name.startswith("left") else "R"
    if name.endswith("hand"):
        return (f"{side}_Shoulder", f"{side}_Elbow", f"{side}_Wrist")
    return (f"{side}_Hip", f"{side}_Knee", f"{side}_Ankle")


def _resolve_swing_targets_against_scene(
    scaled_targets: np.ndarray,
    world_targets: np.ndarray,
    scene_mesh: trimesh.Trimesh | None,
    clearance: float,
    free_mask: np.ndarray | None = None,
    iterations: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fade a morphology-scaled swing target back to world where it hits the scene.

    Shortening a limb about its base pulls the endpoint toward the body, so an
    extended human reach over a step can map to a robot target *inside* that
    step.  Such a target is unattainable and actively drags the limb into the
    geometry, which is exactly the free-space penetration the scaled target was
    meant to avoid.

    The registered world target is the one reference known to sit outside the
    scene, so each offending frame is blended back toward it by the smallest
    fraction that clears the surface.  Free-space frames keep the scaled target
    untouched, and the fade degrades to plain world tracking only where the
    geometry demands it.  Returns the resolved targets and the per-frame blend.
    """
    scaled = np.asarray(scaled_targets, dtype=float)
    world = np.asarray(world_targets, dtype=float)
    blend = np.zeros(len(scaled), dtype=float)
    if scene_mesh is None or len(scaled) == 0:
        return scaled.copy(), blend
    margin = max(float(clearance), 0.0)
    # The registered source markers are the free-side reference that fixes the
    # global normal orientation; the scan is not watertight, so an inside test
    # based on ray containment would be unreliable here.
    orientation_sign = resolve_scene_normal_orientation(scene_mesh, world)

    def outside(points: np.ndarray) -> np.ndarray:
        clearances = scene_signed_clearance(
            scene_mesh, points, orientation_sign
        )[0]
        return clearances >= margin

    pending = ~outside(scaled)
    if free_mask is not None:
        # Contact cores are replaced by their scene anchors downstream, so
        # only free-space frames carry a scaled target worth resolving.
        pending &= np.asarray(free_mask, dtype=bool)
    if not np.any(pending):
        return scaled.copy(), blend
    # The world target itself may be inside the scene on a badly reconstructed
    # frame.  Those frames keep the honest world reference rather than an
    # invented one, matching how world-space effectors are treated.
    lower = np.zeros(len(scaled), dtype=float)
    upper = np.ones(len(scaled), dtype=float)
    for _ in range(max(int(iterations), 1)):
        middle = 0.5 * (lower + upper)
        candidate = scaled + middle[:, None] * (world - scaled)
        clear = outside(candidate)
        upper = np.where(pending & clear, middle, upper)
        lower = np.where(pending & ~clear, middle, lower)
    blend = np.where(pending, upper, 0.0)
    return scaled + blend[:, None] * (world - scaled), blend


def _base_quaternions_wxyz(
    base_orientations: np.ndarray | None,
    num_frames: int,
) -> np.ndarray | None:
    """Normalise a source-neutral base orientation track to wxyz quaternions."""
    if base_orientations is None:
        return None
    base_orientations = np.asarray(base_orientations, dtype=float)
    if base_orientations.shape == (num_frames, 4):
        norms = np.linalg.norm(base_orientations, axis=1, keepdims=True)
        if np.any(norms < 1e-10):
            raise ValueError("base_orientations contains a zero quaternion.")
        quaternions = base_orientations / norms
    elif base_orientations.shape == (num_frames, 3):
        # Backward-compatible axis-angle input. Source adapters should
        # normally convert their native convention to wxyz first.
        quaternions = Rotation.from_rotvec(base_orientations).as_quat(
            scalar_first=True
        )
    else:
        raise ValueError(
            "base_orientations must have shape (T, 4) for source-neutral "
            "wxyz quaternions or legacy shape (T, 3) for rotation vectors; "
            f"got {base_orientations.shape}."
        )
    for frame_idx in range(1, len(quaternions)):
        if np.dot(quaternions[frame_idx - 1], quaternions[frame_idx]) < 0.0:
            quaternions[frame_idx] *= -1.0
    return quaternions


def _root_positions_aligning_robot_body(
    model: mujoco.MjModel,
    source_body_positions: np.ndarray,
    robot_body_name: str,
    root_quaternions_wxyz: np.ndarray,
) -> np.ndarray:
    """Place the floating root so a nominal robot body matches a source body.

    The body offset is measured from the compiled robot at ``qpos0`` and
    rotated by the per-frame source root orientation. This replaces a
    hand-tuned global height offset with explicit anatomical registration.
    """
    targets = np.asarray(source_body_positions, dtype=float)
    quaternions = np.asarray(root_quaternions_wxyz, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 3:
        raise ValueError(
            "source body positions must have shape (T, 3), got "
            f"{targets.shape}."
        )
    if quaternions.shape != (len(targets), 4):
        raise ValueError(
            "root quaternions must have shape "
            f"({len(targets)}, 4), got {quaternions.shape}."
        )
    if model.njnt == 0 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("robot-body base alignment requires a floating-base model.")
    root_body_id = int(model.jnt_bodyid[0])
    robot_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, str(robot_body_name)
    )
    if robot_body_id < 0:
        raise ValueError(
            "base_position_constraint.robot_body refers to missing MuJoCo "
            f"body {robot_body_name!r}."
        )

    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(model.qpos0, dtype=float)
    mujoco.mj_forward(model, data)
    root_rotation0 = data.xmat[root_body_id].reshape(3, 3)
    body_offset_root = root_rotation0.T @ (
        data.xpos[robot_body_id] - data.xpos[root_body_id]
    )
    root_rotations = Rotation.from_quat(
        quaternions, scalar_first=True
    ).as_matrix()
    body_offsets_world = np.einsum(
        "tij,j->ti", root_rotations, body_offset_root
    )
    return targets - body_offsets_world


def _feasibility_options_from_config(
    config: Mapping[str, Any],
    root_warp_config: Mapping[str, Any],
    framerate: float,
) -> FeasibilityOptions:
    """Build one feasibility limit set coordinated with the root shaper."""
    if not np.isfinite(framerate) or float(framerate) <= 0.0:
        raise ValueError(
            f"interaction_feasibility framerate must be positive, got {framerate!r}."
        )
    root_cfg = dict(root_warp_config or {})
    values: Dict[str, Any] = {"dt": 1.0 / float(framerate)}
    if "max_xyz" in root_cfg:
        values["max_root_translation"] = tuple(
            np.asarray(root_cfg["max_xyz"], dtype=float).reshape(3)
        )
    if "max_translation_norm" in root_cfg:
        values["max_root_translation_norm"] = float(
            root_cfg["max_translation_norm"]
        )
    elif "max_norm" in root_cfg and float(root_cfg["max_norm"]) > 0.0:
        values["max_root_translation_norm"] = float(root_cfg["max_norm"])
    if "max_yaw" in root_cfg:
        values["max_root_yaw"] = (
            float(root_cfg["max_yaw"])
            if bool(root_cfg.get("yaw_enabled", False))
            else 0.0
        )
    if root_cfg.get("max_linear_velocity") is not None:
        limit = float(root_cfg["max_linear_velocity"])
        values["max_root_velocity"] = (limit, limit, limit)
    if root_cfg.get("max_linear_acceleration") is not None:
        limit = float(root_cfg["max_linear_acceleration"])
        values["max_root_acceleration"] = (limit, limit, limit)
    if root_cfg.get("max_yaw_rate") is not None:
        values["max_yaw_velocity"] = float(root_cfg["max_yaw_rate"])
    if root_cfg.get("max_yaw_acceleration") is not None:
        values["max_yaw_acceleration"] = float(
            root_cfg["max_yaw_acceleration"]
        )

    raw_options = config.get("options", config)
    if not isinstance(raw_options, Mapping):
        raise ValueError("interaction_feasibility.options must be an object.")
    option_names = set(FeasibilityOptions.__dataclass_fields__)
    values.update(
        {
            str(name): value
            for name, value in raw_options.items()
            if str(name) in option_names
        }
    )
    return FeasibilityOptions(**values)


def _coordinated_root_warp_config(
    config: Mapping[str, Any], options: FeasibilityOptions
) -> Dict[str, Any]:
    """Make the shaper consume exactly the feasibility amplitude decisions."""
    result = dict(config or {})
    result.update(
        {
            "enabled": True,
            "max_xyz": list(options.max_root_translation),
            "max_translation_norm": float(options.max_root_translation_norm),
            "yaw_enabled": bool(options.max_root_yaw > 0.0),
            "max_yaw": float(options.max_root_yaw),
        }
    )
    if "max_linear_velocity" not in result:
        result["max_linear_velocity"] = float(
            np.linalg.norm(np.asarray(options.max_root_velocity, dtype=float))
        )
    if "max_linear_acceleration" not in result:
        result["max_linear_acceleration"] = float(
            np.linalg.norm(np.asarray(options.max_root_acceleration, dtype=float))
        )
    result.setdefault("max_yaw_rate", float(options.max_yaw_velocity))
    result.setdefault("max_yaw_acceleration", float(options.max_yaw_acceleration))
    return result


def _build_interaction_contact_regions(
    tasks: Mapping[str, SceneContactTask],
    scene_mesh: trimesh.Trimesh,
    config: Mapping[str, Any],
) -> Dict[Tuple[str, int], ContactRegion]:
    """Extract one immutable source-conditioned scene region per contact run."""
    affordance_cfg = dict(config.get("affordance", {}) or {})
    default_radius = affordance_cfg.get("neighborhood_radius")
    default_object_id = affordance_cfg.get("object_id")
    per_effector = dict(affordance_cfg.get("effectors", {}) or {})
    regions: Dict[Tuple[str, int], ContactRegion] = {}
    for name, task in tasks.items():
        effector_cfg = dict(per_effector.get(name, {}) or {})
        radius = effector_cfg.get("neighborhood_radius", default_radius)
        object_id = effector_cfg.get("object_id", default_object_id)
        segment_ids = np.asarray(task.contact_segment_ids, dtype=np.int32)
        for raw_segment_id in np.unique(segment_ids[segment_ids >= 0]):
            segment_id = int(raw_segment_id)
            frames = np.flatnonzero(segment_ids == segment_id)
            kwargs: Dict[str, Any] = {}
            if radius is not None:
                kwargs["neighborhood_radius"] = float(radius)
            if object_id is not None:
                kwargs["object_id"] = str(object_id)
            regions[(str(name), segment_id)] = build_contact_region(
                scene_mesh,
                np.asarray(task.surface_points, dtype=float)[frames],
                np.asarray(task.surface_normals, dtype=float)[frames],
                **kwargs,
            )
    return regions


def _apply_feasibility_contact_choices(
    tasks: Mapping[str, SceneContactTask],
    result: InteractionFeasibilityResult,
) -> Dict[str, SceneContactTask]:
    """Copy tasks and replace active solver targets with on-region choices."""
    chosen: Dict[str, np.ndarray] = {
        name: np.asarray(task.contact_positions, dtype=float).copy()
        for name, task in tasks.items()
    }
    for segment in result.segments:
        if segment.effector_name not in chosen:
            raise KeyError(
                f"Feasibility returned unknown effector {segment.effector_name!r}."
            )
        chosen[segment.effector_name][segment.frame_indices] = (
            segment.chosen_contact_points_world
        )
    return {
        name: replace(task, contact_positions=chosen[name])
        for name, task in tasks.items()
    }


def _build_robot_base_reference(
    human_root_positions: np.ndarray,
    tasks: Dict[str, SceneContactTask],
    reach_profiles: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    root_z_offset: float | np.ndarray,
    *,
    base_orientations: np.ndarray | None = None,
    framerate: float = 30.0,
    return_diagnostics: bool = False,
    feasibility_segment_corrections: Mapping[
        Tuple[str, int], Mapping[str, np.ndarray]
    ]
    | None = None,
) -> Any:
    """Create one bounded robot-only root warp around contact segments.

    Translation and yaw share the existing segment-median, minimum-jerk, and
    overlap-averaging path.  Derivative caps apply to the *correction* rather
    than the source root motion: deliberate source speed must remain intact.
    Caps are met by widening the approach/release windows, never by clipping
    individual frames.  ``return_diagnostics`` is opt-in so the historical
    two-array return contract remains available to callers and tests.

    ``root_z_offset`` retains its legacy name for API compatibility, but may
    now be a full xyz reference offset.  A scalar still means global z only.

    When ``feasibility_segment_corrections`` is supplied, feasibility is the
    sole authority for correction amplitude and direction.  This function
    only turns each decided segment core into the existing smooth,
    derivative-limited trajectory.  The legacy reach-excess calculation is
    used only when that argument is ``None``; the two paths are never summed.
    """
    source_root = np.asarray(human_root_positions, dtype=float)
    base_reference = source_root.copy()
    root_offset = np.asarray(root_z_offset, dtype=float)
    if root_offset.ndim == 0:
        reference_offset = np.array([0.0, 0.0, float(root_offset)], dtype=float)
    else:
        reference_offset = root_offset.reshape(3).copy()
    base_reference += reference_offset[None, :]
    unwarped = base_reference.copy()
    num_frames = len(source_root)
    yaw_correction = np.zeros(num_frames, dtype=float)
    empty_diagnostics = {
        "strategy": "minimum_jerk_window_widening",
        "max_translation": 0.0,
        "max_yaw": 0.0,
        "max_linear_velocity": 0.0,
        "max_linear_acceleration": 0.0,
        "max_yaw_rate": 0.0,
        "max_yaw_acceleration": 0.0,
        "derivative_slack": {
            "linear_velocity": 0.0,
            "linear_acceleration": 0.0,
            "yaw_rate": 0.0,
            "yaw_acceleration": 0.0,
        },
        "segment_windows": [],
    }
    if not bool(config.get("enabled", False)) or not tasks:
        if return_diagnostics:
            return base_reference, unwarped, yaw_correction, empty_diagnostics
        return base_reference, unwarped

    if not np.isfinite(framerate) or float(framerate) <= 0.0:
        raise ValueError(f"root_warp framerate must be positive, got {framerate!r}.")
    framerate = float(framerate)
    approach_frames = max(int(config.get("approach_frames", 8)), 0)
    release_frames = max(int(config.get("release_frames", 8)), 0)
    max_xyz = np.asarray(
        config.get("max_xyz", [0.08, 0.08, 0.06]), dtype=float
    ).reshape(3)
    if np.any(~np.isfinite(max_xyz)) or np.any(max_xyz < 0.0):
        raise ValueError("root_warp.max_xyz must contain three non-negative values.")
    uses_new_norm_key = "max_translation_norm" in config
    max_norm = float(
        config.get("max_translation_norm", config.get("max_norm", 0.10))
    )
    if not np.isfinite(max_norm) or max_norm < 0.0:
        raise ValueError("root_warp.max_translation_norm must be non-negative.")
    yaw_enabled = bool(config.get("yaw_enabled", False))
    max_yaw = float(config.get("max_yaw", 0.0))
    if not np.isfinite(max_yaw) or max_yaw < 0.0:
        raise ValueError("root_warp.max_yaw must be non-negative radians.")

    def derivative_limit(name: str) -> float:
        if name not in config or config[name] is None:
            return np.inf
        value = float(config[name])
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"root_warp.{name} must be non-negative.")
        return value

    limits = {
        "linear_velocity": derivative_limit("max_linear_velocity"),
        "linear_acceleration": derivative_limit("max_linear_acceleration"),
        "yaw_rate": derivative_limit("max_yaw_rate"),
        "yaw_acceleration": derivative_limit("max_yaw_acceleration"),
    }
    base_quaternions = _base_quaternions_wxyz(base_orientations, num_frames)
    base_rotations = (
        None
        if base_quaternions is None
        else Rotation.from_quat(base_quaternions, scalar_first=True).as_matrix()
    )
    raw_qualifying_modes = config.get(
        "qualifying_contact_modes", [ContactMode.SUPPORT.name]
    )
    if isinstance(raw_qualifying_modes, (str, int, np.integer, ContactMode)):
        raw_qualifying_modes = [raw_qualifying_modes]
    try:
        qualifying_modes = {
            int(value)
            if isinstance(value, (int, np.integer, ContactMode))
            else int(ContactMode[str(value).strip().upper()])
            for value in raw_qualifying_modes
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "root_warp.qualifying_contact_modes must contain contact-mode "
            "names or ids (NONE, TOUCH, SLIDE, SUPPORT)."
        ) from exc

    segment_warps: List[Dict[str, Any]] = []

    def minimum_yaw_to_reach(
        root_to_target: np.ndarray,
        origin_offset_w: np.ndarray,
        max_reach: float,
    ) -> float:
        """Return the smallest world-z yaw that can reduce a reach excess."""
        current_delta = root_to_target - origin_offset_w
        if float(np.linalg.norm(current_delta)) <= max_reach + 1e-10:
            return 0.0
        target_radius = float(np.linalg.norm(root_to_target[:2]))
        offset_radius = float(np.linalg.norm(origin_offset_w[:2]))
        if target_radius < 1e-10 or offset_radius < 1e-10:
            return 0.0
        alignment = float(
            np.arctan2(root_to_target[1], root_to_target[0])
            - np.arctan2(origin_offset_w[1], origin_offset_w[0])
        )
        alignment = float((alignment + np.pi) % (2.0 * np.pi) - np.pi)
        vertical_error = float(root_to_target[2] - origin_offset_w[2])
        cosine_needed = (
            target_radius**2
            + offset_radius**2
            + vertical_error**2
            - max_reach**2
        ) / max(2.0 * target_radius * offset_radius, 1e-12)
        if cosine_needed <= -1.0:
            return 0.0
        allowed_residual = (
            0.0
            if cosine_needed >= 1.0
            else float(np.arccos(np.clip(cosine_needed, -1.0, 1.0)))
        )
        magnitude = max(abs(alignment) - allowed_residual, 0.0)
        return float(np.copysign(magnitude, alignment))

    feasibility_decided = feasibility_segment_corrections is not None
    for name, task in tasks.items():
        profile = reach_profiles.get(name)
        if not feasibility_decided and not profile:
            continue
        modes = np.asarray(task.contact_modes, dtype=np.uint8)
        segment_ids = np.asarray(task.contact_segment_ids, dtype=np.int32)
        for segment_id in np.unique(segment_ids[segment_ids >= 0]):
            frames = np.flatnonzero(segment_ids == segment_id)
            if len(frames) == 0:
                continue
            # SUPPORT-only remains the default. Experiments may explicitly let
            # other contact semantics qualify without changing their schedule.
            if not np.any(np.isin(modes[frames], tuple(qualifying_modes))):
                continue
            if feasibility_decided:
                assert feasibility_segment_corrections is not None
                target = feasibility_segment_corrections.get(
                    (str(name), int(segment_id))
                )
                if target is None:
                    continue

                def segment_values(
                    field: str, width: int | None = None
                ) -> np.ndarray:
                    values = np.asarray(target[field], dtype=float)
                    expected_tail = () if width is None else (width,)
                    if values.shape == (num_frames, *expected_tail):
                        return values[frames]
                    if values.shape == (len(frames), *expected_tail):
                        return values
                    raise ValueError(
                        "Feasibility segment correction "
                        f"{(name, int(segment_id))!r} field {field!r} has "
                        f"shape {values.shape}; expected a full {num_frames}-frame "
                        f"track or {len(frames)} segment samples."
                    )

                correction = np.median(
                    segment_values("translation", width=3), axis=0
                )
                segment_yaw = float(np.median(segment_values("yaw")))
                norm_cap_active = uses_new_norm_key or max_norm > 0.0
                if np.any(np.abs(correction) > max_xyz + 1e-8) or (
                    norm_cap_active
                    and float(np.linalg.norm(correction)) > max_norm + 1e-8
                ):
                    raise ValueError(
                        "Feasibility root translation exceeds the coordinated "
                        "root_warp amplitude cap; use one shared set of limits."
                    )
                if abs(segment_yaw) > max_yaw + 1e-8:
                    raise ValueError(
                        "Feasibility root yaw exceeds the coordinated root_warp "
                        "amplitude cap; use one shared set of limits."
                    )
            else:
                assert profile is not None
                origin_offset = np.asarray(
                    profile["origin_offset"], dtype=float
                ).reshape(3)
                if yaw_enabled and base_rotations is not None:
                    origin_offsets_w = np.einsum(
                        "tij,j->ti", base_rotations[frames], origin_offset
                    )
                else:
                    # Preserve the exact historical translation numerics unless
                    # yaw has explicitly been enabled with an orientation track.
                    origin_offsets_w = np.repeat(
                        origin_offset[None, :], len(frames), axis=0
                    )
                origins = unwarped[frames] + origin_offsets_w
                deltas = task.contact_positions[frames] - origins
                distances = np.linalg.norm(deltas, axis=1)
                max_reach = max(float(profile["max_reach"]), 1e-6)
                excess = np.maximum(distances - max_reach, 0.0)
                desired = deltas * (
                    excess / np.maximum(distances, 1e-10)
                )[:, None]
                correction = np.median(desired, axis=0)
                correction = np.clip(correction, -max_xyz, max_xyz)
                correction_norm = float(np.linalg.norm(correction))
                # Legacy max_norm=0 meant "no norm clamp".  The new explicit key
                # follows ordinary cap semantics, where zero disables translation.
                norm_cap_active = uses_new_norm_key or max_norm > 0.0
                if norm_cap_active and correction_norm > max_norm:
                    correction *= max_norm / max(correction_norm, 1e-12)

                segment_yaw = 0.0
                if yaw_enabled and max_yaw > 0.0 and base_rotations is not None:
                    desired_yaw = np.asarray(
                        [
                            minimum_yaw_to_reach(
                                task.contact_positions[frame_idx]
                                - (unwarped[frame_idx] + correction),
                                offset_w,
                                max_reach,
                            )
                            for frame_idx, offset_w in zip(frames, origin_offsets_w)
                        ],
                        dtype=float,
                    )
                    segment_yaw = float(
                        np.clip(np.median(desired_yaw), -max_yaw, max_yaw)
                    )
            if (
                float(np.linalg.norm(correction)) < 1e-8
                and abs(segment_yaw) < 1e-8
            ):
                continue
            start = int(frames[0])
            end = int(frames[-1]) + 1
            segment_warps.append(
                {
                    "name": name,
                    "segment_id": int(segment_id),
                    "start": start,
                    "end": end,
                    "approach": min(approach_frames, start),
                    "release": min(release_frames, num_frames - end),
                    "translation": correction.copy(),
                    "yaw": segment_yaw,
                }
            )

    def assemble_correction() -> Tuple[np.ndarray, np.ndarray]:
        correction_sum = np.zeros((num_frames, 3), dtype=float)
        correction_weight = np.zeros(num_frames, dtype=float)
        yaw_sum = np.zeros(num_frames, dtype=float)
        for segment in segment_warps:
            start = int(segment["start"])
            end = int(segment["end"])
            window_start = start - int(segment["approach"])
            window_end = end + int(segment["release"])
            weights = np.ones(window_end - window_start, dtype=float)
            if start > window_start:
                pre = np.arange(window_start, start + 1)
                weights[: len(pre)] = _minimum_jerk_blend(
                    (pre - window_start) / max(start - window_start, 1)
                )
            if window_end > end:
                post = np.arange(end - 1, window_end)
                weights[-len(post) :] = 1.0 - _minimum_jerk_blend(
                    (post - (end - 1)) / max(window_end - end, 1)
                )
            correction_sum[window_start:window_end] += (
                weights[:, None] * segment["translation"]
            )
            yaw_sum[window_start:window_end] += weights * float(segment["yaw"])
            correction_weight[window_start:window_end] += weights

        translation = np.zeros((num_frames, 3), dtype=float)
        yaw = np.zeros(num_frames, dtype=float)
        active = correction_weight > 1e-10
        # A lone segment must retain its minimum-jerk amplitude.  Normalize
        # only the excess weight created by overlapping segment windows;
        # dividing by a sub-unit single-segment weight would cancel the ramp
        # and recreate a one-frame step at each nonzero endpoint.
        normalization = np.maximum(correction_weight[active], 1.0)
        translation[active] = (
            correction_sum[active] / normalization[:, None]
        )
        yaw[active] = yaw_sum[active] / normalization
        return translation, yaw

    def correction_metrics(
        translation: np.ndarray, yaw: np.ndarray
    ) -> Dict[str, float]:
        linear_velocity = np.diff(translation, axis=0) * framerate
        linear_acceleration = np.diff(translation, n=2, axis=0) * framerate**2
        yaw_rate = np.diff(yaw) * framerate
        yaw_acceleration = np.diff(yaw, n=2) * framerate**2

        def max_row_norm(values: np.ndarray) -> float:
            return (
                float(np.max(np.linalg.norm(values, axis=1)))
                if len(values)
                else 0.0
            )

        return {
            "max_translation": max_row_norm(translation),
            "max_yaw": float(np.max(np.abs(yaw))) if len(yaw) else 0.0,
            "max_linear_velocity": max_row_norm(linear_velocity),
            "max_linear_acceleration": max_row_norm(linear_acceleration),
            "max_yaw_rate": (
                float(np.max(np.abs(yaw_rate))) if len(yaw_rate) else 0.0
            ),
            "max_yaw_acceleration": (
                float(np.max(np.abs(yaw_acceleration)))
                if len(yaw_acceleration)
                else 0.0
            ),
        }

    def limit_violations(metrics: Mapping[str, float]) -> Dict[str, float]:
        return {
            name: max(float(metrics[f"max_{name}"]) - limit, 0.0)
            if np.isfinite(limit)
            else 0.0
            for name, limit in limits.items()
        }

    translation_correction, yaw_correction = assemble_correction()
    metrics = correction_metrics(translation_correction, yaw_correction)
    violations = limit_violations(metrics)

    # Strategy (a): widen complete minimum-jerk windows.  This preserves the
    # quintic shape and segment overlap semantics; it never clips a frame.
    for _ in range(max(2 * num_frames, 1)):
        if not any(value > 1e-10 for value in violations.values()):
            break
        linear_bad = (
            violations["linear_velocity"] > 1e-10
            or violations["linear_acceleration"] > 1e-10
        )
        yaw_bad = (
            violations["yaw_rate"] > 1e-10
            or violations["yaw_acceleration"] > 1e-10
        )
        changed = False
        for segment in segment_warps:
            relevant = (
                linear_bad
                and float(np.linalg.norm(segment["translation"])) > 1e-10
            ) or (yaw_bad and abs(float(segment["yaw"])) > 1e-10)
            if not relevant:
                continue
            if int(segment["approach"]) < int(segment["start"]):
                segment["approach"] = int(segment["approach"]) + 1
                changed = True
            if int(segment["release"]) < num_frames - int(segment["end"]):
                segment["release"] = int(segment["release"]) + 1
                changed = True
        if not changed:
            break
        translation_correction, yaw_correction = assemble_correction()
        metrics = correction_metrics(translation_correction, yaw_correction)
        violations = limit_violations(metrics)

    base_reference += translation_correction
    diagnostics = {
        "strategy": (
            "feasibility_decision_minimum_jerk_window_widening"
            if feasibility_decided
            else "minimum_jerk_window_widening"
        ),
        "correction_authority": (
            "interaction_feasibility" if feasibility_decided else "reach_excess"
        ),
        **metrics,
        "derivative_slack": violations,
        "segment_windows": [
            {
                "name": segment["name"],
                "segment_id": segment["segment_id"],
                "start": segment["start"],
                "end": segment["end"],
                "approach_frames": segment["approach"],
                "release_frames": segment["release"],
            }
            for segment in segment_warps
        ],
    }
    nonzero_slack = {
        name: value for name, value in violations.items() if value > 1e-10
    }
    if nonzero_slack:
        formatted = ", ".join(
            f"{name}={value:.6g}" for name, value in nonzero_slack.items()
        )
        print(
            "Root warp derivative-limit slack after exhausting available "
            f"minimum-jerk windows: {formatted}."
        )

    if return_diagnostics:
        return base_reference, unwarped, yaw_correction, diagnostics
    return base_reference, unwarped


@dataclass
class _SupportHandoffEvent:
    """Runtime state for one explicitly configured support-transfer event."""

    kind: str
    old_supports: Tuple[str, ...]
    new_supports: Tuple[str, ...]
    start: int
    end: int
    requested_release: int
    anchor_frames: Dict[str, int]
    hold_frames: int
    position_error_threshold: float
    max_penetration: float
    min_bridge_alpha: float
    stable_count: int = 0
    established_frame: int | None = None
    flight_release_frame: int | None = None


class _SupportHandoffGuard:
    """Delay configured support releases until a new support is established.

    This is deliberately not a second contact-phase state machine.  Locked
    support and requested release are read directly from each task's
    ``(ContactMode, contact_bridge_alpha, contact_segment_ids)`` tracks.  Only
    events explicitly marked ``support_transfer`` or ``quasi_static`` are
    guarded.  Explicit flight/jump/roll/full-release windows always win.
    """

    _GUARDED_KINDS = {"support_transfer", "quasi_static"}
    _FLIGHT_KINDS = {
        "flight",
        "jump",
        "roll",
        "full_release",
        "intentional_full_release",
    }

    def __init__(
        self,
        tasks: Mapping[str, SceneContactTask],
        config: Mapping[str, Any] | bool | None,
    ) -> None:
        self.tasks = dict(tasks)
        self.num_frames = (
            len(next(iter(self.tasks.values())).contact_modes) if self.tasks else 0
        )
        self.events: List[_SupportHandoffEvent] = []
        self.override_sources: Dict[str, Dict[int, int]] = {
            name: {} for name in self.tasks
        }
        self.flight_mask = np.zeros(self.num_frames, dtype=bool)

        if isinstance(config, Mapping):
            policy = dict(config)
            enabled = bool(policy.get("enabled", False))
        else:
            policy = {}
            enabled = bool(config)
        self.enabled = enabled and bool(self.tasks)
        if not self.enabled:
            return

        raw_events = policy.get("events", [])
        if not isinstance(raw_events, list):
            raise ValueError("support_handoff.events must be a list.")
        for raw_event in raw_events:
            if not isinstance(raw_event, Mapping):
                raise ValueError("Each support_handoff event must be an object.")
            event = dict(raw_event)
            kind = str(
                event.get("type", event.get("kind", ""))
            ).strip().lower().replace("-", "_")
            start = int(event.get("start_frame", 0))
            end = int(event.get("end_frame", self.num_frames))
            if not (0 <= start < end <= self.num_frames):
                raise ValueError(
                    "support_handoff event windows must satisfy "
                    f"0 <= start_frame < end_frame <= {self.num_frames}."
                )
            if kind in self._FLIGHT_KINDS:
                self.flight_mask[start:end] = True

        for raw_event in raw_events:
            event = dict(raw_event)
            kind = str(
                event.get("type", event.get("kind", ""))
            ).strip().lower().replace("-", "_")
            if kind in self._FLIGHT_KINDS:
                continue
            if kind not in self._GUARDED_KINDS:
                raise ValueError(
                    "support_handoff event type must be support_transfer, "
                    "quasi_static, flight, jump, roll, or full_release; got "
                    f"{kind!r}."
                )
            start = int(event.get("start_frame", 0))
            end = int(event.get("end_frame", self.num_frames))
            old_supports = self._name_tuple(
                event.get("old_supports", event.get("from", ()))
            )
            new_supports = self._name_tuple(
                event.get("new_supports", event.get("to", ()))
            )
            if not old_supports or not new_supports:
                raise ValueError(
                    "Guarded support_handoff events require non-empty "
                    "old_supports and new_supports."
                )
            unknown = (set(old_supports) | set(new_supports)) - set(self.tasks)
            if unknown:
                raise ValueError(
                    "support_handoff refers to unknown effectors: "
                    + ", ".join(sorted(unknown))
                )
            release_frame = event.get("release_frame")
            if release_frame is None:
                release_frame = self._derive_requested_release(
                    old_supports, start, end
                )
            release_frame = int(release_frame)
            if not (start <= release_frame < end):
                raise ValueError(
                    "support_handoff.release_frame must lie inside its event window."
                )
            anchors = {
                name: self._support_anchor_before(name, release_frame, start)
                for name in old_supports
            }
            hold_frames = max(
                int(event.get("hold_frames", policy.get("hold_frames", 3))), 1
            )
            position_threshold = float(
                event.get(
                    "position_error_threshold",
                    policy.get("position_error_threshold", 0.025),
                )
            )
            max_penetration = float(
                event.get(
                    "max_penetration",
                    policy.get("max_penetration", 0.05),
                )
            )
            min_bridge_alpha = float(
                event.get(
                    "min_bridge_alpha",
                    policy.get("min_bridge_alpha", 1.0 - 1e-8),
                )
            )
            if position_threshold < 0.0 or max_penetration < 0.0:
                raise ValueError(
                    "support_handoff error and penetration thresholds must be "
                    "non-negative."
                )
            if not 0.0 <= min_bridge_alpha <= 1.0:
                raise ValueError(
                    "support_handoff.min_bridge_alpha must lie in [0, 1]."
                )
            self.events.append(
                _SupportHandoffEvent(
                    kind=kind,
                    old_supports=old_supports,
                    new_supports=new_supports,
                    start=start,
                    end=end,
                    requested_release=release_frame,
                    anchor_frames=anchors,
                    hold_frames=hold_frames,
                    position_error_threshold=position_threshold,
                    max_penetration=max_penetration,
                    min_bridge_alpha=min_bridge_alpha,
                )
            )

    @staticmethod
    def _name_tuple(value: Any) -> Tuple[str, ...]:
        if isinstance(value, str):
            return (value,)
        return tuple(str(name) for name in value)

    @staticmethod
    def _bridge_alpha(task: SceneContactTask) -> np.ndarray:
        if task.contact_bridge_alpha is not None:
            return np.asarray(task.contact_bridge_alpha, dtype=float)
        return (
            np.asarray(task.contact_modes, dtype=np.uint8)
            != int(ContactMode.NONE)
        ).astype(float)

    @classmethod
    def _locked_support(
        cls,
        task: SceneContactTask,
        frame_idx: int,
        min_bridge_alpha: float,
    ) -> bool:
        return bool(
            int(task.contact_modes[frame_idx]) == int(ContactMode.SUPPORT)
            and int(task.contact_segment_ids[frame_idx]) >= 0
            and float(cls._bridge_alpha(task)[frame_idx]) >= min_bridge_alpha
        )

    def _derive_requested_release(
        self,
        old_supports: Tuple[str, ...],
        start: int,
        end: int,
    ) -> int:
        releases: List[int] = []
        for name in old_supports:
            task = self.tasks[name]
            alpha = self._bridge_alpha(task)
            for frame_idx in range(max(start, 1), end):
                previous_locked = bool(
                    int(task.contact_modes[frame_idx - 1])
                    == int(ContactMode.SUPPORT)
                    and int(task.contact_segment_ids[frame_idx - 1]) >= 0
                    and float(alpha[frame_idx - 1]) >= 1.0 - 1e-8
                )
                current_locked = self._locked_support(
                    task, frame_idx, 1.0 - 1e-8
                )
                if previous_locked and not current_locked:
                    releases.append(frame_idx)
                    break
        if not releases:
            raise ValueError(
                "support_handoff could not derive a SUPPORT release inside the "
                "event; configure release_frame explicitly."
            )
        return min(releases)

    def _support_anchor_before(
        self, name: str, release_frame: int, event_start: int
    ) -> int:
        task = self.tasks[name]
        candidates = [
            frame_idx
            for frame_idx in range(release_frame - 1, event_start - 1, -1)
            if self._locked_support(task, frame_idx, 1.0 - 1e-8)
        ]
        if not candidates:
            raise ValueError(
                f"support_handoff old support {name!r} has no locked SUPPORT "
                "frame before release."
            )
        return candidates[0]

    @property
    def new_support_names(self) -> Tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                name for event in self.events for name in event.new_supports
            )
        )

    def overrides_for_frame(self, frame_idx: int) -> Dict[str, int]:
        """Return old-support frame anchors that must remain active now."""
        overrides: Dict[str, int] = {}
        if not self.enabled:
            return overrides
        for event in self.events:
            if frame_idx < event.start:
                continue
            if self.flight_mask[frame_idx]:
                if event.flight_release_frame is None:
                    event.flight_release_frame = frame_idx
                continue
            release_allowed = (
                event.flight_release_frame is not None
                or (
                    event.established_frame is not None
                    and frame_idx > event.established_frame
                )
            )
            if frame_idx >= event.requested_release and not release_allowed:
                for name, anchor_frame in event.anchor_frames.items():
                    overrides[name] = anchor_frame
                    self.override_sources[name][frame_idx] = anchor_frame
        return overrides

    def needs_observation(self, frame_idx: int) -> bool:
        """Whether an unresolved guarded event can advance on this frame."""
        return any(
            event.established_frame is None
            and event.flight_release_frame is None
            and event.start <= frame_idx < event.end
            for event in self.events
        )

    def payload(
        self,
        name: str,
        frame_idx: int,
        anchor_frame: int,
    ) -> Dict[str, Any]:
        """Return a locked old-support payload with the current root reference."""
        task = self.tasks[name]
        payload = dict(task.frame(anchor_frame))
        payload["contact_mode"] = ContactMode.SUPPORT.name
        payload["transition_weight"] = 1.0
        if task.robot_base_reference_w is not None:
            payload["robot_base_reference_w"] = task.robot_base_reference_w[
                frame_idx
            ]
        return payload

    def observe(
        self,
        frame_idx: int,
        position_errors: Mapping[str, float],
        penetration_depths: Mapping[str, float] | float,
    ) -> None:
        """Observe the just-solved frame and advance consecutive hold counts."""
        if not self.enabled:
            return
        for event in self.events:
            if (
                event.established_frame is not None
                or event.flight_release_frame is not None
                or not event.start <= frame_idx < event.end
            ):
                continue
            stable = True
            for name in event.new_supports:
                task = self.tasks[name]
                error = float(position_errors.get(name, np.inf))
                if isinstance(penetration_depths, Mapping):
                    penetration = float(penetration_depths.get(name, np.inf))
                else:
                    penetration = float(penetration_depths)
                stable &= self._locked_support(
                    task, frame_idx, event.min_bridge_alpha
                )
                stable &= error <= event.position_error_threshold
                stable &= penetration <= event.max_penetration
            event.stable_count = event.stable_count + 1 if stable else 0
            if event.stable_count >= event.hold_frames:
                event.established_frame = frame_idx

    def materialize(
        self, tasks: Mapping[str, SceneContactTask]
    ) -> Dict[str, SceneContactTask]:
        """Copy runtime support holds into tasks consumed by refinement/export."""
        result = dict(tasks)
        for name, frame_sources in self.override_sources.items():
            if not frame_sources:
                continue
            task = result[name]

            def copied(field_name: str) -> np.ndarray:
                return np.asarray(getattr(task, field_name)).copy()

            contact_positions = copied("contact_positions")
            surface_points = copied("surface_points")
            surface_normals = copied("surface_normals")
            distances = copied("distances")
            relative_speeds = copied("relative_speeds")
            contact_confidence = copied("contact_confidence")
            contact_modes = copied("contact_modes")
            segment_ids = copied("contact_segment_ids")
            transition_weights = copied("transition_weights")
            optional_names = (
                "forward_directions",
                "up_directions",
                "contact_bridge_alpha",
                "contact_bridge_positions",
                "orientation_target_rotations",
                "orientation_bridge_alpha",
                "position_sigma_scale",
                "orientation_sigma_scale",
            )
            optional = {
                field_name: (
                    None
                    if getattr(task, field_name) is None
                    else copied(field_name)
                )
                for field_name in optional_names
            }
            for frame_idx, anchor_frame in frame_sources.items():
                for values in (
                    contact_positions,
                    surface_points,
                    surface_normals,
                    distances,
                    relative_speeds,
                    contact_confidence,
                ):
                    values[frame_idx] = values[anchor_frame]
                contact_modes[frame_idx] = int(ContactMode.SUPPORT)
                segment_ids[frame_idx] = segment_ids[anchor_frame]
                transition_weights[frame_idx] = 1.0
                for field_name, values in optional.items():
                    if values is None:
                        continue
                    values[frame_idx] = values[anchor_frame]
                    if field_name in (
                        "contact_bridge_alpha",
                        "orientation_bridge_alpha",
                    ):
                        values[frame_idx] = 1.0
            result[name] = replace(
                task,
                contact_positions=contact_positions,
                surface_points=surface_points,
                surface_normals=surface_normals,
                distances=distances,
                relative_speeds=relative_speeds,
                contact_confidence=contact_confidence,
                contact_modes=contact_modes,
                contact_segment_ids=segment_ids,
                transition_weights=transition_weights,
                **optional,
            )
        return result

    def diagnostics(self) -> List[Dict[str, Any]]:
        diagnostics: List[Dict[str, Any]] = []
        for event in self.events:
            if event.flight_release_frame is not None:
                actual_release = event.flight_release_frame
            elif event.established_frame is None:
                actual_release = None
            else:
                actual_release = max(
                    event.requested_release, event.established_frame + 1
                )
            diagnostics.append(
                {
                    "type": event.kind,
                    "old_supports": event.old_supports,
                    "new_supports": event.new_supports,
                    "requested_release_frame": event.requested_release,
                    "hold_frames": event.hold_frames,
                    "established_frame": event.established_frame,
                    "actual_release_frame": actual_release,
                    "held_extra_frames": (
                        None
                        if actual_release is None
                        else max(actual_release - event.requested_release, 0)
                    ),
                    "flight_override": event.flight_release_frame is not None,
                }
            )
        return diagnostics


class OmniRetargeter:
    """
    Generic motion retargeting for any humanoid URDF and terrain mesh.

    This class provides functionality to retarget SMPLX trajectories to any humanoid robot
    operating on any terrain mesh by automatically scaling the terrain and computing
    appropriate joint mappings.
    """

    def __init__(
        self,
        robot_urdf_path: Union[str, Path],
        terrain_mesh_path: Union[str, Path],
        joint_mapping: Dict[str, str],
        robot_height: Optional[float] = None,
        smplx_joint_names: Optional[List[str]] = None,
        height_estimation: Optional[Dict[str, Any]] = None,
        base_orientation: Optional[Dict[str, str]] = None,
        retargeting: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the OmniRetargeter.

        Args:
            robot_urdf_path: Path to the humanoid robot URDF file
            terrain_mesh_path: Path to the terrain mesh file (any common format)
            joint_mapping: Dictionary mapping SMPLX joint names to robot link names
            robot_height: Height of the robot in meters (auto-detected if None)
            smplx_joint_names: List of SMPLX joint names in order (required for proper joint mapping)
            height_estimation: Optional settings for human height estimation from SMPLX joints
            base_orientation: Optional joint names used for base orientation estimation
            retargeting: Optional solver/retargeting settings forwarded to interaction retargeter
        """
        self.robot_urdf_path = Path(robot_urdf_path)
        self.terrain_mesh_path = Path(terrain_mesh_path)
        self.joint_mapping = joint_mapping

        # SMPLX joint names (default to standard SMPLX joint ordering)
        if smplx_joint_names is None:
            # Standard SMPLX joint names (first 22 body joints)
            self.smplx_joint_names = [
                "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
                "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
                "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
                "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist"
            ]
        else:
            self.smplx_joint_names = smplx_joint_names

        # Optional per-robot configuration with safe defaults.
        self.height_estimation_config = dict(height_estimation or {})
        self.base_orientation_config = dict(base_orientation or {})
        self.retargeting_config = dict(retargeting or {})

        # Create mapping from SMPLX joint names to indices
        self.smplx_joint_indices = {}
        for idx, name in enumerate(self.smplx_joint_names):
            self.smplx_joint_indices[name] = idx

        # Get indices for mapped joints
        # CRITICAL: Only include joints that exist in SMPLX AND will be validated in robot
        # Store the valid joint names to ensure consistent ordering
        self.valid_joint_names = []  # Joint names that exist in both SMPLX and robot
        self.mapped_joint_indices = []
        
        for smplx_joint_name in joint_mapping.keys():
            if smplx_joint_name in self.smplx_joint_indices:
                self.mapped_joint_indices.append(self.smplx_joint_indices[smplx_joint_name])
                self.valid_joint_names.append(smplx_joint_name)
            else:
                raise ValueError(
                    f"SMPLX joint '{smplx_joint_name}' not found in SMPLX joint names. "
                    f"Available joints: {list(self.smplx_joint_indices.keys())[:10]}..."
                )

        if len(self.mapped_joint_indices) == 0:
            raise ValueError("No valid joint mappings found. Please check your joint_mapping dictionary.")
        
        # Store the filtered joint mapping (only valid joints) for use in retargeter
        self.valid_joint_mapping = {name: joint_mapping[name] for name in self.valid_joint_names}

        # Load robot model.  URDF files are also parsed by yourdfpy for
        # compatibility with older callers; MJCF files are consumed directly by
        # MuJoCo and do not need a URDF scene graph.
        self.robot_urdf = None
        if self.robot_urdf_path.suffix.lower() == ".urdf":
            if yourdfpy is None:
                raise ImportError("yourdfpy is required when loading URDF robot models.")
            self.robot_urdf = yourdfpy.URDF.load(str(robot_urdf_path), load_meshes=True)
        self.robot_model = mujoco.MjModel.from_xml_path(str(robot_urdf_path))
        self.robot_data = mujoco.MjData(self.robot_model)

        # Load terrain mesh
        self.terrain_mesh = trimesh.load(str(terrain_mesh_path))

        # Detect robot height if not provided
        if robot_height is None:
            self.robot_height = self._detect_robot_height()
        else:
            self.robot_height = robot_height

        # Initialize retargeting components
        self._setup_retargeting_components()

    def _detect_robot_height(self) -> float:
        """
        Detect robot height from URDF by calculating the vertical span of the robot in its default configuration.
        
        Since this is a floating-base robot, we assume the default configuration puts the robot
        in a nominal pose (e.g. standing). We calculate the difference between the highest
        and lowest points of all visual meshes to get the full height.
        """
        # Use MuJoCo to get robot height in default configuration
        # This is more reliable than parsing the URDF scene graph
        
        # Set robot to default configuration (zeros)
        mujoco.mj_resetData(self.robot_model, self.robot_data)
        self.robot_data.qpos[3:7] = [1, 0, 0, 0]  # Set base quaternion to identity
        mujoco.mj_forward(self.robot_model, self.robot_data)
        
        min_z = float('inf')
        max_z = float('-inf')
        
        # Iterate through all bodies and get their positions
        for body_idx in range(self.robot_model.nbody):
            body_pos = self.robot_data.xpos[body_idx]
            z = body_pos[2]
            min_z = min(min_z, z)
            max_z = max(max_z, z)
        
        # Also check geometry positions for more accuracy
        for geom_idx in range(self.robot_model.ngeom):
            geom_pos = self.robot_data.geom_xpos[geom_idx]
            z = geom_pos[2]
            
            # Get geometry size to account for extent
            geom_size = self.robot_model.geom_size[geom_idx]
            geom_type = self.robot_model.geom_type[geom_idx]
            
            # Estimate vertical extent based on geometry type
            if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                radius = geom_size[0]
                min_z = min(min_z, z - radius)
                max_z = max(max_z, z + radius)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                radius = geom_size[0]
                half_height = geom_size[1]
                min_z = min(min_z, z - half_height - radius)
                max_z = max(max_z, z + half_height + radius)
            elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                half_size = geom_size[2]  # Z dimension
                min_z = min(min_z, z - half_size)
                max_z = max(max_z, z + half_size)
            else:
                # For other types, just use the position
                min_z = min(min_z, z)
                max_z = max(max_z, z)
        
        height = max_z - min_z
        
        # Sanity check
        if height < 0.3 or height > 3.0:
            print(f"Warning: Detected height {height:.3f}m seems unreasonable. Using default 1.6m")
            return 1.6
        
        print(f"Detected robot height: {height:.4f}m (Min Z: {min_z:.4f}, Max Z: {max_z:.4f})")
        
        return height

    def _setup_retargeting_components(self):
        """Setup internal retargeting components."""
        # Validate joint mapping and filter out invalid joints
        # CRITICAL: Only keep joints that exist in BOTH SMPLX and robot URDF
        # This ensures consistent sizes between human_joints and robot_points
        
        # First, filter out joints missing from robot URDF
        missing_robot_links = self.validate_joint_mapping()
        if missing_robot_links:
            print(f"Warning: The following robot links from joint_mapping were not found in URDF: {missing_robot_links}")
            print("These joints will be removed from the mapping.")
            print(f"\nAvailable robot bodies:")
            for i in range(min(20, self.robot_model.nbody)):
                body_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, i)
                print(f"  {body_name}")
            if self.robot_model.nbody > 20:
                print(f"  ... and {self.robot_model.nbody - 20} more")
            
            # Remove joints with missing robot links from valid_joint_names
            self.valid_joint_names = [
                name for name in self.valid_joint_names 
                if self.valid_joint_mapping[name] not in missing_robot_links
            ]
            # Rebuild mapped_joint_indices and valid_joint_mapping
            self.mapped_joint_indices = [
                self.smplx_joint_indices[name] for name in self.valid_joint_names
            ]
            self.valid_joint_mapping = {name: self.joint_mapping[name] for name in self.valid_joint_names}
        
        if len(self.valid_joint_names) == 0:
            raise ValueError(
                "No valid joint mappings found after filtering. "
                "Please check that joint_mapping contains joints that exist in both SMPLX and robot URDF."
            )
        
        print(f"Successfully initialized retargeting with {len(self.valid_joint_names)} mapped joints.")
        print(f"Valid joints: {self.valid_joint_names}")
        print(f"Robot height: {self.robot_height:.3f}m")
        print(f"Robot DOF: {self.get_robot_dof()}")

    def _end_effector_reach_profiles(
        self,
        end_effector_cfg: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Measure per-chain reach and shoulder/hip origin from robot morphology."""
        specs = resolve_end_effector_chain_specs(
            self.robot_model,
            end_effector_cfg,
            self.valid_joint_mapping,
        )
        if not specs:
            return {}
        morphology = analyze_robot_morphology(
            self.robot_model,
            {"chains": specs},
            configured_joint_limits=end_effector_cfg.get("joint_posture_limits"),
            reach_samples=int(end_effector_cfg.get("morphology_reach_samples", 512)),
        )
        profiles: Dict[str, Dict[str, Any]] = {}
        for name, spec in specs.items():
            chain = morphology.chains[name]
            max_reach = (
                chain.actual_reach
                if spec.configured_reach is None
                else float(spec.configured_reach)
            )
            profiles[name] = {
                "origin_offset": chain.origin_root_local.copy(),
                # Explicit profile values remain backwards-compatible
                # overrides.  The former qpos0 proxy is replaced by sampled
                # joint-limit-aware reach whenever no override is present.
                "max_reach": max_reach,
                "measured_reach": chain.actual_reach,
                "reference_body": spec.reference_body,
                "reference_offset": np.asarray(spec.reference_offset, dtype=float),
                "nominal_length": chain.nominal_length,
                "qpos0_reach": chain.qpos0_reach,
            }
        return profiles

    def _automatic_morphology_scale(
        self,
        name: str,
        spec: Dict[str, Any],
        task: SceneContactTask,
        source_joints: np.ndarray,
        robot_reach: float,
    ) -> float:
        configured = spec.get(
            "swing_morphology_scale",
            self.retargeting_config.get("end_effector_priority", {}).get(
                "swing_morphology_scale", 1.0
            ),
        )
        if not (isinstance(configured, str) and configured.lower() == "auto"):
            return _numeric_morphology_scale(configured)

        chain = _source_limb_chain(name, spec)
        if any(joint not in self.smplx_joint_indices for joint in chain):
            return 1.0
        points = [source_joints[:, self.smplx_joint_indices[joint]] for joint in chain]
        source_reach = np.linalg.norm(points[1] - points[0], axis=1)
        source_reach += np.linalg.norm(points[2] - points[1], axis=1)
        source_reach += np.linalg.norm(task.tracking_positions - points[2], axis=1)
        median_source_reach = float(np.median(source_reach))
        if median_source_reach <= 1e-8:
            return 1.0
        scale = float(robot_reach) / median_source_reach
        bounds = np.asarray(
            spec.get("swing_morphology_scale_bounds", [0.55, 1.15]),
            dtype=float,
        ).reshape(2)
        return float(np.clip(scale, bounds[0], bounds[1]))

    def _source_limb_anchor(
        self,
        name: str,
        spec: Mapping[str, Any],
        source_trajectory: np.ndarray,
    ) -> np.ndarray:
        """Return the source limb-base track that anchors a scaled swing target."""
        joint_name = str(
            spec.get("swing_anchor_source_joint", _source_limb_chain(name, spec)[0])
        )
        if joint_name not in self.smplx_joint_indices:
            raise ValueError(
                f"End-effector '{name}' uses swing_target_space "
                "'limb_morphology_scaled' but its limb-base joint "
                f"'{joint_name}' is not present in the source skeleton."
            )
        return source_trajectory[:, self.smplx_joint_indices[joint_name]]

    def _robot_limb_anchor(
        self,
        name: str,
        profile: Mapping[str, Any],
        robot_base_reference: np.ndarray,
        base_rotations: np.ndarray | None,
    ) -> np.ndarray:
        """Return the robot limb-base track that anchors a scaled swing target.

        The reach reference body (shoulder or hip) is rigidly attached to the
        floating base on the supported profiles, so transporting its qpos0
        base-frame offset by the base orientation reference reproduces its
        world position without running the solver first.
        """
        if profile.get("reference_body") is None:
            raise ValueError(
                f"End-effector '{name}' uses swing_target_space "
                "'limb_morphology_scaled' but no 'reach_reference_body' "
                "resolves on the robot; the limb base is undefined."
            )
        offset = np.asarray(profile["origin_offset"], dtype=float).reshape(3)
        if base_rotations is None:
            return robot_base_reference + offset[None, :]
        return robot_base_reference + np.einsum("tij,j->ti", base_rotations, offset)

    def _prepare_contact_aware_references(
        self,
        source_trajectory: np.ndarray,
        tasks: Dict[str, SceneContactTask],
        end_effector_cfg: Dict[str, Any],
        base_orientations: np.ndarray | None = None,
        scene_mesh: trimesh.Trimesh | None = None,
        framerate: float = 30.0,
    ) -> Dict[str, SceneContactTask]:
        """Attach explicit swing, contact-bridge, and robot-root references."""
        base_cfg = dict(end_effector_cfg.get("base_position_constraint", {}) or {})
        root_name = str(
            base_cfg.get(
                "source_joint",
                self.base_orientation_config.get("root_position", "Pelvis"),
            )
        )
        if root_name not in self.smplx_joint_indices:
            raise ValueError(
                "base_position_constraint.source_joint must name a loaded "
                f"source joint, got {root_name!r}."
            )
        root_idx = self.smplx_joint_indices[root_name]
        source_roots = source_trajectory[:, root_idx].copy()
        raw_landmarks = base_cfg.get("landmarks")
        landmark_specs: List[Tuple[str, str]] = []
        if raw_landmarks is not None:
            if not isinstance(raw_landmarks, list) or not raw_landmarks:
                raise ValueError(
                    "base_position_constraint.landmarks must be a non-empty list."
                )
            for raw_landmark in raw_landmarks:
                if not isinstance(raw_landmark, Mapping):
                    raise ValueError(
                        "Each base_position_constraint.landmarks entry must "
                        "be an object."
                    )
                source_joint = str(raw_landmark.get("source_joint", ""))
                robot_body = str(raw_landmark.get("robot_body", ""))
                if source_joint not in self.smplx_joint_indices:
                    raise ValueError(
                        "base_position_constraint landmark source_joint must "
                        f"name a loaded source joint, got {source_joint!r}."
                    )
                if not robot_body:
                    raise ValueError(
                        "base_position_constraint landmark robot_body is required."
                    )
                landmark_specs.append((source_joint, robot_body))
        root_z_offset = float(base_cfg.get("z_offset", 0.0))
        root_reference_offset = np.asarray(
            base_cfg.get("offset", [0.0, 0.0, root_z_offset]), dtype=float
        ).reshape(3)
        reach_profiles = self._end_effector_reach_profiles(end_effector_cfg)
        root_warp_cfg = dict(end_effector_cfg.get("root_warp", {}) or {})
        raw_feasibility_cfg = end_effector_cfg.get(
            "interaction_feasibility", {}
        )
        if isinstance(raw_feasibility_cfg, Mapping):
            feasibility_cfg = dict(raw_feasibility_cfg)
            feasibility_enabled = bool(feasibility_cfg.get("enabled", False))
        else:
            feasibility_cfg = {}
            feasibility_enabled = bool(raw_feasibility_cfg)
        feasibility_options = (
            _feasibility_options_from_config(
                feasibility_cfg, root_warp_cfg, framerate
            )
            if feasibility_enabled
            else None
        )
        self.last_interaction_feasibility_result = None
        self.last_interaction_contact_regions = {}
        self.last_interaction_feasibility_export_payload = {}
        source_base_quaternions = _base_quaternions_wxyz(
            base_orientations, len(source_trajectory)
        )
        align_robot_body = base_cfg.get("robot_body")
        needs_body_alignment = bool(
            align_robot_body is not None or landmark_specs
        )
        if (
            source_base_quaternions is None
            and (
                needs_body_alignment
                or
                (
                    bool(root_warp_cfg.get("enabled", False))
                    and bool(root_warp_cfg.get("yaw_enabled", False))
                )
                or (
                    feasibility_options is not None
                    and feasibility_options.max_root_yaw > 0.0
                )
            )
        ):
            # Position-only adapters still have a source-neutral body-frame
            # fallback.  Estimate it here so yaw warp is not silently disabled
            # merely because captured root rotations were unavailable.
            estimated_quaternions: List[np.ndarray] = []
            previous_xyzw = None
            for source_joints in source_trajectory:
                xyzw = self._estimate_base_orientation_from_joints(
                    source_joints, previous_xyzw
                )
                if xyzw is None:
                    estimated_quaternions = []
                    break
                estimated_quaternions.append(
                    np.asarray(xyzw, dtype=float)[[3, 0, 1, 2]]
                )
                previous_xyzw = xyzw
            if estimated_quaternions:
                source_base_quaternions = np.asarray(
                    estimated_quaternions, dtype=float
                )
        if needs_body_alignment:
            if source_base_quaternions is None:
                raise ValueError(
                    "base-position body alignment requires a source "
                    "base orientation or a valid joint-position body frame."
                )
            if landmark_specs:
                aligned_roots = [
                    _root_positions_aligning_robot_body(
                        self.robot_model,
                        source_trajectory[
                            :, self.smplx_joint_indices[source_joint]
                        ],
                        robot_body,
                        source_base_quaternions,
                    )
                    for source_joint, robot_body in landmark_specs
                ]
                source_roots = np.mean(aligned_roots, axis=0)
            else:
                source_roots = _root_positions_aligning_robot_body(
                    self.robot_model,
                    source_roots,
                    str(align_robot_body),
                    source_base_quaternions,
                )

        feasibility_result: InteractionFeasibilityResult | None = None
        feasibility_corrections: Dict[
            Tuple[str, int], Mapping[str, np.ndarray]
        ] | None = None
        if feasibility_enabled:
            if scene_mesh is None:
                raise ValueError(
                    "interaction_feasibility requires the original scene mesh "
                    "to construct source-conditioned contact regions."
                )
            regions = _build_interaction_contact_regions(
                tasks, scene_mesh, feasibility_cfg
            )
            self.last_interaction_contact_regions = regions
            if regions:
                chain_specs = resolve_end_effector_chain_specs(
                    self.robot_model,
                    end_effector_cfg,
                    self.valid_joint_mapping,
                )
                morphology = analyze_robot_morphology(
                    self.robot_model,
                    {"chains": chain_specs},
                    configured_joint_limits=end_effector_cfg.get(
                        "joint_posture_limits"
                    ),
                    reach_samples=int(
                        end_effector_cfg.get("morphology_reach_samples", 512)
                    ),
                )
                unwarped_qpos = np.repeat(
                    np.asarray(self.robot_model.qpos0, dtype=float)[None, :],
                    len(source_trajectory),
                    axis=0,
                )
                if unwarped_qpos.shape[1] < 7:
                    raise ValueError(
                        "interaction_feasibility requires a floating-base robot."
                    )
                unwarped_qpos[:, :3] = (
                    source_roots + root_reference_offset[None, :]
                )
                if source_base_quaternions is not None:
                    unwarped_qpos[:, 3:7] = source_base_quaternions
                assert feasibility_options is not None
                feasibility_result = solve_interaction_feasibility(
                    self.robot_model,
                    morphology,
                    tasks,
                    regions,
                    unwarped_qpos,
                    required_contacts=feasibility_cfg.get(
                        "required_contacts", True
                    ),
                    configured_joint_limits=end_effector_cfg.get(
                        "joint_posture_limits"
                    ),
                    options=feasibility_options,
                )
                self.last_interaction_feasibility_result = feasibility_result
                self.last_interaction_feasibility_export_payload = (
                    feasibility_result.to_npz_dict()
                )
                failures = [
                    segment
                    for segment in feasibility_result.segments
                    if segment.required
                    and segment.reachability_status == INFEASIBLE
                ]
                if failures:
                    details = "; ".join(
                        f"{segment.effector_name}[{segment.segment_id}] "
                        f"slack={segment.max_contact_slack:.6g}m: "
                        f"{segment.infeasible_reason}"
                        for segment in failures
                    )
                    message = (
                        "Required interaction feasibility failed; framewise "
                        f"retargeting was not accepted as contact success: {details}"
                    )
                    failure_policy = str(
                        feasibility_cfg.get("infeasible_policy", "raise")
                    ).strip().lower()
                    if failure_policy == "raise":
                        raise RuntimeError(message)
                    if failure_policy != "flag":
                        raise ValueError(
                            "interaction_feasibility.infeasible_policy must be "
                            "'raise' or 'flag'."
                        )
                    print(f"WARNING: {message}")
                tasks = _apply_feasibility_contact_choices(
                    tasks, feasibility_result
                )
                feasibility_corrections = {
                    (segment.effector_name, segment.segment_id): {
                        "translation": segment.root_correction_translation,
                        "yaw": segment.root_correction_yaw,
                    }
                    for segment in feasibility_result.segments
                    if segment.required
                    and segment.reachability_status != DROPPED_OPTIONAL
                }
            else:
                # An enabled prepass with no active contact run still owns the
                # zero correction.  Passing an empty mapping prevents the
                # legacy reach heuristic from becoming a second authority.
                feasibility_corrections = {}
            assert feasibility_options is not None
            root_warp_cfg = _coordinated_root_warp_config(
                root_warp_cfg, feasibility_options
            )
        (
            robot_base_reference,
            unwarped,
            root_yaw_correction,
            root_warp_diagnostics,
        ) = _build_robot_base_reference(
            source_roots,
            tasks,
            reach_profiles,
            root_warp_cfg,
            root_reference_offset,
            base_orientations=source_base_quaternions,
            framerate=framerate,
            return_diagnostics=True,
            feasibility_segment_corrections=feasibility_corrections,
        )
        base_quaternions = source_base_quaternions
        if base_quaternions is not None and np.any(
            np.abs(root_yaw_correction) > 1e-12
        ):
            source_rotations = Rotation.from_quat(
                base_quaternions, scalar_first=True
            )
            yaw_rotations = Rotation.from_euler(
                "z", root_yaw_correction[:, None]
            )
            base_quaternions = (yaw_rotations * source_rotations).as_quat(
                scalar_first=True
            )
            for frame_idx in range(1, len(base_quaternions)):
                if (
                    np.dot(
                        base_quaternions[frame_idx - 1],
                        base_quaternions[frame_idx],
                    )
                    < 0.0
                ):
                    base_quaternions[frame_idx] *= -1.0
        base_rotations = (
            None
            if base_quaternions is None
            else Rotation.from_quat(
                base_quaternions, scalar_first=True
            ).as_matrix()
        )
        bridge_frames = max(
            int(
                end_effector_cfg.get(
                    "contact_bridge_frames",
                    (end_effector_cfg.get("contact_detection", {}) or {}).get(
                        "transition_frames", 0
                    ),
                )
            ),
            0,
        )
        orientation_bridge_frames = max(
            int(end_effector_cfg.get("orientation_bridge_frames", bridge_frames)),
            0,
        )
        specs = dict(end_effector_cfg.get("effectors", {}) or {})
        result: Dict[str, SceneContactTask] = {}
        scales: Dict[str, float] = {}
        scene_blends: Dict[str, np.ndarray] = {}
        for name, task in tasks.items():
            spec = dict(specs.get(name, {}) or {})
            profile = reach_profiles.get(
                name,
                {
                    "origin_offset": np.zeros(3),
                    "max_reach": np.inf,
                    "reference_body": None,
                    "reference_offset": np.zeros(3),
                },
            )
            target_space = _validate_swing_target_space(
                name, spec.get("swing_target_space", "world")
            )
            limb_offsets = None
            swing_anchor_body = None
            swing_anchor_offset = None
            robot_anchors = None
            if target_space == "world":
                scale = 1.0
                swing_targets = task.tracking_positions.copy()
            else:
                scale = self._automatic_morphology_scale(
                    name,
                    spec,
                    task,
                    source_trajectory,
                    float(profile["max_reach"]),
                )
                if target_space == "root_morphology_scaled":
                    source_anchors = source_roots
                    robot_anchors = robot_base_reference
                else:
                    # Anchoring at the limb base keeps the free-space limb
                    # direction identical to the source and only shortens the
                    # limb.  The root anchor cannot: the robot torso carries
                    # its shoulders far below the human ones, so a
                    # root-relative target lands outside the arm's reach shell
                    # and has to be clipped, which is what bends the free arm
                    # away from the captured pose.
                    source_anchors = self._source_limb_anchor(
                        name, spec, source_trajectory
                    )
                    robot_anchors = self._robot_limb_anchor(
                        name, profile, robot_base_reference, base_rotations
                    )
                scaled_offsets = scale * (
                    task.tracking_positions - source_anchors
                )
                if (
                    target_space == "limb_morphology_scaled"
                    and np.isfinite(float(profile["max_reach"]))
                ):
                    # The reach constraint belongs to the relative vector, not
                    # a reference-pose world point.  This keeps the same vector
                    # attached to the live shoulder throughout SQP.
                    scaled_offsets = np.stack(
                        [
                            _clamp_target_to_reach(
                                offset,
                                np.zeros(3, dtype=float),
                                profile["max_reach"],
                            )
                            for offset in scaled_offsets
                        ],
                        axis=0,
                    )
                    limb_offsets = scaled_offsets
                    swing_anchor_body = str(profile["reference_body"])
                    swing_anchor_offset = np.asarray(
                        profile.get("reference_offset", np.zeros(3)),
                        dtype=float,
                    ).reshape(3)
                swing_targets = robot_anchors + scaled_offsets
            scales[name] = scale
            # A world-space source target stays immutable even when unreachable;
            # the resulting residual is the honest morphology error.  Reach
            # projection is reserved for the explicit morphology-scaled modes.
            if (
                target_space in MORPHOLOGY_SWING_TARGET_SPACES
                and np.isfinite(float(profile["max_reach"]))
            ):
                if target_space != "limb_morphology_scaled":
                    reach_origins = robot_base_reference + np.asarray(
                        profile["origin_offset"], dtype=float
                    )[None, :]
                    swing_targets = np.stack(
                        [
                            _clamp_target_to_reach(
                                target, origin, profile["max_reach"]
                            )
                            for target, origin in zip(
                                swing_targets, reach_origins
                            )
                        ],
                        axis=0,
                    )
                swing_targets, scene_blend = _resolve_swing_targets_against_scene(
                    swing_targets,
                    task.tracking_positions,
                    scene_mesh,
                    float(
                        (
                            end_effector_cfg.get("contact_detection", {}) or {}
                        ).get("surface_clearance", 0.005)
                    ),
                    free_mask=np.asarray(task.contact_modes, dtype=np.uint8)
                    == int(ContactMode.NONE),
                )
                faded = int(np.count_nonzero(scene_blend > 1e-6))
                if faded:
                    print(
                        f"  {name}: {faded}/{len(scene_blend)} scaled swing "
                        "targets intersected the scene and were faded toward "
                        "their world reference (max blend "
                        f"{scene_blend.max():.2f})."
                    )
                scene_blends[name] = scene_blend
            else:
                scene_blend = np.zeros(len(task.tracking_positions), dtype=float)
            effective_targets, bridge_alpha = _bridge_swing_to_contact(
                swing_targets,
                task.contact_positions,
                task.contact_modes,
                task.contact_segment_ids,
                bridge_frames,
            )
            if limb_offsets is not None:
                # Scene fade is another transition from a relative target to a
                # world target.  Fold it into the same affine representation
                # without modifying the source-derived limb vector.
                bridge_alpha = 1.0 - (
                    (1.0 - bridge_alpha) * (1.0 - scene_blend)
                )
                relative_preview = robot_anchors + limb_offsets
            else:
                relative_preview = swing_targets
            orientation_rotations, orientation_alpha = (
                _bridge_contact_orientation(
                    task.forward_directions,
                    task.up_directions,
                    task.surface_normals,
                    task.contact_modes,
                    task.contact_segment_ids,
                    bridge_alpha,
                    orientation_bridge_frames,
                )
            )
            position_sigma_scale, orientation_sigma_scale = (
                _build_per_frame_sigma_scales(
                    task.contact_modes,
                    task.contact_segment_ids,
                    bridge_alpha,
                    end_effector_cfg,
                )
            )
            bridge_positions = task.contact_positions.copy()
            bridged = bridge_alpha > 1e-10
            bridge_positions[bridged] = (
                effective_targets[bridged]
                - (1.0 - bridge_alpha[bridged, None])
                * relative_preview[bridged]
            ) / bridge_alpha[bridged, None]
            result[name] = replace(
                task,
                effective_swing_positions=effective_targets,
                swing_limb_offsets=limb_offsets,
                swing_anchor_body=swing_anchor_body,
                swing_anchor_offset=swing_anchor_offset,
                contact_bridge_alpha=bridge_alpha,
                contact_bridge_positions=bridge_positions,
                orientation_target_rotations=orientation_rotations,
                orientation_bridge_alpha=orientation_alpha,
                position_sigma_scale=position_sigma_scale,
                orientation_sigma_scale=orientation_sigma_scale,
                morphology_scale=scale,
                robot_base_reference_w=robot_base_reference,
            )
        self.last_robot_base_reference_w = robot_base_reference
        self.last_unwarped_robot_base_reference_w = unwarped
        self.last_robot_base_yaw_correction = root_yaw_correction
        self.last_robot_base_orientation_reference_w = (
            None if base_quaternions is None else base_quaternions.copy()
        )
        self.last_unwarped_robot_base_orientation_reference_w = (
            None
            if source_base_quaternions is None
            else source_base_quaternions.copy()
        )
        self.last_root_warp_diagnostics = root_warp_diagnostics
        self.last_morphology_scales = scales
        self.last_swing_scene_blends = scene_blends
        return result

    def retarget_motion(
        self,
        smplx_trajectory: np.ndarray,
        base_orientations: np.ndarray | None = None,
        base_translations: np.ndarray | None = None,
        framerate: float | None = None,
        visualize_trajectory: bool = True,
        enable_terrain_scaling: bool = False,
        end_effector_markers: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
    ) -> Tuple[float, np.ndarray]:
        """
        Retarget SMPLX motion to the robot on the terrain.

        Args:
            smplx_trajectory: SMPLX joint positions of shape (T, J, 3) where T is frames, J is joints
            framerate: Motion framerate used by optional post-processing.
            enable_terrain_scaling: If True, estimate a terrain scale factor from the
                SMPLX trajectory and retarget against a scaled terrain mesh. If False,
                keep the original terrain mesh and use a scale factor of 1.0.
            end_effector_markers: Optional SMPL-X palm/sole centers and body
                surface samples, as returned by ``load_smplx_trajectory``.

        Returns:
            Tuple of (terrain_scale, retargeted_trajectory)
            - terrain_scale: Scalar factor to scale the terrain mesh
            - retargeted_trajectory: Robot motion of shape (T, 7 + DOF) with [pos, quat, joints]
        """
        # Step 1: Compute terrain scaling factor
        if enable_terrain_scaling:
            terrain_scale = self._compute_terrain_scale(smplx_trajectory)
            scaled_terrain = self._scale_terrain_mesh(terrain_scale)
        else:
            terrain_scale = 1.0
            scaled_terrain = self.terrain_mesh.copy()

        # Step 3: Process SMPLX trajectory
        processed_trajectory = self._process_smplx_trajectory(smplx_trajectory, terrain_scale)
        processed_base_translations = None
        if base_translations is not None:
            processed_base_translations = base_translations * terrain_scale

        retargeting_config = getattr(self, "retargeting_config", {})
        end_effector_cfg = dict(
            retargeting_config.get("end_effector_priority", {}) or {}
        )
        end_effector_tasks: Dict[str, SceneContactTask] = {}
        if bool(end_effector_cfg.get("enabled", False)) and end_effector_markers:
            processed_markers = {
                name: {
                    key: np.asarray(value, dtype=float) * terrain_scale
                    for key, value in marker.items()
                }
                for name, marker in end_effector_markers.items()
            }
            end_effector_tasks = infer_scene_contact_tasks(
                processed_markers,
                scaled_terrain,
                framerate=float(framerate or 30.0),
                config=end_effector_cfg.get("contact_detection", {}),
            )
            contact_summary = ", ".join(
                f"{name}="
                + "/".join(
                    f"{mode.name}:{int(np.count_nonzero(task.contact_modes == int(mode)))}"
                    for mode in (ContactMode.TOUCH, ContactMode.SLIDE, ContactMode.SUPPORT)
                )
                + f"/{len(task.contact_modes)}f"
                for name, task in end_effector_tasks.items()
            )
            print(f"Inferred SMPL-X scene contacts (frames): {contact_summary}")
            end_effector_tasks = self._prepare_contact_aware_references(
                processed_trajectory,
                end_effector_tasks,
                end_effector_cfg,
                base_orientations=base_orientations,
                scene_mesh=scaled_terrain,
                framerate=float(framerate or 30.0),
            )
        elif bool(end_effector_cfg.get("enabled", False)):
            print(
                "Contact-priority optimization requested, but no SMPL-X "
                "end-effector markers were supplied; using whole-body retargeting."
            )
        self.last_end_effector_tasks = end_effector_tasks

        # Option Step:
        if visualize_trajectory:
            # visualize the processed_trajectory
            self._visualize_trajectory(processed_trajectory, scaled_terrain)

        # Step 4: Perform retargeting using the generic retargeting system
        solver_base_orientations = base_orientations
        if end_effector_tasks:
            solver_base_orientations = getattr(
                self,
                "last_robot_base_orientation_reference_w",
                base_orientations,
            )
        retargeting_kwargs = dict(
            base_orientations=solver_base_orientations,
            base_translations=processed_base_translations,
        )
        if end_effector_tasks:
            retargeting_kwargs["end_effector_tasks"] = end_effector_tasks
        retargeted_motion = self._perform_retargeting(
            processed_trajectory, scaled_terrain, **retargeting_kwargs
        )
        if end_effector_tasks:
            # Runtime handoff holds are finalized by the framewise pass and
            # must be the same schedule consumed by refinement and evaluation.
            end_effector_tasks = getattr(
                self, "last_end_effector_tasks", end_effector_tasks
            )
        # Keep the framewise result immutable so final acceptance metrics can
        # reveal how much refinement and any later smoothing changed the motion.
        pre_refinement_motion = np.asarray(retargeted_motion, dtype=float).copy()
        self.last_pre_refinement_motion = pre_refinement_motion
        refinement_cfg = dict(
            retargeting_config.get("trajectory_refinement", {}) or {}
        )
        penetration_cfg = dict(refinement_cfg.get("penetration", {}) or {})
        if bool(refinement_cfg.get("enabled", False)) and end_effector_tasks:
            report_penetration = bool(penetration_cfg.get("report", True))
            before = (
                measure_scene_penetration(
                    self.robot_model,
                    retargeted_motion,
                    scaled_terrain,
                    penetration_cfg,
                )
                if report_penetration
                else None
            )
            retargeted_motion = refine_joint_trajectory(
                self.robot_model,
                retargeted_motion,
                end_effector_tasks,
                end_effector_cfg,
                refinement_cfg,
                scene_mesh=scaled_terrain,
                framerate=float(framerate or 30.0),
                source_root_positions=processed_trajectory[
                    :,
                    self.smplx_joint_indices.get(
                        self.base_orientation_config.get(
                            "root_position", "Spine3"
                        ),
                        0,
                    ),
                ],
                source_joint_positions=processed_trajectory,
                source_joint_names=self.smplx_joint_names,
                active_contact_offset_ids={
                    name: task.active_contact_offset_ids
                    for name, task in end_effector_tasks.items()
                    if task.active_contact_offset_ids is not None
                },
                joint_mapping=self.valid_joint_mapping,
            )
            if before is not None:
                after = measure_scene_penetration(
                    self.robot_model,
                    retargeted_motion,
                    scaled_terrain,
                    penetration_cfg,
                )
                # Jerk is part of the refinement objective, so its improvement
                # is not evidence of a good result.  Penetration is only ever
                # penalized, which makes it the honest acceptance check.
                for label, stats in (("before", before), ("after", after)):
                    print(
                        f"Scene penetration {label} refinement: "
                        f"max={stats['max_depth'] * 100.0:.2f} cm, "
                        f"mean={stats['mean_depth'] * 100.0:.2f} cm, "
                        f"frames={stats['frames_penetrating']}/{stats['num_frames']}"
                        + (
                            f", worst={stats['worst_body']}@{stats['worst_frame']}"
                            if stats["worst_body"]
                            else ""
                        )
                    )
                deeper = after["max_depth"] > before["max_depth"] + 1e-4
                spread_improved = (
                    after["frames_penetrating"] < before["frames_penetrating"]
                )
                if deeper and not spread_improved:
                    print(
                        "Scene penetration warning: refinement increased the "
                        "worst penetration depth without clearing frames. "
                        "Tighten trajectory_refinement.penetration.sigma or "
                        "relax acceleration_sigma/jerk_sigma."
                    )
                elif deeper:
                    # The worst point is usually an end effector held on its
                    # contact anchor, so its depth is a floor the joint-space
                    # correction cannot go below.  Once refinement has pushed
                    # every other point down to that floor, a fractional change
                    # at the floor itself is not a regression -- the honest
                    # signal is that far fewer frames penetrate at all.
                    print(
                        "Scene penetration note: the worst point deepened by "
                        f"{(after['max_depth'] - before['max_depth']) * 100.0:.2f} cm "
                        f"while penetrating frames fell from "
                        f"{before['frames_penetrating']} to "
                        f"{after['frames_penetrating']}. "
                        f"{after['worst_body']}@{after['worst_frame']} is pinned "
                        "by its contact anchor; loosen that effector's contact "
                        "task if the residual depth matters."
                    )
        retargeted_motion = self._apply_trajectory_smoothing(
            retargeted_motion,
            framerate=float(framerate or 30.0),
        )

        foot_stabilization_cfg = dict(retargeting_config.get("foot_stabilization", {}) or {})
        if end_effector_tasks and bool(foot_stabilization_cfg.get("enabled", False)):
            print(
                "Foot stabilization skipped: contact-priority retargeting already "
                "anchors the feet to the scene, and the post-hoc ground lock would "
                "move the base away from those anchors."
            )
        if not end_effector_tasks and (
            retargeting_config.get("penetration_resolver", "hard_constraint") == "xyz_nudge"
            or bool(foot_stabilization_cfg.get("enabled", False))
        ):
            retargeted_motion = self._apply_foot_stabilization(
                retargeted_motion,
                scaled_terrain,
                framerate=framerate,
            )

        # Evaluate only after every post-processing stage.  This is the exact
        # trajectory returned to the caller and catches collision/contact
        # regressions introduced by either smoothing or legacy foot locking.
        if hasattr(self, "robot_model"):
            root_idx = self.smplx_joint_indices.get(
                self.base_orientation_config.get("root_position", "Spine3"), 0
            )
            self.last_evaluation_metrics = evaluate_retargeted_motion(
                self.robot_model,
                retargeted_motion,
                end_effector_tasks,
                end_effector_cfg,
                framerate=float(framerate or 30.0),
                reference_trajectory=pre_refinement_motion,
                source_root_positions=processed_trajectory[:, root_idx],
                source_joint_positions=processed_trajectory,
                source_joint_names=self.smplx_joint_names,
                scene_mesh=(
                    scaled_terrain
                    if bool(penetration_cfg.get("report", True))
                    else None
                ),
                penetration_config=penetration_cfg,
            )
        else:
            # Lightweight unit-test instances may intentionally omit a MuJoCo
            # model. Preserve the existing mocked retarget_motion contract.
            self.last_evaluation_metrics = {}

        if end_effector_tasks:
            self._report_final_motion_quality(retargeted_motion, end_effector_tasks)

        return terrain_scale, retargeted_motion

    def _apply_trajectory_smoothing(
        self,
        trajectory: np.ndarray,
        framerate: float,
    ) -> np.ndarray:
        """Remove zero-mean framewise IK chatter without moving the root.

        A symmetric Gaussian convolution is zero phase: it suppresses the
        high-frequency redundant-joint solution switching without introducing
        a time delay between the human and robot.  Its positive kernel also
        avoids ringing around contact transitions.  Only scalar robot joints
        are filtered.  Floating-base translation and orientation remain
        exactly as solved, so scene alignment and base position cannot drift.
        """
        cfg = dict(
            getattr(self, "retargeting_config", {}).get(
                "trajectory_smoothing", {}
            )
            or {}
        )
        if not bool(cfg.get("enabled", False)) or len(trajectory) < 8:
            return trajectory
        sigma_frames = float(cfg.get("sigma_frames", 1.0))
        if sigma_frames <= 0.0:
            raise ValueError(
                "trajectory_smoothing.sigma_frames must be positive, got "
                f"{sigma_frames:g}."
            )
        joint_qpos_indices = []
        for joint_id in range(self.robot_model.njnt):
            joint_type = self.robot_model.jnt_type[joint_id]
            if joint_type in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                qpos_index = int(self.robot_model.jnt_qposadr[joint_id])
                if qpos_index >= 7:
                    joint_qpos_indices.append(qpos_index)
        if not joint_qpos_indices:
            return trajectory

        result = np.asarray(trajectory, dtype=float).copy()
        result[:, joint_qpos_indices] = gaussian_filter1d(
            result[:, joint_qpos_indices],
            sigma=sigma_frames,
            axis=0,
            mode="nearest",
        )

        # The zero-phase filter can overshoot a saturated joint by a fraction
        # of a degree.  Restore the model's exact scalar joint limits.
        for joint_id in range(self.robot_model.njnt):
            if not self.robot_model.jnt_limited[joint_id]:
                continue
            qpos_index = int(self.robot_model.jnt_qposadr[joint_id])
            if qpos_index not in joint_qpos_indices:
                continue
            lower, upper = self.robot_model.jnt_range[joint_id]
            result[:, qpos_index] = np.clip(
                result[:, qpos_index], lower, upper
            )
        return result

    def _report_final_motion_quality(
        self,
        trajectory: np.ndarray,
        tasks: Dict[str, SceneContactTask],
    ) -> None:
        """Print acceptance metrics for the trajectory that is delivered.

        End-effector world-position errors, contact residuals, and temporal
        jitter are reported separately after every post-processing stage, so
        the numbers describe the motion actually written to disk.
        """
        end_effector_cfg = dict(
            self.retargeting_config.get("end_effector_priority", {}) or {}
        )
        specs = dict(end_effector_cfg.get("effectors", {}) or {})
        if not tasks or not specs or len(trajectory) == 0:
            return
        data = mujoco.MjData(self.robot_model)
        contact_errors: Dict[str, List[float]] = {name: [] for name in tasks}
        support_errors: Dict[str, List[float]] = {name: [] for name in tasks}
        swing_errors: Dict[str, List[float]] = {name: [] for name in tasks}
        swing_anchor_errors: Dict[str, List[float]] = {
            name: [] for name in tasks
        }
        source_swing_errors: Dict[str, List[float]] = {
            name: [] for name in tasks
        }
        source_contact_errors: Dict[str, List[float]] = {
            name: [] for name in tasks
        }
        normal_errors: Dict[str, List[float]] = {name: [] for name in tasks}
        orientation_errors: Dict[str, List[float]] = {name: [] for name in tasks}
        support_markers: Dict[str, Dict[int, List[np.ndarray]]] = {
            name: {} for name in tasks
        }
        for frame_idx, qpos in enumerate(np.asarray(trajectory, dtype=float)):
            data.qpos[:] = qpos
            mujoco.mj_forward(self.robot_model, data)
            for name, task in tasks.items():
                spec = dict(specs.get(name) or {})
                body_name = spec.get("robot_body")
                if not body_name:
                    continue
                body_id = mujoco.mj_name2id(
                    self.robot_model, mujoco.mjtObj.mjOBJ_BODY, str(body_name)
                )
                if body_id < 0:
                    continue
                rotation = data.xmat[body_id].reshape(3, 3)
                tracking_offset = np.asarray(
                    spec.get("tracking_offset", [0.0, 0.0, 0.0]), dtype=float
                ).reshape(3)
                mode = ContactMode(int(task.contact_modes[frame_idx]))
                tracking_marker = (
                    data.xpos[body_id] + rotation @ tracking_offset
                )
                live_swing_target = None
                if task.swing_limb_offsets is not None:
                    anchor_body_id = mujoco.mj_name2id(
                        self.robot_model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        str(task.swing_anchor_body),
                    )
                    anchor_rotation = data.xmat[anchor_body_id].reshape(3, 3)
                    anchor_offset = (
                        np.zeros(3, dtype=float)
                        if task.swing_anchor_offset is None
                        else task.swing_anchor_offset
                    )
                    live_anchor = (
                        data.xpos[anchor_body_id]
                        + anchor_rotation @ anchor_offset
                    )
                    limb_target = (
                        live_anchor + task.swing_limb_offsets[frame_idx]
                    )
                    alpha = float(task.contact_bridge_alpha[frame_idx])
                    live_swing_target = (
                        (1.0 - alpha) * limb_target
                        + alpha * task.contact_bridge_positions[frame_idx]
                    )
                    swing_anchor_errors[name].append(
                        float(
                            np.linalg.norm(
                                (limb_target - task.swing_limb_offsets[frame_idx])
                                - live_anchor
                            )
                        )
                    )
                source_error = float(
                    np.linalg.norm(
                        tracking_marker - task.tracking_positions[frame_idx]
                    )
                )
                if mode == ContactMode.NONE:
                    source_swing_errors[name].append(source_error)
                else:
                    source_contact_errors[name].append(source_error)
                if mode != ContactMode.NONE:
                    contact_offsets = np.asarray(
                        spec.get("contact_offsets", [tracking_offset.tolist()]),
                        dtype=float,
                    )
                    if contact_offsets.ndim == 1:
                        contact_offsets = contact_offsets[None, :]
                    offset_id = (
                        -1
                        if task.active_contact_offset_ids is None
                        else int(task.active_contact_offset_ids[frame_idx])
                    )
                    if offset_id < 0 or offset_id >= len(contact_offsets):
                        offset_id = 0
                    marker = data.xpos[body_id] + rotation @ contact_offsets[offset_id]
                    error = float(
                        np.linalg.norm(marker - task.contact_positions[frame_idx])
                    )
                    contact_errors[name].append(error)
                    if mode == ContactMode.SUPPORT:
                        support_errors[name].append(error)
                        segment_id = int(task.contact_segment_ids[frame_idx])
                        support_markers[name].setdefault(segment_id, []).append(
                            marker.copy()
                        )

                    local_rotation = _local_semantic_rotation(
                        spec.get("orientation_local_forward", [1.0, 0.0, 0.0]),
                        spec.get("orientation_local_up", [0.0, 0.0, 1.0]),
                    )
                    semantic_rotation = rotation @ local_rotation
                    normal = np.asarray(task.surface_normals[frame_idx], dtype=float)
                    normal /= max(float(np.linalg.norm(normal)), 1e-10)
                    normal_errors[name].append(
                        float(
                            np.arccos(
                                np.clip(
                                    float(np.dot(semantic_rotation[:, 2], normal)),
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                    )
                    target_up = (
                        normal
                        if mode in (ContactMode.SUPPORT, ContactMode.SLIDE)
                        else (
                            None
                            if task.up_directions is None
                            else task.up_directions[frame_idx]
                        )
                    )
                    if task.forward_directions is not None and target_up is not None:
                        target_up = np.asarray(target_up, dtype=float)
                        target_up /= max(float(np.linalg.norm(target_up)), 1e-10)
                        target_forward = np.asarray(
                            task.forward_directions[frame_idx], dtype=float
                        ).copy()
                        target_forward -= target_up * float(
                            np.dot(target_forward, target_up)
                        )
                        target_forward /= max(
                            float(np.linalg.norm(target_forward)), 1e-10
                        )
                        forward_error = np.arccos(
                            np.clip(
                                float(np.dot(semantic_rotation[:, 0], target_forward)),
                                -1.0,
                                1.0,
                            )
                        )
                        up_error = np.arccos(
                            np.clip(
                                float(np.dot(semantic_rotation[:, 2], target_up)),
                                -1.0,
                                1.0,
                            )
                        )
                        orientation_errors[name].append(float(max(forward_error, up_error)))
                else:
                    marker = tracking_marker
                    if task.effective_swing_positions is None:
                        raise RuntimeError(
                            f"{name}: effective swing targets were not prepared."
                        )
                    swing_target = (
                        task.effective_swing_positions[frame_idx]
                        if live_swing_target is None
                        else live_swing_target
                    )
                    swing_errors[name].append(
                        float(np.linalg.norm(marker - swing_target))
                    )
        print("Final motion quality (after all post-processing):")
        for name in tasks:
            parts = []
            for label, errors in (
                ("contact", contact_errors[name]),
                ("support", support_errors[name]),
                ("source_contact", source_contact_errors[name]),
                ("source_swing", source_swing_errors[name]),
                ("effective_swing", swing_errors[name]),
            ):
                if errors:
                    values = np.asarray(errors) * 100.0
                    parts.append(
                        f"{label}[{len(errors)}f] mean={values.mean():.2f} "
                        f"p95={np.percentile(values, 95):.2f} "
                        f"max={values.max():.2f} cm"
                    )
                else:
                    parts.append(f"{label}[0f]")
            slip_values = []
            for segment_id, markers in support_markers[name].items():
                points = np.asarray(markers, dtype=float)
                frames = np.flatnonzero(
                    tasks[name].contact_segment_ids == segment_id
                )
                normal = np.median(tasks[name].surface_normals[frames], axis=0)
                normal /= max(float(np.linalg.norm(normal)), 1e-10)
                delta = points - points[0]
                tangential = delta - np.outer(delta @ normal, normal)
                slip_values.append(float(np.max(np.linalg.norm(tangential, axis=1))))
            if slip_values:
                parts.append(f"support_slip_max={max(slip_values) * 100.0:.2f} cm")
            if swing_anchor_errors[name]:
                parts.append(
                    "live_anchor_err_max="
                    f"{max(swing_anchor_errors[name]) * 100.0:.6f} cm"
                )
            if normal_errors[name]:
                parts.append(
                    "normal_err="
                    f"{np.rad2deg(np.mean(normal_errors[name])):.1f}/"
                    f"{np.rad2deg(np.max(normal_errors[name])):.1f} deg mean/max"
                )
            if orientation_errors[name]:
                parts.append(
                    "palm/sole_err="
                    f"{np.rad2deg(np.mean(orientation_errors[name])):.1f}/"
                    f"{np.rad2deg(np.max(orientation_errors[name])):.1f} deg mean/max"
                )
            in_contact = (
                np.asarray(tasks[name].contact_modes, dtype=np.uint8)
                != int(ContactMode.NONE)
            )
            onset_frames = np.flatnonzero(
                in_contact & ~np.r_[False, in_contact[:-1]]
            )
            release_frames = np.flatnonzero(
                in_contact & ~np.r_[in_contact[1:], False]
            )
            parts.append(
                "timing_offset=0f "
                f"onset={onset_frames.tolist()} release={release_frames.tolist()}"
            )
            print(f"  {name}: " + " | ".join(parts))
        final_metrics = getattr(self, "last_evaluation_metrics", {})
        penetration = final_metrics.get("penetration")
        if penetration:
            print(
                "  final scene penetration: "
                f"mean={penetration['mean_depth'] * 100.0:.2f} cm, "
                f"p95={penetration.get('p95_depth', 0.0) * 100.0:.2f} cm, "
                f"max={penetration['max_depth'] * 100.0:.2f} cm, "
                f"frames={penetration['frames_penetrating']}/"
                f"{penetration['num_frames']}, "
                f"samples={penetration.get('total_penetrating_samples', 0)}, "
                f"bodies={penetration.get('penetrating_body_count', 0)}, "
                f"worst={penetration['worst_body']}@"
                f"{penetration['worst_frame']}"
            )
        deviation = final_metrics.get("deviation_from_pre_refinement")
        if deviation:
            joint_delta = deviation["joint_absolute_rad"]
            body_delta = deviation["whole_body_root_relative_m"]
            print(
                "  deviation from pre-refinement: "
                f"joint mean/p95/max={joint_delta['mean']:.4f}/"
                f"{joint_delta['p95']:.4f}/{joint_delta['max']:.4f} rad, "
                f"whole-body mean/p95/max={body_delta['mean'] * 100.0:.2f}/"
                f"{body_delta['p95'] * 100.0:.2f}/"
                f"{body_delta['max'] * 100.0:.2f} cm"
            )
        for arm_name, arm_metrics in final_metrics.get(
            "arm_pose_fidelity", {}
        ).items():
            parts = []
            for label, key in (
                ("upper", "upper_arm_direction_error_rad"),
                ("fore", "forearm_direction_error_rad"),
                ("bend_plane", "elbow_bend_plane_error_rad"),
                ("flex_diff", "elbow_flexion_difference_rad"),
            ):
                values = arm_metrics[key]
                parts.append(
                    f"{label}={np.rad2deg(values['mean']):.1f}/"
                    f"{np.rad2deg(values['p95']):.1f}/"
                    f"{np.rad2deg(values['max']):.1f} deg"
                )
            robot_flexion = arm_metrics["robot_elbow_flexion_rad"]
            source_flexion = arm_metrics["source_elbow_flexion_rad"]
            parts.append(
                "flex_robot/source_mean="
                f"{np.rad2deg(robot_flexion['mean']):.1f}/"
                f"{np.rad2deg(source_flexion['mean']):.1f} deg"
            )
            print(f"  {arm_name} pose: " + " | ".join(parts))
        base_reference = getattr(self, "last_robot_base_reference_w", None)
        unwarped_reference = getattr(
            self, "last_unwarped_robot_base_reference_w", None
        )
        if base_reference is not None and unwarped_reference is not None:
            warp = np.linalg.norm(base_reference - unwarped_reference, axis=1)
            root_warp_diagnostics = getattr(
                self, "last_root_warp_diagnostics", {}
            )
            print(
                "  robot root warp: "
                f"mean={warp.mean() * 100.0:.2f} cm, "
                f"p95={np.percentile(warp, 95) * 100.0:.2f} cm, "
                f"max={warp.max() * 100.0:.2f} cm, "
                f"yaw={np.rad2deg(root_warp_diagnostics.get('max_yaw', 0.0)):.2f} deg, "
                f"velocity={root_warp_diagnostics.get('max_linear_velocity', 0.0):.3f} m/s, "
                f"acceleration={root_warp_diagnostics.get('max_linear_acceleration', 0.0):.3f} m/s^2, "
                f"yaw_rate={np.rad2deg(root_warp_diagnostics.get('max_yaw_rate', 0.0)):.2f} deg/s, "
                "strategy="
                f"{root_warp_diagnostics.get('strategy', 'minimum_jerk_window_widening')}"
            )
            slack = root_warp_diagnostics.get("derivative_slack", {})
            if any(float(value) > 1e-10 for value in slack.values()):
                print(
                    "  robot root warp derivative slack: "
                    + ", ".join(
                        f"{name}={float(value):.6g}"
                        for name, value in slack.items()
                        if float(value) > 1e-10
                    )
                )
        actuated = np.asarray(trajectory, dtype=float)[:, 7:]
        if len(actuated) >= 4 and actuated.shape[1] > 0:
            velocity = np.diff(actuated, axis=0)
            acceleration = np.diff(actuated, n=2, axis=0)
            jerk = np.diff(actuated, n=3, axis=0)
            base_acceleration = np.diff(
                np.asarray(trajectory, dtype=float)[:, :3], n=2, axis=0
            )
            print(
                "  temporal jitter: joint "
                f"|vel| rms={np.sqrt(np.mean(velocity**2)):.5f} rad/frame, "
                f"rms={np.sqrt(np.mean(acceleration**2)):.5f} "
                f"max={np.max(np.abs(acceleration)):.4f} rad/frame^2, "
                f"|jerk| rms={np.sqrt(np.mean(jerk**2)):.5f}, "
                f"base |acc| rms={np.sqrt(np.mean(base_acceleration**2)):.5f} m/frame^2"
            )

    def _compute_terrain_scale(
        self,
        smplx_trajectory: np.ndarray,
    ) -> float:
        """
        Compute the appropriate scaling factor for the terrain.
        
        This implementation assumes the terrain provided corresponds to the real-world scale
        relative to the robot's size.
        """
        # Default implementation: Scale based on robot vs human height ratio only
        # This assumes the terrain mesh is already at a "human scale" (e.g. captured or designed for humans)
        # and we just need to adjust it for the robot's size.
        
        # Estimate human height from SMPLX trajectory
        # Height ≈ (head Z - min foot Z) in standing pose
        # Or more robustly: Max height difference in trajectory
        
        # Use simple heuristic if we don't have betas to calculate exact height
        # Assuming typical human height of 1.7m if estimation fails or is unreliable
        estimated_human_height = 1.7
        
        if smplx_trajectory is not None and len(smplx_trajectory) > 0:
            # Estimate height from joints
            # SMPLX/SMPL joints: 0: Pelvis, 10/11: Feet, 15: Head (approx)
            # We can find the max vertical extent across all frames
            
            # Find the frame where the person is most upright (max head height)
            # Assuming Z is up
            head_joint_name = self.height_estimation_config.get("head_joint", "Head")
            foot_joint_names = self.height_estimation_config.get("foot_joints", ["L_Foot", "R_Foot"])
            head_top_offset = float(self.height_estimation_config.get("head_top_offset", 0.12))

            head_idx = self.smplx_joint_indices.get(head_joint_name, 15)
            foot_indices = [
                self.smplx_joint_indices[name]
                for name in foot_joint_names
                if name in self.smplx_joint_indices
            ]
            if not foot_indices:
                foot_indices = [10, 11]  # fallback defaults
            
            # Check if we have enough joints
            if smplx_trajectory.shape[1] > head_idx and all(idx < smplx_trajectory.shape[1] for idx in foot_indices):
                # Calculate height per frame: Head Z - Average Foot Z
                head_z = smplx_trajectory[:, head_idx, 2]
                
                # Use min foot Z per frame as ground reference relative to body
                feet_z = np.min(smplx_trajectory[:, foot_indices, 2], axis=1)
                
                # Height per frame
                heights = head_z - feet_z
                
                # Use the maximum height observed (standing pose)
                # Add offset for top of head (head joint is in neck/center of head)
                # Approx 10-15cm from head joint to top of head
                estimated_human_height = np.max(heights) + head_top_offset
                
                # Sanity check: Constrain to reasonable human range [1.4, 2.2]
                estimated_human_height = np.clip(estimated_human_height, 1.4, 2.2)
                
                print(f"Estimated human height from trajectory: {estimated_human_height:.3f}m")

        scale_factor = self.robot_height / estimated_human_height
        
        print(f"Computed terrain scale factor: {scale_factor:.4f} (Robot: {self.robot_height}m, Human Est: {estimated_human_height:.3f}m)")
        
        return float(scale_factor)

    def _extract_foot_positions(self, smplx_trajectory: np.ndarray) -> np.ndarray:
        """Extract foot positions from SMPLX trajectory."""
        # SMPLX joint indices for feet
        # L_Foot: 10, R_Foot: 11 (standard SMPLX ordering)
        foot_indices = [10, 11]

        foot_positions = []
        for frame in smplx_trajectory:
            for foot_idx in foot_indices:
                if foot_idx < len(frame):
                    foot_positions.append(frame[foot_idx])

        return np.array(foot_positions)

    def _scale_terrain_mesh(self, scale_factor: float) -> trimesh.Trimesh:
        """Scale the terrain mesh by the given factor."""
        scaled_mesh = self.terrain_mesh.copy()
        scaled_mesh.apply_scale(scale_factor)
        return scaled_mesh

    def _process_smplx_trajectory(
        self,
        smplx_trajectory: np.ndarray,
        terrain_scale: float
    ) -> np.ndarray:
        """Process SMPLX trajectory for retargeting."""
        # Apply terrain scaling to trajectory coordinates
        scaled_trajectory = smplx_trajectory * terrain_scale

        # Transform coordinate system if needed (SMPLX uses different convention)
        # TODO: Add coordinate system transformation if needed

        return scaled_trajectory

    def _perform_retargeting(
        self,
        processed_trajectory: np.ndarray,
        scaled_terrain: trimesh.Trimesh,
        base_orientations: np.ndarray | None = None,
        base_translations: np.ndarray | None = None,
        end_effector_tasks: Optional[Dict[str, SceneContactTask]] = None,
    ) -> np.ndarray:
        """Perform the actual motion retargeting using generic interaction mesh retargeting."""
        from .retargeting import GenericInteractionRetargeter

        base_quaternions_wxyz = _base_quaternions_wxyz(
            base_orientations, len(processed_trajectory)
        )

        # Create retargeter instance
        # Note: scaled_terrain is already scaled by _scale_terrain_mesh
        # CRITICAL: Pass only valid_joint_mapping to ensure consistent sizes
        # This ensures robot_points and human_joints have the same number of joints
        preserve_root_z = bool(self.retargeting_config.get("preserve_root_z", False))
        retargeter_kwargs: Dict[str, Any] = {}
        if end_effector_tasks:
            retargeter_kwargs["end_effector_config"] = self.retargeting_config.get(
                "end_effector_priority", {}
            )
        if preserve_root_z:
            retargeter_kwargs["preserve_root_z"] = True
        retargeter = GenericInteractionRetargeter(
            self.robot_model,
            self.robot_data,
            scaled_terrain,
            self.valid_joint_mapping,  # Use filtered mapping, not full joint_mapping
            self.robot_height,
            collision_detection_threshold=float(self.retargeting_config.get("collision_detection_threshold", 0.1)),
            terrain_sample_points=int(self.retargeting_config.get("terrain_sample_points", 100)),
            terrain_sample_seed=self.retargeting_config.get("terrain_sample_seed"),
            valid_joint_names=self.valid_joint_names,  # CRITICAL: Pass ordered joint names for consistency
            replace_cylinders_with_capsules=bool(self.retargeting_config.get("replace_cylinders_with_capsules", False)),
            hard_penetration_constraint=self.retargeting_config.get("penetration_resolver", "hard_constraint") == "hard_constraint",
            **retargeter_kwargs,
        )
        end_effector_priority_cfg = dict(
            self.retargeting_config.get("end_effector_priority", {}) or {}
        )
        handoff_guard = _SupportHandoffGuard(
            end_effector_tasks or {},
            end_effector_priority_cfg.get("support_handoff", {}),
        )
        handoff_data = (
            mujoco.MjData(self.robot_model) if handoff_guard.enabled else None
        )

        # Retarget each frame
        retargeted_trajectory = []
        
        # Start from the model's nominal configuration.  Joint-range midpoints
        # are not a neutral pose for humanoids (for G1 they bend the knees by
        # about 1.4 rad), which previously forced the floating base downward to
        # compensate for artificially shortened legs.
        try:
            q_init = np.asarray(self.robot_model.qpos0, dtype=float).copy()
            if q_init.shape != (int(self.robot_model.nq),):
                raise ValueError("invalid model qpos0 shape")
        except (TypeError, ValueError):
            # Lightweight unit-test models and third-party wrappers may not
            # expose qpos0.  Preserve the historical zero initialization for
            # those callers while real MuJoCo models use their neutral pose.
            q_init = np.zeros(int(self.robot_model.nq), dtype=float)
        if len(q_init) >= 7 and np.linalg.norm(q_init[3:7]) < 1e-8:
            q_init[3:7] = [1, 0, 0, 0]  # Identity quaternion (wxyz)
        feasibility_result = getattr(
            self, "last_interaction_feasibility_result", None
        )
        feasibility_warm_start = (
            None
            if feasibility_result is None
            else np.asarray(feasibility_result.configurations, dtype=float)
        )
        if feasibility_warm_start is not None and feasibility_warm_start.shape != (
            len(processed_trajectory),
            int(self.robot_model.nq),
        ):
            raise ValueError(
                "Interaction-feasibility warm start does not match the framewise "
                "trajectory/model dimensions."
            )
        root_position_name = self.base_orientation_config.get("root_position", "Pelvis")
        root_position_idx = getattr(self, "smplx_joint_indices", {}).get(
            root_position_name, 0
        )
        active_contact_offset_ids = {
            name: np.full(len(processed_trajectory), -1, dtype=np.int32)
            for name in (end_effector_tasks or {})
        }
        
        # Initial height: Use processed (scaled) trajectory root height or default
        # Assuming joint 0 is root
        if len(processed_trajectory) > 0:
            q_init[:3] = processed_trajectory[0, root_position_idx] # Start at first frame root pos
            # Maintain initial rotation (identity or from trajectory if available)
        else:
            q_init[2] = self.robot_height * 0.5  # Default height (Z coordinate)

        self._last_estimated_quat = None
        for frame_idx, human_joints in enumerate(processed_trajectory):
            if feasibility_warm_start is not None:
                # Feasibility owns only a warm start and the decided root warp;
                # framewise SQP still produces the delivered configuration.
                q_init = feasibility_warm_start[frame_idx].copy()
                robot_base_reference = getattr(
                    self, "last_robot_base_reference_w", None
                )
                if robot_base_reference is not None:
                    q_init[:3] = robot_base_reference[frame_idx]
                if base_quaternions_wxyz is not None:
                    q_init[3:7] = base_quaternions_wxyz[frame_idx]
            # Only initialize base from SMPLX for the first frame
            # For subsequent frames, the optimization will handle base movement
            if frame_idx == 0:
                # Initialize base position from SMPLX root
                robot_base_reference = getattr(
                    self, "last_robot_base_reference_w", None
                )
                if end_effector_tasks and robot_base_reference is not None:
                    q_init[:3] = robot_base_reference[frame_idx]
                elif base_translations is not None:
                    q_init[:3] = base_translations[frame_idx]
                else:
                    # Use the configured root position as initial guess
                    q_init[:3] = human_joints[root_position_idx]

                # Initialize base orientation
                if base_quaternions_wxyz is not None:
                    q_init[3:7] = base_quaternions_wxyz[frame_idx]
                else:
                    quat_xyzw = self._estimate_base_orientation_from_joints(human_joints)
                    if quat_xyzw is not None:
                        q_init[3:7] = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            # For subsequent frames, q_init already contains the previous optimized configuration
            if preserve_root_z:
                if base_translations is not None:
                    q_init[2] = base_translations[frame_idx, 2]
                else:
                    q_init[2] = human_joints[root_position_idx, 2]
            
            # Extract mapped joints using the proper indices
            # CRITICAL: The order MUST match valid_joint_names exactly
            # mapped_joint_indices[i] corresponds to valid_joint_names[i] for all i
            # human_joints shape: (num_smplx_joints, 3)
            
            # Validate order consistency before extraction
            if len(self.mapped_joint_indices) != len(self.valid_joint_names):
                raise ValueError(
                    f"Order mismatch: mapped_joint_indices has {len(self.mapped_joint_indices)} elements, "
                    f"but valid_joint_names has {len(self.valid_joint_names)} elements."
                )
            
            # Verify that each index corresponds to the correct joint name
            for i, (joint_name, smplx_idx) in enumerate(zip(self.valid_joint_names, self.mapped_joint_indices)):
                expected_idx = self.smplx_joint_indices.get(joint_name)
                if expected_idx != smplx_idx:
                    raise ValueError(
                        f"Order mismatch at position {i}: joint '{joint_name}' has index {smplx_idx} "
                        f"in mapped_joint_indices, but smplx_joint_indices says it should be {expected_idx}."
                    )
            
            mapped_joints = human_joints[self.mapped_joint_indices]
            # mapped_joints shape: (num_mapped_joints, 3)
            # mapped_joints[i] corresponds to valid_joint_names[i] for all i
            
            # CRITICAL: Validate that we extracted the expected number of joints
            expected_num_joints = len(self.valid_joint_names)
            if len(mapped_joints) != expected_num_joints:
                raise ValueError(
                    f"Size mismatch: extracted {len(mapped_joints)} joints from human_joints, "
                    f"but expected {expected_num_joints} joints from valid_joint_names. "
                    f"This indicates an inconsistency in joint mapping."
                )
            
            # Debug: Print order and actual positions for first frame to verify matching
            import sys
            if not hasattr(sys, '_omni_human_joint_order_printed'):
                print(f"\n=== Human Joint Extraction Order ===")
                print(f"valid_joint_names: {self.valid_joint_names}")
                print(f"mapped_joint_indices: {self.mapped_joint_indices}")
                print(f"Extracted {len(mapped_joints)} joints in order: {self.valid_joint_names}")
                print(f"\n=== First Frame Joint Positions (to verify order) ===")
                for i, (name, idx) in enumerate(zip(self.valid_joint_names, self.mapped_joint_indices)):
                    pos = mapped_joints[i] if frame_idx == 0 else mapped_joints[i]
                    if frame_idx == 0:
                        print(f"  {i}: {name} (SMPLX idx {idx}) -> pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                # Specifically check knee vs ankle positions
                if frame_idx == 0:
                    l_knee_idx = self.valid_joint_names.index("L_Knee") if "L_Knee" in self.valid_joint_names else None
                    l_ankle_idx = self.valid_joint_names.index("L_Ankle") if "L_Ankle" in self.valid_joint_names else None
                    if l_knee_idx is not None and l_ankle_idx is not None:
                        l_knee_pos = mapped_joints[l_knee_idx]
                        l_ankle_pos = mapped_joints[l_ankle_idx]
                        print(f"\n=== Knee vs Ankle Position Check ===")
                        print(f"L_Knee (idx {l_knee_idx}, SMPLX {self.mapped_joint_indices[l_knee_idx]}): [{l_knee_pos[0]:.3f}, {l_knee_pos[1]:.3f}, {l_knee_pos[2]:.3f}]")
                        print(f"L_Ankle (idx {l_ankle_idx}, SMPLX {self.mapped_joint_indices[l_ankle_idx]}): [{l_ankle_pos[0]:.3f}, {l_ankle_pos[1]:.3f}, {l_ankle_pos[2]:.3f}]")
                        print(f"Knee should be HIGHER than ankle (knee.z > ankle.z): {l_knee_pos[2] > l_ankle_pos[2]}")
                sys._omni_human_joint_order_printed = True

            # Estimate target base orientation from human joints
            last_target_quat = None
            if frame_idx > 0 and len(retargeted_trajectory) > 0:
                # Get last orientation from previous frame result or previous target estimate
                # We use the result (wxyz) converted back to xyzw for continuity check
                # But wait, retargeted_trajectory has wxyz.
                # _estimate returns xyzw.
                # Better to store the previous estimated xyzw.
                pass
                
            # Prefer the captured SMPL-X root rotation when available.  Joint-
            # position reconstruction remains a source-neutral fallback for
            # position-only inputs.
            if base_quaternions_wxyz is not None:
                target_quat_wxyz = base_quaternions_wxyz[frame_idx]
                target_quat_xyzw = target_quat_wxyz[[1, 2, 3, 0]]
            else:
                target_quat_xyzw = self._estimate_base_orientation_from_joints(
                    human_joints, self._last_estimated_quat
                )
            self._last_estimated_quat = target_quat_xyzw
            
            target_quat_wxyz = None
            if target_quat_xyzw is not None:
                # Convert xyzw to wxyz for MuJoCo
                target_quat_wxyz = np.array([target_quat_xyzw[3], target_quat_xyzw[0], 
                                            target_quat_xyzw[1], target_quat_xyzw[2]])

            if (
                target_quat_wxyz is not None
                and bool(
                    self.retargeting_config.get(
                        "end_effector_priority", {}
                    ).get("lock_base_orientation", False)
                )
            ):
                q_init[3:7] = target_quat_wxyz

            # Retarget frame with previous frame as reference for smoothness
            q_last = retargeted_trajectory[-1] if len(retargeted_trajectory) > 0 else None
            handoff_overrides = handoff_guard.overrides_for_frame(frame_idx)
            frame_kwargs: Dict[str, Any] = dict(
                q_last=q_last,
                target_base_orientation=target_quat_wxyz,
            )
            if end_effector_tasks:
                base_position_cfg = dict(
                    self.retargeting_config.get("end_effector_priority", {}).get(
                        "base_position_constraint", {}
                    )
                    or {}
                )
                if bool(base_position_cfg.get("enabled", False)):
                    robot_base_reference = getattr(
                        self, "last_robot_base_reference_w", None
                    )
                    frame_kwargs["target_base_position"] = (
                        robot_base_reference[frame_idx]
                        if robot_base_reference is not None
                        else human_joints[root_position_idx]
                    )
                frame_kwargs["end_effector_targets"] = {}
                for name, task in end_effector_tasks.items():
                    target_payload = (
                        handoff_guard.payload(
                            name,
                            frame_idx,
                            handoff_overrides[name],
                        )
                        if name in handoff_overrides
                        else task.frame(frame_idx)
                    )
                    frame_kwargs["end_effector_targets"][name] = target_payload
                frame_kwargs["q_prev"] = (
                    retargeted_trajectory[-2]
                    if len(retargeted_trajectory) > 1
                    else None
                )
                frame_kwargs["q_prevprev"] = (
                    retargeted_trajectory[-3]
                    if len(retargeted_trajectory) > 2
                    else None
                )
                frame_kwargs["previous_human_joints"] = (
                    processed_trajectory[frame_idx - 1, self.mapped_joint_indices]
                    if frame_idx > 0
                    else None
                )
                frame_kwargs["previous_previous_human_joints"] = (
                    processed_trajectory[frame_idx - 2, self.mapped_joint_indices]
                    if frame_idx > 1
                    else None
                )
            q_opt = retargeter.retarget_frame(mapped_joints, q_init, **frame_kwargs)
            if preserve_root_z:
                q_opt[2] = q_init[2]
            retargeted_trajectory.append(q_opt)
            if end_effector_tasks:
                for name, task in end_effector_tasks.items():
                    if (
                        task.contact_modes[frame_idx] != int(ContactMode.NONE)
                        or name in handoff_overrides
                    ):
                        offset_id = retargeter._active_contact_offsets.get(name)
                        if offset_id is not None:
                            active_contact_offset_ids[name][frame_idx] = int(offset_id)

            if handoff_guard.needs_observation(frame_idx):
                assert handoff_data is not None
                handoff_data.qpos[:] = q_opt
                mujoco.mj_forward(self.robot_model, handoff_data)
                position_errors: Dict[str, float] = {}
                any_new_support = False
                effectors_cfg = dict(
                    end_effector_priority_cfg.get("effectors", {}) or {}
                )
                for name in handoff_guard.new_support_names:
                    task = end_effector_tasks[name]
                    if not _SupportHandoffGuard._locked_support(
                        task, frame_idx, 0.0
                    ):
                        continue
                    any_new_support = True
                    spec = dict(effectors_cfg.get(name, {}) or {})
                    body_id = mujoco.mj_name2id(
                        self.robot_model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        str(spec.get("robot_body", "")),
                    )
                    if body_id < 0:
                        position_errors[name] = np.inf
                        continue
                    tracking_offset = np.asarray(
                        spec.get("tracking_offset", [0.0, 0.0, 0.0]),
                        dtype=float,
                    ).reshape(3)
                    contact_offsets = np.asarray(
                        spec.get("contact_offsets", [tracking_offset]),
                        dtype=float,
                    )
                    if contact_offsets.ndim == 1:
                        contact_offsets = contact_offsets[None, :]
                    rotation = handoff_data.xmat[body_id].reshape(3, 3)
                    candidates = (
                        handoff_data.xpos[body_id][None, :]
                        + contact_offsets @ rotation.T
                    )
                    offset_id = retargeter._active_contact_offsets.get(name)
                    if offset_id is None or not 0 <= int(offset_id) < len(candidates):
                        offset_id = int(
                            np.argmin(
                                np.linalg.norm(
                                    candidates
                                    - task.contact_positions[frame_idx][None, :],
                                    axis=1,
                                )
                            )
                        )
                    position_errors[name] = float(
                        np.linalg.norm(
                            candidates[int(offset_id)]
                            - task.contact_positions[frame_idx]
                        )
                    )
                penetration_depth = np.inf
                if any_new_support:
                    penetration_config = dict(
                        (
                            self.retargeting_config.get(
                                "trajectory_refinement", {}
                            )
                            or {}
                        ).get("penetration", {})
                        or {}
                    )
                    penetration_depth = float(
                        measure_scene_penetration(
                            self.robot_model,
                            np.asarray(q_opt, dtype=float)[None, :],
                            scaled_terrain,
                            penetration_config,
                        )["max_depth"]
                    )
                handoff_guard.observe(
                    frame_idx,
                    position_errors,
                    penetration_depth,
                )

            # Update initial guess for next frame (smooth trajectory)
            q_init = q_opt.copy()

        if end_effector_tasks:
            end_effector_tasks = handoff_guard.materialize(end_effector_tasks)
            for name, task in list(end_effector_tasks.items()):
                end_effector_tasks[name] = replace(
                    task,
                    active_contact_offset_ids=active_contact_offset_ids[name],
                )
            self.last_support_handoff_diagnostics = handoff_guard.diagnostics()
            for handoff in self.last_support_handoff_diagnostics:
                print(
                    "Support handoff "
                    f"{handoff['type']}: requested release "
                    f"{handoff['requested_release_frame']}, established "
                    f"{handoff['established_frame']}, actual release "
                    f"{handoff['actual_release_frame']}, hold "
                    f"{handoff['hold_frames']} frames, extra hold "
                    f"{handoff['held_extra_frames']}."
                )
            self.last_end_effector_tasks = end_effector_tasks
        return np.array(retargeted_trajectory)

    def _estimate_base_orientation_from_joints(self, joints: np.ndarray, last_quat: np.ndarray | None = None) -> np.ndarray | None:
        """
        Estimate a base orientation from joint positions.

        Uses pelvis/hips/spine to build an approximate body frame:
        - up: pelvis -> spine1
        - right: left_hip -> right_hip
        - forward: right x up
        Returns quaternion in xyzw order.
        
        Args:
            joints: Joint positions array.
            last_quat: Quaternion from previous frame to ensure continuity (xyzw).
        """
        if joints.shape[0] < 4:
            return None

        pelvis_name = self.base_orientation_config.get("pelvis", "Pelvis")
        left_hip_name = self.base_orientation_config.get("left_hip", "L_Hip")
        right_hip_name = self.base_orientation_config.get("right_hip", "R_Hip")
        spine_name = self.base_orientation_config.get("spine", "Spine1")

        pelvis_idx = self.smplx_joint_indices.get(pelvis_name, 0)
        left_hip_idx = self.smplx_joint_indices.get(left_hip_name, 1)
        right_hip_idx = self.smplx_joint_indices.get(right_hip_name, 2)
        spine_idx = self.smplx_joint_indices.get(spine_name, 3)

        max_required_idx = max(pelvis_idx, left_hip_idx, right_hip_idx, spine_idx)
        if joints.shape[0] <= max_required_idx:
            return None

        pelvis = joints[pelvis_idx]
        left_hip = joints[left_hip_idx]
        right_hip = joints[right_hip_idx]
        spine1 = joints[spine_idx]

        up = spine1 - pelvis
        right = right_hip - left_hip

        up_norm = np.linalg.norm(up)
        right_norm = np.linalg.norm(right)
        if up_norm < 1e-8 or right_norm < 1e-8:
            return None

        up = up / up_norm
        right = right / right_norm
        
        # Build forward using right-hand rule
        forward = np.cross(up, right)
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-8:
            return None
        forward = forward / forward_norm

        # Re-orthogonalize right to ensure perfect orthogonality
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)

        # Rotation matrix with body axes in world coordinates.
        # Convention: X=forward, Y=right, Z=up
        rot = np.column_stack([forward, right, up])
        
        # Verify right-handed
        det = np.linalg.det(rot)
        if det < 0:
            forward = -forward
            right = -right
            rot = np.column_stack([forward, right, up])
        
        current_quat = Rotation.from_matrix(rot).as_quat() # xyzw
        
        # Ensure continuity if last quaternion is provided
        if last_quat is not None:
            # Check dot product to see if we need to flip sign
            dot = np.dot(current_quat, last_quat)
            if dot < 0:
                current_quat = -current_quat
                
        return current_quat

    def _get_foot_stabilization_config(self) -> Dict[str, Any]:
        """Return merged foot stabilization settings."""
        defaults = {
            "enabled": False,
            "clearance": 0.01,
            "surface_clearance": 0.005,
            "contact_clearance": 0.04,
            "contact_vertical_speed": 0.18,
            "min_contact_frames": 3,
            "anchor_frames": 3,
            "xy_correction_gain": 1.0,
            "xy_smoothing_window": 5,
            "z_smoothing_window": 5,
            "contact_point_height_band": 0.01,
            "max_xy_correction": 0.08,
            "max_surface_correction": 0.08,
            "surface_iterations": 4,
            "wall_red_axis_only": True,
            "wall_normal_z_threshold": 0.35,
            "wall_x_dominance_threshold": 0.5,
            "body_names": {},
            "ground_lock": True,
            "ground_lock_clearance": 0.0,
            "ground_lock_body_tokens": ["ankle", "foot"],
            "ground_lock_geom_groups": [2, 3],
            "preserve_root_z": False,
        }
        cfg = dict(defaults)
        cfg.update(self.retargeting_config.get("foot_stabilization", {}) or {})
        if bool(self.retargeting_config.get("preserve_root_z", False)):
            cfg["preserve_root_z"] = True
        return cfg

    def _apply_foot_stabilization(
        self,
        retargeted_motion: np.ndarray,
        terrain_mesh: trimesh.Trimesh,
        framerate: float | None = None,
    ) -> np.ndarray:
        """Post-process the retargeted motion to reduce foot penetration and stance slip."""
        cfg = self._get_foot_stabilization_config()
        if not cfg.get("enabled", False):
            return retargeted_motion
        if retargeted_motion.size == 0:
            return retargeted_motion

        foot_specs = self._build_foot_stabilization_specs(cfg)
        if not foot_specs:
            print("Foot stabilization skipped: no foot bodies could be resolved.")
            return retargeted_motion

        ground_lock_specs = []
        if bool(cfg.get("ground_lock", True)):
            ground_lock_specs = self._build_foot_ground_lock_specs(cfg)
            if not ground_lock_specs:
                print("Foot ground lock skipped: no foot geoms could be resolved.")

        stabilized = np.array(retargeted_motion, copy=True)
        preserve_root_z = bool(cfg.get("preserve_root_z", False))
        original_root_z = stabilized[:, 2].copy()
        stabilized, wall_contact_mask = self._apply_surface_collision_corrections(
            stabilized,
            terrain_mesh,
            foot_specs,
            cfg,
        )
        if preserve_root_z:
            stabilized[:, 2] = original_root_z

        contact_band = float(cfg["contact_point_height_band"])
        clearance_target = float(cfg.get("ground_lock_clearance", cfg["clearance"])) if ground_lock_specs else float(cfg["clearance"])
        z_window = int(cfg["z_smoothing_window"])
        xy_window = int(cfg["xy_smoothing_window"])
        min_contact_frames = int(cfg["min_contact_frames"])
        anchor_frames = max(int(cfg["anchor_frames"]), 1)
        xy_gain = float(cfg["xy_correction_gain"])
        max_xy_correction = float(cfg["max_xy_correction"])
        vertical_speed_threshold = float(cfg["contact_vertical_speed"])
        contact_clearance = float(cfg["contact_clearance"])

        if ground_lock_specs:
            positions, min_z = self._compute_foot_ground_lock_series(stabilized, ground_lock_specs, contact_band)
            wall_contact_mask = np.zeros_like(min_z, dtype=bool)
        else:
            positions, min_z = self._compute_foot_contact_series(stabilized, foot_specs, contact_band)
        terrain_heights = self._compute_terrain_heights(terrain_mesh, positions[:, :, :2])
        clearances = min_z - terrain_heights

        lift_clearances = np.where(~wall_contact_mask, clearances, np.inf)
        min_clearance = np.min(lift_clearances, axis=1)
        base_lift = np.maximum(clearance_target - min_clearance, 0.0)
        base_lift[~np.isfinite(min_clearance)] = 0.0
        base_lift_smoothed = self._smooth_signal(base_lift, z_window)
        base_lift = np.maximum(base_lift, base_lift_smoothed)
        if preserve_root_z:
            base_lift[:] = 0.0
        else:
            stabilized[:, 2] += base_lift

        if ground_lock_specs:
            positions, min_z = self._compute_foot_ground_lock_series(stabilized, ground_lock_specs, contact_band)
        else:
            positions, min_z = self._compute_foot_contact_series(stabilized, foot_specs, contact_band)
        terrain_heights = self._compute_terrain_heights(terrain_mesh, positions[:, :, :2])
        clearances = min_z - terrain_heights

        dt = 1.0 / framerate if framerate and framerate > 0 else 1.0
        z_vel = np.zeros_like(min_z)
        if len(stabilized) > 1:
            z_vel[1:] = np.abs(np.diff(min_z, axis=0)) / dt

        contact_mask = (clearances <= contact_clearance) & (z_vel <= vertical_speed_threshold) & (~wall_contact_mask)
        for foot_idx in range(contact_mask.shape[1]):
            contact_mask[:, foot_idx] = self._filter_short_contact_runs(
                contact_mask[:, foot_idx],
                min_contact_frames=min_contact_frames,
            )

        corrections = np.zeros((len(stabilized), 2), dtype=float)
        weights = np.zeros(len(stabilized), dtype=float)

        for foot_idx in range(contact_mask.shape[1]):
            for start, end in self._iter_true_runs(contact_mask[:, foot_idx]):
                anchor_end = min(start + anchor_frames, end)
                anchor_xy = np.median(positions[start:anchor_end, foot_idx, :2], axis=0)
                drift = positions[start:end, foot_idx, :2] - anchor_xy
                corrections[start:end] += -xy_gain * drift
                weights[start:end] += 1.0

        active = weights > 0
        if np.any(active):
            corrections[active] /= weights[active, None]
            corrections = self._smooth_signal(corrections, xy_window)
            norms = np.linalg.norm(corrections, axis=1)
            oversized = norms > max_xy_correction
            if np.any(oversized):
                corrections[oversized] *= (max_xy_correction / norms[oversized])[:, None]
            stabilized[:, :2] += corrections

            if ground_lock_specs:
                positions, min_z = self._compute_foot_ground_lock_series(stabilized, ground_lock_specs, contact_band)
            else:
                positions, min_z = self._compute_foot_contact_series(stabilized, foot_specs, contact_band)
            terrain_heights = self._compute_terrain_heights(terrain_mesh, positions[:, :, :2])
            clearances = min_z - terrain_heights
            lift_clearances = np.where(~wall_contact_mask, clearances, np.inf)
            min_clearance = np.min(lift_clearances, axis=1)
            residual_lift = np.maximum(clearance_target - min_clearance, 0.0)
            residual_lift[~np.isfinite(min_clearance)] = 0.0
            if not preserve_root_z:
                stabilized[:, 2] += residual_lift

        stabilized, _ = self._apply_surface_collision_corrections(
            stabilized,
            terrain_mesh,
            foot_specs,
            cfg,
        )
        if preserve_root_z:
            stabilized[:, 2] = original_root_z

        if np.any(base_lift > 0) or np.any(active):
            print(
                "Applied foot stabilization: "
                f"max base lift={base_lift.max():.4f}m, "
                f"contact frames={int(np.sum(contact_mask))}"
            )

        return stabilized

    def _build_foot_stabilization_specs(self, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Resolve foot bodies and pre-sample contact candidates in body coordinates."""
        specs = []
        body_ids = self._resolve_foot_body_ids(cfg)
        for side, body_id in body_ids.items():
            if body_id < 0:
                continue
            sample_points = self._collect_body_contact_points(body_id)
            if sample_points.size == 0:
                sample_points = np.zeros((1, 3), dtype=float)
            specs.append({
                "side": side,
                "body_id": body_id,
                "body_name": mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                "sample_points": sample_points,
                "collision_points": self._select_collision_probe_points(sample_points),
            })
        return specs

    def _build_foot_ground_lock_specs(self, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect visible/collision foot geoms used for ground locking."""
        body_tokens = tuple(str(token).lower() for token in cfg.get("ground_lock_body_tokens", ["ankle", "foot"]))
        geom_groups = cfg.get("ground_lock_geom_groups", [2, 3])
        geom_groups = set(int(group) for group in geom_groups) if geom_groups is not None else None

        specs = []
        for side, side_tokens in {
            "left": ("left", "l_"),
            "right": ("right", "r_"),
        }.items():
            geom_specs = []
            for geom_id in range(self.robot_model.ngeom):
                if geom_groups is not None and int(self.robot_model.geom_group[geom_id]) not in geom_groups:
                    continue
                body_id = int(self.robot_model.geom_bodyid[geom_id])
                body_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
                body_name_lower = body_name.lower()
                if not any(token in body_name_lower for token in side_tokens):
                    continue
                if not any(token in body_name_lower for token in body_tokens):
                    continue

                sample_points = self._sample_geom_points_in_body_frame(geom_id)
                if sample_points.size == 0:
                    continue
                geom_specs.append({
                    "geom_id": geom_id,
                    "body_id": body_id,
                    "body_name": body_name,
                    "sample_points": sample_points,
                })

            if geom_specs:
                specs.append({"side": side, "geom_specs": geom_specs})
        return specs

    def _resolve_foot_body_ids(self, cfg: Dict[str, Any]) -> Dict[str, int]:
        """Resolve left/right foot body ids from config, mapping, or robot body names."""
        resolved = {}
        explicit = dict(cfg.get("body_names", {}) or {})

        for side, joint_candidates in {
            "left": ["L_Foot", "L_Ankle"],
            "right": ["R_Foot", "R_Ankle"],
        }.items():
            body_id = -1
            explicit_name = explicit.get(side)
            if explicit_name:
                body_id = self._body_name_to_id(explicit_name)

            if body_id < 0:
                for joint_name in joint_candidates:
                    body_name = self.valid_joint_mapping.get(joint_name)
                    if body_name:
                        body_id = self._body_name_to_id(body_name)
                        if body_id >= 0:
                            break

            if body_id < 0:
                body_id = self._search_body_id_by_keywords(side)

            resolved[side] = body_id

        return resolved

    def _body_name_to_id(self, body_name: str) -> int:
        """Convert a body name to a MuJoCo id."""
        try:
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        except Exception:
            return -1
        return int(body_id) if body_id is not None and body_id >= 0 else -1

    def _search_body_id_by_keywords(self, side: str) -> int:
        """Fallback search for foot/ankle body ids when mappings are incomplete."""
        side_tokens = ("left", "l_") if side == "left" else ("right", "r_")
        candidates = []
        for body_idx in range(self.robot_model.nbody):
            body_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_idx)
            if not body_name:
                continue
            name_lower = body_name.lower()
            if not any(token in name_lower for token in side_tokens):
                continue
            if "foot" in name_lower:
                return body_idx
            if "ankle" in name_lower:
                candidates.append(body_idx)
        return candidates[0] if candidates else -1

    def _collect_body_contact_points(self, body_id: int) -> np.ndarray:
        """Collect representative support points for a foot body in body coordinates."""
        geom_ids = [
            int(geom_id)
            for geom_id in collision_geom_ids(self.robot_model)
            if int(self.robot_model.geom_bodyid[geom_id]) == body_id
        ]
        if not geom_ids:
            return np.zeros((0, 3), dtype=float)

        primitive_geom_ids = [
            geom_id for geom_id in geom_ids
            if int(self.robot_model.geom_type[geom_id]) != mujoco.mjtGeom.mjGEOM_MESH
        ]
        if primitive_geom_ids:
            geom_ids = primitive_geom_ids

        points = [self._sample_geom_points_in_body_frame(geom_id) for geom_id in geom_ids]
        points = [pts for pts in points if pts.size > 0]
        if not points:
            return np.zeros((0, 3), dtype=float)
        return np.vstack(points)

    def _sample_geom_points_in_body_frame(self, geom_id: int) -> np.ndarray:
        """Return the shared collision-shell samples in the owning body frame."""
        return geom_surface_points_in_body_frame(self.robot_model, geom_id)

    def _select_collision_probe_points(self, sample_points: np.ndarray) -> np.ndarray:
        """Reduce dense foot samples to a compact set of boundary probes for collision checks."""
        if sample_points.size == 0:
            return np.zeros((0, 3), dtype=float)
        if len(sample_points) <= 24:
            return np.array(sample_points, copy=True)

        directions = np.array([
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, 0.0, -1.0],
            [-1.0, 0.0, -1.0],
            [0.0, 1.0, -1.0],
            [0.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0],
        ], dtype=float)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)

        chosen = set()
        for direction in directions:
            chosen.add(int(np.argmax(sample_points @ direction)))

        local_min_z = np.min(sample_points[:, 2])
        bottom_band = np.where(sample_points[:, 2] <= local_min_z + 0.01)[0]
        chosen.update(int(idx) for idx in bottom_band)

        chosen_points = sample_points[sorted(chosen)]
        unique_points = np.unique(np.round(chosen_points, decimals=6), axis=0)
        return unique_points.astype(float, copy=False)

    def _compute_foot_contact_series(
        self,
        motion: np.ndarray,
        foot_specs: List[Dict[str, Any]],
        contact_band: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-frame foot support positions and minimum support heights."""
        num_frames = len(motion)
        num_feet = len(foot_specs)
        positions = np.zeros((num_frames, num_feet, 3), dtype=float)
        min_z = np.zeros((num_frames, num_feet), dtype=float)

        for frame_idx, qpos in enumerate(motion):
            self.robot_data.qpos[:] = qpos
            mujoco.mj_forward(self.robot_model, self.robot_data)

            for foot_idx, spec in enumerate(foot_specs):
                body_id = spec["body_id"]
                body_pos = np.asarray(self.robot_data.xpos[body_id], dtype=float)
                body_rot = np.asarray(self.robot_data.xmat[body_id], dtype=float).reshape(3, 3)
                world_points = spec["sample_points"] @ body_rot.T + body_pos
                point_z = world_points[:, 2]
                frame_min_z = float(np.min(point_z))
                support_mask = point_z <= frame_min_z + contact_band
                support_points = world_points[support_mask]
                positions[frame_idx, foot_idx] = np.mean(support_points, axis=0)
                min_z[frame_idx, foot_idx] = frame_min_z

        return positions, min_z

    def _compute_foot_ground_lock_series(
        self,
        motion: np.ndarray,
        ground_lock_specs: List[Dict[str, Any]],
        contact_band: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute support points from all configured visible/collision foot geoms."""
        num_frames = len(motion)
        num_feet = len(ground_lock_specs)
        positions = np.zeros((num_frames, num_feet, 3), dtype=float)
        min_z = np.zeros((num_frames, num_feet), dtype=float)

        for frame_idx, qpos in enumerate(motion):
            self.robot_data.qpos[:] = qpos
            mujoco.mj_forward(self.robot_model, self.robot_data)

            for foot_idx, spec in enumerate(ground_lock_specs):
                world_points = []
                for geom_spec in spec["geom_specs"]:
                    body_id = geom_spec["body_id"]
                    body_pos = np.asarray(self.robot_data.xpos[body_id], dtype=float)
                    body_rot = np.asarray(self.robot_data.xmat[body_id], dtype=float).reshape(3, 3)
                    world_points.append(geom_spec["sample_points"] @ body_rot.T + body_pos)
                world_points = np.vstack(world_points)
                point_z = world_points[:, 2]
                frame_min_z = float(np.min(point_z))
                support_mask = point_z <= frame_min_z + contact_band
                support_points = world_points[support_mask]
                positions[frame_idx, foot_idx] = np.mean(support_points, axis=0)
                min_z[frame_idx, foot_idx] = frame_min_z

        return positions, min_z

    def _compute_terrain_heights(self, terrain_mesh: trimesh.Trimesh, xy_points: np.ndarray) -> np.ndarray:
        """Compute terrain heights for a batch of XY positions."""
        heights = np.zeros(xy_points.shape[:2], dtype=float)
        for frame_idx in range(xy_points.shape[0]):
            for foot_idx in range(xy_points.shape[1]):
                x, y = xy_points[frame_idx, foot_idx]
                heights[frame_idx, foot_idx] = compute_mesh_height_at_point(terrain_mesh, float(x), float(y))
        return heights

    def _apply_surface_collision_corrections(
        self,
        motion: np.ndarray,
        terrain_mesh: trimesh.Trimesh,
        foot_specs: List[Dict[str, Any]],
        cfg: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project penetrating foot probe points out of nearby terrain surfaces and flag wall-contact feet."""
        if len(motion) == 0 or not foot_specs:
            return motion, np.zeros((len(motion), len(foot_specs)), dtype=bool)

        triangles = np.asarray(terrain_mesh.triangles, dtype=float)
        if len(triangles) == 0:
            return motion, np.zeros((len(motion), len(foot_specs)), dtype=bool)
        face_normals = np.asarray(terrain_mesh.face_normals, dtype=float)

        surface_clearance = float(cfg["surface_clearance"])
        max_surface_correction = float(cfg["max_surface_correction"])
        surface_iterations = max(int(cfg["surface_iterations"]), 1)
        preserve_root_z = bool(cfg.get("preserve_root_z", False))

        stabilized = np.array(motion, copy=True)
        total_xy_shift = np.zeros(len(stabilized), dtype=float)
        wall_contact_mask = np.zeros((len(stabilized), len(foot_specs)), dtype=bool)

        for frame_idx, qpos in enumerate(stabilized):
            for _ in range(surface_iterations):
                correction_vectors: List[np.ndarray] = []
                wall_x_corrections: List[float] = []
                self.robot_data.qpos[:] = qpos
                mujoco.mj_forward(self.robot_model, self.robot_data)

                for foot_idx, spec in enumerate(foot_specs):
                    collision_points = spec.get("collision_points")
                    if collision_points is None or collision_points.size == 0:
                        continue

                    body_id = spec["body_id"]
                    body_pos = np.asarray(self.robot_data.xpos[body_id], dtype=float)
                    body_rot = np.asarray(self.robot_data.xmat[body_id], dtype=float).reshape(3, 3)
                    world_points = collision_points @ body_rot.T + body_pos

                    for point in world_points:
                        correction, wall_contact = self._compute_surface_point_correction(
                            point,
                            triangles,
                            face_normals,
                            clearance=surface_clearance,
                            cfg=cfg,
                        )
                        wall_contact_mask[frame_idx, foot_idx] |= wall_contact
                        if correction is not None:
                            if wall_contact:
                                wall_x_corrections.append(float(correction[0]))
                            else:
                                correction_vectors.append(correction)

                if not correction_vectors and not wall_x_corrections:
                    break

                correction = np.zeros(3, dtype=float)
                if correction_vectors:
                    correction = np.mean(correction_vectors, axis=0)
                    correction[2] = max(correction[2], 0.0)

                if wall_x_corrections:
                    pos = max((value for value in wall_x_corrections if value > 0.0), default=0.0)
                    neg = min((value for value in wall_x_corrections if value < 0.0), default=0.0)
                    correction[0] = pos if abs(pos) >= abs(neg) else neg

                if preserve_root_z:
                    correction[2] = 0.0

                norm = np.linalg.norm(correction)
                if norm > max_surface_correction and norm > 1e-12:
                    correction *= max_surface_correction / norm

                qpos[:3] += correction
                total_xy_shift[frame_idx] += np.linalg.norm(correction[:2])

            stabilized[frame_idx] = qpos

        if np.any(total_xy_shift > 0):
            print(
                "Applied surface collision correction: "
                f"max xy shift={total_xy_shift.max():.4f}m"
            )

        return stabilized, wall_contact_mask

    def _compute_surface_point_correction(
        self,
        point: np.ndarray,
        triangles: np.ndarray,
        face_normals: np.ndarray,
        clearance: float,
        cfg: Dict[str, Any],
    ) -> Tuple[np.ndarray | None, bool]:
        """Return a correction vector plus whether the point is in wall-contact mode."""
        repeated_points = np.repeat(point[None, :], len(triangles), axis=0)
        closest_points = trimesh.triangles.closest_point(triangles, repeated_points)
        delta = point[None, :] - closest_points
        dist2 = np.einsum("ij,ij->i", delta, delta)
        face_idx = int(np.argmin(dist2))

        closest = closest_points[face_idx]
        normal = face_normals[face_idx]
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-12:
            return None, False
        normal = normal / normal_norm

        signed_offset = float(np.dot(point - closest, normal))
        penetration = clearance - signed_offset
        if penetration <= 0.0:
            return None, False

        wall_contact = False
        if bool(cfg.get("wall_red_axis_only", True)):
            wall_normal_z_threshold = float(cfg.get("wall_normal_z_threshold", 0.35))
            wall_x_dominance_threshold = float(cfg.get("wall_x_dominance_threshold", 0.5))
            if abs(normal[2]) <= wall_normal_z_threshold and abs(normal[0]) >= wall_x_dominance_threshold:
                correction = np.array([penetration * np.sign(normal[0]), 0.0, 0.0], dtype=float)
                wall_contact = True
            else:
                correction = penetration * normal
        else:
            correction = penetration * normal

        if not wall_contact:
            correction[2] = max(correction[2], 0.0)
        if np.linalg.norm(correction) < 1e-9:
            return None, wall_contact
        return correction, wall_contact

    def _smooth_signal(self, values: np.ndarray, window: int) -> np.ndarray:
        """Apply edge-padded moving average smoothing to 1D or 2D arrays."""
        if window <= 1 or values.shape[0] <= 1:
            return np.array(values, copy=True)

        kernel = np.ones(window, dtype=float) / float(window)
        pad_left = window // 2
        pad_right = window - 1 - pad_left

        if values.ndim == 1:
            padded = np.pad(values, (pad_left, pad_right), mode="edge")
            return np.convolve(padded, kernel, mode="valid")

        padded = np.pad(values, ((pad_left, pad_right), (0, 0)), mode="edge")
        smoothed = np.zeros_like(values, dtype=float)
        for col_idx in range(values.shape[1]):
            smoothed[:, col_idx] = np.convolve(padded[:, col_idx], kernel, mode="valid")
        return smoothed

    def _filter_short_contact_runs(self, mask: np.ndarray, min_contact_frames: int) -> np.ndarray:
        """Drop contact runs shorter than the configured minimum."""
        if min_contact_frames <= 1:
            return np.array(mask, copy=True)

        filtered = np.array(mask, copy=True)
        for start, end in self._iter_true_runs(mask):
            if end - start < min_contact_frames:
                filtered[start:end] = False
        return filtered

    def _iter_true_runs(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Return half-open index ranges for all True runs in a boolean mask."""
        runs: List[Tuple[int, int]] = []
        start = None
        for idx, value in enumerate(mask):
            if value and start is None:
                start = idx
            elif not value and start is not None:
                runs.append((start, idx))
                start = None
        if start is not None:
            runs.append((start, len(mask)))
        return runs

    def _estimate_base_orientation_from_joints(self, joints: np.ndarray, last_quat: np.ndarray | None = None) -> np.ndarray | None:
        """
        Estimate a base orientation from joint positions.

        Uses pelvis/hips/spine to build an approximate body frame:
        - up: pelvis -> spine1
        - right: left_hip -> right_hip
        - forward: right x up
        Returns quaternion in xyzw order.
        
        Args:
            joints: Joint positions array.
            last_quat: Quaternion from previous frame to ensure continuity (xyzw).
        """
        if joints.shape[0] < 4:
            return None

        pelvis_name = self.base_orientation_config.get("pelvis", "Pelvis")
        left_hip_name = self.base_orientation_config.get("left_hip", "L_Hip")
        right_hip_name = self.base_orientation_config.get("right_hip", "R_Hip")
        spine_name = self.base_orientation_config.get("spine", "Spine1")

        pelvis_idx = self.smplx_joint_indices.get(pelvis_name, 0)
        left_hip_idx = self.smplx_joint_indices.get(left_hip_name, 1)
        right_hip_idx = self.smplx_joint_indices.get(right_hip_name, 2)
        spine_idx = self.smplx_joint_indices.get(spine_name, 3)

        max_required_idx = max(pelvis_idx, left_hip_idx, right_hip_idx, spine_idx)
        if joints.shape[0] <= max_required_idx:
            return None

        pelvis = joints[pelvis_idx]
        left_hip = joints[left_hip_idx]
        right_hip = joints[right_hip_idx]
        spine1 = joints[spine_idx]

        up = spine1 - pelvis
        right = right_hip - left_hip

        up_norm = np.linalg.norm(up)
        right_norm = np.linalg.norm(right)
        if up_norm < 1e-8 or right_norm < 1e-8:
            return None

        up = up / up_norm
        right = right / right_norm
        
        # Build forward using right-hand rule
        forward = np.cross(up, right)
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-8:
            return None
        forward = forward / forward_norm

        # Re-orthogonalize right to ensure perfect orthogonality
        right = np.cross(up, forward)
        right = right / np.linalg.norm(right)

        # Rotation matrix with body axes in world coordinates.
        # Convention: X=forward, Y=right, Z=up
        rot = np.column_stack([forward, right, up])
        
        # Verify right-handed
        det = np.linalg.det(rot)
        if det < 0:
            forward = -forward
            right = -right
            rot = np.column_stack([forward, right, up])
        
        current_quat = Rotation.from_matrix(rot).as_quat() # xyzw
        
        # Ensure continuity if last quaternion is provided
        if last_quat is not None:
            # Check dot product to see if we need to flip sign
            dot = np.dot(current_quat, last_quat)
            if dot < 0:
                current_quat = -current_quat
                
        return current_quat

    def get_robot_dof(self) -> int:
        """Get the number of degrees of freedom of the robot."""
        return self.robot_model.nq - 7  # Subtract floating base DOF

    def get_joint_names(self) -> List[str]:
        """Get the names of all robot joints (excluding floating base)."""
        return [self.robot_model.joint(i).name for i in range(self.robot_model.njnt)
                if self.robot_model.joint(i).name and self.robot_model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]

    def validate_joint_mapping(self) -> List[str]:
        """Validate that the joint mapping is compatible with the robot.
        
        Note: joint_mapping maps SMPLX joint names to robot BODY (link) names, not joint names.
        So we check for body names in the URDF.
        """
        # Get all body names from the robot model
        robot_bodies = set()
        for i in range(self.robot_model.nbody):
            body_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name:
                robot_bodies.add(body_name)
        
        # Check which mapped body names don't exist
        mapped_bodies = set(self.joint_mapping.values())
        missing_bodies = mapped_bodies - robot_bodies
        return list(missing_bodies)

    def _visualize_trajectory(self, trajectory: np.ndarray, scaled_terrain: trimesh.Trimesh):
        """
        Visualize the SMPLX trajectory using matplotlib 3D animation with terrain mesh.
        
        Args:
            trajectory: Processed trajectory of shape (T, J, 3) where T is frames, J is joints
                       Coordinates are assumed to be in +Z up convention (already transformed)
            scaled_terrain: Scaled terrain mesh
        """
        print(f"Visualizing trajectory with shape: {trajectory.shape}")
        print(f"Terrain mesh: {len(scaled_terrain.vertices)} vertices, {len(scaled_terrain.faces)} faces")
        print("Coordinate system: +Z is up")
        
        num_frames, num_joints, _ = trajectory.shape
        
        # Create figure and 3D axis
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Compute axis limits based on trajectory AND terrain bounds
        all_points = trajectory.reshape(-1, 3)
        traj_bounds = np.array([all_points.min(axis=0), all_points.max(axis=0)])
        
        # Get terrain bounds
        terrain_bounds = scaled_terrain.bounds  # Shape: (2, 3) - min and max
        
        # Combine bounds
        x_min = min(traj_bounds[0, 0], terrain_bounds[0, 0])
        y_min = min(traj_bounds[0, 1], terrain_bounds[0, 1])
        z_min = min(traj_bounds[0, 2], terrain_bounds[0, 2])
        x_max = max(traj_bounds[1, 0], terrain_bounds[1, 0])
        y_max = max(traj_bounds[1, 1], terrain_bounds[1, 1])
        z_max = max(traj_bounds[1, 2], terrain_bounds[1, 2])
        
        # Add some margin
        margin = 0.2
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        
        ax.set_xlim([x_min - margin * x_range, x_max + margin * x_range])
        ax.set_ylim([y_min - margin * y_range, y_max + margin * y_range])
        ax.set_zlim([z_min - margin * z_range, z_max + margin * z_range])
        
        # Set labels (Z is up)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m) - Up')
        ax.set_title('SMPLX Trajectory with Terrain Visualization')
        
        # Set equal aspect ratio
        max_range = max(x_range, y_range, z_range)
        mid_x = (x_max + x_min) / 2
        mid_y = (y_max + y_min) / 2
        mid_z = (z_max + z_min) / 2
        ax.set_xlim([mid_x - max_range/2, mid_x + max_range/2])
        ax.set_ylim([mid_y - max_range/2, mid_y + max_range/2])
        ax.set_zlim([mid_z - max_range/2, mid_z + max_range/2])
        
        # Plot terrain mesh
        print("Rendering terrain mesh...")
        terrain_vertices = scaled_terrain.vertices
        terrain_faces = scaled_terrain.faces
        
        # Create a simplified mesh for visualization if too complex
        if len(terrain_faces) > 10000:
            print(f"Terrain has {len(terrain_faces)} faces, simplifying for visualization...")
            try:
                simplified_terrain = scaled_terrain.simplify_quadric_decimation(10000)
            except ValueError:
                # Fallback for trimesh/fast_simplification version mismatch
                # where target_count is interpreted as target_reduction
                print("Using direct fast_simplification fallback due to trimesh error...")
                import fast_simplification
                vertices, faces = fast_simplification.simplify(
                    scaled_terrain.vertices, 
                    scaled_terrain.faces, 
                    target_count=10000
                )
                simplified_terrain = trimesh.Trimesh(vertices=vertices, faces=faces)
            
            terrain_vertices = simplified_terrain.vertices
            terrain_faces = simplified_terrain.faces
            print(f"Simplified to {len(terrain_faces)} faces")
        
        # Plot terrain as a triangulated surface
        ax.plot_trisurf(
            terrain_vertices[:, 0], 
            terrain_vertices[:, 1], 
            terrain_vertices[:, 2],
            triangles=terrain_faces,
            color='gray',
            alpha=0.3,
            edgecolor='none',
            shade=True,
            linewidth=0
        )
        
        # Initialize scatter plot for joints
        scatter = ax.scatter([], [], [], c='blue', marker='o', s=50, alpha=0.9, edgecolors='black', linewidths=0.5)
        
        # Add text for frame counter
        frame_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
        
        # Define SMPLX skeleton connections (approximate body structure)
        # These are indices in the standard SMPLX joint ordering
        skeleton_connections = [
            # Spine
            (0, 3),   # Pelvis -> Spine1
            (3, 6),   # Spine1 -> Spine2
            (6, 9),   # Spine2 -> Spine3
            (9, 12),  # Spine3 -> Neck
            (12, 15), # Neck -> Head
            
            # Left leg
            (0, 1),   # Pelvis -> L_Hip
            (1, 4),   # L_Hip -> L_Knee
            (4, 7),   # L_Knee -> L_Ankle
            (7, 10),  # L_Ankle -> L_Foot
            
            # Right leg
            (0, 2),   # Pelvis -> R_Hip
            (2, 5),   # R_Hip -> R_Knee
            (5, 8),   # R_Knee -> R_Ankle
            (8, 11),  # R_Ankle -> R_Foot
            
            # Left arm
            (9, 13),  # Spine3 -> L_Collar
            (13, 16), # L_Collar -> L_Shoulder
            (16, 18), # L_Shoulder -> L_Elbow
            (18, 20), # L_Elbow -> L_Wrist
            
            # Right arm
            (9, 14),  # Spine3 -> R_Collar
            (14, 17), # R_Collar -> R_Shoulder
            (17, 19), # R_Shoulder -> R_Elbow
            (19, 21), # R_Elbow -> R_Wrist
        ]
        
        # Filter connections to only include joints we have
        valid_connections = [(i, j) for i, j in skeleton_connections if i < num_joints and j < num_joints]
        
        # Initialize line objects for skeleton
        lines = []
        for _ in valid_connections:
            line, = ax.plot([], [], [], 'r-', linewidth=2.5, alpha=0.8)
            lines.append(line)
        
        # Set viewing angle for better visibility
        ax.view_init(elev=20, azim=45)
        
        def init():
            """Initialize animation."""
            scatter._offsets3d = ([], [], [])
            frame_text.set_text('')
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            return [scatter, frame_text] + lines
        
        def update(frame_idx):
            """Update animation for each frame."""
            # Get joint positions for current frame
            joints = trajectory[frame_idx]  # Shape: (J, 3)
            
            # Update scatter plot
            xs, ys, zs = joints[:, 0], joints[:, 1], joints[:, 2]
            scatter._offsets3d = (xs, ys, zs)
            
            # Update frame counter
            frame_text.set_text(f'Frame: {frame_idx + 1}/{num_frames}')
            
            # Update skeleton lines
            for line, (i, j) in zip(lines, valid_connections):
                x_data = [joints[i, 0], joints[j, 0]]
                y_data = [joints[i, 1], joints[j, 1]]
                z_data = [joints[i, 2], joints[j, 2]]
                line.set_data(x_data, y_data)
                line.set_3d_properties(z_data)
            
            return [scatter, frame_text] + lines
        
        # Create animation
        print(f"Creating animation for {num_frames} frames...")
        anim = FuncAnimation(
            fig, 
            update, 
            frames=num_frames,
            init_func=init,
            interval=33,  # ~30 FPS
            blit=True,
            repeat=True
        )
        
        print("Displaying animation. Close the window to continue...")
        plt.tight_layout()
        plt.show()
        print("Visualization complete.")
