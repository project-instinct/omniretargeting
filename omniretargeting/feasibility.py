"""Joint-limit-aware interaction feasibility before framewise retargeting.

The framewise retargeter should not be asked to satisfy a contact schedule that
the robot cannot realize.  This module performs a small whole-window SQP over
the robot's scalar joints, a bounded robot-only root correction, and contact
points parameterized directly in their source affordance polygons.  It never
modifies the source motion or scene and never substitutes a point on a radial
reach shell for a point on the real object.

Fallbacks are attempted in this order:

0. joint motion with the unwarped root and source surface point,
1. bounded root translation/yaw,
2. re-pick a point in the same affordance region,
3. relax tangential position and orientation,
4. drop optional contacts, and
5. retain required contacts with an explicit residual and reason.

The public result contains only dataclasses, scalars, strings, and NumPy
arrays.  :meth:`InteractionFeasibilityResult.to_npz_dict` flattens the
variable-length segment records for later motion export.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import cvxpy as cp
import mujoco
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from .affordance import ContactRegion
from .contact import SceneContactTask
from .morphology import MorphologyReport
from .retargeting import _apply_configured_joint_limits
from .trajectory_refinement import _scalar_joint_coordinates


REACHABLE = "reachable"
ADAPTED = "adapted"
DROPPED_OPTIONAL = "dropped_optional"
INFEASIBLE = "infeasible"

FALLBACK_NONE = 0
FALLBACK_ROOT = 1
FALLBACK_REGION = 2
FALLBACK_RELAX = 3
FALLBACK_DROP_OPTIONAL = 4
FALLBACK_REQUIRED_SLACK = 5


@dataclass(frozen=True)
class FeasibilityOptions:
    """Bounds, tolerances, and weights for the feasibility prepass.

    Velocity and acceleration limits apply to the robot-only root correction,
    not to the immutable source root trajectory.  Translation bounds are in
    metres, angular values are in radians, and derivative bounds are per
    second and per second squared.
    """

    dt: float = 1.0 / 30.0
    max_root_translation: tuple[float, float, float] = (0.20, 0.20, 0.15)
    max_root_translation_norm: float = 0.28
    max_root_yaw: float = np.deg2rad(20.0)
    max_root_velocity: tuple[float, float, float] = (1.2, 1.2, 0.8)
    max_root_acceleration: tuple[float, float, float] = (8.0, 8.0, 6.0)
    max_yaw_velocity: float = np.deg2rad(120.0)
    max_yaw_acceleration: float = np.deg2rad(720.0)
    max_tangential_relaxation: float = 0.04
    max_orientation_relaxation: float = np.deg2rad(35.0)
    contact_tolerance: float = 3.0e-3
    orientation_tolerance: float = np.deg2rad(4.0)
    adaptation_tolerance: float = 1.0e-5
    max_sqp_iterations: int = 40
    sqp_step_fraction: float = 0.30
    joint_trust_region: float = 0.35
    root_trust_region: float = 0.08
    yaw_trust_region: float = np.deg2rad(8.0)
    root_deviation_weight: float = 80.0
    root_velocity_weight: float = 2.0
    root_acceleration_weight: float = 0.2
    yaw_deviation_weight: float = 30.0
    contact_point_weight: float = 60.0
    contact_point_temporal_weight: float = 3.0
    contact_slack_weight: float = 2.0e5
    orientation_slack_weight: float = 2.0e4
    joint_reference_weight: float = 2.0e-2
    joint_velocity_weight: float = 5.0e-2
    step_regularization_weight: float = 1.0e-3
    solver: str = "CLARABEL"

    def __post_init__(self) -> None:
        positive = {
            "dt": self.dt,
            "contact_tolerance": self.contact_tolerance,
            "orientation_tolerance": self.orientation_tolerance,
            "max_sqp_iterations": self.max_sqp_iterations,
            "sqp_step_fraction": self.sqp_step_fraction,
            "joint_trust_region": self.joint_trust_region,
            "root_trust_region": self.root_trust_region,
            "yaw_trust_region": self.yaw_trust_region,
        }
        if any(float(value) <= 0.0 for value in positive.values()):
            raise ValueError(f"Feasibility options must be positive: {positive}.")
        if self.sqp_step_fraction > 1.0:
            raise ValueError("sqp_step_fraction must lie in (0, 1].")
        for name in (
            "max_root_translation",
            "max_root_velocity",
            "max_root_acceleration",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (3,) or np.any(value < 0.0) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite non-negative three-vector.")
        for name in (
            "max_root_translation_norm",
            "max_root_yaw",
            "max_yaw_velocity",
            "max_yaw_acceleration",
            "max_tangential_relaxation",
            "max_orientation_relaxation",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")


@dataclass(frozen=True)
class SegmentFeasibilityResult:
    """Diagnosable output for one ``(effector, contact_segment_id)`` run."""

    effector_name: str
    segment_id: int
    frame_indices: np.ndarray
    required: bool
    region_id: str
    chosen_contact_points_world: np.ndarray
    chosen_contact_points_region_local: np.ndarray
    chosen_contact_plane_coordinates: np.ndarray
    source_contact_points_world: np.ndarray
    root_correction_translation: np.ndarray
    root_correction_yaw: np.ndarray
    joint_configurations: np.ndarray
    fallback_level: int
    contact_slack: np.ndarray
    orientation_slack: np.ndarray
    contact_errors: np.ndarray
    orientation_errors: np.ndarray
    reachability_status: str
    infeasible_reason: str

    @property
    def max_contact_error(self) -> float:
        return _max_row_norm(self.contact_errors)

    @property
    def max_contact_slack(self) -> float:
        return _max_row_norm(self.contact_slack)

    @property
    def max_root_warp(self) -> float:
        return _max_row_norm(self.root_correction_translation)


@dataclass(frozen=True)
class InteractionFeasibilityResult:
    """Complete prepass trajectory plus variable-length segment diagnostics."""

    configurations: np.ndarray
    root_correction_translation: np.ndarray
    root_correction_yaw: np.ndarray
    segments: tuple[SegmentFeasibilityResult, ...]
    solver_status: str
    fallback_level_reached: int

    @property
    def feasible(self) -> bool:
        return all(
            segment.reachability_status not in {INFEASIBLE}
            for segment in self.segments
            if segment.required
        )

    def segment(self, effector_name: str, segment_id: int) -> SegmentFeasibilityResult:
        """Return one record without making callers depend on tuple ordering."""
        for result in self.segments:
            if result.effector_name == effector_name and result.segment_id == segment_id:
                return result
        raise KeyError((effector_name, segment_id))

    def to_npz_dict(self, prefix: str = "interaction_feasibility") -> dict[str, np.ndarray]:
        """Flatten this result into non-object arrays accepted by ``np.savez``."""
        prefix = str(prefix).rstrip("_")
        segment_count = len(self.segments)
        lengths = np.asarray(
            [len(segment.frame_indices) for segment in self.segments],
            dtype=np.int32,
        )
        offsets = np.r_[0, np.cumsum(lengths, dtype=np.int64)]

        def concatenate(field: str, width: int | None = None) -> np.ndarray:
            arrays = [np.asarray(getattr(segment, field)) for segment in self.segments]
            if arrays:
                return np.concatenate(arrays, axis=0)
            if width is None:
                return np.zeros(0, dtype=float)
            return np.zeros((0, width), dtype=float)

        occurrence_segment = np.repeat(np.arange(segment_count, dtype=np.int32), lengths)
        return {
            f"{prefix}_configurations": np.asarray(self.configurations, dtype=float),
            f"{prefix}_root_translation": np.asarray(self.root_correction_translation, dtype=float),
            f"{prefix}_root_yaw": np.asarray(self.root_correction_yaw, dtype=float),
            f"{prefix}_solver_status": np.asarray(self.solver_status),
            f"{prefix}_fallback_level_reached": np.asarray(
                self.fallback_level_reached, dtype=np.int8
            ),
            f"{prefix}_segment_offsets": offsets,
            f"{prefix}_segment_effector": np.asarray(
                [segment.effector_name for segment in self.segments]
            ),
            f"{prefix}_segment_id": np.asarray(
                [segment.segment_id for segment in self.segments], dtype=np.int32
            ),
            f"{prefix}_segment_required": np.asarray(
                [segment.required for segment in self.segments], dtype=bool
            ),
            f"{prefix}_segment_region_id": np.asarray(
                [segment.region_id for segment in self.segments]
            ),
            f"{prefix}_segment_fallback_level": np.asarray(
                [segment.fallback_level for segment in self.segments], dtype=np.int8
            ),
            f"{prefix}_segment_status": np.asarray(
                [segment.reachability_status for segment in self.segments]
            ),
            f"{prefix}_segment_infeasible_reason": np.asarray(
                [segment.infeasible_reason for segment in self.segments]
            ),
            f"{prefix}_occurrence_segment": occurrence_segment,
            f"{prefix}_frame_index": concatenate("frame_indices"),
            f"{prefix}_contact_point_world": concatenate("chosen_contact_points_world", 3),
            f"{prefix}_contact_point_region_local": concatenate(
                "chosen_contact_points_region_local", 3
            ),
            f"{prefix}_contact_point_plane": concatenate("chosen_contact_plane_coordinates", 2),
            f"{prefix}_source_contact_point_world": concatenate("source_contact_points_world", 3),
            f"{prefix}_contact_slack": concatenate("contact_slack", 3),
            f"{prefix}_orientation_slack": concatenate("orientation_slack", 3),
            f"{prefix}_contact_error": concatenate("contact_errors", 3),
            f"{prefix}_orientation_error": concatenate("orientation_errors", 3),
        }


@dataclass(frozen=True)
class _Occurrence:
    key: tuple[str, int]
    frame: int
    task: SceneContactTask
    region: ContactRegion
    required: bool
    source_world: np.ndarray
    source_uv: np.ndarray
    world_origin: np.ndarray
    world_basis: np.ndarray
    polygon_a: np.ndarray
    polygon_b: np.ndarray
    endpoint_body_id: int
    endpoint_offset: np.ndarray
    orientation_target: np.ndarray | None


@dataclass
class _State:
    joints: np.ndarray
    root_translation: np.ndarray
    root_yaw: np.ndarray
    plane_coordinates: np.ndarray
    tangential_slack: np.ndarray


@dataclass(frozen=True)
class _StageOutcome:
    state: _State
    configurations: np.ndarray
    contact_errors: np.ndarray
    orientation_errors: np.ndarray
    contact_slack: np.ndarray
    orientation_slack: np.ndarray
    dropped: frozenset[tuple[str, int]]
    status: str
    strict_success: bool
    objective: float


def _max_row_norm(value: np.ndarray) -> float:
    array = np.asarray(value, dtype=float)
    return float(np.max(np.linalg.norm(array, axis=-1))) if len(array) else 0.0


def _world_region_frame(region: ContactRegion) -> tuple[np.ndarray, np.ndarray]:
    rotation = np.asarray(region.world_from_object[:3, :3], dtype=float)
    translation = np.asarray(region.world_from_object[:3, 3], dtype=float)
    origin = rotation @ np.asarray(region.plane_origin_local, dtype=float) + translation
    basis = rotation @ np.asarray(region.tangent_basis_local, dtype=float)
    return origin, basis


def _world_to_region_local(region: ContactRegion, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    rotation = np.asarray(region.world_from_object[:3, :3], dtype=float)
    translation = np.asarray(region.world_from_object[:3, 3], dtype=float)
    return (points - translation) @ np.linalg.inv(rotation).T


def _plane_coordinates(region: ContactRegion, points_world: np.ndarray) -> np.ndarray:
    local = _world_to_region_local(region, points_world)
    relative = local - np.asarray(region.plane_origin_local, dtype=float)
    return relative @ np.asarray(region.tangent_basis_local, dtype=float)


def _polygon_halfspaces(polygon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    polygon = np.asarray(polygon, dtype=float)
    hull = ConvexHull(polygon)
    # scipy represents the hull interior as normal @ x + offset <= 0.
    return hull.equations[:, :2].copy(), (-hull.equations[:, 2]).copy()


def _resolve_region(
    regions: Mapping[Any, Any], name: str, segment_id: int
) -> ContactRegion:
    key = (name, int(segment_id))
    candidates: list[Any] = []
    if key in regions:
        candidates.append(regions[key])
    if name in regions:
        value = regions[name]
        if isinstance(value, Mapping):
            if segment_id in value:
                candidates.append(value[segment_id])
            elif str(segment_id) in value:
                candidates.append(value[str(segment_id)])
        else:
            candidates.append(value)
    if segment_id in regions:
        candidates.append(regions[segment_id])
    if str(segment_id) in regions:
        candidates.append(regions[str(segment_id)])
    for candidate in candidates:
        if isinstance(candidate, ContactRegion):
            return candidate
    raise KeyError(
        f"No ContactRegion supplied for contact segment {key!r}; use a "
        "(effector_name, segment_id) key or a nested effector mapping."
    )


def _resolve_required(
    required_contacts: bool | Mapping[Any, Any],
    name: str,
    segment_id: int,
) -> bool:
    if isinstance(required_contacts, (bool, np.bool_)):
        return bool(required_contacts)
    key = (name, int(segment_id))
    if key in required_contacts:
        return bool(required_contacts[key])
    if name in required_contacts:
        value = required_contacts[name]
        if isinstance(value, Mapping):
            if segment_id in value:
                return bool(value[segment_id])
            if str(segment_id) in value:
                return bool(value[str(segment_id)])
        elif isinstance(value, (bool, np.bool_)):
            return bool(value)
    # SUPPORT is semantically required unless a caller explicitly overrides
    # it.  Other modes also default to required: optionality must be explicit,
    # so a mixed-mode segment can never accidentally disappear.
    return True


def _normalize_root_reference(
    model: mujoco.MjModel,
    num_frames: int,
    root_reference_qpos: np.ndarray | None,
) -> np.ndarray:
    if root_reference_qpos is None:
        return np.repeat(np.asarray(model.qpos0, dtype=float)[None, :], num_frames, axis=0)
    reference = np.asarray(root_reference_qpos, dtype=float).copy()
    if reference.shape == (model.nq,):
        reference = np.repeat(reference[None, :], num_frames, axis=0)
    if reference.shape == (num_frames, model.nq):
        result = reference
    elif reference.shape in {(3,), (num_frames, 3)}:
        positions = (
            np.repeat(reference.reshape(1, 3), num_frames, axis=0)
            if reference.shape == (3,)
            else reference
        )
        result = np.repeat(np.asarray(model.qpos0, dtype=float)[None, :], num_frames, axis=0)
        if model.nq < 7 or model.nv < 6:
            raise ValueError("A three-vector root reference requires a floating-base model.")
        result[:, :3] = positions
    else:
        raise ValueError(
            "root_reference_qpos must have shape (nq,), (T,nq), (3,), or (T,3); "
            f"got {reference.shape}."
        )
    if not np.all(np.isfinite(result)):
        raise ValueError("root_reference_qpos must be finite.")
    return result.copy()


def _compose_configuration(
    reference: np.ndarray,
    joint_qpos_indices: np.ndarray,
    joints: np.ndarray,
    root_translation: np.ndarray,
    root_yaw: np.ndarray,
) -> np.ndarray:
    configurations = np.asarray(reference, dtype=float).copy()
    configurations[:, joint_qpos_indices] = joints
    if configurations.shape[1] >= 7:
        configurations[:, :3] = reference[:, :3] + root_translation
        for frame in range(len(configurations)):
            base_rotation = Rotation.from_quat(reference[frame, 3:7], scalar_first=True)
            yaw_rotation = Rotation.from_rotvec([0.0, 0.0, root_yaw[frame]])
            quaternion = (yaw_rotation * base_rotation).as_quat(scalar_first=True)
            if np.dot(quaternion, reference[frame, 3:7]) < 0.0:
                quaternion *= -1.0
            configurations[frame, 3:7] = quaternion
    return configurations


def _point_kinematics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    configuration: np.ndarray,
    body_id: int,
    local_offset: np.ndarray,
    joint_dof_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data.qpos[:] = configuration
    mujoco.mj_forward(model, data)
    rotation = data.xmat[body_id].reshape(3, 3)
    point = data.xpos[body_id] + rotation @ local_offset
    jacobian_position = np.zeros((3, model.nv), dtype=float)
    jacobian_rotation = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jac(
        model,
        data,
        jacobian_position,
        jacobian_rotation,
        point,
        body_id,
    )
    return (
        point.copy(),
        jacobian_position[:, joint_dof_indices].copy(),
        jacobian_rotation[:, joint_dof_indices].copy(),
    )


def _endpoint_rotation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    configuration: np.ndarray,
    body_id: int,
) -> np.ndarray:
    data.qpos[:] = configuration
    mujoco.mj_forward(model, data)
    return data.xmat[body_id].reshape(3, 3).copy()


def _rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(target) @ np.asarray(current).T).as_rotvec()


def _build_occurrences(
    model: mujoco.MjModel,
    morphology: MorphologyReport,
    contact_tasks: Mapping[str, SceneContactTask],
    contact_regions: Mapping[Any, Any],
    required_contacts: bool | Mapping[Any, Any],
) -> tuple[list[_Occurrence], list[tuple[str, int]], int]:
    if not contact_tasks:
        raise ValueError("contact_tasks must contain at least one SceneContactTask.")
    lengths = {len(task.contact_modes) for task in contact_tasks.values()}
    if len(lengths) != 1:
        raise ValueError("All contact tasks must have the same number of frames.")
    num_frames = lengths.pop()
    occurrences: list[_Occurrence] = []
    keys: list[tuple[str, int]] = []
    for name, task in contact_tasks.items():
        if not isinstance(task, SceneContactTask):
            raise TypeError(f"contact_tasks[{name!r}] must be a SceneContactTask.")
        if name not in morphology.chains:
            raise KeyError(f"Morphology report has no chain for effector {name!r}.")
        chain = morphology.chains[name]
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, chain.endpoint_body
        )
        if body_id < 0:
            raise ValueError(f"Missing endpoint body {chain.endpoint_body!r} for {name!r}.")
        segment_ids = np.asarray(task.contact_segment_ids, dtype=np.int32)
        for raw_segment_id in np.unique(segment_ids[segment_ids >= 0]):
            segment_id = int(raw_segment_id)
            key = (str(name), segment_id)
            keys.append(key)
            frames = np.flatnonzero(segment_ids == segment_id)
            region = _resolve_region(contact_regions, str(name), segment_id)
            required = _resolve_required(required_contacts, str(name), segment_id)
            world_origin, world_basis = _world_region_frame(region)
            polygon_a, polygon_b = _polygon_halfspaces(region.polygon_local_2d)
            orientation_track = task.orientation_target_rotations
            for frame in frames:
                source_world = region.query(task.contact_positions[frame]).nearest_point.copy()
                source_uv = _plane_coordinates(region, source_world[None, :])[0]
                orientation_target = (
                    None
                    if orientation_track is None
                    else np.asarray(orientation_track[frame], dtype=float).copy()
                )
                occurrences.append(
                    _Occurrence(
                        key=key,
                        frame=int(frame),
                        task=task,
                        region=region,
                        required=required,
                        source_world=source_world,
                        source_uv=source_uv,
                        world_origin=world_origin.copy(),
                        world_basis=world_basis.copy(),
                        polygon_a=polygon_a.copy(),
                        polygon_b=polygon_b.copy(),
                        endpoint_body_id=int(body_id),
                        endpoint_offset=np.asarray(chain.endpoint_offset, dtype=float).copy(),
                        orientation_target=orientation_target,
                    )
                )
    if not occurrences:
        raise ValueError("contact_tasks contain no active contact segments.")
    return occurrences, keys, num_frames


def _initial_state(
    model: mujoco.MjModel,
    morphology: MorphologyReport,
    reference: np.ndarray,
    joint_indices: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    occurrences: Sequence[_Occurrence],
) -> _State:
    joints = reference[:, joint_indices].copy()
    coordinate_by_qpos = {
        int(qpos_index): coordinate
        for coordinate, qpos_index in enumerate(joint_indices)
    }
    # A fully extended limb is a radial singularity: its first-order point
    # Jacobian cannot see that bending shortens the chain.  Seed active chains
    # modestly toward their joint-range midpoints so SQP can discover interior
    # workspace points.  This is deterministic, respects effective limits,
    # and changes only the robot's local initial iterate.
    for occurrence in occurrences:
        chain = morphology.chains[occurrence.key[0]]
        for joint in chain.joints:
            qpos_index = int(model.jnt_qposadr[joint.joint_id])
            coordinate = coordinate_by_qpos.get(qpos_index)
            if coordinate is None:
                continue
            midpoint = 0.5 * (lower[coordinate] + upper[coordinate])
            if not np.isfinite(midpoint):
                continue
            value = reference[occurrence.frame, qpos_index]
            joints[occurrence.frame, coordinate] = np.clip(
                value + 0.35 * (midpoint - value),
                lower[coordinate],
                upper[coordinate],
            )
    return _State(
        joints=joints,
        root_translation=np.zeros((len(reference), 3), dtype=float),
        root_yaw=np.zeros(len(reference), dtype=float),
        plane_coordinates=np.asarray([occ.source_uv for occ in occurrences], dtype=float),
        tangential_slack=np.zeros((len(occurrences), 2), dtype=float),
    )


def _derivative_objective(
    value: cp.Expression,
    velocity_weight: float,
    acceleration_weight: float,
    dt: float,
) -> cp.Expression:
    objective: cp.Expression = 0.0
    if value.shape[0] > 1 and velocity_weight > 0.0:
        objective += velocity_weight * cp.sum_squares(cp.diff(value, axis=0) / dt)
    if value.shape[0] > 2 and acceleration_weight > 0.0:
        velocity = cp.diff(value, axis=0) / dt
        objective += acceleration_weight * cp.sum_squares(cp.diff(velocity, axis=0) / dt)
    return objective


def _add_root_derivative_bounds(
    constraints: list[cp.Constraint],
    translation: cp.Expression,
    yaw: cp.Expression,
    options: FeasibilityOptions,
) -> None:
    if translation.shape[0] > 1:
        velocity = cp.diff(translation, axis=0) / options.dt
        constraints.extend(
            [
                velocity <= np.asarray(options.max_root_velocity, dtype=float),
                velocity >= -np.asarray(options.max_root_velocity, dtype=float),
            ]
        )
        yaw_velocity = cp.diff(yaw, axis=0) / options.dt
        constraints.extend(
            [yaw_velocity <= options.max_yaw_velocity, yaw_velocity >= -options.max_yaw_velocity]
        )
    if translation.shape[0] > 2:
        acceleration = cp.diff(cp.diff(translation, axis=0), axis=0) / options.dt**2
        constraints.extend(
            [
                acceleration <= np.asarray(options.max_root_acceleration, dtype=float),
                acceleration >= -np.asarray(options.max_root_acceleration, dtype=float),
            ]
        )
        yaw_acceleration = cp.diff(cp.diff(yaw, axis=0), axis=0) / options.dt**2
        constraints.extend(
            [
                yaw_acceleration <= options.max_yaw_acceleration,
                yaw_acceleration >= -options.max_yaw_acceleration,
            ]
        )


def _evaluate(
    model: mujoco.MjModel,
    reference: np.ndarray,
    joint_indices: np.ndarray,
    joint_dof_indices: np.ndarray,
    occurrences: Sequence[_Occurrence],
    state: _State,
    dropped: frozenset[tuple[str, int]],
    level: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    configurations = _compose_configuration(
        reference,
        joint_indices,
        state.joints,
        state.root_translation,
        state.root_yaw,
    )
    data = mujoco.MjData(model)
    contact_errors = np.zeros((len(occurrences), 3), dtype=float)
    orientation_errors = np.zeros((len(occurrences), 3), dtype=float)
    contact_slack = np.zeros((len(occurrences), 3), dtype=float)
    orientation_slack = np.zeros((len(occurrences), 3), dtype=float)
    for index, occurrence in enumerate(occurrences):
        if occurrence.key in dropped:
            continue
        point, _, _ = _point_kinematics(
            model,
            data,
            configurations[occurrence.frame],
            occurrence.endpoint_body_id,
            occurrence.endpoint_offset,
            joint_dof_indices,
        )
        uv = state.plane_coordinates[index]
        if level >= FALLBACK_RELAX:
            uv = uv + state.tangential_slack[index]
        target = occurrence.world_origin + occurrence.world_basis @ uv
        error = point - target
        contact_errors[index] = error
        if level >= FALLBACK_RELAX:
            contact_slack[index] = (
                occurrence.world_basis @ state.tangential_slack[index] + error
            )
        if occurrence.orientation_target is not None:
            current_rotation = _endpoint_rotation(
                model,
                data,
                configurations[occurrence.frame],
                occurrence.endpoint_body_id,
            )
            error_rotation = _rotation_error(occurrence.orientation_target, current_rotation)
            orientation_errors[index] = error_rotation
            if level >= FALLBACK_RELAX:
                orientation_slack[index] = error_rotation
    return configurations, contact_errors, orientation_errors, contact_slack, orientation_slack


def _strict_success(
    occurrences: Sequence[_Occurrence],
    contact_errors: np.ndarray,
    orientation_errors: np.ndarray,
    dropped: frozenset[tuple[str, int]],
    options: FeasibilityOptions,
    level: int,
    *,
    required_only: bool = False,
) -> bool:
    for index, occurrence in enumerate(occurrences):
        if occurrence.key in dropped or (required_only and not occurrence.required):
            continue
        if np.linalg.norm(contact_errors[index]) > options.contact_tolerance:
            return False
        if occurrence.orientation_target is not None:
            orientation_limit = (
                options.max_orientation_relaxation + options.orientation_tolerance
                if level >= FALLBACK_RELAX
                else options.orientation_tolerance
            )
            if np.linalg.norm(orientation_errors[index]) > orientation_limit:
                return False
    return True


def _solve_stage(
    model: mujoco.MjModel,
    reference: np.ndarray,
    joint_indices: np.ndarray,
    joint_dof_indices: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    occurrences: Sequence[_Occurrence],
    initial: _State,
    level: int,
    dropped: frozenset[tuple[str, int]],
    options: FeasibilityOptions,
) -> _StageOutcome:
    state = _State(
        joints=initial.joints.copy(),
        root_translation=initial.root_translation.copy(),
        root_yaw=initial.root_yaw.copy(),
        plane_coordinates=initial.plane_coordinates.copy(),
        tangential_slack=initial.tangential_slack.copy(),
    )
    source_uv = np.asarray([occurrence.source_uv for occurrence in occurrences], dtype=float)
    orientation_indices = [
        index
        for index, occurrence in enumerate(occurrences)
        if occurrence.orientation_target is not None and occurrence.key not in dropped
    ]
    status = "not_solved"
    objective_value = np.inf

    for _ in range(options.max_sqp_iterations):
        configurations = _compose_configuration(
            reference,
            joint_indices,
            state.joints,
            state.root_translation,
            state.root_yaw,
        )
        data = mujoco.MjData(model)
        kinematics: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]] = []
        for occurrence in occurrences:
            point, joint_jacobian, angular_jacobian = _point_kinematics(
                model,
                data,
                configurations[occurrence.frame],
                occurrence.endpoint_body_id,
                occurrence.endpoint_offset,
                joint_dof_indices,
            )
            rotation_error = None
            if occurrence.orientation_target is not None:
                current_rotation = _endpoint_rotation(
                    model,
                    data,
                    configurations[occurrence.frame],
                    occurrence.endpoint_body_id,
                )
                rotation_error = _rotation_error(occurrence.orientation_target, current_rotation)
            kinematics.append((point, joint_jacobian, angular_jacobian, rotation_error))

        dq = cp.Variable(state.joints.shape)
        droot = cp.Variable(state.root_translation.shape)
        dyaw = cp.Variable(state.root_yaw.shape)
        uv = cp.Variable(state.plane_coordinates.shape)
        tangent = cp.Variable(state.tangential_slack.shape)
        elastic_position = cp.Variable((len(occurrences), 3))
        elastic_orientation = cp.Variable((len(occurrences), 3))
        full_slack = (
            cp.Variable((len(occurrences), 3))
            if level >= FALLBACK_REQUIRED_SLACK
            else None
        )
        orientation_relax = (
            cp.Variable((len(occurrences), 3))
            if level >= FALLBACK_RELAX
            else None
        )

        joints_new = state.joints + dq
        root_new = state.root_translation + droot
        yaw_new = state.root_yaw + dyaw
        constraints: list[cp.Constraint] = [
            joints_new >= lower,
            joints_new <= upper,
            cp.abs(dq) <= options.joint_trust_region,
            cp.abs(droot) <= options.root_trust_region,
            cp.abs(dyaw) <= options.yaw_trust_region,
        ]
        if level < FALLBACK_ROOT:
            constraints.extend([root_new == 0.0, yaw_new == 0.0])
        else:
            max_translation = np.asarray(options.max_root_translation, dtype=float)
            constraints.extend(
                [
                    root_new <= max_translation,
                    root_new >= -max_translation,
                    cp.abs(yaw_new) <= options.max_root_yaw,
                ]
            )
            if options.max_root_translation_norm > 0.0:
                constraints.extend(
                    cp.norm(root_new[frame], 2) <= options.max_root_translation_norm
                    for frame in range(len(reference))
                )
            _add_root_derivative_bounds(constraints, root_new, yaw_new, options)

        if level < FALLBACK_REGION:
            constraints.append(uv == source_uv)
        for index, occurrence in enumerate(occurrences):
            constraints.append(occurrence.polygon_a @ uv[index] <= occurrence.polygon_b + 1e-9)
            if level >= FALLBACK_RELAX:
                constraints.extend(
                    [
                        cp.norm(tangent[index], 2) <= options.max_tangential_relaxation,
                        occurrence.polygon_a @ (uv[index] + tangent[index])
                        <= occurrence.polygon_b + 1e-9,
                    ]
                )
            else:
                constraints.append(tangent[index] == 0.0)

        objective: cp.Expression = 0.0
        objective += options.root_deviation_weight * cp.sum_squares(root_new)
        objective += options.yaw_deviation_weight * cp.sum_squares(yaw_new)
        objective += _derivative_objective(
            root_new,
            options.root_velocity_weight,
            options.root_acceleration_weight,
            options.dt,
        )
        objective += _derivative_objective(
            cp.reshape(yaw_new, (len(reference), 1), order="C"),
            options.root_velocity_weight,
            options.root_acceleration_weight,
            options.dt,
        )
        objective += options.joint_reference_weight * cp.sum_squares(
            joints_new - reference[:, joint_indices]
        )
        if len(reference) > 1:
            objective += options.joint_velocity_weight * cp.sum_squares(
                cp.diff(joints_new, axis=0)
            )
        objective += options.contact_point_weight * cp.sum_squares(uv - source_uv)
        objective += options.contact_point_weight * cp.sum_squares(tangent)
        segment_occurrences: dict[tuple[str, int], list[int]] = {}
        for index, occurrence in enumerate(occurrences):
            if occurrence.key not in dropped:
                segment_occurrences.setdefault(occurrence.key, []).append(index)
        for indices in segment_occurrences.values():
            indices.sort(key=lambda index: occurrences[index].frame)
            for previous, current in zip(indices, indices[1:]):
                previous_uv = uv[previous] + tangent[previous]
                current_uv = uv[current] + tangent[current]
                objective += options.contact_point_temporal_weight * cp.sum_squares(
                    current_uv - previous_uv
                )
        objective += options.step_regularization_weight * (
            cp.sum_squares(dq) + cp.sum_squares(droot) + cp.sum_squares(dyaw)
        )

        for index, occurrence in enumerate(occurrences):
            if occurrence.key in dropped:
                constraints.extend(
                    [
                        elastic_position[index] == 0.0,
                        elastic_orientation[index] == 0.0,
                    ]
                )
                if full_slack is not None:
                    constraints.append(full_slack[index] == 0.0)
                if orientation_relax is not None:
                    constraints.append(orientation_relax[index] == 0.0)
                continue
            point, joint_jacobian, angular_jacobian, rotation_error = kinematics[index]
            frame = occurrence.frame
            root_position = configurations[frame, :3]
            yaw_jacobian = np.cross([0.0, 0.0, 1.0], point - root_position)
            linear_point = (
                point
                + joint_jacobian @ dq[frame]
                + droot[frame]
                + yaw_jacobian * dyaw[frame]
            )
            selected_uv = uv[index] + tangent[index]
            target = occurrence.world_origin + occurrence.world_basis @ selected_uv
            explicit_slack: cp.Expression | np.ndarray
            if full_slack is not None and occurrence.required:
                explicit_slack = full_slack[index]
            else:
                explicit_slack = np.zeros(3, dtype=float)
                if full_slack is not None:
                    constraints.append(full_slack[index] == 0.0)
            constraints.append(linear_point == target + explicit_slack + elastic_position[index])
            elastic_weight = options.contact_slack_weight * (
                100.0 if full_slack is not None else 1.0
            )
            objective += elastic_weight * cp.sum_squares(elastic_position[index])
            if full_slack is not None and occurrence.required:
                objective += options.contact_slack_weight * cp.sum_squares(full_slack[index])

            if rotation_error is None:
                constraints.append(elastic_orientation[index] == 0.0)
                if orientation_relax is not None:
                    constraints.append(orientation_relax[index] == 0.0)
                continue
            linear_rotation = (
                angular_jacobian @ dq[frame]
                + np.asarray([0.0, 0.0, 1.0]) * dyaw[frame]
            )
            if orientation_relax is None:
                orientation_value: cp.Expression | np.ndarray = np.zeros(3, dtype=float)
            else:
                orientation_value = orientation_relax[index]
                constraints.append(
                    cp.norm(orientation_relax[index], 2)
                    <= options.max_orientation_relaxation
                )
                # Rotation about the surface normal is the least important
                # contact DOF (for a palm it is in-plane twist).  Normal
                # alignment remains ten times more expensive.
                normal = occurrence.region.normal
                normal_component = normal @ orientation_relax[index]
                important_component = (
                    orientation_relax[index] - normal * normal_component
                )
                objective += options.orientation_slack_weight * (
                    cp.sum_squares(important_component)
                    + 0.1 * cp.square(normal_component)
                )
            constraints.append(
                linear_rotation + orientation_value + elastic_orientation[index]
                == rotation_error
            )
            objective += options.orientation_slack_weight * cp.sum_squares(
                elastic_orientation[index]
            )

        # Unused rows must be pinned or CVXPY can leave their values undefined.
        unused_orientation = set(range(len(occurrences))) - set(orientation_indices)
        for index in unused_orientation:
            if occurrences[index].key not in dropped:
                constraints.append(elastic_orientation[index] == 0.0)

        problem = cp.Problem(cp.Minimize(objective), constraints)
        try:
            # This problem intentionally contains N-D expressions.  CVXPY's
            # CPP canonicalizer cannot represent them and otherwise emits the
            # same benign fallback warning on every stage.  Selecting the
            # equivalent SCIPY canonicalizer explicitly changes no objective,
            # constraints, or numerical solver behavior.
            problem.solve(
                solver=options.solver,
                verbose=False,
                canon_backend=cp.SCIPY_CANON_BACKEND,
            )
        except cp.error.SolverError:
            problem.solve(
                solver="SCS",
                eps=1e-5,
                max_iters=20000,
                verbose=False,
                canon_backend=cp.SCIPY_CANON_BACKEND,
            )
        status = str(problem.status)
        objective_value = float(problem.value) if problem.value is not None else np.inf
        if status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or dq.value is None:
            break

        fraction = min(float(options.sqp_step_fraction), 1.0)
        state.joints = np.clip(
            state.joints + fraction * np.asarray(dq.value), lower, upper
        )
        state.root_translation += fraction * np.asarray(droot.value)
        state.root_yaw += fraction * np.asarray(dyaw.value)
        state.plane_coordinates += fraction * (
            np.asarray(uv.value) - state.plane_coordinates
        )
        state.tangential_slack += fraction * (
            np.asarray(tangent.value) - state.tangential_slack
        )

        (
            evaluated_configurations,
            contact_errors,
            orientation_errors,
            _,
            _,
        ) = _evaluate(
            model,
            reference,
            joint_indices,
            joint_dof_indices,
            occurrences,
            state,
            dropped,
            level,
        )
        if _strict_success(
            occurrences,
            contact_errors,
            orientation_errors,
            dropped,
            options,
            level,
            required_only=False,
        ):
            configurations = evaluated_configurations
            break

    (
        configurations,
        contact_errors,
        orientation_errors,
        contact_slack,
        orientation_slack,
    ) = _evaluate(
        model,
        reference,
        joint_indices,
        joint_dof_indices,
        occurrences,
        state,
        dropped,
        level,
    )
    strict_success = _strict_success(
        occurrences,
        contact_errors,
        orientation_errors,
        dropped,
        options,
        level,
        required_only=False,
    )
    return _StageOutcome(
        state=state,
        configurations=configurations,
        contact_errors=contact_errors,
        orientation_errors=orientation_errors,
        contact_slack=contact_slack,
        orientation_slack=orientation_slack,
        dropped=dropped,
        status=status,
        strict_success=strict_success,
        objective=objective_value,
    )


def _optional_drop_sets(keys: Sequence[tuple[str, int]], required: Mapping[tuple[str, int], bool]):
    optional = [key for key in keys if not required[key]]
    for count in range(1, len(optional) + 1):
        for selected in combinations(optional, count):
            yield frozenset(selected)


def _chosen_points(
    occurrences: Sequence[_Occurrence], state: _State, level: int
) -> tuple[np.ndarray, np.ndarray]:
    uv = state.plane_coordinates.copy()
    if level >= FALLBACK_RELAX:
        uv += state.tangential_slack
    world = np.asarray(
        [
            occurrence.world_origin + occurrence.world_basis @ uv[index]
            for index, occurrence in enumerate(occurrences)
        ],
        dtype=float,
    )
    return world, uv


def _build_result(
    occurrences: Sequence[_Occurrence],
    keys: Sequence[tuple[str, int]],
    outcome: _StageOutcome,
    level: int,
    options: FeasibilityOptions,
) -> InteractionFeasibilityResult:
    chosen_world, chosen_uv = _chosen_points(occurrences, outcome.state, level)
    segment_results: list[SegmentFeasibilityResult] = []
    for key in keys:
        indices = np.asarray(
            [index for index, occurrence in enumerate(occurrences) if occurrence.key == key],
            dtype=int,
        )
        sample = occurrences[int(indices[0])]
        frames = np.asarray([occurrences[index].frame for index in indices], dtype=np.int32)
        selected_world = chosen_world[indices]
        selected_uv = chosen_uv[indices]
        selected_local = _world_to_region_local(sample.region, selected_world)
        source_world = np.asarray([occurrences[index].source_world for index in indices])
        root_translation = outcome.state.root_translation[frames]
        root_yaw = outcome.state.root_yaw[frames]
        contact_error = outcome.contact_errors[indices]
        orientation_error = outcome.orientation_errors[indices]
        contact_slack = outcome.contact_slack[indices]
        orientation_slack = outcome.orientation_slack[indices]

        dropped = key in outcome.dropped
        point_shift = _max_row_norm(selected_world - source_world)
        root_shift = _max_row_norm(root_translation)
        yaw_shift = float(np.max(np.abs(root_yaw))) if len(root_yaw) else 0.0
        contact_miss = _max_row_norm(contact_error)
        orientation_miss = _max_row_norm(orientation_error)
        if dropped:
            fallback = FALLBACK_DROP_OPTIONAL
            status = DROPPED_OPTIONAL
            reason = ""
        else:
            orientation_limit = (
                options.max_orientation_relaxation + options.orientation_tolerance
                if level >= FALLBACK_RELAX
                else options.orientation_tolerance
            )
            if contact_miss > options.contact_tolerance or orientation_miss > orientation_limit:
                fallback = FALLBACK_REQUIRED_SLACK
                status = INFEASIBLE
                reason = (
                    "required_contact_residual_exceeds_tolerance: "
                    f"position={contact_miss:.9g}m, orientation={orientation_miss:.9g}rad; "
                    "bounded root warp, same-region contact selection, tangential/orientation "
                    "relaxation, and joint-limit-aware SQP were exhausted"
                )
            elif _max_row_norm(orientation_slack) > options.adaptation_tolerance or (
                level >= FALLBACK_RELAX
                and _max_row_norm(outcome.state.tangential_slack[indices])
                > options.adaptation_tolerance
            ):
                fallback = FALLBACK_RELAX
                status = ADAPTED
                reason = ""
            elif point_shift > options.adaptation_tolerance:
                fallback = FALLBACK_REGION
                status = ADAPTED
                reason = ""
            elif (
                root_shift > options.adaptation_tolerance
                or yaw_shift > options.adaptation_tolerance
            ):
                fallback = FALLBACK_ROOT
                status = ADAPTED
                reason = ""
            else:
                fallback = FALLBACK_NONE
                status = REACHABLE
                reason = ""
        segment_results.append(
            SegmentFeasibilityResult(
                effector_name=key[0],
                segment_id=key[1],
                frame_indices=frames,
                required=sample.required,
                region_id=sample.region.region_id,
                chosen_contact_points_world=selected_world.copy(),
                chosen_contact_points_region_local=selected_local.copy(),
                chosen_contact_plane_coordinates=selected_uv.copy(),
                source_contact_points_world=source_world.copy(),
                root_correction_translation=root_translation.copy(),
                root_correction_yaw=root_yaw.copy(),
                joint_configurations=outcome.configurations[frames].copy(),
                fallback_level=fallback,
                contact_slack=contact_slack.copy(),
                orientation_slack=orientation_slack.copy(),
                contact_errors=contact_error.copy(),
                orientation_errors=orientation_error.copy(),
                reachability_status=status,
                infeasible_reason=reason,
            )
        )
    return InteractionFeasibilityResult(
        configurations=outcome.configurations.copy(),
        root_correction_translation=outcome.state.root_translation.copy(),
        root_correction_yaw=outcome.state.root_yaw.copy(),
        segments=tuple(segment_results),
        solver_status=outcome.status,
        fallback_level_reached=level,
    )


def solve_interaction_feasibility(
    model: mujoco.MjModel,
    morphology: MorphologyReport,
    contact_tasks: Mapping[str, SceneContactTask],
    contact_regions: Mapping[Any, Any],
    root_reference_qpos: np.ndarray | None = None,
    *,
    required_contacts: bool | Mapping[Any, Any] = True,
    configured_joint_limits: Mapping[str, Any] | None = None,
    options: FeasibilityOptions | None = None,
) -> InteractionFeasibilityResult:
    """Solve the staged interaction-feasibility prepass.

    Parameters
    ----------
    model:
        Compiled robot model.  Only new :class:`mujoco.MjData` instances are
        mutated; model arrays and ``model.qpos0`` remain unchanged.
    morphology:
        Output of :func:`omniretargeting.morphology.analyze_robot_morphology`.
        Chain names must match ``contact_tasks`` keys.
    contact_tasks:
        Shared source schedule.  ``contact_modes`` and
        ``contact_segment_ids`` define the active runs; source arrays are read
        only.
    contact_regions:
        A region per run, keyed by ``(effector_name, segment_id)`` or by an
        effector name containing either one region or a nested id mapping.
    root_reference_qpos:
        Unwarped reference as ``(T,nq)``, ``(T,3)``, ``(nq,)``, or ``(3,)``.
    required_contacts:
        ``True`` by default.  A mapping with the same key conventions marks
        selected runs optional, enabling fallback level 4 explicitly.

    Returns
    -------
    InteractionFeasibilityResult
        Robot configurations, root correction, and a structured record per
        segment.  Required failure is never hidden: it has status
        ``"infeasible"``, nonzero residual/slack, and a reason string.
    """
    if not isinstance(model, mujoco.MjModel):
        raise TypeError("model must be a mujoco.MjModel.")
    if not isinstance(morphology, MorphologyReport):
        raise TypeError("morphology must be a MorphologyReport.")
    options = FeasibilityOptions() if options is None else options
    if not isinstance(options, FeasibilityOptions):
        raise TypeError("options must be a FeasibilityOptions instance.")

    occurrences, keys, num_frames = _build_occurrences(
        model,
        morphology,
        contact_tasks,
        contact_regions,
        required_contacts,
    )
    reference = _normalize_root_reference(model, num_frames, root_reference_qpos)
    joint_indices, joint_dof_indices, mechanical_lower, mechanical_upper = (
        _scalar_joint_coordinates(model)
    )
    lower, upper = _apply_configured_joint_limits(
        model,
        joint_indices,
        mechanical_lower,
        mechanical_upper,
        configured_joint_limits,
    )
    if np.any(lower > upper):
        raise ValueError("Configured joint limits have an empty intersection.")
    reference[:, joint_indices] = np.clip(reference[:, joint_indices], lower, upper)
    initial = _initial_state(
        model,
        morphology,
        reference,
        joint_indices,
        lower,
        upper,
        occurrences,
    )
    required_by_key = {
        key: next(occurrence.required for occurrence in occurrences if occurrence.key == key)
        for key in keys
    }

    outcome: _StageOutcome | None = None
    level_reached = FALLBACK_NONE
    # Carry the best local iterate forward so each fallback relaxes exactly
    # one restriction instead of restarting the nonlinear solve far away.
    state = initial
    for level in range(FALLBACK_NONE, FALLBACK_RELAX + 1):
        outcome = _solve_stage(
            model,
            reference,
            joint_indices,
            joint_dof_indices,
            lower,
            upper,
            occurrences,
            state,
            level,
            frozenset(),
            options,
        )
        level_reached = level
        state = outcome.state
        if outcome.strict_success:
            return _build_result(occurrences, keys, outcome, level_reached, options)

    optional_sets = list(_optional_drop_sets(keys, required_by_key))
    for dropped in optional_sets:
        candidate = _solve_stage(
            model,
            reference,
            joint_indices,
            joint_dof_indices,
            lower,
            upper,
            occurrences,
            state,
            FALLBACK_DROP_OPTIONAL,
            dropped,
            options,
        )
        level_reached = FALLBACK_DROP_OPTIONAL
        if candidate.strict_success:
            return _build_result(occurrences, keys, candidate, level_reached, options)
        if outcome is None or candidate.objective < outcome.objective:
            outcome = candidate

    all_optional = frozenset(key for key in keys if not required_by_key[key])
    outcome = _solve_stage(
        model,
        reference,
        joint_indices,
        joint_dof_indices,
        lower,
        upper,
        occurrences,
        state if outcome is None else outcome.state,
        FALLBACK_REQUIRED_SLACK,
        all_optional,
        options,
    )
    return _build_result(
        occurrences,
        keys,
        outcome,
        FALLBACK_REQUIRED_SLACK,
        options,
    )


# A descriptive alias for pipeline callers.
run_interaction_feasibility_prepass = solve_interaction_feasibility


__all__ = [
    "ADAPTED",
    "DROPPED_OPTIONAL",
    "FALLBACK_DROP_OPTIONAL",
    "FALLBACK_NONE",
    "FALLBACK_REGION",
    "FALLBACK_RELAX",
    "FALLBACK_REQUIRED_SLACK",
    "FALLBACK_ROOT",
    "FeasibilityOptions",
    "INFEASIBLE",
    "InteractionFeasibilityResult",
    "REACHABLE",
    "SegmentFeasibilityResult",
    "run_interaction_feasibility_prepass",
    "solve_interaction_feasibility",
]
