"""Robot-agnostic kinematic morphology and contact-patch measurement.

This module measures each limb independently from a compiled MuJoCo model.  It
does not infer one whole-body scale and apply it to every limb: arm, leg, and
pelvis-support chains retain their own geometry, joint ranges, and reachable
radius.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import mujoco
import numpy as np
from scipy.stats import qmc

from .retargeting import _apply_configured_joint_limits
from .trajectory_refinement import (
    _scalar_joint_coordinates,
    collision_geom_ids,
    geom_surface_offsets,
    geom_surface_points_in_body_frame,
)


@dataclass(frozen=True)
class ResolvedChainSpec:
    """Names and local offsets needed to measure one semantic endpoint."""

    name: str
    endpoint_body: str
    reference_body: str
    endpoint_offset: tuple[float, float, float]
    reference_offset: tuple[float, float, float]
    configured_reach: float | None = None


@dataclass(frozen=True)
class LinkSegment:
    """Nominal qpos0 distance between consecutive points on a body path."""

    parent_body: str
    child_body: str
    length: float


@dataclass(frozen=True)
class JointRange:
    """Mechanical and task-safe bounds for one joint on a chain."""

    joint_id: int
    name: str
    joint_type: str
    mechanical_range: tuple[float, float]
    effective_range: tuple[float, float]
    sampled_for_reach: bool


@dataclass(frozen=True)
class ChainMorphology:
    """Measured geometry and sampled workspace radius of one chain."""

    name: str
    reference_body: str
    endpoint_body: str
    body_ids: tuple[int, ...]
    body_names: tuple[str, ...]
    segments: tuple[LinkSegment, ...]
    nominal_length: float
    joints: tuple[JointRange, ...]
    qpos0_reach: float
    actual_reach: float
    origin_root_local: np.ndarray
    reference_offset: np.ndarray
    endpoint_offset: np.ndarray


@dataclass(frozen=True)
class ContactPatch:
    """Collision-shell samples describing a semantic robot contact patch.

    ``surface_offsets`` and ``surface_normals`` are paired and expressed in
    the owning body's frame.  ``extent`` is the full body-frame axis-aligned
    span of the selected collision geoms, so it remains meaningful even when
    a normal-direction filter leaves one face-centre sample.  Optional local
    offsets from a profile are retained separately as explicit overrides; they
    never replace or masquerade as samples from the collision shell.
    """

    name: str
    body_id: int
    body_name: str
    geom_ids: tuple[int, ...]
    surface_offsets: np.ndarray
    surface_normals: np.ndarray
    extent: np.ndarray
    preferred_normal: np.ndarray | None
    local_offset_overrides: np.ndarray


@dataclass(frozen=True)
class MorphologyReport:
    """Independent per-chain morphology and collision-derived contact patches."""

    root_body_id: int
    root_body_name: str
    chains: dict[str, ChainMorphology]
    contact_patches: dict[str, ContactPatch]


def _as_offset(value: Any, field: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must be a finite three-vector in metres.")
    return tuple(float(component) for component in array)


def _body_id(model: mujoco.MjModel, name: str, field: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(name))
    if body_id < 0:
        raise ValueError(f"{field} refers to missing MuJoCo body {name!r}.")
    return int(body_id)


def resolve_end_effector_chain_specs(
    model: mujoco.MjModel,
    end_effector_config: Mapping[str, Any],
    source_to_robot_bodies: Mapping[str, str] | None = None,
) -> dict[str, ResolvedChainSpec]:
    """Resolve the existing end-effector profile schema into chain specs.

    This is the single configuration-resolution path used by the legacy core
    adapter and the morphology analyzer.  It preserves the existing shoulder
    and hip fallback rules, tracking/reference offsets, and configured reach
    value.  Keeping the configured value in the resolved record preserves the
    legacy adapter's explicit-override contract; the morphology report itself
    always exposes independently measured reach.
    """
    resolved: dict[str, ResolvedChainSpec] = {}
    specs = dict(end_effector_config.get("effectors", {}) or {})
    joint_mapping = dict(source_to_robot_bodies or {})
    for name, raw_spec in specs.items():
        spec = dict(raw_spec or {})
        endpoint_body = str(spec.get("robot_body", ""))
        if mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, endpoint_body
        ) < 0:
            continue

        reference_body = spec.get("reach_reference_body")
        if reference_body is None:
            reference_body = dict(spec.get("arm_posture", {}) or {}).get(
                "shoulder_body"
            )
        if reference_body is None and name.endswith("foot"):
            source_hip = "L_Hip" if name.startswith("left") else "R_Hip"
            reference_body = joint_mapping.get(source_hip)
        if reference_body is None:
            raise ValueError(
                f"End-effector {name!r} has no reach_reference_body; a "
                "shoulder or hip chain origin is required for morphology."
            )
        reference_body = str(reference_body)
        _body_id(model, reference_body, f"effectors[{name!r}].reach_reference_body")

        configured_reach = spec.get(
            "max_contact_reach", spec.get("max_swing_reach")
        )
        resolved[name] = ResolvedChainSpec(
            name=str(name),
            endpoint_body=endpoint_body,
            reference_body=reference_body,
            endpoint_offset=_as_offset(
                spec.get("tracking_offset", [0.0, 0.0, 0.0]),
                f"effectors[{name!r}].tracking_offset",
            ),
            reference_offset=_as_offset(
                spec.get("reach_reference_offset", [0.0, 0.0, 0.0]),
                f"effectors[{name!r}].reach_reference_offset",
            ),
            configured_reach=(
                None if configured_reach is None else float(configured_reach)
            ),
        )
    return resolved


def _coerce_chain_spec(name: str, raw_spec: Any) -> ResolvedChainSpec:
    if isinstance(raw_spec, ResolvedChainSpec):
        if raw_spec.name == name:
            return raw_spec
        return ResolvedChainSpec(
            name=name,
            endpoint_body=raw_spec.endpoint_body,
            reference_body=raw_spec.reference_body,
            endpoint_offset=raw_spec.endpoint_offset,
            reference_offset=raw_spec.reference_offset,
            configured_reach=raw_spec.configured_reach,
        )
    spec = dict(raw_spec or {})
    endpoint_body = spec.get("endpoint_body", spec.get("robot_body"))
    reference_body = spec.get("reference_body", spec.get("reach_reference_body"))
    if endpoint_body is None or reference_body is None:
        raise ValueError(
            f"chains[{name!r}] must define reference_body and endpoint_body."
        )
    configured_reach = spec.get(
        "configured_reach",
        spec.get("max_contact_reach", spec.get("max_swing_reach")),
    )
    return ResolvedChainSpec(
        name=name,
        endpoint_body=str(endpoint_body),
        reference_body=str(reference_body),
        endpoint_offset=_as_offset(
            spec.get("endpoint_offset", spec.get("tracking_offset", [0, 0, 0])),
            f"chains[{name!r}].endpoint_offset",
        ),
        reference_offset=_as_offset(
            spec.get(
                "reference_offset", spec.get("reach_reference_offset", [0, 0, 0])
            ),
            f"chains[{name!r}].reference_offset",
        ),
        configured_reach=(
            None if configured_reach is None else float(configured_reach)
        ),
    )


def _ancestor_path(
    model: mujoco.MjModel, reference_body_id: int, endpoint_body_id: int
) -> tuple[int, ...]:
    """Return reference-to-endpoint body ids using MuJoCo's parent array."""
    reverse_path: list[int] = []
    body_id = endpoint_body_id
    while body_id > 0:
        reverse_path.append(body_id)
        if body_id == reference_body_id:
            return tuple(reversed(reverse_path))
        body_id = int(model.body_parentid[body_id])
    if reference_body_id == 0 and reverse_path:
        return (0, *reversed(reverse_path))
    reference_name = model.body(reference_body_id).name
    endpoint_name = model.body(endpoint_body_id).name
    raise ValueError(
        f"Reference body {reference_name!r} is not an ancestor of endpoint "
        f"body {endpoint_name!r}."
    )


def _top_level_body(model: mujoco.MjModel, body_id: int) -> int:
    while int(model.body_parentid[body_id]) != 0:
        body_id = int(model.body_parentid[body_id])
    return int(body_id)


def _joint_type_name(joint_type: int) -> str:
    names = {
        int(mujoco.mjtJoint.mjJNT_FREE): "free",
        int(mujoco.mjtJoint.mjJNT_BALL): "ball",
        int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
        int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
    }
    return names.get(int(joint_type), f"unknown({int(joint_type)})")


def _chain_joint_ids(model: mujoco.MjModel, body_path: tuple[int, ...]) -> tuple[int, ...]:
    joint_ids: list[int] = []
    for body_id in body_path:
        start = int(model.body_jntadr[body_id])
        count = int(model.body_jntnum[body_id])
        joint_ids.extend(range(start, start + count))
    return tuple(joint_ids)


def _effective_scalar_ranges(
    model: mujoco.MjModel,
    configured_joint_limits: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qpos_indices, _, mechanical_lower, mechanical_upper = _scalar_joint_coordinates(
        model
    )
    effective_lower, effective_upper = _apply_configured_joint_limits(
        model,
        qpos_indices,
        mechanical_lower,
        mechanical_upper,
        configured_joint_limits,
    )
    return qpos_indices, effective_lower, effective_upper


def _joint_ranges(
    model: mujoco.MjModel,
    joint_ids: tuple[int, ...],
    scalar_qpos_indices: np.ndarray,
    effective_lower: np.ndarray,
    effective_upper: np.ndarray,
) -> tuple[JointRange, ...]:
    coordinate_by_qpos = {
        int(qpos_index): coordinate
        for coordinate, qpos_index in enumerate(scalar_qpos_indices)
    }
    ranges: list[JointRange] = []
    for joint_id in joint_ids:
        joint_type = int(model.jnt_type[joint_id])
        limited = bool(model.jnt_limited[joint_id])
        if limited:
            mechanical = tuple(float(x) for x in model.jnt_range[joint_id])
        else:
            mechanical = (-np.inf, np.inf)
        qpos_index = int(model.jnt_qposadr[joint_id])
        coordinate = coordinate_by_qpos.get(qpos_index)
        if coordinate is None:
            effective = mechanical
        else:
            effective = (
                float(effective_lower[coordinate]),
                float(effective_upper[coordinate]),
            )
        ranges.append(
            JointRange(
                joint_id=joint_id,
                name=model.joint(joint_id).name or f"joint_{joint_id}",
                joint_type=_joint_type_name(joint_type),
                mechanical_range=mechanical,
                effective_range=effective,
                sampled_for_reach=coordinate is not None,
            )
        )
    return tuple(ranges)


def _candidate_joint_coordinates(
    qpos0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    sample_count: int,
) -> np.ndarray:
    """Build deterministic limit, axis, and Sobol samples for scalar joints."""
    dimension = len(qpos0)
    if dimension == 0:
        return np.zeros((1, 0), dtype=float)
    finite_lower = lower.copy()
    finite_upper = upper.copy()
    # An unlimited hinge has a periodic workspace, so one full revolution is
    # sufficient.  An unlimited slide is handled before this helper because
    # its translational reach is genuinely unbounded.
    finite_lower = np.where(np.isfinite(finite_lower), finite_lower, qpos0 - np.pi)
    finite_upper = np.where(np.isfinite(finite_upper), finite_upper, qpos0 + np.pi)
    clipped_qpos0 = np.clip(qpos0, finite_lower, finite_upper)
    midpoint = 0.5 * (finite_lower + finite_upper)
    candidates = [clipped_qpos0, finite_lower, finite_upper, midpoint]
    for coordinate in range(dimension):
        at_lower = clipped_qpos0.copy()
        at_upper = clipped_qpos0.copy()
        at_lower[coordinate] = finite_lower[coordinate]
        at_upper[coordinate] = finite_upper[coordinate]
        candidates.extend([at_lower, at_upper])

    remaining = max(int(sample_count) - len(candidates), 0)
    if remaining:
        exponent = int(np.ceil(np.log2(remaining)))
        unit = qmc.Sobol(d=dimension, scramble=False).random_base2(exponent)
        candidates.extend(
            finite_lower + unit[:remaining] * (finite_upper - finite_lower)
        )
    return np.unique(np.asarray(candidates, dtype=float), axis=0)


def _sampled_reach(
    model: mujoco.MjModel,
    reference_body_id: int,
    endpoint_body_id: int,
    reference_offset: np.ndarray,
    endpoint_offset: np.ndarray,
    joint_ids: tuple[int, ...],
    scalar_qpos_indices: np.ndarray,
    effective_lower: np.ndarray,
    effective_upper: np.ndarray,
    sample_count: int,
) -> tuple[float, float]:
    """Estimate maximum radial reach over effective scalar-joint ranges.

    The estimator runs ``mj_forward`` at qpos0, every per-coordinate limit,
    collective lower/upper/midpoint poses, and a deterministic Sobol grid over
    only scalar joints attached to the reference-body -> endpoint ancestor
    path.  Mechanical bounds are intersected with configured task-safe bounds
    before sampling.  The result is therefore a *lower bound* on the true
    maximum reach, not an exact workspace proof or an IK/collision-feasibility
    certificate.  Ball joints remain at qpos0; an unlimited slide makes radial
    reach unbounded and is reported as ``inf``.
    """
    coordinate_by_qpos = {
        int(qpos_index): coordinate
        for coordinate, qpos_index in enumerate(scalar_qpos_indices)
    }
    selected_coordinates: list[int] = []
    selected_qpos: list[int] = []
    for joint_id in joint_ids:
        qpos_index = int(model.jnt_qposadr[joint_id])
        coordinate = coordinate_by_qpos.get(qpos_index)
        if coordinate is None:
            continue
        if (
            int(model.jnt_type[joint_id]) == mujoco.mjtJoint.mjJNT_SLIDE
            and (
                not np.isfinite(effective_lower[coordinate])
                or not np.isfinite(effective_upper[coordinate])
            )
        ):
            return _reach_at_qpos(
                model,
                np.asarray(model.qpos0, dtype=float),
                reference_body_id,
                endpoint_body_id,
                reference_offset,
                endpoint_offset,
            ), np.inf
        selected_coordinates.append(coordinate)
        selected_qpos.append(qpos_index)

    base_qpos = np.asarray(model.qpos0, dtype=float).copy()
    qpos0_reach = _reach_at_qpos(
        model,
        base_qpos,
        reference_body_id,
        endpoint_body_id,
        reference_offset,
        endpoint_offset,
    )
    if not selected_coordinates:
        return qpos0_reach, qpos0_reach

    coordinates = np.asarray(selected_coordinates, dtype=int)
    qpos_indices = np.asarray(selected_qpos, dtype=int)
    candidates = _candidate_joint_coordinates(
        base_qpos[qpos_indices],
        effective_lower[coordinates],
        effective_upper[coordinates],
        sample_count,
    )
    data = mujoco.MjData(model)
    # qpos0 is a diagnostic pose, not automatically a feasible sample: a
    # configured safe interval may intentionally exclude it.
    maximum = 0.0
    for candidate in candidates:
        data.qpos[:] = base_qpos
        data.qpos[qpos_indices] = candidate
        mujoco.mj_forward(model, data)
        origin = data.xpos[reference_body_id] + (
            data.xmat[reference_body_id].reshape(3, 3) @ reference_offset
        )
        endpoint = data.xpos[endpoint_body_id] + (
            data.xmat[endpoint_body_id].reshape(3, 3) @ endpoint_offset
        )
        maximum = max(maximum, float(np.linalg.norm(endpoint - origin)))
    return qpos0_reach, maximum


def _reach_at_qpos(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    reference_body_id: int,
    endpoint_body_id: int,
    reference_offset: np.ndarray,
    endpoint_offset: np.ndarray,
) -> float:
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(qpos, dtype=float)
    mujoco.mj_forward(model, data)
    origin = data.xpos[reference_body_id] + (
        data.xmat[reference_body_id].reshape(3, 3) @ reference_offset
    )
    endpoint = data.xpos[endpoint_body_id] + (
        data.xmat[endpoint_body_id].reshape(3, 3) @ endpoint_offset
    )
    return float(np.linalg.norm(endpoint - origin))


def _measure_chain(
    model: mujoco.MjModel,
    spec: ResolvedChainSpec,
    root_body_id: int,
    scalar_qpos_indices: np.ndarray,
    effective_lower: np.ndarray,
    effective_upper: np.ndarray,
    reach_samples: int,
) -> ChainMorphology:
    reference_body_id = _body_id(
        model, spec.reference_body, f"chains[{spec.name!r}].reference_body"
    )
    endpoint_body_id = _body_id(
        model, spec.endpoint_body, f"chains[{spec.name!r}].endpoint_body"
    )
    body_path = _ancestor_path(model, reference_body_id, endpoint_body_id)
    joint_ids = _chain_joint_ids(model, body_path)
    joint_ranges = _joint_ranges(
        model,
        joint_ids,
        scalar_qpos_indices,
        effective_lower,
        effective_upper,
    )

    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(model.qpos0, dtype=float)
    mujoco.mj_forward(model, data)
    reference_offset = np.asarray(spec.reference_offset, dtype=float)
    endpoint_offset = np.asarray(spec.endpoint_offset, dtype=float)
    origin = data.xpos[reference_body_id] + (
        data.xmat[reference_body_id].reshape(3, 3) @ reference_offset
    )
    endpoint = data.xpos[endpoint_body_id] + (
        data.xmat[endpoint_body_id].reshape(3, 3) @ endpoint_offset
    )

    path_points = [origin]
    point_names = [spec.reference_body]
    for body_id in body_path[1:]:
        path_points.append(data.xpos[body_id].copy())
        point_names.append(model.body(body_id).name)
    if not np.allclose(path_points[-1], endpoint) or len(path_points) == 1:
        path_points.append(endpoint)
        point_names.append(f"{spec.endpoint_body}:tracking_point")
    segments = tuple(
        LinkSegment(
            parent_body=point_names[index],
            child_body=point_names[index + 1],
            length=float(np.linalg.norm(path_points[index + 1] - path_points[index])),
        )
        for index in range(len(path_points) - 1)
    )

    root_rotation = data.xmat[root_body_id].reshape(3, 3)
    origin_root_local = root_rotation.T @ (origin - data.xpos[root_body_id])
    qpos0_reach, actual_reach = _sampled_reach(
        model,
        reference_body_id,
        endpoint_body_id,
        reference_offset,
        endpoint_offset,
        joint_ids,
        scalar_qpos_indices,
        effective_lower,
        effective_upper,
        reach_samples,
    )
    return ChainMorphology(
        name=spec.name,
        reference_body=spec.reference_body,
        endpoint_body=spec.endpoint_body,
        body_ids=body_path,
        body_names=tuple(model.body(body_id).name for body_id in body_path),
        segments=segments,
        nominal_length=float(sum(segment.length for segment in segments)),
        joints=joint_ranges,
        qpos0_reach=qpos0_reach,
        actual_reach=actual_reach,
        origin_root_local=origin_root_local.copy(),
        reference_offset=reference_offset.copy(),
        endpoint_offset=endpoint_offset.copy(),
    )


def _geom_local_surface_normals(model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    """Return candidate normals paired with ``geom_surface_offsets`` samples."""
    offsets = geom_surface_offsets(model, geom_id)
    if not len(offsets):
        return np.zeros((0, 3), dtype=float)
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    normals = np.zeros_like(offsets)
    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        normals = offsets
    elif geom_type in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        normals = offsets / np.maximum(size[:3] ** 2, 1e-12)
    elif geom_type in (
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        mujoco.mjtGeom.mjGEOM_CYLINDER,
    ):
        half_length = float(size[1])
        if geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            cap_centres = np.column_stack(
                [
                    np.zeros(len(offsets)),
                    np.zeros(len(offsets)),
                    np.clip(offsets[:, 2], -half_length, half_length),
                ]
            )
            normals = offsets - cap_centres
        else:
            cap_centres = np.linalg.norm(offsets[:, :2], axis=1) < 1e-10
            normals[:, :2] = offsets[:, :2]
            normals[cap_centres, 2] = np.sign(offsets[cap_centres, 2])
    else:
        # Mesh vertices do not carry a one-to-one face normal in MuJoCo's
        # compact model arrays.  Radial directions are deterministic candidate
        # normals, not claims about the exact interpolated mesh normal.
        normals = offsets
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.where(lengths > 1e-12, normals / np.maximum(lengths, 1e-12), 0.0)


def _geom_normals_in_body_frame(model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    local_normals = _geom_local_surface_normals(model, geom_id)
    geom_quat = np.asarray(model.geom_quat[geom_id], dtype=float)
    rotation = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(rotation, geom_quat)
    return local_normals @ rotation.reshape(3, 3).T


def _descends_from(model: mujoco.MjModel, body_id: int, ancestor_id: int) -> bool:
    while body_id > 0:
        if body_id == ancestor_id:
            return True
        body_id = int(model.body_parentid[body_id])
    return ancestor_id == 0


def _measure_contact_patch(
    model: mujoco.MjModel, name: str, raw_spec: Mapping[str, Any]
) -> ContactPatch:
    spec = dict(raw_spec or {})
    body_name = spec.get("body", spec.get("robot_body"))
    if body_name is None:
        raise ValueError(f"contact_patches[{name!r}] must define body.")
    body_name = str(body_name)
    body_id = _body_id(model, body_name, f"contact_patches[{name!r}].body")
    tokens = tuple(
        str(token).lower()
        for token in spec.get("geom_tokens", ())
        if str(token)
    )
    include_descendants = bool(spec.get("include_descendants", False))
    candidate_geom_ids = collision_geom_ids(
        model, geom_groups=spec.get("geom_groups")
    )
    selected_geom_ids: list[int] = []
    for geom_id in candidate_geom_ids:
        geom_body_id = int(model.geom_bodyid[geom_id])
        on_body = (
            _descends_from(model, geom_body_id, body_id)
            if include_descendants
            else geom_body_id == body_id
        )
        if not on_body:
            continue
        geom_name = model.geom(int(geom_id)).name or ""
        geom_body_name = model.body(geom_body_id).name or ""
        if tokens and not any(
            token in geom_name.lower() or token in geom_body_name.lower()
            for token in tokens
        ):
            continue
        selected_geom_ids.append(int(geom_id))
    if not selected_geom_ids:
        raise ValueError(
            f"contact_patches[{name!r}] resolved no collision geoms on "
            f"body {body_name!r}."
        )

    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(model.qpos0, dtype=float)
    mujoco.mj_forward(model, data)
    patch_rotation = data.xmat[body_id].reshape(3, 3)
    offsets_by_geom: list[np.ndarray] = []
    normals_by_geom: list[np.ndarray] = []
    for geom_id in selected_geom_ids:
        owner_id = int(model.geom_bodyid[geom_id])
        owner_offsets = geom_surface_points_in_body_frame(model, geom_id)
        owner_normals = _geom_normals_in_body_frame(model, geom_id)
        if owner_id == body_id:
            offsets_by_geom.append(owner_offsets)
            normals_by_geom.append(owner_normals)
            continue
        owner_rotation = data.xmat[owner_id].reshape(3, 3)
        world_points = data.xpos[owner_id] + owner_offsets @ owner_rotation.T
        world_normals = owner_normals @ owner_rotation.T
        offsets_by_geom.append(
            (world_points - data.xpos[body_id]) @ patch_rotation
        )
        normals_by_geom.append(world_normals @ patch_rotation)
    nonempty = [index for index, offsets in enumerate(offsets_by_geom) if len(offsets)]
    if not nonempty:
        raise ValueError(
            f"contact_patches[{name!r}] collision geoms provide no finite "
            "surface samples."
        )
    all_offsets = np.vstack([offsets_by_geom[index] for index in nonempty])
    all_normals = np.vstack([normals_by_geom[index] for index in nonempty])
    preferred_normal = spec.get("normal", spec.get("preferred_normal"))
    normal_array: np.ndarray | None = None
    selected = np.ones(len(all_offsets), dtype=bool)
    if preferred_normal is not None:
        normal_array = np.asarray(
            _as_offset(preferred_normal, f"contact_patches[{name!r}].normal")
        )
        normal_length = float(np.linalg.norm(normal_array))
        if normal_length < 1e-12:
            raise ValueError(
                f"contact_patches[{name!r}].normal must be non-zero."
            )
        normal_array /= normal_length
        alignment = all_normals @ normal_array
        threshold = float(spec.get("normal_alignment", 0.5))
        selected = alignment >= threshold
        if not np.any(selected):
            selected[int(np.argmax(alignment))] = True

    raw_overrides = spec.get(
        "local_offset_overrides", spec.get("contact_offsets", [])
    )
    overrides = np.asarray(raw_overrides, dtype=float)
    if overrides.size == 0:
        overrides = np.zeros((0, 3), dtype=float)
    else:
        overrides = overrides.reshape(-1, 3).copy()
        if not np.all(np.isfinite(overrides)):
            raise ValueError(
                f"contact_patches[{name!r}].local_offset_overrides must be finite."
            )
    return ContactPatch(
        name=name,
        body_id=body_id,
        body_name=body_name,
        geom_ids=tuple(selected_geom_ids),
        surface_offsets=all_offsets[selected].copy(),
        surface_normals=all_normals[selected].copy(),
        extent=np.ptp(all_offsets, axis=0),
        preferred_normal=None if normal_array is None else normal_array.copy(),
        local_offset_overrides=overrides,
    )


def analyze_robot_morphology(
    model: mujoco.MjModel,
    semantic_mapping: Mapping[str, Any],
    *,
    configured_joint_limits: Mapping[str, Any] | None = None,
    reach_samples: int = 512,
) -> MorphologyReport:
    """Measure independent robot chains and collision-derived contact patches.

    ``semantic_mapping`` contains ``chains`` keyed by semantic names such as
    ``left_hand``, ``right_hand``, ``left_foot``, ``right_foot``, and
    ``pelvis_support``.  Each entry supplies a reference shoulder/hip/support
    body, endpoint body, and optional metric local offsets.  Actual body paths
    always come from MuJoCo's ``body_parentid`` walk.  A top-level ``root_body``
    controls the frame used for chain origins; when omitted, the highest
    non-world ancestor of the first chain origin is used.

    No model array or input mapping is mutated.  Reach is measured separately
    for each chain and no total-height ratio is computed.
    """
    raw_chains = dict(semantic_mapping.get("chains", {}) or {})
    if not raw_chains:
        raise ValueError("semantic_mapping.chains must contain at least one chain.")
    specs = {
        str(name): _coerce_chain_spec(str(name), raw_spec)
        for name, raw_spec in raw_chains.items()
    }
    first_reference_id = _body_id(
        model,
        next(iter(specs.values())).reference_body,
        "semantic_mapping.chains reference_body",
    )
    root_body_name = semantic_mapping.get("root_body")
    root_body_id = (
        _top_level_body(model, first_reference_id)
        if root_body_name is None
        else _body_id(model, str(root_body_name), "semantic_mapping.root_body")
    )
    root_body_name = model.body(root_body_id).name

    scalar_qpos_indices, effective_lower, effective_upper = _effective_scalar_ranges(
        model, configured_joint_limits
    )
    chains = {
        name: _measure_chain(
            model,
            spec,
            root_body_id,
            scalar_qpos_indices,
            effective_lower,
            effective_upper,
            reach_samples,
        )
        for name, spec in specs.items()
    }
    contact_patches = {
        str(name): _measure_contact_patch(model, str(name), dict(raw_spec or {}))
        for name, raw_spec in dict(
            semantic_mapping.get("contact_patches", {}) or {}
        ).items()
    }
    return MorphologyReport(
        root_body_id=root_body_id,
        root_body_name=root_body_name,
        chains=chains,
        contact_patches=contact_patches,
    )


def per_chain_ratios(
    numerator: MorphologyReport,
    denominator: MorphologyReport,
    *,
    measure: str = "nominal_length",
) -> dict[str, float]:
    """Return independent same-semantic-chain ratios between two robots."""
    if measure not in {"nominal_length", "actual_reach"}:
        raise ValueError("measure must be 'nominal_length' or 'actual_reach'.")
    shared = numerator.chains.keys() & denominator.chains.keys()
    ratios: dict[str, float] = {}
    for name in shared:
        divisor = float(getattr(denominator.chains[name], measure))
        if divisor <= 0.0:
            raise ValueError(f"Cannot ratio zero-length chain {name!r}.")
        ratios[name] = float(getattr(numerator.chains[name], measure)) / divisor
    return ratios
