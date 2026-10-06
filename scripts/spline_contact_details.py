"""Framewise contact details for the structural spline quality report.

The arrays returned here are observations against the immutable prepared task
plan.  In particular, signed gaps are distances to observed task planes and
are not certificates that the original collision geometry is intersection-free.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping

import mujoco
import numpy as np



ENDPOINT_VECTOR = "endpoint_contact_vector_error_m"
ENDPOINT_NORMAL = "endpoint_contact_normal_abs_error_m"
ENDPOINT_GAP = "endpoint_signed_gap_to_observed_plane_m"
ENDPOINT_SLIP = "endpoint_sticking_target_relative_slip_m_s"
SURFACE_VECTOR = "surface_contact_vector_error_m"
SURFACE_NORMAL = "surface_contact_normal_abs_error_m"
SURFACE_GAP = "surface_signed_gap_to_observed_plane_m"
SURFACE_SLIP = "surface_sticking_target_relative_slip_m_s"
ROOT_DRIFT = "source_root_anatomical_correspondence_error_m"


def _input_dictionary(inputs):
    if isinstance(inputs, Mapping):
        return inputs
    path = Path(inputs)
    loaded = np.load(path, allow_pickle=True)
    return loaded.item() if path.suffix == ".npy" else dict(loaded)


def _frame_series(value, count, start, name, dtype=float):
    value = np.asarray(value)
    if value.ndim == 0 or len(value) < start + count:
        raise ValueError(f"{name} must cover absolute frames [{start}, {start + count})")
    return np.asarray(value[start:start + count], dtype=dtype)


def _new_part_rows(metrics, metric, parts, count):
    return metrics.setdefault(metric, {part: np.full(count, np.nan) for part in parts})


def _worst_binding(values, signed=False):
    """Select the largest defect, retaining sign for a signed gap."""
    finite = np.isfinite(values)
    if not finite.any():
        return np.nan, -1
    index = np.argmax(np.abs(values[finite]))
    binding = np.flatnonzero(finite)[index]
    return float(values[binding]), int(binding)


def _endpoint_rows(inputs, model, poses, choices, start, metrics, result, transforms, fps):
    count = len(poses)
    parts = list(inputs.get("tasks", {}))
    for metric in (ENDPOINT_VECTOR, ENDPOINT_NORMAL, ENDPOINT_GAP, ENDPOINT_SLIP):
        _new_part_rows(metrics, metric, parts, count)

    for name, task in inputs.get("tasks", {}).items():
        spec = inputs["effectors"]["effectors"][name]
        body = int(model.body(spec["robot_body"]).id)
        choice = _frame_series(choices[name], count, start, f"choice {name}", int)
        contact = np.asarray(task.contact_mask)
        managed = np.asarray(task.managed_frames)
        modes = (np.asarray(task.contact_modes)
                 if task.contact_modes is not None else None)
        environment = (np.asarray(task.environment_ids)
                       if task.environment_ids is not None else None)
        offsets = np.asarray(spec["contact_offsets"], float)
        applicable = contact[start:start + count] & ~managed[start:start + count]
        points = np.full((count, 3), np.nan)
        targets = np.asarray(task.contact_positions)[start:start + count]
        planes = np.asarray(task.surface_points)[start:start + count]
        normals = np.asarray(task.surface_normals)[start:start + count]

        for local in range(count):
            if not applicable[local]:
                continue
            rotation = transforms["body_xmat"][local, body]
            points[local] = (transforms["body_xpos"][local, body]
                             + rotation @ offsets[int(choice[local])])

        delta = points - targets
        metrics[ENDPOINT_VECTOR][name][applicable] = np.linalg.norm(
            delta[applicable], axis=1)
        normal_length = np.linalg.norm(normals, axis=1)
        has_normal = applicable & (normal_length > 1e-12)
        metrics[ENDPOINT_NORMAL][name][has_normal] = np.abs(
            np.einsum("fi,fi->f", delta[has_normal],
                      normals[has_normal] / normal_length[has_normal, None]))
        metrics[ENDPOINT_GAP][name][has_normal] = np.einsum(
            "fi,fi->f", points[has_normal] - planes[has_normal],
            normals[has_normal] / normal_length[has_normal, None])

        signals = result["binding_signals"]["endpoint"]
        for offset_index, offset in enumerate(offsets):
            fixed_points = (
                transforms["body_xpos"][:, body]
                + np.einsum("fij,j->fi", transforms["body_xmat"][:, body], offset)
            )
            signals.append({
                "id": f"endpoint:{name}:offset_{offset_index}",
                "kind": "endpoint", "part": name,
                "binding_index": int(offset_index),
                "world_positions": fixed_points,
                "active": applicable & (choice == offset_index),
                "source_world_positions": None,
                "units": "m",
                "provenance": (
                    "fixed prepared endpoint contact offset evaluated by full-frame FK; "
                    "active is unmanaged contact with this offset choice"
                ),
                "source_provenance": (
                    "unverified: prepared endpoint tasks have no distinct full-frame world "
                    "source position for each fixed contact offset; target is not substituted"
                ),
            })
        if modes is None:
            continue
        sticking = applicable & (modes[start:start + count] == "sticking")
        continuation = sticking[1:] & sticking[:-1]
        if environment is not None:
            environment_ids = environment[start:start + count]
            continuation &= environment_ids[1:] == environment_ids[:-1]
        velocity = np.linalg.norm(np.diff(points - targets, axis=0), axis=1) * fps
        metrics[ENDPOINT_SLIP][name][1:][continuation] = velocity[continuation]
    return metrics


def _surface_rows(inputs, model, poses, start, metrics, result, transforms, fps):
    surface = inputs.get("surface", {})
    count = len(poses)
    bindings = np.asarray(surface.get("surface_contact_parts", ()), dtype=object)
    parts = [str(part) for part in dict.fromkeys(bindings.tolist())]
    for metric in (SURFACE_VECTOR, SURFACE_NORMAL, SURFACE_GAP, SURFACE_SLIP):
        _new_part_rows(metrics, metric, parts, count)
    if not len(bindings):
        return

    required = (
        "surface_contact_mask", "surface_contact_robot_geoms",
        "surface_contact_robot_local_points", "surface_contact_targets_w",
        "surface_contact_normals_w", "surface_contact_environment_w",
        "surface_contact_source_vectors_w",
        "surface_contact_modes", "surface_contact_surface_ids")
    missing = [name for name in required if name not in surface]
    if missing:
        raise ValueError(f"prepared surface plan is missing {missing}")

    active = _frame_series(surface["surface_contact_mask"], count, start,
                           "surface_contact_mask", bool)
    geoms = np.asarray(surface["surface_contact_robot_geoms"], int)
    local = np.asarray(surface["surface_contact_robot_local_points"], float)
    targets = _frame_series(surface["surface_contact_targets_w"], count, start,
                            "surface_contact_targets_w")
    normals = _frame_series(surface["surface_contact_normals_w"], count, start,
                            "surface_contact_normals_w")
    environment = _frame_series(surface["surface_contact_environment_w"], count, start,
                                "surface_contact_environment_w")
    source_vectors = _frame_series(surface["surface_contact_source_vectors_w"], count, start,
                                   "surface_contact_source_vectors_w")
    modes = _frame_series(surface["surface_contact_modes"], count, start,
                          "surface_contact_modes", object)
    surface_ids = _frame_series(surface["surface_contact_surface_ids"], count, start,
                                "surface_contact_surface_ids")
    points = np.full((count, len(bindings), 3), np.nan)
    fixed_points = np.empty_like(points)

    for frame in range(count):
        for binding in range(len(bindings)):
            geom = int(geoms[binding])
            fixed_points[frame, binding] = (
                transforms["geom_xpos"][frame, geom]
                + transforms["geom_xmat"][frame, geom] @ local[binding])
    points[active] = fixed_points[active]

    vector = np.linalg.norm(points - targets, axis=-1)
    normal_length = np.linalg.norm(normals, axis=-1)
    valid_normal = active & (normal_length > 1e-12)
    normal_error = np.full_like(vector, np.nan)
    signed_gap = np.full_like(vector, np.nan)
    normal_error[valid_normal] = np.abs(np.einsum(
        "ni,ni->n", points[valid_normal] - targets[valid_normal],
        normals[valid_normal] / normal_length[valid_normal, None]))
    signed_gap[valid_normal] = np.einsum(
        "ni,ni->n", points[valid_normal] - environment[valid_normal],
        normals[valid_normal] / normal_length[valid_normal, None])
    sticking = active & (modes == "sticking")
    continuation = sticking[1:] & sticking[:-1]
    continuation &= surface_ids[1:] == surface_ids[:-1]
    velocity = np.linalg.norm(np.diff(points - targets, axis=0), axis=-1) * fps

    result["binding_parts"] = {
        part: np.flatnonzero(np.asarray([str(value) for value in bindings]) == part).tolist()
        for part in parts}
    selections = result.setdefault("binding_selection", {})
    for metric, values, signed in (
            (SURFACE_VECTOR, vector, False), (SURFACE_NORMAL, normal_error, False),
            (SURFACE_GAP, signed_gap, True)):
        selected = selections.setdefault(metric, {})
        for part in parts:
            group = result["binding_parts"][part]
            for frame in range(count):
                value, binding = _worst_binding(values[frame, group], signed=signed)
                metrics[metric][part][frame] = value
                selected.setdefault(part, np.full(count, -1, int))[frame] = group[binding]
    for part in parts:
        group = result["binding_parts"][part]
        binding_continuation = continuation[:, group]
        valid = np.any(binding_continuation & np.isfinite(velocity[:, group]), axis=1)
        # Assign at the second frame of each sticking continuation.
        metrics[SURFACE_SLIP][part][1:][valid] = np.max(
            np.where(binding_continuation[valid], velocity[:, group][valid], -np.inf),
            axis=1)
    signals = result["binding_signals"]["surface"]
    signals.extend({
        "id": f"surface:{part}:binding_{binding}",
        "kind": "surface", "part": str(part),
        "binding_index": int(binding), "world_positions": fixed_points[:, binding],
        "active": active[:, binding], "units": "m",
        "source_world_positions": environment[:, binding] + source_vectors[:, binding],
        "provenance": (
            "fixed prepared surface geom/local material point evaluated by full-frame FK; "
            "active is the prepared surface_contact_mask for this binding"
        ),
        "source_provenance": (
            "prepared surface source skin point, exactly "
            "surface_contact_environment_w + surface_contact_source_vectors_w; never target_w"
        ),
    } for binding, part in enumerate(bindings))


def measure(inputs, model, poses, choices, *, fps, start=0):
    """Return full-frame, per-part contact observations.

    ``metrics[metric][part]`` always has one value per output pose.  The first
    frame of a trajectory and transitions to a different surface/environment
    have NaN slip values.  Surface parts with multiple bindings use the largest
    framewise defect (largest absolute signed gap); ``binding_selection`` records
    the selected binding.  No values are averaged across bodies or parts.
    """
    inputs = _input_dictionary(inputs)
    poses = np.asarray(poses, float)
    if poses.ndim != 2 or poses.shape[1] != model.nq or not len(poses):
        raise ValueError("poses must be nonempty (frames, model.nq)")
    if not np.isfinite(poses).all():
        raise ValueError("poses must be finite")
    fps, start = float(fps), int(start)
    if not np.isfinite(fps) or fps <= 0 or start < 0:
        raise ValueError("fps must be finite/positive and start nonnegative")

    data = mujoco.MjData(model)
    metrics, result = {}, {}
    result["binding_signals"] = {"endpoint": [], "surface": []}
    transforms = {
        "body_xpos": np.empty((len(poses), model.nbody, 3)),
        "body_xmat": np.empty((len(poses), model.nbody, 3, 3)),
        "geom_xpos": np.empty((len(poses), model.ngeom, 3)),
        "geom_xmat": np.empty((len(poses), model.ngeom, 3, 3)),
    }
    for frame, pose in enumerate(poses):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        transforms["body_xpos"][frame] = data.xpos
        transforms["body_xmat"][frame] = data.xmat.reshape(-1, 3, 3)
        transforms["geom_xpos"][frame] = data.geom_xpos
        transforms["geom_xmat"][frame] = data.geom_xmat.reshape(-1, 3, 3)
    _endpoint_rows(inputs, model, poses, choices, start, metrics, result, transforms, fps)
    _surface_rows(inputs, model, poses, start, metrics, result, transforms, fps)

    source_anchor = inputs.get("effectors", {}).get("source_root_robot_body")
    root_part = str(source_anchor) if source_anchor is not None else "source_root"
    metrics[ROOT_DRIFT] = {root_part: np.full(len(poses), np.nan)}
    if source_anchor is not None:
        body = int(model.body(source_anchor).id)
        source = _frame_series(inputs["roots"], len(poses), start, "roots")
        actual = transforms["body_xpos"][:, body]
        metrics[ROOT_DRIFT][root_part] = np.linalg.norm(actual - source, axis=1)

    result.update(
        start=start, frames=len(poses), framerate=fps, metrics=metrics,
        units={ENDPOINT_VECTOR: "m", ENDPOINT_NORMAL: "m", ENDPOINT_GAP: "m",
               ENDPOINT_SLIP: "m/s", SURFACE_VECTOR: "m", SURFACE_NORMAL: "m",
               SURFACE_GAP: "m", SURFACE_SLIP: "m/s", ROOT_DRIFT: "m"},
        provenance={
            ENDPOINT_VECTOR: "unmanaged prepared endpoint contact anchor; selected contact offset",
            ENDPOINT_NORMAL: "absolute endpoint-anchor projection on the prepared observed normal",
            ENDPOINT_GAP: "signed endpoint mapped-point distance to task.surface_points plane; not an original-geom collision certificate",
            ENDPOINT_SLIP: "target-relative endpoint velocity over two sticking frames with the same environment id when ids exist",
            SURFACE_VECTOR: "prepared surface-contact mapped geom point versus target_w; per-part framewise maximum over bindings",
            SURFACE_NORMAL: "absolute surface-point projection error on normals_w; per-part framewise maximum over bindings",
            SURFACE_GAP: "signed point-environment projection on normals_w; observation plane only, not an original-geom collision certificate; largest absolute binding is retained",
            SURFACE_SLIP: "target-relative surface-point velocity over two sticking frames with the same surface id",
            ROOT_DRIFT: "source_root_robot_body world position minus prepared source root; anatomical anchor, not floating-base qpos fallback",
        },
        binding_selection_index="global index into the prepared surface_contact_* arrays",
        limitations=[
            "signed gaps use observed task planes and do not certify original geometry collision freedom",
            "managed endpoint frames are excluded from endpoint metrics and must be read from surface-contact rows",
            "slip values are assigned to the second frame of each valid continuation",
            "surface parts with multiple bindings are reduced by framewise maximum, never by cross-body averaging",
        ],
    )
    return result
