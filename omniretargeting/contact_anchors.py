"""Material-frame rigid placements shared by all contact samples on a body.

Sample activation chooses which part is pinned, not a fresh body placement.
Welded collision geometry is treated as one rigid body. Source and scene are
immutable; a fit residual records incompatible body sizes or surface patches.
"""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .contact_schedule import _iter_true_runs
from .anchor_motion import (fit_moving_anchor, observed_material_targets,
                            source_material_rotations)


# A contact body's rigid placement stays within this angle of its same-frame
# source orientation; a sticking interval whose single placement would exceed
# it follows the source per frame instead (``anchor_motion``). A tolerance on
# following the original motion, not a posture limit against the reference.
SOURCE_ORIENTATION_TOLERANCE_RAD = 0.2


def rigid_placement(local, target, weights, reference_rotation, *, rotation_budget_rad):
    """Weighted Procrustes fit with an explicit orientation-deviation bound."""
    local, target, weights = map(
        lambda x: np.asarray(x, float), (local, target, weights)
    )
    reference_rotation = np.asarray(reference_rotation, float)
    if (
        local.shape != target.shape
        or local.ndim != 2
        or local.shape[1] != 3
        or weights.shape != (len(local),)
        or not len(local)
        or np.any(weights < 0)
        or weights.sum() <= 0
        or not all(
            np.isfinite(a).all() for a in (local, target, weights, reference_rotation)
        )
        or not np.isfinite(rotation_budget_rad)
        or rotation_budget_rad < 0
    ):
        raise ValueError(
            "A rigid anchor needs finite weighted point pairs and an orientation budget"
        )
    weights = weights / weights.sum()
    center_local, center_target = weights @ local, weights @ target
    covariance = ((local - center_local) * weights[:, None]).T @ (
        target - center_target
    )
    left, values, right = np.linalg.svd(covariance)
    fitted = reference_rotation.copy()
    if values[1] > max(values[0], 1e-15) * 1e-10:
        fitted = right.T @ np.diag([1, 1, np.linalg.det(right.T @ left.T)]) @ left.T
    delta = Rotation.from_matrix(fitted @ reference_rotation.T).as_rotvec()
    angle = np.linalg.norm(delta)
    if angle > rotation_budget_rad:
        fitted = (
            Rotation.from_rotvec(delta * rotation_budget_rad / angle).as_matrix()
            @ reference_rotation
        )
    translation = center_target - fitted @ center_local
    residual = np.linalg.norm(local @ fitted.T + translation - target, axis=1)
    return (
        fitted,
        translation,
        {
            "weighted_rms_m": float(np.sqrt(weights @ residual**2)),
            "maximum_m": float(residual.max()),
            "unbounded_rotation_rad": float(angle),
            "rotation_bound_active": bool(angle > rotation_budget_rad),
        },
    )


def _placement_targets(local, fitted, offset, rotation, translation, scale):
    """A material-anchored body origin with fixed metric robot offsets.

    An authored object scale may vary. Its surface cannot then keep every
    point of a rigid robot patch material-fixed; preserve the body size and
    report the fit residual instead of resizing the robot or freezing scale.
    """
    if np.all(scale == scale[0]):
        material = local / scale[0] @ fitted.T + offset
        return np.einsum("ni,tji->tnj", material, rotation) * scale[:, None, None] + translation[:, None]
    points = np.einsum("ni,tji->tnj", local @ fitted.T, rotation)
    origin = np.einsum("tij,j->ti", rotation, offset) * scale[:, None] + translation
    return points + origin[:, None]


def build_rigid_anchors(
    model,
    bindings,
    sample_indices,
    observation,
    surface_motion,
    *,
    scene_present,
    scene_objects=(),
    rotation_budget_rad,
):
    """Return observed material goals and separate rigid-placement diagnostics.

    The complete sticking interval informs the robot placement estimate, but
    its shape/orientation residual cannot change the scene-contact goals.
    Existing placement estimates are never refitted by later events.
    """
    ids = np.asarray(sample_indices, int)
    count = len(surface_motion.points)
    active = observation["active"][:, ids]
    surface_ids = observation["surface_ids"][:, ids]
    modes = observation["modes"][:, ids]
    confidence = observation["confidence"][:, ids]
    environment = observation["environment"][:, ids]
    data = mujoco.MjData(model)
    data.qpos[:] = surface_motion.reference_qpos
    mujoco.mj_forward(model, data)
    geom_ids = np.array([model.geom(b.robot_geom).id for b in bindings])
    bodies = model.body_weldid[model.geom_bodyid[geom_ids]]
    local = np.empty((len(ids), 3))
    for k, (binding, geom, body) in enumerate(zip(bindings, geom_ids, bodies)):
        point = (
            data.geom_xpos[geom]
            + data.geom_xmat[geom].reshape(3, 3) @ binding.local_point
        )
        local[k] = (point - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
    poses = []
    if scene_present:
        poses.append(
            (np.tile(np.eye(3), (count, 1, 1)), np.zeros((count, 3)), np.ones(count))
        )
    for obj in scene_objects:
        poses.append(
            (
                np.asarray(obj.get("rotations", np.tile(np.eye(3), (count, 1, 1)))),
                np.asarray(obj.get("translations", np.zeros((count, 3)))),
                np.asarray(obj.get("scales", np.ones(count))).reshape(-1),
            )
        )
    targets = environment.copy()
    placements, events = [], {}
    for body in np.unique(bodies):
        group = np.flatnonzero(bodies == body)
        source_ids = ids[group]
        body_rotation = data.xmat[body].reshape(3, 3)
        orientation = (
            surface_motion.frames[:, source_ids]
            @ surface_motion.reference_frames[source_ids].transpose(0, 2, 1)
            @ body_rotation
        )
        for surface in np.unique(surface_ids[:, group][active[:, group]]):
            rotation, translation, scale = poses[surface]
            source_rotations = source_material_rotations(orientation, rotation)
            mask = active[:, group] & (surface_ids[:, group] == surface)
            sticking = (mask & (modes[:, group] == "sticking")).any(1)
            runs = [(start, stop, True) for start, stop in _iter_true_runs(sticking)]
            runs += [
                (int(f), int(f + 1), False)
                for f in np.flatnonzero(mask.any(1) & ~sticking)
            ]
            for start, stop, locked in sorted(runs):
                interval = slice(start, stop)
                chosen = mask[interval].copy()
                if locked:
                    chosen &= modes[interval, group] == "sticking"
                frame, point = np.nonzero(chosen)
                f = frame + start
                weights = confidence[f, group[point]] * chosen.sum(1)[frame]
                if not np.any(weights > 0):
                    continue
                material = (
                    np.einsum(
                        "ni,nij->nj",
                        environment[f, group[point]] - translation[f],
                        rotation[f],
                    )
                    / scale[f, None]
                )
                constant_scale = bool(np.all(scale[interval] == scale[start]))
                # Fit a shared material body origin, with metric local offsets.
                # s^2 weights make the residual a world-distance objective even
                # when samples at different frames have different object scales.
                robot_local = local[group[point]] / (
                    scale[start] if constant_scale else scale[f, None]
                )
                fit_weights = weights if constant_scale else weights * scale[f] ** 2
                reference = rotation[f].transpose(0, 2, 1) @ orientation[f, point]
                mean = Rotation.from_matrix(reference).mean(weights=weights).as_matrix()
                fitted, offset, report = rigid_placement(
                    robot_local,
                    material,
                    fit_weights,
                    mean,
                    rotation_budget_rad=rotation_budget_rad,
                )
                # A persistent point contact is not evidence of a locked body
                # orientation. Never impose an interval mean outside any frame's
                # original source-orientation envelope. A mixed sliding patch
                # also cannot inherit a different sample's sticking placement.
                source_error = Rotation.from_matrix(
                    fitted @ source_rotations[interval].transpose(0, 2, 1)
                ).magnitude()
                mixed = mask[interval] & (modes[interval, group] != "sticking")
                if locked and (np.any(source_error > rotation_budget_rad + 1e-7)
                               or np.any(mixed)):
                    world, moving_report = fit_moving_anchor(
                        local[group], environment[interval, group], mask[interval],
                        modes[interval, group], confidence[interval, group],
                        source_rotations[interval], rotation[interval],
                        translation[interval], scale[interval],
                        rotation_budget_rad=rotation_budget_rad)
                    view = targets[interval, group].copy()
                    view[mask[interval]] = world[mask[interval]]
                    targets[interval, group] = view
                    placements.append({
                        "body": int(body), "surface": int(surface),
                        "start": start, "stop": stop, "locked": False,
                        "sample_indices": ids[group],
                        "rejected_locked_source_error_rad": float(source_error.max()),
                        "mixed_contact_modes": bool(np.any(mixed)), **moving_report,
                    })
                    continue
                residual_m = np.linalg.norm(robot_local @ fitted.T + offset - material, axis=1) * scale[f]
                report.update(
                    weighted_rms_m=float(np.sqrt(weights @ residual_m**2 / weights.sum())),
                    maximum_m=float(residual_m.max()),
                    constant_surface_scale=constant_scale,
                    robot_metric_size_preserved=True,
                )
                world = _placement_targets(
                    local[group], fitted, offset, rotation[interval], translation[interval], scale[interval]
                )
                view = targets[interval, group].copy()
                view[mask[interval]] = world[mask[interval]]
                targets[interval, group] = view
                placements.append(
                    {
                        "body": int(body),
                        "surface": int(surface),
                        "start": start,
                        "stop": stop,
                        "locked": locked,
                        "rotation": fitted,
                        "translation": offset,
                        "sample_indices": ids[group],
                        "fit": report,
                        "observed": material,
                        "robot_points": robot_local @ fitted.T + offset,
                    }
                )
                if locked:
                    events.setdefault(start, []).append(len(placements) - 1)
    from .anchor_events import place_anchor_events
    event_reports = place_anchor_events(model, data, placements, poses)
    for placement in placements:
        if not placement["locked"]:
            continue
        group = np.flatnonzero(bodies == placement["body"])
        start, stop = placement["start"], placement["stop"]
        rotation, translation, scale = poses[placement["surface"]]
        mask = active[start:stop, group] & (surface_ids[start:stop, group] == placement["surface"])
        world = _placement_targets(
            local[group], placement["rotation"], placement["translation"],
            rotation[start:stop], translation[start:stop], scale[start:stop],
        )
        view = targets[start:stop, group].copy()
        view[mask] = world[mask]
        targets[start:stop, group] = view
    # Large point clouds are only transient footprint evidence; the immutable
    # source table retains the observations for independent validation.
    for placement in placements:
        placement.pop("observed", None)
        placement.pop("robot_points", None)
    # The rigid placements above describe robot reach/morphology only. Their
    # residual must never move a contact goal away from the observed surface.
    # Keep even incompatible material point pairs as simultaneous IK targets.
    fitted_targets = targets
    targets = environment.copy()
    for surface in np.unique(surface_ids[active]):
        rotation, translation, scale = poses[surface]
        mask = active & (surface_ids == surface)
        anchored = observed_material_targets(
            environment, mask, modes, confidence, rotation, translation, scale)
        targets[mask] = anchored[mask]
    target_fit = np.linalg.norm(fitted_targets - targets, axis=-1)
    targets.setflags(write=False)
    return targets, {
        "method": "observed_material_contacts_with_separate_rigid_fit",
        "targets_on_observed_surface": True,
        "rigid_fit_target_residual_max_m": float(target_fit[active].max(initial=0.)),
        "moving_contact_intervals": sum(p.get("mode") == "source_following_material_contacts" for p in placements),
        "placements": placements,
        "events": events,
        "event_placement": event_reports,
        "observations_modified": False,
        "held_anchors_refitted": False,
    }
