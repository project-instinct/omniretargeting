"""Objective-independent motion measurements for A/B/C/D experiments.

No acceptance policy lives here. Missing evidence is None, never a zero score.
Temporal measurements use the original cadence, without filtering the source.
"""

import itertools

import mujoco
import numpy as np


def summary(values):
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    return {
        "count": len(values),
        "mean": float(values.mean()) if len(values) else None,
        "p95": float(np.percentile(values, 95)) if len(values) else None,
        "max": float(values.max()) if len(values) else None,
    }


def unit(vectors):
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(
        vectors, lengths, out=np.full_like(vectors, np.nan), where=lengths > 1e-8
    )


def angle_degrees(a, b):
    return np.rad2deg(np.arccos(np.clip(np.sum(a * b, axis=-1), -1, 1)))


def trajectory_metrics(
    model, qpos, source_positions, names, mapping, priority, tasks, fps
):
    """Measure identical semantic markers/edges, irrespective of solver loss.

    Source-world free hand errors are diagnostic only: their absolute world
    positions are not the project's free-arm objective. Local bone fidelity and
    amplitude are reported alongside them. Contact errors use the configured
    contact shell offsets and the same source contact plan in every variant.
    """
    qpos, source_positions = np.asarray(qpos), np.asarray(source_positions)
    data = mujoco.MjData(model)
    mapped = list(mapping)
    indices = [names.index(n) for n in mapped]
    robot = []
    marker_positions = {name: [] for name in tasks}
    contact_errors = {name: [] for name in tasks}
    normal_errors = {name: [] for name in tasks}
    for f, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        robot.append([data.xpos[model.body(mapping[n]).id].copy() for n in mapped])
        for name, task in tasks.items():
            spec = priority["effectors"][name]
            body = model.body(spec["robot_body"]).id
            rotation, center = data.xmat[body].reshape(3, 3), data.xpos[body]
            offset = np.asarray(spec.get("tracking_offset", [0, 0, 0]))
            marker_positions[name].append(center + rotation @ offset)
            if task.contact_mask[f]:
                offsets = np.asarray(spec.get("contact_offsets", [offset])).reshape(
                    -1, 3
                )
                errors = (
                    center + offsets @ rotation.T - task.frame(f)["contact_position"]
                )
                best = errors[np.argmin(np.linalg.norm(errors, axis=1))]
                contact_errors[name].append(np.linalg.norm(best))
                if task.surface_normals is not None:
                    normal_errors[name].append(abs(best @ task.surface_normals[f]))
    robot = np.asarray(robot)
    source = source_positions[:, indices]
    edges = [
        tuple(e)
        for e in priority.get("pose_direction", {}).get("edges", [])
        if all(n in mapped for n in e)
    ]
    direction, robot_directions, source_directions = {}, {}, {}
    for a, b in edges:
        ia, ib = mapped.index(a), mapped.index(b)
        rd, sd = unit(robot[:, ib] - robot[:, ia]), unit(source[:, ib] - source[:, ia])
        robot_directions[(a, b)], source_directions[(a, b)] = rd, sd
        source_amplitude = float(np.linalg.norm(np.nanstd(sd, axis=0)))
        robot_amplitude = float(np.linalg.norm(np.nanstd(rd, axis=0)))
        lag_scores = {}
        if len(qpos) >= 8 and source_amplitude > 0.01:
            for lag in range(-3, 4):
                # Positive lag: robot at t+lag follows source at t.
                s, r = (
                    (sd[:-lag], rd[lag:])
                    if lag > 0
                    else (sd[-lag:], rd[:lag])
                    if lag < 0
                    else (sd, rd)
                )
                s, r = s - s.mean(axis=0), r - r.mean(axis=0)
                lag_scores[lag] = float(np.mean((s - r) ** 2))
        direction[a + "->" + b] = {
            "world_angle_deg": summary(angle_degrees(rd, sd)),
            "direction_step_error": summary(
                np.linalg.norm(np.diff(rd, axis=0) - np.diff(sd, axis=0), axis=1)
            ),
            "direction_acceleration_error_per_s2": summary(
                np.linalg.norm(np.diff(rd - sd, n=2, axis=0), axis=1) * fps**2
            ),
            "source_amplitude": source_amplitude,
            "robot_amplitude": robot_amplitude,
            "amplitude_ratio": robot_amplitude / source_amplitude
            if source_amplitude > 0.01
            else None,
            "lag_frames": min(lag_scores, key=lag_scores.get) if lag_scores else None,
        }
    configurations = {}
    for a, b in edges:
        for b2, c in edges:
            if b != b2:
                continue
            ra, rb = robot_directions[a, b], robot_directions[b, c]
            sa, sb = source_directions[a, b], source_directions[b, c]
            rn, sn = unit(np.cross(ra, rb)), unit(np.cross(sa, sb))
            configurations[a + "->" + b + "->" + c] = {
                "bend_angle_error_deg": summary(
                    abs(angle_degrees(ra, rb) - angle_degrees(sa, sb))
                ),
                "bend_plane_error_deg": summary(angle_degrees(rn, sn)),
                "bend_plane_step_deg": summary(angle_degrees(rn[1:], rn[:-1])),
                "source_bend_plane_step_deg": summary(angle_degrees(sn[1:], sn[:-1])),
                "bend_plane_step_error_deg": summary(
                    abs(angle_degrees(rn[1:], rn[:-1]) - angle_degrees(sn[1:], sn[:-1]))
                ),
            }
    velocity = []
    for previous, current in itertools.pairwise(qpos):
        v = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, v, 1 / fps, previous, current)
        velocity.append(v)
    velocity = np.asarray(velocity).reshape(-1, model.nv)
    scalar = [
        j
        for j in range(model.njnt)
        if model.jnt_type[j]
        in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
    ]
    qa = np.array([model.jnt_qposadr[j] for j in scalar], dtype=int)
    va = np.array([model.jnt_dofadr[j] for j in scalar], dtype=int)
    limit_violations = []
    for j in scalar:
        if model.jnt_limited[j]:
            x = qpos[:, model.jnt_qposadr[j]]
            lo, hi = model.jnt_range[j]
            limit_violations.extend(np.maximum(np.maximum(lo - x, x - hi), 0))
    effectors = {}
    for name, task in tasks.items():
        points = np.asarray(marker_positions[name])
        desired = np.asarray(
            [task.frame(f)["tracking_position"] for f in range(len(qpos))]
        )
        mask = task.contact_mask[1:] & task.contact_mask[:-1]
        if task.contact_modes is not None:
            mask &= (task.contact_modes[1:] == "sticking") & (
                task.contact_modes[:-1] == "sticking"
            )
        effectors[name] = {
            "source_world_marker_error_m": summary(
                np.linalg.norm(points - desired, axis=1)
            ),
            "contact_position_error_m": summary(contact_errors[name]),
            "contact_normal_error_m": summary(normal_errors[name]),
            "support_slip_m_per_s": summary(
                np.linalg.norm(np.diff(points, axis=0)[mask, :2], axis=1) * fps
            ),
            "support_vertical_speed_m_per_s": summary(
                abs(np.diff(points, axis=0)[mask, 2]) * fps
            ),
        }
    anchor = priority.get("_position_anchor_source_joint")
    root_error = (
        summary(
            np.linalg.norm(
                robot[:, mapped.index(anchor)] - source[:, mapped.index(anchor)], axis=1
            )
        )
        if anchor in mapped
        else summary([])
    )
    return {
        "root_anchor_world_error_m": root_error,
        "direction": direction,
        "limb_configuration": configurations,
        "end_effectors": effectors,
        "joint_speed_rad_per_s": summary(abs(velocity[:, va])),
        "joint_acceleration_rad_per_s2": summary(
            abs(np.diff(velocity[:, va], axis=0)) * fps
        ),
        "joint_jerk_rad_per_s3": summary(
            abs(np.diff(velocity[:, va], n=2, axis=0)) * fps**2
        ),
        "joint_range_rad": (np.ptp(qpos[:, qa], axis=0).tolist() if len(qpos) else []),
        "mechanical_limit_violation_rad": summary(limit_violations),
        "frames_outside_mechanical_limits": int(
            sum(
                any(
                    model.jnt_limited[j]
                    and not (
                        model.jnt_range[j, 0] - 1e-8
                        <= q[model.jnt_qposadr[j]]
                        <= model.jnt_range[j, 1] + 1e-8
                    )
                    for j in scalar
                )
                for q in qpos
            )
        ),
        "nonfinite_frames": int(np.sum(~np.isfinite(qpos).all(axis=1))),
    }
