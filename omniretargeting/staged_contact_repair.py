"""Sequential contact-group clearance with fixed posture and contact anchors.

This strict sequence-backend entry point also supports replay. An input that
already violates the supplied pre-acquisition budgets is reported unchanged;
it must be reacquired under those budgets, never certified with a new reference.
Original-scene interframe validation is still required by the owning pipeline.
"""

import mujoco
import numpy as np

from ._sequence.contact_hold import ContactHoldAnchors, ContactHoldStage
from ._sequence.continuity_guard import SequenceContinuityGuard
from ._sequence.pose_guard import SequencePoseGuard
from ._sequence.surface_guard import SurfaceSequenceGuard
from .geometry_query import GeometryQueryError
from .sequence_retargeting import prepare_sequence, retarget_sequence, sequence_config


def measure_contact_scene(tasks, poses):
    """Original static shells and moving-mesh forward/reverse witnesses."""
    depths = np.zeros((len(poses), tasks.model.nbody))
    exits = {}
    unknown = 0
    if tasks.scene_cache is not None:
        tasks.scene_cache.prepare(poses)
    moving = [
        t
        for t in tasks.extra_tasks
        if hasattr(t, "query") and hasattr(t, "clearance_rows")
    ]
    for task in moving:
        task.prepare_trajectory(poses)

    def record(frame, geom, clearance, direction):
        if clearance >= 0:
            return
        body = int(tasks.model.body_weldid[tasks.model.geom_bodyid[geom]])
        depth = -float(clearance)
        depths[frame, body] = max(depths[frame, body], depth)
        exits.setdefault((frame, body), []).append(depth * np.asarray(direction))

    for frame, pose in enumerate(poses):
        tasks.data.qpos[:] = pose
        mujoco.mj_forward(tasks.model, tasks.data)
        measured = (
            None if tasks.scene_cache is None else tasks.scene_cache.get(pose, frame)
        )
        begin = 0
        for geom, offsets in tasks.shells:
            end = begin + len(offsets)
            if measured is None:
                points = (
                    offsets @ tasks.data.geom_xmat[geom].reshape(3, 3).T
                    + tasks.data.geom_xpos[geom]
                )
                queried = tasks.scene.query(points)
            else:
                queried = {key: value[begin:end] for key, value in measured.items()}
            begin = end
            for gap, direction, uncertain in zip(
                queried["clearance"], queried["directions"], queried["uncertain"]
            ):
                if (
                    uncertain
                    or not np.isfinite(gap)
                    or not np.isfinite(direction).all()
                ):
                    unknown += 1
                else:
                    record(frame, geom, gap, direction)
        for task in moving:
            queried = task.query.query(pose, frame, [])
            if queried["failures"]:
                unknown += len(queried["failures"])
            for row in queried["rows"]:
                record(frame, row["geom_id"], row["clearance"], row["direction"])
    return {"depths": depths, "exits": exits, "unknown": unknown}


def scene_summary(tasks, measurement):
    depths = measurement["depths"]
    frame, body = np.unravel_index(np.argmax(depths), depths.shape)
    vectors = measurement["exits"].get((frame, body), [np.zeros(3)])
    vector = max(vectors, key=np.linalg.norm)
    return {
        "max_depth_m": float(depths[frame, body]),
        "worst_frame": int(frame),
        "worst_body": tasks.model.body(body).name,
        "worst_exit_xy_m": float(np.linalg.norm(vector[:2])),
        "worst_exit_z_m": float(vector[2]),
        "affected_frames": np.flatnonzero(depths.max(axis=1) > 0.0005).tolist(),
        "unknown_samples": measurement["unknown"],
    }


class MinimumBodyDisplacement:
    """Metric regularization around the fixed repair input, across all stages."""

    def __init__(self, tasks, reference):
        self.tasks = tasks
        self.reference = np.array(reference, copy=True)
        self.reference.setflags(write=False)
        self.bodies = sorted(
            set(tasks.bodyids) | {s["body_id"] for s in tasks.effectors.values()}
        )
        self.points = []
        for pose in reference:
            tasks.data.qpos[:] = pose
            mujoco.mj_forward(tasks.model, tasks.data)
            self.points.append(tasks.data.xpos[self.bodies].copy())
        self.scale = tasks.config["translation_step_m"]

    def rows(self, pose, frame, dofs):
        tasks = self.tasks
        tasks.data.qpos[:] = pose
        mujoco.mj_forward(tasks.model, tasks.data)
        for index, body in enumerate(self.bodies):
            jac = np.zeros((3, tasks.model.nv))
            mujoco.mj_jacBody(tasks.model, tasks.data, jac, None, body)
            yield (
                1.0,
                (tasks.data.xpos[body] - self.points[frame][index]) / self.scale,
                jac[:, dofs] / self.scale,
            )

    def linearize(self, tasks, pose, frame):
        """Clearance and minimal movement, without refitting free limbs."""
        tasks.data.qpos[:] = pose
        mujoco.mj_forward(tasks.model, tasks.data)
        residuals, matrices = [], []
        for residual, jac in tasks.clearance_rows(frame=frame):
            residuals.append(
                np.atleast_1d(residual) / tasks.config["collision_sigma_m"]
            )
            matrices.append(np.atleast_2d(jac) / tasks.config["collision_sigma_m"])
        for weight, residual, jac in self.rows(pose, frame, tasks.coords.cols):
            residuals.append(np.sqrt(weight) * residual)
            matrices.append(np.sqrt(weight) * jac)
        return np.concatenate(residuals), np.vstack(matrices)


def repair_contact_stages(
    model,
    initial,
    source_positions,
    source_names,
    joint_mapping,
    source_root_positions,
    tasks,
    end_effector_config,
    *,
    endpoint_reference_trajectory,
    correction_reference_trajectory,
    framerate,
    config=None,
    extra_tasks=(),
    callback=None,
    **kwargs,
):
    """Repair colliding active contact groups, holding every other attained point.

    First try scalar joints with a fixed base. Only coherent upward exits permit
    a second, height-only base attempt. The existing cumulative translation,
    source-shape, body-angle and per-joint derivative bounds apply to both.
    No support role is inferred here: ordering consumes the existing role plan.
    """
    cfg = sequence_config(config)
    result = np.array(initial, dtype=float, copy=True)
    coords, _, task_set, rotations = prepare_sequence(
        model,
        result,
        source_positions,
        source_names,
        joint_mapping,
        source_root_positions,
        tasks,
        end_effector_config,
        cfg,
        extra_tasks=extra_tasks,
        **kwargs,
    )
    report = {
        "status": "no_active_collision",
        "stages": [],
        "validation_required": True,
    }
    if task_set is None:
        return result, report
    kwargs = {**kwargs, "contact_offset_choice": task_set.choices}
    guard = SurfaceSequenceGuard(
        task_set, endpoint_reference_trajectory, cfg["endpoint_regression_m"]
    )
    guard.pose = SequencePoseGuard(task_set, correction_reference_trajectory)
    guard.continuity = SequenceContinuityGuard(
        task_set, correction_reference_trajectory, rotations
    )
    report["posture_envelope"] = guard.pose.diagnostics(result)
    if not guard.accepts(result):
        report["status"] = "input_exceeds_fixed_budgets"
        report["input_checks"] = {
            "posture": guard.pose.accepts(result),
            "continuity": guard.continuity.accepts(result),
        }
        return result, report
    measurement = measure_contact_scene(task_set, result)
    report["before"] = scene_summary(task_set, measurement)
    if measurement["unknown"]:
        report.update(status="unknown_scene_geometry", after=report["before"])
        return result, report
    # Already-established contacts are held at the repair input. Their accuracy
    # still has the separate immutable acquisition reference above.
    anchors = ContactHoldAnchors(guard, result)
    groups = {body for frame in anchors.bodies for body in frame.values()}
    groups = [b for b in groups if measurement["depths"][:, b].max() > 0.0005]

    def order(body):
        from .contact_phase import CONTACT_ROLES

        roles = []
        policy = task_set.contact_phase
        if policy is not None:
            for frame, items in enumerate(anchors.bodies):
                for key, owner in items.items():
                    if owner != body:
                        continue
                    if key[0] == "endpoint":
                        role = policy.role(key[1], frame)
                    elif task_set.extra_tasks[key[1]] is policy.surface:
                        role = next(
                            (
                                r
                                for r in CONTACT_ROLES
                                if policy.surface_roles[r][frame, key[2]]
                            ),
                            None,
                        )
                    else:
                        role = None
                    if role in CONTACT_ROLES:
                        roles.append(CONTACT_ROLES.index(role))
        return (
            min(roles, default=len(CONTACT_ROLES)),
            -measurement["depths"][:, body].max(),
            body,
        )

    regularizer = MinimumBodyDisplacement(task_set, result)
    for body in sorted(groups, key=order):
        for mode in ("fixed_base", "bounded_root_lift"):
            if measurement["depths"][:, body].max() <= 0.0005:
                break
            stage_cfg = dict(
                cfg, lock_root_xy=True, lock_root_orientation=True, lock_root_z=True
            )
            root_bounds = None
            if mode == "bounded_root_lift":
                if coords.root_q is None or cfg["lock_root_z"]:
                    continue
                root_bounds = np.repeat(result[:, coords.root_q + 2, None], 2, axis=1)
                coherent = 0
                for frame in range(len(result)):
                    vectors = measurement["exits"].get((frame, body), [])
                    if not vectors:
                        continue
                    vectors = np.asarray(vectors)
                    if np.all(vectors[:, 2] >= 0.9 * np.linalg.norm(vectors, axis=1)):
                        root_bounds[frame, 1] += (
                            measurement["depths"][frame, body]
                            + cfg["scene_clearance_m"]
                        )
                        coherent += len(vectors)
                if coherent < 3:
                    continue
                stage_cfg["lock_root_z"] = False
            stage = ContactHoldStage(
                anchors,
                body,
                min(0.0001, cfg["endpoint_regression_m"]),
                root_bounds,
                regularizer,
            )
            before = measurement
            candidate, details = retarget_sequence(
                model,
                result,
                source_positions,
                source_names,
                joint_mapping,
                source_root_positions,
                tasks,
                end_effector_config,
                framerate=framerate,
                config=stage_cfg,
                extra_tasks=(*extra_tasks, regularizer),
                callback=callback,
                endpoint_reference_trajectory=endpoint_reference_trajectory,
                correction_reference_trajectory=correction_reference_trajectory,
                contact_stage=stage,
                **kwargs,
            )
            try:
                checked = measure_contact_scene(task_set, candidate)
                # Compare every frame/body to the preceding accepted assembly.
                # No body can pay for the selected contact's clearance gain.
                safe = not checked["unknown"] and np.all(
                    checked["depths"] <= before["depths"] + 1e-7
                )
                better = (
                    checked["depths"][:, body].sum()
                    < before["depths"][:, body].sum() - 1e-7
                )
            except GeometryQueryError:
                checked, safe, better = before, False, False
            accepted = bool(safe and better)
            report["stages"].append(
                {
                    "mode": mode,
                    "body": model.body(body).name,
                    "accepted": accepted,
                    "reason": "improved"
                    if accepted
                    else "geometry_regression"
                    if not safe
                    else "bounded_residual",
                    "before": scene_summary(task_set, before),
                    "candidate": scene_summary(task_set, checked),
                    "solver": details,
                }
            )
            if accepted:
                result, measurement = candidate, checked
                anchors = anchors.acquired(guard, result, body)
    report["after"] = scene_summary(task_set, measurement)
    report["posture_envelope"] = guard.pose.diagnostics(result)
    report["status"] = (
        (
            "improved_with_residual"
            if measurement["depths"].max() > 0.0005
            else "resolved_sampled_contacts"
        )
        if any(s["accepted"] for s in report["stages"])
        else "bounded_residual"
        if groups
        else "no_active_collision"
    )
    return result, report
