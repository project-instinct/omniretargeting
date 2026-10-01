"""UMR-informed joint surface/contact/collision correction at real metric scale.

The accepted robot trajectory is an immutable reference. All articulated
ancestors needed by active surfaces, collisions and endpoint compensation may
participate in one trajectory solve; floating joints are never variables.
There is no source displacement, object optimization, frame retiming or filter.
"""

import time

import numpy as np
from scipy import sparse

from ._refinement.math import _difference_operator
from .moving_scene import MovingSceneBudgetError, MovingSceneQuery
from .surface_observation import ContactSurface
from .trajectory_validation import QueryBudgetExceeded, local_repair_config
from .whole_body_guard import WholeBodyGuard, WholeBodyValidator, interpolated_pose

WHOLE_BODY_WINDOW_THRESHOLD = 256
WHOLE_BODY_WINDOW_FRAMES = 96

DEFAULTS = {
    "enabled": True,
    "iterations": 3,
    # Locality of one secondary correction relative to the trajectory it
    # receives, not a posture limit against the source motion.
    "max_joint_correction": 0.2,
    "trust_region": 0.06,
    "contact_regression": 0.001,
    "world_position_budget": 0.002,
    "free_position_budget": 0.02,
    "correction_velocity": 1.0,
    "correction_acceleration": 12.0,
    "correction_jerk": 240.0,
    "surface_weight": 1.0,
    "normal_weight": 0.05,
    "reference_sigma": 0.18,
    "velocity_sigma": 0.04,
    "acceleration_sigma": 0.015,
    "jerk_sigma": 0.01,
    "moving_collision_enabled": True,
    "collision_weight": 5.0,
    "collision_sigma": 0.01,
    "collision_regression": 0.0005,
    "penetration_tolerance": 0.0005,
    "motion_spacing": 0.015,
    "max_substeps": 128,
}


def whole_body_config(config=None):
    from .config_validation import known_options

    known_options(config, DEFAULTS, 'whole_body_correction', extra=('moving_scene',))
    cfg = {**DEFAULTS, **dict(config or {})}
    for key, default in DEFAULTS.items():
        if isinstance(default, bool):
            if not isinstance(cfg[key], (bool, np.bool_)):
                raise TypeError(f"whole_body_correction.{key} must be boolean")
        elif not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"whole_body_correction.{key} must be finite and positive")
    for key in ("iterations", "max_substeps"):
        if int(cfg[key]) != cfg[key]:
            raise ValueError(f"whole_body_correction.{key} must be an integer")
    return cfg


def _active_coordinates(validator, moving):
    """Union surface/collision ancestors with endpoint compensation ancestors."""
    v = validator
    bodies = {int(spec["body_id"]) for spec in v.specs.values()}
    for task in (v.surface_contact_tasks, v.self_contact_tasks, v.surface_pose_tasks):
        if task is not None:
            for binding in task.bindings:
                bodies.add(
                    int(v.model.geom_bodyid[v.model.geom(binding.robot_geom).id])
                )
    if moving is not None:
        bodies.update(map(int, v.model.geom_bodyid[moving.geom_ids]))
    ancestors = set()
    for body in bodies:
        while body:
            ancestors.add(body)
            body = int(v.model.body_parentid[body])
    return np.isin(v.model.jnt_bodyid[v.joint_ids], list(ancestors))


class WholeBodyObjective:
    def __init__(self, validator, moving):
        self.v, self.moving = validator, moving
        self.moving_grid = [(f, 0.0) for f in range(len(validator.reference))]
        n, j = len(validator.reference), len(validator.qi)
        cfg = validator.whole_cfg
        self.regularizers = [sparse.eye(n * j, format="csr") / cfg["reference_sigma"]]
        # Regularize the correction, preserving deliberately fast source motion.
        for order, key in (
            (1, "velocity_sigma"),
            (2, "acceleration_sigma"),
            (3, "jerk_sigma"),
        ):
            if n > order:
                self.regularizers.append(
                    sparse.kron(
                        _difference_operator(n, order), sparse.eye(j), format="csr"
                    )
                    / (cfg[key] * (30.0 / validator.fps) ** order)
                )

    def task_rows(self, pose, frame):
        v, cfg = self.v, self.v.whole_cfg
        if v.surface_contact_tasks is not None:
            task = v.surface_contact_tasks
            for weight, residual, jac in task.rows(pose, frame + v.frame_start, v.di, include_normals=False):
                yield cfg["surface_weight"] * weight, residual, jac
            for weight, residual, jac in task.normal_rows(pose, frame + v.frame_start, v.di):
                yield cfg["normal_weight"] * weight, residual, jac
        for task in (v.self_contact_tasks, v.body_relation_tasks, v.surface_pose_tasks):
            if task is not None:
                yield from task.rows(pose, frame + v.frame_start, v.di)

    def linearize(self, poses):
        matrices, targets = [], []
        for f, pose in enumerate(poses):
            rows, rhs = [], []
            for weight, residual, jac in self.task_rows(pose, f):
                weight = np.sqrt(weight)
                rows.append(weight * jac)
                rhs.extend(-weight * residual)
            matrices.append(
                sparse.csr_matrix(np.vstack(rows))
                if rows
                else sparse.csr_matrix((0, len(self.v.qi)))
            )
            targets.extend(rhs)
        task_matrix = sparse.block_diag(matrices, format="csr")
        correction = (poses[:, self.v.qi] - self.v.reference[:, self.v.qi]).ravel()
        matrices = [task_matrix, *self.regularizers]
        targets = [
            np.asarray(targets),
            *[-matrix @ correction for matrix in self.regularizers],
        ]
        return sparse.vstack(matrices, format="csr"), np.concatenate(targets)

    def __call__(self, poses):
        cost = sum(
            weight * float(residual @ residual)
            for f, pose in enumerate(poses)
            for weight, residual, _ in self.task_rows(pose, f)
        )
        correction = (poses[:, self.v.qi] - self.v.reference[:, self.v.qi]).ravel()
        for matrix in self.regularizers:
            residual = matrix @ correction
            cost += float(residual @ residual)
        if self.moving is not None:
            cfg = self.v.whole_cfg
            depths = {}
            for f, alpha in self.moving_grid:
                pose = interpolated_pose(self.v.model, poses, f, alpha)
                result = self.moving.cached(pose, self.v.frame_start + f + alpha)
                for key in ("forward_depth", "reverse_depth"):
                    group = (f, bool(alpha), key)
                    depths[group] = np.maximum(depths.get(group, 0), result[key])
            for depth in depths.values():
                cost += (
                    cfg["collision_weight"]
                    * float(np.sum(depth**2))
                    / cfg["collision_sigma"] ** 2
                )
        return cost


def _surface_scope(task, poses, dofs):
    if task is None:
        return None
    groups = {}
    for f, pose in enumerate(poses):
        for index, residual, jac in task.position_rows(pose, f, dofs):
            part = task.bindings[index].part
            item = groups.setdefault(
                part,
                {
                    "active_frame_points": 0,
                    "movable_frame_points": 0,
                    "immovable_error_max_m": 0.0,
                },
            )
            item["active_frame_points"] += 1
            if np.linalg.norm(jac) > 1e-10:
                item["movable_frame_points"] += 1
            else:
                item["immovable_error_max_m"] = max(
                    item["immovable_error_max_m"], float(np.linalg.norm(residual))
                )
    return groups


def correct_whole_body_trajectory(
    model,
    trajectory,
    tasks,
    priority,
    config=None,
    *,
    scene_query,
    scene_objects=(),
    surface_contact_tasks=None,
    surface_pose_tasks=None,
    self_contact_tasks=None,
    framerate=30.0,
    validation_config=None,
    body_relation_tasks=None,
    contact_offset_choice=None,
    geometry_cache=None,
    candidate_acceptance=None,
):
    """Return a protected robot trajectory and an auditable acceptance report.

    The function also supports replaying a saved trajectory without rerunning
    framewise retargeting. All task objects must use the same compiled model and
    exact timeline. Callers slicing windows must slice original object poses too.
    """
    started = time.perf_counter()
    cfg = whole_body_config(config)
    result = np.array(trajectory, dtype=float, copy=True)
    report = {
        "status": "disabled",
        "configuration": cfg,
        "accepted_update_count": 0,
        "iterations": [],
        "root_invariant": True,
        "max_joint_correction": 0.0,
        "moving_collision": {"status": "disabled"},
    }
    if not cfg["enabled"]:
        return result, report
    if (
        surface_contact_tasks is None
        and self_contact_tasks is None
        and body_relation_tasks is None
        and surface_pose_tasks is None
        and not (scene_objects and cfg["moving_collision_enabled"])
    ):
        report["status"] = "no_active_tasks"
        return result, report
    if (
        result.ndim != 2
        or result.shape[1] != model.nq
        or not len(result)
        or not np.isfinite(result).all()
    ):
        raise ValueError("Whole-body correction requires a finite nonempty trajectory")
    shared_geometry = validation_config is not None
    local = local_repair_config(
        {
            **dict(validation_config or {}),
            **{
                k: cfg[k]
                for k in (
                    "max_joint_correction",
                    "contact_regression",
                    "correction_velocity",
                    "correction_acceleration",
                    "collision_regression",
                )
            },
            **({} if shared_geometry else {
                k: cfg[k] for k in ("penetration_tolerance", "motion_spacing", "max_substeps")
            }),
            "max_query_points": None,
            "swing_position_budget": cfg["free_position_budget"],
        },
        priority=priority,
    )
    validator = WholeBodyValidator(
        model,
        result,
        scene_query,
        tasks,
        priority,
        local,
        framerate,
        whole_body_config=cfg,
        geometry_cache=geometry_cache,
        surface_contact_tasks=surface_contact_tasks,
        surface_pose_tasks=surface_pose_tasks,
        self_contact_tasks=self_contact_tasks,
        body_relation_tasks=body_relation_tasks,
    )
    if contact_offset_choice is not None:
        for name, choice in contact_offset_choice.items():
            if name not in validator.specs:
                raise ValueError(f"Unknown fixed contact marker effector: {name}")
            array = np.array(choice, dtype=int, copy=True)
            if (
                array.shape != (len(result),)
                or np.any(array < 0)
                or np.any(array >= len(validator.specs[name]["contact_offsets"]))
            ):
                raise ValueError(f"Invalid fixed contact marker choices for {name}")
            validator.offset_choice[name] = array
        validator.original_effectors = validator.effectors(result)
    if not len(validator.qi):
        report.update(
            status="no_active_tasks", reason="no_articulated_scalar_coordinates"
        )
        return result, report
    moving = (
        MovingSceneQuery(model, scene_objects, len(result), cfg.get("moving_scene"))
        if (scene_objects and cfg["moving_collision_enabled"])
        else None
    )
    validator.object_surfaces = (
        tuple(obj["surface"] for obj in moving.objects)
        if moving
        else tuple(
            ContactSurface(
                obj["mesh"],
                len(result),
                translations=obj.get("translations"),
                rotations=obj.get("rotations"),
                scales=obj.get("scales"),
            )
            for obj in scene_objects
        )
    )
    # Contact budgets are defined against this immutable robot reference.
    # Every temporal window and SQP sweep uses the same values.
    validator.fixed_contact_errors = {
        id(task): (validator.frame_start, task.errors(validator.reference, validator.frame_start))
        for task in (surface_contact_tasks, self_contact_tasks)
        if task is not None
    }
    guard = WholeBodyGuard(validator, moving)
    objective = WholeBodyObjective(validator, moving)
    active = _active_coordinates(validator, moving)
    report.update(
        status="no_feasible_update",
        fixed_qpos_indices=validator.fixed_qi.tolist(),
        active_joint_names=[
            model.joint(int(jid)).name for jid in validator.joint_ids[active]
        ],
        surface_scope=_surface_scope(
            surface_contact_tasks, result, validator.di[active]
        ),
        surface_before=surface_contact_tasks.metrics(
            result, errors=validator.fixed_contact_errors[id(surface_contact_tasks)][1]
        )
        if surface_contact_tasks is not None
        else None,
        fixed_marker_choices={
            name: values.tolist() for name, values in validator.offset_choice.items()
        },
    )
    try:
        if not validator.fidelity(result)["ok"]:
            report.update(
                reason="baseline_fidelity_failed",
                baseline_fidelity=validator.fidelity(result),
            )
            return result, report
        baseline_tick = time.perf_counter()
        print("Whole-body correction: checking fixed scene baseline", flush=True)
        guard._baseline_geometry()
        report["static_baseline_seconds"] = time.perf_counter() - baseline_tick
        baseline_moving = None
        if moving is not None:
            moving.prepare_trajectory(result)
            reasons, baseline_moving = guard.moving_check(result)
            report["moving_baseline_seconds"] = time.perf_counter() - baseline_tick - report["static_baseline_seconds"]
            report["moving_collision"] = {
                "status": "active",
                "geometry": moving.describe(),
                "before": baseline_moving,
            }
            guard.last_moving = baseline_moving
            if reasons:
                report.update(reason="baseline_moving_geometry_failed", reasons=reasons)
                return result, report
        windowed = len(result) > WHOLE_BODY_WINDOW_THRESHOLD
        if windowed:
            from .whole_body_windows import solve_windowed

            result, window_report = solve_windowed(
                validator, moving, result, cfg, active, guard,
                core_frames=WHOLE_BODY_WINDOW_FRAMES,
                candidate_acceptance=candidate_acceptance,
                baseline_moving=baseline_moving,
            )
            report["iterations"].extend(window_report.pop("iterations"))
            report.update(window_report)
            if moving is not None:
                coverage = [event["moving_coverage"] for event in report["iterations"]]
                report["moving_collision"]["linearization_coverage"] = {
                    key: sum(int(item[key]) for item in coverage)
                    for key in ("poses", "reverse_available_facets", "reverse_selected_facets")
                }
                report["moving_collision"]["linearization_coverage"][
                    "acceptance_checks_all_original_samples"
                ] = True
                report["moving_collision"]["immovable_witnesses"] = [
                    witness for item in coverage
                    for witness in item.get("immovable_witnesses", [])
                ]
        for iteration in range(0 if windowed else int(cfg["iterations"])):
            tick = time.perf_counter()
            guard.collision_rows = []
            if moving is not None:
                moving.prepare_trajectory(result)
                objective.moving_grid = []
                available_facets = selected_facets = 0
                for f in range(len(result)):
                    steps = (
                        1
                        if f + 1 == len(result)
                        else max(
                            moving.interval_steps(
                                poses[f],
                                poses[f + 1],
                                f,
                                cfg["motion_spacing"],
                                cfg["max_substeps"],
                            )
                            for poses in (validator.reference, result)
                        )
                    )
                    for k in range(steps):
                        alpha = k / steps
                        objective.moving_grid.append((f, alpha))
                        pose = interpolated_pose(model, result, f, alpha)
                        query = moving.query(pose, f + alpha, validator.di)
                        if query["failures"]:
                            raise ValueError(
                                "Moving geometry query failed: "
                                + "; ".join(query["failures"])
                            )
                        guard.collision_rows.extend(
                            (f, alpha, row) for row in query["rows"]
                        )
                        available_facets += query["available_rows"]
                        selected_facets += query["selected_rows"]
                report["moving_collision"]["linearization_coverage"] = {
                    "poses": len(objective.moving_grid),
                    "reverse_available_facets": available_facets,
                    "reverse_selected_facets": selected_facets,
                    "acceptance_checks_all_original_samples": True,
                }
                immovable = [
                    (f + alpha, row)
                    for f, alpha, row in guard.collision_rows
                    if row["clearance"] < 0 and np.linalg.norm(row["jacobian"]) < 1e-10
                ]
                report["moving_collision"]["immovable_witnesses"] = [
                    {
                        "frame": f,
                        "body": model.body(int(model.geom_bodyid[r["geom_id"]])).name,
                        "geom_id": r["geom_id"],
                        "depth_m": -r["clearance"],
                        "kind": r["kind"],
                    }
                    for f, r in immovable
                ]
            moving_seconds = time.perf_counter() - tick
            matrix, rhs = objective.linearize(result)
            linearization_seconds = time.perf_counter() - tick - moving_seconds
            print(
                f"Whole-body correction iteration {iteration}: "
                f"moving={moving_seconds:.1f}s linearization={linearization_seconds:.1f}s; solving",
                flush=True,
            )
            change, solve = guard.solve(matrix, rhs, result, active)
            solve["seconds"] = time.perf_counter() - tick - moving_seconds - linearization_seconds
            if change is None:
                report["iterations"].append({"accepted_alpha": 0.0, "solve": solve})
                break
            if np.max(np.abs(change), initial=0) < 1e-7:
                report["iterations"].append(
                    {"accepted_alpha": 0.0, "reason": "stationary", "solve": solve}
                )
                break
            result, acceptance = guard.accept(
                result, change, objective,
                candidate_acceptance=candidate_acceptance,
            )
            acceptance.update(
                iteration=iteration, solve=solve, seconds=time.perf_counter() - tick
            )
            report["iterations"].append(acceptance)
            if acceptance["accepted_alpha"] == 0:
                break
            report["accepted_update_count"] += 1
            report["status"] = "accepted_updates"
            print(
                f"Whole-body correction iteration {iteration}: accepted alpha={acceptance['accepted_alpha']:g}",
                flush=True,
            )
    except (QueryBudgetExceeded, MovingSceneBudgetError) as exc:
        report.update(status="budget_exceeded", reason=str(exc))
    report.update(
        root_invariant=bool(
            np.array_equal(
                result[:, validator.fixed_qi],
                validator.reference[:, validator.fixed_qi],
            )
        ),
        max_joint_correction=float(
            np.max(
                np.abs(result[:, validator.qi] - validator.reference[:, validator.qi]),
                initial=0,
            )
        ),
        surface_after=surface_contact_tasks.metrics(result)
        if surface_contact_tasks is not None
        else None,
        final_fidelity=validator.fidelity(result),
        seconds=time.perf_counter() - started,
    )
    if moving is not None:
        report["moving_collision"].update(
            after=guard.last_moving or report["moving_collision"].get("before"),
            geometry=moving.describe(),
        )
    geometry = guard.last_geometry if guard.last_geometry is not None else guard.before
    if geometry is not None:
        report["final_static_geometry"] = {
            key: geometry[key]
            for key in (
                "max_depth",
                "interframe_max_depth",
                "worst_frame",
                "worst_body",
                "uncertain_frames",
                "uncertain_intervals",
                "subdivision_limit_intervals",
                "self_collision_max_depth",
                "interframe_samples",
            )
        }
    return result, report
