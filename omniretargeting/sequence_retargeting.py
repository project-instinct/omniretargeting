"""Whole-sequence tangent optimization for mapped robot/source trajectories.

No dataset, clip, robot name, joint count or sampling rate is assumed. All
coordinates come from MuJoCo; positions and contact plans are caller inputs.
This returns a kinematic proposal: actual scene/interframe collision validation
and bounded repair remain mandatory in the owning retargeting pipeline.
"""

import os
import time
from copy import deepcopy
from functools import partial

import mujoco
import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation

from ._sequence.coordinates import Coordinates, temporal_operators
from ._sequence.linear import bounded_step
from ._sequence.surface_guard import SurfaceSequenceGuard
from ._sequence.tasks import SequenceTasks
from ._sequence.windows import (
    independent_groups,
    run_forked,
    temporal_windows,
    window_system,
)
from ._solver.targets import (
    _apply_configured_joint_limits,
    _apply_joint_limit_safety_fraction,
)
from .geometry_query import GeometryQueryError

# Numerical termination, not a physical tolerance. Consecutive accepted steps
# below this relative objective decrease, while no frame's contact priority
# improves by its full tolerance, only refine a converged trajectory at
# round-off level. Slower decreases still smooth secondary temporal terms
# and must continue: a larger threshold left measurable root acceleration.
CONVERGED_RELATIVE_DECREASE = 1e-6
CONVERGED_STEPS = 2
LINE_SEARCH_FRACTIONS = (
    1.0,
    0.5,
    0.25,
    0.125,
    0.0625,
    0.03125,
    0.015625,
    0.0078125,
    0.00390625,
)

# Frames per block-coordinate window in guarded correction passes. Longer
# clips alternate half-shifted windows; shorter clips solve globally.
WINDOW_FRAMES = 48
# Forked processes solving independent windows of one group; 1 is serial.
# Results are identical for any value: every member starts from the same
# iterate and only its own frames are merged.
WINDOW_WORKERS = max(1, min(16, (os.cpu_count() or 2) // 2))

SEQUENCE_DEFAULTS = {
    "iterations": 35,
    "angular_step_rad": 0.12,
    "translation_step_m": 0.03,
    "interaction_sigma_m": 0.5,
    "bone_sigma": 0.12,
    "contact_sigma_m": 0.001,
    "contact_normal_sigma_m": 0.0015,
    # Sliding follows the recorded, time-varying surface point. It is not a
    # license to follow the infinite tangent plane far outside a finite patch.
    "contact_tangent_sigma_m": 0.0015,
    "endpoint_regression_m": 0.001,
    "world_tracking_sigma_m": 0.02,
    "free_tracking_sigma_m": 0.05,
    "orientation_sigma": 0.2,
    "reference_sigma": 2.0,
    "root_height_sigma_m": 0.2,
    "root_xy_sigma_m": 0.25,
    "root_orientation_sigma_rad": 0.05,
    "root_xy_derivative_sigmas": [0.8, 3.6, 108.0],  # m/s, m/s², m/s³ of offset
    "joint_derivative_sigmas": [15.0, 22.5, 675.0],  # rad/s, rad/s², rad/s³
    "root_height_derivative_sigmas": [1.2, 3.6, 108.0],  # m/s, m/s², m/s³
    "root_quaternion_derivative_sigmas": [9.0, 5.4, 162.0],  # component/s^order
    "collision_spacing_m": 0.03,
    "collision_sigma_m": 0.001,
    "scene_clearance_m": 0.001,
    "self_clearance_m": 0.002,
    "self_collision_enabled": True,
    "joint_limit_safety_fraction": 1.0,
    "lock_root_xy": False,
    "lock_root_z": False,
    "lock_root_orientation": False,
    "staged_contact_repair": False,
}


def sequence_config(config=None):
    from .config_validation import known_options

    known_options(
        config,
        SEQUENCE_DEFAULTS,
        "sequence",
        extra=(
            "enabled",
            "backend",
            "pose_axes",
            "self_collision",
            "source_side_support",
            "source_shape_priority",
            "moving_scene",
            "linear_solver",
            "linear_solver_tolerance",
            "linear_solver_iterations",
            "frame_iterations",
            "qp_solver",
        ),
    )
    cfg = {**deepcopy(SEQUENCE_DEFAULTS), **dict(config or {})}
    if not isinstance(cfg["staged_contact_repair"], (bool, np.bool_)):
        raise ValueError("sequence staged_contact_repair must be boolean")
    iterations = cfg["iterations"]
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations < 1
    ):
        raise ValueError("sequence iterations must be a positive integer")
    for key in (k for k, v in SEQUENCE_DEFAULTS.items() if isinstance(v, float)):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"sequence {key} must be finite and positive")
    return cfg


def prepare_sequence(
    model,
    initial_trajectory,
    source_positions,
    source_names,
    joint_mapping,
    source_root_positions,
    tasks,
    end_effector_config,
    cfg,
    *,
    environment_points=None,
    scene_query=None,
    contact_offset_choice=None,
    extra_tasks=(),
    root_orientation_targets=None,
    body_orientation_targets=(),
):
    """Coordinates, limits and per-frame task rows shared by both backends."""
    coords = Coordinates(
        model, cfg["lock_root_xy"], cfg["lock_root_z"], cfg["lock_root_orientation"]
    )
    coords.lower, coords.upper = _apply_joint_limit_safety_fraction(
        model, coords.qi, coords.lower, coords.upper, cfg["joint_limit_safety_fraction"]
    )
    coords.lower, coords.upper = _apply_configured_joint_limits(
        model,
        coords.qi,
        coords.lower,
        coords.upper,
        end_effector_config.get("joint_posture_limits"),
    )
    q = coords.normalize(initial_trajectory)
    frames = len(q)
    if not frames or not coords.width:
        return coords, q, None, None
    orientation_targets = list(body_orientation_targets)
    root_rotations = None
    if coords.root_q is not None:
        if root_orientation_targets is None:
            root_orientation_targets = q[:, coords.root_q + 3 : coords.root_q + 7]
        rotations = np.asarray(root_orientation_targets, dtype=float)
        if (
            rotations.shape != (frames, 4)
            or not np.all(np.isfinite(rotations))
            or np.any(np.linalg.norm(rotations, axis=1) < 1e-12)
        ):
            raise ValueError(
                "root_orientation_targets must be finite nonzero (frames, 4) wxyz"
            )
        root_body = int(
            model.jnt_bodyid[
                np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
            ]
        )
        root_rotations = Rotation.from_quat(rotations, scalar_first=True)
        orientation_targets.append(
            {
                "robot_body": root_body,
                "rotations": root_rotations.as_matrix(),
                "sigma": cfg["root_orientation_sigma_rad"],
            }
        )
    task_set = SequenceTasks(
        model,
        coords,
        q,
        source_positions,
        source_names,
        joint_mapping,
        source_root_positions,
        tasks,
        end_effector_config,
        cfg,
        environment_points=environment_points,
        scene_query=scene_query,
        contact_offset_choice=contact_offset_choice,
        extra_tasks=extra_tasks,
        orientation_targets=orientation_targets,
    )
    return coords, q, task_set, root_rotations


def priority_guard(task_set, reference):
    """Role-ordered levels and fixed budgets when surface/role tasks exist."""
    if task_set.config.get("source_shape_priority", False):
        from ._sequence.source_shape import SourceShapeGuard

        return SourceShapeGuard(task_set)
    protected_tasks = [
        task
        for task in task_set.extra_tasks
        if getattr(task, "protect_primary_endpoints", False)
        or getattr(task, "contact_phase", None) is not None
    ]
    if not protected_tasks and task_set.contact_phase is None:
        return None
    return SurfaceSequenceGuard(
        task_set,
        reference,
        min(
            (task.cfg.get("endpoint_regression", 0.001) for task in protected_tasks),
            default=0.001,
        ),
        preserve_contacts=task_set.contact_phase is None,
    )


def retarget_sequence(
    model,
    initial_trajectory,
    source_positions,
    source_names,
    joint_mapping,
    source_root_positions,
    tasks,
    end_effector_config,
    *,
    framerate,
    config=None,
    environment_points=None,
    scene_query=None,
    contact_offset_choice=None,
    extra_tasks=(),
    root_orientation_targets=None,
    body_orientation_targets=(),
    callback=None,
    endpoint_reference_trajectory=None,
    correction_reference_trajectory=None,
    contact_stage=None,
):
    """Fit an entire timeline with source/contact tasks and physical derivatives.

    Source arrays are copied into immutable task geometry. Contact marker choices
    are fixed per run unless supplied by the caller. Root XY has a soft source
    path reference and continuous offset; its orientation is tracked separately.
    All scalar joint limits, including model-specific safety overrides, apply.
    The same objective is used for every frame and nonlinear backtracking step.
    ``correction_reference_trajectory`` fixes cumulative posture/continuity
    budgets across stages even when an improved endpoint reference is acquired.
    ``contact_stage`` additionally fixes the attained vectors of other contacts
    while one welded body is corrected; it never replaces those fixed budgets.
    """
    cfg = sequence_config(config)
    iterations = cfg["iterations"]
    coords, q, task_set, root_rotations = prepare_sequence(
        model,
        initial_trajectory,
        source_positions,
        source_names,
        joint_mapping,
        source_root_positions,
        tasks,
        end_effector_config,
        cfg,
        environment_points=environment_points,
        scene_query=scene_query,
        contact_offset_choice=contact_offset_choice,
        extra_tasks=extra_tasks,
        root_orientation_targets=root_orientation_targets,
        body_orientation_targets=body_orientation_targets,
    )
    frames, width = len(q), coords.width
    if task_set is None:
        return q, {"status": "empty", "history": [], "framerate": framerate}
    operators = temporal_operators(frames, framerate, cfg["joint_derivative_sigmas"])
    reference = q.copy()
    surface_guard = priority_guard(task_set, reference)
    temporal_reference = None
    if (
        endpoint_reference_trajectory is not None
        or correction_reference_trajectory is not None
    ):
        endpoint_reference = coords.normalize(
            reference
            if endpoint_reference_trajectory is None
            else endpoint_reference_trajectory
        )
        if endpoint_reference.shape != reference.shape:
            raise ValueError(
                "Endpoint reference must share the complete trajectory shape"
            )
        surface_guard = SurfaceSequenceGuard(
            task_set,
            endpoint_reference,
            cfg["endpoint_regression_m"],
            preserve_contacts=endpoint_reference_trajectory is not None,
        )
        from ._sequence.continuity_guard import SequenceContinuityGuard
        from ._sequence.pose_guard import SequencePoseGuard

        correction_reference = coords.normalize(
            endpoint_reference
            if correction_reference_trajectory is None
            else correction_reference_trajectory
        )
        if correction_reference.shape != reference.shape:
            raise ValueError(
                "Correction reference must share the complete trajectory shape"
            )
        # The source-shaped robot pose is fixed before contact acquisition.
        # Penalize only changes to that pose in subsequent stages, so a fast
        # source-authored joint movement is not itself treated as chatter.
        temporal_reference = correction_reference[:, coords.qi]
        surface_guard.continuity = SequenceContinuityGuard(
            task_set, correction_reference, root_rotations
        )
        surface_guard.pose = SequencePoseGuard(task_set, correction_reference)
        if correction_reference_trajectory is not None and (
            not surface_guard.pose.accepts(q) or not surface_guard.continuity.accepts(q)
        ):
            raise ValueError(
                "Initial trajectory exceeds the fixed correction reference budgets"
            )
    if contact_stage is not None:
        if endpoint_reference_trajectory is None or correction_reference_trajectory is None:
            raise ValueError("Contact stages require fixed contact and correction references")
        surface_guard.contact_stage = contact_stage.bind(surface_guard)
        if not surface_guard.contact_stage.accepts(q):
            raise ValueError("Initial trajectory exceeds the contact stage hold budgets")
        if contact_stage.objective is not None:
            # Only the objective changes reference here. Hard derivative and
            # posture bounds still use the immutable pre-acquisition reference.
            temporal_reference = contact_stage.objective.reference[:, coords.qi]
    joint_matrices = [
        sparse.kron(op, coords.selector, format="csr") for op in operators
    ]
    if coords.root_q is not None:
        xy_ops = temporal_operators(frames, framerate, cfg["root_xy_derivative_sigmas"])
        quat_slice = slice(coords.root_q + 3, coords.root_q + 7)
        angular_cols = [coords.inverse.get(coords.root_d + d) for d in (3, 4, 5)]
        height_ops = temporal_operators(
            frames, framerate, cfg["root_height_derivative_sigmas"]
        )
        angular_ops = temporal_operators(
            frames, framerate, cfg["root_quaternion_derivative_sigmas"]
        )

    frame_cache = [None] * frames
    root_reference = []
    for frame, pose in enumerate(reference):
        task_set.data.qpos[:] = pose
        mujoco.mj_forward(model, task_set.data)
        root_reference.append(task_set.root_anchor(frame)[0])
    root_reference = np.asarray(root_reference)

    def evaluate(poses, root_continuity, with_jacobian=True):
        if task_set.scene_cache is not None:
            task_set.scene_cache.prepare(poses)
        for task in task_set.extra_tasks:
            if hasattr(task, "prepare_trajectory"):
                task.prepare_trajectory(poses)
        residuals, matrices = [], []
        root_points, root_jacobians = [], []
        statistics = {}
        for frame, pose in enumerate(poses):
            key = np.ascontiguousarray(pose).tobytes()
            cached = frame_cache[frame]
            if cached is None or cached[0] != key:
                rows = task_set.frame_rows(pose, frame)
                if contact_stage is not None and contact_stage.objective is not None:
                    stage_rows = contact_stage.objective.linearize(task_set, pose, frame)
                    rows = (*stage_rows, rows[2])
                cached = (key, *rows, *task_set.root_anchor(frame))
                frame_cache[frame] = cached
            residual, matrix, stored_stats, root_point, root_jac = cached[1:]
            root_points.append(root_point)
            root_jacobians.append(root_jac)
            stats = dict(stored_stats)
            rr = [
                residual,
                (pose[coords.qi] - reference[frame, coords.qi])
                / cfg["reference_sigma"],
            ]
            jj = [matrix, coords.selector / cfg["reference_sigma"]]
            if coords.root_q is not None:
                rr.append(
                    (root_point[:2] - task_set.roots[frame, :2])
                    / cfg["root_xy_sigma_m"]
                )
                jj.append(root_jac[:2] / cfg["root_xy_sigma_m"])
                stats["root_xy_offset_max_m"] = float(
                    np.linalg.norm(root_point[:2] - task_set.roots[frame, :2])
                )
                stats["root_orientation_max_rad"] = float(
                    (
                        root_rotations[frame].inv()
                        * Rotation.from_quat(pose[quat_slice], scalar_first=True)
                    ).magnitude()
                )
                rr.append(
                    np.array(
                        [
                            (root_point[2] - root_reference[frame, 2])
                            / cfg["root_height_sigma_m"]
                        ]
                    )
                )
                jj.append(root_jac[2:3] / cfg["root_height_sigma_m"])
            residuals.append(np.concatenate(rr))
            matrices.append(sparse.vstack(jj, format="csr"))
            for key, value in stats.items():
                statistics[key] = max(statistics.get(key, 0.0), value)
        joint_motion = poses[:, coords.qi]
        if temporal_reference is not None:
            joint_motion = joint_motion - temporal_reference
        residuals += [(op @ joint_motion).ravel() for op in operators]
        all_matrices = [sparse.block_diag(matrices, format="csr"), *joint_matrices]
        root_points = np.asarray(root_points)
        if coords.root_q is not None:
            # Smooth only the correction; fast motion in the original path is
            # not penalized. Use these terms during both continuation phases.
            residuals += [
                (op @ (root_points[:, :2] - task_set.roots[:, :2])).ravel()
                for op in xy_ops
            ]
            root_xy_jac = sparse.block_diag(
                [j[:2] for j in root_jacobians], format="csr"
            )
            all_matrices += [
                sparse.kron(op, sparse.eye(2), format="csr") @ root_xy_jac
                for op in xy_ops
            ]
        # Every root coordinate needs continuity during contact acquisition,
        # too. A spatial-only warm phase can select alternating IK branches
        # whose height/orientation jumps are expensive to undo afterwards.
        if coords.root_q is not None:
            residuals += [
                (op @ (root_points[:, 2] - task_set.roots[:, 2])).ravel()
                for op in height_ops
            ]
            # Smooth orientation error, not the deliberate source rotation.
            relative_quaternions = (
                root_rotations.inv()
                * Rotation.from_quat(poses[:, quat_slice], scalar_first=True)
            ).as_quat(scalar_first=True, canonical=True)
            residuals += [(op @ relative_quaternions).ravel() for op in angular_ops]
            angular_blocks = []
            for quaternion in relative_quaternions:
                w, x, y, z = quaternion
                block = np.zeros((4, width))
                quaternion_basis = 0.5 * np.array(
                    [[-x, -y, -z], [w, -z, y], [z, w, -x], [-y, x, w]]
                )
                if not cfg["lock_root_orientation"]:
                    block[:, angular_cols] = quaternion_basis
                angular_blocks.append(sparse.csr_matrix(block))
            angular_jac = sparse.block_diag(angular_blocks, format="csr")
            root_z_jac = sparse.block_diag(
                [j[2:3] for j in root_jacobians], format="csr"
            )
            all_matrices += [
                *[op @ root_z_jac for op in height_ops],
                *[
                    sparse.kron(op, sparse.eye(4), format="csr") @ angular_jac
                    for op in angular_ops
                ],
            ]
        return (
            np.concatenate(residuals),
            sparse.vstack(all_matrices, format="csr") if with_jacobian else None,
            statistics,
        )

    history = []
    start = time.monotonic()
    # Guarded correction passes of long clips solve the same linearized
    # problem window by window, all other frames held fixed (see
    # _sequence/windows.py). Every hard bound and acceptance check is
    # unchanged; the unguarded source-pose pass stays one global solve.
    windowed = (
        endpoint_reference_trajectory is not None
        or correction_reference_trajectory is not None
    ) and frames > WINDOW_FRAMES

    def windowed_call(method, *args, window=None):
        return method(*args) if window is None else method(*args, frames=window)

    def attempt(window, root_continuity, residual, matrix, stats, q):
        """One guarded step from iterate ``q`` over ``window`` (None: all frames).

        Returns the accepted poses of the window rows; ``q`` is not modified.
        """
        iteration_start = time.monotonic()
        cost = float(residual @ residual)
        span = slice(None) if window is None else slice(window.start, window.stop)
        rhs = -residual
        if window is not None:
            matrix, rhs = window_system(matrix, residual, window, width)
        policy = task_set.contact_phase
        has_priority = policy is not None or cfg.get("source_shape_priority", False)
        support_before = (
            windowed_call(surface_guard.priority_errors, q, window=window)
            if has_priority
            else None
        )
        lower, upper = coords.bounds(
            q[span], cfg["angular_step_rad"], cfg["translation_step_m"]
        )
        linear_start = time.monotonic()
        if surface_guard is not None:
            solution = windowed_call(
                surface_guard.solve, matrix, rhs, lower, upper, q, window=window
            )
        else:
            solution = bounded_step(matrix, rhs, lower.ravel(), upper.ravel(), cfg)
        linear_end = time.monotonic()
        if not np.all(np.isfinite(solution.x)):
            raise RuntimeError("Sequence linear solver returned nonfinite coordinates")
        step = solution.x.reshape(lower.shape)
        if cfg.get("source_shape_priority", False):
            step = windowed_call(surface_guard.feasible_step, q, step, window=window)
        if window is not None:
            full = np.zeros((frames, width))
            full[span] = step
            step = full
        accepted = progressed = False
        # A zero step (for example a failed constrained subproblem)
        # reproduces the current iterate at every fraction, so it can
        # never be accepted; skip those full trajectory evaluations.
        for alpha in LINE_SEARCH_FRACTIONS if np.any(step) else ():
            candidate = coords.integrate(q, step, alpha)
            # Endpoint FK is much cheaper than querying every original
            # scene triangle. Reject the same inadmissible candidates
            # before geometry work, preserving the acceptance contract.
            if surface_guard is not None and not windowed_call(
                surface_guard.accepts, candidate, window=window
            ):
                # Restore the curved feasible set before shrinking further.
                # Contact/cost priorities are checked below;
                # projection is never permission to widen a hard bound.
                candidate = (
                    windowed_call(
                        surface_guard.restore,
                        q,
                        candidate,
                        lower,
                        upper,
                        alpha,
                        window=window,
                    )
                    if hasattr(surface_guard, "restore")
                    else None
                )
                if candidate is None:
                    continue
            try:
                new_residual, _, new_stats = evaluate(
                    candidate, root_continuity, False
                )
            except GeometryQueryError:
                # An unreliable trial must never be accepted as free space.
                # Backtrack; an unreliable initial iterate still fails above.
                continue
            new_cost = float(new_residual @ new_residual)
            support_ok, support_better = True, False
            if has_priority:
                support_after = windowed_call(
                    surface_guard.priority_errors, candidate, window=window
                )
                from .contact_phase import priority_progress

                # A rejected full step may still admit smaller progress
                # inside the fixed pose envelope. Scale the sufficient
                # decrease AND allowed regression with the trial fraction;
                # a millimetre deadband must not erase all such progress.
                support_ok, support_better = priority_progress(
                    support_before,
                    support_after,
                    alpha
                    * (
                        policy.tolerance
                        if policy is not None
                        else surface_guard.priority_tolerance
                    ),
                )
                new_stats["contact_role_rms_max_m"] = float(
                    support_after.max(initial=0)
                )
            if (
                np.isfinite(new_cost)
                and support_ok
                and (new_cost < cost or support_better)
            ):
                accepted = True
                if support_better:
                    # Termination counts only full-tolerance priority gains,
                    # not the fraction-scaled acceptance deadband above.
                    progressed = priority_progress(
                        support_before,
                        support_after,
                        policy.tolerance
                        if policy is not None
                        else surface_guard.priority_tolerance,
                    )[1]
                q, stats = candidate, new_stats
                break
        if not accepted:
            alpha, new_cost = 0.0, cost
        record = dict(
            phase="root_continuity" if root_continuity else "pose",
            cost=cost,
            after=new_cost,
            alpha=alpha,
            priority_progress=bool(progressed),
            linear_solver_status=int(solution.status),
            linear_solver_backend=getattr(solution, "backend", "priority_guard"),
            evaluation_seconds=linear_start - iteration_start,
            linear_solve_seconds=linear_end - linear_start,
            acceptance_seconds=time.monotonic() - linear_end,
            **stats,
        )
        span = slice(None) if window is None else slice(window.start, window.stop)
        return accepted, progressed, record, q[span].copy()

    # Both passes evaluate the same objective. After a pass stops on rejected
    # or converged steps, re-solving the identical deterministic subproblems can
    # only repeat that outcome; a pass that used its whole budget continues.
    finished = False
    slow_steps = 0
    for root_continuity in (False, True):
        if root_continuity and coords.root_q is None:
            continue
        # Windowed passes count whole sweeps; continuing the identical
        # objective in a second pass would only double that budget.
        if finished or (root_continuity and windowed):
            break
        idle_sweeps = 0
        for iteration in range(iterations):
            schedule = (
                temporal_windows(frames, WINDOW_FRAMES, iteration)
                if windowed
                else [None]
            )
            accepted_any = progressed_any = False
            sweep_cost = None
            for group in independent_groups(schedule) if windowed else [schedule]:
                if not group:
                    continue
                # Re-linearize before every group: collision rows appear and
                # vanish with the pose, so row layouts cannot be reused. All
                # members start from, and are accepted against, this iterate.
                evaluation_start = time.monotonic()
                residual, matrix, stats = evaluate(q, root_continuity)
                if windowed:
                    matrix = matrix.tocsc()
                evaluation = time.monotonic() - evaluation_start
                if sweep_cost is None:
                    sweep_cost = float(residual @ residual)
                base = q
                jobs = [
                    partial(attempt, window, root_continuity, residual, matrix, stats, base)
                    for window in group
                ]
                results = run_forked(jobs, WINDOW_WORKERS if windowed else 1)
                q = base.copy()
                for window, (accepted, progressed, record, poses) in zip(group, results):
                    if accepted:
                        span = (
                            slice(None)
                            if window is None
                            else slice(window.start, window.stop)
                        )
                        q[span] = poses
                    accepted_any |= accepted
                    progressed_any |= progressed
                    record["evaluation_seconds"] += evaluation
                    record.update(
                        iteration=iteration,
                        seconds=time.monotonic() - start,
                        window=None if window is None else [window.start, window.stop],
                        window_index=schedule.index(window),
                        window_count=len(schedule),
                    )
                    history.append(record)
                    if callback is not None:
                        callback(q.copy(), dict(record))
            # Members of a group are separated by a whole window, so their row
            # changes add; the merged sweep is re-evaluated once for its cost.
            end_cost = (
                float(np.square(evaluate(q, root_continuity, False)[0]).sum())
                if windowed
                else record["after"]
            )
            slow = not progressed_any and (
                sweep_cost - end_cost <= CONVERGED_RELATIVE_DECREASE * abs(sweep_cost)
            )
            slow_steps = slow_steps + 1 if accepted_any and slow else 0
            # A rejected global step repeats itself exactly. Shifted windows
            # differ between sweeps, so windows stop after one idle pair.
            idle_sweeps = 0 if accepted_any else idle_sweeps + 1
            if idle_sweeps >= (2 if windowed else 1) or slow_steps >= CONVERGED_STEPS:
                finished = True
                break
    if cfg["lock_root_xy"] and coords.root_q is not None:
        assert np.array_equal(
            q[:, coords.root_q : coords.root_q + 2],
            reference[:, coords.root_q : coords.root_q + 2],
        )
    report = {
        "status": "solved",
        "framerate": float(framerate),
        "frames": frames,
        "model_nq": model.nq,
        "model_nv": model.nv,
        "scalar_joint_count": len(coords.qi),
        "config": cfg,
        "history": history,
        "contact_offset_choice": {k: v.tolist() for k, v in task_set.choices.items()},
        "final_residuals": evaluate(q, coords.root_q is not None, False)[2],
        "source_shape": (
            surface_guard.diagnostics(q)
            if cfg.get("source_shape_priority", False)
            else None
        ),
        "posture_envelope": (
            surface_guard.pose.diagnostics(q)
            if surface_guard is not None and surface_guard.pose is not None
            else None
        ),
        "validation_required": "Original moving scene, interframes, source/end-effector errors and temporal metrics",
        "surface_endpoint_guard": {
            "enabled": surface_guard is not None,
            "linear_failures": surface_guard.failures
            if surface_guard is not None
            else 0,
            "preserves_attained_contacts": (
                surface_guard.preserve_contacts if surface_guard is not None else False
            ),
            "reference": "explicit_endpoint_reference"
            if endpoint_reference_trajectory is not None
            else "immutable_input_trajectory",
            "correction_reference": "explicit_fixed_reference"
            if correction_reference_trajectory is not None
            else "endpoint_reference",
        },
        "contact_phase_priority": (
            task_set.contact_phase.diagnostics
            if task_set.contact_phase is not None
            else None
        ),
        "contact_stage": (
            surface_guard.contact_stage.diagnostics(q)
            if getattr(surface_guard, "contact_stage", None) is not None
            else None
        ),
    }
    if task_set.contact_phase is not None:
        from .contact_phase import CONTACT_ROLES

        report["contact_role_residuals"] = {}
        for level, role in enumerate(CONTACT_ROLES):
            errors = np.array(
                [
                    np.sqrt(
                        sum(
                            w * float(r @ r)
                            for w, r, _ in surface_guard.priority_rows(pose, f, level)
                        )
                    )
                    for f, pose in enumerate(q)
                ]
            )
            report["contact_role_residuals"][role] = {
                "weighted_position_norm_max_m": float(errors.max(initial=0)),
                "max_frame": int(errors.argmax()),
                "dynamics_certificate": False,
            }
    return q, report
