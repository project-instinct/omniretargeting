"""Whole-sequence tangent optimization for mapped robot/source trajectories.

No dataset, clip, robot name, joint count or sampling rate is assumed. All
coordinates come from MuJoCo; positions and contact plans are caller inputs.
This returns a kinematic proposal: actual scene/interframe collision validation
and bounded repair remain mandatory in the owning retargeting pipeline.
"""

import time
from copy import deepcopy

import mujoco
import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear
from scipy.spatial.transform import Rotation

from ._sequence.coordinates import Coordinates, temporal_operators
from ._sequence.surface_guard import SurfaceSequenceGuard
from ._sequence.tasks import SequenceTasks
from ._solver.targets import (
    _apply_configured_joint_limits,
    _apply_joint_limit_safety_fraction,
)

SEQUENCE_DEFAULTS = {
    "iterations": 35,
    "angular_step_rad": 0.12,
    "translation_step_m": 0.03,
    "interaction_sigma_m": 0.5,
    "bone_sigma": 0.12,
    "contact_sigma_m": 0.001,
    "contact_normal_sigma_m": 0.0015,
    "contact_tangent_sigma_m": 0.08,
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
}


def sequence_config(config=None):
    cfg = {**deepcopy(SEQUENCE_DEFAULTS), **dict(config or {})}
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
    protected_tasks = [
        task for task in task_set.extra_tasks
        if getattr(task, "protect_primary_endpoints", False)
        or getattr(task, "contact_phase", None) is not None
    ]
    if not protected_tasks and task_set.contact_phase is None:
        return None
    return SurfaceSequenceGuard(
        task_set, reference,
        min((task.cfg.get("endpoint_regression", 0.001) for task in protected_tasks),
            default=0.001),
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
):
    """Fit an entire timeline with source/contact tasks and physical derivatives.

    Source arrays are copied into immutable task geometry. Contact marker choices
    are fixed per run unless supplied by the caller. Root XY has a soft source
    path reference and continuous offset; its orientation is tracked separately.
    All scalar joint limits, including model-specific safety overrides, apply.
    The same objective is used for every frame and nonlinear backtracking step.
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
    joint_matrices = [
        sparse.kron(op, coords.selector, format="csr") for op in operators
    ]
    if coords.root_q is not None:
        xy_slice = slice(coords.root_q, coords.root_q + 2)
        xyselector = np.zeros((2, width))
        for axis in range(2):
            column = coords.inverse.get(coords.root_d + axis)
            if column is not None:
                xyselector[axis, column] = 1.0
        xyselector = sparse.csr_matrix(xyselector)
        xy_ops = temporal_operators(frames, framerate, cfg["root_xy_derivative_sigmas"])
        xy_matrices = [sparse.kron(op, xyselector, format="csr") for op in xy_ops]
        zq, zd = coords.root_q + 2, coords.inverse.get(coords.root_d + 2)
        quat_slice = slice(coords.root_q + 3, coords.root_q + 7)
        angular_cols = [coords.inverse.get(coords.root_d + d) for d in (3, 4, 5)]
        zselector = (
            sparse.csr_matrix((1, width))
            if zd is None
            else sparse.csr_matrix(([1.0], ([0], [zd])), shape=(1, width))
        )
        height_ops = temporal_operators(
            frames, framerate, cfg["root_height_derivative_sigmas"]
        )
        height_matrices = [
            sparse.kron(op, zselector, format="csr") for op in height_ops
        ]
        angular_ops = temporal_operators(
            frames, framerate, cfg["root_quaternion_derivative_sigmas"]
        )

    def evaluate(poses, root_continuity, with_jacobian=True):
        residuals, matrices = [], []
        statistics = {}
        for frame, pose in enumerate(poses):
            residual, matrix, stats = task_set.frame_rows(pose, frame)
            rr = [
                residual,
                (pose[coords.qi] - reference[frame, coords.qi])
                / cfg["reference_sigma"],
            ]
            jj = [matrix, coords.selector / cfg["reference_sigma"]]
            if coords.root_q is not None:
                rr.append(
                    (pose[xy_slice] - task_set.roots[frame, :2])
                    / cfg["root_xy_sigma_m"]
                )
                jj.append(xyselector / cfg["root_xy_sigma_m"])
                stats["root_xy_offset_max_m"] = float(
                    np.linalg.norm(pose[xy_slice] - task_set.roots[frame, :2])
                )
                stats["root_orientation_max_rad"] = float(
                    (
                        root_rotations[frame].inv()
                        * Rotation.from_quat(pose[quat_slice], scalar_first=True)
                    ).magnitude()
                )
                rr.append(
                    np.array(
                        [(pose[zq] - reference[frame, zq]) / cfg["root_height_sigma_m"]]
                    )
                )
                jj.append(zselector / cfg["root_height_sigma_m"])
            residuals.append(np.concatenate(rr))
            matrices.append(sparse.vstack(jj, format="csr"))
            for key, value in stats.items():
                statistics[key] = max(statistics.get(key, 0.0), value)
        residuals += [(op @ poses[:, coords.qi]).ravel() for op in operators]
        all_matrices = [sparse.block_diag(matrices, format="csr"), *joint_matrices]
        if coords.root_q is not None:
            # Smooth only the correction; fast motion in the original path is
            # not penalized. Use these terms during both continuation phases.
            residuals += [
                (op @ (poses[:, xy_slice] - task_set.roots[:, :2])).ravel()
                for op in xy_ops
            ]
            all_matrices += xy_matrices
        if root_continuity and coords.root_q is not None:
            residuals += [
                (op @ (poses[:, zq] - task_set.roots[:, 2])).ravel()
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
            all_matrices += [
                *height_matrices,
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
    for root_continuity in (False, True):
        if root_continuity and coords.root_q is None:
            continue
        for iteration in range(iterations):
            residual, matrix, stats = evaluate(q, root_continuity)
            cost = float(residual @ residual)
            policy = task_set.contact_phase
            support_before = surface_guard.priority_errors(q) if policy is not None else None
            lower, upper = coords.bounds(
                q, cfg["angular_step_rad"], cfg["translation_step_m"]
            )
            if surface_guard is not None:
                solution = surface_guard.solve(matrix, -residual, lower, upper, q)
            else:
                solution = lsq_linear(
                    matrix,
                    -residual,
                    bounds=(lower.ravel(), upper.ravel()),
                    lsmr_tol=1e-4,
                    tol=1e-4,
                    max_iter=50,
                )
            if not np.all(np.isfinite(solution.x)):
                raise RuntimeError(
                    "Sequence linear solver returned nonfinite coordinates"
                )
            step = solution.x.reshape(frames, width)
            accepted = False
            for alpha in (
                1.0,
                0.5,
                0.25,
                0.125,
                0.0625,
                0.03125,
                0.015625,
                0.0078125,
                0.00390625,
            ):
                candidate = coords.integrate(q, step, alpha)
                new_residual, _, new_stats = evaluate(candidate, root_continuity, False)
                new_cost = float(new_residual @ new_residual)
                support_ok, support_better = True, False
                if policy is not None:
                    support_after = surface_guard.priority_errors(candidate)
                    support_ok = bool(np.all(support_after <= support_before + policy.tolerance))
                    support_better = bool(np.sum(support_after ** 2) < np.sum(support_before ** 2) - 1e-10)
                    new_stats["contact_role_rms_max_m"] = float(support_after.max(initial=0))
                if (np.isfinite(new_cost) and support_ok and (new_cost < cost or support_better)
                        and (surface_guard is None or surface_guard.accepts(candidate))):
                    accepted = True
                    q, stats = candidate, new_stats
                    break
            if not accepted:
                alpha, new_cost = 0.0, cost
            record = dict(
                phase="root_continuity" if root_continuity else "pose",
                iteration=iteration,
                cost=cost,
                after=new_cost,
                alpha=alpha,
                linear_solver_status=int(solution.status),
                seconds=time.monotonic() - start,
                **stats,
            )
            history.append(record)
            if callback is not None:
                callback(q.copy(), dict(record))
            if not accepted:
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
        "validation_required": "Original moving scene, interframes, source/end-effector errors and temporal metrics",
        "surface_endpoint_guard": {
            "enabled": surface_guard is not None,
            "linear_failures": surface_guard.failures if surface_guard is not None else 0,
            "reference": "immutable_input_trajectory",
        },
        "contact_phase_priority": (task_set.contact_phase.diagnostics
                                   if task_set.contact_phase is not None else None),
    }
    return q, report
