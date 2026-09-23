"""Per-frame constrained Gauss-Newton retargeting on mink, following UMR.

UMR (arXiv 2609.02134, Table I: 121 FPS on LAFAN1) solves one small
constrained Gauss-Newton QP per frame; its code calls Clarabel directly.
This backend keeps the whole-sequence objective rows, contact roles and
approach hand-over, but solves each frame in the robot tangent space with
mink tasks and limits. Temporal terms are causal differences to the frames
already solved, with the sequence solver's per-second sigmas. Role levels
(clearance > support > surface support > follow) are solved in order; each
level's achieved task values bound the later levels.
"""

import time

import mink
import numpy as np
import qpsolvers
from mink.limits.limit import Constraint
from scipy.spatial.transform import Rotation

from ._sequence.surface_guard import LEVELS
from .contact_phase import PRIORITY_TOLERANCE
from .sequence_retargeting import prepare_sequence, priority_guard, sequence_config

MINK_DEFAULTS = {"frame_iterations": 10, "qp_solver": "clarabel"}
# Causal backward differences of order 1-3 (current frame first).
_DIFFERENCES = ((1.0, -1.0), (1.0, -2.0, 1.0), (1.0, -3.0, 3.0, -1.0))


class _Rows(mink.Task):
    """Stacked rows ``r + J dq`` as one unit-cost mink task."""

    def __init__(self, residual, jacobian):
        super().__init__(cost=np.ones(len(residual)))
        self.residual, self.jacobian = residual, jacobian

    def compute_error(self, configuration):
        return self.residual

    def compute_jacobian(self, configuration):
        return self.jacobian


class _Linear(mink.Limit):
    """Fixed linear inequalities ``G dq <= h``."""

    def __init__(self, G, h):
        self.G, self.h = G, h

    def compute_qp_inequalities(self, configuration, dt):
        return Constraint(G=self.G, h=self.h)


def _quaternion_basis(quaternion):
    w, x, y, z = quaternion
    return 0.5 * np.array([[-x, -y, -z], [w, -z, y], [z, w, -x], [-y, x, w]])


class _FrameProblem:
    """Rows of one frame: the sequence objective with causal temporal terms."""

    def __init__(self, model, coords, reference, task_set, root_rotations, cfg, fps):
        self.model, self.coords, self.reference = model, coords, reference
        self.tasks, self.root_rotations, self.cfg, self.fps = (
            task_set, root_rotations, cfg, fps
        )
        width = coords.width
        self.selector = coords.selector.toarray()
        self.xy = self.z = self.angular = None
        if coords.root_q is not None:
            self.xy = np.zeros((2, width))
            for axis in range(2):
                column = coords.inverse.get(coords.root_d + axis)
                if column is not None:
                    self.xy[axis, column] = 1.0
            self.z = np.zeros((1, width))
            column = coords.inverse.get(coords.root_d + 2)
            if column is not None:
                self.z[0, column] = 1.0
            self.angular = [coords.inverse.get(coords.root_d + d) for d in (3, 4, 5)]

    def relative_rotation(self, pose, frame):
        start = self.coords.root_q + 3
        rotation = Rotation.from_quat(pose[start : start + 4], scalar_first=True)
        return (self.root_rotations[frame].inv() * rotation).as_quat(
            scalar_first=True, canonical=True
        )

    def rows(self, pose, frame, history):
        """``history`` holds the solved poses of frames f-1, f-2, f-3."""
        cfg, coords = self.cfg, self.coords
        residual, matrix, stats = self.tasks.frame_rows(pose, frame)
        rr, jj = [residual], [matrix.toarray()]

        def add(value, jacobian, sigma):
            rr.append(np.atleast_1d(value) / sigma)
            jj.append(np.atleast_2d(jacobian) / sigma)

        add(pose[coords.qi] - self.reference[frame, coords.qi], self.selector,
            cfg["reference_sigma"])
        roots = self.tasks.roots
        if coords.root_q is not None:
            xy, zq = slice(coords.root_q, coords.root_q + 2), coords.root_q + 2
            add(pose[xy] - roots[frame, :2], self.xy, cfg["root_xy_sigma_m"])
            add(pose[zq] - self.reference[frame, zq], self.z, cfg["root_height_sigma_m"])
        poses = [pose, *history]
        for order, coefficients in enumerate(_DIFFERENCES):
            if len(poses) < len(coefficients):
                break
            scale = self.fps ** (order + 1)
            lead = coefficients[0] * scale

            def difference(values):
                return scale * sum(c * v for c, v in zip(coefficients, values))

            add(difference([p[coords.qi] for p in poses]), lead * self.selector,
                cfg["joint_derivative_sigmas"][order])
            if coords.root_q is None:
                continue
            frames = range(frame, frame - len(coefficients), -1)
            add(difference([p[xy] - roots[f, :2] for p, f in zip(poses, frames)]),
                lead * self.xy, cfg["root_xy_derivative_sigmas"][order])
            add(difference([p[zq] - roots[f, 2] for p, f in zip(poses, frames)]),
                lead * self.z, cfg["root_height_derivative_sigmas"][order])
            relative = [self.relative_rotation(p, f) for p, f in zip(poses, frames)]
            jacobian = np.zeros((4, coords.width))
            for axis, column in enumerate(self.angular):
                if column is not None:
                    jacobian[:, column] = _quaternion_basis(relative[0])[:, axis]
            add(difference(relative), lead * jacobian,
                cfg["root_quaternion_derivative_sigmas"][order])
        return np.concatenate(rr), np.vstack(jj), stats


def retarget_mink(
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
    """Same inputs and outputs as ``retarget_sequence``, solved frame by frame."""
    config = dict(config or {})
    options = {**MINK_DEFAULTS, **{k: config.pop(k) for k in MINK_DEFAULTS if k in config}}
    cfg = sequence_config(config)
    coords, reference, task_set, root_rotations = prepare_sequence(
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
    frames = len(reference)
    if task_set is None:
        return reference, {"status": "empty", "history": [], "framerate": framerate}
    fps = float(framerate)
    problem = _FrameProblem(model, coords, reference, task_set, root_rotations, cfg, fps)
    guard = priority_guard(task_set, reference) if task_set.contact_phase else None
    configuration = mink.Configuration(model)
    nv, cols = model.nv, coords.cols
    failures = 0

    def full(jacobian):
        out = np.zeros((len(jacobian), nv))
        out[:, cols] = jacobian
        return out

    def level_rows(pose, frame, level):
        rows = list(guard.priority_rows(pose, frame, level))
        if not rows:
            return None
        return (
            np.concatenate([np.sqrt(w) * np.atleast_1d(r) for w, r, _ in rows]),
            full(np.vstack([np.sqrt(w) * np.atleast_2d(j) for w, _, j in rows])),
        )

    def level_errors(pose, frame):
        errors = []
        for level in range(LEVELS if guard else 0):
            rows = level_rows(pose, frame, level)
            errors.append(0.0 if rows is None else float(np.linalg.norm(rows[0])))
        return np.array(errors)

    def solve(pose, rows, limits):
        configuration.update(pose)
        qp = mink.build_ik(configuration, [_Rows(*rows)], dt=1.0, damping=1e-8,
                           limits=limits)
        result = qpsolvers.solve_problem(qp, solver=options["qp_solver"])
        return result.x if result.found and np.all(np.isfinite(result.x)) else None

    def step(pose, frame, residual, jacobian):
        lower, upper = coords.bounds(pose[None], cfg["angular_step_rad"],
                                     cfg["translation_step_m"])
        low, high = np.zeros(nv), np.zeros(nv)
        low[cols], high[cols] = lower[0], upper[0]
        limits = [_Linear(np.vstack([np.eye(nv), -np.eye(nv)]), np.r_[high, -low])]
        for level in range(LEVELS if guard else 0):
            rows = level_rows(pose, frame, level)
            if rows is None:
                continue
            x = solve(pose, rows, limits)
            if x is None:
                return None
            achieved = rows[1] @ x
            limits.append(_Linear(np.vstack([rows[1], -rows[1]]),
                                  np.r_[achieved + PRIORITY_TOLERANCE,
                                        PRIORITY_TOLERANCE - achieved]))
        return solve(pose, (residual, full(jacobian)), limits)

    solved = reference.copy()
    history, start = [], time.monotonic()
    root = coords.root_q
    for frame in range(frames):
        pose = reference[frame].copy()
        if frame:
            # Warm start: previous solution moved by the source root motion.
            pose = solved[frame - 1].copy()
            if root is not None:
                pose[root : root + 3] += reference[frame, root : root + 3] - reference[
                    frame - 1, root : root + 3
                ]
                turn = Rotation.from_quat(
                    reference[frame, root + 3 : root + 7], scalar_first=True
                ) * Rotation.from_quat(
                    reference[frame - 1, root + 3 : root + 7], scalar_first=True
                ).inv()
                pose[root + 3 : root + 7] = (
                    turn * Rotation.from_quat(pose[root + 3 : root + 7], scalar_first=True)
                ).as_quat(scalar_first=True)
        past = [solved[f] for f in range(frame - 1, max(frame - 4, -1), -1)]
        residual, jacobian, stats = problem.rows(pose, frame, past)
        first_cost = cost = float(residual @ residual)
        iterations = cfg["iterations"] if frame == 0 else options["frame_iterations"]
        done = 0
        for _ in range(iterations):
            before = level_errors(pose, frame)
            x = step(pose, frame, residual, jacobian)
            if x is None:
                failures += 1
                break
            accepted = False
            for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
                candidate = coords.integrate(pose[None], x[None, cols], alpha)[0]
                after = level_errors(candidate, frame)
                new = problem.rows(candidate, frame, past)
                new_cost = float(new[0] @ new[0])
                better = bool(np.sum(after**2) < np.sum(before**2) - 1e-10)
                if np.all(after <= before + PRIORITY_TOLERANCE) and (new_cost < cost or better):
                    pose, (residual, jacobian, stats), cost = candidate, new, new_cost
                    accepted = True
                    break
            done += 1
            if not accepted:
                break
        solved[frame] = pose
        record = dict(frame=frame, iterations=done, cost=first_cost, after=cost,
                      seconds=time.monotonic() - start, **stats)
        history.append(record)
        if callback is not None:
            callback(pose.copy(), dict(record))
    final = {}
    for frame, pose in enumerate(solved):
        for key, value in task_set.frame_rows(pose, frame)[2].items():
            final[key] = max(final.get(key, 0.0), value)
    report = {
        "status": "solved",
        "backend": "mink",
        "framerate": fps,
        "frames": frames,
        "model_nq": model.nq,
        "model_nv": model.nv,
        "scalar_joint_count": len(coords.qi),
        "config": {**cfg, **options},
        "history": history,
        "contact_offset_choice": {k: v.tolist() for k, v in task_set.choices.items()},
        "final_residuals": final,
        "validation_required": "Original moving scene, interframes, source/end-effector errors and temporal metrics",
        "surface_endpoint_guard": {
            "enabled": guard is not None,
            "linear_failures": failures,
            "reference": "role_levels_per_frame",
        },
        "contact_phase_priority": (task_set.contact_phase.diagnostics
                                   if task_set.contact_phase is not None else None),
    }
    return solved, report
