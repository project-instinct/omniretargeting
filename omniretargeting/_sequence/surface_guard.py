"""Fixed endpoint/contact budgets when adding dense surface pose objectives."""

from types import SimpleNamespace

import cvxpy as cp
import mujoco
import numpy as np
from scipy import sparse

from .._solver.targets import _approach_contact_target
from ..conic_constraints import LocalNormBounds, squared_residual_objective
from ..conic_solver import solve_conic
from ..contact import _is_surface_only_contact_mode
from ..contact_phase import CONTACT_ROLES

# Same role means the same level for endpoints and arbitrary body patches.
LEVELS = len(CONTACT_ROLES)


class SurfaceSequenceGuard:
    def __init__(self, tasks, reference, allowance, *, preserve_contacts=True):
        self.tasks, self.allowance = tasks, allowance
        self.preserve_contacts = preserve_contacts
        self.bounds = []
        self.failures = 0
        self.continuity = None
        self.pose = None
        self.contact_stage = None
        self.step_priority_bounds = {}
        self.step_priority_before = {}
        for frame, pose in enumerate(reference if preserve_contacts else ()):
            self.bounds.append(
                {
                    key: max(float(np.linalg.norm(residual)), tolerance) + regression
                    for key, residual, _, tolerance, regression in self.rows(
                        pose, frame
                    )
                }
            )

    def rows(self, pose, frame, *, full_surface_vectors=False):
        tasks = self.tasks
        tasks.data.qpos[:] = pose
        mujoco.mj_forward(tasks.model, tasks.data)
        for name, spec in tasks.effectors.items():
            target = spec["task"].frame(frame)
            active = target["contact"]
            if active and target.get("surface_managed", False):
                continue
            from .._solver.targets import _world_target_regime

            if not active and not _world_target_regime(spec, target):
                continue
            offset = (
                spec["contact_offsets"][tasks.choices[name][frame]]
                if active
                else spec["tracking_offset"]
            )
            if not active:
                share = target["approach_weight"]
                offset = (1 - share) * offset + share * spec["contact_offsets"][
                    tasks.choices[name][frame]
                ]
            point, jac = tasks.point(spec["body_id"], offset)
            residual = point - (
                target["contact_position"]
                if active
                else _approach_contact_target(target["tracking_position"], target)[0]
            )
            key = ("endpoint", name)
            policy = tasks.contact_phase
            # Observed clear of every surface: yields to contact levels.
            if policy is not None and not active and policy.yields(name, frame):
                continue
            if active and not full_surface_vectors and _is_surface_only_contact_mode(target["contact_mode"]):
                normal = np.asarray(target["surface_normal"], float)
                normal = normal / np.linalg.norm(normal)
                tangent = np.eye(3) - np.outer(normal, normal)
                yield (
                    (*key, "normal"),
                    np.atleast_1d(normal @ residual),
                    (normal @ jac)[None],
                    0,
                    self.allowance,
                )
                yield (
                    (*key, "tangent"),
                    tangent @ residual,
                    tangent @ jac,
                    0,
                    self.allowance,
                )
            else:
                yield key, residual, jac, 0, self.allowance
        for index, task in enumerate(tasks.extra_tasks):
            if not hasattr(task, "position_rows") or "tolerance" not in getattr(
                task, "cfg", {}
            ):
                continue
            position_rows = (
                getattr(task, "hold_position_rows", task.position_rows)
                if full_surface_vectors else task.position_rows
            )
            for k, residual, jac in position_rows(pose, frame, tasks.coords.cols):
                yield (
                    ("surface_contact", index, k),
                    residual,
                    jac,
                    task.cfg["tolerance"],
                    task.cfg["regression"],
                )

    def accepts(self, poses, frames=None):
        """Nonlinear budgets; ``frames`` checks only what a window step moved."""
        if self.contact_stage is not None and not self.contact_stage.accepts(poses, frames):
            return False
        if self.continuity is not None and not _windowed(
            self.continuity.accepts, poses, frames=frames
        ):
            return False
        if self.pose is not None and not _windowed(
            self.pose.accepts, poses, frames=frames
        ):
            return False
        return not self.preserve_contacts or all(
            np.linalg.norm(residual) <= self.bounds[f][key] + 1e-7
            for f in (range(len(poses)) if frames is None else frames)
            for key, residual, _, _, _ in self.rows(poses[f], f)
        )

    def priority_rows(self, pose, frame, level):
        """Evidence-ordered contacts precede bounded clearance correction.

        Contact anchors already lie on observed scene surfaces. Clearance must
        not detach a support or buy a lower collision cost with a different pose.
        Incompatible geometry leaves a reported residual.
        """
        policy = self.tasks.contact_phase
        # Scene clearance stays in the secondary objective. Treating it as a
        # per-frame contact role forbids a continuous support correction when
        # it slightly increases clearance error in a neighbouring free frame.
        # The fixed posture, continuity and attained-contact bounds still apply.
        role = CONTACT_ROLES[level]
        yield from policy.rows(pose, frame, self.tasks.coords.cols, role=role)
        for key, residual, jac, _, _ in self.rows(pose, frame):
            if (
                key[0] == "endpoint"
                and key[-1] != "tangent"
                and policy.role(key[1], frame) == role
            ):
                yield 1.0, residual, jac

    def priority_errors(self, poses, frames=None):
        """Rows are frames (of ``frames`` when given), columns priority levels."""
        return np.array(
            [
                [
                    np.sqrt(
                        sum(
                            w * float(r @ r)
                            for w, r, _ in self.priority_rows(poses[f], f, level)
                        )
                    )
                    for level in range(LEVELS)
                ]
                for f in (range(len(poses)) if frames is None else frames)
            ]
        )

    def constraints(self, poses, flat, lower, upper, frames=None):
        """Linearize the same immutable hard bounds for fitting and restoration.

        With ``frames`` the step spans that window only; frames outside keep a
        zero step, so every bound they alone determine is already satisfied.
        """
        count, width = lower.shape
        window = range(len(poses)) if frames is None else frames
        if self.pose is not None:
            _windowed(self.pose.constrain_step, poses, lower, upper, frames=frames)
        if self.contact_stage is not None:
            self.contact_stage.constrain_step(poses, lower, upper, frames)
        constraints = [flat >= lower.ravel(), flat <= upper.ravel()]
        if self.contact_stage is not None:
            constraints.extend(self.contact_stage.constraints(poses, flat, frames))
        if self.continuity is not None:
            constraints.extend(
                _windowed(self.continuity.constraints, poses, flat, frames=frames)
            )
        if self.pose is not None:
            constraints.extend(
                _windowed(self.pose.constraints, poses, flat, frames=frames)
            )
        # One sparse affine map and one vectorized cone for all per-frame
        # bounds: thousands of tiny CVXPY expressions dominate compile time.
        blocks, bounds = [], []
        for f in window if self.preserve_contacts else ():
            for key, residual, jac, _, _ in self.rows(poses[f], f):
                blocks.append((f - window.start, residual, jac))
                bounds.append(self.bounds[f][key])
        if blocks:
            linear, offset, _ = _stack(blocks, count, width, height=3)
            # Measure each cone in units of its own error budget. This is an
            # exact positive scaling, so millimetre contacts and large legacy
            # residuals receive the same numerical feasibility accuracy.
            scale = 1 / np.maximum(np.asarray(bounds), np.finfo(float).eps)
            linear = sparse.diags(np.repeat(scale, 3)) @ linear
            offset = offset * np.repeat(scale, 3)
            constraints.append(
                cp.SOC(
                    np.asarray(bounds) * scale,
                    cp.reshape(linear @ flat + offset, (3, len(blocks)), order="F"),
                    axis=0,
                )
            )
        return constraints

    def restore(self, origin, candidate, lower, upper, fraction=1.0, frames=None):
        """Retract a tangent step onto the unchanged nonlinear feasible set.

        Linearized rotations/bone directions can lie just outside a curved
        boundary even when the conic step is feasible (the Maratos effect).
        A minimum-change correction avoids endlessly shrinking that useful
        tangent step. Every result is still checked against the original bounds.
        """
        from .restoration import restore_candidate

        bounds = {
            key: (1 - fraction) * self.step_priority_before[key] + fraction * bound
            for key, bound in self.step_priority_bounds.items()
        }
        return restore_candidate(
            self, origin, candidate, lower, upper, bounds, frames=frames
        )

    def restoration_constraints(self, poses, flat, lower, upper, bounds, frames=None):
        constraints = self.constraints(poses, flat, lower, upper, frames)
        start = 0 if frames is None else frames.start
        rows = LocalNormBounds(lower.shape[0], self.tasks.coords.width)
        for (level, frame), bound in bounds.items():
            blocks = [
                (np.sqrt(w) * r, np.sqrt(w) * j)
                for w, r, j in self.priority_rows(poses[frame], frame, level)
            ]
            if blocks:
                residual = np.concatenate([np.atleast_1d(r) for r, _ in blocks])
                jac = np.vstack([np.atleast_2d(j) for _, j in blocks])
                rows.add(residual, bound, (frame - start, jac))
        return constraints + rows.constraints(flat)

    def solve(self, matrix, rhs, lower, upper, poses, frames=None):
        """Role-ordered conic step; ``frames`` restricts it to one window."""
        count, width = lower.shape
        window = range(len(poses)) if frames is None else frames
        self.step_priority_bounds = {}
        self.step_priority_before = {}
        flat = cp.Variable(count * width)
        constraints = self.constraints(poses, flat, lower, upper, frames)
        hessian = (matrix.T @ matrix).tocsc()
        hessian += 1e-8 * sparse.eye(hessian.shape[0], format="csc")
        linear_target = np.asarray(matrix.T @ rhs).ravel()
        # A positive scale leaves the minimizer unchanged. At the raw O(1e7)
        # magnitude Clarabel stalls at its iteration cap and returns a point
        # that violates the millimetre bounds; at O(1) it converges.
        # Keep the banded quadratic explicitly. Expressing its Cholesky factor
        # as sum_squares creates one auxiliary residual variable per coordinate
        # in CVXPY's conic reduction and makes long contact trajectories costly.
        # J.T @ J is PSD by construction; no eigenvalue estimate is necessary.
        objective = (
            cp.quad_form(flat, cp.psd_wrap(hessian)) - 2 * linear_target @ flat
        ) / max(float(rhs @ rhs), 1e-12)
        problem = cp.Problem(cp.Minimize(objective), constraints)
        try:
            policy = self.tasks.contact_phase
            if policy is not None:
                improved = np.zeros(count, dtype=bool)
                for level in range(LEVELS):
                    rows = [
                        (f - window.start, np.sqrt(w) * r, np.sqrt(w) * j)
                        for f in window
                        for w, r, j in self.priority_rows(poses[f], f, level)
                    ]
                    if not rows:
                        continue
                    linear, offset, spans = _stack(rows, count, width)
                    padded, padded_offset, height = _pad_frame_rows(
                        linear, offset, spans
                    )
                    expression = cp.reshape(
                        padded @ flat + padded_offset,
                        (height, len(spans)),
                        order="F",
                    )
                    row_frames = sorted({row[0] for row in rows})
                    before = np.array(
                        [np.linalg.norm(offset[a:b]) for a, b in spans]
                    )
                    # The nonlinear acceptance contract is per frame. An
                    # aggregate primary solve must not buy progress in one
                    # frame by worsening another, only to reject that same
                    # tradeoff at every line-search fraction afterwards.
                    protected = ~improved[row_frames]
                    if np.any(protected):
                        constraints.append(
                            cp.SOC(
                                before[protected] + policy.tolerance,
                                expression[:, protected],
                                axis=0,
                            )
                        )
                    primary = cp.Problem(
                        cp.Minimize(
                            (
                                squared_residual_objective(linear, -offset, flat)
                                + 1e-8 * cp.sum_squares(flat)
                            )
                            / max(float(offset @ offset), 1.0)
                        ),
                        constraints,
                    )
                    solve_conic(primary, solver=cp.CLARABEL, direct_solve_method="qdldl")
                    if (
                        primary.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
                        or flat.value is None
                    ):
                        self.failures += 1
                        return SimpleNamespace(x=np.zeros(count * width), status=0)
                    value = linear @ flat.value + offset
                    achieved = np.array(
                        [np.linalg.norm(value[a:b]) for a, b in spans]
                    )
                    constraints.append(
                        cp.SOC(
                            achieved + policy.tolerance, expression, axis=0
                        )
                    )
                    # Later levels may yield only if the retained upper bound
                    # guarantees progress at an earlier level in this frame.
                    improved[row_frames] |= (
                        achieved + policy.tolerance < before - policy.tolerance
                    )
                    for frame, old, error in zip(row_frames, before, achieved):
                        key = level, frame + window.start
                        self.step_priority_before[key] = old
                        self.step_priority_bounds[key] = min(
                            error + policy.tolerance,
                            old + policy.tolerance if not improved[frame] else np.inf,
                        )
                problem = cp.Problem(cp.Minimize(objective), constraints)
            solve_conic(problem, solver=cp.CLARABEL, direct_solve_method="qdldl")
        except cp.error.SolverError:
            pass
        usable = (
            problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
            and flat.value is not None
        )
        if not usable:
            self.failures += 1
        return SimpleNamespace(
            x=flat.value if usable else np.zeros(count * width),
            status=1 if usable else 0,
            backend=(
                f"priority_guard:{problem.solver_stats.solver_name}"
                if usable and problem.solver_stats is not None
                else "priority_guard"
            ),
        )


def _windowed(method, *args, frames=None):
    """Call a guard method; pass a frame window only when one is requested."""
    return method(*args) if frames is None else method(*args, frames)


def _stack(blocks, frames, width, height=None):
    """Per-frame (frame, residual, jacobian) rows as ``A @ step + r``.

    ``height`` zero-pads every row to a fixed cone dimension. Also returns the
    contiguous row span of each frame, in frame order.
    """
    data, row_ids, col_ids, offsets, spans = [], [], [], [], {}
    start = 0
    for frame, residual, jac in blocks:
        residual = np.atleast_1d(np.asarray(residual, dtype=float))
        jac = np.atleast_2d(np.asarray(jac, dtype=float))
        size = height or len(residual)
        rows = start + np.arange(len(residual))
        row_ids.append(np.repeat(rows, width))
        col_ids.append(np.tile(frame * width + np.arange(width), len(residual)))
        data.append(jac.ravel())
        offsets.append(np.r_[residual, np.zeros(size - len(residual))])
        first, _ = spans.get(frame, (start, start))
        spans[frame] = (first, start + size)
        start += size
    linear = sparse.csr_matrix(
        (np.concatenate(data), (np.concatenate(row_ids), np.concatenate(col_ids))),
        shape=(start, frames * width),
    )
    linear.eliminate_zeros()
    return linear, np.concatenate(offsets), [spans[f] for f in sorted(spans)]


def _pad_frame_rows(linear, offset, spans):
    """Stack varying frame residual lengths into one batch of exact norm cones."""
    height = max(end - begin for begin, end in spans)
    row_map = np.empty(linear.shape[0], dtype=int)
    padded_offset = np.zeros((height, len(spans)), dtype=float)
    for index, (begin, end) in enumerate(spans):
        count = end - begin
        row_map[begin:end] = index * height + np.arange(count)
        padded_offset[:count, index] = offset[begin:end]
    coo = linear.tocoo()
    padded = sparse.coo_matrix(
        (coo.data, (row_map[coo.row], coo.col)),
        shape=(height * len(spans), linear.shape[1]),
    ).tocsr()
    return padded, padded_offset.ravel(order="F"), height
