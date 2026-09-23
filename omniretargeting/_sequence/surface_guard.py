"""Fixed endpoint/contact budgets when adding dense surface pose objectives."""

from types import SimpleNamespace

import cvxpy as cp
import mujoco
import numpy as np
from scipy import linalg, sparse

from .._solver.targets import _approach_contact_target
from ..contact import _is_surface_only_contact_mode

# Non-penetration, endpoint support, surface support, following contacts.
LEVELS = 4


class SurfaceSequenceGuard:
    def __init__(self, tasks, reference, allowance):
        self.tasks, self.allowance = tasks, allowance
        self.bounds = []
        self.failures = 0
        for frame, pose in enumerate(reference):
            self.bounds.append(
                {
                    key: max(float(np.linalg.norm(residual)), tolerance) + regression
                    for key, residual, _, tolerance, regression in self.rows(
                        pose, frame
                    )
                }
            )

    def rows(self, pose, frame):
        tasks = self.tasks
        tasks.data.qpos[:] = pose
        mujoco.mj_forward(tasks.model, tasks.data)
        for name, spec in tasks.effectors.items():
            target = spec["task"].frame(frame)
            active = target["contact"]
            if not active and not spec.get("preserve_world_target", False):
                continue
            offset = (
                spec["contact_offsets"][tasks.choices[name][frame]]
                if active
                else spec["tracking_offset"]
            )
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
            if active and _is_surface_only_contact_mode(target["contact_mode"]):
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
            for k, residual, jac in task.position_rows(pose, frame, tasks.coords.cols):
                yield (
                    ("surface_contact", index, k),
                    residual,
                    jac,
                    task.cfg["tolerance"],
                    task.cfg["regression"],
                )

    def accepts(self, poses):
        return all(
            np.linalg.norm(residual) <= self.bounds[f][key] + 1e-7
            for f, pose in enumerate(poses)
            for key, residual, _, _, _ in self.rows(pose, f)
        )

    def priority_rows(self, pose, frame, level):
        """Clearance, endpoint support, surface support, following contacts.

        The source may itself be inside a surface (e.g. SMPL-X feet through
        the floor). Non-penetration therefore comes first; a contact target
        inside the scene then leaves a reported residual, not a penetration.
        """
        policy = self.tasks.contact_phase
        if level == 0:
            tasks = self.tasks
            tasks.data.qpos[:] = pose
            mujoco.mj_forward(tasks.model, tasks.data)
            for residual, jac in tasks.clearance_rows():
                yield 1.0, residual, jac
            return
        if level == 2:
            yield from policy.rows(pose, frame, self.tasks.coords.cols)
            return
        if level == 3:
            yield from policy.rows(
                pose, frame, self.tasks.coords.cols, role="follow"
            )
        role = "support" if level == 1 else "follow"
        for key, residual, jac, _, _ in self.rows(pose, frame):
            if (
                key[0] == "endpoint"
                and key[-1] != "tangent"
                and policy.role(key[1], frame) == role
            ):
                yield 1.0, residual, jac

    def priority_errors(self, poses):
        return np.array(
            [
                [
                    np.sqrt(
                        sum(
                            w * float(r @ r)
                            for w, r, _ in self.priority_rows(q, f, level)
                        )
                    )
                    for level in range(LEVELS)
                ]
                for f, q in enumerate(poses)
            ]
        )

    def solve(self, matrix, rhs, lower, upper, poses):
        frames, width = lower.shape
        flat = cp.Variable(frames * width)
        constraints = [flat >= lower.ravel(), flat <= upper.ravel()]
        # One sparse affine map and one vectorized cone for all per-frame
        # bounds: thousands of tiny CVXPY expressions dominate compile time.
        blocks, bounds = [], []
        for f, pose in enumerate(poses):
            for key, residual, jac, _, _ in self.rows(pose, f):
                blocks.append((f, residual, jac))
                bounds.append(self.bounds[f][key])
        if blocks:
            linear, offset, _ = _stack(blocks, frames, width, height=3)
            constraints.append(
                cp.SOC(
                    np.asarray(bounds),
                    cp.reshape(linear @ flat + offset, (3, len(blocks)), order="F"),
                    axis=0,
                )
            )
        factor, target = _least_squares_factor(matrix, rhs)
        # A positive scale leaves the minimizer unchanged. At the raw O(1e7)
        # magnitude Clarabel stalls at its iteration cap and returns a point
        # that violates the millimetre bounds; at O(1) it converges.
        objective = cp.sum_squares(factor @ flat - target) / max(
            float(target @ target), 1e-12
        )
        problem = cp.Problem(cp.Minimize(objective), constraints)
        try:
            policy = self.tasks.contact_phase
            if policy is not None:
                for level in range(LEVELS):
                    rows = [
                        (f, np.sqrt(w) * r, np.sqrt(w) * j)
                        for f, pose in enumerate(poses)
                        for w, r, j in self.priority_rows(pose, f, level)
                    ]
                    if not rows:
                        continue
                    linear, offset, spans = _stack(rows, frames, width)
                    expression = linear @ flat + offset
                    primary = cp.Problem(
                        cp.Minimize(
                            cp.sum_squares(expression) + 1e-8 * cp.sum_squares(flat)
                        ),
                        constraints,
                    )
                    primary.solve(solver=cp.CLARABEL)
                    if (
                        primary.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
                        or flat.value is None
                    ):
                        self.failures += 1
                        return SimpleNamespace(x=np.zeros(frames * width), status=0)
                    value = linear @ flat.value + offset
                    constraints += [
                        cp.norm(expression[a:b])
                        <= float(np.linalg.norm(value[a:b])) + policy.tolerance
                        for a, b in spans
                    ]
                problem = cp.Problem(cp.Minimize(objective), constraints)
            problem.solve(solver=cp.CLARABEL)
        except cp.error.SolverError:
            pass
        usable = (
            problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
            and flat.value is not None
        )
        if not usable:
            self.failures += 1
        return SimpleNamespace(
            x=flat.value if usable else np.zeros(frames * width),
            status=1 if usable else 0,
        )


def _least_squares_factor(matrix, rhs):
    """``R``, ``c`` with ``|R x - c|^2 = |A x - b|^2 + 1e-8 |x|^2 + const``.

    Frames couple only through the temporal differences, so ``A^T A`` is
    banded and its Cholesky factor has one row per coordinate instead of one
    per residual (every dense surface sample in every frame). The minimizer
    and conditioning are unchanged; the ridge equals the priority levels'
    existing tie-break.
    """
    normal = sparse.triu(matrix.T @ matrix, format="coo")
    size = normal.shape[0]
    band = int(np.max(normal.col - normal.row, initial=0))
    banded = np.zeros((band + 1, size))
    np.add.at(banded, (band + normal.row - normal.col, normal.col), normal.data)
    banded[band] += 1e-8
    upper = linalg.cholesky_banded(banded)
    factor = sparse.diags(
        [upper[band - d, d:] for d in range(band + 1)], range(band + 1), format="csr"
    )
    # R^-T A^T b = R (R^T R)^-1 A^T b
    target = factor @ linalg.cho_solve_banded((upper, False), matrix.T @ rhs)
    return factor, target


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
    return linear, np.concatenate(offsets), [spans[f] for f in sorted(spans)]
