"""Preserve acquired temporal accuracy while optimizing secondary geometry.

Bounds come from the immutable acquired trajectory, not body labels or clip
timings. Root derivatives concern the correction to the unmodified source.
Joint velocity, acceleration and jerk concern each executed scalar joint,
individually. An explicit fast experiment can opt into correction derivatives
relative to its immutable reference; both backends default to legacy semantics. No filtering or
transfer of jitter budgets between joints occurs.
"""

import numpy as np
import mujoco
from collections import OrderedDict
from scipy import sparse
from scipy.spatial.transform import Rotation

from ..conic_constraints import LocalNormBounds
from ..motion_budgets import windowed_budget

LINEARIZATION_SLACK = 1e-7
ACCEPTANCE_SLACK = 2e-7


def _change_size(name, values):
    return np.abs(values) if name == "joints" else np.linalg.norm(values, axis=1)


def _difference_span(frames, count, order):
    """Differences of this order whose stencil reaches the given frame window."""
    if frames is None:
        return 0, count - order
    return max(0, frames.start - order), min(count - order, frames.stop)


class SequenceContinuityGuard:
    def __init__(self, tasks, reference, root_rotations, windowed=False, *, correction=False):
        """``windowed`` uses the relaxed windowed budgets of
        ``motion_budgets.windowed_budget`` instead of stencil-exact ones.

        The optional fast experiment requests ``correction``: keep the same numeric envelopes,
        but constrain deltas from the immutable reference. The sequence
        backend retains its original executed-signal semantics by default.
        """
        self.tasks = tasks
        self.correction = False
        self.rotations = root_rotations
        self._signals_cache = OrderedDict()
        self._joint_matrices = {}
        self.reference = {name: (np.array(value, copy=True), jac)
                          for name, (value, jac) in self.signals(reference).items()}
        for value, _ in self.reference.values():
            value.setflags(write=False)
        self.bounds = {}
        for name, (values, _) in self.reference.items():
            if not values.shape[1]:
                continue
            orders = (1, 2, 3) if name == "joints" else (1, 2)
            for order in orders:
                differences = np.diff(values, n=order, axis=0)
                size = _change_size(name, differences)
                self.bounds[name, order] = windowed_budget(size) if windowed else size

        self.correction = correction
        self.fixed_rotations = (None if tasks.coords.root_q is None else
            Rotation.from_quat(reference[:, tasks.coords.root_q + 3:tasks.coords.root_q + 7],
                               scalar_first=True))
        self._signals_cache.clear()

    def signals(self, poses, span=None):
        """Signals of frames ``span[0]:span[1]`` (all frames by default)."""
        first, stop = (0, len(poses)) if span is None else span
        part = np.asarray(poses)[first:stop]
        key = (first, stop, part.shape, part.dtype.str, part.tobytes())
        if key in self._signals_cache:
            self._signals_cache.move_to_end(key)
            return self._signals_cache[key]
        coords = self.tasks.coords
        count, width = stop - first, coords.width
        joint_jac = np.broadcast_to(
            coords.selector.toarray(), (count, len(coords.qi), width)
        )
        result = {"joints": (poses[first:stop, coords.qi], joint_jac)}
        if self.correction:
            result['joints'] = result['joints'][0] - self.reference['joints'][0][first:stop], joint_jac
        if coords.root_q is None:
            return result
        q, d = coords.root_q, coords.root_d
        linear = np.zeros((3, width))
        for axis in range(3):
            column = coords.inverse.get(d + axis)
            if column is not None:
                linear[axis, column] = 1.0
        result["root_offset"] = (
            poses[first:stop, q : q + 3] - self.tasks.roots[first:stop],
            np.broadcast_to(linear, (count, 3, width)),
        )
        if getattr(self.tasks, "root_body", None) is not None:
            anchors, anchor_jacs = [], []
            for frame in range(first, stop):
                self.tasks.data.qpos[:] = poses[frame]
                mujoco.mj_forward(self.tasks.model, self.tasks.data)
                point, jacobian = self.tasks.root_anchor(frame)
                anchors.append(point)
                anchor_jacs.append(jacobian)
            result["root_anchor_offset"] = (
                np.asarray(anchors) - self.tasks.roots[first:stop],
                np.asarray(anchor_jacs),
            )
        relative = (
            self.rotations[first:stop].inv()
            * Rotation.from_quat(poses[first:stop, q + 3 : q + 7], scalar_first=True)
        ).as_quat(scalar_first=True, canonical=True)
        # Preserve the quaternion chord metric and its original numeric budget.
        # The opt-in fast experiment uses the world-relative quaternion; its log
        # vector coordinate guard independently constrains all three orders.
        if self.correction:
            relative = (Rotation.from_quat(poses[first:stop, q + 3:q + 7], scalar_first=True)
                        * self.fixed_rotations[first:stop].inv()).as_quat(scalar_first=True, canonical=True)
        jac = np.zeros((count, 4, width))
        for frame, (w, x, y, z) in enumerate(relative):
            basis = 0.5 * np.array([[-x, -y, -z], [w, -z, y], [z, w, -x], [-y, x, w]])
            if self.correction:
                basis = basis @ self.fixed_rotations[first + frame].as_matrix()
            for axis in range(3):
                column = coords.inverse.get(d + 3 + axis)
                if column is not None:
                    jac[frame, :, column] = basis[:, axis]
        result["orientation_offset"] = relative, jac
        if self.correction:
            for name in ('root_offset', 'root_anchor_offset'):
                if name in result:
                    value, derivative = result[name]
                    result[name] = value - self.reference[name][0][first:stop], derivative
            result['orientation_offset'] = relative - np.array([1., 0., 0., 0.]), jac
        self._signals_cache[key] = result
        while len(self._signals_cache) > 2:
            self._signals_cache.popitem(last=False)
        return result

    def repair_bounds(self, poses):
        """Whole-trajectory sparse bounds for a fixed-root local correction."""
        count = len(poses)
        signals = self.signals(poses)
        linear, norms = [], []
        for (name, order), bound in self.bounds.items():
            if count <= order:
                continue
            values, jac = signals[name]
            coefficients = {1: [-1, 1], 2: [1, -2, 1], 3: [-1, 3, -3, 1]}[order]
            difference = sparse.diags(coefficients, np.arange(order + 1),
                                      shape=(count - order, count), format="csr", dtype=float)
            operator = sparse.kron(difference, sparse.eye(values.shape[1]), format="csr")
            matrix = operator @ sparse.block_diag(list(jac), format="csr")
            residual = np.diff(values, n=order, axis=0)
            allowance = np.minimum(
                np.maximum(bound + LINEARIZATION_SLACK, _change_size(name, residual)),
                bound + ACCEPTANCE_SLACK,
            )
            if name == "joints":
                linear.append((matrix, -allowance.ravel() - residual.ravel(),
                               allowance.ravel() - residual.ravel()))
            else:
                norms.append((matrix, residual, allowance))
        return linear, norms

    def constraints(self, poses, flat, frames=None, merge_rows=None):
        """Linearized budgets; ``frames`` restricts steps to one frame window.

        Frames outside the window keep a zero step, so only differences whose
        stencil reaches the window can change. ``flat`` then spans the window.
        With ``merge_rows`` (a list), the scalar-joint budgets are appended there as
        ``(matrix, lower, upper)`` instead, for merging across guards.
        """
        window = range(len(poses)) if frames is None else frames
        first = max(0, window.start - 3)
        signals = self.signals(poses, (first, min(len(poses), window.stop + 3)))
        constraints = []
        for (name, order), bounds in self.bounds.items():
            lo, hi = _difference_span(frames, len(poses), order)
            if hi <= lo:
                continue
            values, jac = signals[name]
            rows = LocalNormBounds(len(window), self.tasks.coords.width)
            differences = np.diff(values, n=order, axis=0)[lo - first : hi - first]
            bounds = bounds[lo:hi]
            coefficients = {1: [-1, 1], 2: [1, -2, 1], 3: [-1, 3, -3, 1]}[order]
            if name == "joints":
                # A Euclidean bound lets a noisy ankle borrow improvement from
                # a smoother arm. Each scalar joint gets its own immutable
                # velocity, acceleration AND jerk budget instead. These linear
                # inequalities batch directly, without thousands of tiny cones.
                signature = (len(poses), window.start, window.stop, order)
                matrix = self._joint_matrices.get(signature)
                if matrix is None:
                    difference = sparse.diags(
                        coefficients, np.arange(order + 1),
                        shape=(hi - lo, hi - lo + order), format="csr", dtype=float,
                    )[:, window.start - lo : window.stop - lo]
                    matrix = sparse.kron(difference, self.tasks.coords.selector, format="csr")
                    self._joint_matrices[signature] = matrix
                if merge_rows is not None:
                    bound = bounds.ravel() + LINEARIZATION_SLACK
                    merge_rows.append((matrix, -bound - differences.ravel(), bound - differences.ravel()))
                    continue
                # Measure each row in units of its own budget. This exact
                # positive scaling leaves the feasible set unchanged, but the
                # solver's relative accuracy now scales with every budget.
                # Raw rows let 1e-6 solver error exceed the 2e-7 acceptance
                # slack, forcing tiny line-search fractions or restoration.
                scale = 1.0 / (bounds.ravel() + LINEARIZATION_SLACK)
                residual = (sparse.diags(scale) @ matrix) @ flat
                residual = residual + scale * differences.ravel()
                constraints.extend([residual <= 1.0, residual >= -1.0])
                continue
            for index, (residual, bound) in enumerate(zip(differences, bounds)):
                stencil = [
                    (lo + index + k, c) for k, c in enumerate(coefficients)
                ]
                rows.add(
                    residual,
                    bound + LINEARIZATION_SLACK,
                    *(
                        (f - window.start, c * jac[f - first])
                        for f, c in stencil
                        if f in window
                    ),
                )
            constraints.extend(rows.constraints(flat))
        return constraints

    def window_dependencies(self, frames):
        from .._fast.window_guards import guard_dependencies
        return guard_dependencies(self, frames, len(self.reference['joints'][0]))

    def qp_bounds(self, poses, *, frames=None):
        from .._fast.window_guards import guard_qp_bounds
        return guard_qp_bounds(self, poses, frames=range(len(poses)) if frames is None else frames)

    def qp_values(self, poses, *, frames=None):
        from .._fast.window_guards import guard_qp_values
        linear, cones = guard_qp_values(self, poses, frames=range(len(poses)) if frames is None else frames)
        return ([(lo, hi) for _, lo, hi in linear],
                [(residual, upper) for _, residual, upper in cones])

    def accepts(self, poses, frames=None):
        if frames is not None:
            from .._fast.window_guards import guard_accepts
            good = guard_accepts(self, poses, frames=frames)
            return good
        """Nonlinear check; ``frames`` limits it to differences a window moved."""
        window = range(len(poses)) if frames is None else frames
        first = max(0, window.start - 3)
        signals = self.signals(poses, (first, min(len(poses), window.stop + 3)))
        for (name, order), bound in self.bounds.items():
            lo, hi = _difference_span(frames, len(poses), order)
            if hi <= lo:
                continue
            differences = np.diff(signals[name][0], n=order, axis=0)
            size = _change_size(name, differences[lo - first : hi - first])
            if not np.all(size <= bound[lo:hi] + ACCEPTANCE_SLACK):
                return False
        return True

    def violation_frames(self, poses, *, frames=None):
        if frames is not None:
            from .._fast.window_guards import guard_violation_frames
            return guard_violation_frames(self, poses, frames=frames)
        """Frames of every difference stencil ``accepts`` rejects."""
        signals = self.signals(poses)
        bad = np.zeros(len(poses), bool)
        for (name, order), bound in self.bounds.items():
            if len(poses) <= order:
                continue
            size = _change_size(name, np.diff(signals[name][0], n=order, axis=0))
            rows = ~(size <= bound + ACCEPTANCE_SLACK)
            rows = rows.any(axis=1) if rows.ndim > 1 else rows
            for offset in range(order + 1):
                bad[offset:offset + len(rows)] |= rows
        return bad
