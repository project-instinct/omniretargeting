"""Independent temporal certificates; no reference rebasing or RMS masking."""

import numpy as np
from scipy import sparse
from scipy.ndimage import maximum_filter1d
from scipy.spatial.transform import Rotation

from .._refinement.math import _difference_operator
from .kinematics import skew
from .runtime_options import OPTIONS

from ..motion_budgets import CHATTER_RMS_FRACTION, CHATTER_WINDOW, windowed_budget

ROUND_OFF = 2e-7


def derivatives(coords, poses):
    """Per-frame differences, including world angular increments of the root.

    Quaternion signs do not affect the result. All angular differences use
    radians; translations use metres. Values are per frame, not per second.
    """
    signals = {"joints": (np.asarray(poses)[:, coords.qi], 0)}
    q = coords.root_q
    if q is not None:
        rotation = Rotation.from_quat(poses[:, q + 3:q + 7], scalar_first=True)
        angular = (rotation[1:] * rotation[:-1].inv()).as_rotvec()
        signals["root_translation"] = (poses[:, q:q + 3], 0)
        signals["root_rotation"] = (angular, 1)
    return {
        f"{name}_order_{order}": np.diff(values, n=order - initial, axis=0)
        for name, (values, initial) in signals.items() for order in (1, 2, 3)
        if len(poses) > order
    }


def correction_signals(coords, fixed, motion):
    """Joint/translation deltas and world rotation log relative to fixed frames.

    R_motion R_fixed^-1 gives zero correction for authored root motion, even
    when successive reference rotations do not commute. Quaternion signs have
    no effect. The reference never follows an accepted candidate.
    """
    signals = {"joints": motion[:, coords.qi] - fixed[:, coords.qi]}
    q = coords.root_q
    if q is not None:
        signals["root_translation"] = motion[:, q:q + 3] - fixed[:, q:q + 3]
        current = Rotation.from_quat(motion[:, q + 3:q + 7], scalar_first=True)
        reference = Rotation.from_quat(fixed[:, q + 3:q + 7], scalar_first=True)
        signals["root_rotation"] = (current * reference.inv()).as_rotvec()
    return signals


def correction_derivatives(coords, fixed, motion):
    return {f"{name}_order_{order}": np.diff(value, n=order, axis=0)
            for name, value in correction_signals(coords, fixed, motion).items()
            for order in (1, 2, 3) if len(motion) > order}


def audit_continuity(coords, fixed, motion):
    """Report strict and relaxed checks separately, per coordinate and stencil."""
    before, after = derivatives(coords, fixed), derivatives(coords, motion)
    correction = correction_derivatives(coords, fixed, motion)
    budget_values = correction if OPTIONS.correction_budget else after
    strict_frames = np.zeros(len(motion), bool)
    chatter_frames = np.zeros(len(motion), bool)
    budget_frames = np.zeros(len(motion), bool)
    channels = {}
    for name, original in before.items():
        current = after[name]
        order = int(name[-1])
        old, new = np.abs(original), np.abs(current)
        strict = new > old + ROUND_OFF
        local = maximum_filter1d(old, CHATTER_WINDOW, axis=0, mode="nearest")
        rms = np.sqrt(np.mean(old**2, axis=0))
        chatter = np.abs(budget_values[name]) > local + CHATTER_RMS_FRACTION * rms + ROUND_OFF
        # The enforced per-joint budget: every order, windowed (motion_budgets).
        over = np.abs(budget_values[name]) > windowed_budget(old) + ROUND_OFF
        for start in np.flatnonzero(over.any(axis=1)):
            budget_frames[start:start + order + 1] = True
        for mask, frames in ((strict, strict_frames), (chatter if order > 1 else np.zeros_like(chatter), chatter_frames)):
            for start in np.flatnonzero(mask.any(axis=1)):
                frames[start:start + order + 1] = True
        channels[name] = {
            "correction_rms_per_coordinate": np.sqrt(np.mean(correction[name]**2, axis=0)).tolist(),
            "correction_peak_per_coordinate": np.abs(correction[name]).max(axis=0).tolist(),
            "strict_violations": int(strict.sum()),
            "strict_per_coordinate": strict.sum(axis=0).tolist(),
            "violating_stencil_starts": np.flatnonzero(strict.any(axis=1)).tolist(),
            "added_chatter_violations": int(chatter.sum()),
            "windowed_budget_violations": int(over.sum()),
            "rms_before_per_coordinate": rms.tolist(),
            "rms_after_per_coordinate": np.sqrt(np.mean(new**2, axis=0)).tolist(),
            "peak_before_per_coordinate": old.max(axis=0).tolist(),
            "peak_after_per_coordinate": new.max(axis=0).tolist(),
        }
    return {
        "units": "metres or radians per frame difference; orders 1, 2, 3",
        "budget_signal": ("correction relative to immutable reference" if OPTIONS.correction_budget
                          else "absolute executed derivatives"),
        "round_off_slack": ROUND_OFF,
        "channels": channels,
        "strict_accepted": not bool(strict_frames.any()),
        "relaxed_chatter_accepted": not bool(chatter_frames.any()),
        "windowed_budget_accepted": not bool(budget_frames.any()),
        "windowed_budget_violation_frames": np.flatnonzero(budget_frames).tolist(),
        "strict_violation_frames": np.flatnonzero(strict_frames).tolist(),
        "added_chatter_frames": np.flatnonzero(chatter_frames).tolist(),
    }


class CorrectionContinuity:
    """Preserve absolute derivatives; correction semantics require an explicit opt-in."""

    def __init__(self, coords, acquired, *, root_norm=False, correction=None):
        self.coords = coords
        self.correction = OPTIONS.correction_budget if correction is None else correction
        self.fixed = np.array(acquired, copy=True)
        self.fixed.setflags(write=False)
        # Windowed per-coordinate budgets of the reference (motion_budgets).
        self.bounds = {k: windowed_budget(v) for k, v in derivatives(coords, acquired).items()}
        self.rejections = 0
        self._operators = {}
        self._linear_cache = None
        # The independent final audit measures root acceleration and
        # jerk as 3D norms. Coordinate bounds alone do not imply these balls.
        self.norm_bounds = {}
        if root_norm and coords.root_q is not None:
            for name, value in derivatives(coords, acquired).items():
                if name in ('root_translation_order_2', 'root_translation_order_3'):
                    self.norm_bounds[name] = windowed_budget(np.linalg.norm(value, axis=1))

    def accepts(self, poses):
        values = (correction_derivatives(self.coords, self.fixed, poses) if self.correction
                  else derivatives(self.coords, poses))
        good = all(np.isfinite(values[k]).all() and np.all(np.abs(values[k]) <= v + ROUND_OFF)
                   for k, v in self.bounds.items())
        good = good and all(np.all(np.linalg.norm(values[k], axis=1) <= v + ROUND_OFF)
                            for k, v in self.norm_bounds.items())
        self.rejections += not good
        return bool(good)

    def linearizations(self, poses):
        """Derivatives of the selected signal in model tangent coordinates."""
        key = (poses.shape, poses.dtype.str, poses.tobytes())
        if self._linear_cache is not None and self._linear_cache[0] == key:
            return self._linear_cache[1]
        coords, count = self.coords, len(poses)
        correction = (correction_signals(coords, self.fixed, poses) if self.correction
                      else {'joints': poses[:, coords.qi]})
        signals = {'joints': (correction['joints'],
                   sparse.kron(sparse.eye(count), coords.selector, format='csr'), 0)}
        if coords.root_q is not None:
            q, d, width = coords.root_q, coords.root_d, coords.width
            select = np.zeros((3, width))
            for axis in range(3):
                col = coords.inverse.get(d + axis)
                if col is not None:
                    select[axis, col] = 1.
            signals['root_translation'] = (correction['root_translation'] if self.correction else poses[:, q:q + 3],
                sparse.kron(sparse.eye(count), select, format='csr'), 0)
            rotation = Rotation.from_quat(poses[:, q + 3:q + 7], scalar_first=True)
            angular = (correction['root_rotation'] if self.correction
                       else (rotation[1:] * rotation[:-1].inv()).as_rotvec())
            angle = np.linalg.norm(angular, axis=1)
            coefficient = np.full_like(angle, 1 / 12)
            away = angle > 1e-4
            theta = angle[away]
            coefficient[away] = 1 / theta**2 - (1 + np.cos(theta)) / (2 * theta * np.sin(theta))
            cross = skew(angular)
            inverse = np.eye(3) - .5 * cross + coefficient[:, None, None] * (cross @ cross)
            if self.correction:
                derivative = inverse @ rotation.as_matrix()
                local = np.zeros((count, 3, width))
                for axis in range(3):
                    col = coords.inverse.get(d + 3 + axis)
                    if col is not None:
                        local[:, :, col] = derivative[:, :, axis]
                jac = sparse.block_diag(list(local), format='csr')
                signals['root_rotation'] = angular, jac, 0
            else:
                derivative = inverse @ rotation.as_matrix()[1:]
                row, column, value = [], [], []
                for axis in range(3):
                    col = coords.inverse.get(d + 3 + axis)
                    if col is None:
                        continue
                    for side, sign in ((0, -1.), (1, 1.)):
                        row.extend((np.arange(count - 1)[:, None] * 3 + np.arange(3)).ravel())
                        column.extend(np.repeat((np.arange(count - 1) + side) * width + col, 3))
                        value.extend((sign * derivative[:, :, axis]).ravel())
                jac = sparse.csr_matrix((value, (row, column)), shape=(3 * (count - 1), count * width))
                signals['root_rotation'] = angular, jac, 1
        rows = {}
        for name, (values, jac, initial) in signals.items():
            if not values.shape[1]:
                continue
            for order in (1, 2, 3):
                if count <= order:
                    continue
                signature = (name, count, values.shape[1], order)
                operator = self._operators.get(signature)
                if operator is None:
                    operator = sparse.kron(_difference_operator(len(values), order - initial),
                                           sparse.eye(values.shape[1]), format='csr')
                    self._operators[signature] = operator
                rows[f'{name}_order_{order}'] = (
                    np.diff(values, n=order - initial, axis=0), operator @ jac)
        self._linear_cache = key, rows
        return rows

    def repair_bounds(self, poses):
        linear, norms = [], []
        rows = self.linearizations(poses)
        for name, (values, jacobian) in rows.items():
            fixed = self.bounds[name].ravel()
            bound = np.minimum(np.maximum(fixed + ROUND_OFF / 2, np.abs(values.ravel())),
                               fixed + ROUND_OFF)
            linear.append((jacobian, -bound - values.ravel(), bound - values.ravel()))
        for name, fixed in self.norm_bounds.items():
            values, jacobian = rows[name]
            bound = np.minimum(np.maximum(fixed + ROUND_OFF / 2, np.linalg.norm(values, axis=1)),
                               fixed + ROUND_OFF)
            norms.append((jacobian, values, bound))
        return linear, norms

    def constraints(self, poses, flat, merge_rows=None):
        """Linearized budgets. With ``merge_rows`` (a list), they are appended there
        as ``(matrix, lower, upper)`` instead, for merging across guards."""
        import cvxpy as cp

        constraints = []
        rows = self.linearizations(poses)
        for name, (values, jacobian) in rows.items():
            if merge_rows is not None:
                bound = self.bounds[name].ravel() + ROUND_OFF / 2
                merge_rows.append((jacobian, -bound - values.ravel(), bound - values.ravel()))
                continue
            # Scale each coordinate/stencil by its own immutable budget.
            scale = 1 / (self.bounds[name].ravel() + ROUND_OFF / 2)
            residual = (sparse.diags(scale) @ jacobian) @ flat + scale * values.ravel()
            constraints += [residual <= 1., residual >= -1.]
        for name, fixed in self.norm_bounds.items():
            values, jacobian = rows[name]
            scale = 1 / (fixed + ROUND_OFF / 2)
            residual = (sparse.diags(np.repeat(scale, 3)) @ jacobian) @ flat
            residual = residual + (values * scale[:, None]).ravel()
            constraints.append(cp.SOC(np.ones(len(fixed)),
                cp.reshape(residual, (3, len(fixed)), order='F'), axis=0))
        return constraints
