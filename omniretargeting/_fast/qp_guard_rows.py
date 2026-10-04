"""Direct copies of acquisition guard algebra, with frozen row identities for SOC.

Repair bounds deliberately are not used for continuity: they have different
roundoff reserves. No CVXPY objects are constructed here.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from ..pose_relations import _row_groups
from .rows import _block_rows


class UnsupportedRows(TypeError):
    pass


@dataclass
class HardBlock:
    A: sparse.csr_matrix
    b: np.ndarray
    soc: tuple = ()
    evaluate: object = None


def intervals(matrix, lower, upper, evaluate=None):
    matrix = sparse.csr_matrix(matrix)
    lower, upper = np.asarray(lower), np.asarray(upper)
    lo, hi = np.isfinite(lower), np.isfinite(upper)
    A = sparse.vstack((matrix[hi], -matrix[lo]), format='csr')
    b = np.r_[upper[hi], -lower[lo]]

    def values(poses):
        result = evaluate(poses)
        if result is None:
            return None
        low, high = result
        return np.r_[np.asarray(high)[hi], -np.asarray(low)[lo]]
    return HardBlock(A, b, evaluate=values if evaluate else None)


def norms(matrix, residual, upper, evaluate=None):
    matrix = sparse.coo_matrix(matrix)
    residual = np.asarray(residual)
    count, dim = residual.shape
    # One whole cone per vector, head first; b-A delta = [upper, r+J delta].
    row = (matrix.row // dim) * (dim + 1) + 1 + matrix.row % dim
    A = sparse.csr_matrix((-matrix.data, (row, matrix.col)),
                          shape=(count * (dim + 1), matrix.shape[1]))
    b = np.column_stack((upper, residual)).ravel()

    def values(poses):
        result = evaluate(poses)
        if result is None:
            return None
        value, bound = result
        return np.column_stack((bound, value)).ravel()
    return HardBlock(A, b, (dim + 1,) * count, values if evaluate else None)


def _merged(families, *, scale):
    if not families:
        return []
    matrix = sparse.vstack([item[0] for item in families], format='csr')
    group = _row_groups(matrix)
    count = int(group.max()) + 1
    first = np.full(count, len(group))
    np.minimum.at(first, group, np.arange(len(group)))
    order = np.argsort(first)

    def combine(bounds):
        low, high = np.full(count, -np.inf), np.full(count, np.inf)
        np.maximum.at(low, group, np.concatenate([v[0] for v in bounds]))
        np.minimum.at(high, group, np.concatenate([v[1] for v in bounds]))
        return low[order], high[order]

    lo, hi = combine([(v[1], v[2]) for v in families])
    divisor = (hi - lo) / 2 if scale else np.ones(len(lo))
    if np.any(divisor <= 0) or not np.isfinite(divisor).all():
        raise ValueError('Invalid acquisition row scaling')
    A = sparse.diags(1 / divisor) @ matrix[first[order]]

    def evaluate(poses):
        low, high = combine([v[3](poses) for v in families])
        return low / divisor, high / divisor
    return [intervals(A, lo / divisor, hi / divisor, evaluate)]


def _shape(shape, poses, box):
    if getattr(shape, 'continuity_only', False):
        # Seed initialization's adapter only enables constrained fallback.
        # Its immutable-seed guard is assembled from candidate_guards below.
        return []
    shape.kin.evaluate(poses)
    coords = shape.coords
    blocks, layout = [], []
    if coords.root_q is not None:
        jac = np.zeros((3, coords.width))
        for axis in range(3):
            column = coords.inverse.get(coords.root_d + axis)
            if column is not None:
                jac[axis, column] = 1.
        matrix = sparse.kron(sparse.eye(len(poses)), jac, format='csr')
        q = coords.root_q
        references = [shape.seed] + ([] if shape.acquired_poses is None else [shape.acquired_poses])
        for reference in references:
            bound = np.full(len(poses), shape.xy)
            def value(p, reference=reference, bound=bound):
                return p[:, q:q + 3] - reference[:, q:q + 3], bound
            residual, _ = value(poses)
            blocks.append(norms(matrix, residual, bound, value))
    for envelope in shape.envelopes:
        _, _, (change, pairs) = envelope.measure(shape.kin, with_jacobian=True)
        value, jac = envelope.layout(change, pairs)
        if value is None:
            continue
        keep = envelope.layout_sign != 0
        if box is not None:
            keep &= np.einsum('fpaw,fw->fpa', np.abs(jac), box) >= value
        keep = keep.ravel()
        if not keep.any():
            continue
        matrix = _block_rows(jac.reshape(len(poses), -1, coords.width), coords.width)[keep]
        lower = -np.maximum(value.ravel()[keep], 0.) - envelope.SLACK / 2
        upper = np.full(len(lower), np.inf)
        def bounds(p, envelope=envelope, keep=keep, upper=upper):
            shape.kin.evaluate(p, with_jacobian=False)
            _, _, (change, _) = envelope.measure(shape.kin)
            value, _ = envelope.layout(change)
            # SOC uses the signed nonlinear remainder; it must restore a flip.
            return -value.ravel()[keep] - envelope.SLACK / 2, upper
        layout.append((matrix, lower, upper, bounds))
    return blocks + _merged(layout, scale=False)


def _correction(guard, poses):
    from .continuity import ROUND_OFF
    rows = guard.linearizations(poses)
    families, blocks = [], []
    for name, (values, jacobian) in rows.items():
        bound = guard.bounds[name].ravel() + ROUND_OFF / 2
        def value(p, name=name, bound=bound):
            residual = guard.linearizations(p)[name][0].ravel()
            return -bound - residual, bound - residual
        lo, hi = value(poses)
        families.append((jacobian, lo, hi, value))
    for name, fixed in guard.norm_bounds.items():
        values, jacobian = rows[name]
        scale = 1 / (fixed + ROUND_OFF / 2)
        def value(p, name=name, scale=scale):
            return guard.linearizations(p)[name][0] * scale[:, None], np.ones(len(scale))
        residual, bound = value(poses)
        blocks.append(norms(sparse.diags(np.repeat(scale, 3)) @ jacobian, residual, bound, value))
    return families, blocks


def _sequence(guard, poses):
    from .._sequence.continuity_guard import LINEARIZATION_SLACK
    signals = guard.signals(poses)
    families, blocks = [], []
    for (name, order), fixed in guard.bounds.items():
        count = len(poses)
        if count <= order:
            continue
        values, jac = signals[name]
        coefficients = {1: [-1, 1], 2: [1, -2, 1], 3: [-1, 3, -3, 1]}[order]
        difference = sparse.diags(coefficients, np.arange(order + 1),
                                  shape=(count - order, count), format='csr', dtype=float)
        bound = fixed + LINEARIZATION_SLACK
        def residual(p, name=name, order=order):
            return np.diff(guard.signals(p)[name][0], n=order, axis=0)
        if name == 'joints':
            matrix = sparse.kron(difference, guard.tasks.coords.selector, format='csr')
            def value(p, residual=residual, bound=bound):
                value = residual(p).ravel()
                return -bound.ravel() - value, bound.ravel() - value
            lo, hi = value(poses)
            families.append((matrix, lo, hi, value))
        else:
            matrix = sparse.kron(difference, sparse.eye(values.shape[1])) @ sparse.block_diag(list(jac))
            def value(p, residual=residual, bound=bound):
                return residual(p), bound
            blocks.append(norms(matrix, residual(poses), bound, value))
    return families, blocks


def _generic(guard, poses, *, nonlinear=True):
    # Explicit qp_bounds is available to new guards. AcquisitionNormalGuard's
    # constraints already delegates to repair_bounds with exactly this algebra.
    provider = getattr(guard, 'qp_bounds', None) or guard.repair_bounds
    linear, cones = provider(poses)
    cached_pose, cached_rows = np.array(poses, copy=True), (linear, cones)
    def at(p):
        nonlocal cached_pose, cached_rows
        if not np.array_equal(p, cached_pose):
            cached_rows = provider(p)
            cached_pose = np.array(p, copy=True)
        return cached_rows
    blocks = []
    for i, (matrix, lo, hi) in enumerate(linear):
        def value(p, i=i):
            _, lo, hi = at(p)[0][i]
            return lo, hi
        blocks.append(intervals(matrix, lo, hi, value if nonlinear else None))
    for i, (matrix, residual, bound) in enumerate(cones):
        def value(p, i=i):
            _, residual, bound = at(p)[1][i]
            return residual, bound
        blocks.append(norms(matrix, residual, bound, value if nonlinear else None))
    return blocks


def guard_blocks(objective, poses, lower, upper):
    from .reference import ReferenceShape
    from .continuity import CorrectionContinuity
    from .acquisition_guard import AcquisitionNormalGuard
    from .._sequence.continuity_guard import SequenceContinuityGuard
    from .proposals import guards_of
    shape, guards = guards_of(objective)
    count = lower.size
    blocks = [intervals(sparse.eye(count, format='csr'), lower.ravel(), upper.ravel())]
    merged = []
    for guard in guards:
        if isinstance(guard, ReferenceShape):
            box = np.maximum(np.abs(lower), np.abs(upper))
            blocks += _shape(guard, poses, np.where(np.isfinite(box), box, 1e3))
        elif isinstance(guard, CorrectionContinuity):
            families, extra = _correction(guard, poses)
            merged += families
            blocks += extra
        elif isinstance(guard, SequenceContinuityGuard):
            families, extra = _sequence(guard, poses)
            merged += families
            blocks += extra
        elif isinstance(guard, AcquisitionNormalGuard) or hasattr(guard, 'qp_bounds'):
            # The coordinator's rows freeze original-geometry witnesses at
            # the committed point; qp_bounds(candidate) is intentionally not
            # a nonlinear-value API. It still checks every trial exactly.
            blocks += _generic(guard, poses, nonlinear=guard is not getattr(objective, 'priority_clearance', None))
        else:
            raise UnsupportedRows(f'No acquisition row adapter for {type(guard).__name__}')
    return blocks + _merged(merged, scale=True)
