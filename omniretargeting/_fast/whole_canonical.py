"""The whole-trajectory acquisition program in CVXPY's exact Clarabel order.

No constraint reduction or alternate precision: scalar rows, root balls and
contact cones retain their original arithmetic and packing. Unknown guards
keep the expression-based path. The existing window adapter owns retries and
restoration working sets, including the original Clarabel tolerances.
"""

from types import SimpleNamespace

import numpy as np
from scipy import sparse

from ..conic_solver import gpu_enabled
from ..pose_relations import _tightest_lower
from .._sequence.continuity_guard import SequenceContinuityGuard
from .acquisition_guard import AcquisitionNormalGuard
from .continuity import CorrectionContinuity
from .guard_signals import CachedSequenceContinuityGuard
from .performance_trace import timed_phase
from .reference import ReferenceShape
from .rows import _merge_identical
from .runtime_options import OPTIONS


def supported(objective):
    """Only contracts whose numerical constraint construction is known here."""
    if not OPTIONS.whole_canonical or gpu_enabled() or OPTIONS.window_qp:
        return False
    if type(getattr(objective, 'source_shape', None)) is not ReferenceShape:
        return False
    # Default rollout is limited to the short-clip acquisition workload.
    if len(objective.source_shape.seed) >= 600:
        return False
    for guard in objective.candidate_guards:
        if type(guard) in (SequenceContinuityGuard, CachedSequenceContinuityGuard):
            if any(name != 'joints' for name, _ in guard.bounds):
                return False
        elif type(guard) not in (AcquisitionNormalGuard, CorrectionContinuity):
            return False
    return True


def _root_cones(shape, poses):
    coords, frames = shape.coords, len(poses)
    if coords.root_q is None:
        return []
    jac = np.zeros((3, coords.width))
    for axis in range(3):
        column = coords.inverse.get(coords.root_d + axis)
        if column is not None:
            jac[axis, column] = 1.
    matrix = sparse.kron(sparse.eye(frames), sparse.csr_matrix(jac), format='csr')
    references = [shape.seed] + ([] if shape.acquired_poses is None else [shape.acquired_poses])
    q = coords.root_q
    return [(sparse.vstack([matrix] * len(references), format='csr'),
             np.concatenate([poses[:, q:q + 3] - reference[:, q:q + 3]
                             for reference in references]),
             np.full(frames * len(references), shape.xy))]


def _constraint_data(objective, poses, lower, upper):
    """Nonnegative rows first; SOC families retain expression insertion order."""
    shape, coords = objective.source_shape, objective.coords
    eye = sparse.eye(lower.size, format='csr')
    rows, targets = [-eye, eye], [-lower.ravel(), upper.ravel()]
    cones = _root_cones(shape, poses)
    shape.kin.evaluate(poses)
    box = np.maximum(np.abs(lower), np.abs(upper))
    box = np.where(np.isfinite(box), box, 1e3)
    layouts = [row for envelope in shape.envelopes
               for row in envelope.linear_bounds(shape.kin, coords.width, box)]
    if layouts:
        matrix, lo = _tightest_lower(
            sparse.vstack([row[0] for row in layouts], format='csr'),
            np.concatenate([row[1] for row in layouts]))
        rows.append(-matrix)
        targets.append(-lo)
    merged = []
    for guard in objective.candidate_guards:
        if type(guard) is AcquisitionNormalGuard:
            intervals, norms = guard.repair_bounds(poses)
            for matrix, lo, hi in intervals:
                # AcquisitionNormalGuard.constraints puts the upper row first.
                rows.extend([matrix, -matrix])
                targets.extend([hi, -lo])
            cones.extend(norms)
        elif type(guard) in (SequenceContinuityGuard, CachedSequenceContinuityGuard):
            # With only scalar bounds this returns no expressions. Reuse its
            # stencil cache and the original allowance/difference arithmetic.
            guard.constraints(poses, None, merge_rows=merged)
        elif type(guard) is CorrectionContinuity:
            if not guard.bounds and not guard.norm_bounds:
                continue
            from .continuity import ROUND_OFF
            linear = guard.linearizations(poses)
            for name, (values, matrix) in linear.items():
                if name in guard.bounds:
                    bound = guard.bounds[name].ravel() + ROUND_OFF / 2
                    merged.append((matrix, -bound - values.ravel(), bound - values.ravel()))
            for name, fixed in guard.norm_bounds.items():
                values, matrix = linear[name]
                scale = 1 / (fixed + ROUND_OFF / 2)
                cones.append((sparse.diags(np.repeat(scale, 3)) @ matrix,
                              values * scale[:, None], np.ones(len(fixed))))
        else:
            raise TypeError(f'Unsupported acquisition guard: {type(guard).__name__}')
    if merged:
        matrix, lo, hi = _merge_identical(
            sparse.vstack([row[0] for row in merged], format='csr'),
            np.concatenate([row[1] for row in merged]),
            np.concatenate([row[2] for row in merged]))
        center, half = (hi + lo) / 2, (hi - lo) / 2
        matrix = sparse.diags(1 / half) @ matrix
        # Match residual = matrix @ flat - center / half, <= 1 then >= -1.
        rows.extend([matrix, -matrix])
        targets.extend([1. + center / half, 1. - center / half])
    nonnegative, sizes = sum(len(value) for value in targets), []
    for matrix, residual, allowance in cones:
        count, dimension = residual.shape
        if not count:
            continue
        jac = matrix.tocoo()
        row = jac.row + jac.row // dimension + 1
        rows.append(sparse.csc_matrix((-jac.data, (row, jac.col)),
                                      shape=(count * (dimension + 1), lower.size)))
        targets.append(np.c_[allowance, residual].ravel())
        sizes.extend([dimension + 1] * count)
    matrix = sparse.vstack(rows, format='csc')
    # CVXPY's affine stuffing omits explicit zeros from dense block rows.
    matrix.eliminate_zeros()
    matrix.sort_indices()
    offset = np.concatenate(targets)
    offset[offset == 0.] = 0.
    return dict(A=matrix, b=offset,
                dims=SimpleNamespace(zero=0, nonneg=nonnegative, soc=sizes,
                                     psd=[], exp=0, p3d=[], pnd=[]))


@timed_phase('canonicalization')
def canonical_data(objective, poses, lower, upper, quadratic, linear):
    data = _constraint_data(objective, poses, lower, upper)
    cost = np.array(linear, copy=True)
    cost[cost == 0.] = 0.
    return dict(data, P=quadratic, c=cost)


def restore_step(objective, origin, step):
    """The unchanged four-iteration whole restoration, without expressions."""
    from .._sequence.restoration import RESTORATION_ITERATIONS
    from .shape_step import WORKING_SET_RESTORATION, _ShapeRestoration, _pin_root
    from .window_canonical import solve_data
    coords = objective.coords
    guard = _ShapeRestoration(objective)
    candidate = objective.project(coords.integrate(origin, step.reshape(len(origin), -1), 1.))
    lower, upper = coords.bounds(origin, objective.cfg['angular_step_rad'],
                                 objective.cfg['translation_step_m'])
    scales = np.maximum(np.maximum(abs(lower), abs(upper)), np.finfo(float).eps)
    weight = sparse.diags(1 / scales.ravel() ** 2, format='csc') * 2
    restore_coords = guard.tasks.coords
    for _ in range(RESTORATION_ITERATIONS):
        if guard.accepts(candidate):
            return objective.project(candidate)
        displacement = restore_coords.difference(origin, candidate)
        lo, hi = lower - displacement, upper - displacement
        lo[:, coords.scalar_cols] = np.maximum(lo[:, coords.scalar_cols],
                                               objective.lower - candidate[:, coords.qi])
        hi[:, coords.scalar_cols] = np.minimum(hi[:, coords.scalar_cols],
                                               objective.upper - candidate[:, coords.qi])
        _pin_root(objective, candidate, lo, hi)
        data = canonical_data(objective, candidate, lo, hi, weight, np.zeros(lower.size))
        delta = solve_data(data, lower.size, working_set=WORKING_SET_RESTORATION)
        if delta is None:
            return None
        candidate = restore_coords.integrate(candidate, delta.reshape(lower.shape), 1)
    return objective.project(candidate) if guard.accepts(candidate) else None
