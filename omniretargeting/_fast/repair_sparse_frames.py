"""Assemble frame-local repair constraints only where a proposal can move.

The unchanged nonlinear check certifies the origin. Frames outside the
coordinate selector remain byte-identical to that origin, so their local
constraints are satisfied constants. Temporal/support constraints retain
their full original operators, allowances and crossing boundary stencils.
Unknown guard contracts fall back to the original full assembly.
"""

import numpy as np
from scipy import sparse

from .acquisition_guard import AcquisitionNormalGuard
from .contact_guard import ContactGuard
from .continuity import CorrectionContinuity
from .guard_signals import CachedSequenceContinuityGuard
from .reference import ReferenceShape
from .repair_contact_rows import _shift
from .repair_gate_frames import local_limits
from .repair_limits import NUMERICAL_SLACK
from .repair_projected import ProjectedBounds
from .runtime_options import OPTIONS
from .._sequence.continuity_guard import SequenceContinuityGuard

TEMPORAL = (CorrectionContinuity, SequenceContinuityGuard, CachedSequenceContinuityGuard)
CONTACT = (ContactGuard, AcquisitionNormalGuard)


def _runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(edges[::2], edges[1::2])


def constraints(limits, poses, *, restore_layout=False, projection=None):
    """Projected full-domain bounds, or None when constant safety is unknown."""
    if projection is None or OPTIONS.repair_aggregate:
        return None
    selector = sparse.csc_matrix(projection)
    if (not np.all(np.diff(selector.indptr) == 1) or not np.all(selector.data == 1.)
            or len(np.unique(selector.indices)) != selector.shape[1]):
        return None
    active = np.zeros(limits.frames, bool)
    active[selector.indices // limits.width] = True
    origin = getattr(limits, '_repair_sparse_origin', None)
    if (origin is None or origin.shape != poses.shape
            or not getattr(limits, '_repair_sparse_origin_valid', False)
            or not np.array_equal(poses[~active], origin[~active])):
        return None
    objective = limits.objective
    shape = getattr(objective, 'source_shape', None)
    guards = getattr(objective, 'candidate_guards', ())
    if ((shape is not None and type(shape) is not ReferenceShape)
            or any(type(g) not in CONTACT + TEMPORAL for g in guards)
            or any(getattr(g, 'priority', None) is not None for g in guards)):
        return None
    # Build the existing local rows on private views. No boundary stencils
    # are taken from these views; all temporal families are added below.
    bounds = ProjectedBounds(limits.frames * limits.width, selector, limits.frames, limits.width)
    values = poses[:, limits.coords.qi].ravel()
    bounds.set_box(limits.lower.ravel() - values - 1e-8,
                   limits.upper.ravel() - values + 1e-8)
    for start, stop in _runs(active):
        view = local_limits(limits, int(start), int(stop))
        if view is None:
            return None
        view._repair_frame_view = True
        local = selector[start * limits.width:stop * limits.width]
        used = np.diff(local.indptr) != 0
        part = view.constraints(poses[start:stop], restore_layout=restore_layout,
                                projection=local[:, used])
        # The full box above owns box rows and redundancy. The remainder is
        # embedded without packing/rescaling or shortening a vector cone.
        for matrix, low, high in zip(part.matrices[1:], part.lower[1:], part.upper[1:]):
            bounds.add(_shift(matrix, start * limits.width, bounds.size), low, high)
        for matrix, residual, allowed in part.norms:
            bounds.norms.append((_shift(matrix, start * limits.width, bounds.size), residual, allowed))
    for op, budget in limits.differences:
        current = op @ (values - limits.entry[:, limits.coords.qi].ravel()
                        if OPTIONS.correction_budget else values)
        slack = np.minimum(NUMERICAL_SLACK, np.maximum(NUMERICAL_SLACK / 4, np.abs(current) - budget))
        bounds.add(op, -budget - slack - current, budget + slack - current)
    # Support points can depend on scalar joints, even with a fixed root.
    # Preserve their complete world-space rows, including support handoffs.
    limits.kin.evaluate(poses)
    limits.support_continuity.add(bounds, limits.kin)
    scalar = sparse.kron(sparse.eye(limits.frames), limits.coords.selector.T, format='csr')
    from .repair_continuity_rows import continuity_bounds
    for guard in guards:
        if type(guard) not in TEMPORAL:
            continue
        result = continuity_bounds(guard, poses)
        linear, norms = guard.repair_bounds(poses) if result is None else result
        for matrix, low, high in linear:
            bounds.add(matrix @ scalar, low, high)
        for matrix, residual, allowed in norms:
            bounds.norms.append((matrix @ scalar, residual, allowed))
    return bounds
