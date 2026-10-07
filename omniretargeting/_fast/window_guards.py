"""Exact hard-guard kernels on original stencils touching a variable window.

All inputs remain full trajectories; all Jacobian columns are window-local.
Immutable bounds are sliced, never recalculated on a short trajectory.
Guard wrappers own rejection counters and semantic policy.
"""

from collections import OrderedDict
import copy
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from .frame_kinematics import frame_range, window_kinematics
from .rows import _block_rows


def _kind(guard):
    from .continuity import CorrectionContinuity
    from .reference import ReferenceShape
    from .._sequence.continuity_guard import SequenceContinuityGuard
    if isinstance(guard, CorrectionContinuity):
        return 'correction'
    if isinstance(guard, SequenceContinuityGuard):
        return 'sequence'
    if isinstance(guard, ReferenceShape):
        return 'shape'
    raise TypeError(f'Unsupported window guard: {type(guard).__name__}')


def guard_dependencies(guard, frames, count):
    frames = frame_range(frames, count)
    kind = _kind(guard)
    halo = 0 if kind == 'shape' else 3
    if kind == 'shape' and _layout_deadband_enabled(count):
        from ..pose_relations import LAYOUT_FLIP_WINDOW_FRAMES
        halo = LAYOUT_FLIP_WINDOW_FRAMES - 1
    return range(max(0, frames.start - halo), min(count, frames.stop + halo))


def _layout_deadband_enabled(count):
    return count >= 600 and False


def _span(frames, count, order):
    return max(0, frames.start - order), min(count - order, frames.stop)


def _operator_cache(guard, key, build):
    """Only constant stencil matrices survive a pose/weight change."""
    cache = getattr(guard, '_window_operators', None)
    if cache is None:
        guard._window_operators = cache = OrderedDict()
    if key not in cache:
        cache[key] = build()
    cache.move_to_end(key)
    while len(cache) > 64:
        cache.popitem(last=False)
    return cache[key]


def _correction(guard, poses, frames, jacobian):
    from .continuity import derivatives, correction_derivatives
    context = guard_dependencies(guard, frames, len(poses))
    part = np.asarray(poses)[context.start:context.stop]
    if jacobian:
        local = copy.copy(guard)
        local.fixed = guard.fixed[context.start:context.stop]
        # linearizations caches its pose-dependent values separately from
        # immutable difference operators. Reuse only the latter across poses.
        key = ('correction', guard.correction, len(context), guard.coords.cols.tobytes())
        local._linear_cache = None
        local._operators = _operator_cache(guard, key, dict)
        values = local.linearizations(part)
    else:
        fixed = guard.fixed[context.start:context.stop]
        raw = (correction_derivatives(guard.coords, fixed, part) if guard.correction else
               derivatives(guard.coords, part))
        values = {name: (value, None) for name, value in raw.items()}
    result = {}
    for name in guard.bounds:
        order = int(name.rsplit('_', 1)[1])
        lo, hi = _span(frames, len(poses), order)
        value, matrix = values[name]
        chosen = slice(lo - context.start, hi - context.start)
        if matrix is not None:
            size = value.shape[1]
            matrix = matrix[chosen.start * size:chosen.stop * size,
                            (frames.start - context.start) * guard.coords.width:
                            (frames.stop - context.start) * guard.coords.width]
        result[name] = value[chosen], matrix, lo, hi, order
    return result


def _sequence_signals(guard, poses, context, jacobian):
    """Use the original signal algebra and share only exact root-body FK."""
    from .._sequence.continuity_guard import SequenceContinuityGuard
    from .._sequence.tasks import SequenceTasks
    if type(guard.tasks) is not SequenceTasks or guard.tasks.root_body is None:
        return guard.signals(poses, (context.start, context.stop))
    local = copy.copy(guard)
    local.tasks = copy.copy(guard.tasks)
    local.tasks.root_body = None
    local._signals_cache = OrderedDict()
    # Original scalar/quaternion operations, with no redundant mj_forward.
    signals = SequenceContinuityGuard.signals(local, poses, (context.start, context.stop))
    owner = SimpleNamespace(model=guard.tasks.model, coords=guard.tasks.coords)
    if hasattr(guard, '_window_frame_cache'):
        owner._window_frame_cache = guard._window_frame_cache
    kin = window_kinematics(owner, poses, context, jacobian)
    guard._window_frame_cache = owner._window_frame_cache
    point = kin.xpos[:, guard.tasks.root_body] - guard.tasks.roots[context.start:context.stop]
    if guard.correction:
        point -= guard.reference['root_anchor_offset'][0][context.start:context.stop]
    anchor = point, kin.jacp[:, guard.tasks.root_body] if jacobian else None
    result = {}
    for name, value in signals.items():
        if name == 'orientation_offset':
            result['root_anchor_offset'] = anchor
        result[name] = value
    return result


def _sequence(guard, poses, frames, jacobian):
    context = guard_dependencies(guard, frames, len(poses))
    signals = _sequence_signals(guard, poses, context, jacobian)
    result = {}
    for (name, order) in guard.bounds:
        lo, hi = _span(frames, len(poses), order)
        values, jac = signals[name]
        chosen = slice(lo - context.start, hi - context.start)
        residual = np.diff(values, n=order, axis=0)[chosen]
        matrix = None
        if jacobian:
            coefficients = {1: [-1, 1], 2: [1, -2, 1], 3: [-1, 3, -3, 1]}[order]
            key = ('sequence', len(context), chosen.start, chosen.stop,
                   order, values.shape[1], guard.tasks.coords.cols.tobytes())
            def build():
                difference = sparse.diags(coefficients, np.arange(order + 1),
                    shape=(len(context) - order, len(context)), format='csr', dtype=float)[chosen]
                return difference, sparse.kron(difference, guard.tasks.coords.selector, format='csr') if name == 'joints' else None
            # Include signal kind: a joint pack can also have 3 or 4 columns.
            difference, joints = _operator_cache(guard, (*key, name), build)
            if name == 'joints':
                matrix = joints
            else:
                matrix = sparse.kron(difference, sparse.eye(values.shape[1]), format='csr') @ _block_rows(jac, guard.tasks.coords.width)
            matrix = matrix[:, (frames.start - context.start) * guard.tasks.coords.width:
                               (frames.stop - context.start) * guard.tasks.coords.width]
        result[name, order] = residual, matrix, lo, hi, order
    return result


def _shape_kin(guard, poses, frames, jacobian):
    owner = SimpleNamespace(model=guard.kin.model, coords=guard.coords)
    if hasattr(guard, '_window_frame_cache'):
        owner._window_frame_cache = guard._window_frame_cache
    kin = window_kinematics(owner, poses, frames, jacobian)
    guard._window_frame_cache = owner._window_frame_cache
    return kin


def guard_violation_frames(guard, poses, *, frames):
    frames = frame_range(frames, len(poses))
    bad = np.zeros(len(poses), bool)
    kind = _kind(guard)
    if kind == 'shape':
        if _layout_deadband_enabled(len(poses)):
            return _shape_history_violations(guard, poses, frames)
        part = np.asarray(poses)[frames.start:frames.stop]
        finite = np.isfinite(part).all(axis=1)
        bad[frames.start:frames.stop] = ~finite
        # Invalid quaternion rows cannot be forwarded by MuJoCo reliably.
        if not finite.all():
            return bad
        kin = _shape_kin(guard, poses, frames, False)
        for envelope in guard.envelopes:
            bad[frames.start:frames.stop] |= envelope.for_frames(slice(frames.start, frames.stop)).violations(kin)
        return bad
    from .continuity import ROUND_OFF
    from .._sequence.continuity_guard import ACCEPTANCE_SLACK, _change_size
    values = _correction(guard, poses, frames, False) if kind == 'correction' else _sequence(guard, poses, frames, False)
    for key, (value, _, lo, hi, order) in values.items():
        bound = guard.bounds[key][lo:hi]
        size = np.abs(value) if kind == 'correction' else _change_size(key[0], value)
        rows = ~(size <= bound + (ROUND_OFF if kind == 'correction' else ACCEPTANCE_SLACK))
        if rows.ndim > 1:
            rows = rows.any(axis=1)
        if kind == 'correction' and key in guard.norm_bounds:
            rows |= ~(np.linalg.norm(value, axis=1) <= guard.norm_bounds[key][lo:hi] + ROUND_OFF)
        for offset in range(order + 1):
            bad[lo + offset:hi + offset] |= rows
    return bad


def guard_accepts(guard, poses, *, frames):
    return not guard_violation_frames(guard, poses, frames=frames).any()


def _shape_history_violations(guard, poses, frames):
    """Read the preceding streak; edits can change the following six verdicts."""
    context = guard_dependencies(guard, frames, len(poses))
    affected = slice(frames.start, context.stop)
    chosen = slice(frames.start - context.start, len(context))
    bad = np.zeros(len(poses), bool)
    if not np.isfinite(np.asarray(poses)[context.start:context.stop]).all():
        # Do not forward an invalid quaternion from the historical context.
        bad[affected] = True
        return bad
    kin = _shape_kin(guard, poses, context, False)
    for envelope in guard.envelopes:
        local = envelope.for_frames(slice(context.start, context.stop))
        bad[affected] |= local.violations(kin)[chosen]
    return bad


def _shape_bounds(guard, poses, frames, box, jacobian):
    deadband = _layout_deadband_enabled(len(poses))
    context = guard_dependencies(guard, frames, len(poses)) if deadband else frames
    kin = _shape_kin(guard, poses, context, jacobian or box is not None)
    linear, cones = [], []
    coords, span = guard.coords, slice(frames.start, frames.stop)
    if coords.root_q is not None:
        select = np.zeros((3, coords.width))
        for axis in range(3):
            column = coords.inverse.get(coords.root_d + axis)
            if column is not None:
                select[axis, column] = 1.
        matrix = sparse.kron(sparse.eye(len(frames)), select, format='csr') if jacobian else None
        for reference in [guard.seed] + ([] if guard.acquired_poses is None else [guard.acquired_poses]):
            q = coords.root_q
            cones.append((matrix, poses[span, q:q + 3] - reference[span, q:q + 3],
                          np.full(len(frames), guard.xy)))
    for envelope in guard.envelopes:
        envelope = envelope.for_frames(slice(context.start, context.stop))
        _, _, (change, pairs) = envelope.measure(kin, with_jacobian=jacobian or box is not None)
        value, derivative = envelope.layout(change, pairs)
        if value is None:
            continue
        keep = envelope.layout_sign != 0
        if deadband:
            from ..pose_relations import LAYOUT_FLIP_MARGIN, LAYOUT_FLIP_WINDOW_FRAMES, _sustained
            floor = np.where(_sustained(value < -LAYOUT_FLIP_MARGIN,
                                       LAYOUT_FLIP_WINDOW_FRAMES), 0., -LAYOUT_FLIP_MARGIN)
            chosen = slice(frames.start - context.start, frames.stop - context.start)
            value, keep, floor = value[chosen], keep[chosen], floor[chosen]
            if derivative is not None:
                derivative = derivative[chosen]
        if box is not None:
            reach = np.einsum('fpaw,fw->fpa', np.abs(derivative), box)
            keep &= reach >= (value - floor if deadband else value)
        keep = keep.ravel()
        if keep.any():
            matrix = (_block_rows(derivative.reshape(len(frames), -1, coords.width), coords.width)[keep]
                      if jacobian else None)
            # Signed nonlinear remainder (SOC), including existing flips.
            kept = value.ravel()[keep]
            lower = (floor.ravel()[keep] - kept - envelope.SLACK / 2 if deadband else
                     -(np.maximum(kept, 0.) if jacobian else kept) - envelope.SLACK / 2)
            linear.append((matrix, lower, np.full(len(lower), np.inf)))
    return linear, cones


def _bounds(guard, poses, frames, box, jacobian):
    frames = frame_range(frames, len(poses))
    kind = _kind(guard)
    if kind == 'shape':
        return _shape_bounds(guard, poses, frames, box, jacobian)
    from .continuity import ROUND_OFF
    from .._sequence.continuity_guard import LINEARIZATION_SLACK
    values = _correction(guard, poses, frames, jacobian) if kind == 'correction' else _sequence(guard, poses, frames, jacobian)
    linear, cones = [], []
    for key, (value, matrix, lo, hi, _) in values.items():
        bound = guard.bounds[key][lo:hi]
        if kind == 'correction' or key[0] == 'joints':
            bound = bound.ravel() + (ROUND_OFF / 2 if kind == 'correction' else LINEARIZATION_SLACK)
            linear.append((matrix, -bound - value.ravel(), bound - value.ravel()))
        else:
            cones.append((matrix, value, bound + LINEARIZATION_SLACK))
    if kind == 'correction':
        for key, fixed in guard.norm_bounds.items():
            value, matrix, lo, hi, _ = values[key]
            scale = 1 / (fixed[lo:hi] + ROUND_OFF / 2)
            matrix = sparse.diags(np.repeat(scale, 3)) @ matrix if jacobian else None
            cones.append((matrix, value * scale[:, None], np.ones(len(scale))))
    return linear, cones


def guard_qp_bounds(guard, poses, *, frames, box=None):
    return _bounds(guard, poses, frames, box, True)


def guard_qp_values(guard, poses, *, frames, box=None):
    """Same row order as unscreened qp_bounds, without rebuilding Jacobians.

    Use box=None for SOC: screening at a changed pose would change row IDs.
    """
    return _bounds(guard, poses, frames, box, False)
