"""Window kernels retain original budgets, row identities and quaternion math."""

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.continuity import CorrectionContinuity, ROUND_OFF
from omniretargeting._fast.reference import ReferenceShape
from omniretargeting._fast.window_guards import (
    guard_accepts, guard_dependencies, guard_qp_bounds, guard_qp_values,
    guard_violation_frames,
)
from omniretargeting._sequence.continuity_guard import (
    ACCEPTANCE_SLACK, LINEARIZATION_SLACK, SequenceContinuityGuard, _change_size,
)
from omniretargeting._fast.rows import _block_rows
from tests.test_fast_backend import _objective


def _expected_mask(guard, poses, frames, correction):
    if correction:
        rows = guard.linearizations(poses)
        values = {name: (value, int(name.rsplit('_', 1)[1])) for name, (value, _) in rows.items()}
    else:
        signals = guard.signals(poses)
        values = {key: (np.diff(signals[key[0]][0], n=key[1], axis=0), key[1]) for key in guard.bounds}
    bad = np.zeros(len(poses), bool)
    for key, (value, order) in values.items():
        lo, hi = max(0, frames.start - order), min(len(poses) - order, frames.stop)
        size = np.abs(value[lo:hi]) if correction else _change_size(key[0], value[lo:hi])
        reject = ~(size <= guard.bounds[key][lo:hi] + (ROUND_OFF if correction else ACCEPTANCE_SLACK))
        reject = reject.any(axis=1) if reject.ndim > 1 else reject
        if correction and key in guard.norm_bounds:
            reject |= ~(np.linalg.norm(value[lo:hi], axis=1) <= guard.norm_bounds[key][lo:hi] + ROUND_OFF)
        for offset in range(order + 1):
            bad[lo + offset:hi + offset] |= reject
    return bad


@pytest.mark.parametrize('kind', ['correction', 'sequence'])
@pytest.mark.parametrize('correction', [False, True])
@pytest.mark.parametrize('frames', [range(0, 1), range(0, 32), range(4, 36), range(14, 18), range(35, 36)])
def test_window_continuity_masks_equal_full_stencil_restriction(kind, correction, frames):
    obj, poses, _ = _objective(temporal=True)
    guard = (CorrectionContinuity(obj.coords, poses, root_norm=True, correction=correction, chatter=False) if kind == 'correction'
             else SequenceContinuityGuard(obj.tasks, poses, obj.root_rotations, windowed=True, correction=correction, chatter=False))
    changed = obj.coords.integrate(poses, np.random.default_rng(11).normal(scale=.02, size=(len(poses), obj.width)), 1.)
    changed[::2, 3:7] *= -1
    expected = _expected_mask(guard, changed, frames, kind == 'correction')
    np.testing.assert_array_equal(guard_violation_frames(guard, changed, frames=frames), expected)
    assert guard_accepts(guard, changed, frames=frames) == (not expected.any())
    assert guard_dependencies(guard, frames, len(poses)) == range(max(0, frames.start - 3), min(len(poses), frames.stop + 3))


@pytest.mark.parametrize('correction', [False, True])
@pytest.mark.parametrize('kind', ['correction', 'sequence'])
def test_window_qp_rows_equal_full_rows_with_fixed_columns_removed(kind, correction):
    obj, poses, _ = _objective(temporal=True)
    guard = (CorrectionContinuity(obj.coords, poses, root_norm=True, correction=correction, chatter=False) if kind == 'correction'
             else SequenceContinuityGuard(obj.tasks, poses, obj.root_rotations, windowed=True, correction=correction, chatter=False))
    changed = obj.coords.integrate(poses, np.random.default_rng(14).normal(scale=.002, size=(len(poses), obj.width)), 1.)
    frames = range(7, 21)
    linear, cones = guard_qp_bounds(guard, changed, frames=frames)
    lvalues, cvalues = guard_qp_values(guard, changed, frames=frames)
    for bounds, values in zip(linear + cones, lvalues + cvalues):
        assert values[0] is None
        for actual, expected in zip(values[1:], bounds[1:]):
            np.testing.assert_array_equal(actual, expected)
    cols = slice(frames.start * obj.width, frames.stop * obj.width)
    if kind == 'correction':
        rows = guard.linearizations(changed)
        for (name, (value, matrix)), (actual, lo, hi) in zip(rows.items(), linear):
            order, dim = int(name.rsplit('_', 1)[1]), value.shape[1]
            first, stop = max(0, frames.start - order), min(len(poses) - order, frames.stop)
            chosen = slice(first * dim, stop * dim)
            np.testing.assert_array_equal(actual.toarray(), matrix[chosen, cols].toarray())
            bound = guard.bounds[name][first:stop].ravel() + ROUND_OFF / 2
            np.testing.assert_array_equal(lo, -bound - value[first:stop].ravel())
            np.testing.assert_array_equal(hi, bound - value[first:stop].ravel())
    else:
        signals = guard.signals(changed)
        scalar, vector = iter(linear), iter(cones)
        for (name, order), bound in guard.bounds.items():
            value, jac = signals[name]
            first, stop = max(0, frames.start - order), min(len(poses) - order, frames.stop)
            coefficients = {1: [-1, 1], 2: [1, -2, 1], 3: [-1, 3, -3, 1]}[order]
            difference = sparse.diags(coefficients, range(order + 1), shape=(len(poses) - order, len(poses)), format='csr')
            matrix = (sparse.kron(difference, obj.coords.selector, format='csr') if name == 'joints'
                      else sparse.kron(difference, sparse.eye(value.shape[1]), format='csr') @ _block_rows(jac, obj.width))
            actual, low, high = next(scalar if name == 'joints' else vector)
            dim = value.shape[1]
            np.testing.assert_array_equal(actual.toarray(), matrix[first * dim:stop * dim, cols].toarray())
            residual = np.diff(value, n=order, axis=0)[first:stop]
            if name == 'joints':
                np.testing.assert_array_equal(low, -(bound[first:stop].ravel() + LINEARIZATION_SLACK) - residual.ravel())
            else:
                np.testing.assert_array_equal(low, residual)


@pytest.mark.parametrize('frames', [range(0, 1), range(5, 25), range(35, 36)])
def test_window_layout_decisions_and_unscreened_rows_are_full_restrictions(frames):
    obj, poses, _ = _objective(temporal=True)
    guard = ReferenceShape(obj, poses)
    guard.freeze_reference(poses)
    changed = obj.coords.integrate(poses, np.random.default_rng(19).normal(scale=.2, size=(len(poses), obj.width)), 1.)
    changed[::2, 3:7] *= -1
    expected = guard.violation_frames(changed)
    expected[:frames.start] = False
    expected[frames.stop:] = False
    np.testing.assert_array_equal(guard_violation_frames(guard, changed, frames=frames), expected)
    assert guard_accepts(guard, changed, frames=frames) == (not expected.any())
    assert guard_dependencies(guard, frames, len(poses)) == frames
    linear, cones = guard_qp_bounds(guard, changed, frames=frames)
    guard.kin.evaluate(changed)
    for envelope, (matrix, lo, hi) in zip(guard.envelopes, linear):
        _, _, (change, pairs) = envelope.measure(guard.kin, with_jacobian=True)
        value, jac = envelope.layout(change, pairs)
        keep = envelope.layout_sign[frames.start:frames.stop].ravel() != 0
        original = _block_rows(jac[frames.start:frames.stop].reshape(len(frames), -1, obj.width), obj.width)[keep]
        np.testing.assert_array_equal(matrix.toarray(), original.toarray())
        np.testing.assert_array_equal(lo, -np.maximum(value[frames.start:frames.stop].ravel()[keep], 0.) - envelope.SLACK / 2)
    assert cones[0][0].shape == (3 * len(frames), obj.width * len(frames))


def test_window_threshold_ties_and_unrelated_outside_failures():
    obj, poses, _ = _objective(temporal=True)
    guard = CorrectionContinuity(obj.coords, poses, chatter=False)
    # Set immutable test bounds to the current exact derivative magnitudes:
    # equality at the existing acceptance reserve must still pass.
    for name, (value, _) in guard.linearizations(poses).items():
        guard.bounds[name] = np.maximum(0., np.abs(value) - ROUND_OFF)
    assert guard_accepts(guard, poses, frames=range(10, 20))
    outside = poses.copy()
    outside[0, 7] += .1
    assert not guard.accepts(outside)
    assert guard_accepts(guard, outside, frames=range(10, 20))


def test_default_chatter_guard_has_no_local_derivative_rows():
    obj, poses, _ = _objective(temporal=True)
    direct = CorrectionContinuity(obj.coords, poses, chatter=True)
    sequence = SequenceContinuityGuard(obj.tasks, poses, obj.root_rotations, windowed=True,
                                       chatter=True)
    assert direct.bounds == {}
    assert sequence.bounds == {}
    assert direct.constraints(poses, np.zeros(1)) == []
    assert sequence.constraints(poses, np.zeros(1)) == []
