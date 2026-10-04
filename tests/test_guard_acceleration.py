"""Exact guard primitives, including sparse storage and reusable inputs."""

from types import SimpleNamespace

import numpy as np
import pytest

from omniretargeting._fast.kinematics import TrajectoryKinematics
from omniretargeting._fast.rows import _block_rows


def _same_bits(first, second):
    assert first.shape == second.shape
    assert first.dtype == second.dtype
    assert first.tobytes() == second.tobytes()


@pytest.mark.parametrize('shape', [(0, 3, 7), (5, 0, 7), (4, 3, 0), (5, 6, 9)])
def test_csr_preserves_original_storage_and_dense_input(monkeypatch, shape):
    blocks = np.random.default_rng(63).normal(size=shape)
    blocks[abs(blocks) < .9] = -0.
    before = blocks.copy()
    monkeypatch.setenv('OMNIRETARGETING_FAST_GUARDS', '0')
    expected = _block_rows(blocks, shape[-1])
    monkeypatch.setenv('OMNIRETARGETING_FAST_GUARDS', '1')
    actual = _block_rows(blocks, shape[-1])
    _same_bits(actual.data, expected.data)
    np.testing.assert_array_equal(actual.indices, expected.indices)
    np.testing.assert_array_equal(actual.indptr, expected.indptr)
    _same_bits(blocks, before)


@pytest.mark.parametrize('width', [1, 7, 35])
@pytest.mark.parametrize('varying', [False, True])
def test_native_point_jacobians_keep_unfused_arithmetic(monkeypatch, width, varying):
    rng = np.random.default_rng(413)
    kin = TrajectoryKinematics.__new__(TrajectoryKinematics)
    kin.model = SimpleNamespace(nbody=11)
    # Subset columns produce non-contiguous arrays in the real repair path.
    kin.jacp = rng.normal(size=(9, 11, 3, width * 2))[..., ::2]
    kin.jacr = rng.normal(size=(9, 11, 3, width * 2))[..., ::2]
    kin.xpos = rng.normal(size=(9, 11, 3))
    bodies = rng.integers(0, 11, size=(9, 17) if varying else 17)
    points = rng.normal(size=(9, 17, 3))
    monkeypatch.setenv('OMNIRETARGETING_FAST_GUARDS', '0')
    expected = kin.point_jacobians(bodies, points)
    monkeypatch.setenv('OMNIRETARGETING_FAST_GUARDS', '1')
    _same_bits(kin.point_jacobians(bodies, points), expected)
    # A new pose/Jacobian must invalidate the retained contiguous buffers.
    kin.jacp = kin.jacp.copy() * 1.2
    monkeypatch.setenv('OMNIRETARGETING_FAST_GUARDS', '0')
    expected = kin.point_jacobians(bodies, points)
    monkeypatch.setenv('OMNIRETARGETING_FAST_GUARDS', '1')
    _same_bits(kin.point_jacobians(bodies, points), expected)


def test_native_failure_and_negative_indices_retain_numpy(monkeypatch):
    from omniretargeting._fast import native_point_rows
    from omniretargeting._fast.batched_self_collision import NativeUnavailable
    from tests.test_fast_backend import _problem

    model, _, coords, poses, _, _ = _problem(frames=8)
    kin = TrajectoryKinematics(model, coords.cols).evaluate(poses)
    points = kin.xpos[:, [1, 2]].copy()
    monkeypatch.setenv('OMNIRETARGETING_FAST_GUARDS', '0')
    expected = kin.point_jacobians([1, 2], points)

    def unavailable(*args):
        raise NativeUnavailable('fixture missing compiler')

    monkeypatch.setattr(native_point_rows, 'NativePointRows', unavailable)
    monkeypatch.setenv('OMNIRETARGETING_FAST_GUARDS', '1')
    _same_bits(kin.point_jacobians([1, 2], points), expected)
    _same_bits(kin.point_jacobians([1 - model.nbody, 2], points), expected)


@pytest.mark.parametrize('correction', [False, True])
@pytest.mark.parametrize('span', [None, (2, 11)])
def test_temporal_root_signals_reuse_fk_and_keep_guard_decisions(monkeypatch, correction, span):
    from scipy.spatial.transform import Rotation
    from omniretargeting._sequence.continuity_guard import SequenceContinuityGuard
    from omniretargeting._fast.guard_signals import prepare
    from tests.test_fast_backend import _objective

    objective, poses, _ = _objective(temporal=True)
    rotations = Rotation.from_quat(poses[:, 3:7], scalar_first=True)
    guard = SequenceContinuityGuard(objective.tasks, poses, rotations, windowed=True,
                                    correction=correction)
    objective.candidate_guards = [guard]
    changed = objective.coords.integrate(poses,
        np.random.default_rng(73).normal(scale=.01, size=(len(poses), objective.width)), 1.)
    monkeypatch.setenv('OMNIRETARGETING_FAST_GUARDS', '0')
    original = guard.signals(changed, span)
    old_accepts = guard.accepts(changed)
    old_bad = guard.violation_frames(changed)
    guard._signals_cache.clear()
    monkeypatch.setenv('OMNIRETARGETING_FAST_GUARDS', '1')
    prepare(objective)
    assert type(guard).__name__ == 'SequenceContinuityGuard'
    actual = guard.signals(changed, span)
    assert list(actual) == list(original)
    for name in original:
        for before, after in zip(original[name], actual[name]):
            _same_bits(before, after)
    assert guard.accepts(changed) == old_accepts
    np.testing.assert_array_equal(guard.violation_frames(changed), old_bad)
    assert bool(old_bad.any()) == (not old_accepts)


def _same_sparse(first, second):
    assert first.shape == second.shape
    for name in ('data', 'indices', 'indptr'):
        _same_bits(getattr(first, name), getattr(second, name))


def test_guard_constraints_repair_bounds_and_per_frame_checks_are_exact(monkeypatch):
    import cvxpy as cp
    from scipy.spatial.transform import Rotation
    from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
    from omniretargeting._fast.contact_guard import ContactGuard
    from omniretargeting._fast.continuity import CorrectionContinuity
    from omniretargeting._fast.guard_signals import prepare
    from omniretargeting._fast.reference import ReferenceShape
    from omniretargeting._fast.repair_limits import RepairLimits
    from omniretargeting._fast.shape_step import _ShapeRestoration
    from omniretargeting._sequence.continuity_guard import SequenceContinuityGuard
    from tests.test_fast_contact_guard import _acquired

    monkeypatch.setenv('OMNIRETARGETING_FAST_GUARDS', '0')
    objective, poses = _acquired(surfaces=True, moving=True)
    objective.source_shape = ReferenceShape(objective, poses)
    objective.candidate_guards = [
        AcquisitionNormalGuard(objective, poses),
        SequenceContinuityGuard(objective.tasks, poses,
            Rotation.from_quat(poses[:, 3:7], scalar_first=True), windowed=True),
        CorrectionContinuity(objective.coords, poses),
        ContactGuard(objective, poses, preserve_positions=True),
    ]
    limits = RepairLimits(objective, poses)
    restoration = _ShapeRestoration(objective)
    flat = cp.Variable(objective.frames * objective.width)
    lo, hi = objective.coords.bounds(poses, .1, .005)

    def canonical():
        guards = objective.candidate_guards
        objective.candidate_guards = [g for g in guards if hasattr(g, 'constraints')]
        try:
            constraints = restoration.restoration_constraints(poses, flat, lo.copy(), hi.copy(), {}, None)
        finally:
            objective.candidate_guards = guards
        problem = cp.Problem(cp.Minimize(cp.sum_squares(flat)), constraints)
        data = problem.get_problem_data(cp.CLARABEL)[0]
        return {name: data[name] for name in ('A', 'b', 'P', 'c')}

    expected = canonical()
    old_bounds = limits.constraints(poses)
    old_report = limits.check(poses)
    monkeypatch.setenv('OMNIRETARGETING_FAST_GUARDS', '1')
    prepare(objective)
    for guard in objective.candidate_guards:
        if hasattr(guard, '_signals_cache'):
            guard._signals_cache.clear()
    actual = canonical()
    for name in ('A', 'P'):
        _same_sparse(actual[name], expected[name])
    for name in ('b', 'c'):
        _same_bits(actual[name], expected[name])
    bounds = limits.constraints(poses)
    for first, second in zip(bounds.matrices, old_bounds.matrices):
        _same_sparse(first, second)
    for name in ('lower', 'upper'):
        for first, second in zip(getattr(bounds, name), getattr(old_bounds, name)):
            _same_bits(first, second)
    for new, old in zip(bounds.norms, old_bounds.norms):
        _same_sparse(new[0], old[0])
        _same_bits(new[1], old[1])
        _same_bits(new[2], old[2])
    assert limits.check(poses) == old_report
    for mode in ('0', '1'):
        monkeypatch.setenv('OMNIRETARGETING_FAST_GUARDS', mode)
        for change in (0., .01):
            candidate = poses.copy()
            candidate[:, 2] += change
            for guard in objective.candidate_guards:
                assert bool(guard.violation_frames(candidate).any()) == (not guard.accepts(candidate))
