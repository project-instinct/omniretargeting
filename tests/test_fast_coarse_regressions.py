"""Review regressions: omitted touches, cached guards, lifetime and transfer."""

import weakref

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting._fast import coarse_acquisition as driver
from omniretargeting._fast.coarse_problem import make_coarse_problem
from omniretargeting._fast.coarse_transfer import (
    interpolate_correction, scale_transfer_correction, transfer_correction,
)
from omniretargeting._fast.continuity import CorrectionContinuity
from omniretargeting._fast.guard_signals import CachedSequenceContinuityGuard
from omniretargeting._fast.legacy_window import unsupported_reason
from omniretargeting._sequence.continuity_guard import SequenceContinuityGuard
from tests.test_fast_legacy_window import problem


@pytest.mark.parametrize('all_odd', [False, True])
def test_odd_only_protected_groups_and_names_are_dropped_only_at_coarse_rate(all_odd):
    objective, poses = problem(extra=True)
    guard = objective.candidate_guards[-1]
    assert len(guard.groups) >= 2
    changed = []
    for index, group in enumerate(guard.groups):
        values = list(group)
        active = np.zeros_like(group[5])
        active[1 if all_odd or index == 0 else 2] = True
        values[5] = active
        changed.append(tuple(values))
    guard.groups = changed
    names, originals = guard.names.copy(), [group[5].tobytes() for group in guard.groups]
    coarse, indices = make_coarse_problem(objective)
    sampled = coarse.hinge_guards[-1]
    assert sampled.names == ([] if all_odd else names[1:])
    assert len(sampled.names) == len(sampled.groups)
    assert all(group[5].any() for group in sampled.groups)
    # This entry check/diagnostic is the path that crashed the real 957 run.
    assert unsupported_reason(coarse, poses[indices]) is None
    assert sampled.diagnostics(poses[indices])['accepted']
    composed = coarse.candidate_guards[-1]
    assert composed.full_guard.names == names  # Odd holds are now enforced during coarse solve.
    assert composed.accepts(poses[indices])
    assert guard.names == names and [group[5].tobytes() for group in guard.groups] == originals


def test_surviving_transfer_uses_one_exact_accepts_without_a_window_screen():
    objective, fixed = problem()
    calls = []
    class Accept:
        def accepts(self, poses):
            calls.append(poses.copy())
            return True
        def violation_frames(self, *args, **kwargs):
            raise AssertionError('A surviving candidate needs no repeated screen')
    objective.source_shape, objective.candidate_guards = None, [Accept()]
    coarse = fixed[::2].copy()
    coarse[:, 7] += .001
    warm, report = transfer_correction(objective, fixed, coarse, np.arange(0, len(fixed), 2))
    assert report['accepted'] and len(calls) == 1
    np.testing.assert_array_equal(warm, calls[0])


def test_local_scaling_cannot_pay_for_a_crossing_jerk_stencil():
    objective, fixed = problem()
    fixed[:, 7] = .003 * np.arange(len(fixed))
    guard = CorrectionContinuity(objective.coords, fixed, correction=False)
    objective.source_shape, objective.candidate_guards = None, [guard]
    objective.upper[:] = fixed[:, objective.coords.qi] + .3
    objective.upper[8:12, 0] = fixed[8:12, 7] + .025
    frozen_jerk = guard.bounds['joints_order_3'].copy()
    coarse = fixed[::2].copy()
    coarse[:, 7] += .1
    unsafe = fixed.copy()
    unsafe[:, 7] += .1
    unsafe[8:12, 7] -= .075
    # The scaling boundary changes stencils beginning outside the reduced area.
    assert not guard.accepts(unsafe)
    assert guard.violation_frames(unsafe)[5:8].all()
    warm, report = scale_transfer_correction(objective, fixed, coarse, np.arange(0, len(fixed), 2),
                                              spread_frames=2)
    local = [row for row in report['attempts'] if row['kind'] == 'local']
    assert local and any(any('CorrectionContinuity' in name for name in row['failures']) for row in local)
    assert any(any(frame < 8 or frame >= 12 for frame in frames)
               for row in local for name, frames in row['failures'].items()
               if 'CorrectionContinuity' in name)
    assert report['accepted'] and report['retained_scale'] == pytest.approx(.25)
    np.testing.assert_allclose(report['scales'], .25)
    assert guard.accepts(warm)
    np.testing.assert_array_equal(guard.bounds['joints_order_3'], frozen_jerk)


@pytest.mark.parametrize('count', [35, 36])
def test_time_varying_noncommuting_corrections_cross_pi_without_a_branch_jump(count):
    objective, poses = problem()
    fixed = poses[:count].copy()
    t = np.linspace(0., 1., count)
    reference = Rotation.from_euler('xyz', np.column_stack((.6 * t, .3 * t**2, -.4 * t)))
    fixed[:, 3:7] = reference.as_quat(scalar_first=True)
    indices = np.arange(0, count, 2)
    time = t[indices]
    axes = np.column_stack((np.ones(len(indices)), .18 * np.sin(2 * time), .12 * np.cos(3 * time)))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = np.linspace(np.pi - .2, np.pi + .2, len(indices))
    correction = Rotation.from_rotvec(axes * angles[:, None])
    commutator = (correction[:-1] * correction[1:]) * (correction[1:] * correction[:-1]).inv()
    assert commutator.magnitude().max() > .001
    coarse = fixed[indices].copy()
    coarse[:, 3:7] = (correction * reference[indices]).as_quat(scalar_first=True)
    delta = interpolate_correction(objective.coords, fixed, coarse, indices)
    warm = objective.coords.integrate(fixed, delta, 1.)
    actual = Rotation.from_quat(warm[:, 3:7], scalar_first=True) * reference.inv()
    assert (actual[indices] * correction.inv()).magnitude().max() < 1e-12
    # Consecutive physical rotations remain close even at the pi log cut.
    assert (actual[1:] * actual[:-1].inv()).magnitude().max() < .1
    assert actual.magnitude().min() > np.pi - .25
    assert (actual[-1] * correction[-1].inv()).magnitude() < 1e-12
    fixed[1::2, 3:7] *= -1
    coarse[::2, 3:7] *= -1
    np.testing.assert_allclose(interpolate_correction(objective.coords, fixed, coarse, indices),
                               delta, atol=1e-12)


def test_coarse_and_full_guards_are_prepared_without_pools_and_coarse_is_collected(monkeypatch):
    monkeypatch.setenv('OMNIRETARGETING_FAST_GUARDS', '1')
    monkeypatch.setattr(driver, 'MIN_WINDOW_FRAMES', 30)
    objective, fixed = problem()
    refs = []
    make, transfer = driver.make_coarse_problem, driver.transfer_correction
    def sample(obj):
        coarse, indices = make(obj)
        refs.extend([weakref.ref(coarse), *(weakref.ref(g) for g in coarse.candidate_guards)])
        coarse.cycle = coarse  # Includes guard -> objective cycles from the real problem.
        return coarse, indices
    def pool():
        raise AssertionError('Preparing guards must not start a pool')
    monkeypatch.setattr(objective, 'start_workers', pool)
    def root_anchor(frame):
        raise AssertionError('Uncached per-frame root signal path')
    monkeypatch.setattr(objective.tasks, 'root_anchor', root_anchor)
    def solve(obj, entry, **kwargs):
        seq = next(g for g in (getattr(wrapper, 'full_guard', wrapper) for wrapper in obj.candidate_guards)
                   if isinstance(g, SequenceContinuityGuard))
        assert type(seq) is CachedSequenceContinuityGuard and getattr(obj, 'pool', None) is None
        seq._signals_cache.clear()
        assert seq.accepts(obj.coarse_embedding.embed(entry) if obj is not objective else entry)
        if obj is objective:
            assert all(ref() is None for ref in refs)
        moved = entry.copy()
        if obj is not objective:
            moved[:, 7] += .001
        return moved, []
    def certify(*args, **kwargs):
        assert all(ref() is None for ref in refs)
        return transfer(*args, **kwargs)
    monkeypatch.setattr(driver, 'make_coarse_problem', sample)
    monkeypatch.setattr(driver, 'optimize_legacy_windows', solve)
    monkeypatch.setattr(driver, 'transfer_correction', certify)
    _, _, report = driver.optimize_coarse_acquisition(objective, fixed, iterations=35)
    assert not report['fallback_ran'] and all(ref() is None for ref in refs)


@pytest.mark.parametrize('phase', ['construction', 'entry', 'solve'])
def test_coarse_errors_fall_back_after_partial_state_is_collected(monkeypatch, phase):
    objective, fixed = problem()
    monkeypatch.setattr(driver, 'MIN_WINDOW_FRAMES', 30)
    make, check = driver.make_coarse_problem, driver.unsupported_reason
    refs, calls = [], []
    def sample(obj):
        coarse, indices = make(obj)
        refs.append(weakref.ref(coarse))
        coarse.cycle = coarse
        if phase == 'construction':
            raise RuntimeError('partial construction failed')
        return coarse, indices
    def entry(obj, *args, **kwargs):
        if obj is not objective and phase == 'entry':
            raise ValueError('entry diagnostics failed')
        return check(obj, *args, **kwargs)
    def solve(obj, entry, **kwargs):
        if obj is not objective:
            raise ArithmeticError('coarse solve failed')
        assert all(ref() is None for ref in refs)
        np.testing.assert_array_equal(entry, fixed)
        calls.append(kwargs)
        return entry.copy(), []
    monkeypatch.setattr(driver, 'make_coarse_problem', sample)
    monkeypatch.setattr(driver, 'unsupported_reason', entry)
    monkeypatch.setattr(driver, 'optimize_legacy_windows', solve)
    _, _, report = driver.optimize_coarse_acquisition(objective, fixed, iterations=35)
    assert report['fallback_ran'] and phase in report['fallback_reason']
    assert len(calls) == 1 and calls[0]['iterations'] == 35
    assert 'sweep_iterations' not in calls[0]


def test_grid_summary_matches_existing_depths_without_workers(monkeypatch):
    from omniretargeting._fast.sdf import GridSDF
    objective, poses = problem()
    lower, spacing = np.array([-1., -1., -.2]), .04
    shape = (50, 50, 50)
    heights = lower[2] + spacing * np.arange(shape[2])
    objective.scene_sdf = GridSDF(lower, spacing, np.broadcast_to(heights - .2, shape).copy(),
                                 np.zeros(shape + (3,)))
    expected = objective.penetration_summary(poses)
    def pool():
        raise AssertionError('Grid diagnostics must not start workers')
    monkeypatch.setattr(objective, 'start_workers', pool)
    monkeypatch.setattr(objective, 'deepest_fields', pool)
    summary = driver._grid_summary(objective, poses, chunk_size=11)
    for name, row in expected.items():
        actual = summary['fields'][name]
        assert actual['max_m'] == pytest.approx(row['max_m'])
        assert actual['frames_over_2cm'] == row['frames_over_2cm']
        assert actual['frames_over_1cm'] >= actual['frames_over_2cm']
        assert actual['worst']['body'] and 0 <= actual['worst']['frame'] < len(poses)
