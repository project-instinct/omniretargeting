"""Restoration is a fallback; nonlinear and geometry rejection stay independent."""
from types import SimpleNamespace
import numpy as np
from omniretargeting._fast.repair_candidates import candidate_steps


def limits(bound):
    return SimpleNamespace(coords=SimpleNamespace(qi=np.array([1])),
                           check=lambda q: (bool(q[0, 1] <= bound), {'value': float(q[0, 1])}))


def test_backtracked_valid_step_avoids_restoration_and_keeps_root():
    def expensive(q):
        raise AssertionError('unnecessary restoration')
    current = np.array([[2., 0.]])
    trials = candidate_steps(limits(.6), current, np.ones((1, 1)), expensive, {}, lazy=True,
                             fixed_q=np.array([True, False]))
    alpha, q = next(trials)
    assert alpha == .5 and q[0, 0] == current[0, 0] and q[0, 1] == .5
    np.testing.assert_array_equal(current, [[2., 0.]])


def test_geometry_rejections_exhaust_raw_steps_before_restoring():
    calls = []
    def restore(q):
        calls.append(q.copy())
        return np.array([[0., .007]]), {'status': 'restored'}
    record = {}
    trials = candidate_steps(limits(.6), np.zeros((1, 2)), np.ones((1, 1)), restore, record, lazy=True)
    # Simulate the independent geometry check rejecting all raw candidates.
    for alpha, q in trials:
        if q[0, 1] == .007:
            assert len(calls) == 1 and alpha == 1.
            break
        assert not calls
    else:
        raise AssertionError('restoration fallback was lost')
    assert record['restoration']['status'] == 'restored'


def test_restored_candidate_still_must_pass_nonlinear_guard():
    restore = lambda q: (q, {'status': 'unresolved'})
    record = {}
    trials = list(candidate_steps(limits(-1), np.zeros((1, 2)), np.ones((1, 1)), restore, record, lazy=True))
    assert trials == [] and len(record['rejected_limits']) == 8


def test_legacy_order_restores_first_and_does_not_repeat_valid_geometry_failure():
    calls = []
    def restore(q):
        calls.append(1)
        return np.array([[0., .55]]), {}
    args = (limits(.6), np.zeros((1, 2)), np.ones((1, 1)), restore, {})
    alpha, q = next(candidate_steps(*args))
    assert alpha == 1 and q[0, 1] == .55 and calls == [1]
    calls.clear()
    assert len(list(candidate_steps(limits(2), args[1], args[2], restore, {}, lazy=True))) == 7
    assert not calls


def test_shallow_correction_restores_before_repeating_tiny_steps():
    calls = []
    def restore(q):
        calls.append(1)
        return np.array([[0., .19]]), {'status': 'restored'}
    args = (limits(.2), np.zeros((1, 2)), np.ones((1, 1)), restore, {})
    alpha, q = next(candidate_steps(*args, lazy=True, restore_below=.5))
    assert alpha == 1. and q[0,1] == .19 and calls == [1]
    calls.clear()
    alpha, q = next(candidate_steps(limits(.6), args[1], args[2], restore, {},
                                     lazy=True, restore_below=.5))
    assert alpha == .5 and not calls
