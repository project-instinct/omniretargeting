"""Scheduling changes must preserve consumed proposals and guard accounting."""
import gc
import multiprocessing
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from omniretargeting._fast.attempt_schedule import LazyAttempts, attempts_type
from omniretargeting._fast.proposals import Attempts


class Guard:
    rejections = 0


def conic(objective, poses, system, gradient, diagonal, damping, guards):
    for guard in guards:
        guard.rejections += int(damping)
    if damping == 2 and getattr(objective, 'fail', False):
        raise ValueError('consumed failure')
    if damping == 3 and getattr(objective, 'blocked', None) is not None:
        objective.blocked.wait(5)
    return np.array([damping]), np.array([[damping]]), .25


def make(cls, *, fail=False, blocked=None):
    objective = SimpleNamespace(candidate_guards=[Guard()], source_shape=Guard(), fail=fail,
                                blocked=blocked)
    return cls(objective, None, None, None, None, None, None, None, [1, 2, 3], None, conic)


@pytest.mark.skipif(not hasattr(multiprocessing, 'get_context'), reason='fork unavailable')
def test_consumed_proposals_and_counters_equal_eager_reference():
    reference, fast = make(Attempts), make(LazyAttempts)
    try:
        for value in (reference, fast):
            value._fork('conic', [0, 1, 2], value._conic)
        for attempt in (0, 1):
            old, new = reference.conic(attempt), fast.conic(attempt)
            assert old[0].tobytes() == new[0].tobytes()
            assert old[1].tobytes() == new[1].tobytes()
            assert old[2] == new[2]
        fast.close()
        assert reference.objective.source_shape.rejections == fast.objective.source_shape.rejections == 3
        assert reference.objective.candidate_guards[0].rejections == fast.objective.candidate_guards[0].rejections == 3
    finally:
        fast.close()


def test_waits_only_for_consumed_child_and_cancels_unused_work():
    blocked = multiprocessing.get_context('fork').Event()
    fast = make(LazyAttempts, blocked=blocked)
    try:
        fast._fork('conic', [0, 2], fast._conic)
        slow = fast.pending['conic', 2][0]
        assert fast.conic(0)[2] == .25
        assert slow.is_alive()
        fast.close()
        assert not slow.is_alive()
        assert not blocked.is_set()
        assert fast.objective.source_shape.rejections == 1
    finally:
        fast.close()


def test_unused_exception_does_not_change_consumed_result():
    fast = make(LazyAttempts, fail=True)
    try:
        fast._fork('conic', [0, 1], fast._conic)
        assert fast.conic(0)[2] == .25
        assert fast.objective.source_shape.rejections == 1
    finally:
        fast.close()


def test_consumed_exception_has_reference_counter_semantics():
    for cls in (Attempts, LazyAttempts):
        value = make(cls, fail=True)
        try:
            value._fork('conic', [0, 1], value._conic)
            value.conic(0)
            with pytest.raises(ValueError, match='consumed failure'):
                value.conic(1)
            assert value.objective.source_shape.rejections == 1
        finally:
            if isinstance(value, LazyAttempts):
                value.close()


def test_iteration_release_cleans_up_children():
    blocked = multiprocessing.get_context('fork').Event()
    fast = make(LazyAttempts, blocked=blocked)
    fast._fork('conic', [2], fast._conic)
    process = fast.pending['conic', 2][0]
    reference = weakref.ref(fast)
    del fast
    gc.collect()
    assert reference() is None
    assert not process.is_alive()


def test_switch_retains_reference(monkeypatch):
    from omniretargeting._fast import runtime_options
    monkeypatch.setattr(runtime_options, 'OPTIONS', SimpleNamespace(lazy_attempts=False))
    assert attempts_type() is Attempts
    monkeypatch.setattr(runtime_options, 'OPTIONS', SimpleNamespace(lazy_attempts=True))
    assert attempts_type() is LazyAttempts


@pytest.mark.parametrize('diverse', [False, True])
def test_complete_lm_history_with_real_rejected_attempts(monkeypatch, diverse):
    from dataclasses import replace
    from omniretargeting._fast import runtime_options, solver
    from tests.test_fast_proposals import _solve

    monkeypatch.setattr(runtime_options, 'OPTIONS',
                        replace(runtime_options.OPTIONS, conic_trust_damping=diverse))
    reference = _solve(monkeypatch, True)
    monkeypatch.setattr(solver, 'Attempts', LazyAttempts)
    fast = _solve(monkeypatch, True)
    assert reference[3] and fast[3], 'must exercise rejected first attempts'
    assert fast[0].tobytes() == reference[0].tobytes()
    assert fast[1] == reference[1]
    assert fast[2] == reference[2]
