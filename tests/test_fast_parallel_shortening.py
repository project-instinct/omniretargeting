"""Parallel fraction restoration preserves results, order and consumed counters."""
from dataclasses import replace
from types import SimpleNamespace
import multiprocessing
import os
import numpy as np
import pytest

from omniretargeting._fast.shortening import FractionRestorations
from omniretargeting._fast import runtime_options, shortening

pytestmark = pytest.mark.skipif(not hasattr(os, 'fork'), reason='fork speculation')


def _restore(objective, origin, step):
    objective.guard.rejections += 3
    if np.max(step) < .3:
        raise ValueError('second fraction')
    return origin + step


def _context(enabled):
    guard = SimpleNamespace(rejections=7)
    objective = SimpleNamespace(guard=guard)
    pending = FractionRestorations(_restore, objective, np.zeros((2, 3)), np.ones((2, 3)),
                                  1., (.5, .25, .125), 2, [guard], enabled=enabled)
    return objective, pending


@pytest.mark.parametrize('enabled', [False, True])
def test_unused_errors_and_rejections_do_not_change_consumed_result(enabled):
    objective, pending = _context(enabled)
    children = []
    with pending:
        children = [job.process for job in pending.jobs.values()]
        result = pending.restore(.5)
    np.testing.assert_array_equal(result, np.full((2, 3), .5))
    assert objective.guard.rejections == 10
    assert not any(p.is_alive() for p in children)
    if enabled:
        assert objective.parallel_shortening_stats['consumed'] == 1
        assert objective.parallel_shortening_stats['discarded'] == 1


@pytest.mark.parametrize('enabled', [False, True])
def test_consumed_error_propagates_and_closes_other_children(enabled):
    objective, pending = _context(enabled)
    children = []
    with pytest.raises(ValueError, match='second fraction'):
        with pending:
            children = [job.process for job in pending.jobs.values()]
            pending.restore(.25)
    assert objective.guard.rejections == 10
    assert not any(p.is_alive() for p in children)


def test_unprefetched_fraction_uses_serial_function_without_extra_children():
    objective, pending = _context(True)
    with pending:
        assert len(pending.jobs) == 2
        np.testing.assert_array_equal(pending.restore(.75), np.full((2, 3), .75))
    assert objective.parallel_shortening_stats['serial'] == 1
    assert objective.parallel_shortening_stats['consumed'] == 0
    assert objective.guard.rejections == 10


def test_full_conic_shortening_matches_serial_numerical_path(monkeypatch):
    from tests.test_fast_proposals import _solve
    calls = []
    original = shortening.FractionRestorations.__enter__
    def enter(self):
        calls.append(self.enabled)
        return original(self)
    monkeypatch.setattr(shortening.FractionRestorations, '__enter__', enter)
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS, parallel_shortening=False))
    serial = _solve(monkeypatch, False)
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS, parallel_shortening=True))
    parallel = _solve(monkeypatch, False)
    assert any(calls), 'the integration case must exercise speculative restorations'
    assert serial[0].tobytes() == parallel[0].tobytes()
    assert serial[1] == parallel[1]
    assert serial[2] == parallel[2]
