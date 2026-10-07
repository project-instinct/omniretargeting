"""Capacity-bound speculation keeps bytes, errors and consumed guard counters."""
import multiprocessing
import os
from types import SimpleNamespace

import numpy as np
import pytest

from omniretargeting._fast import window_restoration_parallel as parallel
from omniretargeting._fast.shortening import FractionRestorations
from tests.test_fast_parallel_shortening import _restore

pytestmark = pytest.mark.skipif(not hasattr(os, 'fork'), reason='fork unavailable')


def context(capacity):
    permits = multiprocessing.get_context('fork').BoundedSemaphore(capacity)
    guard = SimpleNamespace(rejections=7)
    objective = SimpleNamespace(guard=guard, _window_restoration_permits=permits)
    pending = parallel.WindowFractionRestorations(_restore, objective, np.zeros((2, 3)),
        np.ones((2, 3)), 1., (.5, .25, .125), 2, [guard], enabled=False)
    return objective, permits, pending


@pytest.mark.parametrize('capacity', [0, 1, 2])
def test_capacity_exhaustion_and_discard_preserve_serial_result(capacity):
    objective, permits, pending = context(capacity)
    with pending:
        children = [job.process for job in pending.jobs.values()]
        assert len(children) == capacity
        assert not permits.acquire(block=False)
        actual = pending.restore(.5)
    assert actual.tobytes() == np.full((2, 3), .5).tobytes()
    assert objective.guard.rejections == 10
    assert not any(child.is_alive() for child in children)
    for _ in range(capacity):
        assert permits.acquire(block=False)
    assert not permits.acquire(block=False)


def test_consumed_error_releases_all_permits_and_discards_peer():
    objective, permits, pending = context(2)
    with pytest.raises(ValueError, match='second fraction'):
        with pending:
            children = [job.process for job in pending.jobs.values()]
            pending.restore(.25)
    assert objective.guard.rejections == 10
    assert not any(child.is_alive() for child in children)
    assert permits.acquire(block=False) and permits.acquire(block=False)
    assert not permits.acquire(block=False)


def test_setup_error_releases_its_reserved_capacity(monkeypatch):
    _, permits, pending = context(1)
    def broken(*args):
        raise RuntimeError('cannot fork')
    monkeypatch.setattr(parallel, 'GuardedCall', broken)
    with pytest.raises(RuntimeError, match='cannot fork'):
        with pending:
            pass
    assert permits.acquire(block=False)
    assert not permits.acquire(block=False)


def test_default_and_nonwindow_callers_keep_existing_fraction_class(monkeypatch):
    assert parallel.WINDOW_PARALLEL_RESTORATIONS
    assert parallel.fraction_class(SimpleNamespace(), FractionRestorations) is FractionRestorations
    objective, _, _ = context(1)
    monkeypatch.setattr(parallel, 'WINDOW_PARALLEL_RESTORATIONS', False)
    assert parallel.fraction_class(objective, FractionRestorations) is FractionRestorations
    monkeypatch.setattr(parallel, 'WINDOW_PARALLEL_RESTORATIONS', True)
    assert parallel.fraction_class(SimpleNamespace(), FractionRestorations) is FractionRestorations




def test_window_pool_preserves_bytes_and_releases_ownership(monkeypatch):
    from threadpoolctl import threadpool_limits
    from omniretargeting._fast.legacy_window import _workers
    from omniretargeting._fast.legacy_window_qp import solve_local
    from omniretargeting._fast.solver import CONVERGED_DECREASE
    from tests.test_fast_legacy_window import problem
    obj, poses = problem()
    frames = range(9, 18)
    obj.prepare_window_evaluation()
    with threadpool_limits(limits=1):
        expected = solve_local(obj, poses, frames, 2, CONVERGED_DECREASE)
        monkeypatch.setattr(parallel, 'WINDOW_PARALLEL_RESTORATIONS', True)
        with _workers(obj, 2) as pool:
            assert all(not process.daemon for process, _ in pool.jobs)
            permits = obj._window_restoration_permits
            actual, = pool.solve(poses, [SimpleNamespace(frames=frames)], 2, CONVERGED_DECREASE)
            assert permits.acquire(block=False) and permits.acquire(block=False)
            assert not permits.acquire(block=False)
            permits.release()
            permits.release()
    assert not hasattr(obj, '_window_restoration_permits')
    assert actual['poses'].tobytes() == expected['poses'].tobytes()
    def clean(rows):
        return [{k: v for k, v in row.items() if k != 'seconds'} for row in rows]
    assert clean(actual['history']) == clean(expected['history'])
    assert actual['commit']['accepted'] == expected['commit']['accepted']
