"""A speculative child must never wait on its parent's inherited pool."""
import multiprocessing
import os
from types import SimpleNamespace

import numpy as np
import pytest

from omniretargeting._fast.objective import _owned_pool
from tests.test_fast_backend import _objective


class ParentPool:
    def __init__(self, owner):
        self._pool = [SimpleNamespace(_parent_pid=owner)]
    def map(self, *args, **kwargs):
        raise AssertionError('inherited pool was used')
    close = join = map


@pytest.mark.parametrize('jacobian', [False, True])
def test_inherited_pool_evaluation_keeps_reference_values(jacobian):
    objective, poses, _ = _objective(temporal=True)
    expected = objective.evaluate(poses, jacobian)
    objective.pool = ParentPool(os.getpid()-1)
    actual = objective.evaluate(poses, jacobian)
    assert actual[0].cost == expected[0].cost
    for field in ('frame_cost', 'hessian', 'gradient') if jacobian else ('frame_cost',):
        assert getattr(actual[0],field).tobytes() == getattr(expected[0],field).tobytes()
    assert actual[1].tobytes() == expected[1].tobytes()
    if jacobian:
        for field in ('data','indices','indptr'):
            assert getattr(actual[2],field).tobytes() == getattr(expected[2],field).tobytes()
    assert actual[3] == expected[3]


def test_inherited_pool_deepest_fields_and_shutdown_are_local():
    objective, poses, _ = _objective(temporal=True)
    expected = objective.deepest_fields(poses)
    objective.pool = ParentPool(os.getpid()-1)
    actual = objective.deepest_fields(poses)
    for old, new in zip(expected, actual):
        for before, after in zip(old, new):
            assert np.asarray(before).tobytes() == np.asarray(after).tobytes()
    objective.stop_workers()
    assert objective.pool is None


def test_parent_owner_and_explicit_pid_marker():
    objective = SimpleNamespace(pool=ParentPool(os.getpid()))
    assert _owned_pool(objective) is objective.pool
    objective._pool_pid = os.getpid()-1
    assert _owned_pool(objective) is None


def test_daemonic_child_drops_pool_without_starting_more_children(monkeypatch):
    from omniretargeting._fast import objective as module
    objective, _, _ = _objective(temporal=True)
    objective.pool = ParentPool(os.getpid()-1)
    monkeypatch.setattr(module, 'MIN_CHUNK_FRAMES', 1)
    monkeypatch.setattr(multiprocessing, 'current_process', lambda:SimpleNamespace(daemon=True))
    assert objective.start_workers()
    assert objective.pool is None
    objective.stop_workers()


def _child_evaluate(objective, poses, pipe):
    try:
        value = objective.evaluate(poses)
        pipe.send(('ok', value[0].cost, value[0].gradient.tobytes(), value[3]))
    except BaseException as error:
        pipe.send(('error', repr(error)))
    finally:
        pipe.close()


def test_actual_fork_inherits_no_usable_parent_pool():
    objective, poses, _ = _objective(temporal=True)
    expected = objective.evaluate(poses)
    objective.pool = ParentPool(os.getpid())
    context = multiprocessing.get_context('fork')
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_child_evaluate, args=(objective,poses,child),daemon=True)
    try:
        process.start()
        child.close()
        assert parent.poll(10), 'child blocked on the inherited pool'
        actual = parent.recv()
        assert actual == ('ok',expected[0].cost,expected[0].gradient.tobytes(),expected[3])
        process.join(10)
        assert not process.is_alive()
    finally:
        if process.is_alive():
            process.terminate()
        process.join()
        parent.close()


def test_temporary_deepest_pool_matches_serial(monkeypatch):
    import omniretargeting._fast.objective as module
    objective, poses, _ = _objective(temporal=True)
    monkeypatch.setattr(module, 'TEMPORARY_DEEPEST_FRAMES', 10**9)
    expected = objective.deepest_fields(poses)
    monkeypatch.setattr(module, 'TEMPORARY_DEEPEST_FRAMES', 1)
    monkeypatch.setattr(module, 'MIN_CHUNK_FRAMES', 1)
    registry = dict(module._WORKER)
    actual = objective.deepest_fields(poses)
    assert dict(module._WORKER) == registry
    assert _owned_pool(objective) is None
    for old, new in zip(expected, actual, strict=True):
        for before, after in zip(old, new, strict=True):
            assert np.asarray(before).tobytes() == np.asarray(after).tobytes()
