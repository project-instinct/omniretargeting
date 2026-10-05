"""Fork-shared borrowing never alters self rows or survives the next fork."""
from contextlib import ExitStack
import ctypes
import multiprocessing
import os
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from omniretargeting._fast.window_cpu_budget import WindowCpuBudget, native_workers
from tests.test_batched_self_collision import _batch, _poses
from tests.test_convex_mesh_distance import mesh_pair
from omniretargeting._fast.kinematics import TrajectoryKinematics
from omniretargeting.self_collision import SelfCollisionQuery
from omniretargeting.geometry_query import GeometryQueryError


def test_validated_default_and_explicit_native_thread_limits(monkeypatch):
    monkeypatch.delenv('OMNIRETARGETING_WINDOW_NATIVE_THREADS', raising=False)
    assert native_workers() == 4
    for value, expected in [('1', 1), ('2', 2), ('4', 4), ('0', 1), ('16', 4)]:
        monkeypatch.setenv('OMNIRETARGETING_WINDOW_NATIVE_THREADS', value)
        assert native_workers() == expected


@pytest.mark.parametrize('windows,expected', [(1, 4), (2, 4), (3, 2), (4, 1), (8, 1)])
def test_busy_windows_keep_one_unit_and_tail_borrows_only_spares(windows, expected):
    budget = WindowCpuBudget(8)
    with ExitStack() as leases:
        for _ in range(windows):
            leases.enter_context(budget.window())
        with budget.borrow(4) as actual:
            assert actual == expected
            held = []
            while budget.permits.acquire(block=False):
                held.append(1)
            assert len(held) == 8-windows-(actual-1)
            for _ in held:
                budget.permits.release()
    assert budget.windows.value == 0
    for _ in range(8):
        assert budget.permits.acquire(block=False)
    assert not budget.permits.acquire(block=False)


def test_fraction_tokens_share_budget_and_query_errors_release_helpers():
    budget = WindowCpuBudget(4)
    with budget.window(), budget.permits, budget.permits:
        with pytest.raises(RuntimeError, match='query failed'):
            with budget.borrow(4) as count:
                assert count == 2
                raise RuntimeError('query failed')
        assert budget.permits.acquire(block=False)
        assert not budget.permits.acquire(block=False)
        budget.permits.release()
    assert budget.report()['native_calls_by_threads']['2'] == 1
    for _ in range(4):
        assert budget.permits.acquire(block=False)


def signature(rows, failures, unknown):
    return (failures, unknown, [[[(key, np.asarray(value).dtype.str, np.asarray(value).shape,
                                np.asarray(value).tobytes()) for key, value in row.items()]
                              for row in frame] for frame in rows])


@pytest.mark.parametrize('workers', [1, 2, 4])
def test_thread_counts_preserve_ties_unknowns_witness_and_jacobian_bytes(workers):
    with threadpool_limits(limits=1):
        model = mesh_pair()
        poses = _poses(model)
        kin = TrajectoryKinematics(model, np.arange(model.nv)).evaluate(poses)
        query = SelfCollisionQuery(model)
        serial, candidate = _batch(query.fresh_state(), 1), _batch(query.fresh_state(), workers)
        expected = serial.rows(poses, kin, True)
        actual = candidate.rows(poses, kin, True)
        assert signature(*actual, candidate.unknown_frames) == signature(*expected, serial.unknown_frames)
        assert candidate.diagnostics['native_workers'] == workers
        assert threading.active_count() == 1


def _daemon_query(query, poses, kin, budget, sender):
    with budget.window():
        candidate = _batch(query, 4)
        candidate.cpu_budget = budget
        rows, failures = candidate.rows(poses, kin, True)
        sender.send((signature(rows, failures, candidate.unknown_frames),
                     candidate.diagnostics['native_workers'], threading.active_count()))
    sender.close()


def test_daemon_can_borrow_native_threads_and_all_threads_join_before_return():
    with threadpool_limits(limits=1):
        model = mesh_pair()
        poses = _poses(model)
        kin = TrajectoryKinematics(model, np.arange(model.nv)).evaluate(poses)
        query = SelfCollisionQuery(model)
        serial = _batch(query.fresh_state(), 1)
        expected = serial.rows(poses, kin, True)
        budget = WindowCpuBudget(4)
        context = multiprocessing.get_context('fork')
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=_daemon_query, daemon=True,
                                  args=(query.fresh_state(), poses, kin, budget, sender))
        process.start()
        sender.close()
        actual, threads, active = receiver.recv()
        receiver.close()
        process.join()
        assert process.exitcode == 0
        assert actual == signature(*expected, serial.unknown_frames)
        assert threads == 4 and active == 1
        for _ in range(4):
            assert budget.permits.acquire(block=False)


@pytest.mark.parametrize('workers', [1, 2, 4])
def test_parallel_native_fallback_preserves_unknown_frame_bytes(monkeypatch, workers):
    with threadpool_limits(limits=1):
        model = mesh_pair()
        poses = _poses(model)
        kin = TrajectoryKinematics(model, np.arange(model.nv)).evaluate(poses)
        batch = _batch(SelfCollisionQuery(model), workers)
        original = batch.library.ws3_pairs
        def indeterminate(*args):
            original(*args)
            status = np.ctypeslib.as_array(ctypes.cast(args[15], ctypes.POINTER(ctypes.c_uint8)),
                                          shape=(args[0]*args[2],))
            status[:] = 2
        batch.library = SimpleNamespace(ws3_pairs=indeterminate)
        def failed(*args):
            raise GeometryQueryError('fixture indeterminate')
        monkeypatch.setattr(batch.query.mesh_distance, 'query', failed)
        rows, unknown = batch.rows(poses, kin, True)
        assert rows == [[] for _ in poses]
        assert unknown == len(poses) and batch.unknown_frames == list(range(len(poses)))
        assert batch.diagnostics['native_workers'] == workers
        assert threading.active_count() == 1


def _fraction_borrow(budget, sender):
    with budget.borrow(4) as count:
        sender.send(count)
    sender.close()


def test_disposable_fraction_child_cannot_inherit_its_parents_helper_leases():
    budget = WindowCpuBudget(4)
    with budget.window(), budget.permits:
        context = multiprocessing.get_context('fork')
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=_fraction_borrow, args=(budget, sender), daemon=True)
        process.start()
        sender.close()
        assert receiver.recv() == 1
        receiver.close()
        process.join()
        assert process.exitcode == 0
        assert budget.permits.acquire(block=False) and budget.permits.acquire(block=False)
        assert not budget.permits.acquire(block=False)
        budget.permits.release()
        budget.permits.release()


def test_disposable_child_bypass_does_not_even_touch_inherited_shared_locks():
    budget = WindowCpuBudget(4)
    budget.window_owner_pid = os.getpid()+1
    class Forbidden:
        def get_lock(self):
            raise AssertionError('Disposable child accessed an inherited lock')
    budget.windows = budget.calls = Forbidden()
    budget.calls_by_kind = {'self': Forbidden()}
    with budget.borrow(4) as count:
        assert count == 1
