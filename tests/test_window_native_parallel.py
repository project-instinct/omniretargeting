"""Threaded shell frames keep first minima, witnesses and budget ownership."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting._fast.bulk_geometry import BulkGeometry
from omniretargeting._fast.window_cpu_budget import WindowCpuBudget
from omniretargeting._fast.window_query_native import WindowQueries
from omniretargeting._fast import window_native_parallel
from tests.test_bulk_geometry import _shell_problem


@pytest.mark.parametrize('workers', [1, 2, 3, 4])
@pytest.mark.parametrize('scale', [1., -.5, np.float32(1.07)])
def test_shell_frames_keep_every_output_byte_including_moving_fields_and_unknowns(monkeypatch, workers, scale):
    monkeypatch.setattr(window_native_parallel, 'native_workers', lambda: workers)
    obj, poses, sdf = _shell_problem(unknown=True)
    reference, accelerated = BulkGeometry(obj), BulkGeometry(obj)
    reference.window_queries, accelerated.window_queries = WindowQueries(obj), WindowQueries(obj)
    rng = np.random.default_rng(14)
    rotation = Rotation.from_rotvec(rng.normal(scale=.1, size=(obj.frames, 3))).as_matrix()
    translation = rng.normal(scale=.01, size=(obj.frames, 3))
    scales = np.full(obj.frames, scale, dtype=np.asarray(scale).dtype)
    budget = WindowCpuBudget(4)
    for delta in (0., .001, .02, -.003):
        moved = poses.copy()
        moved[:, 7] += delta
        obj.kin.evaluate(moved, False)
        obj._window_cpu_budget = None
        expected = reference.deepest(sdf, rotation, translation, scales)
        obj._window_cpu_budget = budget
        with budget.window():
            actual = accelerated.deepest(sdf, rotation, translation, scales)
        for a, b in zip(actual, expected):
            assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        assert 'shell' not in accelerated.window_queries.__dict__
        assert threading.active_count() == 1
    for _ in range(4):
        assert budget.permits.acquire(block=False)


@pytest.mark.parametrize('value', [0., np.nan, np.inf, -np.inf])
def test_parallel_seed_at_later_tie_still_returns_first_sample(monkeypatch, value):
    monkeypatch.setattr(window_native_parallel, 'native_workers', lambda: 4)
    obj, poses, sdf = _shell_problem()
    sdf.clearance.fill(value)
    reference, accelerated = BulkGeometry(obj), BulkGeometry(obj)
    reference.window_queries, accelerated.window_queries = WindowQueries(obj), WindowQueries(obj)
    for geom in range(len(obj.shell_geoms)):
        accelerated.window_queries.seeds[sdf, geom] = np.full(obj.frames, 3, np.int64)
    expected = reference.deepest(sdf, None, None)
    obj._window_cpu_budget = WindowCpuBudget(4)
    with obj._window_cpu_budget.window():
        actual = accelerated.deepest(sdf, None, None)
    for a, b in zip(actual, expected):
        assert a.tobytes() == b.tobytes()


def test_native_exception_joins_threads_restores_method_and_returns_all_permits(monkeypatch):
    monkeypatch.setattr(window_native_parallel, 'native_workers', lambda: 4)
    obj, poses, sdf = _shell_problem()
    bulk = BulkGeometry(obj)
    bulk.window_queries = WindowQueries(obj)
    budget = obj._window_cpu_budget = WindowCpuBudget(4)
    def failed(*args):
        raise RuntimeError('native failed')
    bulk.window_queries.library = SimpleNamespace(window_shell=failed)
    with pytest.raises(RuntimeError, match='native failed'):
        with budget.window():
            bulk.deepest(sdf, None, None)
    assert 'shell' not in bulk.window_queries.__dict__
    assert threading.active_count() == 1
    for _ in range(4):
        assert budget.permits.acquire(block=False)


def test_unknown_custom_shell_keeps_its_original_hook(monkeypatch):
    monkeypatch.setattr(window_native_parallel, 'native_workers', lambda: 4)
    obj, poses, sdf = _shell_problem()
    bulk = BulkGeometry(obj)
    bulk.window_queries = WindowQueries(obj)
    shell = bulk.window_queries.shell
    calls = []
    def custom(*args):
        calls.append(1)
        return shell(*args)
    bulk.window_queries.shell = custom
    budget = obj._window_cpu_budget = WindowCpuBudget(4)
    with budget.window():
        bulk.deepest(sdf, None, None)
    assert calls and bulk.window_queries.shell is custom
    assert not sum(budget.report()['native_calls_by_threads'].values())
