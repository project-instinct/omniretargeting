"""Serial/parallel evidence equality and the shared mechanics pool contract."""

import dataclasses

import numpy as np
import pytest

from omniretargeting.contact_mechanics import ContactCandidate
from omniretargeting.object_support_evidence import _decide_task
from omniretargeting.source_support_evidence import _explain_and_combine
from omniretargeting.support_parallel import SupportPool


def exact(value):
    if isinstance(value, np.ndarray):
        return value.dtype.str, value.shape, value.tobytes()
    if dataclasses.is_dataclass(value):
        return {f.name: exact(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {k: exact(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [exact(v) for v in value]
    return value


def candidate(name, x, normal=(0., 0., 1.), gap=0.):
    points = np.array([[x, 0., 0.]])
    return ContactCandidate(name, points + [0., 0., gap], points, np.array(normal), .6, .01)


def test_shared_pool_keeps_complete_source_and_object_evidence_exact(monkeypatch):
    monkeypatch.setenv("OMNIRETARGETING_CPUS", "2")
    frames = 40
    candidates = tuple((candidate("left", -.2), candidate("right", .2),
                        candidate("gap", 0., gap=.02)) for _ in range(frames))
    regions = {"left": ("l", 0), "right": ("r", 0), "gap": ("g", 0)}
    wrench = np.tile([0., 0., 1., 0., 0., 0.], (frames, 1))
    wrench[10:20, 4] = .5
    wrench[20:30, 2] = -1.
    demand = dict(wrench=wrench, origin=np.zeros(3),
                  derivative_disagreement=np.full((frames, 6), .001), derivative_window=3)
    serial = _explain_and_combine([candidates] * 3, regions, demand, map_function=map)
    jobs = [(wrench[f], np.zeros(3), [], list(candidates[f]), regions) for f in range(frames)]
    expected = list(map(_decide_task, jobs))
    with SupportPool() as pool:
        parallel = _explain_and_combine([candidates] * 3, regions, demand, map_function=pool.map)
        first_pool = pool.pool
        observed = pool.map(_decide_task, jobs)
        assert pool.pool is first_pool and first_pool is not None
        assert first_pool._processes == 2
    assert pool.pool is None
    assert exact(vars(serial)) == exact(vars(parallel))
    assert observed == expected


def test_object_counters_include_carried_infeasible_and_candidate_limit_frames(monkeypatch):
    monkeypatch.setenv("OMNIRETARGETING_CPUS", "2")
    from omniretargeting.object_support_evidence import MAX_CANDIDATES
    up = candidate("up", 0.)
    too_many = [candidate(str(i), 0.) for i in range(MAX_CANDIDATES + 1)]
    jobs = [([0., 0., 1., 0., 0., 0.], np.zeros(3), [up], [], {}),
            ([0., 0., -1., 0., 0., 0.], np.zeros(3), [], [up], {"up": ("hand", 1)}),
            ([0., 0., 1., 0., 0., 0.], np.zeros(3), [], too_many, {})] * 12
    expected = list(map(_decide_task, jobs))
    with SupportPool() as pool:
        assert pool.map(_decide_task, jobs) == expected
    assert expected[0][1]["environment_carried_frames"] == 1
    assert expected[1][1]["unexplained_frames"] == 1
    assert expected[2][1]["skipped_frames"] == 1


def test_worker_cap_lazy_serial_paths_and_exception_cleanup(monkeypatch):
    monkeypatch.setenv("OMNIRETARGETING_CPUS", "100")
    with SupportPool() as pool:
        assert pool.workers == 8
        assert pool.map(abs, [-1, -2]) == [1, 2]
        assert pool.map(abs, []) == []
        assert pool.pool is None
    monkeypatch.setenv("OMNIRETARGETING_CPUS", "1")
    with SupportPool() as pool:
        assert pool.map(abs, range(-40, 0)) == list(range(40, 0, -1))
        assert pool.pool is None
    monkeypatch.setenv("OMNIRETARGETING_CPUS", "2")
    workers = []
    with pytest.raises(ValueError):
        with SupportPool() as pool:
            pool.map(abs, range(40))
            workers = list(pool.pool._pool)
            pool.map(int, ["bad"] * 40)
    assert workers and all(not worker.is_alive() for worker in workers)


def test_no_fork_platform_uses_the_serial_functions(monkeypatch):
    import omniretargeting.support_parallel as parallel

    monkeypatch.setenv("OMNIRETARGETING_CPUS", "8")
    monkeypatch.delattr(parallel.os, "fork", raising=False)
    with SupportPool() as pool:
        assert pool.map(abs, range(-40, 0)) == list(range(40, 0, -1))
        assert pool.pool is None
