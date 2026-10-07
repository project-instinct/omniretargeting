"""Window adapter parity across global frame overlaps, thresholds and forks."""

import os

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from omniretargeting._fast.batched_self_collision import BatchedSelfCollision
from omniretargeting._fast.kinematics import TrajectoryKinematics
from omniretargeting._fast.self_collision_window import SelfCollisionWindow
from omniretargeting._fast.window_self_collision import WindowSelfCollision
from omniretargeting.geometry_query import GeometryQueryError
from omniretargeting.self_collision import SelfCollisionQuery
from tests.test_batched_self_collision import _poses
from tests.test_convex_mesh_distance import mesh_pair


@pytest.fixture(autouse=True)
def serial_blas():
    with threadpool_limits(limits=1):
        yield


def equal(left, right):
    if isinstance(left, np.ndarray):
        assert left.shape == right.shape and left.dtype == right.dtype
        assert left.tobytes() == right.tobytes()
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    else:
        assert left == right


@pytest.mark.parametrize('target', [None, 0., .004, .012])
def test_adapter_retains_rows_jacobians_and_failures_across_overlapping_frames(target):
    model = mesh_pair()
    query = SelfCollisionQuery(model)
    poses = _poses(model)
    provider = SelfCollisionWindow(query).prepare()
    adapter = WindowSelfCollision(provider, query)
    kin = TrajectoryKinematics(model, np.arange(model.nv))
    for frames, jacobian in [(range(0, 24), False), (range(0, 24), True),
                             (range(12, 36), True), (range(4, 32), False)]:
        current = kin.evaluate(poses[frames.start:frames.stop], jacobian)
        adapter.bind(poses, frames)
        actual = adapter.rows(poses[frames.start:frames.stop], current, jacobian, target)
        expected = BatchedSelfCollision(query, workers=1).rows(
            poses[frames.start:frames.stop], current, jacobian, target)
        equal(actual, expected)
    assert adapter.diagnostics['pose_reused_frames'] == len(frames)
    query.cfg['detection_distance'] = .2
    equal(adapter.rows(poses[frames.start:frames.stop], current, False, target),
          BatchedSelfCollision(query, workers=1).rows(
              poses[frames.start:frames.stop], current, False, target))
    assert adapter.fallback is not None


def test_unknown_frames_are_requeried_and_never_cached(monkeypatch):
    model = mesh_pair()
    query = SelfCollisionQuery(model)
    provider = SelfCollisionWindow(query).prepare()
    adapter = WindowSelfCollision(provider, query)
    poses = _poses(model)[:4]
    kin = TrajectoryKinematics(model, np.arange(model.nv)).evaluate(poses)

    def fallback(position, rotation, near, clearance, points, directions, status):
        status[:] = 2

    def unknown(*args):
        raise GeometryQueryError('fixture uncertain')

    monkeypatch.setattr(provider, '_run_native', fallback)
    monkeypatch.setattr(provider.query.mesh_distance, 'query', unknown)
    adapter.bind(poses, range(4))
    for _ in range(2):
        rows, failures = adapter.rows(poses, kin, True, .004)
        assert failures == 4 and not any(rows)
        assert not provider.pose_cache




@pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires fork')
def test_worker_private_frame_cache_matches_original_serial_solver(monkeypatch):
    from omniretargeting._fast import runtime_options, window_self_collision
    from omniretargeting._fast.legacy_window import WindowWorkers
    from omniretargeting._fast.legacy_window_qp import solve_local
    from omniretargeting._fast.proposals import guards_of
    from omniretargeting._fast.runtime_options import RuntimeOptions
    from omniretargeting._fast.window_schedule import dependency_window
    from tests.test_fast_legacy_window import problem

    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions())
    monkeypatch.setattr(window_self_collision, 'WINDOW_SELF_FRAME_CACHE', False)
    obj, poses = problem(extra=True)
    obj.self_collision = SelfCollisionQuery(obj.model)
    obj.prepare_window_evaluation()
    window = dependency_window(range(8, 16), len(poses), [obj, *guards_of(obj)[1]])
    expected = solve_local(obj, poses, window.frames, 1, .003)
    monkeypatch.setattr(window_self_collision, 'WINDOW_SELF_FRAME_CACHE', True)
    obj.prepare_window_evaluation()
    workers = WindowWorkers(obj, 1)
    try:
        for _ in range(2):
            actual = workers.solve(poses, [window], 1, .003)[0]
            equal(actual['poses'], expected['poses'])
            assert actual['commit']['accepted'] == expected['commit']['accepted']
            assert actual['stop_reason'] == expected['stop_reason']
    finally:
        workers.close()
