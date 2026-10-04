"""Early rejection saves queries without turning partial frames into evidence."""

from copy import deepcopy

import numpy as np
import pytest

from omniretargeting._fast.mesh_repair import _regressed
from omniretargeting._fast.original_geometry_window import remember_frames
from tests.test_original_geometry_window import _geometry
from tests.test_original_geometry_batches import _assert_same


@pytest.fixture(autouse=True)
def cpu_queries(monkeypatch):
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "0")


def _rejected(measured, protected):
    return (not measured["complete"] or bool(measured["failures"])
            or any(_regressed(depth, protected.get(key, 0.))
                   for key, depth in measured["depths"].items()))


def test_cached_regression_precedes_any_missing_gpu_preparation(monkeypatch):
    geometry, poses = _geometry()
    full = geometry.measure(poses)
    key = next(k for k, d in full["depths"].items() if d > .0001 and k[0] > 0)
    protected = dict(full["depths"])
    protected[key] = 0.
    trial = poses.copy()
    trial[0, 0] += .001  # A miss before the rejecting cached frame.

    def fail(*args, **kwargs):
        pytest.fail("Known cached rejection must precede all GPU preparation")
    monkeypatch.setattr(geometry, "_prepare_gpu_static", fail)
    monkeypatch.setattr(geometry, "_prepare_gpu_moving", fail)
    measured = geometry.measure(trial, frames=range(len(trial)), protected_depths=protected)
    assert _rejected(measured, protected)
    assert not measured["complete"] and not measured["frame_complete"][0]
    assert measured["query_scope"]["queried_frame_count"] == 0


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("shift", [0., .0001, .02])
def test_cached_and_full_queries_make_identical_decisions(parallel, shift, monkeypatch):
    from omniretargeting._fast import parallel_geometry
    geometry, poses = _geometry()
    if parallel:
        monkeypatch.setattr(parallel_geometry, "WORKERS", 3)
        monkeypatch.setattr(parallel_geometry, "MIN_FRAMES_PER_WORKER", 2)
        geometry.start_workers(len(poses))
    try:
        before = geometry.measure(poses)
        trial = poses.copy()
        trial[3:5, 7] += shift
        protected = before["depths"]
        cached = geometry.measure(trial, frames=range(len(trial)), protected_depths=protected)
        full = geometry.measure(trial)
        assert _rejected(cached, protected) == _rejected(full, protected)
        if not _rejected(cached, protected):
            _assert_same(cached, full)
            assert cached["frame_complete"].all() and cached["reliable"]
        else:
            # A later unprotected request completes/requeries all missing work.
            completed = geometry.measure(trial, frames=range(len(trial)))
            _assert_same(completed, full)
            assert completed["frame_complete"].all()
    finally:
        geometry.stop_workers()


def test_partial_rejection_is_not_cached_but_completed_prefix_is():
    geometry, poses = _geometry()
    part = geometry.measure_frames(poses, [1, 2])
    part["rejection"] = 2
    remember_frames(geometry, poses, [part])
    assert 1 in geometry._window_frame_cache and 2 not in geometry._window_frame_cache
    measured = geometry.measure(poses, frames=range(1, 3))
    assert measured["complete"] and measured["frame_complete"].all()
    assert measured["query_scope"]["queried_frames"] == [2]


def test_cached_failure_remains_unknown_without_a_geometry_query(monkeypatch):
    geometry, poses = _geometry()
    geometry._install_reference_observations({}, {}, None, "missing-evidence")
    full = geometry.measure(poses, frames=range(2, 4))
    assert full["complete"] and full["failures"]
    measured = geometry.measure(poses, frames=range(2, 4), protected_depths=full["depths"])
    assert not measured["complete"] and measured["failures"]
    assert not measured["reliable"]
    assert measured["query_scope"]["queried_frames"] == []


def test_linearization_requires_exact_pose_and_witness_content():
    geometry, poses = _geometry()
    full = geometry.measure(poses, jacobians=True)
    again = geometry.linearize(poses, full)
    assert geometry.linearization_stats == dict(reused=len(full["rows"]), computed=0)
    _assert_same(again, full)
    changed = deepcopy(full)
    row = next(row for key, row in changed["rows"] if key[1] == "static")
    row["point"][0] += .001
    geometry.linearize(poses, changed)
    assert geometry.linearization_stats["computed"] == 1
    # Public result mutation cannot corrupt a retained Jacobian.
    again["rows"][0][1]["jacobian"][:] = 17.
    _assert_same(geometry.linearize(poses, full), full)
    trial = poses.copy()
    frame = full["rows"][0][0][0]
    trial[frame, 7] = np.nextafter(trial[frame, 7], 1.)
    geometry.linearize(trial, full)
    assert geometry.linearization_stats["computed"] == sum(k[0] == frame for k, _ in full["rows"])


@pytest.mark.parametrize("reject", [False, True])
def test_probe_prioritizes_recent_rejection_and_completes_every_accepted_frame(monkeypatch, reject):
    from omniretargeting._fast.mesh_repair import OriginalGeometry
    from tests.test_original_geometry_batches import _problem, _scene
    objective, poses = _problem()
    poses = np.tile(poses, (5, 1))
    objective.frames = len(poses)
    geometry = OriginalGeometry(objective, (), _scene("closed"))
    full = geometry.measure(poses)
    protected = dict(full["depths"])
    if reject:
        key = next(k for k, d in protected.items() if k[0] == 39 and d > .0001)
        protected[key] = 0.
    geometry._window_frame_cache.clear()
    geometry._window_rejections = [39]
    visits = []
    original = geometry.measure_frames

    def record(poses, frames, **kwargs):
        visits.append(tuple(frames))
        return original(poses, frames, **kwargs)
    monkeypatch.setattr(geometry, "measure_frames", record)
    actual = geometry.measure(poses, frames=range(len(poses)), protected_depths=protected)
    assert visits[0] == (39,)
    assert _rejected(actual, protected) == reject
    if reject:
        assert len(visits) == 1 and not actual["frame_complete"][39]
        assert 39 not in geometry._window_frame_cache
        _assert_same(geometry.measure(poses, frames=range(len(poses))), full)
    else:
        assert visits[1] == tuple(range(31)) and len(visits[2]) == 8
        assert actual["frame_complete"].all()
        _assert_same(actual, full)


def test_probe_uses_most_changed_frames_and_retains_stable_ties(monkeypatch):
    from omniretargeting._fast.mesh_repair import OriginalGeometry
    from tests.test_original_geometry_batches import _problem, _scene
    objective, poses = _problem()
    poses = np.tile(poses, (5, 1))
    poses[:, 0] = 0.  # Equal binary displacement, including every tie.
    objective.frames = len(poses)
    geometry = OriginalGeometry(objective, (), _scene("closed"))
    geometry.measure(poses)
    reference = geometry._window_reference.copy()
    trial = poses.copy()
    trial[:, 0] += .0001
    trial[39, 0] += .001
    visits = []
    original = geometry.measure_frames

    def record(poses, frames, **kwargs):
        visits.append(tuple(frames))
        return original(poses, frames, **kwargs)
    monkeypatch.setattr(geometry, "measure_frames", record)
    geometry.measure(trial, frames=range(len(trial)), protected_depths={})
    assert visits[0] == (*range(31), 39)
    # A partial rejection must not become the next sorting reference.
    np.testing.assert_array_equal(geometry._window_reference, reference)

