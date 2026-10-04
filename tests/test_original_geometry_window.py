"""Window geometry preserves global witness identities and exact legacy rows."""

import multiprocessing
import os

import numpy as np
import pytest

from omniretargeting._fast.mesh_repair import OriginalGeometry
from tests.test_original_geometry_api import _objects, _rotated_problem
from tests.test_original_geometry_batches import _assert_same, _scene


@pytest.fixture(autouse=True)
def cpu_queries(monkeypatch):
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "0")


def _geometry(*, columns=None):
    objective, poses = _rotated_problem()
    if columns is None:
        columns = objective.coords.cols
    geometry = OriginalGeometry(objective, _objects(), _scene("closed"), manifold=True,
                                columns=columns, reference_poses=poses)
    return geometry, poses


def _restricted(measured, frames):
    return dict(rows=[item for item in measured["rows"] if item[0][0] in frames],
                depths={key: value for key, value in measured["depths"].items() if key[0] in frames},
                failures=[], complete=True)


@pytest.mark.parametrize("parallel", [False, True])
def test_window_is_bit_identical_to_restricted_full_root_self_and_moving(parallel, monkeypatch):
    from omniretargeting._fast import parallel_geometry
    geometry, poses = _geometry(columns=np.array([5, 0, 7, 3, 2, 6, 1, 4]))
    if parallel:
        monkeypatch.setattr(parallel_geometry, "WORKERS", 3)
        monkeypatch.setattr(parallel_geometry, "MIN_FRAMES_PER_WORKER", 2)
        geometry.start_workers(len(poses))
    try:
        frames = range(2, 6)
        # Cold window, then full trajectory, then cache-only window.
        cold = geometry.measure(poses, frames=frames, jacobians=True)
        assert geometry.window_stats["queried_frames"] == len(frames)
        assert cold["query_scope"] == dict(requested_frames=list(frames), requested_frame_count=4,
                                           queried_frames=list(frames), queried_frame_count=4,
                                           cached_frame_count=0)
        full = geometry.measure(poses, jacobians=True)
        expected = _restricted(full, frames)
        _assert_same(cold, expected)
        warm = geometry.measure(poses, frames=frames, jacobians=True)
        assert geometry.window_stats["queried_frames"] == 0
        assert warm["query_scope"] == dict(requested_frames=list(frames), requested_frame_count=4,
                                           queried_frames=[], queried_frame_count=0,
                                           cached_frame_count=4)
        _assert_same(warm, expected)
        matrix, phi = geometry.linear_rows(warm, frames=frames)
        full_matrix, full_phi = geometry.linear_rows(full)
        selected = np.array([key[0] in frames for key, _ in full["rows"]])
        reference = full_matrix[selected, frames.start * geometry.width:frames.stop * geometry.width]
        assert matrix.shape == (selected.sum(), len(frames) * geometry.width)
        assert matrix.toarray().tobytes() == reference.toarray().tobytes()
        assert phi.tobytes() == full_phi[selected].tobytes()
        assert warm["queried_frames"] == tuple(frames)
        assert warm["frame_complete"].all() and warm["reliable"]
    finally:
        geometry.stop_workers()


def test_exact_pose_cache_only_queries_changed_selected_frames(monkeypatch):
    geometry, poses = _geometry()
    baseline = geometry.measure(poses)
    visits = []
    original = geometry.measure_frames

    def record(poses, frames, **kwargs):
        visits.append(tuple(frames))
        return original(poses, frames, **kwargs)

    monkeypatch.setattr(geometry, "measure_frames", record)
    candidate = poses.copy()
    candidate[0, 0] += .1  # Outside the window: no query.
    candidate[3, 0] = np.nextafter(candidate[3, 0], 1.)
    window = geometry.measure(candidate, frames=range(2, 6))
    assert visits == [(3,)]
    assert geometry.window_stats == dict(requested_frames=4, cached_frames=3, queried_frames=1)
    assert window["query_scope"] == dict(requested_frames=[2, 3, 4, 5], requested_frame_count=4,
                                         queried_frames=[3], queried_frame_count=1, cached_frame_count=3)
    _assert_same(geometry.measure(poses, frames=range(2, 6)), _restricted(baseline, range(2, 6)))
    assert visits == [(3,)]
    # Returned arrays/dictionaries may be modified without corrupting evidence.
    window["rows"][0][1]["direction"][:] = 17.
    window["depths"].clear()
    fresh = geometry.measure(candidate, frames=range(2, 6))
    assert fresh["depths"] and not np.all(fresh["rows"][0][1]["direction"] == 17.)


def test_activation_filter_exact_omitted_rows_and_unscreened_acceptance_depths():
    geometry, poses = _geometry()
    frames = range(1, 7)
    full = geometry.measure(poses, frames=frames)
    screened = geometry.measure(poses, frames=frames, activation_distance=0.)
    expected = [(key, row) for key, row in full["rows"] if row["clearance"] < 0]
    assert [key for key, _ in screened["rows"]] == [key for key, _ in expected]
    assert screened["depths"] == full["depths"]
    assert screened["max_depth"].tobytes() == full["max_depth"].tobytes()
    counts = np.zeros(len(frames), dtype=np.int64)
    for key, row in full["rows"]:
        counts[key[0] - frames.start] += row["clearance"] >= 0
    assert screened["omitted_near_misses"].tobytes() == counts.tobytes()
    depths = geometry.max_depth_per_body(poses, frames)
    assert geometry.window_stats["queried_frames"] == 0
    assert depths["max_depth"].tobytes() == full["max_depth"].tobytes()
    for key, depth in full["depths"].items():
        geoms = key[2:] if key[1] == "self" else key[-1:]
        for geom in geoms:
            owner = geometry.model.body_weldid[geometry.model.geom_bodyid[geom]]
            assert depths["max_depth"][key[0] - frames.start, owner] >= depth


def test_unknown_reference_masks_depth_and_keeps_frame_complete():
    geometry, poses = _geometry()
    geometry._install_reference_observations({}, {}, None, "unknown-reference")
    measured = geometry.measure(poses, frames=range(2, 3))
    assert measured["complete"] and measured["frame_complete"].all()
    assert measured["failures"] and not measured["reliable"]
    assert np.isnan(measured["max_depth"]).all() and not measured["body_reliable"].any()


@pytest.mark.parametrize("frames", [range(-1, 2), range(2, 1), range(0, 9), range(0, 8, 2), [1, 2]])
def test_invalid_window_rejected(frames):
    geometry, poses = _geometry()
    with pytest.raises(ValueError, match="range"):
        geometry.measure(poses, frames=frames)
    with pytest.raises(ValueError, match="range"):
        geometry.linear_rows(dict(rows=[]), frames=frames)


@pytest.mark.parametrize("activation", [-.001, .009, np.nan, np.inf])
def test_unsupported_activation_never_silently_omits_rows(activation):
    geometry, poses = _geometry()
    with pytest.raises(ValueError, match="activation_distance"):
        geometry.measure(poses, frames=range(2, 3), activation_distance=activation)


def test_empty_window_and_zero_width_columns_keep_shapes():
    geometry, poses = _geometry(columns=np.array([], dtype=int))
    measured = geometry.measure(poses, frames=range(3, 3), jacobians=True)
    matrix, phi = geometry.linear_rows(measured, frames=range(3, 3))
    assert matrix.shape == (0, 0) and phi.shape == (0,)
    assert measured["max_depth"].shape == (0, geometry.model.nbody)
    measured = geometry.measure(poses, frames=range(3, 4), jacobians=True)
    matrix, phi = geometry.linear_rows(measured, frames=range(3, 4))
    assert matrix.shape == (len(measured["rows"]), 0) and len(phi)


def _child_query(geometry, poses, connection):
    try:
        with pytest.raises(RuntimeError, match="fork_private_client"):
            geometry.measure(poses, frames=range(2, 6))
        client = geometry.fork_private_client()
        assert client.parallel is None and not client._gpu_allowed
        assert client.reference_observation_id == geometry.reference_observation_id
        assert client._moving_evidence_baseline is geometry._moving_evidence_baseline
        assert client.data is not geometry.data and client.moving.data is not geometry.moving.data
        result = client.measure(poses, frames=range(2, 6), jacobians=True)
        assert client.window_stats["queried_frames"] == 0
        changed = poses.copy()
        changed[3, 7] += .01
        client.measure(changed, frames=range(2, 6))
        assert client.window_stats["queried_frames"] == 1
        connection.send((True, result))
    except Exception as error:
        connection.send((False, repr(error)))
    finally:
        connection.close()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork unavailable")
def test_fork_private_client_does_not_requery_reference_or_touch_parent_pool(monkeypatch):
    from omniretargeting._fast import parallel_geometry
    monkeypatch.setattr(parallel_geometry, "WORKERS", 3)
    monkeypatch.setattr(parallel_geometry, "MIN_FRAMES_PER_WORKER", 2)
    geometry, poses = _geometry()
    geometry.start_workers(len(poses))
    context = multiprocessing.get_context("fork")
    parent, child = context.Pipe()
    try:
        expected = geometry.measure(poses, jacobians=True)
        process = context.Process(target=_child_query, args=(geometry, poses, child))
        process.start()
        child.close()
        assert parent.poll(20), "private geometry child stalled"
        ok, result = parent.recv()
        assert ok, result
        _assert_same(result, _restricted(expected, range(2, 6)))
        process.join(timeout=10)
        assert process.exitcode == 0
        _assert_same(geometry.measure(poses, jacobians=True), expected)
    finally:
        geometry.stop_workers()
        parent.close()
