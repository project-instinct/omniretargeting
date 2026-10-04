"""Legacy witness bits, frame identity, motion bounds and fork-safe caches."""

import os
import select
import signal
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from omniretargeting._fast.batched_self_collision import NativeUnavailable
from omniretargeting._fast.kinematics import TrajectoryKinematics
from omniretargeting._fast.self_collision_window import SelfCollisionWindow
from omniretargeting.geometry_query import GeometryQueryError
from omniretargeting.self_collision import SelfCollisionQuery
from tests.test_batched_self_collision import _poses
from tests.test_convex_mesh_distance import mesh_pair


@pytest.fixture(autouse=True)
def serial_blas():
    with threadpool_limits(limits=1):
        yield


def client(model, columns=None):
    result = SelfCollisionWindow(SelfCollisionQuery(model), columns=columns).prepare()
    assert result.prepared
    return result


def assert_legacy(query, poses, frames, actual, *, columns=None, activation=None):
    expected_rows, expected_depths, failures = [], {}, []
    omitted = []
    for frame in frames:
        try:
            rows = query.query(poses[frame])
        except GeometryQueryError as error:
            failures.append(f'unknown self frame {frame}: {error}')
            omitted.append(0)
            continue
        missed = 0
        for row in rows:
            key = (frame, 'self', *row['pair'])
            expected_depths[key] = max(0., -row['clearance'])
            if activation is not None and row['clearance'] >= activation:
                missed += 1
                continue
            if columns is not None:
                row['jacobian'] = query.jacobian(row, query.data)[columns]
            expected_rows.append((key, row))
        omitted.append(missed)
    assert actual['depths'] == expected_depths
    assert actual['failures'] == failures
    assert actual['complete']
    assert actual['reliable'] == (not failures)
    assert actual['queried_frames'] == tuple(frames)
    np.testing.assert_array_equal(actual['omitted_near_misses'], omitted)
    assert len(actual['rows']) == len(expected_rows)
    for (key, row), (expected_key, expected) in zip(actual['rows'], expected_rows):
        assert key == expected_key
        assert row.keys() == expected.keys()
        for field in row:
            np.testing.assert_array_equal(row[field], expected[field], err_msg=str((key, field)))
            if isinstance(expected[field], np.ndarray):
                assert row[field].dtype == expected[field].dtype
                assert row[field].tobytes() == expected[field].tobytes(), (key, field, 'bytes')
            elif isinstance(expected[field], (float, np.floating)):
                assert np.asarray(row[field]).tobytes() == np.asarray(expected[field]).tobytes(), (key, field, 'bytes')
    return expected_rows


@pytest.mark.parametrize('columns', [None, np.array([0, 3, 6, 8]), np.empty(0, dtype=int)])
def test_mesh_witnesses_and_jacobians_match_legacy_bits(columns):
    model = mesh_pair()
    c = client(model, columns)
    poses = _poses(model)
    frames = range(7, 66)
    selected = np.arange(model.nv) if columns is None else columns
    result = c.measure(poses, frames=frames, jacobians=True)
    assert_legacy(c.query, poses, frames, result, columns=selected)
    again = c.measure(poses, frames=frames, jacobians=True)
    assert again['diagnostics']['computed_frames'] == 0
    assert_legacy(c.query, poses, frames, again, columns=selected)


@pytest.mark.parametrize('activation', [None, 0., .002, .008])
def test_activation_preserves_near_zero_depth_keys_and_strict_ties(activation):
    model = mesh_pair()
    c = client(model)
    poses = _poses(model)[:10]
    result = c.measure(poses, frames=range(10), activation_distance=activation)
    assert_legacy(c.query, poses, range(10), result, activation=activation)
    assert any(value == 0 for value in result['depths'].values())
    assert any(value > 0 for value in result['depths'].values())


def test_sparse_frame_ids_external_kinematics_and_transform_reuse():
    model = mesh_pair()
    c = client(model)
    poses = _poses(model)
    ids = [2, 10, 30, 61]
    kin = TrajectoryKinematics(model, []).evaluate(poses[ids], False, native=True)
    result = c.query_frames(poses, ids, kinematics=SimpleNamespace(
        geom_xpos=kin.geom_xpos, geom_xmat=kin.geom_xmat))
    assert_legacy(c.query, poses, ids, result)
    changed = poses.copy()
    # Quaternion sign gives exactly the same geom placements, a different pose.
    changed[ids, 3:7] *= -1
    result = c.query_frames(changed, ids)
    assert_legacy(c.query, changed, ids, result)
    assert result['diagnostics']['transform_reused_pairs'] > 0
    assert result['diagnostics']['native_pairs'] == 0
    assert_legacy(c.query, poses, range(3, 8), c.measure(poses, frames=range(3, 8)))


def test_sphere_displacement_certificates_follow_global_frame_and_never_drop_pairs():
    model = mesh_pair()
    c = client(model)
    poses = _poses(model)
    poses[:, 7] += 3
    c.measure(poses, frames=range(30, 62))
    for shift in (.01, -.03, -2.97, .001):
        poses[30:62, 7] += shift
        poses[30:62, 10:14] = [np.cos(.02), 0, 0, np.sin(.02)]
        result = c.measure(poses, frames=range(30, 62))
        assert_legacy(c.query, poses, range(30, 62), result)
        if shift == .01:
            assert result['diagnostics']['sphere_reused_pairs'] == 32
    # A different global window must not reuse its predecessor's certificates.
    assert_legacy(c.query, poses, range(0, 30), c.measure(poses, frames=range(0, 30)))


def test_primitive_normals_and_sphere_threshold_ties_match_original():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><freejoint/><geom type="sphere" size=".1"/></body>
      <body pos=".15 0 0"><freejoint/><geom type="sphere" size=".1"/></body>
      </worldbody></mujoco>''')
    c = client(model)
    poses = _poses(model)
    cutoff = sum(model.geom_rbound) + c.threshold
    poses[:3, 7] = [np.nextafter(cutoff, -np.inf), cutoff, np.nextafter(cutoff, np.inf)]
    result = c.measure(poses, frames=range(len(poses)), jacobians=True)
    assert_legacy(c.query, poses, range(len(poses)), result, columns=np.arange(model.nv))
    assert result['diagnostics']['sphere_tie_pairs'] >= 3


def test_unknown_frames_are_not_cached_or_reported_free(monkeypatch):
    model = mesh_pair()
    c = client(model)
    poses = _poses(model)[:5]

    def fallback(position, rotation, near, clearance, points, directions, status):
        status[near.astype(bool)] = 2

    def unknown(*args):
        raise GeometryQueryError('fixture indeterminate')

    monkeypatch.setattr(c, '_run_native', fallback)
    monkeypatch.setattr(c.query.mesh_distance, 'query', unknown)
    for repeat in range(2):
        result = c.measure(poses, frames=range(5))
        assert len(result['failures']) == 5
        assert result['complete'] and not result['reliable']
        assert not result['rows'] and not result['depths']
        assert np.isnan(result['max_depth']).all()
        assert not result['body_reliable'].any()
        assert result['diagnostics']['computed_frames'] == 5
        assert not c.states and not c.pose_cache


def test_unavailable_native_uses_exact_serial_reference(monkeypatch):
    import omniretargeting._fast.self_collision_window as module

    def unavailable(*args, **kwargs):
        raise NativeUnavailable('fixture missing compiler')

    monkeypatch.setattr(module, 'BatchedSelfCollision', unavailable)
    model = mesh_pair()
    c = client(model)
    poses = _poses(model)[:10]
    result = c.measure(poses, frames=range(10), jacobians=True)
    assert not result['diagnostics']['native_available']
    assert_legacy(c.query, poses, range(10), result, columns=np.arange(model.nv))


def test_private_clone_output_and_buffers_are_independent():
    model = mesh_pair()
    c = client(model)
    poses = _poses(model)[:10]
    before = c.measure(poses, frames=range(10))
    clone = c.fork_private_client()
    assert clone.query.data is not c.query.data
    assert clone.query.pairs is not c.query.pairs
    result = clone.measure(poses, frames=range(10))
    result['rows'][0][1]['points'][:] = 100
    result['rows'][0][1]['pair'][0] = -100
    assert_legacy(c.query, poses, range(10), c.measure(poses, frames=range(10)))
    assert c.measure(poses, frames=range(10))['depths'] == before['depths']


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='requires POSIX fork')
def test_fork_child_never_compiles_or_starts_threads_or_pools(monkeypatch):
    model = mesh_pair()
    c = client(model)
    poses = _poses(model)[:32]
    c.measure(poses, frames=range(32))
    reader, writer = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(reader)
        try:
            def forbidden(*args, **kwargs):
                raise AssertionError('fork child attempted preparation/threads/processes')
            subprocess.run = threading.Thread.start = ThreadPoolExecutor.__init__ = forbidden
            import threadpoolctl
            threadpoolctl.ThreadpoolController.__init__ = forbidden
            clone = c.fork_private_client()
            candidate = poses.copy()
            candidate[:, 7] += 1e-5
            result = clone.measure(candidate, frames=range(32), jacobians=True)
            assert_legacy(clone.query, candidate, range(32), result, columns=np.arange(model.nv))
            assert len(os.listdir('/proc/self/task')) == 1
            os.write(writer, b'OK')
        except BaseException as error:
            os.write(writer, repr(error).encode())
        finally:
            os.close(writer)
        os._exit(0)
    os.close(writer)
    try:
        ready, _, _ = select.select([reader], [], [], 15)
        if not ready:
            os.kill(pid, signal.SIGKILL)
            pytest.fail('fork self query timed out')
        assert os.read(reader, 4096) == b'OK'
    finally:
        os.close(reader)
        os.waitpid(pid, 0)


@pytest.mark.parametrize('bad_frames', [[1, 1], [2, 1], [-1], [100]])
def test_frame_validation(bad_frames):
    c = client(mesh_pair())
    with pytest.raises(ValueError):
        c.query_frames(_poses(c.model), bad_frames)


def test_bad_kinematics_and_frozen_band_validation():
    c = client(mesh_pair())
    poses = _poses(c.model)
    with pytest.raises(ValueError, match='kinematics'):
        c.measure(poses, frames=range(10), kinematics=SimpleNamespace(
            geom_xpos=np.zeros((1, c.model.ngeom, 3)), geom_xmat=np.zeros((1, c.model.ngeom, 3, 3))))
    c.query.cfg['detection_distance'] *= 2
    with pytest.raises(ValueError, match='frozen'):
        c.measure(poses, frames=range(10))


def test_seed_completed_imports_only_reported_pairs_without_remeasurement(monkeypatch):
    model = mesh_pair()
    c = client(model)
    poses = _poses(model)[:20]
    frames = range(2, 17)
    kin = TrajectoryKinematics(model, []).evaluate(poses[list(frames)], False, native=True)
    source = c.measure(poses, frames=frames, kinematics=kin)
    new = client(model)

    def forbidden(*args, **kwargs):
        raise AssertionError('Seeding must not measure anything')

    with monkeypatch.context() as patch:
        patch.setattr(new.query, 'query', forbidden)
        patch.setattr(new.query.mesh_distance, 'query', forbidden)
        patch.setattr(new, '_run_native', forbidden)
        seeded = new.seed_completed(poses, source, kin)
    assert seeded['seeded_pairs'] == len(source['rows'])
    assert not new.pose_cache
    assert sum(int((state[3] == 1).sum()) for state in new.states.values()) == len(source['rows'])
    actual = new.measure(poses, frames=frames, kinematics=kin)
    assert_legacy(new.query, poses, frames, actual)
    assert actual['diagnostics']['transform_reused_pairs'] == len(source['rows'])
    # A narrowed seed cannot certify the omitted near-zero depth entries.
    narrow = c.measure(poses, frames=frames, kinematics=kin, activation_distance=0.)
    another = client(model)
    assert another.seed_completed(poses, narrow, kin)['seeded_pairs'] == len(narrow['rows'])
    assert_legacy(another.query, poses, frames, another.measure(poses, frames=frames, kinematics=kin))


@pytest.mark.parametrize('metadata', [dict(complete=False), dict(failures=['unknown self frame 2']),
                                     dict(reliable=False), dict(frame_complete=np.zeros(10, bool))])
def test_seed_incomplete_or_unknown_measurement_grants_no_cache(metadata):
    model = mesh_pair()
    c = client(model)
    poses = _poses(model)[:10]
    kin = TrajectoryKinematics(model, []).evaluate(poses, False, native=True)
    source = c.measure(poses, frames=range(10))
    new = client(model)
    assert new.seed_completed(poses, {**source, **metadata}, kin)['seeded_pairs'] == 0
    assert not new.states and not new.pose_cache


def legacy_reuse_seed(c, poses):
    frames = range(len(poses))
    kin = TrajectoryKinematics(c.model, []).evaluate(poses, False, native=True)
    reuse, rows = {}, []
    for frame in frames:
        rows.extend(((frame, 'self', *row['pair']), row)
                    for row in c.query.query(poses[frame], reuse=reuse.setdefault(frame, {})))
    return dict(rows=rows, complete=True, reliable=True, failures=[], queried_frames=tuple(frames)), kin, reuse


def test_explicit_legacy_none_reuse_is_an_exact_absence_certificate_not_a_distance(monkeypatch):
    c = client(mesh_pair())
    poses = _poses(c.model)[:10]
    poses[:, 7] = np.linspace(.235, .295, 10)
    source, kin, reuse = legacy_reuse_seed(c, poses)
    assert not source['rows']
    assert all(list(value.values()) == [None] for value in reuse.values())
    new = client(c.model)
    new.seed_query_reuse(poses, source, kin, legacy_reuse=reuse)
    assert all(np.all(state[3] == 3) for state in new.states.values())
    assert not new.pose_cache

    def no_narrow(position, rotation, near, *args):
        assert not near.any()

    monkeypatch.setattr(new, '_run_native', no_narrow)
    candidate = poses.copy()
    candidate[:, 3:7] *= -1
    result = new.measure(candidate, frames=range(10))
    assert_legacy(new.query, candidate, range(10), result)
    assert not result['rows'] and not result['depths']
    assert result['diagnostics']['transform_reused_pairs'] == 10


def test_explicit_none_certificate_never_survives_a_changed_geom():
    c = client(mesh_pair())
    poses = _poses(c.model)[:10]
    poses[:, 7] = .24
    source, kin, reuse = legacy_reuse_seed(c, poses)
    c.seed_query_reuse(poses, source, kin, legacy_reuse=reuse)
    candidate = poses.copy()
    candidate[:, 7] = .1999
    result = c.measure(candidate, frames=range(10), jacobians=True)
    assert_legacy(c.query, candidate, range(10), result, columns=np.arange(c.model.nv))
    assert result['diagnostics']['transform_reused_pairs'] == 0
    assert len(result['depths']) == 10
    assert all(value > 0 for value in result['depths'].values())


def test_row_absence_or_mismatched_explicit_none_keys_never_infer_certificates():
    c = client(mesh_pair())
    poses = _poses(c.model)[:10]
    poses[:, 7] = .24
    source, kin, reuse = legacy_reuse_seed(c, poses)
    empty = client(c.model)
    assert empty.seed_query_reuse(poses, source, kin, legacy_reuse={})['seeded_inactive_pairs'] == 0
    assert not empty.states
    result = empty.measure(poses, frames=range(10))
    assert result['diagnostics']['native_pairs'] == 10
    changed = poses.copy()
    changed[:, 7] += .001
    changed_kin = TrajectoryKinematics(c.model, []).evaluate(changed, False, native=True)
    stale = client(c.model)
    stats = stale.seed_query_reuse(changed, source, changed_kin, legacy_reuse=reuse)
    assert stats['seeded_inactive_pairs'] == 0 and stats['skipped_inactive_keys'] == 10
    assert not stale.states
    wrong_band = {frame: {(key[0], key[1], .008, *key[3:]): value
                          for key, value in entries.items()} for frame, entries in reuse.items()}
    assert stale.seed_query_reuse(poses, source, kin, legacy_reuse=wrong_band)['seeded_inactive_pairs'] == 0


def test_explicit_none_conflict_is_atomic():
    c = client(mesh_pair())
    poses = _poses(c.model)[:10]
    source, kin, reuse = legacy_reuse_seed(c, poses)
    frame, row = source['rows'][0]
    key = next(iter(reuse[frame[0]]))
    reuse[frame[0]][key] = None
    assert row['clearance'] < c.threshold
    with pytest.raises(ValueError, match='conflict'):
        c.seed_query_reuse(poses, source, kin, legacy_reuse=reuse)
    assert not c.states


def test_fork_runtime_blas_check_uses_parent_resolved_getters(monkeypatch):
    c = client(mesh_pair())
    c.blas_checked_pid = -1
    c.blas_controls = (SimpleNamespace(get_num_threads=lambda: 2),)
    with pytest.raises(RuntimeError, match='single-thread BLAS'):
        c.measure(_poses(c.model), frames=range(1))
