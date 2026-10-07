"""Native query parity, fallback semantics and conservative motion certificates."""

import mujoco
import numpy as np
import pytest

from omniretargeting._fast.batched_self_collision import BatchedSelfCollision, NativeUnavailable
from omniretargeting._fast.kinematics import TrajectoryKinematics
from omniretargeting.geometry_query import GeometryQueryError
from omniretargeting.self_collision import SelfCollisionQuery
from tests.test_convex_mesh_distance import mesh_pair


def _batch(query, workers=3):
    try:
        return BatchedSelfCollision(query, workers)
    except NativeUnavailable as error:
        pytest.skip(str(error))


def _poses(model):
    rng = np.random.default_rng(315)
    poses = np.tile(model.qpos0, (70, 1))
    poses[:10, 7] = [.15, .05, 0, .23, .2, .199999, .200001, .229999, .230001, 2]
    poses[10:, 7:10] = rng.normal(scale=.10, size=(60, 3)) + [.16, 0, 0]
    for start in (3, 10):
        quat = rng.normal(size=(60, 4))
        poses[10:, start:start + 4] = quat / np.linalg.norm(quat, axis=1)[:, None]
    return poses


def _assert_rows(query, poses, rows, unknown):
    errors = 0
    for pose, actual in zip(poses, rows):
        try:
            expected = query.query(pose)
        except GeometryQueryError:
            errors += 1
            assert not actual
            continue
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            for key in ('pair', 'pair_id', 'pair_index', 'body_ids', 'clearance', 'points', 'direction'):
                np.testing.assert_array_equal(left[key], right[key], err_msg=key)
                assert np.asarray(left[key]).tobytes() == np.asarray(right[key]).tobytes(), key
            expected_jacobian = query.jacobian(right, query.data)
            np.testing.assert_array_equal(left['_jacobian'], expected_jacobian)
            assert left['_jacobian'].tobytes() == expected_jacobian.tobytes()
    assert errors == unknown


def test_mesh_rows_match_original_including_ties_containment_and_jacobians():
    model = mesh_pair()
    query = SelfCollisionQuery(model)
    poses = _poses(model)
    kin = TrajectoryKinematics(model, np.arange(model.nv)).evaluate(poses)
    batch = _batch(query)
    _assert_rows(query, poses, *batch.rows(poses, kin, True))
    _assert_rows(query, poses, *batch.rows(poses, kin, True))
    assert batch.diagnostics['exact_pose_reused']
    # The distance cutoff is part of an exact-pose query, too.
    query.cfg['detection_distance'] = .2
    _assert_rows(query, poses, *batch.rows(poses, kin, True))
    assert not batch.diagnostics['exact_pose_reused']


@pytest.mark.parametrize('offset', [0., 1e4, 1e10])
def test_support_certificates_preserve_bytes_near_ties_and_large_origins(offset):
    model = mesh_pair()
    query = SelfCollisionQuery(model)
    poses = _poses(model)[:18]
    # Exact face ties and progressively smaller rotations exercise both
    # certified choices and the original centered BLAS tie resolution.
    angles = np.tile([0., 1e-14, -1e-12, 1e-9, .03, -.2], 3)
    poses[:, 3:7] = [1., 0., 0., 0.]
    poses[:, 10:14] = np.column_stack((np.cos(angles), np.zeros((18, 2)), np.sin(angles)))
    poses[:, :3] += [offset, -offset, offset]
    poses[:, 7:10] += [offset, -offset, offset]
    kin = TrajectoryKinematics(model, np.arange(model.nv)).evaluate(poses)
    _assert_rows(query, poses, *_batch(query).rows(poses, kin, True))


def test_asymmetric_seed_axes_keep_original_facet_and_witness_bytes():
    model = mesh_pair()
    query = SelfCollisionQuery(model)
    rng = np.random.default_rng(901)
    axes = rng.normal(size=(26, 3))
    query.mesh_distance.seed_axes = axes / np.linalg.norm(axes, axis=1)[:, None]
    poses = _poses(model)
    kin = TrajectoryKinematics(model, np.arange(model.nv)).evaluate(poses)
    _assert_rows(query, poses, *_batch(query).rows(poses, kin, True))


def test_displacement_certificate_never_drops_a_reference_pair():
    model = mesh_pair()
    query = SelfCollisionQuery(model)
    batch = _batch(query)
    poses = _poses(model)
    poses[:, 7] += 3
    kin = TrajectoryKinematics(model, np.arange(model.nv))
    batch.rows(poses, kin.evaluate(poses), True)
    for displacement in (.01, -2.95):
        changed = poses.copy()
        changed[:, 7] += displacement
        changed[:, 10:14] = [np.cos(.02), 0, 0, np.sin(.02)]
        rows, unknown = batch.rows(changed, kin.evaluate(changed), True)
        _assert_rows(query, changed, rows, unknown)
        if displacement == .01:
            assert batch.diagnostics['sphere_reused_pairs'] == len(poses)


def test_partial_frame_reuse_preserves_candidate_bytes_rows_and_jacobians():
    model, poses = mesh_pair(), None
    poses = _poses(model)
    query = SelfCollisionQuery(model)
    batch, kin = _batch(query), TrajectoryKinematics(model, np.arange(model.nv))
    batch.rows(poses, kin.evaluate(poses), True)
    for jacobians in (True, False, True):
        changed = poses.copy()
        changed[::3, 7] += .001
        current_kin = kin.evaluate(changed, jacobians)
        actual = batch.candidates(current_kin)
        assert batch.diagnostics['transform_reused_pairs'] > 0
        fresh = _batch(SelfCollisionQuery(model))
        expected = fresh.candidates(current_kin)
        for a, b in zip(actual, expected):
            assert a.tobytes() == b.tobytes()
        if jacobians:
            _assert_rows(query, changed, *batch.rows(changed, current_kin, True))
        else:
            rows, unknown = batch.rows(changed, current_kin, False)
            expected_rows, expected_unknown = fresh.rows(changed, current_kin, False)
            assert unknown == expected_unknown
            for a, b in zip(rows, expected_rows):
                assert len(a) == len(b)
                for left, right in zip(a, b):
                    for key in left:
                        np.testing.assert_array_equal(left[key], right[key])
        poses = changed


def test_pair_reuse_with_changed_third_geom_and_threshold_invalidation():
    vertices = ' '.join(map(str, np.array(np.meshgrid(*[[-.1, .1]] * 3)).reshape(3, -1).T.ravel()))
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><asset>
      <mesh name="cube" vertex="{vertices}"/></asset><worldbody>
      <body><freejoint/><geom type="mesh" mesh="cube"/></body>
      <body pos=".15 0 0"><freejoint/><geom type="mesh" mesh="cube"/></body>
      <body pos="0 .15 0"><freejoint/><geom type="mesh" mesh="cube"/></body>
      </worldbody></mujoco>''')
    query, poses = SelfCollisionQuery(model), np.tile(model.qpos0, (6, 1))
    batch, kin = _batch(query), TrajectoryKinematics(model, np.arange(model.nv))
    _assert_rows(query, poses, *batch.rows(poses, kin.evaluate(poses), True))
    poses[:, 14] += .003
    current = kin.evaluate(poses)
    result = batch.rows(poses, current, True)
    assert batch.diagnostics['transform_reused_pairs'] == len(poses)
    _assert_rows(query, poses, *result)
    query.cfg['detection_distance'] = .2
    result = batch.rows(poses, current, True)
    assert batch.diagnostics['transform_reused_pairs'] == 0
    _assert_rows(query, poses, *result)




def test_native_failure_keeps_unknown_frames_and_exact_python_fallback(monkeypatch):
    model = mesh_pair()
    query = SelfCollisionQuery(model)
    poses = _poses(model)[:10]
    kin = TrajectoryKinematics(model, np.arange(model.nv)).evaluate(poses)
    batch = _batch(query)
    shape = (len(poses), len(query.pairs))
    monkeypatch.setattr(batch, 'candidates', lambda kin: (
        np.full(shape, np.inf), np.zeros((*shape, 2, 3)), np.zeros((*shape, 3)), np.full(shape, 2, np.uint8)))
    _assert_rows(query, poses, *batch.rows(poses, kin, True))

    def uncertain(*args):
        raise GeometryQueryError('fixture indeterminate')

    monkeypatch.setattr(query.mesh_distance, 'query', uncertain)
    rows, unknown = batch.rows(poses, kin, True, target=.002)
    assert unknown == len(poses)
    assert not any(rows)


def test_primitive_contact_normals_keep_original_forward_fallback():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><freejoint/><geom type="sphere" size=".1"/></body>
      <body pos=".15 0 0"><freejoint/><geom type="sphere" size=".1"/></body>
      </worldbody></mujoco>''')
    query = SelfCollisionQuery(model)
    poses = _poses(model)
    kin = TrajectoryKinematics(model, np.arange(model.nv)).evaluate(poses)
    _assert_rows(query, poses, *_batch(query).rows(poses, kin, True))


def test_native_point_rows_keep_reference_arithmetic_for_selected_columns():
    from omniretargeting._fast.native_point_rows import NativePointRows
    model = mesh_pair()
    query = SelfCollisionQuery(model)
    _batch(query)
    poses = _poses(model)[:20]
    rng = np.random.default_rng(37)
    for columns in (np.arange(model.nv), np.array([0, 3, 6, 8])):
        kin = TrajectoryKinematics(model, columns).evaluate(poses)
        native = NativePointRows(kin)
        ids = rng.integers(0, model.nbody, size=(len(poses), 17))
        points, direction = rng.normal(size=(2, len(poses), 17, 3))
        jac = kin.point_jacobians(ids, points)
        np.testing.assert_array_equal(native.jacobians(ids, points), jac)
        np.testing.assert_array_equal(native.rows(ids, points, direction),
                                      np.einsum('fsi,fsiw->fsw', direction, jac))


def test_native_fk_calls_original_mujoco_stages_bit_for_bit():
    model = mesh_pair()
    _batch(SelfCollisionQuery(model))
    poses = _poses(model)
    for columns in (np.arange(model.nv), np.array([0, 3, 6, 8])):
        for jacobian in (False, True):
            reference = TrajectoryKinematics(model, columns).evaluate(poses, jacobian)
            accelerated = TrajectoryKinematics(model, columns).evaluate(poses, jacobian, native=True)
            names = ['xpos', 'xmat', 'geom_xpos', 'geom_xmat']
            if jacobian:
                names += ['jacp', 'jacr']
            for name in names:
                np.testing.assert_array_equal(getattr(accelerated, name), getattr(reference, name))
                assert not getattr(accelerated, name).flags.writeable
            np.testing.assert_array_equal(accelerated.data.qpos, reference.data.qpos)


def test_native_collision_jacobians_keep_mujoco_and_blas_reduction_order():
    from omniretargeting._fast.native_collision_rows import collision_jacobians

    model = mesh_pair()
    query = SelfCollisionQuery(model)
    _batch(query)
    rng = np.random.default_rng(619)
    for pose in _poses(model)[10:25]:
        query.data.qpos[:] = pose
        mujoco.mj_forward(model, query.data)
        rows = [dict(body_ids=[body, other], points=rng.normal(size=(2, 3)),
                     direction=rng.normal(size=3))
                for body, other in ((1, 2), (2, 1), (1, 0), (0, 2))]
        expected = np.array([query.jacobian(row, query.data) for row in rows])
        np.testing.assert_array_equal(collision_jacobians(query, rows), expected)
