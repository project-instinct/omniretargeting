"""Physical acquisition distances, fixed eligibility and conservative reuse."""

from copy import copy
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest

from omniretargeting._fast.batched_self_collision import BatchedSelfCollision
from omniretargeting._fast.fast_self_collision import FastSelfCollision, acquisition_self_collision
from omniretargeting._fast.kinematics import TrajectoryKinematics
from omniretargeting._fast import runtime_options
from omniretargeting.geometry_query import GeometryQueryError
from omniretargeting.self_collision import SelfCollisionQuery
from tests.test_convex_mesh_distance import mesh_pair


def evaluate(batch, poses, jacobian=False, target=None):
    kin = TrajectoryKinematics(batch.model, np.arange(batch.model.nv)).evaluate(poses, False)
    return batch.rows(poses, kin, jacobian, target)


def test_eligible_pairs_exclusions_and_model_options_are_unchanged():
    model = mesh_pair()
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    query = SelfCollisionQuery(model)
    pairs, flags, tolerance = query.pairs.copy(), model.opt.disableflags, model.opt.ccd_tolerance
    batch = FastSelfCollision(query)
    np.testing.assert_array_equal(np.stack((batch.first, batch.second), axis=1), pairs)
    assert batch.query is query
    assert model.opt.disableflags == flags
    assert model.opt.ccd_tolerance == tolerance
    assert batch.distance_model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD) == 0
    assert batch.distance_model is not model


@pytest.mark.parametrize('shift', [.15, .05, 0., .2000001, .201, .22, .23])
def test_signed_mesh_distance_including_containment_and_near_contact(shift):
    model = mesh_pair()
    query = SelfCollisionQuery(model)
    q = model.qpos0.copy()
    q[7] = shift
    batch = FastSelfCollision(query)
    rows, unknown = evaluate(batch, q[None], True)
    assert unknown == 0
    expected = query.query(q)
    assert [r['pair_id'] for r in rows[0]] == [r['pair_id'] for r in expected]
    for actual, exact in zip(rows[0], expected):
        assert actual['clearance'] == pytest.approx(exact['clearance'], abs=2e-8)
        if actual['clearance'] < 0:
            np.testing.assert_allclose(actual['points'][0] - actual['points'][1],
                                       actual['clearance'] * actual['direction'], atol=2e-8)


def test_random_mesh_poses_keep_penetrations_and_separating_bound_order():
    model = mesh_pair()
    query, rng = SelfCollisionQuery(model), np.random.default_rng(892)
    poses = np.tile(model.qpos0, (90, 1))
    poses[:, 7:10] = rng.uniform(-.28, .28, (len(poses), 3))
    for start in (3, 10):
        quat = rng.normal(size=(len(poses), 4))
        poses[:, start:start + 4] = quat / np.linalg.norm(quat, axis=1)[:, None]
    rows, unknown = evaluate(FastSelfCollision(query), poses)
    assert unknown == 0
    penetrations = 0
    for q, frame in zip(poses, rows):
        exact = query.query(q)
        actual = {r['pair_id']: r for r in frame}
        for row in exact:
            if row['clearance'] < -1e-7:
                penetrations += 1
                assert row['pair_id'] in actual
                assert actual[row['pair_id']]['clearance'] == pytest.approx(row['clearance'], abs=2e-8)
            elif row['pair_id'] in actual:
                assert actual[row['pair_id']]['clearance'] >= row['clearance'] - 2e-8
    assert penetrations > 10


@pytest.mark.parametrize('shift', [.16, .255])
def test_witness_jacobians_match_tangent_finite_differences(shift):
    model = mesh_pair()
    query = SelfCollisionQuery(model, {'detection_distance': .3})
    q = model.qpos0.copy()
    q[7:10] = [shift, .035, .022]
    q[10:14] = [np.cos(.17), 0., 0., np.sin(.17)]
    batch = FastSelfCollision(query)
    row = evaluate(batch, q[None], True)[0][0][0]
    for column in (0, 1, 5, 6, 7, 11):
        values = []
        for sign in (-1, 1):
            tangent = np.zeros(model.nv)
            tangent[column] = sign * 1e-5
            moved = q.copy()
            mujoco.mj_integratePos(model, moved, tangent, 1.)
            values.append(evaluate(batch, moved[None])[0][0][0]['clearance'])
        assert (values[1] - values[0]) / 2e-5 == pytest.approx(row['_jacobian'][column], abs=3e-6)


def test_displacement_reuse_catches_approach_rotation_and_cutoff_changes():
    model = mesh_pair()
    query, batch = SelfCollisionQuery(model), None
    batch = FastSelfCollision(query)
    poses = np.tile(model.qpos0, (12, 1))
    poses[:, 7] = .6
    evaluate(batch, poses, target=.002)
    for shift, rotation, cutoff in ((.59, .02, .002), (.199, .1, .002), (.24, .3, .03)):
        changed = poses.copy()
        changed[:, 7] = shift
        changed[:, 10:14] = [np.cos(rotation), 0., 0., np.sin(rotation)]
        rows = evaluate(batch, changed, target=cutoff)[0]
        if shift == .59:
            assert batch.diagnostics['bound_reused_pairs'] == len(poses)
        fresh = evaluate(FastSelfCollision(query), changed, target=cutoff)[0]
        for actual, expected in zip(rows, fresh):
            assert [r['pair_id'] for r in actual] == [r['pair_id'] for r in expected]
            np.testing.assert_allclose([r['clearance'] for r in actual], [r['clearance'] for r in expected])
        poses = changed


def test_exact_pose_cache_upgrades_jacobians_and_returns_independent_rows():
    model = mesh_pair()
    batch = FastSelfCollision(SelfCollisionQuery(model))
    poses = model.qpos0[None].copy()
    kin = TrajectoryKinematics(model, np.arange(model.nv)).evaluate(poses, False)
    first = batch.rows(poses, kin, False)[0][0][0]
    first['points'][:] = 100
    second = batch.rows(poses, kin, True)[0][0][0]
    assert batch.diagnostics['exact_pose_reused']
    assert batch.diagnostics['native_pairs'] == 0
    assert '_jacobian' in second
    assert abs(second['points']).max() < 1
    fork = copy(batch)
    assert fork.data is not batch.data
    assert not fork.cache


def test_inconsistent_and_coincident_mesh_witnesses_use_exact_authority(monkeypatch):
    model = mesh_pair()
    query, batch = SelfCollisionQuery(model), None
    batch = FastSelfCollision(query)

    def unreliable(model, data, a, b, maximum, points):
        points[:] = 0
        return -.05

    monkeypatch.setattr(mujoco, 'mj_geomDistance', unreliable)
    rows, unknown = evaluate(batch, model.qpos0[None])
    assert unknown == 0
    assert rows[0][0]['clearance'] == pytest.approx(-.05, abs=1e-8)
    assert batch.diagnostics['fallback_pairs'] == 1

    def uncertain(*args):
        raise GeometryQueryError('fixture indeterminate')

    monkeypatch.setattr(query.mesh_distance, 'query', uncertain)
    rows, unknown = evaluate(FastSelfCollision(query), model.qpos0[None])
    assert unknown == 1
    assert not rows[0]


def test_primitive_touch_retains_original_contact_normal_path():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><freejoint/><geom type="sphere" size=".1"/></body>
      <body pos=".2 0 0"><freejoint/><geom type="sphere" size=".1"/></body>
      </worldbody></mujoco>''')
    query = SelfCollisionQuery(model)
    actual, unknown = evaluate(FastSelfCollision(query), model.qpos0[None], True)
    expected = query.query(model.qpos0)
    assert unknown == 0
    assert [r['pair_id'] for r in actual[0]] == [r['pair_id'] for r in expected]
    np.testing.assert_allclose(actual[0][0]['direction'], expected[0]['direction'])


def test_default_off_selects_original_rows_and_window_cache(monkeypatch):
    from tests.test_fast_backend import _objective
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS, fast_self_collision=False))
    assert runtime_options.RuntimeOptions().fast_self_collision
    model = mesh_pair()
    query = SelfCollisionQuery(model)
    chosen = acquisition_self_collision(query)
    assert type(chosen) is BatchedSelfCollision
    poses = model.qpos0[None]
    actual, unknown = evaluate(chosen, poses, True, .002)
    expected, failures = evaluate(BatchedSelfCollision(query, workers=1), poses, True, .002)
    assert unknown == failures
    for a, b in zip(actual[0], expected[0]):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    obj, _, _ = _objective(temporal=True, extra=True)
    obj.self_collision = SelfCollisionQuery(obj.model)
    obj.evaluation_backend = 'cpu'
    obj.prepare_window_evaluation()
    assert type(obj._window_collision_template) is BatchedSelfCollision
    assert obj._window_self_cache is not None


def test_enabled_band_is_window_only_including_same_size_chunks(monkeypatch):
    from tests.test_fast_backend import _objective
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS, fast_self_collision=True))
    obj, poses, _ = _objective(temporal=True, extra=True)
    obj.self_collision = SelfCollisionQuery(obj.model)
    obj.evaluation_backend = 'cpu'
    obj.prepare_window_evaluation()
    assert type(obj._window_collision_template) is FastSelfCollision
    assert not hasattr(obj, '_window_self_cache')
    obj.evaluate(poses, frames=range(4, 12))
    local = obj._window_views[4, 12][0]
    assert type(local._self_batch) is FastSelfCollision
    obj.evaluate(poses)
    assert type(obj._self_batch) is BatchedSelfCollision
    chunk = obj.for_frames(slice(4, 12))
    chunk.evaluate_frames(poses[4:12], True, obj.envelope_weight[4:12])
    assert type(chunk._self_batch) is BatchedSelfCollision
    assert type(acquisition_self_collision(obj.self_collision, window=True)) is FastSelfCollision


@pytest.mark.parametrize('jacobian', [False, True])
def test_whole_trajectory_band_switch_keeps_cost_and_normal_equations(monkeypatch, jacobian):
    from tests.test_fast_backend import _objective
    obj, poses, _ = _objective(temporal=True, extra=True)
    obj.self_collision = SelfCollisionQuery(obj.model)
    obj.evaluation_backend = 'cpu'
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS, fast_self_collision=False))
    before = obj.evaluate(poses, jacobian)
    obj.__dict__.pop('_self_batch')
    monkeypatch.setattr(runtime_options, 'OPTIONS', replace(runtime_options.OPTIONS, fast_self_collision=True))
    after = obj.evaluate(poses, jacobian)
    assert type(obj._self_batch) is BatchedSelfCollision
    assert before[0].cost == after[0].cost
    for name in ('frame_cost', 'gradient', 'hessian') if jacobian else ('frame_cost',):
        np.testing.assert_array_equal(getattr(before[0], name), getattr(after[0], name))
    np.testing.assert_array_equal(before[1], after[1])
    if jacobian:
        for name in ('data', 'indices', 'indptr'):
            np.testing.assert_array_equal(getattr(before[2], name), getattr(after[2], name))
    assert before[3] == after[3]


def test_saved_g1_poses_preserve_all_penetrations_and_physical_active_pair_set():
    root = Path(__file__).resolve().parents[1]
    run = root / 'out/speedup_20261001/runs/trumans_2023-02-19_22-45-43_0_957_g1_h_ctrl'
    if not (run / 'prepared.npy').exists():
        pytest.skip('Saved acquisition clip is unavailable')
    prepared = np.load(run / 'prepared.npy', allow_pickle=True).item()
    model = mujoco.MjModel.from_xml_path(prepared['model_path'])
    query = SelfCollisionQuery(model)
    batch = FastSelfCollision(query)
    np.testing.assert_array_equal(np.stack((batch.first, batch.second), 1), query.pairs)
    for name in ('before_mesh_repair', 'fixed_reference'):
        indices = np.unique(np.r_[np.linspace(0, 956, 24).astype(int), 203, 214, 365])
        poses = np.load(run / (name + '.npz'))['qpos'][indices]
        rows, unknown = evaluate(batch, poses, True, target=.002)
        assert unknown == 0
        for pose, actual in zip(poses, rows):
            indexed = {r['pair_id']: r for r in actual}
            expected = query.query(pose)
            old_active = {r['pair_id'] for r in expected if r['clearance'] < .002}
            fast_active = {r['pair_id'] for r in actual if r.get('hinge_clearance', r['clearance']) < .002}
            assert old_active == fast_active
            for row in expected:
                if row['clearance'] < .002:
                    hinge = indexed[row['pair_id']]
                    assert hinge.get('hinge_clearance', hinge['clearance']) == pytest.approx(
                        row['clearance'], abs=2e-8)
                    np.testing.assert_allclose(hinge['_jacobian'], query.jacobian(row, query.data),
                                               atol=2e-12, rtol=1e-10)
                if row['clearance'] < -1e-7:
                    assert row['pair_id'] in indexed
                    assert indexed[row['pair_id']]['clearance'] == pytest.approx(row['clearance'], abs=2e-8)
            # An exhaustive native query verifies the broad phase against
            # every eligible pair, independent of the old plane-gap cutoff.
            batch.data.qpos[:] = pose
            mujoco.mj_kinematics(batch.distance_model, batch.data)
            physical = set()
            for i, (a, b) in enumerate(query.pairs):
                measured = batch._distance(int(a), int(b), batch.activation_distance)
                if measured is not None:
                    physical.add(int(query.pair_ids[i]))
            assert set(indexed) == physical


@pytest.mark.parametrize('activation_distance', [.013, .018, .030])
@pytest.mark.parametrize('gap', [.001, .0025, .008, .0129, .014, .0179, .0181, .0299, .0301])
def test_activation_band_keeps_physical_distance_and_unchanged_hinge_cost(gap, activation_distance):
    from types import SimpleNamespace
    from omniretargeting._fast.rows import _Accumulator
    from omniretargeting._fast.terms import FrameTerms
    model = mesh_pair()
    query = SelfCollisionQuery(model)
    poses = model.qpos0[None].copy()
    poses[0, 7] = .2 + gap
    kin = TrajectoryKinematics(model, np.arange(model.nv)).evaluate(poses, False)
    batch = FastSelfCollision(query, activation_distance=activation_distance)
    rows, unknown = batch.rows(poses, kin, True, .002)
    assert unknown == 0
    assert len(rows[0]) == int(gap < activation_distance)
    obj = SimpleNamespace(self_collision=query, evaluation_backend='cpu', _self_batch=batch,
                          kin=kin, coords=SimpleNamespace(cols=np.arange(model.nv)),
                          cfg=dict(self_clearance_m=.002, collision_sigma_m=.01), clearance_scale=1.)
    acc = _Accumulator(1, model.nv, True)
    FrameTerms._self_collision_terms(obj, poses, acc, {})
    obj._self_batch = FastSelfCollision(query, activation_distance=0.)
    physical = _Accumulator(1, model.nv, True)
    FrameTerms._self_collision_terms(obj, poses, physical, {})
    assert acc.cost == pytest.approx(physical.cost, abs=1e-12)
    if rows[0]:
        row = rows[0][0]
        assert row['clearance'] == pytest.approx(gap, abs=2e-8)
        if gap > .002:
            assert not row.get('hinge_clearance', row['clearance']) < .002
            np.testing.assert_array_equal(acc.gradient, 0.)
            np.testing.assert_array_equal(acc.hessian, 0.)
        else:
            assert row['hinge_clearance'] == pytest.approx(row['clearance'], abs=2e-8)
    else:
        np.testing.assert_array_equal(acc.hessian, 0.)


def test_band_disabled_reproduces_unbanded_pair_set_and_invalid_band_rejected():
    model = mesh_pair()
    query = SelfCollisionQuery(model)
    poses = model.qpos0[None].copy()
    poses[0, 7] = .208
    unbanded = FastSelfCollision(query, activation_distance=0.)
    assert not evaluate(unbanded, poses, True, .002)[0][0]
    banded = FastSelfCollision(query)
    assert evaluate(banded, poses, True, .002)[0][0][0]['clearance'] > .002
    for invalid in (-1., np.nan, np.inf):
        with pytest.raises(ValueError, match='activation_distance'):
            FastSelfCollision(query, activation_distance=invalid)


def test_band_cutoff_participates_in_pose_cache_and_detects_boundary_approach():
    model = mesh_pair()
    batch = FastSelfCollision(SelfCollisionQuery(model), activation_distance=0.)
    poses = model.qpos0[None].copy()
    poses[0, 7] = .208
    kin = TrajectoryKinematics(model, np.arange(model.nv)).evaluate(poses, False)
    assert not batch.rows(poses, kin, False, .002)[0][0]
    batch.activation_distance = .013
    assert batch.rows(poses, kin, True, .002)[0][0][0]['clearance'] > .002
    assert not batch.diagnostics['exact_pose_reused']
    changed = poses.copy()
    changed[0, 7] = .212
    assert evaluate(batch, changed, True, .002)[0][0][0]['clearance'] > .002


def test_early_band_row_reproduces_old_hinge_force_and_curvature_exactly():
    from types import SimpleNamespace
    from omniretargeting._fast.rows import _Accumulator
    from omniretargeting._fast.terms import FrameTerms
    model = mesh_pair()
    query = SelfCollisionQuery(model)
    poses = model.qpos0[None].copy()
    poses[0, 7:10] = [.2018, .2018, .015]
    kin = TrajectoryKinematics(model, np.arange(model.nv)).evaluate(poses, False)
    fast = FastSelfCollision(query)
    row = fast.rows(poses, kin, True, .002)[0][0][0]
    assert row['clearance'] > .002
    assert row['hinge_clearance'] < .002
    obj = SimpleNamespace(self_collision=query, evaluation_backend='cpu', kin=kin,
                          coords=SimpleNamespace(cols=np.arange(model.nv)),
                          cfg=dict(self_clearance_m=.002, collision_sigma_m=.01), clearance_scale=1.)
    results = []
    for batch in (BatchedSelfCollision(query, workers=1), fast):
        obj._self_batch = batch
        acc = _Accumulator(1, model.nv, True)
        FrameTerms._self_collision_terms(obj, poses, acc, {})
        results.append(acc)
    assert results[0].cost == results[1].cost > 0
    np.testing.assert_array_equal(results[0].gradient, results[1].gradient)
    np.testing.assert_array_equal(results[0].hessian, results[1].hessian)


def test_positive_band_python_fallback_keeps_old_early_hinge(monkeypatch):
    from omniretargeting._fast import fast_self_planes
    from omniretargeting._fast.batched_self_collision import NativeUnavailable
    def unavailable(*args, **kwargs):
        raise NativeUnavailable('fixture compiler unavailable')
    monkeypatch.setattr(fast_self_planes, '_RestrictedPlanes', unavailable)
    model = mesh_pair()
    query = SelfCollisionQuery(model)
    poses = model.qpos0[None].copy()
    poses[0, 7:10] = [.2018, .2018, .015]
    row = evaluate(FastSelfCollision(query), poses, True, .002)[0][0][0]
    expected = query.query(poses[0])[0]
    assert row['clearance'] > .002
    assert row['hinge_clearance'] == expected['clearance']
    np.testing.assert_array_equal(row['_jacobian'], query.jacobian(expected, query.data))


def test_banded_penetration_keeps_original_material_witness_and_jacobian():
    model = mesh_pair()
    query = SelfCollisionQuery(model)
    poses = model.qpos0[None].copy()
    poses[0, 7:10] = [.16, .035, .022]
    old = evaluate(BatchedSelfCollision(query, workers=1), poses, True, .002)[0][0][0]
    fast = evaluate(FastSelfCollision(query), poses, True, .002)[0][0][0]
    for key in ('clearance', 'points', 'direction', '_jacobian'):
        np.testing.assert_array_equal(fast[key], old[key])


def test_band_consumers_keep_authority_buffers_and_each_other_private():
    model = mesh_pair()
    query = SelfCollisionQuery(model)
    untouched = query.data.qpos.copy()
    first = FastSelfCollision(query)
    evaluate(first, model.qpos0[None], True, .002)
    second = copy(first)
    moved = model.qpos0[None].copy()
    moved[0, 7] = .201
    evaluate(second, moved, True, .002)
    np.testing.assert_array_equal(query.data.qpos, untouched)
    assert first.positive_hinges.query.data is not second.positive_hinges.query.data
    assert first.positive_hinges.query.data is not query.data


def test_unreliable_band_distances_share_authority_batch_and_cache_is_independent(monkeypatch):
    def unreliable(model, data, a, b, maximum, points):
        points[:] = 0.
        return 0.
    monkeypatch.setattr(mujoco, 'mj_geomDistance', unreliable)
    model = mesh_pair()
    query = SelfCollisionQuery(model)
    poses = model.qpos0[None].copy()
    poses[0, 7:10] = [.2018, .2018, .015]
    kin = TrajectoryKinematics(model, np.arange(model.nv)).evaluate(poses, False)
    fast = FastSelfCollision(query)
    first, unknown = fast.rows(poses, kin, True, .002)
    old = evaluate(BatchedSelfCollision(query, workers=1), poses, True, .002)[0][0][0]
    assert unknown == 0 and fast.diagnostics['fallback_pairs'] == 1
    assert '_original_required' not in first[0][0]
    for key in ('clearance', 'points', 'direction', '_jacobian'):
        np.testing.assert_array_equal(first[0][0][key], old[key])
    first[0][0]['points'][:] = 100.
    cached = fast.rows(poses, kin, True, .002)[0][0][0]
    np.testing.assert_array_equal(cached['points'], old['points'])
