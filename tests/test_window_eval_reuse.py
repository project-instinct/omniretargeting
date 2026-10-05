"""Exact window geometry reuse must preserve witnesses and normal equations."""

from types import SimpleNamespace

import numpy as np
import pytest

from omniretargeting._fast.bulk_geometry import BulkGeometry
from omniretargeting._fast.window_geometry import WindowGeometry
from omniretargeting.self_collision import SelfCollisionQuery
from tests.test_bulk_geometry import _shell_problem
from tests.test_convex_mesh_distance import mesh_pair
from tests.test_fast_backend import _objective


def _bytes_equal(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


def test_value_jacobian_upgrade_preserves_geom_identity_and_collision_witnesses():
    obj, poses, _ = _objective(temporal=True, extra=True)
    obj.self_collision = SelfCollisionQuery(obj.model)
    obj.evaluation_backend = 'cpu'
    obj.prepare_window_evaluation()
    frames = range(4, 24)
    value = obj.evaluate(poses, False, frames=frames)
    local = obj._window_views[frames.start, frames.stop][0]
    arrays = local.kin.geom_xpos, local.kin.geom_xmat
    jacobian = obj.evaluate(poses, True, frames=frames)
    assert local.kin.geom_xpos is arrays[0]
    assert local.kin.geom_xmat is arrays[1]
    assert local._self_batch.diagnostics['exact_pose_reused']
    _bytes_equal(value[0].frame_cost, jacobian[0].frame_cost)
    # A new pose must query its own geometry, and retained equations are owned.
    retained = jacobian[0].hessian.copy()
    changed = poses.copy()
    changed[10, 7] += .001
    obj.evaluate(changed, False, frames=frames)
    assert local.kin.geom_xpos is not arrays[0]
    _bytes_equal(jacobian[0].hessian, retained)
    again = obj.evaluate(poses, True, frames=frames)
    assert local.kin.geom_xpos is arrays[0]
    _bytes_equal(again[0].hessian, jacobian[0].hessian)


def test_fixed_neighbour_and_weight_changes_are_not_cached_as_objective_values():
    obj, poses, _ = _objective(temporal=True, extra=True)
    frames = range(5, 17)
    before = obj.evaluate(poses, True, frames=frames)
    moved = poses.copy()
    moved[4, 7] += .02
    obj.envelope_weight *= 1.5
    obj.proximities[0]['scale'] *= 1.3
    after = obj.evaluate(moved, True, frames=frames)
    assert before[1].tobytes() != after[1].tobytes()
    obj.__dict__.pop('_window_views')
    fresh = obj.evaluate(moved, True, frames=frames)
    for name in ('frame_cost', 'gradient', 'hessian'):
        _bytes_equal(getattr(after[0], name), getattr(fresh[0], name))
    _bytes_equal(after[1], fresh[1])
    for name in ('data', 'indices', 'indptr'):
        _bytes_equal(getattr(after[2], name), getattr(fresh[2], name))
    assert after[3] == fresh[3]


@pytest.mark.parametrize('moving', [False, True])
def test_deepest_cache_keeps_unknown_bits_owned_arrays_and_transform_changes(moving):
    obj, poses, sdf = _shell_problem(unknown=True)
    obj.kin.evaluate(poses, False)
    geometry = WindowGeometry(obj)
    rotation = np.tile(np.eye(3), (obj.frames, 1, 1)) if moving else None
    translation = np.zeros((obj.frames, 3)) if moving else None
    scales = np.linspace(.9, 1.1, obj.frames) if moving else None
    expected = geometry._deepest(sdf, rotation, translation, scales)
    first = geometry.deepest(sdf, rotation, translation, scales)
    first[0].fill(-123.)
    second = geometry.deepest(sdf, rotation, translation, scales)
    for actual, reference in zip(second, expected):
        _bytes_equal(actual, reference)
    assert geometry.diagnostics['hits'] == 1
    if moving:
        translation[4, 0] += .004
        scales[7] += .01
    obj.cfg['scene_clearance_m'] += .003
    changed = geometry.deepest(sdf, rotation, translation, scales)
    fresh = WindowGeometry(obj).deepest(sdf, rotation, translation, scales)
    for actual, reference in zip(changed, fresh):
        _bytes_equal(actual, reference)
    assert geometry.diagnostics['misses'] == 2


def test_proximity_cache_tracks_in_place_normals_and_geom_transforms():
    obj, poses, _ = _objective(extra=True)
    obj.kin.evaluate(poses, False)
    geometry = WindowGeometry(obj)
    prox = obj.proximities[0]
    expected = geometry.proximity_points(prox)
    geometry.proximity_points(prox).fill(-123.)
    _bytes_equal(geometry.proximity_points(prox), expected)
    prox['normals'][0, 0] *= -1
    changed = geometry.proximity_points(prox)
    _bytes_equal(changed, WindowGeometry(obj).proximity_points(prox))
    moved = poses.copy()
    moved[2, 7] += .04
    obj.kin.evaluate(moved, False)
    _bytes_equal(geometry.proximity_points(prox), WindowGeometry(obj).proximity_points(prox))


def test_replacing_the_immutable_shell_pack_invalidates_geometry_and_native_points():
    obj, poses, sdf = _shell_problem()
    obj.kin.evaluate(poses, False)
    geometry = WindowGeometry(obj)
    geometry.deepest(sdf, None, None)
    obj.shell_local = obj.shell_local.copy()
    obj.shell_local[:, 2] += .01
    actual = geometry.deepest(sdf, None, None)
    expected = WindowGeometry(obj).deepest(sdf, None, None)
    for a, b in zip(actual, expected):
        _bytes_equal(a, b)


def test_directional_certificate_retains_full_batch_ties_and_winner_changes():
    from omniretargeting._fast.window_native import support
    model = mesh_pair()
    helper = BulkGeometry(SimpleNamespace(model=model))
    reference = BulkGeometry(SimpleNamespace(model=model))
    helper.window_support_reuse = True
    rng = np.random.default_rng(787)
    initial = np.vstack((rng.normal(size=(4096, 3)),
                         [[0., 0., 0.], [1., 0., 0.], [1., 1e-17, 0.],
                          [1., 1., 0.], [1., 1., 1.]]))
    for amplitude in (0., 1e-6, .001, .1, 1., 0.):
        directions = initial + rng.normal(scale=amplitude, size=initial.shape)
        for geom in range(model.ngeom):
            actual = helper.support_unique(geom, directions, fallback=support)
            expected = reference.support_unique(geom, directions, fallback=support)
            _bytes_equal(actual, expected)
    assert helper.diagnostics['support_reused'] > 12000


def test_window_cache_capacities_are_bounded():
    obj, poses, _ = _objective(extra=True)
    frames = range(3, 18)
    for i in range(18):
        moved = poses.copy()
        moved[10, 7] += i * .001
        obj.evaluate(moved, False, frames=frames)
    local = obj._window_views[frames.start, frames.stop][0]
    assert len(local._window_assemblies) == 4
    assert len(local._window_geometry.results) <= 12
    assert len(local._window_geometry.packs) == 1


def test_frozen_solve_guard_bounds_reuse_keeps_owned_canonical_buffers(monkeypatch):
    from omniretargeting._fast.legacy_window_qp import WindowObjective
    from tests.test_fast_legacy_window import problem
    obj, poses = problem(extra=True)
    local = WindowObjective(obj, poses, range(5, 17))
    guard = local.source_shape
    original = guard.guard.qp_bounds
    calls = []

    def measured(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(guard.guard, 'qp_bounds', measured)
    part = poses[local.span]
    expected = guard.qp_bounds(part)
    local._cache_guard_bounds = True
    first = guard.qp_bounds(part)
    for family in first:
        for matrix, lo, hi in family:
            matrix.data.fill(-123.)
            lo.fill(-123.)
            hi.fill(-123.)
    actual = guard.qp_bounds(part)
    assert len(calls) == 2
    for before, after in zip(expected, actual):
        for a, b in zip(before, after):
            for name in ('data', 'indices', 'indptr'):
                _bytes_equal(getattr(a[0], name), getattr(b[0], name))
            _bytes_equal(a[1], b[1])
            _bytes_equal(a[2], b[2])
