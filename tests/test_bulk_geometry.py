"""CPU acquisition kernels retain supports, unknown grids and original rows."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting._fast import bulk_geometry
from omniretargeting._fast.bulk_geometry import BulkGeometry
from omniretargeting._fast.sdf import GridSDF
from omniretargeting._fast.terms import FrameTerms
from omniretargeting._sequence.proximity import support_points
from tests.test_convex_mesh_distance import mesh_pair
from tests.test_fast_backend import _objective


def _grid(shape=(17, 15, 19), unknown=False):
    rng = np.random.default_rng(315)
    values = rng.normal(size=shape).astype(np.float32)
    if unknown:
        values[0, 0, 0] = np.nan
        values[4, 3, 8] = np.nan
        values[8, 5, 11] = np.inf
    return GridSDF([-.6, -.5, -.4], .1, values, np.zeros((*shape, 3)))


@pytest.mark.parametrize('shape', [(17, 15, 19), (1, 4, 7)])
@pytest.mark.parametrize('unknown', [False, True])
def test_grid_arithmetic_boundaries_and_unknown_corners(shape, unknown):
    sdf = _grid(shape, unknown and shape[0] > 1)
    rng = np.random.default_rng(29)
    index = rng.uniform(-1, np.array(shape), size=(4096, 3))
    nodes = np.array(np.meshgrid(*[np.arange(n) for n in shape], indexing='ij')).reshape(3, -1).T
    points = sdf.lower+sdf.spacing*np.vstack((index, nodes, [[np.nan, 1, 2], [np.inf, 1, 2]]))
    helper = BulkGeometry(SimpleNamespace())
    expected = sdf.query(points, gradient=False)
    actual = helper.query(sdf, points)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype == np.float32
    # Multidimensional/noncontiguous queries have the same scalar shape.
    np.testing.assert_array_equal(helper.query(sdf, points[:60].reshape(5, 4, 3, 3)[:, ::2]),
                                  sdf.query(points[:60].reshape(5, 4, 3, 3)[:, ::2], False))


def test_zero_weight_unknown_corner_stays_unknown():
    sdf = GridSDF([0, 0, 0], 1., np.ones((3, 3, 3)), np.zeros((3, 3, 3, 3)))
    sdf.clearance[1, 1, 1] = np.nan
    helper = BulkGeometry(SimpleNamespace())
    assert np.isnan(helper.query(sdf, [[0, 0, 0]])[0])
    np.testing.assert_array_equal(helper.query(sdf, [[0, 0, 0], [2, 2, 2]]),
                                  sdf.query([[0, 0, 0], [2, 2, 2]], False))


def test_invalid_native_grid_shapes_fail_before_pointer_access():
    helper = BulkGeometry(SimpleNamespace())
    with pytest.raises(ValueError):
        helper.query(_grid(), [1., 2.])
    with pytest.raises(ValueError):
        helper.query(GridSDF([0, 0, 0], 1., np.ones((2, 2)), np.zeros((2, 2, 3))), [[0, 0, 0]])


@pytest.mark.parametrize('tied', [False, True])
def test_mesh_supports_match_full_blas_batch_including_ties(tied):
    model = mesh_pair()
    rng = np.random.default_rng(83)
    normals = rng.normal(size=(11, 25, 3))
    normals[:, 4:10] = normals[:, :1]
    if tied:
        normals[:, 0] = [1., 1., 0.]
        normals[:, 10:15] = [0., 0., 1.]
        normals[:, 24] = 0.
    prox = dict(geoms=np.zeros(25, int), normals=normals)
    helper = BulkGeometry(SimpleNamespace(model=model))
    directions = normals.reshape(-1, 3)
    actual = helper.support(prox, 0, directions)
    np.testing.assert_array_equal(actual, support_points(model, 0, directions))
    if tied:
        assert helper.diagnostics['support_fallbacks'] > 0
    assert helper.diagnostics['support_queries'] < len(directions)
    assert helper.group(prox, 0) is helper.group(prox, 0)


def _shell_problem(unknown=False):
    obj, poses, _ = _objective(temporal=True, extra=True)
    obj.kin.evaluate(poses, True)
    return obj, poses, _grid(unknown=unknown)


@pytest.mark.parametrize('moving', [False, True])
@pytest.mark.parametrize('unknown', [False, True])
def test_fused_shell_matches_first_minimum_with_rotation_and_scale(moving, unknown):
    obj, _, sdf = _shell_problem(unknown)
    rng = np.random.default_rng(8)
    rotation = Rotation.from_rotvec(rng.normal(scale=.1, size=(obj.frames, 3))).as_matrix() if moving else None
    translation = rng.normal(scale=.04, size=(obj.frames, 3)) if moving else None
    scales = rng.uniform(.9, 1.1, obj.frames) if moving else None
    expected = FrameTerms._deepest_samples(obj, sdf, rotation, translation, scales)
    helper = BulkGeometry(obj)
    actual = helper.deepest(sdf, rotation, translation, scales)
    for before, after in zip(expected, actual):
        np.testing.assert_array_equal(after, before)
    assert helper.diagnostics['shell_queries'] > 0
    assert sdf.__class__ is GridSDF


def test_shell_ties_and_all_unknown_keep_first_sample():
    obj, _, sdf = _shell_problem()
    helper = BulkGeometry(obj)
    for value in (-.01, np.nan):
        sdf.clearance.fill(value)
        expected = FrameTerms._deepest_samples(obj, sdf, None, None)
        actual = helper.deepest(sdf, None, None)
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)


def test_known_centre_with_every_shell_sample_outside_stays_unknown():
    sdf = GridSDF([0, 0, 0], .1, np.ones((2, 2, 2)), np.zeros((2, 2, 2, 3)))
    obj = SimpleNamespace(frames=1, model=SimpleNamespace(ngeom=1),
        cfg=dict(scene_clearance_m=.002), shell_geoms=[0], shell_radius=np.array([3.]),
        shell_index=np.array([[0, 1]]), shell_local=np.array([[2., 0, 0], [-2., 0, 0]]),
        kin=SimpleNamespace(geom_xpos=np.full((1, 1, 3), .05), geom_xmat=np.eye(3)[None, None]))
    expected = FrameTerms._deepest_samples(obj, sdf, None, None)
    actual = BulkGeometry(obj).deepest(sdf, None, None)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    assert np.isinf(actual[0][0, 0])
    np.testing.assert_array_equal(actual[3], [[2.05, .05, .05]])


def test_analytic_fields_retain_their_own_query():
    obj, _, _ = _shell_problem()

    class Sphere:
        def query(self, points, gradient=False):
            return np.linalg.norm(points, axis=-1)-.9
    sdf = Sphere()
    expected = FrameTerms._deepest_samples(obj, sdf, None, None)
    actual = BulkGeometry(obj).deepest(sdf, None, None)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize('float32_scale', [False, True])
def test_mixed_precision_object_fields_retain_reference_rounding(float32_scale):
    obj, _, sdf = _shell_problem()
    rotation = np.broadcast_to(np.eye(3, dtype=np.float32), (obj.frames, 3, 3))
    translation = np.zeros((obj.frames, 3), np.float32)
    scales = np.full(obj.frames, 1.07, np.float32 if float32_scale else np.float64)
    actual = BulkGeometry(obj).deepest(sdf, rotation, translation, scales)
    expected = FrameTerms._deepest_samples(obj, sdf, rotation, translation, scales)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)




def test_default_off_and_native_unavailable_retain_reference(monkeypatch):
    pass
    obj, _, _ = _objective()
    pass
    calls = []
    def unavailable():
        calls.append(True)
        raise bulk_geometry.NativeUnavailable('test compiler unavailable')
    monkeypatch.setattr(bulk_geometry, 'native_library', unavailable)
    bulk_geometry.prepare(obj)
    bulk_geometry.prepare(obj)
    assert len(calls) == 1
    assert '_proximity_support' not in obj.__dict__
    assert '_deepest_samples' not in obj.__dict__






def test_parallel_native_evaluation_keeps_rows_owned(monkeypatch):
    obj, poses, _ = _shell_problem()
    obj.scene_sdf = _grid()
    expected = obj.evaluate(poses, True)
    monkeypatch.setenv('OMNIRETARGETING_CPUS', '3')
    pass
    obj.start_workers()
    try:
        actual = obj.evaluate(poses, True)
        before = actual[0].gradient.copy()
        obj.evaluate(poses+.001, False)
        np.testing.assert_array_equal(actual[0].gradient, before)
    finally:
        obj.stop_workers()
    np.testing.assert_array_equal(actual[0].hessian, expected[0].hessian)
    np.testing.assert_array_equal(actual[0].gradient, expected[0].gradient)
    assert actual[0].cost == expected[0].cost




def test_window_supports_preserve_input_and_existing_tie_fallback():
    model = mesh_pair()
    directions = np.vstack((np.random.default_rng(3).normal(size=(51, 3)),
                            [[1., 1., 0.], [0., 0., 1.], [0., 0., 0.]]))
    retained = directions.copy()
    from omniretargeting._fast.window_native import support
    helper = BulkGeometry(SimpleNamespace(model=model))
    np.testing.assert_array_equal(helper.support_unique(0, directions, fallback=support),
                                  support(model, 0, directions))
    np.testing.assert_array_equal(directions, retained)
