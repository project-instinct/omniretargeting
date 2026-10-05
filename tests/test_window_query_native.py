"""Bounds prune work while preserving original first extrema and witnesses."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting._fast.bulk_geometry import BulkGeometry
from omniretargeting._fast.sdf import GridSDF
from omniretargeting._fast.window_query_native import BLOCK, WindowQueries, grid_bounds
from tests.test_bulk_geometry import _shell_problem


def _equal(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


def test_support_bvh_preserves_first_winner_and_original_margin():
    from omniretargeting._fast.window_native import support
    from omniretargeting._sequence.proximity import _hull_vertices
    rng = np.random.default_rng(101)
    vertices = rng.normal(size=(1500, 3))
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    xml = '<mujoco><asset><mesh name="m" vertex="' + ' '.join(map(str, vertices.ravel()))
    xml += '"/></asset><worldbody><geom type="mesh" mesh="m"/></worldbody></mujoco>'
    model = mujoco.MjModel.from_xml_string(xml)
    obj = SimpleNamespace(model=model)
    reference, accelerated = BulkGeometry(obj), BulkGeometry(obj)
    accelerated.window_support_reuse = True
    accelerated.window_queries = WindowQueries(obj)
    hull = _hull_vertices(model, 0)
    directions = np.vstack((rng.normal(size=(2000, 3)), hull[:10], np.zeros((3, 3)),
                            [[1., 0., 0.], [-0., -0., -0.]]))
    for delta in (0., 1e-7, .03, -.1):
        moved = directions+delta
        _equal(accelerated.support_unique(0, moved, fallback=support),
               reference.support_unique(0, moved, fallback=support))
        # BVH second maximum must reproduce the complete scan's gap too.
        normalized = moved/np.maximum(np.linalg.norm(moved, axis=1, keepdims=True), 1e-12)
        scan_choice, scan_gap = reference._mesh_support(0, moved)
        choices, margins = np.empty(len(moved), np.int64), np.empty(len(moved))
        accelerated.window_queries.support(0, hull, normalized, choices, margins,
                                          64*np.finfo(float).eps*(np.abs(hull).max()+1))
        _equal(hull[choices], scan_choice)
        scores = normalized @ hull.T
        ordered = np.sort(scores, axis=1)
        assert np.all(margins >= 0)
        # Native scan/BLAS arithmetic can differ within their original slack.
        np.testing.assert_allclose(margins, ordered[:, -1]-ordered[:, -2], atol=1e-14)
        assert not scan_gap[:-3].all()


@pytest.mark.parametrize('moving', [False, True])
@pytest.mark.parametrize('scale', [1., 1.07, -.5])
def test_bounded_shells_and_old_witness_seed_match_every_reference_byte(moving, scale):
    obj, poses, sdf = _shell_problem(unknown=True)
    reference, accelerated = BulkGeometry(obj), BulkGeometry(obj)
    accelerated.window_queries = WindowQueries(obj)
    rng = np.random.default_rng(14)
    rotation = Rotation.from_rotvec(rng.normal(scale=.1, size=(obj.frames, 3))).as_matrix() if moving else None
    translation = rng.normal(scale=.01, size=(obj.frames, 3)) if moving else None
    scales = np.full(obj.frames, scale)
    for delta in (0., .001, .02, -.003):
        moved = poses.copy()
        moved[:, 7] += delta
        obj.kin.evaluate(moved, False)
        before = reference.deepest(sdf, rotation, translation, scales)
        after = accelerated.deepest(sdf, rotation, translation, scales)
        for a, b in zip(after, before):
            _equal(a, b)
    assert accelerated.window_queries.diagnostics['shell_interpolated'] > 0


def test_structured_field_prunes_interpolations_without_changing_the_minimum():
    obj, poses, _ = _shell_problem()
    obj.kin.evaluate(poses, False)
    shape = (101, 101, 201)
    values = np.broadcast_to((np.arange(shape[2])*.01-.7).astype(np.float32), shape).copy()
    sdf = GridSDF([-.5, -.5, -.5], .01, values, np.zeros((*shape, 3), np.float32))
    reference, accelerated = BulkGeometry(obj), BulkGeometry(obj)
    accelerated.window_queries = WindowQueries(obj)
    for a, b in zip(accelerated.deepest(sdf, None, None), reference.deepest(sdf, None, None)):
        _equal(a, b)
    assert accelerated.window_queries.diagnostics['shell_pruned'] > 0


def test_seed_at_later_equal_sample_restores_first_minimum():
    obj, poses, _ = _shell_problem()
    obj.kin.evaluate(poses, False)
    values = np.zeros((31, 31, 31), np.float32)
    sdf = GridSDF([-1, -1, -1], .1, values, np.zeros((*values.shape, 3), np.float32))
    reference, accelerated = BulkGeometry(obj), BulkGeometry(obj)
    accelerated.window_queries = WindowQueries(obj)
    for geom in range(len(obj.shell_geoms)):
        accelerated.window_queries.seeds[sdf, geom] = np.full(obj.frames, 3, np.int64)
    for a, b in zip(accelerated.deepest(sdf, None, None), reference.deepest(sdf, None, None)):
        _equal(a, b)


@pytest.mark.parametrize('magnitude', [1e-30, 1., 1e30])
def test_grid_block_bound_covers_interpolation_rounding_and_unknown_nodes(magnitude):
    rng = np.random.default_rng(445)
    values = (rng.normal(size=(18, 23, 27))*magnitude).astype(np.float32)
    values[4, 8, 12] = np.nan
    values[9, 13, 17] = np.inf
    values[10, 14, 18] = -np.inf
    sdf = GridSDF([-.2, -.3, -.4], .07, values, np.zeros((*values.shape, 3), np.float32))
    index = rng.uniform(0, sdf.shape-1, size=(20000, 3))
    points = sdf.lower+sdf.spacing*index
    actual = sdf.query(points, gradient=False)
    bound, _ = grid_bounds(sdf)
    # Use the exact coordinate division consumed by the native kernel.
    cells = np.floor((points-sdf.lower)/sdf.spacing).astype(int)//BLOCK
    lower = bound[tuple(cells.T)]
    assert np.all(lower[np.isfinite(actual)] <= actual[np.isfinite(actual)])
