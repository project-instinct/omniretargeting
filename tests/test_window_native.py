"""Native short-window grids keep unknown masks and first-minimum choices."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting._fast.window_native import NativeGrid, deepest, support
from omniretargeting._fast.sdf import GridSDF
from tests.test_fast_backend import _objective


def _grid():
    rng = np.random.default_rng(444)
    values = rng.normal(scale=.05, size=(31, 29, 27)).astype(np.float32)
    values[3, 5, 7] = np.nan
    return GridSDF([-1., -1., -1.], .1, values, np.zeros((*values.shape, 3), np.float32))


def test_native_grid_is_bit_identical_to_scipy_at_nodes_edges_unknowns():
    grid = _grid()
    rng = np.random.default_rng(87)
    points = rng.uniform(-1.3, 2.2, size=(10000, 3))
    nodes = np.array(np.meshgrid(*(np.arange(n) for n in grid.shape), indexing='ij')).reshape(3, -1).T
    points = np.vstack((points, grid.lower + grid.spacing * nodes,
                        [[np.nan, 0., 0.], [np.inf, 0., 0.], [-np.inf, 0., 0.]]))
    expected = grid.query(points, gradient=False)
    actual = NativeGrid(grid).query(points, gradient=False)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))


@pytest.mark.parametrize('moving', [False, True])
def test_native_shell_first_minimum_witness_and_unknowns_equal_reference(moving):
    obj, poses, _ = _objective(temporal=True)
    obj.kin.evaluate(poses, False)
    rotation = Rotation.from_rotvec(np.random.default_rng(7).normal(scale=.03, size=(obj.frames, 3))).as_matrix() if moving else None
    translation = np.tile([.02, -.01, .03], (obj.frames, 1)) if moving else None
    scales = np.linspace(.9, 1.1, obj.frames) if moving else None
    grid = _grid()
    for values in (grid.clearance, np.zeros_like(grid.clearance), np.full_like(grid.clearance, np.nan)):
        field = GridSDF(grid.lower, grid.spacing, values, grid.directions)
        expected = obj._deepest_samples(field, rotation, translation, scales)
        actual = deepest(obj, field, rotation, translation, scales)
        for a, e in zip(actual, expected):
            np.testing.assert_array_equal(a, e)


def test_certified_support_winners_and_ties_use_original_vertex_indices():
    from tests.test_convex_mesh_distance import mesh_pair
    from omniretargeting._sequence.proximity import support_points
    model = mesh_pair()
    directions = np.vstack((np.random.default_rng(44).normal(size=(800, 3)),
                            [[0., 0., 0.], [1., 0., 0.], [1., 1e-17, 0.], [0., 0., 1.]]))
    for geom in range(2):
        np.testing.assert_array_equal(support(model, geom, directions), support_points(model, geom, directions))


def test_window_proximity_native_support_preserves_bytes_and_group_cache():
    from types import SimpleNamespace
    from tests.test_convex_mesh_distance import mesh_pair
    from omniretargeting._fast.window_geometry import WindowGeometry
    model, frames = mesh_pair(), 19
    rng = np.random.default_rng(111)
    kin = SimpleNamespace(
        geom_xpos=rng.normal(size=(frames, model.ngeom, 3)),
        geom_xmat=Rotation.from_rotvec(rng.normal(size=(frames * model.ngeom, 3))).as_matrix().reshape(
            frames, model.ngeom, 3, 3))
    objective = SimpleNamespace(model=model, frames=frames, kin=kin)
    native, reference = WindowGeometry(objective), WindowGeometry(objective)
    reference.native = False
    assert native.native
    normals = rng.normal(size=(frames, 12, 3))
    normals[:, 1:4] = normals[:, :1]  # duplicated within each frame
    normals[:, 5] = 0.  # support tie: retain the original first vertex
    prox = dict(normals=normals, geoms=np.tile(np.arange(2), 6))
    for _ in range(3):
        expected, actual = reference.proximity_points(prox), native.proximity_points(prox)
        assert expected.tobytes() == actual.tobytes()
        kin.geom_xpos += .001
        kin.geom_xmat = Rotation.from_rotvec(rng.normal(size=(frames * model.ngeom, 3))).as_matrix().reshape(
            frames, model.ngeom, 3, 3)
    assert len(native.groups) == len(reference.groups) == 2
