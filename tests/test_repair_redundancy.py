"""Projected-box restoration omits only constraints of fixed coordinates."""

import itertools

import numpy as np
from scipy import sparse

from omniretargeting._fast.repair_redundancy import RestorationBounds


def test_interval_matches_all_corners_of_the_selected_box():
    rng = np.random.default_rng(7)
    projection = sparse.eye(6, format="csc")[:, [1, 4]]
    bounds = RestorationBounds(6, projection, 2, 3)
    low, high = -rng.random(6), rng.random(6)
    bounds.set_box(low, high)
    matrix = sparse.csr_matrix(rng.normal(size=(5, 6)))
    bottom, top = bounds._range(matrix)
    corners = []
    for choice in itertools.product((0, 1), repeat=2):
        step = np.zeros(6)
        step[[1, 4]] = [high[j] if side else low[j] for j, side in zip([1, 4], choice)]
        corners.append(matrix @ step)
    np.testing.assert_allclose(bottom, np.min(corners, axis=0), rtol=0, atol=1e-15)
    np.testing.assert_allclose(top, np.max(corners, axis=0), rtol=0, atol=1e-15)


def test_redundant_projected_cone_drops_but_fixed_infeasible_cone_survives():
    projection = sparse.eye(2, format="csc")[:, [0]]
    bounds = RestorationBounds(2, projection, 1, 2)
    bounds.set_box([-1., -1.], [1., 1.])
    bounds.norm(np.array([[[.01, 10.], [0., 10.], [0., 10.]]]),
                np.array([[.1, .1, .1]]), .5)
    # A guard can append sparse norm rows directly; retain violations even
    # with no selected coefficient and no feasible point in the box.
    bounds.norms.append((sparse.csr_matrix((3, 2)), np.array([[1., 0., 0.]]), np.array([.5])))
    cones = list(bounds.cones())
    assert cones[0][1].shape == (0, 3)
    np.testing.assert_array_equal(cones[1][1], [[1., 0., 0.]])
