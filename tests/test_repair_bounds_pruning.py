import numpy as np
from scipy import sparse

from omniretargeting._fast.repair_limits import LinearBounds


def _bounds(rng, size=6):
    bounds = LinearBounds(size)
    lower, upper = -rng.uniform(0, .1, size), rng.uniform(0, .1, size)
    bounds.set_box(lower, upper)
    matrix = sparse.random(40, size, density=.4, random_state=1, format="csr")
    bounds.add(matrix, -rng.uniform(0, .2, 40), rng.uniform(0, .2, 40))
    residual = rng.normal(0, .02, (20, 3))
    jacobian = sparse.random(60, size, density=.4, random_state=2, format="csr")
    bounds.norms.append((jacobian, residual, rng.uniform(.01, .2, 20)))
    return bounds, lower, upper, matrix, jacobian


def test_box_implied_rows_and_cones_are_the_only_ones_omitted():
    rng = np.random.default_rng(0)
    bounds, lower, upper, matrix, jacobian = _bounds(rng)
    packed, low, high = bounds.packed()
    kept = packed.shape[0] - bounds.size
    assert 0 < kept < matrix.shape[0]
    # The box rows themselves are always present.
    np.testing.assert_array_equal(np.abs(packed[:bounds.size].toarray()) > 0, np.eye(bounds.size) > 0)
    cones = list(bounds.cones())
    assert 0 < len(cones[0][1]) < 20
    # Omitted rows and cones hold at every vertex of the box (and inside it).
    corners = np.array(np.meshgrid(*zip(lower, upper), indexing="ij")).reshape(len(lower), -1).T
    samples = np.concatenate([corners, rng.uniform(lower, upper, (500, len(lower)))])
    row_lower, row_upper = bounds.lower[1], bounds.upper[1]
    values = samples @ matrix.T.toarray()
    violated = ((values < row_lower) | (values > row_upper)).any(axis=0)
    kept_rows = packed[bounds.size:].toarray()
    scale = 1 / np.maximum(np.maximum(np.abs(row_lower), np.abs(row_upper)), 1e-4)
    for row in np.flatnonzero(violated):
        assert np.any(np.all(np.isclose(kept_rows, scale[row] * matrix[row].toarray()), axis=1))
    residual, allowed = bounds.norms[0][1], bounds.norms[0][2]
    norms = np.linalg.norm((samples @ jacobian.T.toarray()).reshape(len(samples), -1, 3) + residual, axis=2)
    needed = (norms > allowed).any(axis=0)
    kept_cones = {tuple(r) for r in cones[0][1]}
    assert {tuple(r) for r in residual[needed]} <= kept_cones


def test_identical_rows_merge_into_their_intersection():
    bounds = LinearBounds(3)
    stencil = sparse.csr_matrix([[1., -2., 1.], [0., 1., -1.]])
    bounds.add(stencil, [-.3, -.1], [.2, .4])
    bounds.add(sparse.csr_matrix([[2., 0., 0.]]), -1., 1.)
    bounds.add(stencil, [-.1, -.5], [.5, .3])
    matrix, lower, upper = bounds.packed()
    scale = 1 / np.maximum(np.maximum(np.abs([-.1, -.1, -1.]), np.abs([.2, .3, 1.])), 1e-4)
    np.testing.assert_allclose(matrix.toarray(), scale[:, None] * np.array(
        [[1., -2., 1.], [0., 1., -1.], [2., 0., 0.]]))
    np.testing.assert_allclose(lower, scale * [-.1, -.1, -1.])
    np.testing.assert_allclose(upper, scale * [.2, .3, 1.])
