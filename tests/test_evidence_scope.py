"""The cell-grid evidence scope equals the bounded nearest-neighbour test."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

import omniretargeting.evidence_scope as evidence_scope
from omniretargeting.evidence_scope import EvidenceScope


def _direct(tree, points, radius):
    distance, _ = tree.query(points, distance_upper_bound=radius * (1 + 1e-9) + 1e-12)
    return distance <= radius


@pytest.mark.parametrize("seed, radius, scale", [(0, 0.15, 1.0), (1, 0.05, 1.0), (2, 0.15, 30.0), (3, 0.3, 0.01)])
def test_scope_equals_direct_query(seed, radius, scale):
    rng = np.random.default_rng(seed)
    evidence = scale * rng.uniform(-0.5, 0.5, size=(300, 3))
    radius *= scale
    tree = cKDTree(evidence)
    scope = EvidenceScope(tree, evidence, radius)
    # Random points, points exactly on and near the radius of an evidence
    # point, and points outside the grid.
    direction = rng.normal(size=(2000, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    anchor = evidence[rng.integers(0, len(evidence), 2000)]
    shell = anchor + direction * radius * (1 + rng.choice([-1e-9, 0.0, 1e-9, -1e-3, 1e-3], 2000))[:, None]
    points = np.concatenate([scale * rng.uniform(-1.5, 1.5, size=(20000, 3)), shell,
                             evidence, scale * np.array([[10.0, 0, 0], [-10.0, 3, 3]])])
    for chunk in np.array_split(points, 7):  # cells are classified across calls
        np.testing.assert_array_equal(scope(chunk), _direct(tree, chunk, radius))
    assert scope(np.zeros((0, 3))).shape == (0,)


def test_coarse_cells_when_the_grid_would_be_large(monkeypatch):
    monkeypatch.setattr(evidence_scope, "MAX_CELLS", 5000)
    rng = np.random.default_rng(4)
    evidence = rng.uniform(-2, 2, size=(100, 3))
    tree = cKDTree(evidence)
    scope = EvidenceScope(tree, evidence, 0.1)
    assert np.prod(scope.counts) <= 5000
    points = rng.uniform(-3, 3, size=(5000, 3))
    np.testing.assert_array_equal(scope(points), _direct(tree, points, 0.1))
