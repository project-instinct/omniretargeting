"""Existential norm epigraph elimination retains complete cones and constants."""
import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.repair_canonical import canonical_data
from omniretargeting._fast.repair_epigraph import eliminate_epigraph
from omniretargeting._fast.repair_restoration import _trust_terms
from tests.test_fast_restoration_trust import _problem, Bounds


@pytest.mark.parametrize('seed', range(5))
def test_cone_slacks_match_at_the_original_upper_epigraph_bounds(seed):
    bounds, projection, movement, radius = _problem(seed)
    variables = projection.shape[1]
    original = canonical_data(bounds, projection, *_trust_terms(projection, movement), radius)
    reduced = eliminate_epigraph(original, variables)
    scalar_count = len(original['c']) - variables
    linear = reduced['dims'].nonneg
    budgets = original['b'][linear:original['dims'].nonneg]
    for delta in np.random.default_rng(seed).normal(size=(10, variables)):
        complete = np.r_[delta, budgets]
        first = original['b'] - original['A'] @ complete
        second = reduced['b'] - reduced['A'] @ delta
        assert first[:linear].tobytes() == second[:linear].tobytes()
        assert not first[linear:linear + scalar_count].any()
        assert first[linear + scalar_count:].tobytes() == second[linear:].tobytes()
    assert reduced['P'].data.tobytes() == original['P'].data.tobytes()
    assert reduced['dims'].soc == original['dims'].soc


def test_fixed_violated_cone_is_preserved():
    projection = sparse.eye(3, format='csc')[:, [1]]
    bounds = Bounds(sparse.csr_matrix((0, 3)), np.empty(0), np.empty(0),
                    [(sparse.csr_matrix((3, 3)), np.array([[1., 2., 3.]]), np.array([.5]))])
    original = canonical_data(bounds, projection, *_trust_terms(projection, np.zeros(3)), .1)
    reduced = eliminate_epigraph(original, 1)
    assert reduced['dims'].soc == [3, 4]
    assert np.linalg.norm(reduced['b'][-3:]) > reduced['b'][-4]
