"""A positive lower bound skips futile weight trials, without accepting a step."""

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.priority_bounds import box_level_lower_bounds
from omniretargeting._fast.qp_assembly import quadratic, solve_model
from omniretargeting._fast.qp_reduction import restrict_model
from tests.test_fast_qp_assembly import small_model


def _positive_model(copies=1):
    return small_model([(sparse.csr_matrix(-np.ones((copies, 1))), np.full(copies, -.2),
                         np.full(copies, 3.), [('penetration', 0, 1)] * copies)],
                        gradient=100., upper=.1)


def test_box_bound_is_conservative_and_duplicate_witnesses_charge_once():
    for count in (1, 10):
        bound = box_level_lower_bounds(_positive_model(count))
        assert 0. < bound[0] < .3
        assert bound[0] == pytest.approx(.3, abs=1e-13)
        np.testing.assert_array_equal(bound[1:], 0.)


def test_cancellation_and_unknown_domain_do_not_prove_a_positive_level():
    model = small_model([(sparse.csr_matrix([[1.]]), np.array([np.nextafter(1., 0.)]),
                         np.ones(1), [('penetration', 0, 1)])], lower=1., upper=1.)
    np.testing.assert_array_equal(box_level_lower_bounds(model), 0.)
    model.step_lower[:] = -np.inf
    np.testing.assert_array_equal(box_level_lower_bounds(model), 0.)


def test_restricted_coordinates_never_reuse_the_original_box_for_a_proof():
    model = _positive_model()
    P, c = quadratic(model, .001)
    submodel, _, _ = restrict_model(model, P, c, -sparse.eye(1), np.zeros(1))
    np.testing.assert_array_equal(box_level_lower_bounds(submodel), 0.)
    submodel, _, _ = restrict_model(model, P, c, -sparse.eye(1), np.zeros(1),
                                     lower=np.array([-.1]), upper=np.array([1.]))
    assert box_level_lower_bounds(submodel)[0] == pytest.approx(.3, abs=1e-13)


def test_positive_bound_starts_at_strict_lp_and_still_requires_certificate(monkeypatch):
    from omniretargeting._fast import priority_qp
    original, calls = priority_qp.solve_program, []
    def solve(program, P, c, **kwargs):
        calls.append(P.nnz)
        return original(program, P, c, **kwargs)
    monkeypatch.setattr(priority_qp, 'solve_program', solve)
    model = _positive_model()
    result = solve_model(model, .001, priority=True)
    assert result is not None and result.certificate['certified']
    assert calls[0] == 0
    assert result.certificate['weighted_attempts'] == 0
    assert result.certificate['box_lower_bounds'][0] > 0.
    monkeypatch.setattr(priority_qp, 'solve_program', lambda *args, **kwargs: None)
    assert solve_model(_positive_model(), .001, priority=True) is None
