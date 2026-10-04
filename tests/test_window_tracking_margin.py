"""Original-geometry positive margins belong solely to the tracking merit."""

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.qp_assembly import quadratic
from omniretargeting._fast.qp_types import PriorityMerit, TRACKING_LEVEL, priority_level
from omniretargeting._fast.window_assembly import solve_window_model
from omniretargeting._fast.window_local import assess_window
from tests.test_fast_qp_assembly import small_model

KEY = ('clearance_margin', 0, 'static', 1)
NEW_KEY = ('clearance_margin', 0, 'self', 1, 2)


def model_with_margin(*, gradient=0., weight=2., matrix=-1., bound=-.2):
    return small_model([(sparse.csr_matrix([[matrix]]), np.array([bound]),
                          np.array([weight]), [KEY])], gradient=gradient)


def merit(slacks, *, cost=None, valid=True, penetration=0.):
    details = {} if cost is None else dict(tracking_margin_cost=cost)
    return PriorityMerit(valid, (penetration, 0., 0., 0.), (1e-7,) * 4, slacks, details)


def test_tracking_slot_never_enters_four_high_levels_or_penalty():
    model = model_with_margin()
    assert priority_level(KEY) == TRACKING_LEVEL == 4
    assert model.soft.levels == [4]
    np.testing.assert_array_equal(model.soft.level_values({KEY: .2}), np.zeros(4))
    assert model.soft.penalty({KEY: .2}, [1e9, 1e8, 1e7, 1e6]) == 0.
    assert model.soft.tracking_penalty({KEY: .2}) == pytest.approx(.4)
    assert len(merit({KEY: .2}).levels) == 4


def test_qp_tracking_weight_is_only_divided_by_numeric_model_scale():
    model = model_with_margin(weight=.2)
    model.scale = 31.
    _, c = quadratic(model, .001)
    assert c[-1] == pytest.approx(.2 / 31.)
    mandatory = np.ones(len(model.program.b), bool)
    first, _ = solve_window_model(model, .001, mandatory, penetration_multiplier=1.)
    escalated, _ = solve_window_model(model, .001, mandatory, penetration_multiplier=1000.)
    assert first is not None and escalated is not None
    # min d^2 + .2*max(.2-d,0) (plus unchanged LM regularizer).
    assert first.step[0] == pytest.approx(.1 / 1.001, abs=1e-6)
    np.testing.assert_array_equal(first.step, escalated.step)
    assert len(first.certificate['coefficients']) == 4


def test_margin_gain_can_pay_for_tracking_rise_and_enters_both_lm_decreases():
    model = model_with_margin()
    before, after = merit({KEY: .2}, cost=.4), merit({KEY: .1}, cost=.2)
    accepted, ratio, details = assess_window(model, np.array([.1]), before, after, 0., .01, np.ones(4))
    assert accepted and ratio == pytest.approx(1.)
    assert details['tracking_actual_decrease'] == pytest.approx(.19)
    assert details['tracking_predicted_decrease'] == pytest.approx(.19)
    assert details['actual_merit_decrease'] == pytest.approx(.19)  # No double charging.
    assert details['predicted_merit_decrease'] == pytest.approx(.19)
    assert details['improved_level'] is None


def test_new_candidate_key_is_charged_even_without_any_prepared_qp_row():
    model = model_with_margin(gradient=-1.)
    before = merit({KEY: .2}, cost=.4)
    after = merit({KEY: .1, NEW_KEY: .5}, cost=1.2)
    accepted, ratio, details = assess_window(model, np.array([.1]), before, after, 1., .9, np.ones(4))
    assert not accepted and ratio < 0.
    assert details['tracking_actual_decrease'] == pytest.approx(-.7)
    assert details['tracking_predicted_decrease'] == pytest.approx(.39)


def test_missing_complete_cost_with_unknown_key_fails_closed_even_on_penetration_progress():
    model = model_with_margin()
    before = merit({KEY: .2}, penetration=.01)
    after = merit({KEY: .1, NEW_KEY: .001}, penetration=0.)
    accepted, _, details = assess_window(model, np.array([.1]), before, after, 1., .9, np.ones(4))
    assert not accepted and not details['tracking_margin_cost_valid']


def test_known_keys_keep_compatibility_without_complete_cost_field():
    model = model_with_margin()
    accepted, ratio, _ = assess_window(model, np.array([.1]), merit({KEY: .2}), merit({KEY: .1}),
                                        0., .01, np.ones(4))
    assert accepted and ratio == pytest.approx(1.)


@pytest.mark.parametrize('cost', [float('nan'), float('inf'), -.1, 'invalid'])
def test_invalid_full_cost_cannot_authorize_acceptance(cost):
    model = model_with_margin()
    accepted, _, details = assess_window(model, np.array([.1]), merit({KEY: .2}, cost=.4),
                                          merit({KEY: .1}, cost=cost), 1., .9, np.ones(4))
    assert not accepted and not details['tracking_margin_cost_valid']


def test_margin_descent_cannot_override_hard_penetration_or_contact_rejection():
    model = model_with_margin()
    accepted, _, _ = assess_window(model, np.array([.1]), merit({KEY: .2}, cost=.4),
                                    merit({KEY: .1}, cost=.2, valid=False), 1., .9, np.ones(4))
    assert not accepted


def test_penetration_progress_still_outranks_tracking_margin_loss():
    model = model_with_margin()
    accepted, _, details = assess_window(model, np.array([.1]), merit({KEY: .2}, cost=.4, penetration=.01),
                                          merit({KEY: .3}, cost=.6), 0., 1., np.ones(4))
    assert accepted and details['improved_level'] == 0
    assert details['tracking_actual_decrease'] < 0


def test_legacy_priority_solver_keeps_tracking_epigraph_out_of_strict_levels():
    from omniretargeting._fast.qp_assembly import solve_model
    model = model_with_margin(weight=.2)
    result = solve_model(model, .001, priority=True)
    assert result is not None
    assert result.step[0] == pytest.approx(.1 / 1.001, abs=1e-6)
    assert len(result.certificate['predicted_reductions']['levels']) == 4
