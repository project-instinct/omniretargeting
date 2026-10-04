"""Step tolerance is current-relative; historical clearance is never new credit."""

import numpy as np
import pytest

from omniretargeting._fast.priority_window import WINDOW_PENETRATION_STEP_TOLERANCE as TAU
from tests.test_priority_window import case


@pytest.mark.parametrize('increase,accepted', [(TAU, True), (TAU + 1e-9, False)])
def test_current_key_body_and_sum_boundary(increase, accepted):
    c, _ = case()
    q = c.current.copy()
    q[2, 0] = -increase
    report = c.evaluate_trial(q, frames=range(2, 3))['details']
    assert report['accepted'] is accepted
    assert bool(report['penetration_regressions']) is not accepted
    assert bool(report['body_regressions']) is not accepted
    assert report['total_penetration_regression'] is not accepted
    assert report['penetration_step_tolerance_m'] == .0005
    assert report['penetration_step_reference'] == 'current_committed'


def test_centimeter_improvement_can_accompany_ten_micron_regression():
    c, _ = case()
    q = c.current.copy()
    q[2] = .02  # Contact remains within its original allowance.
    q[3, 0] = -1e-5
    report = c.evaluate_trial(q, frames=range(2, 4))['details']
    assert report['accepted'] and not report['rejection_details']
    assert report['actual_reductions'][0] == pytest.approx(.02 - 1e-5)


def test_total_tolerance_is_once_per_window_not_per_key():
    c, _ = case()
    q = c.current.copy()
    q[2:4, 0] = -.0003
    report = c.evaluate_trial(q, frames=range(2, 4))['details']
    assert not report['accepted']
    assert not report['penetration_regressions'] and not report['body_regressions']
    row, = report['rejection_details']
    assert row['family'] == 'window_total'
    assert row['excess_m'] == pytest.approx(.0001)
    assert row['tolerance_m'] == TAU


def test_current_reference_not_historical_best_and_qp_half_reserve():
    c, _ = case()
    frames = range(2, 3)
    for value in (-.0004, -.0008):
        q = c.current.copy()
        q[2, 0] = value
        assert c.accepts(q, frames=frames)
        c.commit(q, frames=frames)
    assert c.current_depth[2, 1] == pytest.approx(.0508)
    assert c.best_depth[2, 1] == .05
    assert c.best_keys[(2, 'static', 0)] == .05
    linear, _ = c.qp_bounds(c.current, frames=frames)
    matrix, _, upper = linear[0]
    assert upper[0] == pytest.approx(TAU / 2)
    assert (matrix @ np.array([-TAU / 2, 0.]))[0] <= upper[0] + 1e-15
    assert not c.spent and not c.yields


@pytest.mark.parametrize('new_depth,accepted', [(TAU, True), (TAU + 1e-9, False)])
def test_new_key_on_previously_clear_body_has_same_absolute_cap(new_depth, accepted):
    c, provider = case()
    c.contacts.clear()
    # Start this frame at zero penetration, with no historical/current key.
    c.current[2, 0] = .05
    c.current_depth[2] = 0.
    c.best_depth[2] = 0.
    c.current_measurement = provider.measure(c.current)
    del c.current_measurement['depths'][(2, 'static', 0)]
    c.best_keys.pop((2, 'static', 0))
    original = provider.measure

    def measure(poses, **kwargs):
        measured = original(poses, **kwargs)
        if poses[2, 0] < .05:
            # Exact measured boundary, independent of pose-subtraction roundoff.
            measured['depths'][(2, 'static', 0)] = new_depth
            measured['rows'][0][1]['clearance'] = -new_depth
        return measured

    provider.measure = measure
    q = c.current.copy()
    q[2, 0] -= new_depth
    report = c.evaluate_trial(q, frames=range(2, 3))['details']
    assert report['accepted'] is accepted
    if not accepted:
        assert report['penetration_regressions'][0][2] == 0.


def test_new_key_cannot_hide_behind_an_already_penetrating_body():
    c, provider = case()
    c.contacts.clear()
    original = provider.measure

    def measure(poses, **kwargs):
        measured = original(poses, **kwargs)
        # Second witness on the same welded body; dominant body maximum drops.
        if poses[2, 0] > 0:
            measured['depths'][(2, 'moving', 0, 0)] = TAU + 1e-6
        return measured

    provider.measure = measure
    q = c.current.copy()
    q[2, 0] = .01
    report = c.evaluate_trial(q, frames=range(2, 3))['details']
    assert not report['accepted'] and report['penetration_regressions']
    assert not report['body_regressions'] and not report['total_penetration_regression']


def test_rebound_and_recovery_never_mint_cross_body_credit():
    c, _ = case()
    frames = range(2, 3)
    q = c.current.copy()
    q[2] = .01
    c.commit(q, frames=frames)
    best = c.best_depth.copy()
    q[2, 0] = .0096  # Approved temporary 0.4 mm regression, no contact yield.
    c.commit(q, frames=frames)
    np.testing.assert_array_equal(c.best_depth, best)
    assert not c.spent and not c.yields
    for x, y in [(.0099, .008), (.0102, .0089)]:
        trial = q.copy()
        trial[2] = x, y
        report = c.evaluate_trial(trial, frames=frames)['details']
        assert report['actual_reductions'][0] > 0
        assert not report['accepted'] and report['unfunded_contacts']
        with pytest.raises(ValueError, match='unfunded_contact_yield'):
            c.commit(trial, frames=frames)
        assert not c.spent and not c.yields and c.epoch == 2
    trial[2] = .0102, .0091  # 0.1 mm yield funded by 0.2 mm below best, not 0.6 mm recovery.
    c.commit(trial, frames=frames)
    record, = c.yields
    assert record['available_m'] == pytest.approx(.0002)
    assert record['gap_m'] == pytest.approx(.0001)
    assert sum(c.spent.values()) == pytest.approx(.0001)


def test_tolerance_never_authorizes_unknown_geometry_or_joint_violation():
    c, provider = case()
    q = c.current.copy()
    q[2, 0] = -TAU / 2
    c.begin_iteration(c.current, frames=range(2, 3))
    provider.fail = True
    assert not c.accepts(q, frames=range(2, 3))
    provider.fail = False
    q[2, 1] = 1. + 1e-8
    report = c.evaluate_trial(q, frames=range(2, 3))['details']
    assert not report['accepted'] and report['hard_failures'] == ['joint_box']


def test_parent_net_commit_gets_one_tolerance_not_one_per_private_step():
    c, _ = case()
    frames = range(2, 3)
    child = c.fork_window(frames)
    q = child.current.copy()
    q[2, 0] = -.0004
    child.advance_local(q, frames=frames)
    q[2, 0] = -.0008
    assert not child.accepts(q, frames=frames)
    with pytest.raises(ValueError, match='penetration_regression'):
        child.advance_local(q, frames=frames)
    assert child.epoch == 1
    with pytest.raises(ValueError, match='penetration_regression'):
        c.commit(q, frames=frames)
    assert c.epoch == 0 and not c.spent and not c.yields
    np.testing.assert_array_equal(c.current, 0.)
