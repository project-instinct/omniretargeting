import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting.anchor_events import place_event


def anchor(center):
    center = np.array(center, float)
    footprint = np.array([[-.1, -.1, 0], [-.1, .1, 0], [.1, -.1, 0], [.1, .1, 0]])
    return dict(translation=center, observed=footprint + center,
                robot_points=footprint * .5 + center,
                surface_rotation=np.eye(3), surface_scale=1., surface_translation=np.zeros(3))


def test_new_contact_moves_only_inside_measured_footprint_held_stays_put():
    held, new = anchor([0, 0, 0]), anchor([1, 0, 0])
    before = held['translation'].copy()
    change, report = place_event([new], [held], reach_bounds={(0, 1): .97})
    np.testing.assert_allclose(change[0], [-.03, 0, 0], atol=2e-5)
    np.testing.assert_array_equal(held['translation'], before)
    assert report['remaining_reach_violation_m'][0] < 1e-7
    # An impossible reach never spends more than the observed 5 cm margin.
    change, report = place_event([new], [held], reach_bounds={(0, 1): .8})
    np.testing.assert_allclose(change[0], [-.05, 0, 0], atol=2e-5)
    assert report['remaining_reach_violation_m'][0] > .149


def test_simultaneous_contacts_jointly_share_available_margin_on_rotated_surface():
    a, b = anchor([-.5, 0, 0]), anchor([.5, 0, 0])
    turn = Rotation.from_euler('xy', [.5, .8]).as_matrix()
    for c in (a, b):
        c['surface_rotation'] = turn
        c['surface_translation'] = np.array([2, 3, 4])
    change, report = place_event([a, b], [], reach_bounds={(0, 1): .94})
    np.testing.assert_allclose(change, [[.03, 0, 0], [-.03, 0, 0]], atol=2e-5)
    assert report['status'] == 'placed'


def test_nonplanar_patch_never_gains_translation_through_its_interior():
    a = anchor([0, 0, 0]); a['observed'][0, 2] = .1
    held = anchor([1, 0, 0])
    change, _ = place_event([a], [held], reach_bounds={(0, 1): .8})
    np.testing.assert_allclose(change, 0, atol=1e-7)


def test_footprint_qhull_rejects_is_a_segment(monkeypatch):
    """qhull merges nearly coincident points more coarsely than the rank test."""
    import cvxpy as cp
    from scipy.spatial import QhullError

    import omniretargeting.anchor_events as events

    def reject(points):
        raise QhullError("QH6214 qhull input error: not enough points(2) to construct initial simplex")

    monkeypatch.setattr(events, "ConvexHull", reject)
    observed = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1e-6, 0.], [0., 1e-6, 0.]])
    delta = cp.Variable(3)
    rows, report = events.footprint_constraints(delta, observed, observed[:1])
    assert report["rank"] == 1 and report["translation_freedom"]
    problem = cp.Problem(cp.Maximize(delta[0]), rows)
    problem.solve()
    # Along the segment the anchor may move to its far end; across it, not at all.
    assert delta.value[0] == pytest.approx(1., abs=1e-6)
