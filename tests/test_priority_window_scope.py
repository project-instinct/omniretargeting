"""Real geometry window scopes, cache accounting and nonlinear rejection evidence."""

from types import SimpleNamespace

import numpy as np
import pytest

from omniretargeting._fast.priority_clearance import PriorityClearance
from omniretargeting._fast.priority_geometry import PriorityGeometry, body_depths, window_body_depths
from tests.test_fast_backend import _objective
from tests.test_priority_window import case


def test_real_begin_trial_local_and_parent_commit_never_query_full_trajectory(monkeypatch):
    from omniretargeting._fast.mesh_repair import OriginalGeometry
    obj, poses, _ = _objective()
    provider = PriorityGeometry(obj)
    coordinator = PriorityClearance(obj, provider, poses)
    obj.priority_clearance, obj.candidate_guards = coordinator, [coordinator]
    frames = range(2, 6)
    requests, actual_frames = [], []
    original_measure = OriginalGeometry.measure
    original_frames = OriginalGeometry.measure_frames

    def measure(geometry, q, *, frames=None, **kwargs):
        assert frames is not None, 'An iteration/commit queried the whole trajectory'
        assert set(frames).issubset({2, 3, 4, 5})
        requests.append(tuple(frames))
        return original_measure(geometry, q, frames=frames, **kwargs)

    def measured_frames(geometry, q, selected, **kwargs):
        actual_frames.extend(selected)
        assert set(selected).issubset({2, 3, 4, 5})
        return original_frames(geometry, q, selected, **kwargs)

    monkeypatch.setattr(OriginalGeometry, 'measure', measure)
    monkeypatch.setattr(OriginalGeometry, 'measure_frames', measured_frames)
    child = coordinator.fork_window(frames)
    child.begin_iteration(poses, frames=frames)
    q = poses.copy()
    q[3, 2] += .0001  # One changed frame; all fixed neighbors are exact cache hits.
    merit = child.evaluate_trial(q, frames=frames)
    assert merit['valid']
    scope = merit['details']['query_scope']
    assert scope['requested_frames'] == list(frames)
    assert scope['queried_frames'] == [3]
    assert scope['queried_frame_count'] == 1 and scope['cached_frame_count'] == 3
    repeated = child.evaluate_trial(q, frames=frames)['details']['query_scope']
    assert repeated['adapter_cache_hit'] and repeated['queried_frame_count'] == 0
    child.advance_local(q, frames=frames)
    coordinator.commit(q, frames=frames, transactions=child.export_transactions(frames=frames))
    assert coordinator.epoch == 1
    # Private and parent caches each query the changed frame once. Neither
    # re-queries the fixed three window neighbors or any outside frame.
    assert actual_frames == [3, 3]
    assert requests and set(requests) == {tuple(frames)}
    np.testing.assert_array_equal(coordinator.current[:2], poses[:2])
    np.testing.assert_array_equal(coordinator.current[6:], poses[6:])


def test_r4_window_maxima_count_weld_owners_once_and_reject_unknown():
    model = SimpleNamespace(nbody=4, body_weldid=np.array([0, 1, 1, 3]), geom_bodyid=np.array([2, 3]))
    measured = dict(depths={(2, 'self', 0, 1): .01}, failures=[], complete=True,
                    reliable=True, frame_complete=np.ones(1, bool), queried_frames=(2,),
                    max_depth=np.array([[0., .01, .01, .01]]), body_reliable=np.ones((1, 4), bool))
    value = window_body_depths(model, 5, range(2, 3), measured)
    expected, _ = body_depths(model, 5, measured)
    np.testing.assert_array_equal(value, expected)
    assert value.sum() == pytest.approx(.02)
    measured['body_reliable'][0, 2] = False
    with pytest.raises(ValueError, match='unknown'):
        window_body_depths(model, 5, range(2, 3), measured)


def test_collision_rejection_reports_key_body_and_original_tolerance_excess():
    from omniretargeting._fast.priority_window import WINDOW_PENETRATION_STEP_TOLERANCE
    c, _ = case()
    q = c.current.copy()
    q[3, 0] -= .002
    report = c.evaluate_trial(q, frames=range(2, 5))['details']
    rows = report['rejection_details']
    row = next(r for r in rows if r['family'] == 'key_nonregression')
    assert row['key'] == (3, 'static', 0) and row['bodies'] == [1]
    assert row['frame'] == 3 and row['excess_m'] == pytest.approx(.002 - WINDOW_PENETRATION_STEP_TOLERANCE)
    assert row['tolerance_m'] == WINDOW_PENETRATION_STEP_TOLERANCE


def test_hard_box_rejection_identifies_frame_coordinate_and_excess():
    c, _ = case()
    q = c.current.copy()
    q[3, 1] = 1.002
    report = c.evaluate_trial(q, frames=range(2, 5))['details']
    row = next(r for r in report['rejection_details'] if r['guard'] == 'joint_box')
    assert row['frame'] == 3 and row['qpos_column'] == 1
    assert row['excess'] == pytest.approx(.002 - 1e-9)


def test_temporal_rejection_identifies_crossing_stencil_and_exact_budget():
    from omniretargeting._fast.continuity import CorrectionContinuity, ROUND_OFF
    from omniretargeting._fast.priority_rejections import hard_details
    obj, poses, _ = _objective()
    fixed = np.repeat(poses[:1], len(poses), axis=0)
    guard = CorrectionContinuity(obj.coords, fixed)
    obj.candidate_guards, obj.source_shape = [guard], None
    q = fixed.copy()
    q[3, obj.coords.qi[0]] += .002
    frames = range(3, 4)
    c = SimpleNamespace(objective=obj)
    rows = hard_details(c, q, frames, ['CorrectionContinuity'])
    row = next(r for r in rows if r['family'] == 'joints_order_1')
    assert row['stencil'] in ([2, 3], [3, 4])
    assert row['qpos_column'] == int(obj.coords.qi[0])
    assert row['budget'] == 0.
    assert row['tolerance'] == ROUND_OFF
    assert row['excess'] == pytest.approx(.002 - ROUND_OFF)
    assert row['bodies']
