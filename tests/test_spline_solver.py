"""Nonlinear acceptance and source-time regression of cubic corrections."""
from contextlib import ExitStack
import time

import numpy as np

from omniretargeting._fast.continuity import chatter_reports
from omniretargeting._fast.spline_geometry import SplineGeometry
from omniretargeting._fast.spline_solver import SplineSolve
from omniretargeting._fast.spline_pipeline import local_controls, local_windows
from tests.test_fast_whole_canonical import problem


def test_real_all_frame_solve_descends_without_rebasing_reference_or_chatter():
    objective, q = problem()
    saved = q.copy()
    with ExitStack() as resources:
        geometry = SplineGeometry(objective, (), None, resources)
        state = SplineSolve(objective, geometry, deadline=time.monotonic() + 15)
        before = objective.evaluate(state.poses, False)[0].cost
        state.optimize(3)
        after = objective.evaluate(state.poses, False)[0].cost
        assert state.accepted > 0 and after < before
        assert not chatter_reports(objective.coords, saved, state.poses)[1].any()
        assert state.spline.count == len(q)
        np.testing.assert_array_equal(state.spline.reference, saved)
        np.testing.assert_array_equal(objective.reference, saved)
        assert all(row['original_frames'] == len(q) for row in state.history)


def test_expired_budget_returns_current_accepted_curve_without_full_restart():
    objective, q = problem()
    with ExitStack() as resources:
        geometry = SplineGeometry(objective, (), None, resources)
        state = SplineSolve(objective, geometry, deadline=time.monotonic() - 1)
        result = state.optimize(12)
        np.testing.assert_array_equal(result, q)
        assert state.report()['budget_exhausted']
        assert state.accepted == state.conic_calls == 0
        assert state.report()['full_rate_fallback_count'] == 0


def test_shared_weight_change_requires_a_fresh_model_before_the_next_step(monkeypatch):
    objective, q = problem()
    snapshots = []
    evaluate = objective.evaluate

    def observed(poses, with_jacobian=True):
        if with_jacobian:
            snapshots.append(objective.envelope_weight.copy())
        return evaluate(poses, with_jacobian)

    monkeypatch.setattr(objective, 'evaluate', observed)
    updates = []

    def update(motion, phase):
        if updates:
            return False
        objective.envelope_weight[2] *= 10.
        updates.append((motion.copy(), phase))
        return True

    with ExitStack() as resources:
        geometry = SplineGeometry(objective, (), None, resources)
        state = SplineSolve(objective, geometry, deadline=time.monotonic() + 15.)
        state.optimize(3, weight_updater=update)
        assert updates and len(snapshots) >= 2
        assert snapshots[1][2] == 10 * snapshots[0][2]
        assert state.history[0]['new_weight_model_required_before_next_step']
        np.testing.assert_array_equal(state.spline.reference, q)


def test_source_fast_impulse_stays_in_reference_even_between_control_times():
    objective, q = problem()
    # Source-authored pulse is deliberately narrower than initial knot spacing.
    q = q.copy()
    q[9, -1] += .4
    objective.reference = q
    with ExitStack() as resources:
        geometry = SplineGeometry(objective, (), None, resources)
        state = SplineSolve(objective, geometry)
        np.testing.assert_array_equal(state.poses, q)
        refined, a = state.spline.insert_knots([.285, .325], state.a)
        np.testing.assert_array_equal(refined.embed(a), q)
        assert refined.embed(a)[9, -1] - refined.embed(a)[8, -1] > .3


def test_local_refinement_closure_keeps_complete_filter_and_difference_support():
    objective, q = problem()
    # This fixture otherwise has independently randomized surface contacts at
    # every frame, already saturating the full-rate basis at construction.
    objective.surfaces = []
    with ExitStack() as resources:
        geometry = SplineGeometry(objective, (), None, resources)
        state = SplineSolve(objective, geometry, adaptive=True)
        focus = np.zeros(len(q), bool)
        focus[17] = True
        assert state.refine(focus)
        windows = local_windows(state.spline, focus)
        assert windows == [(0, len(q))]  # Four cubic spans + Gaussian halo + differences.
        controls = local_controls(state.spline, *windows[0])
        assert len(controls) == state.spline.controls
        assert state.refinements[0]['inherited_curve']
        np.testing.assert_array_equal(state.poses, q)


def test_guard_curvature_recovery_shrinks_steps_without_inserting_knots(monkeypatch):
    objective, q = problem()
    objective.surfaces = []
    lower, upper = objective.lower.copy(), objective.upper.copy()
    monkeypatch.setattr(objective.source_shape, 'accepts', lambda _candidate: False)
    with ExitStack() as resources:
        geometry = SplineGeometry(objective, (), None, resources)
        state = SplineSolve(objective, geometry, adaptive=True, deadline=time.monotonic() + 15.)
        limits = []

        def proposal(model):
            limits.append(np.max(np.abs(model.step_upper)))
            return np.full(model.gradient.size, .001), {'backend': 'synthetic_curvature_probe'}

        monkeypatch.setattr(state, '_solve', proposal)
        controls = state.spline.controls
        state.optimize(2)
        assert len(limits) == 6 and limits[-1] < limits[0]
        assert state.spline.controls == controls and not state.refinements
        assert state.accepted == 0
        assert state.history[-1]['solver']['rejected_guard'] == 'ReferenceShape'
        assert state.history[-1]['recovery'] == 'smaller_increment_trust_region'
        np.testing.assert_array_equal(state.poses, q)
        np.testing.assert_array_equal(objective.lower, lower)
        np.testing.assert_array_equal(objective.upper, upper)
