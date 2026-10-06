"""Real tiny spline LP/QP proposals through exact nonlinear merit/acceptance.

Two scalar coordinates and two original frames; no robot FK or geometry
workers. Analytic signed distances supply independent nonlinear evidence.
Production model assembly, Clarabel hierarchy, guards, chatter, geometry
non-regression, commit and optimize all run without mocking their decisions.
"""

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

import omniretargeting._fast.spline_solver as solver
from omniretargeting._fast.continuity import chatter_reports
from omniretargeting._fast.mesh_repair import OriginalGeometry
from omniretargeting._fast.spline_geometry import SplineGeometry
from omniretargeting._fast.spline_model import prepare_spline_model


class ScalarCoordinates:
    """A Euclidean two-coordinate fixture, with the real coordinate contract."""
    def __init__(self):
        self.model = SimpleNamespace(nq=2)
        self.width = 2
        self.qi = self.di = self.cols = self.scalar_cols = np.arange(2)
        self.inverse = {0: 0, 1: 1}
        self.root_q = self.root_d = None
        self.selector = sparse.eye(2, format='csr')
        self.lower, self.upper = np.full(2, -1.), np.full(2, 1.)

    def bounds(self, poses, angular, translation):
        return (np.maximum(np.full_like(poses, -angular), self.lower - poses),
                np.minimum(np.full_like(poses, angular), self.upper - poses))

    def difference(self, origin, target):
        return target - origin

    def integrate(self, origin, step, alpha):
        return origin + alpha * step


class QuadraticTracking:
    """Independent actual squared cost and its original J.T r moments."""
    def __init__(self, target=(-.005, 0.), weights=(1e6, 1.), quartic=0.):
        self.coords = ScalarCoordinates()
        self.model, self.width = self.coords.model, self.coords.width
        self.frames, self.framerate = 2, 30.
        self.reference = np.zeros((self.frames, self.width))
        self.lower, self.upper = self.coords.lower.copy(), self.coords.upper.copy()
        self.fixed_root = self.source_shape = None
        self.candidate_guards, self.effectors, self.surfaces = [], [], []
        self.cfg = dict(angular_step_rad=.01, translation_step_m=.01, iterations=8)
        self.target, self.weights = np.asarray(target), np.asarray(weights)
        self.quartic = quartic
        self.evaluations = []

    def cost(self, poses):
        return float(np.sum(self.weights * (poses - self.target)**2)
                     + self.quartic * np.sum(poses[:, 1]**4))

    def evaluate(self, poses, with_jacobian=True):
        self.evaluations.append((poses.copy(), with_jacobian))
        diagonal = np.broadcast_to(self.weights, poses.shape).copy()
        diagonal[:, 1] += 4 * self.quartic * poses[:, 1]**2
        hessian = np.array([np.diag(value) for value in diagonal])
        gradient = self.weights * (poses - self.target)
        gradient[:, 1] += 2 * self.quartic * poses[:, 1]**3
        accumulator = SimpleNamespace(cost=self.cost(poses), hessian=hessian, gradient=gradient)
        return accumulator, np.empty(0), sparse.csr_matrix((0, poses.size)), {}, {}


class AnalyticProvider:
    """Known, full-frame nonlinear witness distances; no FK/query backend."""
    def __init__(self, objective, *, depth=.03, stationary=False, curvature=0.,
                 second_curvature=None, new_key_depth=0., query_change=None):
        self.objective, self.width = objective, objective.width
        self.depth, self.stationary, self.curvature = depth, stationary, curvature
        self.second_curvature, self.new_key_depth = second_curvature, new_key_depth
        self.query_change = query_change
        self.calls, self.row_calls = [], 0

    def measure(self, poses, *, jacobians=False):
        self.calls.append((poses.copy(), jacobians))
        rows, depths = [], {}
        for frame, pose in enumerate(poses):
            x, y = pose
            clearance = -self.depth if self.stationary else -self.depth + x - self.curvature * y*y
            jacobian = np.zeros(2) if self.stationary else np.array([1., -2*self.curvature*y])
            entries = [((frame, 'static', 21), clearance, jacobian)]
            if self.second_curvature is not None:
                entries.append(((frame, 'static', 22), -.02 - self.second_curvature*y*y,
                                np.array([0., -2*self.second_curvature*y])))
            if self.new_key_depth and np.any(pose != 0.):
                entries.append(((frame, 'self', 11, 24), -self.new_key_depth, np.zeros(2)))
            for key, phi, jac in entries:
                row = dict(clearance=float(phi), jacobian=jac, sample_id=0)
                rows.append((key, row))
                depths[key] = max(depths.get(key, 0.), -float(phi), 0.)
        result = dict(rows=rows, depths=depths, failures=[], complete=True,
                      frame_complete=np.ones(len(poses), bool), reliable=True,
                      query_scope={'requested_frames': list(range(len(poses)))})
        if self.query_change is not None:
            self.query_change(result, poses)
        return result

    def linear_rows(self, measurement):
        self.row_calls += 1
        return OriginalGeometry.linear_rows(self, measurement)

    def summary(self, measurement):
        return dict(max_depth=max(measurement['depths'].values(), default=0.),
                    complete=measurement.get('complete'), failures=list(measurement['failures']))

    def relinearize_witnesses(self, *args, **kwargs):
        # No material registry in this analytic fixture. Its distance functions
        # already provide the true origin derivative of every modelled key.
        return dict(rows=[], failures=[], witness_ids=[])


def fixture(*, tracking=None, **geometry_options):
    objective = QuadraticTracking() if tracking is None else tracking
    provider = AnalyticProvider(objective, **geometry_options)
    geometry = SplineGeometry.__new__(SplineGeometry)
    geometry.provider = provider
    geometry.current = provider.measure(objective.reference)
    geometry.best = dict(geometry.current['depths'])
    geometry.reference_summary = provider.summary(geometry.current)
    geometry.query_seconds = 0.
    geometry.rejections = geometry.unknown_trials = 0
    geometry.history, geometry.witness_failures = [], []
    geometry.witness_memory = {}
    geometry.witness_batches = geometry.witness_queries = 0
    state = solver.SplineSolve(objective, geometry, identity=True, deadline=time.monotonic() + 15.)
    return state


def prepared(state, *, trust_scale=1.):
    objective, poses = state.objective, state.poses
    return prepare_spline_model(objective, state.spline, state.a, objective.evaluate(poses),
                                chatter=state.chatter, geometry=state.geometry, trust_scale=trust_scale)


def actual_trial(state, model, step, alpha=1.):
    direction = np.zeros(state.a.size)
    direction[model.spline_selected] = step
    candidate = state.spline.embed(state.a + alpha * direction.reshape(state.a.shape))
    before, after = state.objective.cost(state.poses), state.objective.cost(candidate)
    good, measured, bad = state.geometry.accepts(candidate)
    merit = solver.step_merit(model, step, alpha, before, after, measured)
    return candidate, before, after, good, measured, bad, merit


def test_model_has_only_tracking_moments_and_frozen_full_frame_geometry_cache(monkeypatch):
    state = fixture()
    objective = state.objective
    saved_reference, saved_coefficients = state.spline.reference.copy(), state.a.copy()

    def forbidden(*args, **kwargs):
        pytest.fail('Original geometry must not be reintroduced into tracking H/g or cost')

    monkeypatch.setattr(state.geometry, 'residual_rows', forbidden)
    monkeypatch.setattr(state.geometry, 'cost', forbidden)
    model, mapping, selected = prepared(state)
    expected_hessian = sparse.diags(np.tile(objective.weights, objective.frames))
    expected_gradient = np.tile(-objective.weights * objective.target, objective.frames)
    np.testing.assert_array_equal(model.hessian.toarray(), expected_hessian.toarray())
    np.testing.assert_array_equal(model.gradient, expected_gradient)
    assert model.spline_mapping is mapping and model.spline_selected is selected
    assert model.spline_geometry is state.geometry.linearized
    assert len(model.spline_geometry['rows']) == objective.frames
    assert state.geometry.provider.calls[-1][1] is True
    assert not hasattr(model, 'spline_collision_layer')
    count = len(state.geometry.provider.calls)
    step, diagnostics = state._solve(model)
    assert step is not None and diagnostics['certification']['certified']
    assert len(state.geometry.provider.calls) == count  # rows, not another measurement
    assert diagnostics['reused_collision_layer']
    assert model.spline_collision_layer.matches(model.spline_geometry, mapping, selected)
    np.testing.assert_array_equal(state.spline.reference, saved_reference)
    np.testing.assert_array_equal(state.a, saved_coefficients)


def test_real_lp_step_collision_improves_despite_actual_tracking_increase():
    state = fixture()
    model, _, _ = prepared(state)
    step, diagnostics = state._solve(model)
    assert step is not None and diagnostics['conic_calls'] >= 2
    assert diagnostics['clarabel_calls'][0]['stage'] == 'collision'
    assert diagnostics['clarabel_calls'][-1]['stage'] == 'tracking'
    candidate, before, after, good, measured, bad, result = actual_trial(state, model, step)
    accepted, ratio, reason, record = result
    assert good and not bad and solver.projection_domain(state.objective, candidate)
    assert after > before * 8 and np.isfinite(after)
    assert accepted and ratio > 0 and not reason
    # Independently evaluated distances establish the actual higher-level gain.
    expected = np.maximum(.03 - candidate[:, 0], 0.).sum()
    assert expected < .041 and record['collision_after_m'] == pytest.approx(expected)
    assert record['accepted_priority'] == 'collision'
    assert model.tracking_decrease(step) < 0
    assert all(measured['depths'][key] < .03 for key in measured['depths'])


def test_optimize_commits_collision_gain_with_finite_tracking_loss_and_immutable_reference():
    state = fixture()
    saved_reference, saved_root_limits = state.objective.reference.copy(), state.objective.upper.copy()
    before = state.objective.cost(state.poses)
    result = state.optimize(1)
    assert state.accepted == 1 and state.rejected == 0 and state.conic_calls >= 2
    assert state.objective.cost(result) > before
    assert np.max(.03 - result[:, 0]) < .020001
    assert state.history[0]['solver']['nonlinear_merit']['accepted_priority'] == 'collision'
    assert len(state.geometry.history) == 1
    for key, depth in state.geometry.best.items():
        assert depth == pytest.approx(.03 - result[key[0], 0])
    np.testing.assert_array_equal(state.objective.reference, saved_reference)
    np.testing.assert_array_equal(state.spline.reference, saved_reference)
    np.testing.assert_array_equal(state.objective.upper, saved_root_limits)
    assert not chatter_reports(state.objective.coords, saved_reference, result)[1].any()


def test_millions_of_actual_tracking_gain_cannot_buy_nonlinear_collision_regression():
    tracking = QuadraticTracking(target=(-.005, .02), weights=(1e6, 1e10))
    state = fixture(tracking=tracking, curvature=100.2)
    model, _, _ = prepared(state)
    step, diagnostics = state._solve(model)
    assert step is not None and diagnostics['predicted_improvement'] > .019
    candidate, before, after, nonreg, measured, bad, result = actual_trial(state, model, step)
    # The original per-key non-regression tolerance permits this 20 um
    # curvature error; the strict layer must still reject the cost bargain.
    assert nonreg and not bad and before - after > 5e6
    depth = .03 - candidate[:, 0] + 100.2*candidate[:, 1]**2
    assert np.min(depth - .03) > 1e-5
    accepted, _, reason, record = result
    assert not accepted and reason == 'original_geometry_curvature_or_new_witness'
    assert record['collision_after_m'] == pytest.approx(depth.sum())
    assert all(value > .03 for value in measured['depths'].values())
    # Recompute the nonlinear geometry for the half step, not endpoint merit
    # interpolation. It has genuine gain and can be accepted.
    half = actual_trial(state, model, step, .5)[-1]
    assert half[0] and half[3]['accepted_priority'] == 'collision'


def test_optimize_backtracks_tracking_bargain_before_committing_real_collision_gain():
    state = fixture(tracking=QuadraticTracking(target=(-.005, .02), weights=(1e6, 1e10)),
                    curvature=100.2)
    result = state.optimize(1)
    assert state.accepted == 1
    assert state.history[0]['alpha'] == .5
    depths = .03 - result[:, 0] + 100.2*result[:, 1]**2
    assert np.max(depths) < .028
    assert state.history[0]['solver']['nonlinear_merit']['collision_after_m'] == pytest.approx(depths.sum())
    for frame in range(2):
        assert state.geometry.best[(frame, 'static', 21)] == pytest.approx(depths[frame])


class CurvedGuard:
    def __init__(self, cap):
        self.cap, self.rejections = cap, 0

    def qp_bounds(self, poses):
        count = len(poses)
        matrix = sparse.csr_matrix((2*poses[:, 0],
            (np.arange(count), np.arange(count)*2)), shape=(count, poses.size))
        return [(matrix, np.full(count, -np.inf), self.cap**2 - poses[:, 0]**2)], []

    def accepts(self, poses):
        good = bool(np.all(poses[:, 0]**2 <= self.cap**2))
        self.rejections += not good
        return good


def test_collision_gain_does_not_bypass_exact_original_guard():
    tracking = QuadraticTracking()
    guard = CurvedGuard(.003)
    tracking.candidate_guards = [guard]
    state = fixture(tracking=tracking)
    result = state.optimize(1)
    assert state.accepted == 1 and state.history[0]['alpha'] == .25
    assert guard.rejections >= 2 and guard.accepts(result)
    assert state.history[0]['solver']['rejected_guard'] == 'CurvedGuard'
    assert state.history[0]['solver']['nonlinear_merit']['accepted_priority'] == 'collision'
    assert np.max(result[:, 0]) <= .003


def test_collision_gain_does_not_bypass_any_key_original_nonregression():
    state = fixture(tracking=QuadraticTracking(target=(-.005, .02), weights=(1e6, 1e10)),
                    second_curvature=1.)
    model, _, _ = prepared(state)
    step, _ = state._solve(model)
    assert step is not None
    _, _, _, nonreg, _, bad, merit = actual_trial(state, model, step)
    # Aggregate collision improves, but the other original key worsens beyond
    # its unchanged per-key geometry guard. optimize must enforce that guard.
    assert merit[0] and not nonreg and {key[-1] for key in bad} == {22}
    result = state.optimize(1)
    assert state.accepted == 1 and state.history[0]['alpha'] == .5
    assert state.geometry.rejections >= 1
    assert np.max(.02 + result[:, 1]**2) < .02005
    assert all(state.geometry.best[(frame, 'static', 22)] == .02 for frame in range(2))


def test_stationary_collision_lp_allows_only_tracking_descent_and_keeps_depth_ledger():
    state = fixture(tracking=QuadraticTracking(target=(0., .005), weights=(1., 1e6)), stationary=True)
    model, _, _ = prepared(state)
    step, diagnostics = state._solve(model)
    assert step is not None and diagnostics['stage_optima']['collision'] == pytest.approx(.06)
    assert diagnostics['predicted_improvement'] == 0.
    _, before, after, _, _, _, result = actual_trial(state, model, step)
    assert after < before and model.tracking_decrease(step) > 0
    assert result[0] and result[3]['accepted_priority'] == 'tracking_at_collision_tie'
    ledger = dict(state.geometry.best)
    state.optimize(1)
    assert state.accepted == 1 and state.geometry.best == ledger
    assert state.history[0]['solver']['nonlinear_merit']['accepted_priority'] == 'tracking_at_collision_tie'


def test_stationary_collision_with_true_quartic_tracking_increase_is_rejected():
    tracking = QuadraticTracking(target=(0., .005), weights=(1., 1e6), quartic=1e12)
    state = fixture(tracking=tracking, stationary=True)
    model, _, _ = prepared(state)
    step, _ = state._solve(model)
    _, before, after, _, _, _, result = actual_trial(state, model, step)
    assert model.tracking_decrease(step) > 0 and after > before
    assert not result[0] and result[2] == 'collision_layer_stationary'


def test_predicted_collision_gain_with_actual_stationarity_is_rejected_despite_tracking_gain():
    # Current policy conservatively asks for the predicted collision gain to
    # materialize. A tracking tie is allowed only after a stationary LP layer.
    state = fixture(tracking=QuadraticTracking(target=(-.005, .02), weights=(1e6, 1e10)), curvature=100.)
    model, _, _ = prepared(state)
    step, diagnostic = state._solve(model)
    assert diagnostic['predicted_improvement'] > .019
    _, before, after, nonreg, measurement, _, _ = actual_trial(state, model, step)
    # Remove only numerical solver roundoff from the independently nonlinear
    # distance, making this an exact layer tie rather than a tiny regression.
    tied = dict(measurement, depths=dict(model.spline_collision_layer.baseline['per_key']),
                rows=[(key, dict(row, clearance=-.03)) for key, row in measurement['rows']])
    result = solver.step_merit(model, step, 1., before, after, tied)
    assert nonreg and after < before
    assert not result[0] and result[2] == 'original_geometry_curvature_or_new_witness'


def test_unknown_current_measurement_stops_before_any_solver_call():
    state = fixture()
    model, _, _ = prepared(state)
    model.spline_geometry = dict(model.spline_geometry, failures=['unknown_query'], complete=False)
    step, diagnostic = state._solve(model)
    assert step is None and diagnostic['status'] in {'unknown', 'unknown_geometry'}
    assert diagnostic['conic_calls'] == state.conic_calls == 0
    assert not model.solve_diagnostics


def test_nonfinite_tracking_cost_cannot_be_accepted_for_real_collision_gain():
    state = fixture()
    model, _, _ = prepared(state)
    step, _ = state._solve(model)
    _, before, _, _, measurement, _, _ = actual_trial(state, model, step)
    result = solver.step_merit(model, step, 1., before, np.nan, measurement)
    assert not result[0] and result[2] == 'nonfinite_objective'


def test_new_self_key_is_in_actual_layer_even_if_all_original_nonreg_rows_pass():
    state = fixture(tracking=QuadraticTracking(target=(-.005, .02), weights=(1e6, 1e10)),
                    curvature=100., new_key_depth=8e-7)
    model, _, _ = prepared(state)
    step, _ = state._solve(model)
    _, before, after, nonreg, measured, bad, result = actual_trial(state, model, step)
    assert nonreg and not bad and after < before
    assert {(key[1]) for key in measured['depths']} == {'static', 'self'}
    assert all(key[1] == 'static' for key in model.spline_collision_layer.keys)
    assert not result[0] and result[2] == 'original_geometry_curvature_or_new_witness'
    assert result[3]['collision_after_m'] > .060001


def corrupt_candidate(kind):
    def change(measured, poses):
        if not np.any(poses != 0.):
            return
        if kind == 'failures':
            measured['failures'] = ['unknown_query']
        elif kind == 'missing_complete':
            del measured['complete']
        elif kind == 'frame_scope':
            measured['query_scope']['requested_frames'] = [0]
        elif kind == 'frame_complete':
            measured['frame_complete'][1] = False
        elif kind == 'nan_depth':
            measured['depths'][(1, 'static', 21)] = np.nan
        else:
            measured['reliable'] = False
    return change


@pytest.mark.parametrize('kind', ['failures', 'missing_complete', 'frame_scope',
                                 'frame_complete', 'nan_depth', 'reliable'])
def test_positive_collision_merit_rejects_unknown_or_partial_actual_frames(kind):
    state = fixture(query_change=corrupt_candidate(kind))
    model, _, _ = prepared(state)
    step, _ = state._solve(model)
    result = actual_trial(state, model, step)[-1]
    assert not result[0] and result[2] == 'unknown_original_geometry'
    assert result[3]['collision_after_m'] is None


@pytest.mark.parametrize('kind', ['failures', 'frame_scope', 'frame_complete', 'reliable'])
def test_positive_collision_optimize_keeps_reference_and_ledger_on_unknown_candidate(kind):
    state = fixture(query_change=corrupt_candidate(kind))
    reference, ledger = state.poses.copy(), dict(state.geometry.best)
    result = state.optimize(1)
    assert state.accepted == 0 and not state.geometry.history
    assert state.history[-1]['rejected_reason'] == 'unknown_original_geometry'
    np.testing.assert_array_equal(result, reference)
    assert state.geometry.best == ledger


def test_known_zero_collision_still_allows_original_tracking_without_collision_lp(monkeypatch):
    import omniretargeting._fast.spline_collision_priority as priority

    def forbidden(*args, **kwargs):
        pytest.fail('Zero known collision should use the ordinary tracking subproblem')

    monkeypatch.setattr(priority, 'collision_first_proposal', forbidden)
    state = fixture(tracking=QuadraticTracking(target=(0., .005), weights=(1., 1e6)),
                    depth=0., stationary=True)
    before = state.objective.cost(state.poses)
    result = state.optimize(1)
    assert state.accepted == 1 and state.objective.cost(result) < before
    assert state.conic_calls == 0
    assert all(depth == 0. for depth in state.geometry.best.values())


def test_selected_local_columns_predict_the_full_original_frame_measurement():
    state = fixture()
    model, mapping, selected = prepare_spline_model(state.objective, state.spline, state.a,
        state.objective.evaluate(state.poses), chatter=state.chatter, geometry=state.geometry,
        controls=np.array([1]), columns=np.array([0]))
    np.testing.assert_array_equal(selected, [2])
    assert mapping.shape == (4, 4) and model.gradient.size == 1
    step, _ = state._solve(model)
    assert step is not None
    candidate, _, _, nonreg, measured, _, merit = actual_trial(state, model, step)
    np.testing.assert_array_equal(candidate[0], np.zeros(2))
    assert candidate[1, 0] > 0 and candidate[1, 1] == 0.
    assert nonreg and merit[0] and merit[3]['accepted_priority'] == 'collision'
    assert len(measured['depths']) == 2
    assert merit[3]['collision_after_m'] == pytest.approx(.06 - candidate[1, 0])


@pytest.mark.parametrize('kind', ['frame_scope', 'frame_complete', 'nan_depth', 'reliable', 'missing_complete'])
def test_zero_collision_optimize_cannot_commit_unknown_geometry_despite_tracking_gain(kind):
    # The normal tracking branch is deliberately exercised: no known entry
    # penetration exists and geometry's per-key non-regression sees no bad key.
    state = fixture(tracking=QuadraticTracking(target=(0., .005), weights=(1., 1e6)),
                    depth=0., stationary=True, query_change=corrupt_candidate(kind))
    reference, ledger = state.poses.copy(), dict(state.geometry.best)
    result = state.optimize(1)
    assert state.accepted == 0, f'Unknown {kind} candidate was committed by zero-depth tracking path'
    np.testing.assert_array_equal(result, reference)
    assert state.geometry.best == ledger and not state.geometry.history
    assert state.history[-1]['rejected_reason'] == 'unknown_original_geometry'


@pytest.mark.parametrize('depth', [0., .03])
def test_solve_rejects_partial_current_frame_scope_including_zero_depth(depth):
    def partial(measured, poses):
        measured['query_scope']['requested_frames'] = [0]

    state = fixture(tracking=QuadraticTracking(target=(0., .005), weights=(1., 1e6)),
                    depth=depth, stationary=True, query_change=partial)
    model, _, _ = prepared(state)
    step, diagnostics = state._solve(model)
    assert step is None, 'Partial current geometry was allowed to produce a tracking step'
    assert diagnostics['status'] in {'unknown', 'unknown_geometry'}
    assert diagnostics['conic_calls'] == state.conic_calls == 0


@pytest.mark.parametrize('kind', ['nan_jacobian', 'missing_jacobian'])
def test_solve_returns_unknown_status_for_missing_collision_derivative_without_crashing(kind):
    state = fixture()
    model, _, _ = prepared(state)
    row = model.spline_geometry['rows'][0][1]
    if kind == 'nan_jacobian':
        row['jacobian'][0] = np.nan
    else:
        del row['jacobian']
    step, diagnostics = state._solve(model)
    assert step is None and diagnostics['status'] in {'unknown', 'unknown_geometry'}
    assert diagnostics['unknown_reasons'] and diagnostics['conic_calls'] == 0


def test_zero_entry_new_key_cannot_purchase_collision_with_large_tracking_gain():
    state = fixture(tracking=QuadraticTracking(target=(0., .005), weights=(1., 1e12)),
                    depth=0., stationary=True, new_key_depth=8e-7)
    model, _, _ = prepared(state)
    step, _ = state._solve(model)
    _, before, after, nonreg, _, bad, result = actual_trial(state, model, step)
    assert before - after > 4e7 and nonreg and not bad
    assert not result[0], 'Zero-entry tracking descent purchased a newly known self collision'


def test_internal_tuple_keys_survive_while_history_and_report_are_json_serializable():
    state = fixture()
    model, _, _ = prepared(state)
    step, raw = state._solve(model)
    assert step is not None and isinstance(next(iter(raw['key_ownership'])), tuple)
    with pytest.raises(TypeError):
        json.dumps(raw)
    json.dumps(solver._recordable(raw), allow_nan=False)
    assert isinstance(next(iter(raw['key_ownership'])), tuple)
    assert all(isinstance(key, tuple) for key in model.spline_collision_layer.keys)
    state.optimize(1)
    decoded = json.loads(json.dumps({'history': state.history, 'report': state.report()}, allow_nan=False))
    assert decoded['history'][0]['solver']['certification']['certified']
    assert all(isinstance(key, tuple) for key in state.geometry.best)


def test_trust_scale_shrinks_only_current_increment_without_changing_tracking_or_reference():
    state = fixture()
    original, _, _ = prepared(state)
    smaller, _, _ = prepared(state, trust_scale=.25)
    np.testing.assert_array_equal(smaller.hessian.toarray(), original.hessian.toarray())
    np.testing.assert_array_equal(smaller.gradient, original.gradient)
    np.testing.assert_array_equal(smaller.step_lower, .25*original.step_lower)
    np.testing.assert_array_equal(smaller.step_upper, .25*original.step_upper)
    assert state.geometry.best == {(0, 'static', 21): .03, (1, 'static', 21): .03}
    np.testing.assert_array_equal(state.objective.reference, np.zeros((2, 2)))
    np.testing.assert_array_equal(state.objective.lower, [-1., -1.])
    np.testing.assert_array_equal(state.objective.upper, [1., 1.])
