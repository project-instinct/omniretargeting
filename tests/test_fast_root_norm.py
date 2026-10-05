"""Acquisition sees the same root norm budgets as final validation."""

import pytest
import cvxpy as cp
from tests.test_fast_budget_modes import select_budget_mode
from omniretargeting._fast import runtime_options


@pytest.fixture(autouse=True, params=[False, True])
def budget_mode(request, monkeypatch):
    select_budget_mode(monkeypatch, request.param)

import numpy as np
from scipy.ndimage import maximum_filter1d

from omniretargeting._fast.continuity import CorrectionContinuity, ROUND_OFF
from omniretargeting._fast.pipeline import verify_budgets
from omniretargeting._fast.reweight import offending_frames, reweight_temporal_signals
from tests.test_fast_backend import _objective


def _norm_counterexample():
    objective, poses, _ = _objective(temporal=True)
    poses = np.repeat(poses[:1], len(poses), axis=0)
    acceleration = np.eye(3)[np.arange(len(poses) - 2) % 3] * 1e-4
    def positions(a):
        velocity = np.vstack([np.full(3, .01), .01 + np.cumsum(a, axis=0)])
        return np.vstack([np.zeros(3), np.cumsum(velocity, axis=0)])
    q = objective.coords.root_q
    poses[:, q:q + 3] = positions(acceleration)
    candidate = poses.copy()
    acceleration[9] = .7e-4
    candidate[:, q:q + 3] = positions(acceleration)
    if runtime_options.OPTIONS.correction_budget:
        candidate[:, q:q + 3] += poses[:, q:q + 3]
    return objective, poses, candidate


def test_combined_root_chatter_is_rejected_even_when_each_coordinate_passes():
    objective, fixed, candidate = _norm_counterexample()
    coords = objective.coords
    assert CorrectionContinuity(coords, fixed).accepts(candidate)
    guard = CorrectionContinuity(coords, fixed, root_norm=True)
    assert guard.accepts(fixed)
    assert not guard.accepts(candidate)
    report, _, _ = verify_budgets(objective.model, coords, fixed, candidate)
    assert report['root_order_2']['derivative_screen_violations'] > 0
    for order in (2, 3):
        original = np.linalg.norm(np.diff(fixed[:, :3], n=order, axis=0), axis=1)
        independent = maximum_filter1d(original, 7, mode='nearest') + .1 * np.sqrt(np.mean(original**2))
        np.testing.assert_array_equal(guard.norm_bounds[f'root_translation_order_{order}'], independent)


def test_root_norm_is_present_in_conic_and_repair_rows_with_roundoff_reserve():
    objective, fixed, candidate = _norm_counterexample()
    guard = CorrectionContinuity(objective.coords, fixed, root_norm=True)
    flat = cp.Variable(len(fixed) * objective.width)
    flat.value = np.zeros(flat.shape)
    merged = []
    constraints = guard.constraints(candidate, flat, merge_rows=merged)
    assert merged and len(constraints) == 2
    assert max(float(np.max(c.violation())) for c in constraints) > 1e-4
    linear, norms = guard.repair_bounds(fixed)
    assert linear and len(norms) == 2
    direction = np.zeros_like(fixed[:, :objective.width])
    for row, (name, bound) in zip(norms, guard.norm_bounds.items()):
        matrix, residual, allowance = row
        np.testing.assert_array_equal(allowance, bound + ROUND_OFF / 2)
        order = int(name[-1])
        # A step at a frame affects every crossing stencil, including earlier starts.
        direction[:] = 0
        direction[10, objective.coords.inverse[objective.coords.root_d]] = 1e-3
        expected = np.diff(direction[:, :3], n=order, axis=0)
        np.testing.assert_allclose((matrix @ direction.ravel()).reshape(-1, 3), expected, atol=0)


def test_reweight_root_only_preserves_joint_rotation_and_world_path_weights():
    objective, fixed, candidate = _norm_counterexample()
    candidate = fixed.copy()
    candidate[:, objective.coords.root_q:objective.coords.root_q + 3] += .02 * (
        -1.0
    ) ** np.arange(len(fixed))[:, None]
    report = reweight_temporal_signals(objective, fixed, candidate, 10.)
    assert not report['joint_columns'] and report['root_translation_frames']
    for weights in (objective.temporal_weight, objective.joint_temporal_weight, objective.root_rotation_weight):
        assert all(np.all(w == 1) for w in weights)
    assert all(np.any(w == 10) for w in objective.root_translation_weight)
    joint, root = offending_frames(objective.coords, fixed, candidate)
    assert not joint.any()
    marked = np.flatnonzero(root)
    for order, weights in enumerate(objective.root_translation_weight, start=1):
        expected = [i for i in range(len(weights)) if any(i <= f <= i + order for f in marked)]
        assert np.flatnonzero(weights == 10).tolist() == expected


def test_reweight_one_joint_does_not_change_other_joint_or_root_channels():
    objective, fixed, _ = _objective(temporal=True)
    candidate = fixed.copy()
    candidate[15, objective.coords.qi[1]] += .2
    report = reweight_temporal_signals(objective, fixed, candidate, 10.)
    assert report['joint_columns'] == [1]
    for weights in objective.joint_temporal_weight:
        assert np.any(weights[:, 1] == 10)
        assert np.all(np.delete(weights, 1, axis=1) == 1)
    assert all(np.all(w == 1) for w in objective.root_translation_weight)


def test_signal_weights_change_only_their_rows_and_keep_exact_jacobians():
    objective, poses, _ = _objective(temporal=True)
    objective.root_translation_weight[1][3:8] *= 3.
    objective.root_rotation_weight[2][2:7] *= 5.
    objective.joint_temporal_weight[0][4:9, 1] *= 7.
    scale = objective.correction_temporal_scale
    for matrix, op, i in zip(objective._temporal_matrices(scale), objective.joint_matrices, range(3)):
        row_scale = (objective.temporal_weight[i][:, None] * scale * objective.joint_temporal_weight[i]).ravel()
        np.testing.assert_array_equal(matrix.toarray(), op.toarray() * row_scale[:, None])
    # A finite difference checks that residual and Jacobian weighting agree,
    # including the independent root translation/orientation paths.
    rng = np.random.default_rng(13)
    direction = rng.normal(size=(len(poses), objective.width)) * .01
    _, residual, jac, _, _ = objective.evaluate(poses)
    epsilon = 1e-5
    plus = objective.evaluate(objective.coords.integrate(poses, direction, epsilon), False)[1]
    minus = objective.evaluate(objective.coords.integrate(poses, direction, -epsilon), False)[1]
    np.testing.assert_allclose(jac @ direction.ravel(), (plus - minus) / (2 * epsilon), atol=3e-6, rtol=3e-5)
    assert np.isfinite(residual).all()
