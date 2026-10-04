"""Objective-preserving globalization and full-QP window certificates."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.globalization import globalize
from omniretargeting._fast.qp_assembly import prepare_model, solve_model
from omniretargeting._fast.qp_types import QPSolution
from omniretargeting._fast.second_order_step import corrected_candidates
from omniretargeting._fast.window_qp import solve_standard


class Coordinates:
    root_q = None
    def __init__(self, width):
        self.width = width
        self.qi = self.scalar_cols = np.arange(width)
    def bounds(self, poses, angular, translation):
        return np.full_like(poses, -angular), np.full_like(poses, angular)
    def integrate(self, poses, step, alpha):
        return poses + alpha * step
    def difference(self, poses, candidate):
        return candidate - poses


def objective_for(target, guards, *, step=1.):
    frames, width = target.shape
    objective = SimpleNamespace(coords=Coordinates(width), frames=frames, width=width,
                                cfg=dict(angular_step_rad=step, translation_step_m=step),
                                lower=np.full(width, -2.), upper=np.full(width, 2.),
                                source_shape=None, candidate_guards=guards, project=lambda q: q)
    objective.evaluate = lambda q, **kw: (SimpleNamespace(cost=float(np.sum((q - target)**2))),)
    return objective


def test_soc_keeps_original_objective_and_crossing_temporal_rows():
    from tests.test_fast_local_restoration import _CurvedGuard, _TemporalGuard, _support
    class Curved(_CurvedGuard):
        qp_bounds = _CurvedGuard.repair_bounds
    class Temporal(_TemporalGuard):
        qp_bounds = _TemporalGuard.repair_bounds
    target = np.zeros((64, 2))
    target[:, 0] = .1
    guards = [Curved(), Temporal(64)]
    objective = objective_for(target, guards, step=.2)
    poses = np.zeros_like(target)
    model = prepare_model(objective, poses, np.ones((1, poses.size)), -target.ravel(), np.ones(poses.size))
    solution = solve_model(model, .001)
    results = list(corrected_candidates(objective, model, solution, .001, guards))
    assert results
    corrected, candidate, fraction = results[0]
    assert fraction == 1.
    assert all(guard.accepts(candidate) for guard in guards)
    # Unchanged tracking Hessian/gradient retain the tangential objective's
    # minimizer even though it would be cheaper in correction norm to back off.
    np.testing.assert_allclose(candidate[:, 0], target[:, 0] / 1.001, atol=2e-6)
    assert candidate[19, 1] > 0 and candidate[21, 1] > 0
    outside = ~_support(guards[0].violation_frames(solution.step.reshape(target.shape)))
    np.testing.assert_allclose(candidate[outside], solution.step.reshape(target.shape)[outside], atol=1e-12)


def test_frame_fractions_keep_exact_temporal_rows_and_improve_over_uniform():
    class Guard:
        def __init__(self):
            self.B = sparse.diags([-np.ones(7), np.ones(7)], [0, 1], shape=(7, 8), format='csr')
        def qp_bounds(self, q):
            values = self.B @ q.ravel()
            return [(self.B, -.2 - values, .2 - values)], []
        def violation_frames(self, q):
            bad = np.zeros(8, bool)
            bad[3] = q[3, 0]**2 > .04
            excess = np.abs(self.B @ q.ravel()) > .2 + 1e-7
            bad[:-1] |= excess
            bad[1:] |= excess
            return bad
        def accepts(self, q):
            return not self.violation_frames(q).any()
    guard = Guard()
    target, poses = np.ones((8, 1)), np.zeros((8, 1))
    objective = objective_for(target, [guard])
    model = prepare_model(objective, poses, np.ones((1, 8)), -np.ones(8), np.ones(8))
    solution = QPSolution(np.ones(8))
    _, candidate, _ = globalize(objective, model, solution, .001, [guard], frame_lengths=True)
    assert guard.accepts(candidate)
    assert np.max(np.abs(np.diff(candidate[:, 0]))) <= .2 + 1e-7
    assert np.sum((candidate - target)**2) < np.sum((.125 - target)**2)
    assert candidate[0, 0] > .5 and candidate[3, 0] < .2


def test_nonlinear_cost_cannot_be_worsened_by_a_better_fraction_model(monkeypatch):
    import omniretargeting._fast.frame_step as module
    target, poses = np.ones((3, 1)), np.zeros((3, 1))
    objective = objective_for(target, [])
    model = prepare_model(objective, poses, np.ones((1, 3)), -np.ones(3), np.ones(3))
    solution = QPSolution(np.ones(3))
    def bad(*a, **kw):
        yield QPSolution(np.full(3, .5)), np.full((3, 1), .5), .5
    monkeypatch.setattr(module, 'frame_candidates', bad)
    _, candidate, alpha = globalize(objective, model, solution, .001, [], frame_lengths=True)
    np.testing.assert_array_equal(candidate, target)
    assert alpha == 1.


def test_disabled_globalization_does_not_duplicate_objective_evaluation():
    target, poses = np.ones((3, 1)), np.zeros((3, 1))
    objective = objective_for(target, [])
    model = prepare_model(objective, poses, np.ones((1, 3)), -np.ones(3), np.ones(3))
    def unexpected(*args, **kwargs):
        raise AssertionError('The acceptance caller must own this evaluation')
    objective.evaluate = unexpected
    _, candidate, alpha = globalize(objective, model, QPSolution(np.ones(3)), .001, [])
    np.testing.assert_array_equal(candidate, target)
    assert alpha == 1.


def test_window_result_requires_global_stationarity_not_coordinate_stagnation():
    # Equality couples two disjoint windows: neither coordinate can move, but
    # the full problem's optimum is (1,1). Two stalled sweeps must fall back.
    P = sparse.eye(2, format='csc')
    A = sparse.csc_matrix([[1., -1.], [-1., 1.]])
    result = solve_standard(P, -np.ones(2), A, np.zeros(2), 2, [], np.arange(2), 2,
                            size=1, sweeps=8)
    assert result is None


def test_window_result_is_certified_for_separable_problem():
    count = 8
    result = solve_standard(sparse.eye(count), np.full(count, -.3),
                            sparse.vstack((sparse.eye(count), -sparse.eye(count))), np.ones(2 * count),
                            2 * count, [], np.arange(count), count, size=2)
    assert result is not None
    np.testing.assert_allclose(result, .3, atol=1e-7)


def test_window_never_hides_a_violated_constant_cone():
    # An impossible constant cone has no column incidence in any window.
    A = sparse.csc_matrix((4, 2))
    assert solve_standard(sparse.eye(2), np.zeros(2), A, np.array([1., 2., 0., 0.]),
                          0, [4], np.arange(2), 2, size=1) is None
