"""Coupled direction bounds, release conditions and constrained fallback."""
from types import SimpleNamespace

import numpy as np
import pytest

from omniretargeting._fast.box_step import (
    BoxStepFailure, banded_product, direction_bounds, solve_box_banded,
)
from omniretargeting._fast.solver import _banded_solve, _hold
from omniretargeting._fast.proposals import Attempts


def band(matrix):
    result = np.zeros_like(matrix)
    for offset in range(len(matrix)):
        result[offset, :len(matrix) - offset] = np.diag(matrix, -offset)
    return result


def test_coupled_direction_can_cross_a_bound_despite_inward_negative_gradient():
    hessian = np.array([[2., 1.], [1., 2.]])
    rhs = np.array([1., 3.])
    assert np.linalg.solve(hessian, rhs)[0] < 0 < rhs[0]
    result = solve_box_banded(band(hessian), rhs, np.zeros(2), np.full(2, np.inf),
                              _banded_solve, _hold)
    np.testing.assert_allclose(result, [0., 1.5], atol=1e-15)
    # Clipping the unconstrained answer would leave the coupled variable 1 wrong.
    assert abs(result[1] - np.clip(np.linalg.solve(hessian, rhs), 0, None)[1]) > .1


def test_nonzero_bound_elimination_updates_the_other_variables():
    hessian = np.array([[2., 1.], [1., 2.]])
    result = solve_box_banded(band(hessian), np.array([1., 3.]),
                              np.array([.2, -np.inf]), np.full(2, np.inf), _banded_solve, _hold)
    np.testing.assert_allclose(result, [.2, 1.4], atol=1e-15)


@pytest.mark.parametrize('seed', range(16))
def test_releasing_variables_produces_box_feasibility_and_kkt_certificate(seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(4, 4))
    hessian = a.T @ a + np.eye(4) * .3
    rhs = rng.normal(size=4) * 4
    states = []

    def solve(matrix, rhs):
        states.append(set(np.flatnonzero(matrix[0] == 1.)))
        return _banded_solve(matrix, rhs)

    result = solve_box_banded(band(hessian), rhs, np.zeros(4), np.ones(4), solve, _hold)
    assert np.all(result >= 0) and np.all(result <= 1)
    residual = hessian @ result - rhs
    assert np.all(residual[result == 0] >= -1e-12)
    assert np.all(residual[result == 1] <= 1e-12)
    np.testing.assert_allclose(residual[(result > 0) & (result < 1)], 0, atol=1e-12)
    if seed == 1:
        assert any(old - new for old, new in zip(states, states[1:]))


def test_work_cap_requests_fallback_instead_of_returning_clipped_iterate():
    with pytest.raises(BoxStepFailure, match='budget'):
        solve_box_banded(band(np.eye(2)), np.ones(2), np.zeros(2), np.ones(2) * .1,
                         _banded_solve, _hold, limit=1)
    objective = SimpleNamespace(candidate_guards=[])

    def failed(*args):
        raise BoxStepFailure('budget')

    attempts = Attempts(objective, np.zeros((1, 2)), None, None, None, None, None, None,
                        [.001], failed, None)
    assert attempts.banded(0) == (None, None, 0.)


def test_position_boxes_and_unconditional_root_holds_are_separate():
    coords = SimpleNamespace(qi=np.array([3, 4]), scalar_cols=np.array([2, 3]))
    objective = SimpleNamespace(coords=coords, width=4, lower=np.array([-.2, -.3]), upper=np.array([.2, .3]))
    poses = np.zeros((2, 5))
    poses[:, 3:] = [.1, -.1]
    lower, upper = direction_bounds(objective, poses, np.array([0, 4]))
    np.testing.assert_allclose(lower[[2, 3, 6, 7]], [-.3, -.2, -.3, -.2])
    np.testing.assert_allclose(upper[[2, 3, 6, 7]], [.1, .4, .1, .4])
    assert np.array_equal(lower[[0, 4]], upper[[0, 4]])
    assert np.isneginf(lower[[1, 5]]).all()


def test_symmetric_banded_product_matches_dense():
    matrix = np.array([[3., -1., 0.], [-1., 4., 2.], [0., 2., 5.]])
    vector = np.array([.4, -.2, .7])
    np.testing.assert_allclose(banded_product(band(matrix), vector), matrix @ vector)


def test_source_reference_fit_is_byte_identical_with_box_experiment(monkeypatch):
    from dataclasses import replace
    from omniretargeting._fast import runtime_options
    from omniretargeting._fast.solver import optimize
    from tests.test_fast_backend import _objective
    results = []
    for enabled in (False, True):
        monkeypatch.setattr(runtime_options, 'OPTIONS',
                            replace(runtime_options.OPTIONS, active_joint_bounds=enabled))
        objective, poses, _ = _objective(temporal=True)
        assert getattr(objective, 'source_shape', None) is None
        result, history = optimize(objective, poses, iterations=3, label='source_pose')
        results.append((result.tobytes(), [{k: v for k, v in row.items() if k != 'seconds'} for row in history]))
    assert results[0] == results[1]
