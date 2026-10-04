"""Eliminating private norm auxiliaries preserves the full pose domain."""
from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.window_restore_reduction import reduce_epigraphs
from omniretargeting._fast.window_canonical import canonical_data, solve_data


def program(seed=13):
    rng = np.random.default_rng(seed)
    size, count = 5, 7
    jac = sparse.csr_matrix(rng.normal(size=(count*3, size)))
    residual = rng.normal(scale=.02, size=(count, 3))
    allowed = np.linalg.norm(residual, axis=1) + rng.uniform(.01, .1, count)
    guard = SimpleNamespace(qp_bounds=lambda poses: (
        [(sparse.csr_matrix([[1., 0., 0., 0., 0.]]), np.array([-.02]), np.array([.03]))],
        [(jac, residual, allowed)]))
    obj = SimpleNamespace(source_shape=guard, candidate_guards=[])
    return canonical_data(obj, None, np.full((1, size), -.2), np.full((1, size), .2),
                          sparse.eye(size, format='csc')*2, rng.normal(size=size))


def margins(data, point):
    slack = data['b'] - data['A'] @ point
    start = data['dims'].nonneg
    answer = list(slack[:start])
    for size in data['dims'].soc:
        answer.append(slack[start] - np.linalg.norm(slack[start+1:start+size]))
        start += size
    return np.asarray(answer)


@pytest.mark.parametrize('seed', [13, 42, 817])
def test_pose_domain_and_quadratic_optimum_are_unchanged(seed):
    original = program(seed)
    reduced = reduce_epigraphs(original, 5)
    rng = np.random.default_rng(seed)
    for point in [np.zeros(5), *rng.normal(scale=.04, size=(60, 5))]:
        # The largest allowed t gives a feasible lift exactly when x is
        # feasible in the reduced program, including every box/linear row.
        starts = reduced['dims'].nonneg + np.r_[0, np.cumsum(reduced['dims'].soc)[:-1]]
        lift = np.r_[point, reduced['b'][starts]]
        assert (margins(original, lift).min() >= 0) == (margins(reduced, point).min() >= 0)
        np.testing.assert_allclose(original['c'] @ lift, reduced['c'] @ point, atol=0.)
    a, b = solve_data(original, 5), solve_data(reduced, 5)
    assert a is not None and b is not None
    np.testing.assert_allclose(a, b, atol=2e-5, rtol=0.)
    assert margins(reduced, b).min() >= -1e-7


def test_true_norm_slack_allows_restoration_constraint_generation():
    data = program()
    reduced = reduce_epigraphs(data, 5)
    # The old SOCs look violated at the all-zero epigraph point, even though
    # every physical norm has positive slack at the zero pose correction.
    assert margins(data, np.zeros(len(data['c']))).min() < 0.
    assert margins(reduced, np.zeros(5)).min() > 0.
    full = solve_data(reduced, 5)
    generated = solve_data(reduced, 5, working_set=True)
    np.testing.assert_allclose(full, generated, atol=2e-5, rtol=0.)


@pytest.mark.parametrize('corrupt', ['objective', 'head', 'upper'])
def test_other_auxiliary_contracts_are_rejected(corrupt):
    data = program()
    if corrupt == 'objective':
        data['c'][-1] = .01
    elif corrupt == 'head':
        data['b'][data['dims'].nonneg] = .01
    else:
        a = data['A'].tolil()
        a[0, -1] = 1.
        data['A'] = a.tocsc()
    with pytest.raises(ValueError, match='epigraph'):
        reduce_epigraphs(data, 5)


@pytest.mark.parametrize('amplitude', [0., .0001, .001, .01])
def test_restored_window_passes_original_nonlinear_and_crossing_guards(amplitude, monkeypatch):
    from omniretargeting._fast import window_restore_reduction
    from omniretargeting._fast.legacy_window_qp import WindowObjective, guard_failures
    from omniretargeting._fast.window_canonical import restore_step
    from tests.test_fast_legacy_window import problem

    obj, poses = problem(extra=True)
    window = WindowObjective(obj, poses, range(9, 18))
    delta = np.random.default_rng(721).normal(scale=amplitude, size=(window.frames, window.width))
    monkeypatch.setattr(window_restore_reduction, 'EXACT_WINDOW_RESTORATION', False)
    legacy = restore_step(window, poses[window.span], delta)
    monkeypatch.setattr(window_restore_reduction, 'EXACT_WINDOW_RESTORATION', True)
    restored = restore_step(window, poses[window.span], delta)
    # Large random root/joint jumps are rejected by both four-step loops;
    # removal of auxiliaries must never turn a rejection into acceptance.
    assert (restored is None) == (legacy is None)
    if restored is None:
        return
    embedded = window.embed(restored)
    assert not guard_failures(obj, embedded)
    np.testing.assert_array_equal(embedded[:9], poses[:9])
    np.testing.assert_array_equal(embedded[18:], poses[18:])
