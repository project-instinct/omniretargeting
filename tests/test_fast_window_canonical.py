"""Window canonicalization preserves solver input bytes, not just feasibility."""

import cvxpy as cp
import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.legacy_window_qp import WindowObjective, serial_proposals
from omniretargeting._fast.proposals import guards_of
from omniretargeting._fast.window_canonical import canonical_data, solve_data, restore_step
from omniretargeting._fast.window_conic import step_bounds
from omniretargeting.conic_solver import solve_conic, solve_conic_working_set
from tests.test_fast_legacy_window import problem


def assert_program_equal(expected, actual):
    for key in ('A', 'P'):
        first, second = expected[key], actual[key]
        assert first.shape == second.shape
        np.testing.assert_array_equal(first.indptr, second.indptr)
        np.testing.assert_array_equal(first.indices, second.indices)
        assert first.data.tobytes() == second.data.tobytes(), key
    for key in ('b', 'c'):
        assert expected[key].tobytes() == actual[key].tobytes(), key
    for key in ('zero', 'nonneg', 'soc', 'psd', 'exp', 'p3d', 'pnd'):
        assert getattr(expected['dims'], key) == getattr(actual['dims'], key)


@pytest.mark.parametrize('frames', [range(0, 8), range(8, 20), range(28, 36)])
@pytest.mark.parametrize('amplitude', [0., .001, .05])
@pytest.mark.parametrize('fixed_root', [False, True])
def test_canonical_window_coefficients_and_order_are_exact(frames, amplitude, fixed_root):
    obj, poses = problem(extra=True)
    if fixed_root:
        obj.fixed_root = poses[:, :7].copy()
    window = WindowObjective(obj, poses, frames)
    origin = poses[window.span]
    candidate = obj.coords.integrate(origin, np.random.default_rng(611).normal(
        scale=amplitude, size=(len(frames), obj.width)), 1.)
    lo, hi = step_bounds(window, candidate)
    count = lo.size
    rng = np.random.default_rng(12)
    diagonal, gradient = rng.uniform(1., 3., count), rng.normal(size=count)
    matrix = sparse.diags(diagonal, format='csc')
    scale = max(float(np.linalg.norm(gradient, ord=np.inf)), 1.)
    flat = cp.Variable(count)
    constraints = [flat >= lo.ravel(), flat <= hi.ravel()]
    for guard in guards_of(window)[1]:
        constraints += guard.constraints(candidate, flat)
    reference = cp.Problem(cp.Minimize((cp.quad_form(flat, cp.psd_wrap(matrix))
                                       + 2 * gradient @ flat) / scale), constraints)
    expected, _, _ = reference.get_problem_data(cp.CLARABEL)
    actual = canonical_data(window, candidate, lo, hi, matrix * (2 * (1 / scale)),
                            (2 * gradient) * (1 / scale))
    assert_program_equal(expected, actual)
    for working_set, solve in ((False, solve_conic), (True, solve_conic_working_set)):
        # Separate CVXPY Problems, just like separate restoration attempts.
        reference = cp.Problem(reference.objective, constraints)
        try:
            solve(reference, solver=cp.CLARABEL, direct_solve_method='qdldl')
        except cp.error.SolverError:
            expected_step = None
        else:
            expected_step = flat.value if reference.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) else None
        step = solve_data(actual, count, working_set=working_set)
        if expected_step is None:
            assert step is None
        else:
            assert step is not None
            assert step.tobytes() == expected_step.tobytes()


@pytest.mark.parametrize('amplitude', [0., .001, .01, .1])
def test_restoration_matches_legacy_pose_bytes_and_guard_decisions(amplitude, monkeypatch):
    from omniretargeting._fast import shape_step
    monkeypatch.setattr(shape_step, '_canonical_window', lambda objective: False)
    obj, poses = problem(extra=True)
    window = WindowObjective(obj, poses, range(9, 18))
    origin = poses[window.span]
    step = np.random.default_rng(721).normal(scale=amplitude, size=(window.frames, window.width))
    with serial_proposals():
        expected = shape_step.restore_shape_step(window, origin, step)
        actual = restore_step(window, origin, step)
    if expected is None:
        assert actual is None
    else:
        assert actual is not None
        assert actual.tobytes() == expected.tobytes()
        assert all(g.accepts(actual) == g.accepts(expected) for g in guards_of(window)[1])
        full = window.embed(actual)
        assert full[:9].tobytes() == poses[:9].tobytes()
        assert full[18:].tobytes() == poses[18:].tobytes()


def test_canonical_retry_retains_the_same_solver_cache(monkeypatch):
    from types import SimpleNamespace
    from omniretargeting._fast import window_canonical
    calls = []

    def solve(self, data, warm_start, verbose, solver_opts, solver_cache):
        calls.append(dict(solver_opts))
        if len(calls) == 1:
            assert not solver_cache
            solver_cache['CLARABEL'] = 'retained'
            return SimpleNamespace(status='InsufficientProgress', x=[0.], obj_val=0.)
        assert solver_cache['CLARABEL'] == 'retained'
        return SimpleNamespace(status='AlmostSolved', x=[.0123], obj_val=0.)
    monkeypatch.setattr(window_canonical.CLARABEL, 'solve_via_data', solve)
    value = solve_data(dict(c=np.zeros(1)), 1)
    assert value[0] == .0123
    assert calls == [dict(direct_solve_method='qdldl'),
                     dict(direct_solve_method='qdldl', tol_gap_abs=1e-6,
                          tol_gap_rel=1e-6, tol_feas=1e-6)]


def test_complete_window_lm_history_and_pose_bytes_match_cvxpy(monkeypatch):
    from omniretargeting._fast import shape_step, window_qoco
    # Canonical Clarabel assembly vs CVXPY; the default QOCO proposals are off here.
    monkeypatch.setattr(window_qoco, 'WINDOW_QOCO_PROPOSALS', False)
    from omniretargeting._fast.legacy_window_qp import solve_local
    original = shape_step._canonical_window
    outputs = []
    for canonical in (False, True):
        monkeypatch.setattr(shape_step, '_canonical_window', original if canonical else lambda obj: False)
        obj, poses = problem(extra=True)
        outputs.append(solve_local(obj, poses, range(8, 20), 4, .003))
    before, after = outputs
    assert before['poses'].tobytes() == after['poses'].tobytes()
    def decisions(history):
        return [{k: v for k, v in row.items() if k != 'seconds'} for row in history]
    assert decisions(before['history']) == decisions(after['history'])
    for key in ('converged', 'stop_reason'):
        assert before[key] == after[key]
    for key in ('cost', 'after', 'candidate_cost', 'accepted', 'parent_rejections'):
        assert before['commit'][key] == after['commit'][key]


def test_shared_cone_descriptors_preserve_repeated_working_set_solutions(monkeypatch):
    import clarabel
    from types import SimpleNamespace
    from omniretargeting import conic_solver
    variables, count = 3, 100
    # One binding cone, many safely inactive cones: exercise the reduced
    # working-set solve with repeated descriptors and new RHS values.
    matrix = sparse.vstack([sparse.csc_matrix((1, variables)), -sparse.eye(variables)], format='csc')
    bounds = np.full(count, 10.)
    bounds[0] = .01
    cached = conic_solver._soc_descriptor
    results = []
    for factory in (clarabel.SecondOrderConeT, cached, cached):
        monkeypatch.setattr(conic_solver, '_soc_descriptor', factory)
        values = []
        for shift in (0., .002, -.003):
            data = dict(P=sparse.eye(variables, format='csc') * 2,
                        c=np.array([-1., .2, .3]), A=sparse.vstack([matrix] * count, format='csc'),
                        b=np.c_[bounds, np.tile([.0099 + shift, 0., 0.], (count, 1))].ravel(),
                        dims=SimpleNamespace(zero=0, nonneg=0, soc=[4] * count,
                                             psd=[], exp=0, p3d=[], pnd=[]))
            values.append(solve_data(data, variables, working_set=True).tobytes())
        results.append(values)
    assert results[0] == results[1] == results[2]
