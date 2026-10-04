"""Banded primal-dual step KKT: exact affine balls and degenerate faces."""

import numpy as np
from scipy import sparse

from omniretargeting._fast.barrier_rows import ball, scalar
from omniretargeting._fast.constrained_ip import QuadraticDomain, solve
from tests.test_fast_constrained import active_diagnostics, band


def domain(families, lower, upper, held=()):
    return QuadraticDomain(families, np.asarray(lower), np.asarray(upper), np.asarray(held, int))


def test_ip_linear_dense_qp_with_wrong_cached_duals():
    rows = scalar([.1, 2., 2., 2.], sparse.csr_matrix([[-1., 0.], [1., 0.],
        [0., -1.], [0., 1.]]), ('test',), [(i,) for i in range(4)])
    guards = domain([rows], [-3., -3.], [3., 3.])
    diagnostics = active_diagnostics()
    point = solve(band(np.array([[2., 1.], [1., 2.]])), np.array([-3., -.2]),
        guards, dict(ip_dual=np.full(guards.count, 10.)), diagnostics)
    np.testing.assert_allclose(point, [.1, .05], atol=2e-6)
    assert guards.evaluate(point)[0].max() < 1e-9


def test_ip_retains_curved_ball_at_zero_vector_and_box_equality():
    rows = ball(np.zeros((1, 2)), np.array([.3]), sparse.eye(2, format='csr'), ('ball',), [(0,)])
    guards = domain([rows], [-1., -1.], [1., 1.])
    point = solve(band(np.eye(2)), np.array([-3., -4.]), guards, {}, active_diagnostics())
    np.testing.assert_allclose(point, [.18, .24], atol=2e-6)
    assert np.linalg.norm(point) <= .3
    guards = domain([rows], [0., -1.], [0., 1.], held=[0])
    point = solve(band(np.eye(2)), np.array([0., -4.]), guards, {}, active_diagnostics())
    np.testing.assert_allclose(point, [0., .3], atol=2e-6)


def test_ip_ball_jacobian_and_psd_curvature_match_dense():
    matrix = sparse.csr_matrix([[1., .1], [.2, -2.]])
    rows = ball(np.array([[.1, -.1]]), np.array([.5]), matrix, ('ball',), [(0,)])
    guards = domain([rows], [-1., -1.], [1., 1.])
    point = np.array([.05, .02]); epsilon = 1e-6
    _, jacobian = guards.evaluate(point)
    numeric = np.column_stack([(guards.evaluate(point + epsilon * d)[0] -
        guards.evaluate(point - epsilon * d)[0]) / (2 * epsilon) for d in np.eye(2)])
    np.testing.assert_allclose(jacobian.toarray(), numeric, atol=1e-9)
    curvature = band(np.zeros((2, 2)))
    dual = np.r_[np.zeros(len(guards.bounds)), 2.]
    guards.curvature(curvature, dual)
    expected = 2. / (guards.balls[0][2][0] * guards.balls[0][3][0]) * (matrix.T @ matrix).toarray()
    np.testing.assert_allclose(curvature, band(expected))
    assert np.linalg.eigvalsh(expected).min() >= 0


def test_soc_retains_remainder_after_face_becomes_feasible():
    from omniretargeting._fast.constrained_soc import shifted
    from dataclasses import replace
    row = scalar([.01], sparse.csr_matrix([[-1.]]), ('contact',), [(0,)])
    # Current nonlinear slack passes, but still includes a 2 mm curvature
    # defect. Dropping that defect would return to the original failed step.
    measured = replace(row, slack=np.array([.003]))
    corrected = shifted([row], [measured], np.array([.005]), {})
    np.testing.assert_allclose(corrected[0].slack, [.00798])
    np.testing.assert_array_equal(row.slack, [.01])
    np.testing.assert_array_equal(corrected[0].matrix.toarray(), row.matrix.toarray())


def test_soc_linear_crossing_stencil_has_no_curvature_offset():
    from omniretargeting._fast.constrained_soc import shifted
    from dataclasses import replace
    row = scalar([.001], sparse.csr_matrix([[-1., 3., -3., 1.]]), ('continuity',), [(12,)])
    point = np.array([.01, -.02, .03, -.04])
    measured = replace(row, slack=row.slack + row.matrix @ point)
    assert shifted([row], [measured], point, {}) is None


def test_ip_opposing_zero_budget_stencil_keeps_exact_acceptance():
    rows = scalar([0., 0.], sparse.csr_matrix([[-1., 1.], [1., -1.]]),
                  ('zero',), [(0,), (0,)])
    guards = domain([rows], [-.1, -.1], [.1, .1])
    point = solve(band(np.eye(2)), np.array([-3., 2.]), guards, {}, active_diagnostics())
    assert abs(point[0] - point[1]) < 1e-9
    np.testing.assert_allclose(point, [.1, .1], atol=2e-6)


def test_blas_band_product_matches_dense_with_full_width_band():
    from omniretargeting._fast.constrained_ip import band_product
    rng = np.random.default_rng(32)
    matrix = rng.normal(size=(12, 12))
    matrix += matrix.T
    vector = rng.normal(size=12)
    np.testing.assert_allclose(band_product(band(matrix), vector), matrix @ vector, atol=1e-12)


def test_ip_all_columns_held_has_no_empty_complementarity_division():
    row = scalar([0., 0.], sparse.eye(2, format='csr'), ('fixed',), [(0,), (1,)])
    guards = domain([row], [0., 0.], [0., 0.], held=[0, 1])
    assert guards.count == 0
    with np.errstate(all='raise'):
        point = solve(band(np.eye(2)), np.zeros(2), guards, {}, active_diagnostics())
    np.testing.assert_array_equal(point, [0., 0.])


def test_ip_window_route_keeps_zero_box_fixed_root_quaternion_and_commit(monkeypatch):
    from omniretargeting._fast import runtime_options
    from omniretargeting._fast.runtime_options import RuntimeOptions
    from omniretargeting._fast.constrained_rows import ConstrainedRows
    from omniretargeting._fast.legacy_window_qp import WindowObjective, solve_local
    from omniretargeting._fast.proposals import guards_of
    from tests.test_fast_legacy_window import problem
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions(active_set_acquisition=True))
    monkeypatch.setattr(ConstrainedRows, 'primal_dual', True, raising=False)
    monkeypatch.setattr(ConstrainedRows, 'banded_restoration', True, raising=False)
    objective, poses = problem()
    reference = objective.reference.tobytes()
    objective.fixed_root = poses[:, :7].copy()
    joint = objective.coords.qi[-1]
    objective.lower = np.broadcast_to(objective.lower, (len(poses), len(objective.coords.qi))).copy()
    objective.upper = np.broadcast_to(objective.upper, objective.lower.shape).copy()
    objective.lower[:, -1] = objective.upper[:, -1] = poses[:, joint]
    result = solve_local(objective, poses, range(8, 20), 2, .003)
    window = WindowObjective(objective, poses, range(8, 20))
    np.testing.assert_array_equal(result['poses'][:, :7], poses[window.span, :7])
    np.testing.assert_array_equal(result['poses'][:, joint], poses[window.span, joint])
    assert all(g.accepts(result['poses']) for g in guards_of(window)[1])
    assert all(h['after'] < h['cost'] for h in result['history'] if h['alpha'])
    assert objective.reference.tobytes() == reference
    assert result['commit']['accepted'] == (result['commit']['candidate_cost'] < result['commit']['cost'])
