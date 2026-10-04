"""Proposal margins, crossing rows and unchanged acquisition acceptance."""
import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.constrained_rows import ConstrainedRows
from omniretargeting._fast.constrained_step import with_boxes
from omniretargeting._fast.legacy_window_qp import WindowObjective, solve_local
from omniretargeting._fast.runtime_options import RuntimeOptions
from tests.test_fast_legacy_window import problem


def active_diagnostics():
    return dict(factorizations=0, band_solves=0, schur_iterations=0, schur_incomplete=0,
                working_sets=0, dropped_rows=0, added_rows=0)


def band(matrix):
    result = np.zeros(matrix.shape, order='F')
    for k in range(len(matrix)):
        result[k, :len(matrix)-k] = np.diag(matrix, -k)
    return result


def test_banded_equality_schur_matches_dense_kkt_stationarity_and_psd():
    from omniretargeting._fast.constrained_active import equality_step, normalized_rows
    rng = np.random.default_rng(43)
    physical = rng.normal(size=(7, 7))
    physical = physical.T @ physical + np.eye(7)
    gradient = rng.normal(size=7)
    matrix = sparse.csr_matrix(rng.normal(size=(3, 7)))
    bounds = rng.normal(size=3)
    a, h, _ = normalized_rows(matrix, bounds, np.diag(physical))
    d, multipliers = equality_step(band(physical), gradient, a, h, np.zeros(3), active_diagnostics())
    dense = a.toarray()
    expected = np.linalg.solve(np.block([[physical, dense.T], [dense, np.zeros((3, 3))]]),
                               np.r_[-gradient, h])
    np.testing.assert_allclose(d, expected[:7], atol=2e-8)
    np.testing.assert_allclose(multipliers, expected[7:], atol=2e-8)
    np.testing.assert_allclose(physical @ d + gradient + a.T @ multipliers, 0., atol=2e-8)


def test_active_set_coupled_boxes_releases_wrong_multiplier_and_warm_starts():
    from omniretargeting._fast.constrained_active import solve_active
    physical = np.array([[2., 1.], [1., 2.]])
    matrix = sparse.csr_matrix([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    bounds = np.array([.1, 2., 2., 2.])
    warm = dict(active=np.array([False, True, False, False]), multipliers=np.zeros(4))
    diagnostics = active_diagnostics()
    direction = solve_active(band(physical), np.array([-3., -.2]), matrix, bounds, warm, diagnostics)
    np.testing.assert_allclose(direction, [.1, .05], atol=1e-9)
    assert diagnostics['dropped_rows'] >= 1
    assert np.all(matrix @ direction <= bounds + 1e-10)
    second = active_diagnostics()
    np.testing.assert_allclose(solve_active(band(physical), np.array([-3., -.2]),
                                          matrix, bounds, warm, second), direction, atol=1e-9)
    assert second['warm_rows'] == 1 and second['factorizations'] == 1


def test_active_set_dependent_faces_drop_the_looser_face():
    from omniretargeting._fast.constrained_active import solve_active
    diagnostics = active_diagnostics()
    direction = solve_active(band(np.eye(2)), np.array([-3., 0.]),
        sparse.csr_matrix([[1., 0.], [1., 0.]]), np.array([1., 2.]), {}, diagnostics)
    np.testing.assert_allclose(direction, [1., 0.], atol=1e-8)
    assert diagnostics['dropped_rows'] >= 1


def test_active_set_reports_binding_rows_without_free_variables():
    from omniretargeting._fast.constrained_active import ActiveSetFailure, solve_active
    with pytest.raises(ActiveSetFailure, match='no free tangent'):
        solve_active(band(np.eye(2)), np.zeros(2), sparse.csr_matrix((1, 2)),
                     np.array([-.01]), {}, active_diagnostics())


def test_projected_dual_identifies_coupled_binding_face_without_dense_normals():
    from omniretargeting._fast.constrained_active import normalized_rows
    from omniretargeting._fast.constrained_warm import working_set
    physical = np.array([[2., 1.], [1., 2.]])
    matrix = sparse.csr_matrix([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    bounds = np.array([.1, 2., 2., 2.])
    normal, limits, _ = normalized_rows(matrix, bounds, np.diag(physical))
    diagnostics = active_diagnostics()
    direction, dual = working_set(band(physical), np.array([-3., -.2]), normal, limits, {}, diagnostics)
    np.testing.assert_allclose(direction, [.1, .05], atol=1e-6)
    assert dual[0] > 0 and np.max(dual[1:]) < 1e-6
    assert diagnostics['factorizations'] == 1


def test_inner_hinges_have_zero_reference_residual_including_fixed_neighbours():
    objective, poses = problem(extra=True)
    window = WindowObjective(objective, poses, range(9, 18))
    rows = ConstrainedRows(window)
    families = rows.evaluate(poses[window.span], True)
    matrix, bounds, _ = rows.inequalities(families, np.zeros(window.frames * window.width))
    assert np.all(bounds >= 0)
    assert np.all(matrix @ np.zeros(matrix.shape[1]) - bounds <= 0)
    assert any(f.label[0].endswith('ContinuityGuard') and f.locations[0][0] < 9 for f in families)
    assert any(f.label[0] == 'RootDomain' for f in families)


def test_complete_one_sided_hinge_gradient_matches_nonlinear_finite_difference():
    objective, poses = problem(extra=True)
    window = WindowObjective(objective, poses, range(9, 18))
    rows = ConstrainedRows(window)
    rng = np.random.default_rng(18)
    moved = window.coords.integrate(poses[window.span],
        rng.normal(scale=.008, size=(window.frames, window.width)), 1.)
    families = rows.evaluate(moved, True)
    matrix, bounds, _ = rows.inequalities(families, np.zeros(matrix_size := window.frames * window.width))
    gradient = np.asarray(matrix.T @ np.maximum(-bounds, 0.)).ravel()

    def merit(candidate):
        total = 0.
        for f in rows.evaluate(candidate, False):
            if f.ball:
                radius = np.sqrt(f.slack + np.einsum('ij,ij->i', f.residual, f.residual))
                excess = np.linalg.norm(f.residual, axis=1) - radius + rows.margins(f)
            else:
                excess = rows.margins(f) - f.slack
            total += .5 * np.maximum(excess, 0.) @ np.maximum(excess, 0.)
        return total

    tangent = rng.normal(scale=.03, size=matrix_size)
    epsilon = 1e-5
    plus = window.coords.integrate(moved, tangent.reshape(window.frames, window.width), epsilon)
    minus = window.coords.integrate(moved, tangent.reshape(window.frames, window.width), -epsilon)
    numeric = (merit(plus) - merit(minus)) / (2 * epsilon)
    assert float(gradient @ tangent) == pytest.approx(numeric, rel=2e-5, abs=2e-9)


def test_norm_cut_observes_direction_even_at_zero_norm_residual():
    objective, poses = problem()
    window = WindowObjective(objective, poses, range(8, 20))
    rows = ConstrainedRows(window)
    families = rows.evaluate(poses[window.span], True)
    direction = np.random.default_rng(7).normal(scale=.003, size=window.frames * window.width)
    matrix, bounds, _ = rows.inequalities(families, direction)
    cursor = 0
    for f in families:
        if f.ball:
            predicted = f.residual + (f.matrix @ direction).reshape(f.residual.shape)
            radius = np.sqrt(f.slack + np.einsum('ij,ij->i', f.residual, f.residual))
            np.testing.assert_allclose((matrix @ direction - bounds)[cursor:cursor + len(f.slack)],
                np.linalg.norm(predicted, axis=1) - radius + rows.margins(f), atol=1e-12)
        cursor += len(f.slack)


def test_crossing_hinge_matrix_keeps_nonzero_fixed_neighbour_constants():
    objective, poses = problem(extra=True)
    moved = poses.copy()
    moved[:, 7:] += .004
    window = WindowObjective(objective, moved, range(9, 18))
    full = WindowObjective(objective, moved, range(len(moved)))
    a, b = ConstrainedRows(window), ConstrainedRows(full)
    small = a.evaluate(moved[window.span], True)
    large = b.evaluate(moved, True)
    for f in small:
        if 'Continuity' not in f.label[0]:
            continue
        g = next(g for g in large if g.label == f.label)
        index = {g.identity(i): i for i in range(len(g.slack))}
        for i in range(len(f.slack)):
            j = index[f.identity(i)]
            assert f.slack[i] == g.slack[j]
            assert a.margins(f)[i] == b.margins(g)[j]


def test_box_rows_are_actual_step_intersection_without_a_new_guard():
    matrix, bounds = with_boxes(sparse.csr_matrix([[1., 1.]]), np.array([1.]),
                               np.array([0., -np.inf]), np.array([0., .2]))
    assert np.all(matrix @ [0., .1] <= bounds)
    assert np.any(matrix @ [.01, .1] > bounds)


def test_hinge_route_keeps_exact_guards_cost_and_reference(monkeypatch):
    from omniretargeting._fast import runtime_options
    from omniretargeting._fast.proposals import guards_of
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions(continuity_hinge_acquisition=True))
    objective, poses = problem()
    reference = objective.reference.tobytes()
    result = solve_local(objective, poses, range(8, 20), 2, .003)
    window = WindowObjective(objective, poses, range(8, 20))
    assert all(g.accepts(result['poses']) for g in guards_of(window)[1])
    assert all(r['after'] < r['cost'] for r in result['history'] if r['alpha'])
    assert objective.reference.tobytes() == reference
    assert result['constrained']['mode'] == 'continuity_hinge'


def test_active_route_keeps_zero_box_fixed_root_guards_and_commit(monkeypatch):
    from omniretargeting._fast import runtime_options
    from omniretargeting._fast.proposals import guards_of
    monkeypatch.setattr(runtime_options, 'OPTIONS', RuntimeOptions(active_set_acquisition=True))
    objective, poses = problem()
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
    assert all(r['after'] < r['cost'] for r in result['history'] if r['alpha'])
    assert result['constrained']['mode'] == 'active_set'
    assert result['commit']['accepted'] == (result['commit']['candidate_cost'] < result['commit']['cost'])
