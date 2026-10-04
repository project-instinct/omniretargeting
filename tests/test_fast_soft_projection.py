"""Closest-point projection honors frozen budgets and nonlinear certificates."""

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.soft_projection import certify, project_bulk, project_joints
from omniretargeting._fast.soft_projection_qp import closest
from tests.test_fast_legacy_window import problem


def test_sparse_closest_qp_with_boxes_equalities_and_whole_vector_ball():
    matrix = sparse.eye(3, format='csr')
    ball = (sparse.csr_matrix([[1., 0., 0.], [0., 1., 0.]]), np.zeros((1, 2)), np.ones(1))
    point = closest([3., 4., 1.], matrix, [-np.inf, -np.inf, .2], [np.inf, np.inf, .2], balls=[ball])
    np.testing.assert_allclose(point, [.6, .8, .2], atol=2e-8)


def test_scaled_projection_preserves_tiny_soc_and_large_equality_units():
    matrix = sparse.eye(3, format='csr')
    ball = (sparse.csr_matrix([[1., 0., 0.], [0., 1., 0.]]), np.zeros((1, 2)), np.array([1e-5]))
    point = closest([3., 4., 1.], matrix, [-np.inf, -np.inf, .2], [np.inf, np.inf, .2],
        balls=[ball], variable_scale=[.03, .03, .1])
    np.testing.assert_allclose(point, [6e-6, 8e-6, .2], atol=2e-9)


def test_numerical_qp_exit_requires_original_linear_feasibility(monkeypatch):
    from types import SimpleNamespace
    from omniretargeting._fast import soft_projection_qp
    result = SimpleNamespace(status='NumericalError', x=[.2], iterations=3, r_prim=1e-9, r_dual=0.)
    monkeypatch.setattr(soft_projection_qp.clarabel, 'DefaultSolver',
        lambda *a:SimpleNamespace(solve=lambda:result))
    matrix = sparse.eye(1, format='csr')
    np.testing.assert_array_equal(closest([1.], matrix, [0.], [.3]), [.2])
    result.x = [.4]
    with pytest.raises(soft_projection_qp.ProjectionFailure):
        closest([1.], matrix, [0.], [.3])


def test_projection_quadratic_retains_cross_coordinate_physical_curvature():
    # Holding the second coordinate changes the closest point of the first
    # through the physical cross term. Diagonal-only distance loses it.
    point = closest([1.], sparse.eye(1, format='csr'), [-10.], [10.],
        quadratic=sparse.csr_matrix([[2.]]), linear=np.array([-3.]), variable_scale=[.03])
    np.testing.assert_allclose(point,[1.5],atol=1e-8)


def test_projection_metric_band_matches_original_dense_curvature():
    from omniretargeting._fast.soft_projection_metrics import measurements
    from scipy.linalg import block_diag
    objective, poses = problem(extra=True)
    value = objective.evaluate(poses)
    expected = block_diag(*value[0].hessian) + (value[2].T@value[2]).toarray()
    expected += np.diag(1e-6*np.maximum(np.diag(expected),1.))
    with measurements(objective, poses) as tracker:
        np.testing.assert_allclose(tracker.system.toarray(),expected,atol=1e-8,rtol=1e-12)


def test_production_projection_uses_local_regions_and_exact_final_certificate():
    objective, poses = problem(extra=True)
    moved = poses.copy()
    moved[8:20,7] += .08
    projected, report = project_bulk(objective,moved,poses,measure_costs=True)
    assert not certify(objective,projected)
    assert report['cost_by_guard_family']
    assert 'geometry_projection' in report or not report['geometry_violation_frames']


@pytest.mark.parametrize('correction', [False, True])
def test_joint_projection_intersects_distinct_fixed_reference_budgets(correction):
    objective, poses = problem(extra=True)
    for guard in objective.candidate_guards[:2]:
        guard.correction = correction
    # Different allowances must be intersected, never averaged or rebased.
    guard = objective.candidate_guards[0]
    for name in guard.bounds:
        if name.startswith('joints'):
            guard.bounds[name] *= 1.01
    original = {name: value.tobytes() for name, value in guard.bounds.items()}
    desired = poses.copy()
    desired[8:20, 7:] += .07
    projected = project_joints(objective, desired, report={})
    assert all(g.accepts(projected) for g in objective.candidate_guards[:2])
    assert any(np.any(projected[:, 7:] != poses[:, 7:]) for _ in [0])
    assert {name: value.tobytes() for name, value in guard.bounds.items()} == original
    # A convex blend with the feasible fixed trajectory remains feasible.
    blend = projected.copy()
    blend[:, 7:] = .5 * projected[:, 7:] + .5 * poses[:, 7:]
    assert all(g.accepts(blend) for g in objective.candidate_guards[:2])


def test_projection_certifies_contacts_layout_quaternions_and_keeps_reference():
    objective, poses = problem(extra=True)
    tangent = np.zeros((len(poses), objective.width))
    tangent[:, objective.coords.scalar_cols] = .04 * np.sin(np.arange(len(poses))[:, None] / 9.)
    tangent[8:16, :6] = .005
    desired = objective.coords.integrate(poses, tangent, 1.)
    reference = objective.reference.tobytes()
    projected, report = project_bulk(objective, desired, poses)
    assert not certify(objective, projected)
    assert report['removed_correction_fraction'] < 1.1
    assert np.linalg.norm(projected[:, 3:7], axis=1) == pytest.approx(np.ones(len(poses)), abs=1e-12)
    assert objective.reference.tobytes() == reference
    assert not report['exact_guard_failures']


def test_zero_width_boxes_fixed_roots_and_zero_correction_survival():
    objective, poses = problem()
    objective.fixed_root = poses[:, :7].copy()
    objective.lower = np.broadcast_to(objective.lower, (len(poses), len(objective.coords.qi))).copy()
    objective.upper = np.broadcast_to(objective.upper, objective.lower.shape).copy()
    objective.lower[:, -1] = objective.upper[:, -1] = poses[:, objective.coords.qi[-1]]
    desired = poses.copy()
    desired[8:16, 7] += .01
    projected, report = project_bulk(objective, desired, poses)
    np.testing.assert_array_equal(projected[:, :7], poses[:, :7])
    np.testing.assert_array_equal(projected[:, objective.coords.qi[-1]], poses[:, objective.coords.qi[-1]])
    assert not certify(objective, projected)
    _, zero = project_bulk(objective, poses, poses)
    assert zero['retained_correction_fraction'] == 1.


def test_failed_coupled_sqp_never_returns_unchecked_crossing_stencils(monkeypatch):
    from omniretargeting._fast import soft_projection
    objective, poses = problem()
    desired = poses.copy()
    desired[10, 7] += .3
    monkeypatch.setattr(soft_projection, 'project_regions', lambda obj, q, target, report: desired.copy())
    projected, report = project_bulk(objective, desired, poses)
    assert report['fallback_failures']
    assert not certify(objective, projected)
    assert report['fallback_fraction'] < 1.
