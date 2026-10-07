"""Trial cuts retain frozen allowances and translate to origin coordinates."""
import numpy as np
from scipy import sparse
from omniretargeting._fast.acquisition_soc import translated_bounds, _override_bounds


def test_interval_translation_and_untouched_rows():
    a=sparse.eye(2,format='csr'); j=2*a
    original=([(a,np.array([-1.,-1.]),np.array([1.,1.]))],[])
    trial=([(j,np.array([-2.,-2.]),np.array([-.2,1.]))],[])
    (rows,_),changed=translated_bounds(original,trial,np.array([.6,.5]))
    matrix,lo,hi=rows[0]
    assert changed==1
    np.testing.assert_array_equal(matrix.toarray(),np.diag([2.,1.]))
    np.testing.assert_allclose(lo,[-.8,-1.])
    np.testing.assert_allclose(hi,[1.-1e-6,1.])
    # At p=d the trial bound is still violated; correction must move inward.
    assert (matrix@np.array([.6,.5]))[0]>hi[0]


def test_cone_translation_and_reserve():
    a=sparse.eye(6,format='csr');j=2*a
    r=np.array([[0.,0.,0.],[.2,0.,0.]])
    trial_r=np.array([[1.2,0.,0.],[.4,0.,0.]])
    original=([],[(a,r,np.ones(2))]); trial=([],[(j,trial_r,np.ones(2))])
    d=np.array([.5,0.,0.,.1,0.,0.])
    (_,cones),changed=translated_bounds(original,trial,d)
    matrix,residual,allowed=cones[0]
    assert changed==1
    np.testing.assert_array_equal(matrix.toarray(),np.diag([2,2,2,1,1,1]))
    np.testing.assert_allclose(residual,[[.2,0,0],[.2,0,0]])
    np.testing.assert_allclose(allowed,[1.-1e-6,1.])
    np.testing.assert_allclose((matrix@d).reshape(2,3)[0]+residual[0],trial_r[0])


def test_lower_side_tightens_and_origin_is_preserved():
    a=sparse.eye(1,format='csr')
    origin=([(a,np.array([-1.]),np.array([1.]))],[])
    trial=([(a,np.array([.1]),np.array([2.1]))],[])
    (rows,_),_=translated_bounds(origin,trial,np.array([-1.1]))
    np.testing.assert_allclose(rows[0][1],[-1.+1e-6])
    np.testing.assert_array_equal(origin[0][0][1],[-1.])


def test_bounds_override_restores_after_exception():
    class Guard:
        def repair_bounds(self,poses): return 'original'
    guard=Guard()
    try:
        with _override_bounds(guard,'trial'):
            assert guard.repair_bounds(None)=='trial'
            raise ValueError
    except ValueError: pass
    assert guard.repair_bounds(None)=='original'
    assert 'repair_bounds' not in guard.__dict__


def test_all_rows_translates_unviolated_trial_without_loosening_allowance():
    a=sparse.eye(1,format='csr'); j=2*a
    origin=([(a,np.array([-1.]),np.array([1.]))],[])
    trial=([(j,np.array([-2.]),np.array([.1]))],[])
    (rows,_),changed=translated_bounds(origin,trial,np.array([.4]),all_rows=True)
    assert changed==1
    np.testing.assert_allclose(rows[0][0].toarray(),[[2.]])
    np.testing.assert_allclose(rows[0][1],[-1.2])
    np.testing.assert_allclose(rows[0][2],[.9])


def test_rejected_correction_keeps_original_feasible_fraction(monkeypatch):
    from types import SimpleNamespace
    from omniretargeting._fast import solver, shape_step, whole_canonical
    from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
    from omniretargeting._fast.acquisition_soc import conic_proposal

    guard = AcquisitionNormalGuard.__new__(AcquisitionNormalGuard)
    guard.accepts = lambda q: bool(q.max() <= .5)
    guard.diagnostics = lambda q: {'accepted': guard.accepts(q)}
    def bounds(q):
        return [(sparse.eye(1, format='csr'), np.array([-1.-q.item()]),
                 np.array([.5-q.item()]))], []
    guard.repair_bounds = bounds
    coords = SimpleNamespace(
        integrate=lambda q, d, alpha: q + alpha*d,
        difference=lambda q, trial: trial-q)
    objective = SimpleNamespace(coords=coords, frames=1, width=1,
                                project=lambda q: q.copy())
    directions = iter([np.array([1.]), np.array([.8])])
    monkeypatch.setattr(shape_step, 'bounded_shape_step', lambda *a: next(directions))
    monkeypatch.setattr(whole_canonical, 'supported', lambda obj: True)
    monkeypatch.setattr(solver, 'backtrack', lambda *a: (np.array([[.5]]), .5))
    result = conic_proposal(objective, np.array([[0.]]), None, None, None, 1., [guard])
    np.testing.assert_array_equal(result[0], [1.])
    np.testing.assert_array_equal(result[1], [[.5]])
    assert result[2] == .5
    assert objective._acquisition_soc_events[0]['correction_solves'] == 1
    assert not objective._acquisition_soc_events[0]['corrected_full']
    assert guard.repair_bounds is bounds


def test_mismatched_family_counts_fail_instead_of_dropping_rows():
    import pytest
    with pytest.raises(ValueError):
        translated_bounds(([], []), ([(sparse.eye(1), np.zeros(1), np.ones(1))], []),
                          np.zeros(1))


def test_projection_metric_targets_original_full_step():
    from types import SimpleNamespace
    from omniretargeting._fast.acquisition_soc import projection_metric
    coords = SimpleNamespace(bounds=lambda *a: (np.array([[-.2, -.1]]),
                                                np.array([[.2, .1]])))
    objective = SimpleNamespace(coords=coords, cfg={'angular_step_rad': .2,
                                                   'translation_step_m': .1})
    step = np.array([.1, .05])
    metric, gradient = projection_metric(objective, None, step)
    assert metric.shape == (1, 2)
    np.testing.assert_allclose(metric[0]*step + gradient, 0.)
    np.testing.assert_allclose(metric[0], [25., 100.])


def test_root_transport_matches_origin_step_finite_difference():
    import mujoco
    from omniretargeting._sequence.coordinates import Coordinates
    from omniretargeting._fast.acquisition_soc import transport_trial_bounds
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/>'
                                          '<geom type="sphere" size=".1"/></body></worldbody></mujoco>')
    coords = Coordinates(model)
    origin = np.array([[0., 0., 1., 1., 0., 0., 0.]])
    d = np.array([[.1, -.1, 0., .4, -.3, .2]])
    trial = coords.integrate(origin, d, 1.)
    jac = sparse.eye(6, format='csr')
    bounds = ([(jac, -np.ones(6), np.ones(6))], [])
    rows, _ = transport_trial_bounds(coords, origin, trial, d.ravel(), bounds)
    transport = rows[0][0]
    perturbation = np.array([[.3, -.2, .1, .1, .2, -.3]])
    epsilon = 1e-5
    actual = coords.difference(trial, coords.integrate(origin, d+epsilon*perturbation, 1.)).ravel()/epsilon
    np.testing.assert_allclose(transport@perturbation.ravel(), actual, atol=1e-6)
    assert np.linalg.norm(transport.toarray()[3:, 3:]-np.eye(3)) > .1


def test_second_projection_uses_new_trial_and_respects_two_solve_cap(monkeypatch):
    from types import SimpleNamespace
    from omniretargeting._fast import solver, shape_step, whole_canonical
    from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
    from omniretargeting._fast.acquisition_soc import conic_proposal
    guard = AcquisitionNormalGuard.__new__(AcquisitionNormalGuard)
    guard.accepts = lambda q: bool(q.max() <= .7)
    guard.diagnostics = lambda q: {'accepted': guard.accepts(q)}
    guard.repair_bounds = lambda q: ([(sparse.eye(1, format='csr'),
                                      np.array([-1.-q.item()]), np.array([.7-q.item()]))], [])
    coords = SimpleNamespace(integrate=lambda q, d, a: q+a*d,
                             difference=lambda q, t: t-q,
                             bounds=lambda *a: (-np.ones((1, 1)), np.ones((1, 1))))
    objective = SimpleNamespace(coords=coords, frames=1, width=1,
                                project=lambda q: q.copy(),
                                cfg={'angular_step_rad': 1., 'translation_step_m': 1.},
                                evaluate=lambda q, **kw: [SimpleNamespace(cost=10-q.item())])
    directions = iter([np.array([1.]), np.array([.9]), np.array([.6])])
    monkeypatch.setattr(shape_step, 'bounded_shape_step', lambda *a: next(directions))
    monkeypatch.setattr(whole_canonical, 'supported', lambda obj: True)
    monkeypatch.setattr(solver, 'backtrack', lambda *a: (np.array([[.5]]), .5))
    step, candidate, alpha = conic_proposal(objective, np.zeros((1, 1)), None, None,
                                          None, 1., [guard], mode='soc_projection2')
    np.testing.assert_allclose(step, [.6])
    np.testing.assert_allclose(candidate, [[.6]])
    assert alpha == 1.
    event = objective._acquisition_soc_events[0]
    assert event['correction_solves'] == 2
    assert event['corrected_full']
