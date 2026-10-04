"""Full physical QP tangent cuts retain original cones and linear bounds."""
import numpy as np
from scipy import sparse
from omniretargeting._fast.soft_projection_linear_qp import solve


def test_qp_retains_cross_terms_equalities_and_box_activity():
    hessian=sparse.csr_matrix([[2.,.5],[.5,3.]])
    gradient=np.array([-4.,-3.])
    point=solve([0.,0.],sparse.eye(2),[.2,-1.],[.2,1.],
        quadratic=hessian,linear=gradient,report={})
    np.testing.assert_allclose(point,[.2,2.9/3],atol=1e-8)


def test_ball_cuts_match_tiny_intact_cone_and_feasible_zero_width_hold():
    ball=(sparse.csr_matrix([[1.,0.,0.],[0.,1.,0.]]),np.zeros((1,2)),np.array([1e-5]))
    report={}
    point=solve([3.,4.,1.],sparse.eye(3),[-np.inf,-np.inf,.2],[np.inf,np.inf,.2],
        balls=[ball],quadratic=sparse.eye(3),linear=np.array([-3.,-4.,-1.]),report=report)
    assert np.linalg.norm(point[:2]) <= 1e-5+1e-8
    np.testing.assert_allclose(point,[6e-6,8e-6,.2],atol=1e-8)
    assert report['linear_qp_solves'] >= 2


def test_inaccurate_linear_exit_routes_to_original_conic_solver(monkeypatch):
    from types import SimpleNamespace
    from omniretargeting._fast import soft_projection_linear_qp
    monkeypatch.setattr(soft_projection_linear_qp.osqp,'OSQP',lambda:SimpleNamespace(
        setup=lambda **kw:None,solve=lambda **kw:SimpleNamespace(x=np.array([2.]),
        info=SimpleNamespace(status='maximum iterations',iter=1))))
    report={}
    point=solve([1.],sparse.eye(1),[0.],[.3],quadratic=sparse.eye(1),
                linear=np.array([-1.]),report=report)
    np.testing.assert_allclose(point,[.3],atol=1e-8)
    assert report['linear_qp_fallbacks'] == ['linear_qp_original_residual']
