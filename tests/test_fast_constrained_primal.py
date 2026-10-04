"""Feasible KKT working sets do not activate incompatible violated faces."""
import numpy as np
from scipy import sparse
from omniretargeting._fast.constrained_primal import solve_primal
from tests.test_fast_constrained import active_diagnostics,band


def test_primal_polish_matches_dense_qp_and_releases_wrong_duals():
    physical=np.array([[2.,1.],[1.,2.]])
    matrix=sparse.csr_matrix([[1.,0.],[-1.,0.],[0.,1.],[0.,-1.]])
    bounds=np.array([.1,2.,2.,2.])
    diagnostics=active_diagnostics()
    direction=solve_primal(band(physical),np.array([-3.,-.2]),matrix,bounds,
                           np.array([.15,.05]),diagnostics)
    np.testing.assert_allclose(direction,[.1,.05],atol=1e-8)
    assert np.max(matrix@direction-bounds)<1e-8


def test_dependent_and_opposing_box_faces_keep_feasible_kkt_point():
    matrix=sparse.csr_matrix([[1.,0.],[-1.,0.],[1.,0.],[0.,1.]])
    diagnostics=active_diagnostics()
    direction=solve_primal(band(np.eye(2)),np.array([-3.,-2.]),matrix,
        np.array([0.,0.,1.,.2]),np.array([.1,.5]),diagnostics)
    np.testing.assert_allclose(direction,[0.,.2],atol=1e-8)
    assert diagnostics['dropped_rows']>0


def test_b_reserve_keeps_feasible_origin_without_hinge_springs():
    from omniretargeting._fast.constrained_rows import ConstrainedRows
    from omniretargeting._fast.legacy_window_qp import WindowObjective
    from tests.test_fast_legacy_window import problem
    objective,poses=problem(extra=True)
    window=WindowObjective(objective,poses,range(9,18))
    rows=ConstrainedRows(window)
    families=rows.evaluate(poses[window.span],True)
    matrix,bounds,_=rows.inequalities(families,np.zeros(window.frames*window.width),inner=False)
    assert np.min(bounds)>-1e-12
    _,hinges,_=rows.inequalities(families,np.zeros(window.frames*window.width))
    assert np.all(bounds>=hinges-1e-12)
