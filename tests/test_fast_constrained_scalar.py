"""Scalar band projection matches an independent small QP and exact stencils."""
import numpy as np
from scipy import sparse
from omniretargeting._fast.constrained_scalar import project_one
from omniretargeting._fast.soft_projection_qp import closest


def test_native_scalar_band_matches_closest_qp_with_fixed_neighbour_constants():
    n=19
    matrices=[sparse.diags([-np.ones(n-k),np.ones(n-k)],[0,k],shape=(n-k,n)) for k in (1,2,3)]
    matrix=sparse.vstack([*matrices,*[-a for a in matrices],sparse.eye(n),-sparse.eye(n)],format='csr')
    bounds=np.r_[np.full(sum(a.shape[0] for a in matrices)*2,.003),np.full(2*n,.1)]
    # Crossing constraints contain a nonzero fixed neighbour in their RHS.
    bounds[0]=.001;bounds[sum(a.shape[0] for a in matrices)]=.0005
    target=.03*np.sin(np.arange(n)/2)
    point,iterations,fraction,_=project_one(matrix,bounds,target)
    expected=closest(target,matrix,np.full(len(bounds),-np.inf),bounds)
    assert np.max(matrix@point-bounds)<1e-12
    np.testing.assert_allclose(point,expected,atol=3e-5)
    assert iterations>0 and 0<=fraction<=1.


def test_scalar_zero_width_box_and_duplicate_intersection_are_exact():
    matrix=sparse.csr_matrix([[1.,0.],[-1.,0.],[1.,0.],[0.,1.],[0.,-1.]])
    point,*_=project_one(matrix,np.array([0.,0.,.1,.03,.02]),[.2,.1])
    np.testing.assert_allclose(point,[0.,.03],atol=1e-7)
