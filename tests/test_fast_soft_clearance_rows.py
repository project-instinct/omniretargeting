"""Clearance recovery derivatives, immutable entry and rejected-trial credit."""
from types import SimpleNamespace
import numpy as np
import pytest
from omniretargeting._fast.soft_clearance_rows import ClearanceRows
from tests.test_fast_legacy_window import problem


class Plane:
    def gradient(self, point):
        return np.broadcast_to([0., 0., 1.], point.shape).copy()


def plane_problem(monkeypatch):
    obj, q = problem(extra=True)
    geom = int(obj.model.body(obj.effectors[0]['body']).geomadr[0])
    if geom < 0:geom=1
    plane = Plane()
    obj.shell_geoms = np.array([geom])
    obj.self_collision = None
    obj._clearance_fields = lambda: [('scene_depth_m', plane, None, None, None)]
    local = np.array([.015, -.02, -.03])
    def deepest(*args):
        point = obj.kin.geom_xpos[:, geom]+np.einsum('fij,j->fi',obj.kin.geom_xmat[:, geom],local)
        return point[:, 2, None], np.arange(len(q)), np.zeros(len(q),int), point, point
    obj._deepest_samples = deepest
    obj._plane_clearance = lambda: (np.full((len(q),1),np.inf),np.zeros((len(q),1,3)))
    from omniretargeting._fast import soft_clearance_rows
    monkeypatch.setattr(soft_clearance_rows, 'prepare', lambda obj: None)
    return obj, q


def test_material_clearance_gradient_includes_root_quaternion(monkeypatch):
    obj, q = plane_problem(monkeypatch)
    rows = ClearanceRows(obj,q,{})
    direction = np.random.default_rng(42).normal(scale=.004,size=(len(q),obj.width))
    _, families = rows.measure(q,True)
    eps = 1e-5
    plus = rows.measure(obj.coords.integrate(q,direction,eps),False)[1][0].slack
    minus = rows.measure(obj.coords.integrate(q,direction,-eps),False)[1][0].slack
    np.testing.assert_allclose((plus-minus)/(2*eps),families[0].matrix@direction.ravel(),rtol=2e-7,atol=1e-10)


def test_trials_never_rebase_clearance_limits_and_new_pairs_get_zero_depth():
    rows = ClearanceRows.__new__(ClearanceRows)
    rows.limits = {}
    key = ('self', 2, 3, 7)
    rows.commit({key: -.02})
    assert not rows.check({key: -.018})
    assert rows.check({key: -.03})
    assert rows.limits[key] == .02
    rows.commit({key: -.018})
    assert rows.limits[key] == .018
    assert rows.check({key: -.019})
    assert rows.check({('self', 4, 8, 9): -.001})
    assert not rows.check({('self', 4, 8, 9): .001})


def test_partial_geometry_rows_equal_full_rows_with_absolute_identities():
    from tests.test_fast_backend import _floor
    obj,q=problem(extra=True)
    obj.scene_sdf=_floor(.1)
    obj.self_collision=None
    rows=ClearanceRows(obj,q,{})
    full,ff=rows.measure(q,True)
    scope=range(7,19)
    part,pf=rows.measure(q,True,scope=scope)
    assert part
    for key,value in part.items():
        assert scope.start<=key[2]<scope.stop
        assert value==pytest.approx(full[key],abs=1e-12)
    source={key:(family.slack[i],family.matrix.getrow(i))
            for family in ff for i,key in enumerate(family.locations)}
    for family in pf:
        for i,key in enumerate(family.locations):
            slack,jac=source[key]
            assert family.slack[i]==pytest.approx(slack,abs=1e-12)
            np.testing.assert_allclose(family.matrix.getrow(i).toarray(),jac.toarray(),rtol=1e-12,atol=1e-12)


def test_maximum_depth_cuts_keep_geometry_families_and_original_rows():
    from scipy import sparse
    from omniretargeting._fast.barrier_rows import scalar
    keys=[('field',0,0,1),('field',0,1,1),('self',0,2,3)]
    values=dict(zip(keys,[-.02,-.04,.005]))
    matrix=sparse.eye(3,format='csr')
    first=scalar(np.zeros(2),matrix[:2],('RecoveryClearance','field',0),keys[:2])
    second=scalar(np.zeros(1),matrix[2:],('RecoveryClearance','self'),keys[2:])
    cuts=ClearanceRows.maximum_faces(values,[first,second])
    np.testing.assert_allclose(cuts[0].slack,[.02,0.])
    np.testing.assert_allclose(cuts[1].slack,[.005])
    np.testing.assert_array_equal(first.slack,np.zeros(2))
    assert cuts[0].matrix is first.matrix
    assert cuts[0].locations is first.locations
