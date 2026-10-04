"""Source-depth yield and generic persistent-world chain recovery rows."""
import numpy as np
import pytest
from dataclasses import replace
from types import SimpleNamespace
from omniretargeting._fast.soft_swing_rows import SwingRows, target_yield
from omniretargeting._fast.soft_bulk import bulk_context
from omniretargeting._fast.soft_support_rows import SupportWeights
from tests.test_fast_soft_support import ranked


class Plane:
    def query(self, points, gradient=True):
        points=np.asarray(points);value=points[...,2]
        return (value,np.broadcast_to([0.,0.,1.],points.shape).copy()) if gradient else value


def test_target_own_depth_yields_only_normal_and_never_modifies_source():
    obj=SimpleNamespace(_clearance_fields=lambda:[('scene',Plane(),None,None,None)])
    target=np.array([[2.,3.,-.04],[1.,4.,.1]]);before=target.copy()
    depth,normal=target_yield(obj,target,[0,1])
    np.testing.assert_allclose(depth,[.04,0.])
    surface=target+depth[:,None]*normal
    np.testing.assert_array_equal(surface[:,:2],target[:,:2])
    np.testing.assert_allclose(surface[:,2],[0.,.1])
    np.testing.assert_array_equal(target,before)


def test_unknown_geometry_creates_no_yield_allowance():
    class Unknown(Plane):
        def query(self,point,gradient=True):
            return np.full(len(point),np.nan),np.full_like(point,np.nan)
    obj=SimpleNamespace(_clearance_fields=lambda:[('scene',Unknown(),None,None,None)])
    depth,normal=target_yield(obj,np.array([[0.,0.,-.04]]),[0])
    np.testing.assert_array_equal(depth,[0.]);np.testing.assert_array_equal(normal,np.zeros((1,3)))


def swing_problem():
    obj,q=ranked()
    name=obj.effectors[0]['name'];task=obj.tasks.effectors[name]['task']
    obj.tasks.effectors[name]['task']=replace(task,preserve_world_tracking=True)
    obj.effectors[0]['share'][:]=0.
    obj._clearance_fields=lambda:[('scene',Plane(),None,None,None)]
    return obj,q


@pytest.mark.parametrize('contact_sigma',[None,.01])
def test_world_recovery_gradient_root_quaternion_and_fixed_targets(contact_sigma):
    obj,q=swing_problem();rows=SwingRows(obj,include_contacts=True,contact_tangent_sigma=contact_sigma)
    assert rows.groups
    original=obj.effectors[0]['tracking'].copy()
    direction=np.random.default_rng(14).normal(scale=.003,size=(len(q),obj.width))
    r,j,_=rows.evaluate(q)
    numeric=[]
    for alpha in (-1e-5,1e-5):
        rr,_,_=rows.evaluate(obj.coords.integrate(q,direction,alpha),False);numeric.append(rr@rr)
    assert (numeric[1]-numeric[0])/2e-5==pytest.approx(2*(j.T@r)@direction.ravel(),rel=2e-5,abs=1e-6)
    np.testing.assert_array_equal(obj.effectors[0]['tracking'],original)
    assert not rows.movable[:,:6].any()


def test_world_recovery_facade_reports_original_physical_cost():
    from tests.test_fast_legacy_window import problem
    obj,q=problem(extra=True);obj._clearance_fields=lambda:[]
    for spec in obj.tasks.effectors.values():
        spec['task']=replace(spec['task'],preserve_world_tracking=True)
    original=obj.evaluate(q)[0].cost
    report=dict(evaluation_seconds=0.,evaluations=0.)
    with bulk_context(obj,q,report,support_weights=SupportWeights(),source_yield=True,world_recovery=True) as facade:
        value=facade.evaluate(q)
        assert facade.physical_cost(value)==pytest.approx(original,rel=1e-12,abs=1e-9)
        assert value[4]['world_recovery_cost']>=0.
    assert not hasattr(obj,'_soft_bulk_signals')


def test_added_tracking_excludes_observed_support_and_hold_phases():
    obj,q=swing_problem()
    name=obj.effectors[0]['name']
    task=obj.tasks.effectors[name]['task']
    roles=obj.tasks.contact_phase.roles[name]
    expected=np.flatnonzero(~(np.asarray(task.contact_mask,bool)&(np.asarray(roles)!='free')))
    rows=SwingRows(obj)
    selected=[g for g in rows.groups if g['name']==name]
    if len(expected):
        np.testing.assert_array_equal(selected[0]['frames'],expected)
    else:
        assert not selected


def test_contact_source_yield_gradient_keeps_raw_targets_and_reported_gap():
    from omniretargeting._fast.soft_support_rows import SupportRows
    obj,q=ranked()
    obj._clearance_fields=lambda:[('scene',Plane(),None,None,None)]
    obj._field_index=lambda ids:np.zeros(np.asarray(ids).shape,int)
    for surface in obj.surfaces:surface['targets'][...,2]=-.04
    for eff in obj.effectors:eff['contact'][...,2]=-.04
    original=[s['targets'].copy() for s in obj.surfaces]
    rows=SupportRows(obj,yield_targets=True)
    assert rows.diagnostics['source_normal_yield']
    assert rows.diagnostics['source_yield_max_m']==pytest.approx(.04)
    for group in rows.groups:
        np.testing.assert_array_equal(group['fitting_target'][...,:2],group['target'][...,:2])
        assert not group['fitting_target'].flags.writeable
    direction=np.random.default_rng(37).normal(scale=.003,size=(len(q),obj.width))
    r,j,stats=rows.evaluate(q)
    values=[]
    for alpha in (-1e-5,1e-5):
        residual,_,_=rows.evaluate(obj.coords.integrate(q,direction,alpha),False)
        values.append(residual@residual)
    assert (values[1]-values[0])/2e-5==pytest.approx(2*(j.T@r)@direction.ravel(),rel=2e-5)
    assert stats['observed_contact_normal_max_m']>=.04
    for before,surface in zip(original,obj.surfaces):
        np.testing.assert_array_equal(before,surface['targets'])


@pytest.mark.parametrize('sigma',[0.,float('nan'),float('inf')])
def test_invalid_recovery_sigmas_rejected(sigma):
    obj,_=ranked()
    with pytest.raises(ValueError,match='finite and positive'):
        SwingRows(obj,tangent_sigma=sigma)


def test_yielded_holding_does_not_follow_source_velocity_inside_floor():
    from omniretargeting._fast.soft_support_rows import SupportRows
    obj,q=ranked();q[:]=q[0]
    obj._clearance_fields=lambda:[('scene',Plane(),None,None,None)]
    obj._field_index=lambda ids:np.zeros(np.asarray(ids).shape,int)
    surf=obj.surfaces[0]
    kin=obj.window_kinematics(q,range(len(q)),with_jacobian=False)
    surf['targets'][:]=kin.geom_points(surf['geoms'],surf['local'])
    surf['targets'][...,2]=(-.04-.01*np.sin(np.arange(len(q))*.2))[:,None]
    original=surf['targets'].copy()
    rows=SupportRows(obj,yield_targets=True);rows.groups=rows.groups[:1]
    r,_,stats=rows.evaluate(q,False)
    # A stationary robot point over a stationary plane earns no fitting
    # velocity penalty from the source's normal motion below that plane.
    rows.groups[0]['velocity']=(np.empty(0,int),np.empty(0,int))
    without,_,_=rows.evaluate(q,False)
    assert r@r==pytest.approx(without@without,rel=1e-12)
    assert stats['sticking_slip_max_mps']>.01  # raw audit is never hidden
    np.testing.assert_array_equal(original,surf['targets'])


def test_sticking_velocity_is_tangential_and_leaves_normal_clearance_free():
    from omniretargeting._fast.soft_support_rows import SupportRows
    obj,q=ranked();q[:]=q[0]
    rows=SupportRows(obj);rows.groups=rows.groups[:1]
    group=rows.groups[0]
    group['normal'][:]=[0.,0.,1.]
    group['fitting_target']=group['target'].copy()
    group['fitting_target'][...,2]+=np.arange(len(q))[:,None]*.01
    for family in ('anchor','touch_anchor','normal_rows'):
        group[family]=(np.empty(0,int),np.empty(0,int))
    residual,matrix,_=rows.evaluate(q)
    np.testing.assert_allclose(residual,0.,atol=1e-12)
    direction=np.zeros((len(q),obj.width))
    direction[:,2]=np.arange(len(q))*.01
    np.testing.assert_allclose(matrix@direction.ravel(),0.,atol=1e-12)
