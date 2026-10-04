"""World support anchors, role handoffs and surface-relative banded rows."""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
from omniretargeting._fast.soft_support_rows import SupportRows, SupportWeights, surface_roles
from omniretargeting._fast.soft_bulk import bulk_context
from omniretargeting._fast.runtime_options import RuntimeOptions
from omniretargeting._fast.barrier_assembly import scatter
from tests.test_fast_contact_guard import _acquired
from tests.test_fast_legacy_window import problem


def ranked(*, moving=False):
    obj,q=_acquired(surfaces=True,moving=moving)
    surf=obj.surfaces[0]
    surf['sticking'][:]=True
    surf['task'].surface_ids=np.zeros(surf['active'].shape,int)
    roles={n:np.full(len(q),'hold','U7') for n in obj.tasks.effectors}
    roles['foot'][4:14]='support'
    support=np.zeros(surf['active'].shape,bool)
    support[4:14]=True
    policy=SimpleNamespace(surface=surf['task'],roles=roles,
        surface_roles={'support':support,'hold':~support})
    obj.tasks.contact_phase=policy
    return obj,q


def test_support_only_anchors_and_observed_sticking_handoffs():
    obj,q=ranked()
    rows=SupportRows(obj)
    assert rows.diagnostics['support_anchor_points']>0
    for g in rows.groups:
        f,k=g['anchor']
        assert np.all(g['roles'][f,k]=='support')
        f,k=g['velocity']
        if g['kind']=='geom':assert 13 in f  # stencil crosses support -> hold
    obj.tasks.contact_phase.roles={n:np.full(len(q),'hold') for n in obj.tasks.effectors}
    obj.tasks.contact_phase.surface_roles={'hold':np.ones(obj.surfaces[0]['active'].shape,bool)}
    hold=SupportRows(obj)
    assert hold.diagnostics['support_anchor_points']==0
    assert hold.diagnostics['sticking_velocity_points']>0
    assert hold.diagnostics['contact_normal_points']>0


def test_moving_world_targets_no_rebase_and_frame_gaps():
    obj,q=ranked(moving=True)
    rows=SupportRows(obj)
    r,j,_=rows.evaluate(q)
    np.testing.assert_allclose(r,0.,atol=1e-10)
    assert rows.groups[0]['target'] is obj.surfaces[0]['targets']
    obj.surfaces[0]['task'].surface_ids[8:]+=1
    broken=SupportRows(obj)
    assert 7 not in broken.groups[0]['velocity'][0]
    moved=q.copy();moved[7,0]+=.01
    r2,_,_=rows.evaluate(moved,False)
    assert np.linalg.norm(r2)>0
    np.testing.assert_array_equal(rows.groups[0]['target'],obj.surfaces[0]['targets'])


def test_support_gradient_quaternion_and_psd_direct_band():
    obj,q=ranked(moving=True)
    rows=SupportRows(obj)
    rng=np.random.default_rng(940)
    q=obj.coords.integrate(q,rng.normal(scale=.02,size=(len(q),obj.width)),1.)
    direction=rng.normal(scale=.01,size=(len(q),obj.width))
    r,j,_=rows.evaluate(q)
    numeric=[]
    for a in (-1e-5,1e-5):
        rr,_,_=rows.evaluate(obj.coords.integrate(q,direction,a),False)
        numeric.append(rr@rr)
    assert (numeric[1]-numeric[0])/2e-5==pytest.approx(2*(j.T@r)@direction.ravel(),rel=3e-6,abs=1e-6)
    n=obj.frames*obj.width
    ab=np.zeros((2*obj.width,n),order='F');scatter(ab,j,1.)
    dense=np.zeros((n,n))
    for k in range(len(ab)):
        ids=np.arange(n-k);dense[ids+k,ids]=ab[k,:n-k];dense[ids,ids+k]=ab[k,:n-k]
    expected=(j.T@j).toarray()
    np.testing.assert_allclose(dense,expected,atol=1e-8,rtol=1e-12)
    assert direction.ravel()@dense@direction.ravel()>=0


def test_facade_original_physical_cost_and_opt_in_equivalence():
    assert not RuntimeOptions().soft_acquisition_v5
    obj,q=problem(extra=True)
    report=lambda:dict(evaluation_seconds=0.,evaluations=0.)
    with bulk_context(obj,q,report()) as facade:
        ev=facade.evaluate(q);original=facade.physical_cost(ev)
    with bulk_context(obj,q,report(),support_weights=SupportWeights()) as facade:
        ev=facade.evaluate(q)
        assert facade.physical_cost(ev)==pytest.approx(original,rel=1e-12)
        assert ev[4]['support_cost']>=0
        assert ev[2].shape[1]==obj.frames*obj.width
    assert not hasattr(obj,'_soft_bulk_signals')


def test_bad_sigmas_rejected():
    obj,_=problem()
    with pytest.raises(ValueError):SupportRows(obj,SupportWeights(anchor_sigma=0.))


def test_priority_scaling_gradient_keeps_unweighted_cost_and_fixed_budgets():
    obj,q=problem(extra=True)
    rng=np.random.default_rng(911)
    q=obj.coords.integrate(q,rng.normal(scale=.005,size=(len(q),obj.width)),1.)
    original=obj.evaluate(q)[0].cost
    bounds=[g.bounds for g in obj.candidate_guards if hasattr(g,'bounds')]
    report=dict(evaluation_seconds=0.,evaluations=0.,fitting_multiplier=.1,clearance_multiplier=4.)
    with bulk_context(obj,q,report,clearance_multiplier=40.,support_weights=SupportWeights()) as facade:
        ev=facade.evaluate(q)
        assert facade.physical_cost(ev)==pytest.approx(original,rel=1e-12)
        direction=rng.normal(scale=.01,size=(len(q),obj.width))
        numeric=[]
        for a in (-1e-5,1e-5):
            trial=facade.evaluate(obj.coords.integrate(q,direction,a),False)
            numeric.append(trial[0].cost)
        gradient=ev[0].gradient.ravel()+ev[2].T@ev[1]
        assert (numeric[1]-numeric[0])/2e-5==pytest.approx(2*gradient@direction.ravel(),rel=3e-5,abs=1e-5)
    assert bounds==[g.bounds for g in obj.candidate_guards if hasattr(g,'bounds')]


def test_active_observations_keep_holding_rows_at_zero_confidence():
    obj,q=ranked()
    surf=obj.surfaces[0]
    before=SupportRows(obj).evaluate(q)[1]
    surf['task'].weights=np.zeros_like(surf['task'].weights)
    surf['task'].confidence=np.zeros(surf['active'].shape)
    after=SupportRows(obj).evaluate(q)[1]
    np.testing.assert_array_equal(before.toarray(),after.toarray())
    rows=SupportRows(obj)
    assert np.all(rows.groups[0]['weights'][surf['active']]>0.)
    assert rows.diagnostics['holding_weights']=='sample_density_only_no_confidence_attenuation'


def test_contact_projection_holds_anchor_dependencies_bitwise(monkeypatch):
    from omniretargeting._fast import soft_projection
    obj, q = problem(extra=True)
    enabled = np.zeros(len(obj.coords.qi), bool)
    held = SimpleNamespace(enabled=enabled)
    def unexpected(*args, **kwargs):
        raise AssertionError('Held coordinates must not enter a numerical QP')
    monkeypatch.setattr(soft_projection, 'closest', unexpected)
    candidate = soft_projection.project_joints(obj, q, report={'_contact_program': held})
    np.testing.assert_array_equal(candidate, q)


def test_support_subset_preserves_original_targets_bounds_yield_and_roles():
    from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
    from omniretargeting._fast.soft_support_guard import support_guard
    obj, q = ranked()
    obj.tasks.contact_phase.role = lambda name, frame: str(obj.tasks.contact_phase.roles[name][frame])
    frozen = AcquisitionNormalGuard(obj, q)
    selected = support_guard(obj, frozen)
    assert selected.groups and selected.accepts(q)
    assert sum(g[5].sum() for g in selected.groups) < sum(g[5].sum() for g in frozen.groups)
    for name, group in zip(selected.names, selected.groups):
        old = frozen.groups[frozen.names.index(name)]
        for index in (1,2,3,4,6,7,8):
            assert group[index] is old[index]
        assert not group[5].flags.writeable
        assert not np.any(group[5] & ~old[5])
    obj.tasks.contact_phase.surface_roles = {'hold': np.ones(obj.surfaces[0]['active'].shape, bool)}
    obj.tasks.contact_phase.roles = {name: np.full(len(q), 'hold') for name in obj.tasks.effectors}
    assert not support_guard(obj, frozen).groups


def test_softer_touch_anchors_do_not_promote_observations_to_support():
    obj, q = ranked()
    rows = SupportRows(obj)
    for group in rows.groups:
        frames, samples = group['touch_anchor']
        assert np.all(group['roles'][frames, samples] != 'free')
        assert not np.any((group['roles'][frames, samples] == 'support') & group['sticking'][frames, samples])
    assert rows.weights.observed_anchor_sigma > rows.weights.anchor_sigma


def test_coordinate_layout_intervals_equal_independent_linear_programs():
    from scipy.optimize import linprog
    from omniretargeting._fast.soft_support_projection import ContactProjection
    model=ContactProjection.__new__(ContactProjection)
    model.cols=2
    model.objective=SimpleNamespace(coords=SimpleNamespace(qi=np.array([0,1])))
    model.target=np.array([[1.,2.],[3.,4.]])
    model.layout=sparse.csr_matrix([[2.,3.,0.,0.],[-1.,1.,0.,0.],
                                   [0.,0.,4.,0.],[0.,0.,-2.,5.]])
    model.layout_slack=np.array([.2,.4,.3,.5])
    result=model.target+np.array([[.01,-.02],[.03,.04]])
    lower,upper=model.boxes(0,result)
    matrix=model.layout.toarray()
    for frame in range(2):
        column=2*frame
        other=(result-model.target).ravel();other[column]=0.
        b=model.layout_slack+matrix@other
        for sign,expected in ((1.,lower[frame]),(-1.,upper[frame])):
            solved=linprog([sign],A_ub=-matrix[:,column:column+1],b_ub=b,
                           bounds=[(None,None)],method='highs')
            assert solved.success
            assert model.target[frame,0]+solved.x[0]==pytest.approx(expected,abs=1e-12)


def test_hold_anchor_strength_preserves_mechanics_priority():
    obj,q=ranked();rows=SupportRows(obj)
    assert rows.weights.anchor_sigma<rows.weights.hold_anchor_sigma<rows.weights.observed_anchor_sigma
    before=surface_roles(obj,obj.surfaces[0]).copy()
    rows.evaluate(q)
    np.testing.assert_array_equal(before,surface_roles(obj,obj.surfaces[0]))


def test_contact_projection_rolls_back_a_failed_coordinate_program(monkeypatch):
    from omniretargeting._fast import soft_projection
    from omniretargeting._fast.soft_projection_qp import ProjectionFailure
    from omniretargeting._fast.soft_support_projection import refine
    obj,q=problem(extra=True)
    def rejected(*args,**kwargs):raise ProjectionFailure('binding layout interval')
    monkeypatch.setattr(soft_projection,'project_joints',rejected)
    candidate,report=refine(obj,q,q,np.ones((len(q),obj.width)))
    np.testing.assert_array_equal(candidate,q)
    assert report['alpha']==0.
    assert report['stall']=='binding layout interval'
    assert '_contact_program' not in report
