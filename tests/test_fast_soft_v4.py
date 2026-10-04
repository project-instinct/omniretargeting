"""Soft-v4 changes internal normal semantics, never final budgets/layout."""
import numpy as np
import pytest
from omniretargeting._fast import soft_acquisition_v4 as mod
from omniretargeting._fast.runtime_options import RuntimeOptions
from omniretargeting._fast.soft_bulk import bulk_context, SoftObjective, run_bulk
from tests.test_fast_legacy_window import problem
from tests.test_fast_contact_guard import _acquired
from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard


def test_default_off_final_scope_and_exception_restore(monkeypatch):
    assert not RuntimeOptions().soft_acquisition_v4
    obj,q=problem(extra=True)
    guards=obj.candidate_guards
    normal=guards[-1]
    reference=obj.reference.tobytes()
    monkeypatch.setattr(normal,'accepts',lambda *a,**k:False)
    projected,report=mod.project_final(obj,q,q,measure_costs=False)
    np.testing.assert_array_equal(projected,q)
    assert obj.candidate_guards is guards
    assert not report['protected_normal_projection']
    assert reference==obj.reference.tobytes()
    with pytest.raises(RuntimeError):
        with mod.final_guards(obj):
            assert normal not in obj.candidate_guards
            raise RuntimeError('injected')
    assert obj.candidate_guards is guards


@pytest.mark.parametrize('surface',[False,True])
def test_acquired_max_bounds_immutable_targets_and_no_further_regression(surface):
    obj,q=_acquired(surfaces=surface)
    obj.tasks.contact_phase=None
    for eff in obj.effectors:eff['normal'][:]=[0,0,1]
    for surf in obj.surfaces:surf['normals'][:]=[0,0,1]
    old=AcquisitionNormalGuard(obj,q)
    saved=[g[7].copy() for g in old.groups]
    acquired=q.copy();acquired[:,2]+=.02
    assert not old.accepts(acquired)
    new=mod.acquired_normal_guard(old,acquired)
    assert new.accepts(acquired)
    assert new.accepts(acquired,frames=range(2,6))
    further=acquired.copy();further[:,2]+=.01
    assert not new.accepts(further)
    assert not new.accepts(further,frames=range(2,6))
    for a,b,c in zip(old.groups,new.groups,saved):
        np.testing.assert_array_equal(a[7],c)
        for index in (1,2,3,4,5,6,8):assert a[index] is b[index]
        assert not b[7].flags.writeable
        assert np.all(b[7]>=a[7])
    assert not new.diagnostics(acquired)['fixed_pre_acquisition_reference']
    assert old.diagnostics(acquired)['fixed_pre_acquisition_reference']


def test_global_clearance_scaling_preserves_original_cost_and_gradient_fd():
    obj,q=problem(extra=True)
    reference=obj.reference.tobytes()
    rows=[]
    for multiplier in (1.,16.):
        with bulk_context(obj,q,dict(evaluation_seconds=0.,evaluations=0.),clearance_multiplier=multiplier) as f:
            ev=f.evaluate(q)
            rows.append((ev,f.physical_cost(ev)))
            if multiplier==16:
                direction=np.random.default_rng(89).normal(scale=.001,size=(len(q),obj.width))
                numeric=[]
                for alpha in (-1e-5,1e-5):
                    v=f.evaluate(obj.coords.integrate(q,direction,alpha),False)
                    numeric.append(f.acceptance_cost(v)+v[4]['guard_cost'])
                gradient=ev[0].gradient.ravel()+ev[2].T@ev[1]
                assert (numeric[1]-numeric[0])/2e-5==pytest.approx(2*gradient@direction.ravel(),rel=3e-5,abs=.01)
    first,second=rows
    assert first[1]==pytest.approx(second[1],rel=1e-12)
    assert second[0][4]['priority_extra']==pytest.approx(15*second[0][4]['clearance_cost'])
    assert obj.reference.tobytes()==reference
    assert not hasattr(obj,'_soft_clearance_multiplier')


def test_transaction_rolls_back_no_polish_and_only_final_callback(monkeypatch):
    obj,q=problem()
    private=q.copy();private[5,7]+=.3
    callbacks=[]
    monkeypatch.setattr(mod,'run_bulk',lambda *a,**k:(private,[],{'seconds':.1}))
    monkeypatch.setattr(mod,'project_final',lambda *a,**k:(private,{'seconds':.1}))
    monkeypatch.setattr(mod,'commit_verdict',lambda *a,**k:{'accepted':False})
    result,history,report=mod.optimize_soft_acquisition_v4(obj,q,iterations=10,
        callback=lambda x,r:callbacks.append(x))
    np.testing.assert_array_equal(result,q)
    assert history==[] and report['polish_window_iterations']==0
    assert all(np.array_equal(x,q) for x in callbacks)
    assert obj.soft_acquired_normal_guard.accepts(result)
