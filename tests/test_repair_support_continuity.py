"""Evidence masks, independent immutable point budgets and exact Jacobians."""
from types import SimpleNamespace
import mujoco
import numpy as np
from omniretargeting._fast.kinematics import TrajectoryKinematics
from omniretargeting._fast.repair_support import SupportContinuity
from omniretargeting._fast.repair_limits import LinearBounds, NUMERICAL_SLACK
from omniretargeting.motion_budgets import windowed_budget

def problem():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="region_a"><joint axis="0 0 1"/><geom size=".03"/>
        <body name="tip_a" pos=".7 0 0"><geom size=".03"/></body></body>
      <body name="region_b" pos="0 1 0"><joint axis="0 0 1"/><geom size=".03"/>
        <body name="tip_b" pos=".7 0 0"><geom size=".03"/></body></body>
      </worldbody></mujoco>''')
    frames = 15
    q = np.zeros((frames, 2)); q[:, 0] = .001 * np.arange(frames)**2
    q[:, 1] = .2 * np.sin(np.arange(frames))
    roles = np.array(['free']*6 + ['support']*3 + ['free']*6)
    policy = SimpleNamespace(roles={'anything':roles,'elsewhere':np.full(frames,'hold')})
    effectors = [{'name':n,'body':model.body(b).id} for n,b in [('anything','tip_a'),('elsewhere','tip_b')]]
    tasks = SimpleNamespace(contact_phase=policy, effectors={n:{'tracking_offset':np.array([.1,.02,0])} for n in policy.roles})
    obj = SimpleNamespace(model=model,frames=frames,effectors=effectors,tasks=tasks,surfaces=[])
    kin = TrajectoryKinematics(model,[0,1]); kin.evaluate(q)
    return obj,kin,q

def test_evidence_handoff_and_separate_point_budgets_are_immutable():
    obj,kin,q = problem(); guard=SupportContinuity(obj,kin,slack=NUMERICAL_SLACK)
    group=next(g for g in guard.groups if g[0]=='anything' and g[3]==2)
    assert group[4].tolist()==list(range(4,9))
    point=kin.points(np.array([group[1]]),group[2][None])[:,0]
    expected=windowed_budget(np.linalg.norm(np.diff(point,n=2,axis=0),axis=1))[group[4]]
    np.testing.assert_array_equal(expected,group[6])
    before=group[6].copy()
    changed=q.copy();changed[7,0]+=.02;kin.evaluate(changed,False)
    valid,report=guard.check(kin);assert not valid and report['point']=='anything'
    np.testing.assert_array_equal(before,group[6])
    kin.evaluate(q,False);assert guard.check(kin)[0]
    # Large motion of another supported endpoint never increases this budget.
    assert group[6].max()<.01

def test_cross_frame_cones_match_nonlinear_finite_difference():
    obj,kin,q=problem();guard=SupportContinuity(obj,kin,slack=NUMERICAL_SLACK)
    bounds=LinearBounds(q.size);guard.add(bounds,kin)
    direction=np.random.default_rng(17).normal(size=q.shape)*1e-6
    predicted=[r+(j@direction.ravel()).reshape(r.shape) for j,r,a in bounds.norms]
    kin.evaluate(q+direction,False)
    for group,linear in zip(guard.groups,predicted):
        _,body,offset,_,_,op,_=group
        actual=(op@kin.points(np.array([body]),offset[None]).ravel()).reshape(-1,3)
        np.testing.assert_allclose(actual,linear,atol=1e-11,rtol=0)

def test_no_role_and_follow_do_not_invent_support():
    obj,kin,q=problem();obj.tasks.contact_phase=None
    assert not SupportContinuity(obj,kin,slack=NUMERICAL_SLACK).groups
    obj.tasks.contact_phase=SimpleNamespace(roles={e['name']:np.full(len(q),'follow') for e in obj.effectors})
    assert not SupportContinuity(obj,kin,slack=NUMERICAL_SLACK).groups

def test_joint_budget_alone_cannot_certify_a_supported_endpoint():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <joint type="slide" axis="1 0 0"/><geom size=".02"/>
      <body name="supported_region"><joint type="slide" axis="1 0 0"/>
        <geom size=".02"/></body></body></worldbody></mujoco>''')
    frames=17; signal=.02*np.sin(np.arange(frames)*.4)
    q=np.stack([signal,-signal],axis=1)
    policy=SimpleNamespace(roles={'region':np.full(frames,'support')})
    obj=SimpleNamespace(model=model,frames=frames,surfaces=[],
        effectors=[dict(name='region',body=model.body('supported_region').id)],
        tasks=SimpleNamespace(contact_phase=policy,effectors={'region':{'tracking_offset':np.zeros(3)}}))
    kin=TrajectoryKinematics(model,[0,1]);kin.evaluate(q)
    guard=SupportContinuity(obj,kin,slack=NUMERICAL_SLACK)
    changed=q.copy();changed[8,1]+=.00001
    for order in (1,2,3):
        assert np.all(np.abs(np.diff(changed,n=order,axis=0))<=windowed_budget(np.diff(q,n=order,axis=0)))
    kin.evaluate(changed,False)
    assert not guard.check(kin)[0]
