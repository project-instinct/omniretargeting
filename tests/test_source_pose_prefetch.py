"""Source prefetch reuse requires exact immutable input equality."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from omniretargeting._core.source_pose_prefetch import SourcePosePrefetch, source_key
from tests._contact_fixtures import _task


def _inputs():
    q=np.zeros((3,4));p=np.zeros((3,2,3));task=_task(np.zeros((3,3)))
    return [q,p,['a','b'],{'a':'x'},np.zeros((3,3)),{'foot':task},{},{},
            dict(framerate=30.,environment_points=np.zeros((3,1,3)),
                 root_orientation_targets=None,body_orientation_targets=[])]


def test_key_includes_bits_order_and_each_source_phase_input():
    from copy import deepcopy
    inputs=_inputs();expected=source_key(*inputs)
    assert expected is not None and source_key(*deepcopy(inputs))==expected
    for index in (0,1,4):
        other=deepcopy(inputs);other[index].flat[0]=-0.
        assert source_key(*other)!=expected
    other=deepcopy(inputs);other[-1]['framerate']=31.
    assert source_key(*other)!=expected
    other=deepcopy(inputs);other[5]['foot'].tracking_positions[0,0]+=1.
    assert source_key(*other)!=expected
    other=deepcopy(inputs);other[8]['body_orientation_targets']=[object()]
    assert source_key(*other) is None


def test_temporary_contact_phase_is_removed_from_source_key():
    inputs=_inputs();expected=source_key(*inputs)
    inputs[5]['foot']=replace(inputs[5]['foot'],contact_phase=object())
    assert source_key(*inputs)==expected


def test_observed_contacts_invalidate_height_registration_prefetch():
    # Joint fitting ignores contact roles, but the translation-only height
    # registration now reads the observed sole contact mask. Reusing the old
    # result after a mask change would register to a different support plane.
    inputs=_inputs();expected=source_key(*inputs)
    inputs[5]['foot']=replace(inputs[5]['foot'],contact_mask=np.ones(3,bool))
    assert source_key(*inputs)!=expected


class Result:
    def __init__(self,result):self.result=result;self.closed=False
    def receive(self):self.closed=True;return self.result
    def close(self):self.closed=True


def test_mismatch_does_not_deliver_callbacks_or_result(capsys):
    future=SourcePosePrefetch.__new__(SourcePosePrefetch)
    future.job=Result(dict(key=b'old',output='discard me',callbacks=[(None,{})]))
    assert future.consume(b'new',lambda *_:(_ for _ in ()).throw(AssertionError())) is None
    assert future.job is None and capsys.readouterr().out==''


def test_consumption_preserves_callback_order_and_reports_wait(capsys):
    future=SourcePosePrefetch.__new__(SourcePosePrefetch)
    future.job=Result(dict(key=b'same',output='source output\n',callbacks=[(None,{'id':i}) for i in range(3)],stage={'seconds':2.}))
    records=[];result=future.consume(b'same',lambda pose,record:records.append(record['id']))
    assert records==[0,1,2] and result['stage']['precomputed_seconds']==2.
    assert result['stage']['seconds']>=0. and result['stage']['precompute_reused']
    assert capsys.readouterr().out=='source output\n'


def test_inactive_prefetch_creates_no_worker(monkeypatch):
    import omniretargeting._core.source_pose_prefetch as module
    monkeypatch.setattr(module,'ForkJob',lambda *_:(_ for _ in ()).throw(AssertionError()))
    owner=SimpleNamespace()
    with SourcePosePrefetch(owner,None,np.zeros((128,1,3)),{},None,None,None,None,False):
        assert not hasattr(owner,'_source_pose_prefetch')


def test_surface_coverage_is_inert_in_source_tasks():
    from omniretargeting._fast.reference import kinematic_tracking_tasks
    inputs=_inputs();expected=source_key(*inputs)
    inputs[5]['foot']=replace(inputs[5]['foot'],surface_managed=True,
                             surface_managed_mask=np.array([True,False,True]))
    assert source_key(*inputs)==expected
    assert not kinematic_tracking_tasks(inputs[5],{})['foot'].managed_frames.any()


def test_completed_jacobians_are_rebuilt_bitwise_after_transport_eviction():
    import mujoco
    from omniretargeting._fast.kinematics import TrajectoryKinematics
    from omniretargeting._core.source_pose_prefetch import compact_source_transport
    model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint axis="0 1 0"/><geom size=".1"/><body pos="0 0 1"><geom size=".1"/></body></body></worldbody></mujoco>')
    kin=TrajectoryKinematics(model,np.arange(model.nv));poses=np.array([[.1],[.2],[.3]])
    kin.evaluate(poses,with_jacobian=True,native=False)
    expected={name:getattr(kin,name).tobytes() for name in ('xpos','xmat','jacp','jacr')}
    compact_source_transport(SimpleNamespace(kin=kin),SimpleNamespace(kin=kin))
    assert not hasattr(kin,'jacp') and not hasattr(kin,'jacr')
    kin.evaluate(poses,with_jacobian=True,native=False)
    assert all(getattr(kin,name).tobytes()==value for name,value in expected.items())
