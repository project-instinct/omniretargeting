import cvxpy as cp
import mujoco
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from omniretargeting.retargeting import GenericInteractionRetargeter, _source_orientation_step
from tests.test_contact_priority import ANKLE_ARM_XML


def payload(current, previous):
    return dict(forward_direction=current[:,0].copy(),up_direction=current[:,2].copy(),
                previous_forward_direction=previous[:,0].copy(),previous_up_direction=previous[:,2].copy(),
                contact=False,tracking_position=np.zeros(3))


def test_source_step_handles_fast_rotation_and_fixed_local_frame_calibration():
    previous=Rotation.from_euler('xyz',[.2,-.6,.9]).as_matrix()
    step=Rotation.from_rotvec([.4,-.2,.1])
    current=step.as_matrix()@previous
    local=Rotation.from_euler('xyz',[-.8,.4,.3]).as_matrix()
    for offset in (np.eye(3),local):
        data=payload(current@offset,previous@offset)
        original={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in data.items()}
        np.testing.assert_allclose(_source_orientation_step(data),step.as_rotvec(),atol=1e-12)
        for key in original:
            np.testing.assert_array_equal(data[key],original[key])
    reverse=_source_orientation_step(payload(previous,current))
    np.testing.assert_allclose(reverse,-step.as_rotvec(),atol=1e-12)


def test_one_normal_does_not_invent_full_orientation():
    assert _source_orientation_step({'up_direction':np.array([0,0,1])}) is None
    data=payload(np.eye(3),np.eye(3));data['previous_forward_direction']=data['previous_up_direction'].copy()
    assert _source_orientation_step(data) is None


@pytest.mark.parametrize('reference,actual,source,expected',[
    ('source',.45,.45,True),('absolute',.45,.45,False),
    ('source',-.45,.45,False),('source',.25,0.,False),('source',.04,0.,True),
])
def test_source_relative_cap_preserves_intentional_rotation_and_rejects_extra_motion(reference,actual,source,expected):
    model=mujoco.MjModel.from_xml_string(ANKLE_ARM_XML);data=mujoco.MjData(model)
    solver=GenericInteractionRetargeter(model,data,trimesh.creation.box(),{'base':'root','endpoint':'foot'},1.,
        valid_joint_names=['base','endpoint'],terrain_sample_seed=0,
        end_effector_config={'enabled':True,'effectors':{'effector':{
            'robot_body':'foot','track_orientation':True,'track_swing_orientation':True,
            'max_orientation_step':.1,'orientation_step_reference':reference,
        }}})
    data.qpos[7]=actual;mujoco.mj_forward(model,data)
    dqa=cp.Variable(len(solver.q_a_indices));dqa.value=np.zeros(len(solver.q_a_indices))
    target=payload(Rotation.from_rotvec([0,source,0]).as_matrix(),np.eye(3))
    constraints=solver._build_end_effector_objectives(dqa,{'effector':target},{'effector':np.eye(3)})[3]
    assert constraints
    violation=max(float(np.max(c.violation())) for c in constraints)
    assert (violation<1e-8) == expected


@pytest.mark.parametrize('contact_mode', ['none', 'sticking', 'sliding', 'transition'])
def test_solver_keeps_chronological_source_directions_read_only(contact_mode):
    model=mujoco.MjModel.from_xml_string(ANKLE_ARM_XML);data=mujoco.MjData(model)
    solver=GenericInteractionRetargeter(model,data,trimesh.creation.box(),{'base':'root','endpoint':'foot'},1.,
        valid_joint_names=['base','endpoint'],terrain_sample_seed=0,
        end_effector_config={'enabled':True,'effectors':{'effector':{
            'robot_body':'foot','track_orientation':True,'track_swing_orientation':True,
            'max_orientation_step':.1,
        }}})
    target=payload(Rotation.from_euler('xyz',[.2,-.3,.4]).as_matrix(),np.eye(3))
    target.update(contact=contact_mode!='none',contact_mode=contact_mode,
                  contact_position=np.zeros(3),surface_normal=np.array([0.,0.,1.]),confidence=1.)
    for value in target.values():
        if isinstance(value,np.ndarray):value.setflags(write=False)
    dqa=cp.Variable(len(solver.q_a_indices))
    solver._build_end_effector_objectives(dqa,{'effector':target},{'effector':np.eye(3)})
