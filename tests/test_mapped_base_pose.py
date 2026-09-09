import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting.data_sources.base import MotionData
from omniretargeting.data_sources.smplx import SmplxDataSource, _SMPLX_ROOT_OFFSET
from omniretargeting.main import select_mapped_source_base_pose


def test_adapter_calibrates_every_body_frame_without_changing_source(monkeypatch, tmp_path):
    rotations = Rotation.from_rotvec([[0, 0, 0], [0, np.pi / 2, 0]]).as_quat(scalar_first=True)
    raw = np.tile(rotations[:, None], (1, 22, 1))
    raw[:, 0] = [1, 0, 0, 0]  # Pelvis stays still while the trunk turns.
    positions = np.zeros((2, 22, 3))
    root = np.zeros((2, 3))
    saved = raw.copy()
    adapter = SmplxDataSource(tmp_path/'dummy.npz')
    monkeypatch.setattr(adapter, '_load_arrays', lambda _: (positions, raw, root, root, 25, {}))
    motion = adapter.load()
    aligned = motion.metadata['body_aligned_joint_orientations']
    expected = Rotation.from_quat(raw.reshape(-1,4), scalar_first=True).as_matrix() @ _SMPLX_ROOT_OFFSET
    np.testing.assert_allclose(Rotation.from_quat(aligned.reshape(-1,4), scalar_first=True).as_matrix(), expected, atol=1e-7)
    np.testing.assert_array_equal(raw, saved)
    np.testing.assert_array_equal(motion.positions, positions)
    # The adapter's explicit disable still takes precedence.
    adapter.use_smplx_base_pose = False; adapter._motion_data = None
    assert 'body_aligned_joint_orientations' not in adapter.load().metadata


@pytest.mark.parametrize('root_body,source', [('upper','Chest'), ('lower','Hips')])
def test_actual_compiled_free_body_selects_both_position_and_orientation(root_body, source):
    # Arbitrary body and source names: no G1 / clip / SMPL-X name matching.
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody><body name="{root_body}"><freejoint/>
        <geom type="sphere" size=".1"/></body></worldbody></mujoco>''')
    poses = np.array([[[0,0,1], [0,0,1.5]], [[.2,0,1], [.2,0,1.4]]])
    quats = np.tile([1.,0,0,0], (2,2,1));quats[1,1] = Rotation.from_euler('x', 150, degrees=True).as_quat(scalar_first=True)
    motion = MotionData(positions=poses, target_names=['Hips','Chest'], framerate=25,
                        metadata={'body_aligned_joint_orientations': quats})
    q,p,name = select_mapped_source_base_pose(motion,model,{'Hips':'lower','Chest':'upper'})
    i = motion.target_names.index(source)
    assert name == source
    np.testing.assert_array_equal(q,quats[:,i]);np.testing.assert_array_equal(p,poses[:,i])
    assert not np.shares_memory(q,quats) and not np.shares_memory(p,poses)


def test_unavailable_calibration_keeps_original_source_path():
    motion = MotionData(positions=np.zeros((2,2,3)), target_names=['a','b'])
    assert select_mapped_source_base_pose(motion, None, {}) is None


def test_position_anchor_is_independent_of_floating_body_orientation():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body name="upper"><freejoint/>
        <geom type="sphere" size=".1"/></body></worldbody></mujoco>''')
    positions=np.array([[[1,2,3],[1,2,3.3]]])
    orientations=np.tile([1.,0,0,0],(1,2,1))
    orientations[0,1]=Rotation.from_rotvec([0,.8,0]).as_quat(scalar_first=True)
    motion=MotionData(positions=positions,target_names=['Hip','Chest'],
                      metadata={'body_aligned_joint_orientations':orientations})
    quat,pos,name=select_mapped_source_base_pose(motion,model,{'Hip':'lower','Chest':'upper'},position_source='Hip')
    assert name=='Chest'
    np.testing.assert_allclose(quat,orientations[:,1])
    np.testing.assert_array_equal(pos,positions[:,0])


def test_mapped_base_frames_follow_explicit_output_resampling():
    quats = Rotation.from_rotvec([[0,0,0], [np.pi/2,0,0]]).as_quat(scalar_first=True)[:,None,:]
    motion = MotionData(positions=np.zeros((2,1,3)), target_names=['body'], framerate=25,
                        metadata={'body_aligned_joint_orientations':quats})
    resampled = motion.resample(50)
    aligned = resampled.metadata['body_aligned_joint_orientations']
    assert aligned.shape[0] == len(resampled.positions)
    np.testing.assert_allclose(Rotation.from_quat(aligned[1,0],scalar_first=True).as_rotvec(), [np.pi/4,0,0], atol=1e-7)
    np.testing.assert_array_equal(motion.metadata['body_aligned_joint_orientations'],quats)


def test_locked_initial_registration_uses_joint_anchor_not_model_translation():
    from unittest.mock import Mock, patch
    from omniretargeting.core import OmniRetargeter
    source=np.array([[[10.,20.,1.]]])
    translations=np.array([[10.,20.36,.9]])
    original_translations=translations.copy()
    model=Mock(nq=7,njnt=0);model.qpos0=np.array([0.,0.,.8,1.,0.,0.,0.])
    retargeter=OmniRetargeter.__new__(OmniRetargeter)
    retargeter.robot_model=model;retargeter.robot_data=Mock()
    retargeter.valid_joint_mapping={'Anchor':'base'};retargeter.valid_joint_names=['Anchor']
    retargeter.mapped_joint_indices=[0];retargeter.smplx_joint_indices={'Anchor':0};retargeter.smplx_joint_names=['Anchor']
    retargeter.base_orientation_config={'root_position':'Anchor'};retargeter.robot_height=1.
    retargeter.retargeting_config={'end_effector_priority':{'base_position_constraint':{
        'enabled':True,'use_initial_offset':True,'lock_initial_xy':True,
        'track_xy':True,'lock_xy':True,'track_z':False}}}
    task=Mock();task.frame.return_value={'contact':False,'contact_mode':'none'}
    with patch('omniretargeting.retargeting.GenericInteractionRetargeter') as constructor:
        solver=Mock();constructor.return_value=solver
        solver.retarget_frame.side_effect=lambda joints,q_init,**kwargs:q_init.copy()
        result=retargeter._perform_retargeting(source,Mock(),
            base_orientations=np.array([[1.,0.,0.,0.]]),base_translations=translations,
            end_effector_tasks={'marker':task})
        initial=solver.retarget_frame.call_args.args[1]
    np.testing.assert_array_equal(initial[:2],source[0,0,:2])
    np.testing.assert_array_equal(result[0,:2],source[0,0,:2])
    np.testing.assert_array_equal(translations,original_translations)
    assert initial[2]==translations[0,2]  # Existing independent height policy.


def test_profile_torso_policy_requires_reference_and_preserves_explicit_source_choices():
    from omniretargeting.main import _source_config_with_profile_torso
    motion=MotionData(positions=np.zeros((1,3,3)),target_names=['anchor','left','right'])
    profile={'torso_mapping':{'preset':'pelvis_shoulders','origin':'anchor',
                             'left_shoulder':'left','right_shoulder':'right','source_body':'anchor'}}
    assert _source_config_with_profile_torso(motion,{},profile)=={}
    motion.metadata['joint_rest_positions']=np.array([[0.,0.,0.],[0.,1.,1.],[0.,-1.,1.]])
    cfg=_source_config_with_profile_torso(motion,{},profile)
    assert cfg['torso_mapping']==profile['torso_mapping']
    assert cfg['lock_root_orientation_to_source'] is False
    for explicit in [{'torso_mapping':False},{'base_orientation_joint':'anchor'},
                     {'torso_mapping':{'source_body':'another'}}]:
        assert _source_config_with_profile_torso(motion,explicit,profile)==explicit


def test_root_predictor_advances_source_motion_without_locking_or_mutating_previous_output():
    from unittest.mock import Mock, patch
    from omniretargeting.core import OmniRetargeter
    source=np.array([[[10.,20.,1.]],[[11.,22.,2.]]])
    rotations=Rotation.from_rotvec([[0.,0.,0.],[0.,np.pi/2,0.]]).as_quat(scalar_first=True)
    rotations.setflags(write=False)
    model=Mock(nq=7,njnt=0);model.qpos0=np.array([0.,0.,.8,1.,0.,0.,0.])
    core=OmniRetargeter.__new__(OmniRetargeter);core.robot_model=model;core.robot_data=Mock()
    core.valid_joint_mapping={'Anchor':'base'};core.valid_joint_names=['Anchor'];core.mapped_joint_indices=[0]
    core.smplx_joint_indices={'Anchor':0};core.smplx_joint_names=['Anchor'];core.robot_height=1.
    core.base_orientation_config={'root_position':'Anchor'}
    core.retargeting_config={'end_effector_priority':{'source_root_prediction':True,'base_position_constraint':{
        'enabled':True,'use_initial_offset':True,'lock_initial_xy':True,
        'track_xy':True,'lock_xy':True,'track_z':False}}}
    task=Mock();task.frame.return_value={'contact':False,'contact_mode':'none'}
    previous=np.r_[source[0,0],Rotation.from_rotvec([0.,0.,.4]).as_quat(scalar_first=True)]
    with patch('omniretargeting.retargeting.GenericInteractionRetargeter') as constructor:
        solver=Mock();constructor.return_value=solver
        solver.retarget_frame.side_effect=[previous.copy(),previous.copy()]
        core._perform_retargeting(source,Mock(),base_orientations=rotations,end_effector_tasks={'marker':task})
        call=solver.retarget_frame.call_args_list[1];predicted=call.args[1]
    np.testing.assert_array_equal(predicted[:3],source[1,0])
    expected=Rotation.from_quat(rotations[1],scalar_first=True)*Rotation.from_rotvec([0.,0.,.4])
    np.testing.assert_allclose(Rotation.from_quat(predicted[3:7],scalar_first=True).as_matrix(),expected.as_matrix(),atol=1e-12)
    np.testing.assert_array_equal(call.kwargs['q_last'],previous)
    assert task.frame.call_count==2  # Original chronological plan is read once.
