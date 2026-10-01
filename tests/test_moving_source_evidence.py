"""Open scans need source-side evidence and must not hide deep local crossings."""
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from omniretargeting.moving_evidence import source_object_evidence
from omniretargeting.moving_scene import MovingSceneQuery


def panel():
    return trimesh.Trimesh([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
                          [[0, 1, 2], [0, 2, 3]], process=False)


def robot():
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="arbitrary_contact_region"><freejoint/><geom type="box" size=".03 .04 .05" mass="1"/></body>
    </worldbody></mujoco>''')


def spec(evidence=True):
    result = dict(mesh=panel(), translations=np.zeros((1, 3)),
                  rotations=np.eye(3)[None], scales=np.ones(1))
    if evidence:
        result['collision_evidence'] = dict(points=np.array([[0., 0, 0]]),
                                            normals=np.array([[0., 0, 1]]), radius=.4)
    return result


def test_deep_crossing_has_full_exit_when_reverse_samples_miss_entire_robot():
    model = robot()
    q = model.qpos0.copy();q[2] = -.2
    unknown = MovingSceneQuery(model, [spec(False)], 1).query(q, 0)
    assert unknown['forward_depth'].max() == 0
    assert unknown['reverse_depth'].max() == 0
    query = MovingSceneQuery(model, [spec()], 1)
    measured = query.query(q, 0)
    assert measured['unknown_components'] == 1
    assert measured['open_evidence_depth'].max() == pytest.approx(.25)
    row = min(measured['rows'], key=lambda x:x['clearance'])
    assert row['kind'] == 'observed_open_surface'
    np.testing.assert_allclose(row['direction'], [0, 0, 1], atol=1e-9)
    query.prepare_trajectory(q[None])
    np.testing.assert_allclose(query.query(q, 0)['forward_depth'], measured['forward_depth'])


def test_source_evidence_rotates_and_scales_with_object_not_world_vertical():
    model = robot();obj = spec()
    rotation = Rotation.from_rotvec([.4, -.3, .2]).as_matrix()
    obj.update(rotations=rotation[None], translations=np.array([[.2, -.3, .7]]), scales=np.array([1.5]))
    q = model.qpos0.copy();q[:3] = rotation @ [0, 0, -.3] + obj['translations'][0]
    q[3:7] = Rotation.from_matrix(rotation).as_quat(scalar_first=True)
    measured = MovingSceneQuery(model, [obj], 1).query(q, 0)
    row = min(measured['rows'], key=lambda x:x['clearance'])
    assert row['clearance'] == pytest.approx(-.35)
    np.testing.assert_allclose(row['direction'], rotation[:, 2], atol=1e-9)


@pytest.mark.parametrize('translation', [[.7, 0, -.2], [1.1, 0, -.2], [0, 0, -.7]])
def test_evidence_is_not_extrapolated_to_unobserved_patch_or_past_edge(translation):
    model = robot();q=model.qpos0.copy();q[:3]=translation
    query=MovingSceneQuery(model,[spec()],1)
    result=query.query(q,0)
    assert result['open_evidence_depth'].max() == 0
    assert result['unknown_components'] == 1


def test_conflicted_free_sides_remain_unknown():
    obj=spec();obj['collision_evidence'].update(points=np.zeros((2,3)), normals=np.array([[0,0,1.],[0,0,-1.]]))
    model=robot();q=model.qpos0.copy();q[2]=-.2
    result=MovingSceneQuery(model,[obj],1).query(q,0)
    assert result['open_evidence_depth'].max()==0


def test_evidence_uses_material_side_and_does_not_promote_penetrating_source():
    rotations=Rotation.from_euler('z', [[0],[90]], degrees=True).as_matrix()
    local_normal=np.tile([1.,0,0],(2,3,1))
    points=np.arange(18,dtype=float).reshape(2,3,3)/100
    obs=dict(surface_poses=((rotations,np.zeros((2,3)),np.array([2.,3.])),),
             surface_ids=np.zeros((2,3),int),material_environment=points,
             normals=np.einsum('tij,tnj->tni',rotations,local_normal),
             confidence=np.array([[1.,1.,.1],[1.,1.,1.]]),
             clear_side=np.array([[True,False,True],[True,False,True]]))
    table=SimpleNamespace(observation=obs,valid=np.ones((2,3),bool),active=np.ones((2,3),bool))
    original=spec(False);before=original['mesh'].vertices.copy()
    result=source_object_evidence([original],table)
    evidence=result[0]['collision_evidence']
    assert len(evidence['points'])==3
    np.testing.assert_allclose(evidence['normals'],np.tile([1.,0,0],(3,1)),atol=1e-12)
    assert evidence['radius']==pytest.approx(.05)
    assert 'collision_evidence' not in original
    np.testing.assert_array_equal(original['mesh'].vertices,before)
    np.testing.assert_array_equal(obs['material_environment'],points)


def test_losing_side_evidence_cannot_count_as_resolving_penetration():
    from omniretargeting.moving_scene import moving_regression_reasons

    model=robot();query=MovingSceneQuery(model,[spec()],1)
    q=model.qpos0.copy();q[2]=-.2
    before=query.query(q,0)
    q[0]=.7
    after=query.query(q,0)
    assert 'moving_local_evidence_lost' in moving_regression_reasons(before,after)


def test_exact_material_cache_preserves_geometry_and_fresh_jacobians():
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body pos="-.1 0 -.1"><joint name="swing" type="hinge" axis="0 1 0"/>
        <geom type="box" pos=".05 0 0" size=".02 .03 .04" mass="1"/></body>
      <body pos=".1 0 -.1"><joint name="other" type="slide" axis="0 0 1"/>
        <geom type="box" size=".02 .03 .04" mass="1"/></body>
    </worldbody></mujoco>''')
    obj=spec()
    obj.update(translations=np.array([[0.,0,0],[.01,0,.01]]),
               rotations=Rotation.from_rotvec([[0.,0,0],[.02,0,0]]).as_matrix(),scales=np.array([1.,1.1]))
    cached=MovingSceneQuery(model,[obj],2)
    fresh=MovingSceneQuery(model,[obj],2)
    pose=model.qpos0.copy()
    for index,time in enumerate([0,0,0,.5,.5,1,1,0]):
        pose[0]=.01*index
        actual=cached.query(pose,time)
        fresh.material_cache.clear()
        expected=fresh.query(pose,time)
        for key in ['forward_depth','reverse_depth','open_evidence_depth']:
            np.testing.assert_array_equal(actual[key],expected[key])
        for key in ['open_known_samples','open_penetrating_samples','unknown_open_samples','failures']:
            assert actual[key]==expected[key]
        assert len(actual['rows'])==len(expected['rows'])
        for a,b in zip(actual['rows'],expected['rows']):
            assert a.keys()==b.keys()
            for key in a:
                np.testing.assert_equal(a[key],b[key])
    # Preparing whole-trajectory winding data uses the same material contract.
    poses=np.array([[.03,0],[.04,0]])
    cached.prepare_trajectory(poses)
    for frame,pose in enumerate(poses):
        a=cached.query(pose,frame);fresh.material_cache.clear();b=fresh.query(pose,frame)
        np.testing.assert_array_equal(a['forward_depth'],b['forward_depth'])
