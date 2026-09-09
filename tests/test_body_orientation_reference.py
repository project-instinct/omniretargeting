import cvxpy as cp
import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from omniretargeting.retargeting import GenericInteractionRetargeter


XML = '''<mujoco><compiler angle="radian"/><worldbody>
  <body name="base"><freejoint/><geom type="sphere" size=".03"/>
    <body name="segment" pos=".1 0 0"><joint name="hinge" axis="0 1 0" range="-2 2"/>
      <geom type="sphere" size=".03"/>
      <body name="tip" pos=".2 0 0"><geom type="sphere" size=".03"/></body>
    </body>
    <body name="side" pos="0 .2 .1"><geom type="sphere" size=".03"/></body>
  </body></worldbody></mujoco>'''


def make_retargeter(scene, *, relative=True):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    names = ['root', 'bend', 'end', 'lateral']
    mapping = dict(zip(names, ['base','segment','tip','side']))
    retargeter = GenericInteractionRetargeter(
        model, data, scene, mapping, .5, terrain_sample_seed=0,
        valid_joint_names=names, hard_penetration_constraint=True,
        end_effector_config={
            '_base_orientation_source_joint':'root',
            'body_orientation':{'_source_orientations_available':True,
                                'frame':'parent_relative' if relative else 'world',
                                'calibration_iterations':3},
        },
    )
    return retargeter, names, mapping


def test_position_anchor_orientation_weight_uses_source_contract():
    residuals = []
    for weight in (1.0, 4.0):
        solver, names, _ = make_retargeter(trimesh.creation.box())
        solver.end_effector_config["_position_anchor_source_joint"] = "bend"
        solver.end_effector_config["body_orientation"]["position_anchor_weight"] = weight
        solver._setup_body_orientations()
        source = np.tile([1.0, 0.0, 0.0, 0.0], (len(names), 1))
        solver.configure_source_orientation_reference(names, source)
        source[1:3] = Rotation.from_rotvec([0.0, 0.3, 0.0]).as_quat(scalar_first=True)
        dqa = cp.Variable(len(solver.q_a_indices))
        dqa.value = np.zeros(len(solver.q_a_indices))
        residuals.append(solver._build_body_orientation_residual(dqa, source).value)
    assert np.linalg.norm(residuals[0]) > 0.1
    np.testing.assert_allclose(residuals[1], 2.0 * residuals[0], atol=1e-12)


def test_bind_calibration_does_not_encode_incidental_core_twist(monkeypatch):
    solver, names, mapping = make_retargeter(trimesh.creation.box())
    solver.end_effector_config["_position_anchor_source_joint"] = "bend"
    solver._setup_body_orientations()
    model, data = solver.robot_model, solver.robot_data
    original = data.qpos.copy()
    positions = np.array([data.xpos[model.body(mapping[n]).id] for n in names])
    source = np.tile([1.0, 0.0, 0.0, 0.0], (len(names), 1))
    fitted = model.qpos0.copy()
    fitted[3:7] = Rotation.from_rotvec([0.0, 0.4, 0.0]).as_quat(scalar_first=True)
    fitted[7] = 0.7
    monkeypatch.setattr(solver, "_optimize_configuration", lambda *a, **k: fitted.copy())
    solver.configure_source_orientation_reference(
        names, source, positions, reference_base_orientation=source[0],
        preserve_core_reference=True,
    )
    for name in ("root", "bend"):
        np.testing.assert_allclose(solver._robot_orientation_references[name], np.eye(3), atol=1e-12)
    np.testing.assert_array_equal(data.qpos, original)
    np.testing.assert_array_equal(source, np.tile([1.0, 0.0, 0.0, 0.0], (len(names), 1)))
    # An arbitrary first-frame reference retains its fitted pose semantics.
    solver.configure_source_orientation_reference(
        names, source, positions, reference_base_orientation=source[0],
        preserve_core_reference=False,
    )
    assert np.linalg.norm(solver._robot_orientation_references["bend"] - np.eye(3)) > 0.5


def test_reference_calibration_does_not_depend_on_actual_scene():
    references = []
    for location in ([0,0,0], [10,20,-5]):
        mesh = trimesh.creation.box(extents=[2,2,2]);mesh.apply_translation(location)
        solver,names,mapping = make_retargeter(mesh)
        original_q = solver.robot_data.qpos.copy()
        original_scene_points = solver.terrain_points.copy()
        positions = np.array([solver.robot_data.xpos[solver.robot_model.body(mapping[name]).id] for name in names])
        solver.configure_source_orientation_reference(
            names, np.tile([1.,0,0,0],(4,1)), positions,
            reference_base_orientation=np.array([1.,0,0,0]),
        )
        references.append(solver._robot_orientation_references)
        assert solver.hard_penetration_constraint  # Runtime collision policy restored.
        np.testing.assert_array_equal(solver.robot_data.qpos, original_q)
        np.testing.assert_array_equal(solver.terrain_points, original_scene_points)
    for name in references[0]:
        np.testing.assert_allclose(references[0][name], references[1][name], atol=1e-12)


def test_relative_pose_is_invariant_to_whole_body_rotation_and_has_no_root_gradient():
    solver,names,_ = make_retargeter(trimesh.creation.box())
    solver.configure_source_orientation_reference(names,np.tile([1.,0,0,0],(4,1)))
    model,data = solver.robot_model,solver.robot_data
    dqa = cp.Variable(len(solver.q_a_indices));dqa.value = np.zeros(len(solver.q_a_indices))
    desired = .45
    global_source = Rotation.from_euler('xyz',[.8,-.7,1.1])
    source = np.tile(global_source.as_quat(scalar_first=True),(4,1))
    source[1:3] = (global_source*Rotation.from_rotvec([0,desired,0])).as_quat(scalar_first=True)
    values = []
    for global_robot in (Rotation.identity(),Rotation.from_euler('xyz',[-1.1,.9,-2.])):
        data.qpos[:]=model.qpos0
        data.qpos[3:7]=global_robot.as_quat(scalar_first=True)
        data.qpos[7]=.15
        mujoco.mj_forward(model,data)
        residual=solver._build_body_orientation_residual(dqa,source)
        values.append(residual.value.copy())
        # The same root perturbation acts on parent and child and cancels.
        for qa,qi in enumerate(solver.q_a_indices):
            if qi>=7:continue
            step=np.zeros(len(solver.q_a_indices));step[qa]=.1;dqa.value=step
            np.testing.assert_allclose(residual.value,values[-1],atol=1e-12)
        dqa.value=np.zeros(len(solver.q_a_indices))
    np.testing.assert_allclose(values[0],values[1],atol=1e-12)
    data.qpos[7]=desired;mujoco.mj_forward(model,data)
    np.testing.assert_allclose(solver._build_body_orientation_residual(dqa,source).value,0,atol=1e-12)


def test_relative_hinge_orientation_jacobian_matches_finite_difference():
    solver,names,_ = make_retargeter(trimesh.creation.box())
    solver.configure_source_orientation_reference(names,np.tile([1.,0,0,0],(4,1)))
    model,data=solver.robot_model,solver.robot_data
    data.qpos[3:7]=Rotation.from_euler('xyz',[1.,.8,-1.]).as_quat(scalar_first=True)
    data.qpos[7]=.2;mujoco.mj_forward(model,data)
    source=np.tile([1.,0,0,0],(4,1))
    dqa=cp.Variable(len(solver.q_a_indices));dqa.value=np.zeros(len(solver.q_a_indices))
    residual=solver._build_body_orientation_residual(dqa,source);value=residual.value.copy()
    qa=int(np.flatnonzero(solver.q_a_indices==7)[0]);step=np.zeros(len(solver.q_a_indices));step[qa]=1.;dqa.value=step
    analytic=residual.value-value;dqa.value=np.zeros(len(solver.q_a_indices))
    actual=[]
    for sign in [-1,1]:
        data.qpos[7]=.2+sign*1e-6;mujoco.mj_forward(model,data)
        actual.append(solver._build_body_orientation_residual(dqa,source).value.copy())
    np.testing.assert_allclose(analytic,(actual[1]-actual[0])/2e-6,atol=1e-8)


def test_orientation_override_exposes_terminal_rotation_axes_without_moving_position_mapping():
    model=mujoco.MjModel.from_xml_string('''<mujoco><compiler angle="radian"/><worldbody>
      <body name="floating"><freejoint/><geom type="sphere" size=".1"/>
       <body name="axis_x"><joint axis="1 0 0"/><geom type="sphere" size=".03"/>
        <body name="axis_y"><joint axis="0 1 0"/><geom type="sphere" size=".03"/>
         <body name="terminal"><joint axis="0 0 1"/><geom type="sphere" size=".03"/>
         </body></body></body></body></worldbody></mujoco>''')
    data=mujoco.MjData(model)
    mapping={'source_root':'floating','source_limb':'axis_y'}
    solver=GenericInteractionRetargeter(model,data,trimesh.creation.box(),mapping,1.,
        end_effector_config={'_base_orientation_source_joint':'source_root',
            'body_orientation':{'_source_orientations_available':True,
                'frame':'parent_relative','tasks':{'source_limb':{'robot_body':'terminal'}}}})
    source=np.tile([1.,0,0,0],(2,1))
    solver.configure_source_orientation_reference(list(mapping),source)
    assert solver.joint_mapping==mapping
    assert solver.body_orientation_tasks['source_limb']['body_name']=='terminal'
    dqa=cp.Variable(len(solver.q_a_indices));dqa.value=np.zeros(len(solver.q_a_indices))
    residual=solver._build_body_orientation_residual(dqa,source);origin=residual.value.copy();columns=[]
    for index in (7,8,9):
        step=np.zeros(len(solver.q_a_indices));step[np.flatnonzero(solver.q_a_indices==index)[0]]=1.;dqa.value=step
        columns.append(residual.value-origin)
    assert np.linalg.matrix_rank(np.stack(columns,axis=1))==3


def test_visible_bone_pose_is_preserved_before_incompatible_full_frame_rotation():
    from omniretargeting.utils import calculate_laplacian_coordinates
    model=mujoco.MjModel.from_xml_string(XML);data=mujoco.MjData(model)
    mapping={'root':'base','end':'tip'}
    solver=GenericInteractionRetargeter(model,data,trimesh.creation.box(),mapping,.5,
        terrain_sample_seed=0,end_effector_config={
            'enabled':True,'hierarchy':True,'effectors':{'marker':{'robot_body':'tip','track_swing':False}},
            '_base_orientation_source_joint':'root',
            'lock_root_orientation_to_source':True,
            'base_position_constraint':{'enabled':True,'track_xy':True,'lock_xy':True,
                                        'track_z':True,'max_z_error':1e-8},
            'body_orientation':{'_source_orientations_available':True},
            'pose_direction':{'enabled':True,'edges':[['root','end']]},
            'pose_priority_absolute_tolerance':1e-5,
        })
    source=np.tile([1.,0,0,0],(2,1));solver.configure_source_orientation_reference(list(mapping),source)
    mujoco.mj_forward(model,data)
    positions=np.array([data.xpos[model.body(n).id] for n in mapping.values()])
    source[1]=Rotation.from_rotvec([0,.4,0]).as_quat(scalar_first=True)
    adjacency=[[1],[0]]
    target=calculate_laplacian_coordinates(positions,adjacency,uniform_weight=True)
    q,cost=solver._single_optimization_step(model.qpos0.copy(),target,adjacency,np.empty((0,3)),
        target_human_joints=positions,source_joint_orientations=source,
        target_base_orientation=np.array([1.,0,0,0]),target_base_position=np.zeros(3))
    assert np.isfinite(cost)
    assert abs(q[7])<1e-3  # The frame target cannot rotate the correctly posed bone.


def test_balanced_free_motion_does_not_trade_away_a_trusted_contact():
    from omniretargeting.utils import calculate_laplacian_coordinates
    model=mujoco.MjModel.from_xml_string(XML);data=mujoco.MjData(model)
    solver=GenericInteractionRetargeter(model,data,trimesh.creation.box(),{'root':'base','end':'tip'},.5,
        terrain_sample_seed=0,end_effector_config={
            'enabled':True,'hierarchy':True,'free_motion_priority':'balanced',
            'priority_relative_tolerance':0.,'priority_absolute_tolerance':1e-7,
            'lock_root_orientation_to_source':True,
            'effectors':{
                'contact':{'robot_body':'tip','contact_sigma':.01},
                'free':{'robot_body':'tip','swing_sigma':.001,'preserve_world_target':True},
            }})
    mujoco.mj_forward(model,data)
    positions=np.array([data.xpos[model.body(n).id] for n in ['base','tip']]);adjacency=[[1],[0]]
    lap=calculate_laplacian_coordinates(positions,adjacency,uniform_weight=True)
    targets={'contact':{'contact':True,'contact_mode':'sticking','confidence':1.,
                        'contact_position':np.array([.2,0,0]),'surface_normal':np.array([0.,0,1.])},
             'free':{'contact':False,'tracking_position':np.array([.4,0,0])}}
    q,cost=solver._single_optimization_step(model.qpos0.copy(),lap,adjacency,np.empty((0,3)),
        target_human_joints=positions,end_effector_targets=targets)
    assert np.isfinite(cost)
    data.qpos[:]=q;mujoco.mj_forward(model,data)
    np.testing.assert_allclose(data.xpos[model.body('tip').id],targets['contact']['contact_position'],atol=1e-5)
    np.testing.assert_array_equal(targets['free']['tracking_position'],[.4,0,0])


def test_balanced_motion_without_contact_preserves_weighted_solver_behavior():
    from omniretargeting.utils import calculate_laplacian_coordinates
    outputs=[]
    for hierarchy in (False,True):
        model=mujoco.MjModel.from_xml_string(XML);data=mujoco.MjData(model)
        solver=GenericInteractionRetargeter(model,data,trimesh.creation.box(),{'root':'base','end':'tip'},.5,
            terrain_sample_seed=0,end_effector_config={
                'enabled':True,'hierarchy':hierarchy,'free_motion_priority':'balanced',
                'lock_root_orientation_to_source':True,
                'effectors':{'free':{'robot_body':'tip','swing_sigma':.02,'preserve_world_target':True}}})
        mujoco.mj_forward(model,data)
        positions=np.array([data.xpos[model.body(n).id] for n in ['base','tip']]);adjacency=[[1],[0]]
        lap=calculate_laplacian_coordinates(positions,adjacency,uniform_weight=True)
        outputs.append(solver._single_optimization_step(model.qpos0.copy(),lap,adjacency,np.empty((0,3)),
            target_human_joints=positions,end_effector_targets={'free':{'contact':False,'tracking_position':np.array([.25,0,.02])}}))
    np.testing.assert_allclose(outputs[0][0],outputs[1][0],atol=1e-10)
    np.testing.assert_allclose(outputs[0][1],outputs[1][1],atol=1e-10)
