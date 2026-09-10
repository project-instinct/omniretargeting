import copy

import mujoco
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from omniretargeting.body_relations import BodyRelationBinding, BodyRelationTasks, build_body_relation_tasks
from omniretargeting.retargeting import GenericInteractionRetargeter
from omniretargeting.local_collision_repair import select_safe_postprocessing
from omniretargeting.trajectory_refinement import refine_joint_trajectory
from omniretargeting.trajectory_validation import TrajectoryValidator, local_repair_config
from omniretargeting.scene_geometry import SceneQuery
from tests.test_contact_priority import _task
from tests.test_local_collision_repair import setup_motion


def case(ball=False):
    xml = f'''<mujoco><compiler angle="radian"/><worldbody>
    <body name="base" pos="0 0 1"><freejoint/><geom size="0.02"/>
      <body name="chest" pos="0 0 .2"><joint name="waist" axis="0 0 1"/>
        <geom size="0.02"/>
        <body name="shoulder" pos="0 .2 0">
          <joint name="shoulder_joint" type="{'ball' if ball else 'hinge'}" axis="1 0 0"/>
          <geom type="capsule" fromto="0 0 0 0 .3 0" size=".02"/>
          <body name="elbow" pos="0 .3 0"><joint name="elbow_joint" axis="0 0 1"/>
            <geom type="capsule" fromto="0 0 0 0 .25 0" size=".02"/>
            <body name="hand" pos="0 .25 0"><geom size=".02"/></body>
          </body>
        </body>
      </body>
    </body></worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    mapping = dict(core="chest", upper="shoulder", middle="elbow", end="hand")
    names = list(mapping)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    positions = np.array([data.xpos[model.body(mapping[n]).id] for n in names])
    # Deliberately unrelated source link axes, as with SMPL-X vs a robot.
    gauge = Rotation.from_euler("xyz", [.4, -.7, .3])
    orientations = np.tile(gauge.as_quat(scalar_first=True), (len(names), 1))
    priority = {"effectors": {"arm": {"arm_posture": dict(
        source_shoulder="upper", source_elbow="middle", source_wrist="end")}}}
    config = dict(enabled=True, reference_source="core", weight=1., sigma=.12)
    args = [model, config, mapping, priority, names, np.tile(positions, (4, 1, 1)),
            np.tile(orientations, (4, 1, 1)), positions.copy(), orientations.copy(), model.qpos0.copy()]
    return args


def build(args):
    tasks, report = build_body_relation_tasks(*args)
    assert tasks is not None, report
    return tasks


@pytest.mark.parametrize("ball", [False, True])
def test_relation_jacobian_all_nv_and_moving_reference_finite_difference(ball):
    args = case(ball)
    model = args[0]
    tasks = build(args)
    q = model.qpos0.copy()
    velocity = np.linspace(-.4, .7, model.nv)
    mujoco.mj_integratePos(model, q, velocity, .8)
    _, jac = tasks.at(q, 0)
    assert model.nq != model.nv
    for dof in range(model.nv):
        v = np.eye(model.nv)[dof] * 1e-6
        qm, qp = q.copy(), q.copy()
        mujoco.mj_integratePos(model, qm, v, -1)
        mujoco.mj_integratePos(model, qp, v, 1)
        numeric = (tasks.at(qp, 0)[0] - tasks.at(qm, 0)[0]) / 2e-6
        np.testing.assert_allclose(jac[:, dof], numeric, atol=2e-9)
    # Common floating base and chest rotations cancel, not just translation.
    np.testing.assert_allclose(jac[:, :7], 0, atol=1e-14)
    subset = np.arange(7, model.nv)
    row = list(tasks.rows(q, 0, subset))[0]
    np.testing.assert_allclose(row[2], jac[:, subset] / tasks.sigma, atol=1e-14)


def test_qpos_conversion_matches_actual_main_solver_active_dofs():
    args = case()
    model = args[0]
    tasks = build(args)
    solver = GenericInteractionRetargeter(model, mujoco.MjData(model), trimesh.creation.box(),
                                         args[2], 1.5, terrain_sample_points=5, hard_penetration_constraint=False)
    q = model.qpos0.copy()
    mujoco.mj_integratePos(model, q, np.linspace(-.3, .5, model.nv), .7)
    solver.robot_data.qpos[:] = q
    mujoco.mj_forward(model, solver.robot_data)
    jac = tasks.at(q, 0)[1] @ solver._build_transform_qdot_to_qvel_fast()
    for qi in solver.q_a_indices:
        qm, qp = q.copy(), q.copy()
        qm[qi] -= 1e-6
        qp[qi] += 1e-6
        mujoco.mj_normalizeQuat(model, qm)
        mujoco.mj_normalizeQuat(model, qp)
        np.testing.assert_allclose(jac[:, qi], (tasks.at(qp, 0)[0] - tasks.at(qm, 0)[0]) / 2e-6, atol=2e-9)


def test_reference_axis_calibration_and_common_world_rigid_transform():
    args = case()
    tasks = build(args)
    q = args[0].qpos0
    np.testing.assert_allclose(tasks.targets[0, 0], tasks.at(q, 0)[0], atol=1e-15)
    transformed = copy.deepcopy(args[1:])
    args2 = [args[0], *transformed]
    rigid = Rotation.from_euler("xyz", [.3, .8, -.5])
    for i in (5, 7):
        args2[i] = args[i] @ rigid.as_matrix().T + [3, 2, -1]
    for i in (6, 8):
        shape = args[i].shape
        args2[i] = (rigid * Rotation.from_quat(args[i].reshape(-1, 4), scalar_first=True)).as_quat(scalar_first=True).reshape(shape)
    q2 = args2[9]
    q2[:3] = rigid.apply(q[:3]) + [3, 2, -1]
    q2[3:7] = (rigid * Rotation.from_quat(q[3:7], scalar_first=True)).as_quat(scalar_first=True)
    other = build(args2)
    np.testing.assert_allclose(tasks.targets, other.targets, atol=2e-15)
    np.testing.assert_allclose(tasks.at(q, 0)[0], other.at(q2, 0)[0], atol=2e-15)
    # Known local arm motion transported through arbitrary source axes.
    local_delta = np.array([.1, -.2, .05])
    args[5][:, -1] += local_delta
    moved = build(args)
    np.testing.assert_allclose(moved.targets[0, 0] - tasks.targets[0, 0], local_delta, atol=1e-15)


def test_morphology_uses_arm_length_and_robot_shoulder_width_without_editing_source():
    args = case()
    original = build(args)
    # Human twice the robot's shoulder width and arm length.
    for i in (5, 7):
        chest = args[i][..., :1, :].copy()
        args[i] = chest + 2 * (args[i] - chest)
    source = args[5].copy()
    scaled = build(args)
    np.testing.assert_allclose(scaled.targets, original.targets, atol=1e-15)
    assert scaled.diagnostics["bindings"][0]["arm_length_ratio"] == pytest.approx(.5)
    np.testing.assert_array_equal(source, args[5])


@pytest.mark.parametrize("missing", [6, 7, 8, 9])
def test_missing_data_disables_with_reason(missing):
    args = case()
    args[missing] = None
    tasks, report = build_body_relation_tasks(*args)
    assert tasks is None
    assert "missing_explicit" in report["reason"]


def test_degenerate_motion_reference_and_weight_do_not_create_nan_or_flip():
    args = case()
    args[5][1, -1] = np.nan
    args[5][2, -1] = args[5][2, 0]
    args[6][3, 0] = 0
    tasks = build(args)
    assert np.isfinite(tasks.targets).all()
    assert tasks.diagnostics["bindings"][0]["invalid_frames"] == [1, 2, 3]
    for frame in (1, 2, 3):
        assert not list(tasks.rows(args[9], frame, np.arange(args[0].nv)))
    args[7][:] = 0
    disabled, report = build_body_relation_tasks(*args)
    assert disabled is None and report["disabled"][0]["reason"] == "degenerate_reference_chain"


def translation_relation(model, q, weight):
    target = np.tile([.3, .1, 0], (len(q), 1, 1))
    return BodyRelationTasks(model, [BodyRelationBinding("arm", "foot", "root", "end", "core")],
                             target, np.ones((len(q), 1)), weight=weight, sigma=.02, diagnostics={})


def test_zero_weight_is_exact_main_solver_and_refinement_noop():
    model, q, _ = setup_motion(4)
    mapping = {"core": "root", "end": "foot", "other": "other"}
    scene = trimesh.creation.box(); scene.apply_translation([0, 0, -5])
    solver = GenericInteractionRetargeter(model, mujoco.MjData(model), scene, mapping, 1.,
                                         terrain_sample_points=5, terrain_sample_seed=0,
                                         hard_penetration_constraint=False)
    data = mujoco.MjData(model); data.qpos[:] = q[0]; mujoco.mj_forward(model, data)
    points = np.array([data.xpos[model.body(b).id] for b in mapping.values()])
    a = solver.retarget_frame(points, q[0], max_iter=3)
    solver.body_relation_tasks = translation_relation(model, q, 0.)
    b = solver.retarget_frame(points, q[0], max_iter=3, relation_frame=0)
    np.testing.assert_array_equal(a, b)
    contact = _task(np.c_[q[:, 7:9], np.zeros(len(q))], np.zeros(len(q), bool))
    priority = {"effectors": {"arm": {"robot_body": "foot"}}}
    plain = refine_joint_trajectory(model, q, {"arm": contact}, priority, {})
    zero = refine_joint_trajectory(model, q, {"arm": contact}, priority, {}, body_relation_tasks=solver.body_relation_tasks)
    np.testing.assert_array_equal(plain, zero)


def test_contact_protection_during_main_solve_refinement_acceptance_and_saved_output(tmp_path):
    model, q, scene = setup_motion(4)
    q[:, 7:9] = [.1, .1]
    scene.apply_translation([0, 0, -5])
    relation = translation_relation(model, q, 1e5)
    targets = np.c_[q[:, 7:9], np.zeros(len(q))]
    task = _task(targets, np.ones(len(q), bool), ["sticking"] * len(q))
    priority = {"enabled": True, "hierarchy": True, "free_motion_priority": "balanced",
                "pose_direction": {"enabled": False}, "priority_absolute_tolerance": .001,
                "effectors": {"arm": {"robot_body": "foot", "track_orientation": False}}}
    solver = GenericInteractionRetargeter(model, mujoco.MjData(model), scene,
        {"core": "root", "end": "foot", "other": "other"}, 1.,
        terrain_sample_points=5, terrain_sample_seed=0, hard_penetration_constraint=False,
        end_effector_config=priority)
    solver.body_relation_tasks = relation
    data = mujoco.MjData(model); data.qpos[:] = q[0]; mujoco.mj_forward(model, data)
    points = np.array([data.xpos[model.body(b).id] for b in solver.joint_mapping.values()])
    solved = solver.retarget_frame(points, q[0], max_iter=10, relation_frame=0,
                                   end_effector_targets={"arm": task.frame(0)})
    data.qpos[:] = solved; mujoco.mj_forward(model, data)
    assert np.linalg.norm(data.xpos[model.body("foot").id] - targets[0]) < .001
    candidate = refine_joint_trajectory(model, q, {"arm": task}, priority, {"iterations": 2}, body_relation_tasks=relation)
    cfg = local_repair_config({})
    accepted, _, _ = select_safe_postprocessing(model, q, candidate, scene, {"arm": task}, priority, cfg,
                                               body_relation_tasks=relation)
    v = TrajectoryValidator(model, q, SceneQuery(scene), {"arm": task}, priority, cfg, 30, body_relation_tasks=relation)
    assert v.fidelity(accepted)["ok"]
    assert np.max(np.abs(accepted[:, 7:9] - q[:, 7:9])) <= cfg["contact_regression"] + 1e-8
    # A separate free, small source-driven correction must survive acceptance.
    free = _task(targets, np.zeros(len(q), bool))
    relation = translation_relation(model, q, .0001)
    free_priority = {"effectors": {"arm": {"robot_body": "foot", "preserve_world_target": False}}}
    candidate = refine_joint_trajectory(model, q, {"arm": free}, free_priority,
                                        {"iterations": 2, "reference_sigma": .8}, body_relation_tasks=relation)
    accepted, _, _ = select_safe_postprocessing(model, q, candidate, scene, {"arm": free}, free_priority, cfg,
                                               body_relation_tasks=relation)
    assert np.max(np.abs(accepted - q)) > 1e-6
    path = tmp_path / "accepted.npz"
    np.savez(path, qpos=accepted)
    np.testing.assert_array_equal(np.load(path)["qpos"], accepted)


def test_segment_rows_keep_original_frame_indices_and_source_weights():
    model, q, scene = setup_motion(5)
    task = translation_relation(model, q, 1.)
    v = TrajectoryValidator(model, q, SceneQuery(scene), {}, {}, local_repair_config({}), 30,
                            body_relation_tasks=task)
    with v.segment(1, 5) as part:
        with part.segment(1, 3) as nested:
            np.testing.assert_array_equal(list(nested.relation_rows(q[2], 0, v.di))[0][1],
                                          list(task.rows(q[2], 2, v.di))[0][1])
    original_weight = list(task.rows(q[0], 0, v.di))[0][0]
    worse = q[0].copy(); worse[7] += .5
    assert list(task.rows(worse, 0, v.di))[0][0] == original_weight
