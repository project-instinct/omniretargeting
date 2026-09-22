"""Physical increments must agree with MuJoCo, including rotated roots."""
import cvxpy as cp
import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting._solver.kinematics import RobotKinematicsMixin
from omniretargeting._solver import tangent


@pytest.fixture
def context():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><freejoint/><geom type="sphere" size=".05"/>
        <body name="tip" pos=".1 .2 .3"><joint range="-90 90"/>
          <geom type="sphere" pos=".2 0 0" size=".02"/>
        </body></body></worldbody></mujoco>''')
    result = RobotKinematicsMixin()
    result.robot_model = model
    result.robot_data = mujoco.MjData(model)
    result.q_a_indices = np.arange(model.nq)
    result.step_size = .2
    result.end_effector_config = {'step_coordinates': 'tangent'}
    return result


@pytest.mark.parametrize('sign', [-1, 1])
@pytest.mark.parametrize('fixed_xy', [False, True])
def test_embedding_and_integration_match_native_mujoco(context, sign, fixed_xy):
    model = context.robot_model
    q = model.qpos0.copy()
    q[3:7] = sign * Rotation.from_rotvec([.8, -.3, .5]).as_quat(scalar_first=True)
    if fixed_xy:
        context.q_a_indices = context.q_a_indices[2:]
    embedding, dofs, limits = tangent.tangent_basis(context, q)
    matrix = context._build_transform_qdot_to_qvel_fast()[:, context.q_a_indices]
    np.testing.assert_allclose(matrix @ embedding, np.eye(model.nv)[:, dofs], atol=1e-14)
    np.testing.assert_allclose(q[3:7] @ embedding[np.isin(context.q_a_indices, range(3, 7))], 0, atol=1e-14)
    velocity = np.linspace(-.12, .14, len(dofs))
    velocity = np.clip(velocity, -limits, limits)
    actual = tangent.integrate_step(context, q, embedding @ velocity)
    full = np.zeros(model.nv)
    full[dofs] = velocity
    expected = q.copy()
    mujoco.mj_integratePos(model, expected, full, 1.)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert np.linalg.norm(actual[3:7]) == pytest.approx(1., abs=1e-14)
    if fixed_xy:
        np.testing.assert_array_equal(actual[:2], q[:2])


def test_world_point_jacobian_matches_tangent_finite_difference(context):
    model = context.robot_model
    data = context.robot_data
    q = model.qpos0.copy()
    q[3:7] = Rotation.from_rotvec([.7, -.2, 1.2]).as_quat(scalar_first=True)
    basis, dofs, _ = tangent.tangent_basis(context, q)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    body = model.body('tip').id
    local = np.array([.25, .01, -.02])
    jac = context._calc_contact_jacobian_from_point(body, local) @ basis
    epsilon = 1e-6
    for column in range(len(dofs)):
        points = []
        for sign in [-1, 1]:
            data.qpos[:] = tangent.integrate_step(context, q, sign * epsilon * basis[:, column])
            mujoco.mj_forward(model, data)
            points.append(data.xpos[body] + data.xmat[body].reshape(3, 3) @ local)
        np.testing.assert_allclose((points[1]-points[0])/(2*epsilon), jac[:, column], atol=1e-8)


def test_feasible_status_does_not_hide_violated_constraints(context):
    value = cp.Variable()
    problem = cp.Problem(cp.Minimize(cp.square(value - .001)), [value <= .002])
    problem.solve(solver=cp.CLARABEL)
    assert tangent.usable(problem, context)
    value.value = .05
    assert problem.status == cp.OPTIMAL
    assert tangent.constraint_violation(problem) == pytest.approx(.048)
    assert not tangent.usable(problem, context)


def test_tangent_contact_remains_primary_over_conflicting_surface():
    from tests.test_surface_contact import contact_case
    from tests.test_surface_endpoint_priority import (
        _make_solver, _foot_position, _mapped_points, _shifted_surface_tasks,
    )
    from tests._contact_fixtures import _task

    model, poses, source_tasks, _ = contact_case(frames=2)
    solver = _make_solver(model)
    solver.end_effector_config['step_coordinates'] = 'tangent'
    solver.surface_contact_tasks = _shifted_surface_tasks(model, source_tasks, [.12, 0., 0.])
    target = _foot_position(model, poses[0])
    result = solver.retarget_frame(
        _mapped_points(model, poses[0]), poses[0], max_iter=4, relation_frame=0,
        end_effector_targets={'left_foot': _task(np.tile(target, (2, 1)), np.ones(2, dtype=bool)).frame(0)},
    )
    assert np.isfinite(result).all()
    assert np.linalg.norm(_foot_position(model, result) - target) < .002
    assert np.linalg.norm(result[3:7]) == pytest.approx(1., abs=1e-10)


def test_only_batch_tangent_proposals_defer_interval_collisions():
    from omniretargeting.retargeting import GenericInteractionRetargeter
    from omniretargeting.self_collision import SelfCollisionQuery
    from tests.test_self_collision import XML

    model = mujoco.MjModel.from_xml_string(XML)
    query = SelfCollisionQuery(model)
    before = model.qpos0.copy()
    before[7] = -.2
    candidate = before.copy()
    candidate[7] = .38
    assert max(query.depths(before).values(), default=0.) == 0.
    assert max(query.depths(candidate).values(), default=0.) == 0.
    assert max(query.segment(before, candidate)['depths'].values()) > .1
    solver = GenericInteractionRetargeter.__new__(GenericInteractionRetargeter)
    solver.robot_model = model
    solver.self_collision_enabled = True
    solver._self_collision_query = query
    solver.end_effector_config = {'step_coordinates': 'tangent'}
    solver._defer_self_collision_path_repair = False
    standalone = solver._guard_self_collision_step(before, candidate, before)
    assert not np.array_equal(standalone, candidate)
    solver._defer_self_collision_path_repair = True
    batch_proposal = solver._guard_self_collision_step(before, candidate, before)
    np.testing.assert_allclose(batch_proposal, candidate, atol=1e-14)
    # Current-pose collision checking is still mandatory for batch proposals.
    penetrating = before.copy()
    penetrating[7] = .08
    accepted = solver._guard_self_collision_step(before, penetrating, before)
    assert max(query.depths(accepted).values(), default=0.) <= query.cfg['penetration_tolerance']
