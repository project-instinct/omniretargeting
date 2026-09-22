"""Independent seeds and a source-fixed operator; real MuJoCo Jacobians."""

from copy import deepcopy

import mujoco
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from omniretargeting.retargeting import GenericInteractionRetargeter
from omniretargeting.utils import (
    calculate_exponential_edge_weights,
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
    calculate_source_laplacian,
)

XML = """<mujoco><compiler angle="radian"/><worldbody>
<body name="root" pos="0 0 1"><freejoint/><geom size=".03"/>
 <body name="trunk"><joint name="trunk_twist" axis="0 0 1" range="-1 1"/><geom size=".03"/>
  <body name="upper" pos="0 .15 .15">
   <joint name="swing" axis="0 1 0" range="-2 2"/>
   <joint name="plane" axis="1 0 0" range="-1.5 1.5"/><geom size=".03"/>
   <body name="lower" pos=".3 0 0"><joint name="bend" axis="0 0 1" range="-1.8 1.8"/>
    <geom size=".03"/><body name="tip" pos=".25 0 0"><geom size=".03"/></body>
   </body>
  </body>
 </body>
</body></worldbody></mujoco>"""


def make_solver(*, q_a_init_idx=-7, **overrides):
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    names = ["center", "chest", "proximal", "middle", "distal"]
    mapping = dict(zip(names, ["root", "trunk", "upper", "lower", "tip"]))
    config = {
        "pose_initialization": {"enabled": True},
        "pose_direction": {
            "enabled": True,
            "edges": [["proximal", "middle"], ["middle", "distal"]],
        },
        "body_orientation": {
            "_source_orientations_available": True,
            "frame": "parent_relative",
        },
        **overrides,
    }
    floor = trimesh.creation.box(extents=[4, 4, 0.1])
    floor.apply_translation([0, 0, -0.05])
    solver = GenericInteractionRetargeter(
        model,
        data,
        floor,
        mapping,
        1.5,
        valid_joint_names=names,
        end_effector_config=config,
        terrain_sample_seed=0,
        terrain_sample_points=8,
        hard_penetration_constraint=True,
        q_a_init_idx=q_a_init_idx,
    )
    _, ori = source_at(solver, model.qpos0)
    solver.configure_source_orientation_reference(names, ori)
    return solver


def source_at(solver, q):
    data = mujoco.MjData(solver.robot_model)
    data.qpos[:] = q
    mujoco.mj_forward(solver.robot_model, data)
    ids = [
        solver.robot_model.body(solver.joint_mapping[n]).id
        for n in solver.valid_joint_names
    ]
    return data.xpos[ids].copy(), Rotation.from_matrix(
        data.xmat[ids].reshape(-1, 3, 3)
    ).as_quat(scalar_first=True)


def direction_error(solver, q, target):
    positions, _ = source_at(solver, q)
    vectors = np.diff(positions[2:], axis=0)
    desired = np.diff(target[2:], axis=0)
    return np.linalg.norm(
        vectors / np.linalg.norm(vectors, axis=1)[:, None]
        - desired / np.linalg.norm(desired, axis=1)[:, None]
    )


def test_seed_preserves_already_satisfied_pose_and_all_fixed_state():
    solver = make_solver()
    q = solver.robot_model.qpos0.copy()
    q[:3] += [2, 3, 0.4]
    q[3:7] = Rotation.from_rotvec([0.2, -0.1, 0.7]).as_quat(scalar_first=True)
    q[7:] = [0.3, -0.6, 0.5, 0.8]
    positions, orientations = source_at(solver, q)
    saved = deepcopy(
        (
            solver._source_orientation_references,
            solver._robot_orientation_references,
            solver.robot_reference_qpos,
            solver.robot_data.qpos.copy(),
        )
    )
    result = solver._initialize_pose(q, positions, orientations)
    np.testing.assert_array_equal(q, result)
    assert solver.last_pose_initialization["status"] == "unchanged"
    for old, new in zip(
        saved[:2],
        (solver._source_orientation_references, solver._robot_orientation_references),
    ):
        for key in old:
            np.testing.assert_array_equal(old[key], new[key])
    np.testing.assert_array_equal(saved[2], solver.robot_reference_qpos)
    np.testing.assert_array_equal(saved[3], solver.robot_data.qpos)


@pytest.mark.parametrize("enabled", [False, True])
def test_disabled_or_failed_seed_preserves_input_dtype(enabled):
    solver = make_solver(pose_initialization={"enabled": enabled})
    q = solver.robot_model.qpos0.astype(np.float32)
    source, _ = source_at(solver, q)
    result = solver._initialize_pose(q, source, None)
    assert result.dtype == q.dtype
    np.testing.assert_array_equal(result, q)


def test_two_segment_seed_reduces_error_without_moving_root_or_trunk():
    solver = make_solver()
    q = solver.robot_model.qpos0.copy()
    q[7:] = [0.4, 0, 0, 0.2]
    target_q = q.copy()
    target_q[8:] = [-0.8, 0.7, 1.1]
    positions, orientations = source_at(solver, target_q)
    before = positions.copy()
    result = solver._initialize_pose(q, positions, orientations)
    assert solver.last_pose_initialization["status"] == "accepted"
    assert direction_error(solver, result, positions) < 1e-4
    assert direction_error(solver, q, positions) > 0.5
    np.testing.assert_array_equal(result[:8], q[:8])
    np.testing.assert_array_equal(positions, before)
    assert np.all(result[solver.q_a_indices] >= solver.q_a_lb)
    assert np.all(result[solver.q_a_indices] <= solver.q_a_ub)


def test_seed_keeps_bending_plane_not_just_bone_inner_product():
    solver = make_solver()
    q = solver.robot_model.qpos0.copy()
    q[10] = 0.8
    target = q.copy()
    target[9] = 1.1  # Rotate bending plane around the upper bone's own axis.
    pos, ori = source_at(solver, target)
    result = solver._initialize_pose(q, pos, ori)
    assert direction_error(solver, result, pos) < 1e-4
    assert result[9] > 1.0


def test_fast_source_step_has_no_new_velocity_bound():
    solver = make_solver()
    q = solver.robot_model.qpos0.copy()
    q[8:] = [-1.0, 0.3, 0.7]
    target = q.copy()
    target[8] = 1.0
    pos, ori = source_at(solver, target)
    result = solver._initialize_pose(q, pos, ori)
    assert abs(result[8] - q[8]) > 1.9
    assert direction_error(solver, result, pos) < 1e-4


@pytest.mark.parametrize(
    "failure",
    [
        "nan",
        "collapsed",
        "missing_orientation",
        "missing_mapping",
        "solver",
        "nonfinite_result",
    ],
)
def test_seed_fallback_is_exact_and_reason_is_recorded(failure, monkeypatch):
    solver = make_solver()
    q = solver.robot_model.qpos0.copy()
    goal = q.copy()
    goal[8:] = [0.4, 0.5, 0.8]
    pos, ori = source_at(solver, goal)
    if failure == "nan":
        pos[2, 0] = np.nan
    elif failure == "collapsed":
        pos[3] = pos[2]
    elif failure == "missing_orientation":
        ori = None
    elif failure == "missing_mapping":
        solver.pose_bones = []
    elif failure == "solver":

        def fail(*args, **kwargs):
            raise RuntimeError("injected failure")

        monkeypatch.setattr(
            "omniretargeting._solver.initialization.least_squares", fail
        )
    else:
        from types import SimpleNamespace

        monkeypatch.setattr(
            "omniretargeting._solver.initialization.least_squares",
            lambda *a, **k: SimpleNamespace(
                success=True, x=np.full(3, np.nan), fun=np.zeros(6)
            ),
        )
    np.testing.assert_array_equal(solver._initialize_pose(q, pos, ori), q)
    assert solver.last_pose_initialization["status"] == "fallback"
    assert solver.last_pose_initialization["reason"]


def test_unreachable_seed_stays_inside_existing_joint_limits():
    solver = make_solver()
    q = solver.robot_model.qpos0.copy()
    goal = q.copy()
    goal[8:] = [2.3, 0.5, 0.8]
    pos, ori = source_at(solver, goal)
    result = solver._initialize_pose(q, pos, ori)
    assert np.all(result[solver.q_a_indices] <= solver.q_a_ub)
    assert np.all(result[solver.q_a_indices] >= solver.q_a_lb)


def test_exponential_rows_and_coordinates_match_definition():
    vertices = np.array([[0.0, 0, 0], [1, 0, 0], [3, 0, 0], [7, 0, 0]])
    adjacency = [[2, 1], [0, 2], [1, 0], []]
    rows = calculate_exponential_edge_weights(vertices, adjacency, kappa=2)
    for row in rows[:3]:
        assert row.sum() == pytest.approx(1)
    assert rows[0][1] > rows[0][0]
    assert np.exp(-2 * 1) > np.exp(-2 * 3)
    expected = np.zeros_like(vertices)
    for i, neighbors in enumerate(adjacency):
        for j, weight in zip(neighbors, rows[i]):
            expected[i] += weight * (vertices[i] - vertices[j])
    np.testing.assert_allclose(
        calculate_source_laplacian(vertices, adjacency, kappa=2) @ vertices, expected
    )
    np.testing.assert_allclose(
        calculate_source_laplacian(vertices, adjacency, kappa=0),
        calculate_laplacian_matrix(vertices, adjacency, uniform_weight=True),
    )


@pytest.mark.parametrize("scale", [0.0, 1e-300, 1e4, 1e308])
def test_extreme_distances_and_duplicates_are_finite(scale):
    vertices = np.array([[0.0, 0, 0], [scale, scale, 0], [-scale, 0, 0]])
    rows = calculate_exponential_edge_weights(
        vertices, [[1, 2], [0, 2], [0, 1]], kappa=1e308
    )
    for row in rows:
        assert np.isfinite(row).all()
        assert row.sum() == pytest.approx(1)


@pytest.mark.parametrize(
    "vertices,adjacency,kappa",
    [
        ([[np.nan, 0, 0]], [[]], 1),
        ([[0, 0, 0]], [], 1),
        ([[0, 0, 0]], [[1]], 1),
        ([[0, 0, 0]], [[0]], 1),
        ([[0, 0, 0]], [[]], -1),
        ([[0, 0, 0]], [[]], np.inf),
        ([[0, 0, 0], [1, 0, 0]], [[1, 1], [0]], 1),
    ],
)
def test_invalid_graph_inputs_are_rejected(vertices, adjacency, kappa):
    with pytest.raises(ValueError):
        calculate_source_laplacian(vertices, adjacency, kappa=kappa)


def test_source_operator_is_frozen_in_production_frame_iterations(monkeypatch):
    solver = make_solver(interaction_mesh={"weighting": "exponential", "kappa": 3.0})
    q = solver.robot_model.qpos0.copy()
    pos, ori = source_at(solver, q)
    seen = []

    def step(
        current, target, adjacency, terrain, *args, source_laplacian=None, **kwargs
    ):
        seen.append(source_laplacian)
        np.testing.assert_allclose(target, source_laplacian @ np.vstack([pos, terrain]))
        updated = current.copy()
        updated[8] += 0.03
        return updated, 1.0 / len(seen)

    monkeypatch.setattr(solver, "_single_optimization_step", step)
    solver.retarget_frame(pos, q, max_iter=3, source_joint_orientations=ori)
    assert len(seen) == 3
    assert all(operator is seen[0] for operator in seen)
    assert not seen[0].flags.writeable


def test_uniform_retains_legacy_target_and_no_operator_argument(monkeypatch):
    solver = make_solver(pose_initialization={"enabled": False})
    q = solver.robot_model.qpos0.copy()
    pos, _ = source_at(solver, q)

    def step(current, target, adjacency, terrain, *args, **kwargs):
        assert "source_laplacian" not in kwargs
        np.testing.assert_array_equal(
            target,
            calculate_laplacian_coordinates(np.vstack([pos, terrain]), adjacency),
        )
        np.testing.assert_array_equal(current, q)
        return current, 0.0

    monkeypatch.setattr(solver, "_single_optimization_step", step)
    solver.retarget_frame(pos, q, max_iter=1)


def test_laplacian_residual_jacobian_uses_mujoco_tangent_perturbation():
    solver = make_solver()
    q = solver.robot_model.qpos0.copy()
    q[3:7] = Rotation.from_rotvec([0.3, -0.4, 0.8]).as_quat(scalar_first=True)
    q[7:] = [0.1, 0.3, 0.2, 0.6]
    vertices, _ = source_at(solver, q)
    terrain = np.array([[1.0, 2.0, 0.0]])
    source = np.vstack([vertices + [0.02, 0, 0.1], terrain])
    adjacency = [[j for j in range(len(source)) if j != i] for i in range(len(source))]
    operator = calculate_source_laplacian(source, adjacency, kappa=3)
    solver.robot_data.qpos[:] = q
    mujoco.mj_forward(solver.robot_model, solver.robot_data)
    jac, _, _ = solver._compute_robot_jacobians(q)
    transform = solver._build_transform_qdot_to_qvel_fast()
    jac_tangent = jac @ np.linalg.pinv(transform)
    linear = np.kron(operator, np.eye(3)) @ np.vstack(
        [jac_tangent, np.zeros((3, solver.robot_model.nv))]
    )
    columns = []
    eps = 1e-6
    for i in range(solver.robot_model.nv):
        velocity = np.eye(solver.robot_model.nv)[i]
        values = []
        for sign in (1, -1):
            perturbed = q.copy()
            mujoco.mj_integratePos(solver.robot_model, perturbed, velocity, sign * eps)
            points, _ = source_at(solver, perturbed)
            values.append((operator @ (np.vstack([points, terrain]) - source)).ravel())
        columns.append((values[0] - values[1]) / (2 * eps))
    np.testing.assert_allclose(linear, np.array(columns).T, atol=5e-9)


def test_seed_direction_jacobian_matches_scalar_tangent_differences():
    solver = make_solver()
    q = solver.robot_model.qpos0.copy()
    q[7:] = [0.2, -0.5, 0.4, 0.7]
    bindings, indices = solver._pose_initialization_bindings()
    targets = {("proximal", "middle"): np.zeros(3), ("middle", "distal"): np.zeros(3)}
    _, jac = solver._pose_seed_rows(q, targets, bindings, indices)
    columns = []
    for qa in solver.q_a_indices[indices]:
        joint = np.flatnonzero(solver.robot_model.jnt_qposadr == qa)[0]
        velocity = np.eye(solver.robot_model.nv)[solver.robot_model.jnt_dofadr[joint]]
        values = []
        for sign in (1, -1):
            perturbed = q.copy()
            mujoco.mj_integratePos(solver.robot_model, perturbed, velocity, sign * 1e-6)
            values.append(
                solver._pose_seed_rows(perturbed, targets, bindings, indices)[0]
            )
        columns.append((values[0] - values[1]) / 2e-6)
    np.testing.assert_allclose(jac, np.array(columns).T, atol=5e-9)


def test_production_step_uses_source_operator_for_both_value_and_jacobian(monkeypatch):
    solver = make_solver()
    solver.self_collision_enabled = False
    solver.Q_diag[:] = 0.
    q = solver.robot_model.qpos0.copy()
    q[7:] = [0.2, -0.5, 0.4, 0.7]
    source, _ = source_at(solver, solver.robot_model.qpos0)
    terrain = solver.terrain_points
    vertices = np.vstack([source, terrain])
    adjacency = [
        [j for j in range(len(vertices)) if j != i] for i in range(len(vertices))
    ]
    operator = calculate_source_laplacian(vertices, adjacency, kappa=4)
    target = operator @ vertices
    robot, _ = source_at(solver, q)
    solver.robot_data.qpos[:] = q
    mujoco.mj_forward(solver.robot_model, solver.robot_data)
    jac, _, _ = solver._compute_robot_jacobians(q)
    expected_jac = np.kron(operator, np.eye(3)) @ np.vstack(
        [jac, np.zeros((3 * len(terrain), solver.nq_a))]
    )
    expected_value = (operator @ np.vstack([robot, terrain])).ravel()

    def inspect(**kwargs):
        dq = kwargs["dqa"]
        for direction in [np.zeros(solver.nq_a), *np.eye(solver.nq_a), *-np.eye(solver.nq_a)]:
            dq.value = .01 * direction
            residual = expected_value - target.ravel() + expected_jac @ dq.value
            assert kwargs["body_obj"].value == pytest.approx(10. * (residual @ residual), abs=1e-12)
        return q.copy(), 0.0

    def forbidden(*args, **kwargs):
        raise AssertionError("Robot iteration must not recompute source weights")

    monkeypatch.setattr("omniretargeting._solver.step.solve_hierarchy", inspect)
    monkeypatch.setattr(
        "omniretargeting._solver.step.calculate_laplacian_matrix", forbidden
    )
    solver._single_optimization_step(
        q, target, adjacency, terrain, source_laplacian=operator
    )


def test_seed_is_invariant_to_source_native_axes_and_world_translation():
    solver = make_solver()
    q = solver.robot_model.qpos0.copy()
    goal = q.copy()
    goal[8:] = [-0.7, 0.4, 1.0]
    pos, ori = source_at(solver, goal)
    expected = solver._initialize_pose(q, pos, ori)
    assert solver.last_pose_initialization["status"] == "accepted"
    local_change = Rotation.from_rotvec([0.6, -0.4, 0.2]).as_matrix()
    for name in solver._source_orientation_references:
        solver._source_orientation_references[name] = (
            solver._source_orientation_references[name] @ local_change
        )
    changed = Rotation.from_matrix(
        Rotation.from_quat(ori, scalar_first=True).as_matrix() @ local_change
    ).as_quat(scalar_first=True)
    actual = solver._initialize_pose(q, pos + [30, -20, 10], changed)
    assert solver.last_pose_initialization["status"] == "accepted"
    np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_graph_vertex_permutation_and_empty_graph():
    vertices = np.array([[0.0, 0, 0], [0.2, 0, 0], [1.0, 0.3, 0]])
    adjacency = [[2, 1], [0], [0]]
    matrix = calculate_source_laplacian(vertices, adjacency, kappa=3.0)
    permutation = np.array([2, 0, 1])
    inverse = np.argsort(permutation)
    changed = [[int(inverse[j]) for j in adjacency[i]] for i in permutation]
    reordered = calculate_source_laplacian(vertices[permutation], changed, kappa=3.0)
    np.testing.assert_array_equal(reordered, matrix[np.ix_(permutation, permutation)])
    assert calculate_source_laplacian(np.empty((0, 3)), [], kappa=3.0).shape == (0, 0)


def test_quality_metrics_separate_vertical_chatter_from_support_slip():
    from omniretargeting.motion_quality import trajectory_metrics
    from tests._contact_fixtures import _task

    solver = make_solver()
    poses = np.tile(solver.robot_model.qpos0, (4, 1))
    poses[:, 2] += np.arange(4) * 0.01
    source = np.array([source_at(solver, q)[0] for q in poses])
    fixed_targets = np.tile(source[0, -1], (4, 1))
    priority = {
        "effectors": {"left_foot": {"robot_body": "tip"}},
        "_position_anchor_source_joint": "center",
    }

    def measure(mask):
        return trajectory_metrics(
            solver.robot_model,
            poses,
            source,
            solver.valid_joint_names,
            solver.joint_mapping,
            priority,
            {"left_foot": _task(fixed_targets, contact_mask=mask)},
            30.0,
        )

    active = measure(np.ones(4, dtype=bool))
    foot = active["end_effectors"]["left_foot"]
    assert foot["support_slip_m_per_s"]["max"] == 0.0
    assert foot["support_vertical_speed_m_per_s"]["mean"] == pytest.approx(0.3)
    assert foot["contact_position_error_m"]["max"] == pytest.approx(0.03)
    assert active["root_anchor_world_error_m"]["max"] == 0.0
    inactive = measure(np.zeros(4, dtype=bool))["end_effectors"]["left_foot"]
    assert inactive["support_slip_m_per_s"]["mean"] is None
    assert inactive["contact_position_error_m"]["max"] is None


@pytest.mark.parametrize("variant", list("ABCD"))
def test_experiments_preserve_contact_priority_and_base_anchor(variant):
    solver = make_solver(
        # Isolate contact/seed compatibility with the existing actuated-only
        # solve mode. Floating-base linearization is tested with mj_integratePos
        # above; this case must not depend on base-orientation soft tolerances.
        q_a_init_idx=0,
        enabled=True,
        pose_initialization={"enabled": variant in "BD"},
        interaction_mesh={"weighting": "exponential" if variant in "CD" else "uniform"},
        effectors={
            "endpoint": {"robot_body": "tip", "contact_offsets": [[0.0, 0.0, 0.0]]}
        },
        base_position_constraint={
            "enabled": True,
            "max_xy_error": 1e-5,
            "max_z_error": 1e-5,
        },
    )
    q = solver.robot_model.qpos0.copy()
    goal = q.copy()
    goal[8:] = [0.25, 0.1, 0.5]
    source, orientations = source_at(solver, goal)
    target = source[-1].copy()
    result = solver.retarget_frame(
        source,
        q,
        max_iter=10,
        target_base_position=q[:3],
        target_base_orientation=q[3:7],
        source_joint_orientations=orientations,
        end_effector_targets={
            "endpoint": {
                "contact": True,
                "contact_position": target,
                "tracking_position": target,
                "confidence": 1.0,
            }
        },
    )
    actual, _ = source_at(solver, result)
    assert np.linalg.norm(actual[-1] - target) < 0.002
    np.testing.assert_allclose(result[:3], q[:3], atol=1.1e-5)
    assert solver.self_collision_enabled and solver.hard_penetration_constraint
