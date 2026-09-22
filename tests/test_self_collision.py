"""Collision geometry, near-contact prediction and integrated self avoidance."""

import cvxpy as cp
import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting.retargeting import GenericInteractionRetargeter
from omniretargeting.retargeting_defaults import CONTACT_RETARGETING_DEFAULTS
from omniretargeting.self_collision import (
    SelfCollisionQuery,
    bounded_collision_constraints,
    summarize_self_collisions,
)
from omniretargeting.trajectory_refinement import refine_joint_trajectory
from omniretargeting.utils import calculate_laplacian_coordinates


XML = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="root">
      <freejoint/>
      <geom name="visual" type="sphere" size="1" mass="1" contype="0" conaffinity="0"/>
      <body name="left" pos="-0.08 0 0">
        <joint name="left_slide" type="slide" axis="1 0 0" range="-0.5 0.5"/>
        <geom name="left_collision" type="sphere" size="0.1"/>
      </body>
      <body name="right" pos="0.08 0 0">
        <joint name="right_slide" type="slide" axis="1 0 0" range="-0.5 0.5"/>
        <geom name="right_collision" type="sphere" size="0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_visual_only_geoms_are_ignored_even_in_explicit_contact_pairs():
    model = mujoco.MjModel.from_xml_string(XML.replace(
        "</mujoco>", '<contact><pair geom1="visual" geom2="left_collision"/></contact></mujoco>'
    ))
    query = SelfCollisionQuery(model)
    assert query.geom_ids.tolist() == [1, 2]
    assert query.pairs.tolist() == [[1, 2]]
    assert query.query(model.qpos0)[0]["clearance"] == pytest.approx(-0.04)
    q = model.qpos0.copy()
    q[7:9] = [-0.2, 0.2]
    # Both collision bodies still overlap the large visual sphere.
    assert query.query(q) == []


@pytest.mark.parametrize("exclusion", ["mask", "xml", "weld", "parent"])
def test_collision_pair_filters(exclusion):
    xml = XML
    if exclusion == "xml":
        xml = xml.replace("</mujoco>", '<contact><exclude body1="left" body2="right"/></contact></mujoco>')
    if exclusion == "weld":
        xml = xml.replace('<joint name="left_slide" type="slide" axis="1 0 0" range="-0.5 0.5"/>', "")
        xml = xml.replace('<joint name="right_slide" type="slide" axis="1 0 0" range="-0.5 0.5"/>', "")
    if exclusion == "parent":
        xml = xml.replace('</body>\n      <body name="right"', '<body name="right"')
        xml = xml.replace('  </worldbody>', '    </body>\n  </worldbody>')
    model = mujoco.MjModel.from_xml_string(xml)
    if exclusion == "mask":
        model.geom_contype[1:] = [1, 2]
        model.geom_conaffinity[1:] = [1, 2]
    assert len(SelfCollisionQuery(model).pairs) == 0


def test_nearby_pairs_are_detected_before_contact_without_changing_model():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    data.qpos[8] = 0.065  # 25 mm clearance; no ordinary MuJoCo contact.
    mujoco.mj_forward(model, data)
    assert data.ncon == 0
    margins = model.geom_margin.copy()
    query = SelfCollisionQuery(model)
    assert query.query(data=data)[0]["clearance"] == pytest.approx(0.025)
    np.testing.assert_array_equal(model.geom_margin, margins)


@pytest.mark.parametrize("separation", [0.0, 0.04, 0.06])
def test_signed_relative_jacobian_matches_finite_difference(separation):
    model = mujoco.MjModel.from_xml_string(XML)
    query = SelfCollisionQuery(model)
    q = model.qpos0.copy()
    q[8] = separation
    margin = model.geom_margin.copy()
    row = query.query(q)[0]
    jac = query.jacobian(row, query.data)
    finite = np.zeros(model.nv)
    for dof in range(model.nv):
        tangent = np.zeros(model.nv)
        tangent[dof] = 1
        samples = []
        for sign in (-1, 1):
            perturbed = q.copy()
            mujoco.mj_integratePos(model, perturbed, tangent, sign * 1e-6)
            samples.append(query.query(perturbed)[0]["clearance"])
        finite[dof] = (samples[1] - samples[0]) / 2e-6
    np.testing.assert_allclose(jac, finite, atol=1e-8)
    np.testing.assert_allclose(jac[:6], 0, atol=1e-12)
    np.testing.assert_array_equal(model.geom_margin, margin)


def test_deep_penetration_recovers_with_bounded_steps_and_slack():
    model = mujoco.MjModel.from_xml_string(XML)
    query = SelfCollisionQuery(model)
    q = model.qpos0.copy()
    original_root = q[:7].copy()
    depths = []
    for _ in range(6):
        rows = query.query(q)
        depths.append(max(-rows[0]["clearance"], 0))
        step = cp.Variable(2)
        constraints, penalty = bounded_collision_constraints(
            query, rows, query.data, step, np.array([6, 7])
        )
        problem = cp.Problem(cp.Minimize(penalty + cp.sum_squares(step)),
                             constraints + [cp.abs(step) <= 0.005])
        problem.solve(solver="CLARABEL")
        assert problem.status == cp.OPTIMAL
        before = query.depths(q)
        q[7:9] += step.value
        assert query.acceptable(before, query.depths(q))
    assert depths[0] == pytest.approx(0.04)
    assert depths[-1] < 0.001
    assert np.all(np.diff(depths) <= 1e-8)
    np.testing.assert_array_equal(q[:7], original_root)


def test_framewise_self_collision_works_without_scene_constraints():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    targets = data.xpos[[model.body(n).id for n in ("left", "right")]].copy()
    adjacency = [[1], [0]]
    laplacian = calculate_laplacian_coordinates(targets, adjacency, uniform_weight=True)
    depths = []
    for enabled in (False, True):
        solver = GenericInteractionRetargeter(
            model, data, trimesh.creation.box(), {"a": "left", "b": "right"}, 1.0,
            hard_penetration_constraint=False, self_collision_enabled=enabled,
            terrain_sample_seed=0,
        )
        q = model.qpos0.copy()
        for _ in range(6):
            q, cost = solver._single_optimization_step(
                q, laplacian, adjacency, np.empty((0, 3)), target_human_joints=targets,
            )
            assert np.isfinite(cost)
        depths.append(summarize_self_collisions(SelfCollisionQuery(model), q[None])["max_depth"])
    assert depths[0] > 0.035
    assert depths[1] < 0.001
    assert CONTACT_RETARGETING_DEFAULTS["self_collision_enabled"] is True


def test_refinement_handles_self_collision_without_scene_or_effector_tasks():
    model = mujoco.MjModel.from_xml_string(XML)
    before = np.tile(model.qpos0, (8, 1))
    before[:, 0] = np.linspace(0, 2, len(before))
    cfg = {"iterations": 6, "trust_region": 0.01}
    off = refine_joint_trajectory(model, before, {}, {}, {**cfg, "self_collision_enabled": False})
    after = refine_joint_trajectory(model, before, {}, {}, {**cfg, "self_collision_enabled": True})
    np.testing.assert_array_equal(off, before)
    np.testing.assert_array_equal(after[:, :7], before[:, :7])
    assert summarize_self_collisions(SelfCollisionQuery(model), after)["max_depth"] < 0.001
    # Constant internal correction has no additional temporal jitter.
    np.testing.assert_allclose(np.diff(after[:, 7:], axis=0), 0.0, atol=1e-8)


def test_refinement_rechecks_collisions_created_by_smoothing():
    model = mujoco.MjModel.from_xml_string(XML)
    before = np.tile(model.qpos0, (8, 1))
    before[:, 7] = np.where(np.arange(8) % 2, -0.2, 0.36)
    query = SelfCollisionQuery(model)
    assert summarize_self_collisions(query, before)["max_depth"] < 1e-12
    after = refine_joint_trajectory(
        model, before, {}, {}, {"iterations": 3, "reference_sigma": 10.0,
                                "acceleration_sigma": 0.001, "trust_region": 0.2},
    )
    assert summarize_self_collisions(query, after)["max_depth"] <= 0.001001


def test_unknown_distance_is_an_error_instead_of_zero_depth(monkeypatch):
    model = mujoco.MjModel.from_xml_string(XML)
    monkeypatch.setattr(mujoco, "mj_geomDistance", lambda *_args: float("nan"))
    with pytest.raises(RuntimeError, match="nonfinite"):
        SelfCollisionQuery(model).query(model.qpos0)


def test_reachable_contact_is_preserved_while_other_body_exits():
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    targets = data.xpos[[model.body(n).id for n in ("left", "right")]].copy()
    solver = GenericInteractionRetargeter(
        model, data, trimesh.creation.box(), {"a": "left", "b": "right"}, 1.0,
        q_a_init_idx=0, hard_penetration_constraint=False,
        end_effector_config={
            "enabled": True, "hierarchy": True, "free_motion_priority": "balanced",
            "priority_absolute_tolerance": 1e-5,
            "effectors": {"anchor": {"robot_body": "left", "contact_sigma": 0.01}},
        },
    )
    laplacian = calculate_laplacian_coordinates(targets, [[1], [0]], uniform_weight=True)
    q = model.qpos0.copy()
    for _ in range(6):
        q, cost = solver._single_optimization_step(
            q, laplacian, [[1], [0]], np.empty((0, 3)), target_human_joints=targets,
            end_effector_targets={"anchor": {
                "contact": True, "contact_mode": "sticking", "contact_position": targets[0],
                "tracking_position": targets[0], "surface_normal": np.array([0., 0., 1.]),
            }},
        )
        assert np.isfinite(cost)
    assert abs(q[7]) < 1e-5
    assert q[8] > 0.039
    np.testing.assert_array_equal(q[:3], model.qpos0[:3])
    assert summarize_self_collisions(SelfCollisionQuery(model), q[None])["max_depth"] < 0.001


def test_clear_endpoints_cannot_jump_through_another_collision_body():
    model = mujoco.MjModel.from_xml_string(XML)
    query = SelfCollisionQuery(model)
    previous = model.qpos0.copy()
    previous[7] = -0.2
    candidate = previous.copy()
    candidate[7] = 0.38
    assert max(query.depths(previous).values(), default=0) == 0
    assert max(query.depths(candidate).values(), default=0) == 0
    assert max(query.segment(previous, candidate)["depths"].values()) > 0.1
    solver = GenericInteractionRetargeter.__new__(GenericInteractionRetargeter)
    solver.robot_model = model
    solver.self_collision_enabled = True
    solver._self_collision_query = query
    guarded = solver._guard_self_collision_step(previous, candidate, previous)
    assert max(query.segment(previous, guarded)["depths"].values(), default=0) <= 0.001001
    assert guarded[7] < candidate[7]


def test_self_motion_sampling_ignores_common_root_translation_and_rotation():
    model = mujoco.MjModel.from_xml_string(XML)
    query = SelfCollisionQuery(model)
    previous = model.qpos0.copy()
    candidate = previous.copy()
    candidate[:3] = [100, -20, 30]
    candidate[3:7] = [0, 1, 0, 0]
    segment = query.segment(previous, candidate)
    assert segment["samples"] == 2
    assert not segment["subdivision_limit"]


def test_segment_query_reports_incomplete_subdivision_budget():
    model = mujoco.MjModel.from_xml_string(XML)
    query = SelfCollisionQuery(model, {"max_substeps": 1})
    previous = model.qpos0.copy()
    candidate = previous.copy()
    previous[7], candidate[7] = -0.2, 0.38
    assert query.segment(previous, candidate)["subdivision_limit"]
