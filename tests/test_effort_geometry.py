"""Actual finite-triangle contacts, unsupported dynamics and tolerance analysis."""

import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting.effort_audit import evaluate_effort
from omniretargeting.effort_contacts import contact_evidence, load_fixed_objects
from omniretargeting.effort_dynamics import allocate_circular, motor_demand
from omniretargeting.effort_geometry import EffortGeometry
from omniretargeting.effort_certificate import certify_circular_unsupported


def _case(height=.06, *, object_surface=False):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="torso"><freejoint/><geom type="sphere" size=".05" mass="1"/>
        <body name="foot" pos=".2 0 0"><joint name="waist" axis="0 1 0"/>
          <geom type="sphere" size=".05" mass="1"/></body>
      </body></worldbody></mujoco>''')
    floor = trimesh.creation.box([2., 2., .02])
    floor.apply_translation([0., 0., -.01])
    saved = {"surface_contact_mask": np.ones((3, 2), bool),
             "surface_contact_robot_geoms": np.array([0, 1]),
             "surface_contact_robot_local_points": np.tile([0., 0., -.05], (2, 1)),
             "surface_contact_environment_w": np.tile([[0., 0., 0.], [.2, 0., 0.]], (3, 1, 1)),
             "surface_contact_normals_w": np.tile([0., 0., 1.], (3, 2, 1)),
             "surface_contact_surface_ids": np.full((3, 2), int(object_surface)),
             "contact_role_surface_support": np.ones((3, 2), bool)}
    evidence = contact_evidence(saved, model, np.zeros((int(object_surface), 3), bool))
    scene = floor.copy()
    if object_surface:
        scene.apply_translation([0., 0., -2.])
    evidence["geometry"] = EffortGeometry(model, scene, [{"mesh": floor}] if object_surface else [], 3, evidence, enclosed=False)
    poses = np.tile(model.qpos0, (3, 1)); poses[:, 2] = height
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    return model, poses, evidence


def test_ten_mm_air_gap_is_definite_unsupported_and_tolerance_is_separate():
    model, poses, evidence = _case()
    evaluation = evaluate_effort(model, poses, 30., np.array([1]), np.array([5.]), evidence)
    row = evaluation["rows"][0]
    assert row["status"] == "infeasible_unsupported"
    assert row["reason"] == "source_support_not_attained"
    assert row["tolerance_15mm"]["status"] == "feasible"
    assert row["closest_outer_cone_residual"][2] == pytest.approx(19.62)
    assert len(row["missing_source_support_pairs"]) == 2


def test_tangential_correspondence_drift_does_not_turn_actual_floor_into_air():
    model, poses, evidence = _case(height=.05)
    poses[:, 0] = .3
    row = evaluate_effort(model, poses, 30., np.array([1]), np.array([5.]), evidence)["rows"][0]
    assert row["status"] == "feasible"
    pair = next(d for d in row["original_triangle_pairs"] if d["geom"] == 0)
    assert pair["actual_contact"]
    assert pair["source_correspondence_error_range_m"][0] == pytest.approx(.3)


def test_finite_surface_is_not_extended_past_its_edge():
    model, poses, evidence = _case(height=.05)
    poses[:, 0] = 2.
    row = evaluate_effort(model, poses, 30., np.array([1]), np.array([5.]), evidence)["rows"][0]
    assert row["status"] == "infeasible_unsupported"
    assert row["tolerance_15mm"]["status"] == "infeasible_unsupported"


def test_ten_mm_penetration_cannot_gain_reaction_from_the_crossing_ring():
    model, poses, evidence = _case(height=.04)
    row = evaluate_effort(model, poses, 30., np.array([1]), np.array([5.]), evidence)["rows"][0]
    assert row["status"] == "infeasible_unsupported"
    assert row["tolerance_15mm"]["status"] == "feasible"
    assert row["reason"] == "penetrating_support_geom"
    assert all(p["gap_m"] < -.005 for p in row["missing_source_support_pairs"])


def test_object_load_unknown_requires_actual_robot_object_contact():
    model, poses, evidence = _case(height=.06, object_surface=True)
    rows = evaluate_effort(model, poses, 30., np.array([1]), np.array([5.]), evidence)["rows"]
    assert rows[0]["status"] == "infeasible_unsupported"
    assert rows[0]["unmodelled_object_contacts"] == 0
    assert rows[0]["tolerance_15mm"]["status"] == "unknown_object_load"
    poses[:, 2] = .05
    row = evaluate_effort(model, poses, 30., np.array([1]), np.array([5.]), evidence)["rows"][0]
    assert row["status"] == "unknown_object_load"
    assert {p["body"] for p in row["unmodelled_object_contact_pairs"]} == {"torso", "foot"}
    assert all(abs(p["gap_m"]) <= .005 for p in row["unmodelled_object_contact_pairs"])


def test_no_contact_can_still_be_feasible_in_ballistic_free_fall():
    model, poses, evidence = _case(height=1.)
    time = np.arange(3) / 30.
    poses[:, 2] -= .5 * 9.81 * time**2
    row = evaluate_effort(model, poses, 30., np.array([1]), np.array([5.]), evidence)["rows"][0]
    assert row["status"] == "feasible"


def test_missing_geometry_does_not_prove_unsupported():
    model, poses, evidence = _case()
    evidence.pop("geometry")
    row = evaluate_effort(model, poses, 30., np.array([1]), np.array([5.]), evidence)["rows"][0]
    assert row["status"] == "unknown_contact_geometry"


def test_unoriented_near_face_reports_missing_direction_evidence():
    model, poses, evidence = _case(height=.05)
    # An open scan plus tangential source normals cannot specify which side
    # supplies a compressive reaction; proximity alone must not invent one.
    floor = trimesh.creation.box([2., 2., .02])
    floor.apply_translation([0., 0., -.01])
    floor.update_faces(floor.face_normals[:, 2] > .9)
    evidence["normals_w"][:] = [1., 0., 0.]
    evidence["geometry"] = EffortGeometry(model, floor, [], 3, evidence)
    row = evaluate_effort(model, poses, 30., np.array([1]), np.array([5.]), evidence)["rows"][0]
    assert row["status"] == "unknown_contact_geometry"
    assert row["uncertain_near_surface_points"] > 0
    for pair in row["original_triangle_pairs"]:
        assert pair["uncertain_points_5mm"] > 0
        assert pair["orientation_evidence_distance_m"] < .15
        assert pair["orientation_evidence_abs_alignment"] == pytest.approx(0.)
        assert not pair["surface_watertight_oriented"]


def test_exact_circular_cone_resolves_inner_polygon_false_failure():
    model, poses, _ = _case(height=.05)
    data = mujoco.MjData(model)
    acceleration = np.zeros(model.nv)
    acceleration[:2] = 9.81 * .59 * np.array([np.cos(np.pi/8), np.sin(np.pi/8)])
    demand = motor_demand(model, data, poses[1], np.zeros(model.nv), acceleration)
    # Place the reaction through the total CoM to avoid an unrelated tipping moment.
    point = np.array([.1, 0., .05])
    result = allocate_circular(model, data, demand, [{"body": 1, "point": point, "normal": [0, 0, 1]}],
                               np.arange(6), np.array([6]), np.array([5.]))
    assert result["status"] == "balanced"
    assert result["base_residual"] < 1e-6


def test_circular_separator_proves_outside_pressure_footprint_without_fake_root_force():
    model, poses, _ = _case(height=.05)
    data = mujoco.MjData(model)
    motor_demand(model, data, poses[1], np.zeros(model.nv), np.zeros(model.nv))
    contacts = [{"body": 1, "point": np.array([x, 0., .05]), "normal": np.array([0., 0., 1.])}
                for x in (-.1, .1)]
    demand = np.zeros(model.nv); demand[2] = 10.; demand[4] = 3.
    certificate = certify_circular_unsupported(model, data, demand, contacts, np.arange(6))
    assert certificate is not None
    assert certificate["circular_base_separator_min_cone_slack"] >= 0
    assert certificate["circular_base_separator_dot_demand"] < 0
    demand[4] = .5
    assert certify_circular_unsupported(model, data, demand, contacts, np.arange(6)) is None


def test_circular_separator_does_not_misclassify_polygon_boundary_feasibility():
    model, poses, _ = _case(height=.05)
    data = mujoco.MjData(model)
    motor_demand(model, data, poses[1], np.zeros(model.nv), np.zeros(model.nv))
    demand = np.zeros(model.nv)
    demand[:2] = .59 * 10. * np.array([np.cos(np.pi/8), np.sin(np.pi/8)])
    demand[2] = 10.
    assert certify_circular_unsupported(model, data, demand,
        [{"body": 1, "point": data.xpos[1].copy(), "normal": np.array([0.,0.,1.])}], np.arange(6)) is None


@pytest.mark.parametrize("primal_status", ["numerical_unknown", "base_infeasible"])
def test_failed_primal_gets_a_verified_circular_base_certificate(monkeypatch, primal_status):
    import omniretargeting.effort_audit as audit_module
    model, poses, evidence = _case(height=.05)
    time = (np.arange(3)-1)/30.
    poses[:,0] = .5 * (.62*9.81) * time**2
    monkeypatch.setattr(audit_module, "allocate_circular", lambda *a, **k: {
        "status": primal_status, "torque": None, "utilization": None})
    row = evaluate_effort(model, poses, 30., np.array([1]), np.array([5.]), evidence)["rows"][0]
    assert row["status"] == "infeasible_unsupported"
    assert row["reason"] == "verified_circular_cone_separating_wrench"
    assert row["circular_base_separator_min_cone_slack"] >= 0
    assert row["circular_base_separator_dot_demand"] < -row["circular_base_separator_residual_allowance"]


def test_uncertified_primal_infeasibility_is_checker_error_not_missing_data(monkeypatch):
    import omniretargeting.effort_audit as audit_module
    model, poses, evidence = _case(height=.05)
    time = (np.arange(3)-1)/30.
    poses[:,0] = .5 * (.62*9.81) * time**2
    monkeypatch.setattr(audit_module, "allocate_circular", lambda *a, **k: {
        "status": "base_infeasible", "torque": None, "utilization": None})
    monkeypatch.setattr(audit_module, "certify_circular_unsupported", lambda *a, **k: None)
    row = evaluate_effort(model, poses, 30., np.array([1]), np.array([5.]), evidence)["rows"][0]
    assert row["status"] == "audit_error_numerical"
    assert row["reason"] == "contact_force_solver_did_not_certify"


def test_selected_frame_keeps_original_neighbors_and_frame_number():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <joint axis="0 1 0"/><geom size=".1" mass="1" pos="1 0 0"/>
    </body></worldbody></mujoco>''')
    poses = np.array([0., .2, 0., .1, .1, 0.])[:,None]
    args = model, poses, 30., np.array([0]), np.array([1000.])
    whole = evaluate_effort(*args)
    selected = evaluate_effort(*args, frame_indices=[4])
    assert selected["frames"].tolist() == [4]
    for key in ("velocity", "acceleration", "torque"):
        np.testing.assert_array_equal(selected[key][0], whole[key][3])


def test_attained_contact_with_impossible_upward_reaction_is_definite_failure():
    model, poses, evidence = _case(height=.05)
    # Downward acceleration greater than gravity needs a pulling reaction.
    time = np.arange(3) / 30.
    poses[:, 2] = .05 - .5 * 20. * (time - time[1])**2
    row = evaluate_effort(model, poses, 30., np.array([1]), np.array([5.]), evidence)["rows"][0]
    assert row["status"] == "infeasible_unsupported"
    assert row["reason"] == "friction_cone_or_support_wrench"


def test_still_object_ten_mm_above_floor_is_not_primary_world_support(tmp_path):
    floor = trimesh.creation.box([2., 2., .02]); floor.apply_translation([0., 0., -.01])
    floor.export(tmp_path / "static_scene.obj")
    cube = trimesh.creation.box([.1, .1, .1])
    prepared = tmp_path / "prepared.npy"
    np.save(prepared, {"surface_config": {"distance": .03, "sticking_speed": .04}})
    objects = tmp_path / "objects.npy"
    np.save(objects, [{"mesh": cube, "translations": np.tile([0.,0.,.06], (3,1))}])
    primary = load_fixed_objects(prepared, objects, 3, 30.)
    sensitivity = load_fixed_objects(prepared, objects, 3, 30., distance=.015)
    assert not primary.any()
    assert sensitivity.all()


def test_contact_footprint_extremes_balance_com_without_inventing_a_root_motor():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="torso"><freejoint/><geom type="sphere" size=".05" mass="1"/>
        <body name="foot" pos=".2 0 -.3"><joint name="ankle" axis="0 1 0"/>
          <geom type="box" size=".2 .05 .01" mass="1"/></body>
      </body></worldbody></mujoco>''')
    _, _, old = _case(height=.31)
    evidence = dict(old); evidence.pop("geometry")
    floor = trimesh.creation.box([2.,2.,.02]); floor.apply_translation([0.,0.,-.01])
    evidence["geometry"] = EffortGeometry(model, floor, [], 3, evidence, enclosed=False)
    poses = np.tile(model.qpos0, (3,1)); poses[:,2] = .31
    row = evaluate_effort(model, poses, 30., np.array([1]), np.array([5.]), evidence)["rows"][0]
    assert row["status"] == "feasible"
    assert row["base_residual"] < 1e-6
    # The sole center is x=.2 but COM x=.1; a center-only force would tip.
    loaded = row["loaded_contacts"]
    normal = sum(c["normal_force_N"] for c in loaded)
    cop = sum(c["normal_force_N"] * c["point_w_m"][0] for c in loaded) / normal
    assert cop == pytest.approx(.1, abs=1e-6)


def test_side_penetration_does_not_erase_a_real_floor_support_patch():
    model, poses, evidence = _case(height=.05)
    floor = trimesh.creation.box([2.,2.,.02]); floor.apply_translation([0.,0.,-.01])
    wall = trimesh.creation.box([.1,.2,.2]); wall.apply_translation([.09,0.,.1])
    scene = trimesh.util.concatenate([floor, wall])
    evidence["geometry"] = EffortGeometry(model, scene, [], 3, evidence, enclosed=False)
    row = evaluate_effort(model, poses, 30., np.array([1]), np.array([5.]), evidence)["rows"][0]
    assert row["status"] == "feasible"
    pair = next(p for p in row["original_triangle_pairs"] if p["geom"] == 0)
    assert pair["minimum_any_surface_gap_m"] < -.005
    assert pair["gap_m"] == pytest.approx(0., abs=1e-6)
    assert not row["missing_source_support_pairs"]


def test_thin_floor_opposite_side_cannot_hide_deep_support_intrusion():
    model, poses, evidence = _case(height=.04)
    floor = trimesh.creation.box([2.,2.,.002]); floor.apply_translation([0.,0.,-.001])
    evidence["geometry"] = EffortGeometry(model, floor, [], 3, evidence, enclosed=False)
    row = evaluate_effort(model, poses, 30., np.array([1]), np.array([5.]), evidence)["rows"][0]
    assert row["status"] == "infeasible_unsupported"
    assert len(row["missing_source_support_pairs"]) == 2
    assert all(p["gap_m"] == pytest.approx(-.01) for p in row["missing_source_support_pairs"])
