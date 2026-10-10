"""Analytic balance, friction bracketing, and false-collision-force regression."""

import mujoco
import numpy as np
import pytest

from omniretargeting.effort_contacts import contact_evidence, frame_contacts
from omniretargeting.effort_dynamics import (
    allocate, contact_matrix, derivatives, friction_rays, mechanical_limits, motor_demand,
)


def _floating():
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="torso"><freejoint/><inertial mass="1" pos="0 0 0" diaginertia=".1 .1 .1"/>
        <geom type="sphere" size=".05"/>
        <body name="lower" pos=".4 0 0"><joint name="waist" axis="0 1 0"/>
          <inertial mass="2" pos="0 0 0" diaginertia=".1 .1 .1"/>
          <geom type="sphere" size=".05"/>
        </body>
      </body></worldbody></mujoco>''')


def test_seat_and_foot_support_balance_weight_without_fake_base_or_waist_torque():
    model = _floating()
    data = mujoco.MjData(model)
    demand = motor_demand(model, data, model.qpos0, np.zeros(model.nv), np.zeros(model.nv))
    contacts = [{"body": body, "point": data.xpos[body], "normal": [0, 0, 1]} for body in (1, 2)]
    matrix = contact_matrix(model, data, contacts)
    result = allocate(demand, matrix, np.arange(6), np.array([6]), np.array([5.0]))
    assert result["status"] == "balanced"
    assert result["base_residual"] < 1e-6
    assert abs(result["torque"][0]) < 1e-6
    forces = result["coefficients"].reshape(2, -1).sum(axis=1)
    np.testing.assert_allclose(forces, [9.81, 19.62], atol=1e-6)
    unsupported = allocate(demand, np.empty((model.nv, 0)), np.arange(6), [6], [5.0])
    assert unsupported["status"] == "base_infeasible"
    assert unsupported["torque"] is None


def test_alternative_contact_split_can_satisfy_effort():
    # f1+f2=10. Minimum-norm f=(5,5) would demand 5 Nm > 2 Nm;
    # the admissible f=(10,0) unloads the joint completely.
    result = allocate(np.array([10., 10.]), np.array([[1., 1.], [1., 0.]]), [0], [1], np.array([2.]))
    assert result["utilization"] < 1e-6
    np.testing.assert_allclose(result["coefficients"], [10, 0])


def test_true_motor_overload_is_detected_with_balanced_base():
    result = allocate(np.array([10., 8.]), np.array([[1.], [0.]]), [0], [1], np.array([5.]))
    assert result["status"] == "balanced"
    assert result["utilization"] == pytest.approx(1.6)
    assert result["torque"][0] == pytest.approx(8)


def test_friction_inner_outer_bracket_never_claims_inner_failure_as_proof():
    inner = friction_rays([0, 0, 1], .6)
    outer = friction_rays([0, 0, 1], .6, outer=True)
    assert np.max(np.linalg.norm(inner[:2], axis=0)) == pytest.approx(.6)
    assert np.min(np.linalg.norm(outer[:2], axis=0)) > .6
    # Midway between rays: circular cone admits a .59 tangential load.
    direction = np.array([np.cos(np.pi/8), np.sin(np.pi/8)])
    demand = np.r_[.59 * direction, 1., 0.]
    first = allocate(demand, np.r_[inner, np.zeros((1, 8))], [0, 1, 2], [3], np.array([1.]))
    second = allocate(demand, np.r_[outer, np.zeros((1, 8))], [0, 1, 2], [3], np.array([1.]))
    assert first["status"] == "base_infeasible"
    assert second["status"] == "balanced"


def test_centered_frame_and_raw_spike_acceleration():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <joint type="slide"/><geom type="sphere" size=".1" mass="1"/>
    </body></worldbody></mujoco>''')
    time = np.arange(6) / 30
    frames, velocity, acceleration = derivatives(model, (time**2)[:, None], 30.)
    np.testing.assert_array_equal(frames, [1, 2, 3, 4])
    np.testing.assert_allclose(velocity[:, 0], 2 * time[1:-1])
    np.testing.assert_allclose(acceleration[:, 0], 2.)
    spike = np.zeros((6, 1)); spike[3] = .001
    _, _, acceleration = derivatives(model, spike, 30.)
    assert acceleration[2, 0] == pytest.approx(-1.8)


def test_motor_demand_includes_armature_and_compensates_passive_damping():
    model = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/>
      <worldbody><body><joint type="slide" armature=".1" damping="2"/>
        <geom type="sphere" size=".1" mass="1"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    demand = motor_demand(model, data, model.qpos0, np.array([3.]), np.array([4.]))
    assert demand[0] == pytest.approx(1.1 * 4 + 2 * 3)


def test_soft_self_collision_is_not_motor_demand():
    model = mujoco.MjModel.from_xml_string('''<mujoco><option gravity="0 0 0"/>
      <worldbody><body pos="0 0 0"><geom type="sphere" size=".1" mass="1"/></body>
      <body pos=".15 0 0"><joint type="slide" axis="1 0 0"/>
        <geom type="sphere" size=".1" mass="1"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_inverse(model, data)
    assert data.ncon > 0 and abs(data.qfrc_inverse[0]) > 20
    force = motor_demand(model, data, model.qpos0, np.zeros(1), np.zeros(1))
    assert abs(force[0]) < 1e-6


def test_unattained_or_carried_contact_cannot_support_floating_robot():
    model = _floating(); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    saved = {"surface_contact_mask": np.ones((3, 1), bool),
             "surface_contact_robot_geoms": np.array([0]),
             "surface_contact_robot_local_points": np.zeros((1, 3)),
             "surface_contact_environment_w": np.zeros((3, 1, 3)),
             "surface_contact_normals_w": np.tile([0., 0., 1.], (3, 1, 1)),
             "surface_contact_surface_ids": np.zeros((3, 1), int)}
    evidence = contact_evidence(saved, model)
    attained, _, _ = frame_contacts(evidence, model, data, 1)
    assert len(attained) == 1
    saved["surface_contact_environment_w"] += .01
    attained, optimistic, stats = frame_contacts(contact_evidence(saved, model), model, data, 1)
    assert len(attained) == 0 and len(optimistic) == 1
    assert stats["rejected_geometry"] == 1
    saved["surface_contact_surface_ids"][:] = 1
    attained, optimistic, stats = frame_contacts(contact_evidence(saved, model), model, data, 1)
    assert not attained and not optimistic
    assert stats["unmodelled_object_contacts"] == 1


def test_mesh_model_reads_original_urdf_velocity_and_effort_limits():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "robot_models/unitree_g1/g1_29dof_torsobase_mesh.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    velocity, effort, provenance = mechanical_limits(path, model)
    assert len(velocity) == len(effort) == 29
    assert velocity["waist_pitch_joint"] == 37
    assert effort["waist_pitch_joint"] == 50
    assert provenance["limits_urdf"].endswith("g1_29dof_popsicle.urdf")


def test_mismatched_model_and_urdf_effort_is_an_error(tmp_path):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <joint name="motor" actuatorfrcrange="-5 5"/><geom size=".1"/>
    </body></worldbody></mujoco>''')
    urdf = tmp_path / "limits.urdf"
    urdf.write_text('<robot><joint name="motor"><limit velocity="2" effort="6"/></joint></robot>')
    with pytest.raises(ValueError, match="effort mismatch"):
        mechanical_limits(urdf, model, urdf)
