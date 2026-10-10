"""The smoothness audit follows source chatter and URDF mechanics."""

import numpy as np
import mujoco
import pytest

from scripts.review_joint_smoothness import audit
from omniretargeting.collision_manifest import collision_manifest


def _robot(tmp_path):
    text = """<robot name="slider">
  <link name="base"/>
  <link name="link"><inertial><mass value="1"/>
    <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
  </inertial></link>
  <joint name="slide" type="prismatic">
    <parent link="base"/><child link="link"/><axis xyz="1 0 0"/>
    <limit lower="-10" upper="10" effort="20" velocity="2"/>
  </joint>
</robot>"""
    path = tmp_path / "slider.urdf"
    path.write_text(text)
    return path


def _motion(path, values):
    path.write_bytes(b"")
    np.savez(path, qpos=values[:, None], framerate=30.0)
    return path


def test_smooth_low_frequency_correction_is_not_chatter(tmp_path):
    model = _robot(tmp_path)
    frames = np.arange(30)
    before = np.zeros(30)
    after = 0.1 * (1 - np.cos(np.pi * frames / 29)) / 2
    report = audit(str(model), _motion(tmp_path / "before.npz", before),
                   _motion(tmp_path / "after.npz", after))
    assert report["violating_joint_time_order_entries"] == 0
    assert report["velocity_limit_violations"] == 0
    assert report["effort_limit_violations"] == 0
    assert report["effort_dynamics"]["unaudited_endpoint_frames"] == [0, 29]
    assert not report["all_individual_budgets_pass"]


def test_alternating_correction_is_chatter_even_below_velocity_limit(tmp_path):
    model = _robot(tmp_path)
    before = np.zeros(30)
    after = 0.02 * (-1.0) ** np.arange(30)
    report = audit(str(model), _motion(tmp_path / "before.npz", before),
                   _motion(tmp_path / "after.npz", after))
    assert report["violating_joint_time_order_entries"] > 0
    assert report["velocity_limit_violations"] == 0
    assert report["effort_limit_violations"] == 28
    effort = report["joints"]["slide"]["mechanical_limits"]["effort"]
    assert effort["worst_frame"] == 1
    assert all(row["classification"] == "added_inertial_demand" for row in effort["attribution"])


def test_short_motion_does_not_invent_endpoint_acceleration(tmp_path):
    report = audit(_robot(tmp_path), _motion(tmp_path / "before.npz", np.zeros(2)),
                   _motion(tmp_path / "after.npz", np.zeros(2)))
    assert report["effort_dynamics"]["unaudited_endpoint_frames"] == [0, 1]
    assert report["joints"]["slide"]["mechanical_limits"]["effort"]["peak"] is None
    assert not report["all_individual_budgets_pass"]


def test_saved_collision_hash_uses_compiled_semantics_not_xml_bytes(tmp_path):
    path = _robot(tmp_path)
    model = mujoco.MjModel.from_xml_path(str(path))
    before = _motion(tmp_path / "before.npz", np.zeros(3))
    after = tmp_path / "after.npz"
    np.savez(after, qpos=np.zeros((3, 1)), framerate=30.,
             collision_model_sha256=collision_manifest(model)["sha256"])
    report = audit(path, before, after)
    assert report["mechanical_model"]["collision_model_sha256"] == str(np.load(after)["collision_model_sha256"])
    np.savez(after, qpos=np.zeros((3, 1)), framerate=30., collision_model_sha256="wrong")
    with pytest.raises(ValueError, match="collision_model_sha256"):
        audit(path, before, after)


def test_added_gravity_leverage_is_not_labelled_as_source_or_acceleration(tmp_path):
    path = tmp_path / "pendulum.xml"
    path.write_text('''<mujoco><worldbody><body>
      <joint name="pitch" axis="0 1 0" actuatorfrcrange="-20 20"/>
      <inertial mass="3" pos="1 0 0" diaginertia=".1 .1 .1"/>
    </body></worldbody></mujoco>''')
    before = _motion(tmp_path / "before.npz", np.full(3, -np.pi / 2))
    after = _motion(tmp_path / "after.npz", np.zeros(3))
    report = audit(path, before, after)
    assert report["effort_limit_violations"] == 1
    attribution = report["effort_dynamics"]["overload_frame_attribution"][0]
    assert attribution["classification"] == "added_pose_or_contact_geometry"
    assert attribution["source_inner_cone_feasible"]
    assert attribution["source_derivatives_at_final_pose_overload"]


def test_effort_roundoff_is_in_force_units_not_limit_scaled(tmp_path):
    before = _motion(tmp_path / "before.npz", np.zeros(3))
    after = _motion(tmp_path / "after.npz", np.array([0., 0., 20.000002 / 30**2]))
    report = audit(_robot(tmp_path), before, after)
    assert report["effort_limit_violations"] == 1
