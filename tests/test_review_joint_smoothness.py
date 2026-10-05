"""The smoothness audit follows source chatter and URDF mechanics."""

import numpy as np

from scripts.review_joint_smoothness import audit


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
    assert report["all_individual_budgets_pass"]


def test_alternating_correction_is_chatter_even_below_velocity_limit(tmp_path):
    model = _robot(tmp_path)
    before = np.zeros(30)
    after = 0.02 * (-1.0) ** np.arange(30)
    report = audit(str(model), _motion(tmp_path / "before.npz", before),
                   _motion(tmp_path / "after.npz", after))
    assert report["violating_joint_time_order_entries"] > 0
    assert report["velocity_limit_violations"] == 0
