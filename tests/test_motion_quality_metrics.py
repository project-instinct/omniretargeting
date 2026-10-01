"""Velocity-direction, object-relative drift and orientation-target metrics."""

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting.motion_quality import (
    object_relative_drift_degrees,
    orientation_target_degrees,
    velocity_direction_degrees,
)


def test_velocity_direction_ignores_scale_speed_and_offset():
    t = np.linspace(0.0, 1.0, 31)[:, None]
    source = np.stack([np.hstack([t, t**2, 0 * t]), np.hstack([0 * t, t, t])], axis=1)
    robot = 0.6 * source + np.array([1.0, -2.0, 0.5])
    np.testing.assert_allclose(velocity_direction_degrees(robot, source, 30.0), 0.0, atol=1e-5)
    reversed_robot = robot[::-1]
    reversed_source = source[::-1]
    angles = velocity_direction_degrees(reversed_robot, source, 30.0)
    assert np.nanmax(angles) == pytest.approx(180.0, abs=1e-5)
    np.testing.assert_allclose(
        velocity_direction_degrees(reversed_robot, reversed_source, 30.0), 0.0, atol=1e-5
    )


def test_velocity_direction_is_undefined_for_slow_source_or_static_robot():
    source = np.zeros((10, 1, 3))
    source[:, 0, 0] = np.arange(10) * 0.001  # 3 cm/s at 30 fps
    robot = source.copy()
    assert np.isnan(velocity_direction_degrees(robot, source, 30.0)).all()
    fast = source * 100
    assert np.isnan(velocity_direction_degrees(np.zeros_like(fast), fast, 30.0)).all()
    single = velocity_direction_degrees(robot[:1], source[:1], 30.0)
    assert single.shape == (1, 1) and np.isnan(single).all()


def _rotations(axis, angles_deg):
    return Rotation.from_rotvec(np.outer(np.deg2rad(angles_deg), axis)).as_matrix()


def test_object_drift_cancels_rigid_grip_and_constant_frame_offset():
    frames = 12
    objects = _rotations([0, 0, 1], np.linspace(0, 90, frames))
    grip_source = Rotation.from_euler("xyz", [10, -20, 30], degrees=True).as_matrix()
    grip_robot = Rotation.from_euler("xyz", [-40, 5, 70], degrees=True).as_matrix()
    offset = Rotation.from_euler("xyz", [0, 25, -15], degrees=True).as_matrix()
    source = objects @ grip_source
    # A different grip and a constant robot/source body-frame offset.
    robot = objects @ grip_robot @ offset
    mask = np.ones(frames, bool)
    np.testing.assert_allclose(
        object_relative_drift_degrees(robot, source, objects, mask), 0.0, atol=1e-5
    )


def test_object_drift_measures_extra_turn_per_segment():
    frames = 10
    objects = _rotations([1, 0, 0], np.linspace(0, 45, frames))
    source = objects.copy()
    turn = np.r_[np.linspace(0, 30, 5), np.linspace(40, 60, 5)]
    robot = objects @ _rotations([0, 1, 0], turn)
    mask = np.r_[np.ones(5, bool), [False], np.ones(4, bool)]
    drift = object_relative_drift_degrees(robot, source, objects, mask)
    np.testing.assert_allclose(drift[:5], np.linspace(0, 30, 5), atol=1e-5)
    assert np.isnan(drift[5])
    # A new run starts from its own first frame.
    np.testing.assert_allclose(drift[6:], turn[6:] - turn[6], atol=1e-5)


XML = """<mujoco><worldbody>
<body name="arm"><joint name="twist" axis="1 0 0"/><geom size=".05"/>
 <body name="hand" pos=".3 0 0"><joint name="flex" axis="0 1 0"/><geom size=".03"/></body>
</body></worldbody></mujoco>"""


def test_orientation_targets_world_and_parent_relative():
    model = mujoco.MjModel.from_xml_string(XML)
    qpos = np.array([[0.0, 0.0], [0.2, 0.3], [0.4, -0.1]])
    flex = _rotations([0, 1, 0], np.rad2deg(qpos[:, 1]))
    world = _rotations([1, 0, 0], np.rad2deg(qpos[:, 0])) @ flex
    targets = [
        {"robot_body": "hand", "rotations": world},
        {"robot_body": "arm", "rotations": _rotations([1, 0, 0], np.zeros(3))},
    ]
    errors = orientation_target_degrees(model, qpos, targets)
    np.testing.assert_allclose(errors["hand"], 0.0, atol=1e-5)
    np.testing.assert_allclose(errors["arm"], np.rad2deg(qpos[:, 0]), atol=1e-5)
    relative = orientation_target_degrees(
        model, qpos, [{"robot_body": "hand", "parent_body": "arm", "rotations": flex}]
    )
    np.testing.assert_allclose(relative["hand"], 0.0, atol=1e-5)
