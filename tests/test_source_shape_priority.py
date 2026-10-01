"""Unreachable endpoint evidence must not define an unrecognizable reference."""

import mujoco
import numpy as np

from omniretargeting._sequence.source_pose import acquire_source_pose
from omniretargeting._sequence.source_shape import SourceShapeGuard
from omniretargeting.sequence_retargeting import prepare_sequence, sequence_config
from tests._contact_fixtures import _task


def test_source_reference_preserves_limb_direction_when_a_contact_conflicts():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><worldbody><body name="origin"><geom size=".01"/>'
        '<body><joint axis="0 0 1" range="-2 2"/><geom size=".01"/>'
        '<body name="tip" pos="1 0 0"><geom size=".01"/></body>'
        "</body></body></worldbody></mujoco>"
    )
    source = np.tile([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], (5, 1, 1))
    observed = _task(np.tile([0.0, 1.0, 0.0], (5, 1)), np.ones(5, bool))
    q, report = acquire_source_pose(
        model,
        np.full((5, 1), 0.6),
        source,
        ["a", "b"],
        {"a": "origin", "b": "tip"},
        source[:, 0],
        {"probe": observed},
        {"effectors": {"probe": {"robot_body": "tip"}}},
        config={"iterations": 10},
        framerate=30,
        root_orientation_targets=None,
        body_orientation_targets=(),
    )
    assert np.max(abs(q)) < 0.201
    assert report["final_residuals"]["sticking_max_m"] > 1.0
    np.testing.assert_allclose(np.diff(q, axis=0), 0, atol=1e-7)


def test_source_frame_that_would_worsen_does_not_strand_other_frames():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/><worldbody><body name="origin"><geom size=".01"/>'
        '<body><joint axis="0 0 1" range="-2 2"/><geom size=".01"/>'
        '<body name="tip" pos="1 0 0"><geom size=".01"/></body>'
        "</body></body></worldbody></mujoco>"
    )
    poses = np.array([[0.19], [1.1]])
    source = np.tile([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], (2, 1, 1))
    _, _, tasks, _ = prepare_sequence(
        model,
        poses,
        source,
        ["a", "b"],
        {"a": "origin", "b": "tip"},
        source[:, 0],
        {},
        {},
        sequence_config(),
    )
    guard = SourceShapeGuard(tasks)
    step = guard.feasible_step(poses, np.array([[0.3], [-0.4]]))
    # No angular allowance: a step away from the source direction is removed.
    assert step[0, 0] == 0
    assert step[1, 0] == -0.4
    candidate = tasks.coords.integrate(poses, step, 1)
    assert np.all(
        guard.priority_errors(candidate) <= guard.priority_errors(poses) + 1e-7
    )
