"""Secondary objectives cannot detach an acquired endpoint contact."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting.contact import build_world_tracking_tasks
from omniretargeting.sequence_retargeting import retarget_sequence


@pytest.mark.parametrize("mode", ["sticking", "sliding"])
def test_acquired_contact_survives_conflicting_secondary_objective(mode):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="tool"><joint type="slide" '
        'axis="1 0 0" range="-1 1"/><geom size=".02" mass="1"/>'
        "</body></worldbody></mujoco>"
    )
    frames = 5
    points = np.zeros((frames, 3))
    task = build_world_tracking_tasks({"tool": {"tracking_positions": points}})["tool"]
    task = replace(
        task,
        contact_mask=np.ones(frames, bool),
        contact_modes=np.full(frames, mode),
        surface_normals=np.tile([0.0, 0.0, 1.0], (frames, 1)),
    )

    class Secondary:
        def rows(self, pose, frame, dofs):
            yield 1e10, np.array([pose[0] - 0.2]), np.ones((1, len(dofs)))

    arguments = {
        "model": model,
        "initial_trajectory": np.zeros((frames, 1)),
        "source_positions": points[:, None],
        "source_names": ["source"],
        "joint_mapping": {"source": "tool"},
        "source_root_positions": points,
        "tasks": {"tool": task},
        "end_effector_config": {"effectors": {"tool": {"robot_body": "tool"}}},
        "config": {"self_collision_enabled": False, "iterations": 4},
        "extra_tasks": (Secondary(),),
        "framerate": 30.0,
    }
    unprotected, _ = retarget_sequence(**arguments)
    reference = np.zeros((frames, 1))
    protected, _ = retarget_sequence(
        **arguments, endpoint_reference_trajectory=reference
    )
    assert np.max(unprotected) > 0.1
    assert np.max(np.abs(protected)) <= 0.001001
    assert not np.any(reference)


def test_all_root_coordinates_are_continuous_during_contact_acquisition():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="root"><freejoint/>'
        '<geom size=".03" mass="1"/><body pos="0 0 .1">'
        '<joint/><geom size=".02" mass="1"/></body></body></worldbody></mujoco>'
    )
    frames = 15
    source = np.zeros((frames, 3))
    source[:, 0] = np.linspace(0, 0.5, frames)
    source[:, 2] = 1.0
    seed = np.tile(model.qpos0, (frames, 1))
    seed[:, :3] = source
    alternating = (-1.0) ** np.arange(frames)
    seed[:, :3] += 0.02 * alternating[:, None]
    rotations = np.zeros((frames, 3))
    rotations[:, 0] = 0.04 * alternating
    seed[:, 3:7] = Rotation.from_rotvec(rotations).as_quat(scalar_first=True)
    phases = []
    retarget_sequence(
        model,
        seed,
        source[:, None],
        ["source"],
        {"source": "root"},
        source,
        {},
        {},
        framerate=30.0,
        root_orientation_targets=np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1)),
        config={"iterations": 3, "self_collision_enabled": False},
        callback=lambda poses, report: phases.append((poses, report["phase"])),
    )
    poses = [q for q, phase in phases if phase == "pose"][-1]
    assert np.linalg.norm(
        np.diff(poses[:, :3] - source, n=2, axis=0)
    ) < 0.1 * np.linalg.norm(np.diff(seed[:, :3] - source, n=2, axis=0))
    angles = Rotation.from_quat(poses[:, 3:7], scalar_first=True).as_rotvec()
    assert np.linalg.norm(np.diff(angles, n=2, axis=0)) < 0.1 * np.linalg.norm(
        np.diff(rotations, n=2, axis=0)
    )


def test_secondary_geometry_cannot_add_joint_chatter_to_acquired_motion():
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="tool"><joint type="slide" '
        'axis="1 0 0" range="-1 1"/><geom size=".02" mass="1"/>'
        "</body></worldbody></mujoco>"
    )
    frames = 9
    points = np.zeros((frames, 3))
    reference = np.zeros((frames, 1))

    class Secondary:
        def rows(self, pose, frame, dofs):
            yield 1e8, np.array([pose[0] - 0.1 * (-1.0) ** frame]), np.ones((1, 1))

    result, _ = retarget_sequence(
        model,
        reference,
        points[:, None],
        ["source"],
        {"source": "tool"},
        points,
        {},
        {},
        framerate=60.0,
        config={"iterations": 3, "self_collision_enabled": False},
        extra_tasks=(Secondary(),),
        endpoint_reference_trajectory=reference,
    )
    assert np.max(np.abs(np.diff(result[:, 0], n=2))) <= 2e-7
    assert np.max(np.abs(result)) > 0.001  # A smooth useful change is still possible.


def test_secondary_geometry_change_is_reported_against_the_fixed_reference():
    model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
    <worldbody><body name="a"><joint axis="0 0 1" range="-2 2"/><geom size=".02"/>
    <body name="b" pos=".1 0 0"><joint axis="0 0 1" range="-2 2"/><geom size=".02"/>
    <body name="c" pos=".1 0 0"><joint axis="0 0 1" range="-2 2"/><geom size=".02"/>
    </body></body></body></worldbody></mujoco>""")
    frames = 5
    reference = np.zeros((frames, 3))
    source = np.tile(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], (frames, 1, 1)
    )

    class ConflictingGeometry:
        def rows(self, pose, frame, dofs):
            yield 1e8, pose - 0.8, np.eye(3)

    result, report = retarget_sequence(
        model,
        reference,
        source,
        ["a", "b", "c"],
        {n: n for n in ["a", "b", "c"]},
        source[:, 0],
        {},
        {},
        framerate=30.0,
        config={"iterations": 8, "self_collision_enabled": False},
        extra_tasks=(ConflictingGeometry(),),
        endpoint_reference_trajectory=reference,
    )
    # Posture has no angular bound: the change is reported, not capped.
    assert np.max(abs(result)) > 0.6
    assert report["posture_envelope"]["angular_bound"] is None
    assert report["posture_envelope"]["maximum_joint_change_rad"] == pytest.approx(np.max(abs(result)))
    # A subsequent optimization reports against the original reference, even
    # though its initial pose has already moved: passes cannot rebase it.
    again, report = retarget_sequence(
        model,
        result,
        source,
        ["a", "b", "c"],
        {n: n for n in ["a", "b", "c"]},
        source[:, 0],
        {},
        {},
        framerate=30.0,
        config={"iterations": 8, "self_collision_enabled": False},
        extra_tasks=(ConflictingGeometry(),),
        correction_reference_trajectory=reference,
    )
    assert report["posture_envelope"]["maximum_joint_change_rad"] == pytest.approx(np.max(abs(again)))
    assert report["posture_envelope"]["within_translation_bound"]
