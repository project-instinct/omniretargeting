"""A declared bind pose must replace the positional calibration fit.

A positional fit cannot observe rotation about a bone's own axis, so it leaves
that twist wherever the optimizer stopped, independently per side. The declared
pose removes that freedom: the robot reference frames become a pure function of
the profile, not of the reference positions the fit was given.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from omniretargeting.retargeting import GenericInteractionRetargeter

REPO_ROOT = Path(__file__).resolve().parents[1]

ARMS_XML = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="torso" pos="0 0 1">
      <freejoint/>
      <geom type="box" size=".1 .1 .1"/>
      <body name="left_upper" pos="0 .2 0">
        <joint name="left_shoulder_pitch" axis="0 1 0" range="-3 3"/>
        <joint name="left_shoulder_roll" axis="1 0 0" range="-3 3"/>
        <geom type="capsule" fromto="0 0 0 0 .2 0" size=".02"/>
        <body name="left_lower" pos="0 .2 0">
          <joint name="left_elbow" axis="0 0 1" range="-1.5 1.5"/>
          <geom type="capsule" fromto="0 0 0 0 .2 0" size=".02"/>
        </body>
      </body>
      <body name="right_upper" pos="0 -.2 0">
        <joint name="right_shoulder_pitch" axis="0 1 0" range="-3 3"/>
        <joint name="right_shoulder_roll" axis="1 0 0" range="-3 3"/>
        <geom type="capsule" fromto="0 0 0 0 -.2 0" size=".02"/>
        <body name="right_lower" pos="0 -.2 0">
          <joint name="right_elbow" axis="0 0 1" range="-1.5 1.5"/>
          <geom type="capsule" fromto="0 0 0 0 -.2 0" size=".02"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

SOURCE_NAMES = ["Chest", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow"]
MAPPING = {
    "Chest": "torso",
    "L_Shoulder": "left_upper",
    "R_Shoulder": "right_upper",
    "L_Elbow": "left_lower",
    "R_Elbow": "right_lower",
}
DECLARED = {
    "left_shoulder_pitch": 0.3,
    "left_shoulder_roll": 0.7,
    "left_elbow": 0.4,
    "right_shoulder_pitch": 0.3,
    "right_shoulder_roll": -0.7,
    "right_elbow": -0.4,
}


def build(reference_pose):
    model = mujoco.MjModel.from_xml_string(ARMS_XML)
    body_orientation = {
        "_source_orientations_available": True,
        "frame": "parent_relative",
    }
    if reference_pose is not None:
        body_orientation["reference_pose"] = reference_pose
    return GenericInteractionRetargeter(
        model,
        mujoco.MjData(model),
        trimesh.creation.box(),
        MAPPING,
        1.0,
        valid_joint_names=list(MAPPING),
        terrain_sample_seed=0,
        end_effector_config={
            "enabled": True,
            "effectors": {"left_hand": {"robot_body": "left_lower"}},
            "body_orientation": body_orientation,
            "_base_orientation_source_joint": "Chest",
        },
    )


def positions(spread):
    """Two plainly different reference skeletons for the same semantic pose."""
    return np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, spread, 1.0],
            [0.0, -spread, 1.0],
            [0.0, 2 * spread, 1.0],
            [0.0, -2 * spread, 1.0],
        ]
    )


def calibrate(retargeter, spread):
    retargeter.configure_source_orientation_reference(
        SOURCE_NAMES,
        np.tile([1.0, 0.0, 0.0, 0.0], (len(SOURCE_NAMES), 1)),
        positions(spread),
    )
    return {
        name: value.copy()
        for name, value in retargeter._robot_orientation_references.items()
    }


def test_declared_pose_is_applied_exactly_and_ignores_the_positional_fit():
    retargeter = build(DECLARED)
    references = calibrate(retargeter, 0.2)

    model = retargeter.robot_model
    for joint_name, angle in DECLARED.items():
        address = int(model.jnt_qposadr[model.joint(joint_name).id])
        assert retargeter.robot_reference_qpos[address] == pytest.approx(angle)

    data = mujoco.MjData(model)
    data.qpos[:] = retargeter.robot_reference_qpos
    mujoco.mj_forward(model, data)
    for source_name, body_name in MAPPING.items():
        expected = data.xmat[model.body(body_name).id].reshape(3, 3)
        np.testing.assert_allclose(references[source_name], expected, atol=1e-12)

    # A different reference skeleton must not move a declared robot frame.
    widened = build(DECLARED)
    for name, value in calibrate(widened, 0.35).items():
        np.testing.assert_allclose(references[name], value, atol=1e-12)
    np.testing.assert_allclose(
        retargeter.robot_reference_qpos, widened.robot_reference_qpos, atol=0
    )


def test_without_a_declaration_the_solved_fit_still_runs():
    retargeter = build(None)
    solved = calibrate(retargeter, 0.2)
    assert set(solved) == set(MAPPING)

    model = retargeter.robot_model
    data = mujoco.MjData(model)
    data.qpos[:] = retargeter.robot_reference_qpos
    mujoco.mj_forward(model, data)
    for source_name, body_name in MAPPING.items():
        expected = data.xmat[model.body(body_name).id].reshape(3, 3)
        np.testing.assert_allclose(solved[source_name], expected, atol=1e-12)

    # The fit moved the robot, and the declaration is a different mechanism.
    assert np.abs(retargeter.robot_reference_qpos - model.qpos0).max() > 1e-6
    declared = calibrate(build(DECLARED), 0.2)
    assert max(
        Rotation.from_matrix(solved[name] @ declared[name].T).magnitude()
        for name in solved
    ) > 1e-3


def test_mirrored_declaration_gives_mirrored_parent_relative_references():
    retargeter = build(DECLARED)
    references = calibrate(retargeter, 0.2)
    tasks = retargeter.body_orientation_tasks
    mirror = np.diag([1.0, -1.0, 1.0])
    for left, right in (("L_Shoulder", "R_Shoulder"), ("L_Elbow", "R_Elbow")):
        left_relative = (
            references[tasks[left]["parent_source"]].T @ references[left]
        )
        right_relative = (
            references[tasks[right]["parent_source"]].T @ references[right]
        )
        deviation = Rotation.from_matrix(
            left_relative @ (mirror @ right_relative @ mirror).T
        ).magnitude()
        assert deviation == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "reference_pose",
    [
        {"no_such_joint": 0.0},
        {"left_elbow": 99.0},
        {"left_elbow": float("nan")},
        {"root": 0.0},
    ],
)
def test_an_unusable_declaration_is_reported_instead_of_silently_ignored(reference_pose):
    with pytest.raises(ValueError):
        calibrate(build(reference_pose), 0.2)


def test_bundled_g1_profile_declares_a_left_right_mirrored_bind_pose():
    profile = json.loads(
        (
            REPO_ROOT / "robot_models" / "unitree_g1" / "unitree_g1_instinctmj.json"
        ).read_text()
    )
    pose = profile["retargeting"]["end_effector_priority"]["body_orientation"][
        "reference_pose"
    ]
    # Roll and yaw flip across the sagittal plane; pitch and the hinges do not.
    for name, angle in pose.items():
        if not name.startswith("left_"):
            continue
        opposite = pose["right_" + name[len("left_"):]]
        flips = name.endswith(("_roll_joint", "_yaw_joint"))
        assert opposite == pytest.approx(-angle if flips else angle), name
