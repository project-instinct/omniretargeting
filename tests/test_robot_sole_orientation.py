"""A renamed terminal still follows its calibrated physical sole direction."""

from dataclasses import replace
import mujoco, numpy as np
from scipy.spatial.transform import Rotation
from omniretargeting.foot_frames import robot_sole_directions
from tests.test_limb_repair_scope import _floor_problem, XML
from omniretargeting.core import OmniRetargeter


def test_rotated_geometry_defines_axes_in_body_coordinates():
    rotation = Rotation.from_euler("z", [[0.7]]).as_quat(scalar_first=True)[0]
    quat = " ".join(map(str, rotation))
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody><body name="root"><freejoint/><geom type="sphere" size=".01"/>
      <body name="foot" pos="0 0 -.4"><joint/><geom type="box" pos=".03 0 -.05" quat="{quat}" size=".1 .03 .01"/></body>
      </body></worldbody></mujoco>''')
    forward, up = robot_sole_directions(model, model.body("foot").id)
    np.testing.assert_allclose(up, [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(forward, [np.cos(0.7), np.sin(0.7), 0], atol=1e-12)


def test_profile_enables_geometric_sole_axes_and_respects_explicit_disable():
    owner = OmniRetargeter.__new__(OmniRetargeter)
    owner.robot_model = mujoco.MjModel.from_xml_string(XML)
    owner.valid_joint_mapping = {"end_a": "tip1", "end_b": "tip2"}
    owner.retargeting_config = {
        "foot_stabilization": {"body_names": {"left": "tip1", "right": "tip2"}},
        "end_effector_priority": {
            "effectors": {
                "renamed_b": {"robot_body": "tip2", "track_orientation": False}
            }
        },
    }
    owner._infer_end_effector_geometry()
    specs = owner.retargeting_config["end_effector_priority"]["effectors"]
    assert specs["end_a"]["track_orientation"]
    assert not specs["renamed_b"]["track_orientation"]
    np.testing.assert_allclose(
        specs["end_a"]["orientation_local_up"], [0, 0, 1], atol=1e-12
    )


def test_sole_direction_contributes_to_source_pass_and_not_position_statistics():
    objective, poses, _, _ = _floor_problem(0.004)
    for spec in objective.tasks.effectors.values():
        spec["track_orientation"] = True
        spec["task"] = replace(
            spec["task"],
            forward_directions=np.tile([1.0, 0, 0], (len(poses), 1)),
            up_directions=np.tile([0.0, np.sin(0.3), np.cos(0.3)], (len(poses), 1)),
            contact_mask=np.zeros(len(poses), bool),
            swing_world_mask=np.ones(len(poses), bool),
        )
    objective._pack_effectors(None)
    first = objective.evaluate(poses)
    before = first[3]
    for effector in objective.effectors:
        effector["orientation"] = []
    second = objective.evaluate(poses)
    after = second[3]
    assert first[0].cost > second[0].cost + 0.1
    assert before["world_tracking_max_m"] == after["world_tracking_max_m"]
    assert before["free_tracking_max_m"] == after["free_tracking_max_m"]
