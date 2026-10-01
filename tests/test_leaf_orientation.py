"""Leaf segments no other task observes follow the source orientation closely."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from omniretargeting._core.orientation_targets import (
    UNOBSERVED_LEAF_SIGMA_SCALE,
    sequence_body_orientation_targets,
    unobserved_leaf_bodies,
)

XML = """
<mujoco><worldbody>
  <body name="torso"><freejoint/><geom size=".1"/>
    <body name="upper" pos="0 .2 0"><joint axis="1 0 0"/><geom size=".03"/>
      <body name="lower" pos="0 .3 0"><joint axis="0 0 1"/><geom size=".03"/>
        <body name="hand" pos="0 .3 0"><joint axis="0 1 0"/><geom size=".03"/>
          <body name="palm" pos="0 .05 0"><geom size=".02"/></body>
        </body>
      </body>
    </body>
    <body name="thigh" pos="0 -.1 -.2"><joint axis="0 1 0"/><geom size=".03"/>
      <body name="ankle" pos="0 0 -.4"><joint axis="0 1 0"/><geom size=".03"/>
        <body name="foot" pos="0 0 -.05"><joint axis="1 0 0"/><geom size=".03"/></body>
      </body>
    </body>
  </body>
</worldbody></mujoco>
"""
MAPPING = {"Chest": "torso", "L_Shoulder": "upper", "L_Elbow": "lower", "L_Wrist": "hand",
           "L_Hip": "thigh", "L_Ankle": "ankle"}
EFFECTORS = {"hand": {"robot_body": "palm"},
             "foot": {"robot_body": "foot", "track_orientation": True}}


def _ids(model, names):
    return {int(model.body(name).id) for name in names}


def test_only_unobserved_terminal_chains_are_leaves():
    model = mujoco.MjModel.from_xml_string(XML)
    mapped = [model.body(name).id for name in MAPPING.values()]
    leaves = unobserved_leaf_bodies(model, mapped, EFFECTORS)
    # The hand chain has no mapped body below it and nothing tracks its
    # orientation. The foot's sole frame observes the whole ankle chain, even
    # though it sits on an unmapped body below the mapped ankle.
    assert leaves == _ids(model, ["hand", "palm"])
    tracked = unobserved_leaf_bodies(model, mapped, {**EFFECTORS, "hand": {"robot_body": "palm", "track_orientation": True}})
    assert tracked == set()
    untracked = unobserved_leaf_bodies(model, mapped, {"hand": {"robot_body": "palm"}})
    assert untracked == _ids(model, ["hand", "palm", "ankle", "foot"])


def test_leaf_orientation_sigma_is_tightened_and_others_are_unchanged():
    model = mujoco.MjModel.from_xml_string(XML)
    frames = 3
    tasks = {name: {"body_id": int(model.body(body).id), "body_name": body, "sigma": 0.2, "weight": 1.0}
             for name, body in MAPPING.items() if name != "Chest"}
    identity = np.eye(3)
    retargeter = SimpleNamespace(
        body_orientation_enabled=True, robot_model=model, joint_mapping=MAPPING,
        _source_orientation_indices={name: index for index, name in enumerate(tasks)},
        _source_orientation_references={name: identity for name in tasks},
        _robot_orientation_references={name: identity for name in tasks},
        end_effector_config={"effectors": EFFECTORS}, body_orientation_tasks=tasks,
    )
    source = np.tile([1.0, 0.0, 0.0, 0.0], (frames, len(tasks), 1))
    specs = {spec["robot_body"]: spec for spec in sequence_body_orientation_targets(retargeter, source, None)}
    assert specs["hand"]["sigma"] == pytest.approx(0.2 * UNOBSERVED_LEAF_SIGMA_SCALE)
    for body in ("upper", "lower", "thigh", "ankle"):
        assert specs[body]["sigma"] == pytest.approx(0.2)
