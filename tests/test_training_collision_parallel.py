"""Parallel saved-frame self-collision audit preserves serial aggregation."""

import mujoco
import numpy as np
import pytest

from omniretargeting.training_collision_parallel import query_self_frames
from scripts.review_training_collision import TrainingTrajectory, audit_self


ROBOT_XML = """<mujoco><compiler angle="radian"/><worldbody>
  <body name="left" pos="-0.12 0 0"><freejoint/>
    <geom name="left-collider" type="sphere" size="0.04" mass="1"/>
  </body>
  <body name="right" pos="0.12 0 0"><freejoint/>
    <geom name="right-collider" type="sphere" size="0.04" mass="1"/>
  </body>
</worldbody></mujoco>"""


def test_parallel_self_collision_matches_serial_report():
    model = mujoco.MjModel.from_xml_string(ROBOT_XML)
    poses = np.repeat(model.qpos0[None], 8, axis=0)
    poses[:, 2] = np.linspace(-0.03, 0.03, len(poses))
    trajectory = TrainingTrajectory(
        poses.copy(),
        np.arange(len(poses), dtype=float),
        np.arange(len(poses), dtype=int),
        {"status": "input_frames"},
    )
    geom_ids = np.flatnonzero((model.geom_contype | model.geom_conaffinity) != 0)

    serial, serial_query = audit_self(model, trajectory, geom_ids, workers=1)
    parallel, parallel_query = audit_self(model, trajectory, geom_ids, workers=2)

    assert serial_query.describe()["pair_count"] > 0
    assert parallel_query.describe()["pair_count"] > 0
    assert serial == parallel
    assert serial["per_pair"] == parallel["per_pair"]
    assert serial["per_body"] == parallel["per_body"]


def test_worker_count_is_validated():
    with pytest.raises(ValueError, match="positive integer"):
        query_self_frames(object(), np.zeros((0, 0)), workers=0)
