"""Small MuJoCo models and contact tasks shared by contact regression tests."""

import numpy as np
import pytest

from omniretargeting.contact import (
    SceneContactTask,
)

ANKLE_ARM_XML = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="root">
      <freejoint/>
      <geom type="sphere" size="0.02" mass="1"/>
      <body name="foot">
        <joint name="ankle" type="hinge" axis="0 1 0" range="-0.6 0.6"/>
        <geom type="capsule" fromto="0 0 0 1 0 0" size="0.02"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _straight_arm_setup(num_frames=12, amplitude=0.12):
    """Return a model, a chattering trajectory and a fixed endpoint task."""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(ANKLE_ARM_XML)
    trajectory = np.tile(model.qpos0, (num_frames, 1))
    trajectory[:, 3] = 1.0
    trajectory[:, 7] = amplitude * (-1.0) ** np.arange(num_frames)
    targets = np.tile([1.0, 0.0, 0.0], (num_frames, 1))
    return model, trajectory, targets


def _task(
    targets,
    contact_mask=None,
    contact_modes=None,
    surface_normals=None,
    transition_weights=None,
):
    num_frames = len(targets)
    if contact_mask is None:
        contact_mask = np.zeros(num_frames, dtype=bool)
    if surface_normals is None:
        surface_normals = np.tile([0.0, 0.0, 1.0], (num_frames, 1))
    return SceneContactTask(
        name="left_foot",
        tracking_positions=targets,
        contact_positions=targets,
        surface_points=targets,
        surface_normals=np.asarray(surface_normals, dtype=float),
        distances=np.zeros(num_frames),
        relative_speeds=np.zeros(num_frames),
        contact_confidence=np.ones(num_frames),
        contact_mask=np.asarray(contact_mask, dtype=bool),
        forward_directions=np.tile([1.0, 0.0, 0.0], (num_frames, 1)),
        up_directions=np.tile([0.0, 0.0, 1.0], (num_frames, 1)),
        contact_modes=(
            None
            if contact_modes is None
            else np.asarray(contact_modes)
        ),
        transition_weights=(
            None
            if transition_weights is None
            else np.asarray(transition_weights, dtype=float)
        ),
    )


def _effector_config(contact_offsets=None, track_orientation=True):
    return {
        "effectors": {
            "left_foot": {
                "robot_body": "foot",
                "tracking_offset": [1.0, 0.0, 0.0],
                "contact_offsets": contact_offsets or [[1.0, 0.0, 0.0]],
                "track_orientation": track_orientation,
                "orientation_sigma": 0.1,
            }
        }
    }

