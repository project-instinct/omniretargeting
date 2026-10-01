"""Calibrated rotations disambiguate IK without altering registered source roots."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from omniretargeting._sequence.orientation_seed import source_orientation_seed


def test_calibrated_branch_and_mapped_anchor_follow_source_across_quaternion_seam():
    model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/><worldbody>
      <body name="floating"><freejoint/><geom size=".03"/>
        <body name="anchor" pos="0 0 -.2"><joint name="bend" axis="0 1 0" range="-1 1"/><geom size=".02"/>
          <body name="terminal" pos="0 0 -.3"><joint name="turn" axis="1 0 0" range="-1 1"/><geom size=".02"/>
          </body></body></body></worldbody></mujoco>""")
    frames = 15
    initial = np.tile(model.qpos0, (frames, 1))
    original = initial.copy()
    roots = np.column_stack(
        [np.linspace(0, 0.1, frames), np.zeros(frames), np.ones(frames)]
    )
    saved_roots = roots.copy()
    bend, turn = np.linspace(-0.3, 0.4, frames), np.linspace(0.2, -0.2, frames)
    world = Rotation.from_euler("z", np.linspace(3.0, 3.4, frames)[:, None]).as_matrix()
    specifications = [
        {"robot_body": "floating", "rotations": world, "sigma": 1.0},
        {
            "robot_body": "anchor",
            "parent_body": "floating",
            "rotations": Rotation.from_euler("y", bend[:, None]).as_matrix(),
            "sigma": 1.0,
        },
        {
            "robot_body": "terminal",
            "parent_body": "anchor",
            "rotations": Rotation.from_euler("x", turn[:, None]).as_matrix(),
            "sigma": 1.0,
        },
    ]
    result = source_orientation_seed(
        model, initial, roots, {"source_root_robot_body": "anchor"}, specifications, {}
    )
    np.testing.assert_allclose(result[:, 7], bend, atol=1e-6)
    np.testing.assert_allclose(result[:, 8], turn, atol=1e-6)
    data = mujoco.MjData(model)
    for frame, pose in enumerate(result):
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(
            data.xpos[model.body("anchor").id], roots[frame], atol=1e-12
        )
        np.testing.assert_allclose(
            data.xmat[model.body("floating").id].reshape(3, 3), world[frame], atol=1e-12
        )
    np.testing.assert_array_equal(initial, original)
    np.testing.assert_array_equal(roots, saved_roots)
    locked = source_orientation_seed(
        model,
        initial,
        roots,
        {},
        specifications,
        {"lock_root_xy": True, "lock_root_z": True, "lock_root_orientation": True},
    )
    np.testing.assert_array_equal(locked[:, :7], initial[:, :7])
