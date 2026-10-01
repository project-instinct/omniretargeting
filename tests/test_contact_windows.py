"""Physical material-point continuity and globally consistent window targets."""

from dataclasses import replace

import mujoco
import numpy as np

from omniretargeting._sequence.coordinates import Coordinates
from omniretargeting._sequence.surface_guard import SurfaceSequenceGuard
from omniretargeting._sequence.tasks import SequenceTasks
from omniretargeting.contact import build_world_tracking_tasks
from omniretargeting.contact_windows import slice_contact_task
from omniretargeting.sequence_retargeting import retarget_sequence, sequence_config


def test_window_preserves_approach_even_when_contact_is_outside_crop():
    path = np.c_[np.linspace(1, 0, 31), np.zeros((31, 2))]
    task = build_world_tracking_tasks({"hand": {"tracking_positions": path}})["hand"]
    task = replace(
        task,
        contact_mask=np.arange(31) >= 25,
        contact_positions=np.tile([0.0, 0.1, 0.0], (31, 1)),
    )
    cropped = slice_contact_task(task, 10, 20)
    assert not cropped.contact_mask.any()
    assert (cropped.approach[0] > 0).all()
    for f in range(10):
        assert (
            cropped.frame(f)["approach_weight"] == task.frame(f + 10)["approach_weight"]
        )
        np.testing.assert_array_equal(
            cropped.frame(f)["approach_position"],
            task.frame(f + 10)["approach_position"],
        )
    nested = slice_contact_task(cropped, 2, 6)
    np.testing.assert_array_equal(nested.approach[0], task.approach[0][12:16])


def test_contact_handover_tracks_the_same_robot_surface_point():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="root"><freejoint/><geom size=".02"/>
        <body name="hand" pos=".3 0 0"><geom size=".02"/></body>
      </body></worldbody></mujoco>""")
    source = np.tile([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]], (3, 1, 1))
    task = build_world_tracking_tasks({"hand": {"tracking_positions": source[:, 1]}})[
        "hand"
    ]
    task = replace(
        task,
        contact_mask=np.array([False, False, True]),
        contact_modes=np.array(["none", "none", "sticking"]),
        contact_positions=np.tile([0.4, 0, 0], (3, 1)),
        approach_reference=(np.array([0.0, 0.999, 0.0]), np.tile([0.4, 0, 0], (3, 1))),
    )
    priority = {
        "effectors": {
            "hand": {
                "robot_body": "hand",
                "preserve_world_target": True,
                "tracking_offset": [0, 0, 0],
                "contact_offsets": [[0.1, 0, 0]],
            }
        }
    }
    poses = np.tile(model.qpos0, (3, 1))
    tasks = SequenceTasks(
        model,
        Coordinates(model),
        poses,
        source,
        ["root", "hand"],
        {"root": "root", "hand": "hand"},
        source[:, 0],
        {"hand": task},
        priority,
        sequence_config({"self_collision_enabled": False}),
    )
    # Without blending the robot-side material point the penultimate residual
    # is -9.99 cm although the pose and the observed contact are stationary.
    for f in range(3):
        np.testing.assert_allclose(
            tasks.frame_rows(poses[f], f)[0][-3:], 0.0, atol=1e-10
        )
    guard = SurfaceSequenceGuard(tasks, poses, 0.001)
    for f in range(3):
        np.testing.assert_allclose(
            next(iter(guard.rows(poses[f], f)))[1], 0.0, atol=1e-10
        )


def test_sliding_follows_recorded_positions_on_a_finite_contact_patch():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
       <body name="point"><freejoint/><geom size=".01"/></body>
       </worldbody></mujoco>""")
    targets = np.c_[np.linspace(0.1, 0.2, 15), np.zeros((15, 2))]
    task = build_world_tracking_tasks({"touch": {"tracking_positions": targets}})[
        "touch"
    ]
    task = replace(
        task,
        contact_mask=np.ones(15, bool),
        contact_modes=np.full(15, "sliding"),
        surface_normals=np.tile([0.0, 0.0, 1.0], (15, 1)),
    )
    source = np.zeros((15, 1, 3))
    poses, _ = retarget_sequence(
        model,
        np.tile(model.qpos0, (15, 1)),
        source,
        ["point"],
        {"point": "point"},
        source[:, 0],
        {"touch": task},
        {"effectors": {"touch": {"robot_body": "point"}}},
        framerate=30,
        config={
            "iterations": 15,
            "self_collision_enabled": False,
            "lock_root_orientation": True,
        },
    )
    np.testing.assert_allclose(poses[:, :3], targets, atol=0.001)
