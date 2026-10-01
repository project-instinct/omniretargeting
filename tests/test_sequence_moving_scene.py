"""Moving object transforms participate in the sequence clearance level."""

import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting._sequence.coordinates import Coordinates
from omniretargeting._sequence.scene import PrescribedSceneClearance
from omniretargeting._sequence.tasks import SequenceTasks
from omniretargeting.geometry_query import GeometryQueryError
from omniretargeting.sequence_retargeting import retarget_sequence, sequence_config


def test_clearance_uses_each_recorded_object_pose_without_relocating_it():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
       <body name="body"><freejoint/><geom type="box" size=".1 .1 .1"/></body>
       </worldbody></mujoco>""")
    trajectory = np.tile(model.qpos0, (2, 1))
    translations = np.array([[2.0, 0.0, 0.0], [0.15, 0.0, 0.0]])
    before = translations.copy()
    moving = PrescribedSceneClearance(
        model,
        [
            {
                "mesh": trimesh.creation.box(extents=[0.2, 0.2, 0.2]),
                "translations": translations,
            }
        ],
        2,
        clearance=0.001,
    )
    tasks = SequenceTasks(
        model,
        Coordinates(model),
        trajectory,
        np.zeros((2, 1, 3)),
        ["body"],
        {"body": "body"},
        np.zeros((2, 3)),
        {},
        {},
        sequence_config({"self_collision_enabled": False}),
        extra_tasks=[moving],
    )
    clear = tasks.frame_rows(trajectory[0], 0)
    colliding = tasks.frame_rows(trajectory[1], 1)
    assert clear[2]["moving_depth_m"] == 0
    assert colliding[2]["moving_depth_m"] > 0.04
    rows = list(tasks.clearance_rows(frame=1))
    assert any(abs(jac[0, 0]) > 0.9 for _, jac in rows)
    np.testing.assert_array_equal(translations, before)


def test_uncertain_collision_trial_backtracks_but_initial_uncertainty_fails():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="tool"><joint type="slide" axis="1 0 0" range="-1 1"/>
      <geom size=".1"/></body></worldbody></mujoco>""")

    class ObservedRegion:
        def rows(self, pose, frame, dofs):
            if pose[0] > 0.0151:
                raise GeometryQueryError("Trial left reliably observed geometry")
            yield 1.0, np.array([pose[0] - 0.1]), np.ones((1, 1))

    def solve(initial):
        return retarget_sequence(
            model,
            np.array([[initial]]),
            np.array([[[0.1, 0, 0]]]),
            ["source"],
            {"source": "tool"},
            np.zeros((1, 3)),
            {},
            {},
            framerate=30.0,
            config={"iterations": 1, "self_collision_enabled": False},
            extra_tasks=[ObservedRegion()],
        )

    result, report = solve(0.0)
    assert 0 < result[0, 0] <= 0.0151
    assert 0 < report["history"][0]["alpha"] < 1
    with pytest.raises(GeometryQueryError):
        solve(0.05)
