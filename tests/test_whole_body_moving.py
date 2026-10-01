"""Small real-MuJoCo regressions for the moving-scene whole-body stage."""

from __future__ import annotations

import mujoco
import numpy as np
import trimesh

from omniretargeting.contact import SceneContactTask
from omniretargeting.moving_scene import MovingSceneQuery
from omniretargeting.scene_geometry import SceneQuery
from omniretargeting.whole_body_correction import correct_whole_body_trajectory

MOVING_XML = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="root">
      <freejoint name="floating_base"/>
      <body name="slider">
        <joint name="slide" type="slide" axis="0 0 1" range="-.2 .2"/>
        <geom name="probe" type="box" size=".02 .02 .02"
              contype="1" conaffinity="1"/>
      </body>
      <body name="hand" pos="0 0 1">
        <geom name="hand_marker" type="sphere" size=".02"
              contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

ROOT_COLLISION_XML = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="root">
      <freejoint name="floating_base"/>
      <geom name="root_probe" type="box" size=".02 .02 .02"
            contype="1" conaffinity="1"/>
      <body name="slider" pos="0 0 1">
        <joint name="slide" type="slide" axis="0 0 1" range="-.2 .2"/>
        <geom name="probe" type="box" size=".02 .02 .02"
              contype="1" conaffinity="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _model_and_trajectory(xml=MOVING_XML):
    model = mujoco.MjModel.from_xml_string(xml)
    return model, np.tile(model.qpos0, (2, 1))


def _moving_box(*, static=False):
    return {
        "name": "moving_box",
        "mesh": trimesh.creation.box(extents=[.18, .2, .06]),
        "translations": np.asarray(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            if static else [[-.15, 0.0, 0.0], [.15, 0.0, 0.0]],
            dtype=float,
        ),
        "rotations": np.tile(np.eye(3), (2, 1, 1)),
        "scales": np.ones(2),
    }


def _far_scene():
    transform = np.eye(4)
    transform[:3, 3] = [10.0, 10.0, 10.0]
    return SceneQuery(trimesh.creation.box(extents=[.1, .1, .1], transform=transform))


def _config(**overrides):
    config = {
        "enabled": True,
        "iterations": 3,
        "max_joint_correction": .2,
        "trust_region": .06,
        "correction_velocity": 5.0,
        "correction_acceleration": 72.0,
        "correction_jerk": 240.0,
        "motion_spacing": .015,
        "max_substeps": 64,
        "moving_collision_enabled": True,
        "collision_weight": 20.0,
    }
    config.update(overrides)
    return config


def _hand_task(model, baseline):
    data = mujoco.MjData(model)
    data.qpos[:] = baseline[0]
    mujoco.mj_forward(model, data)
    hand = data.xpos[model.body("hand").id].copy()
    targets = np.tile(hand, (len(baseline), 1))
    return SceneContactTask(
        name="left_hand",
        tracking_positions=targets.copy(),
        contact_positions=targets.copy(),
        surface_points=targets.copy(),
        surface_normals=np.tile([0.0, 0.0, 1.0], (len(baseline), 1)),
        distances=np.zeros(len(baseline)),
        relative_speeds=np.zeros(len(baseline)),
        contact_confidence=np.ones(len(baseline)),
        contact_mask=np.ones(len(baseline), dtype=bool),
        contact_modes=np.full(len(baseline), "sticking"),
        transition_weights=np.ones(len(baseline)),
        environment_ids=np.zeros(len(baseline), dtype=int),
    )


def _priority():
    return {
        "effectors": {
            "left_hand": {
                "robot_body": "hand",
                "tracking_offset": [0.0, 0.0, 0.0],
                "contact_offsets": [[0.0, 0.0, 0.0]],
            }
        }
    }


def test_interframe_moving_box_is_repaired_with_endpoints_collision_free():
    model, baseline = _model_and_trajectory()
    obj = _moving_box()
    query = MovingSceneQuery(model, [obj], len(baseline))
    dof = int(model.joint("slide").dofadr[0])
    endpoint_depths = [query.query(pose, frame)["forward_depth"].max() for frame, pose in enumerate(baseline)]
    assert endpoint_depths == [0.0, 0.0]
    midpoint = query.query(baseline[0], .5, [dof])
    assert midpoint["forward_depth"].max() > .01
    assert any(row["geom_id"] == model.geom("probe").id for row in midpoint["rows"])

    result, report = correct_whole_body_trajectory(
        model,
        baseline,
        {},
        {},
        _config(),
        scene_query=_far_scene(),
        scene_objects=[obj],
    )

    assert report["status"] == "accepted_updates"
    assert report["accepted_update_count"] > 0
    assert report["root_invariant"] is True
    assert report["max_joint_correction"] > 1e-4
    assert report["moving_collision"]["before"]["interframe_forward_max_depth"] > .01
    assert report["moving_collision"]["after"]["interframe_forward_max_depth"] < (
        report["moving_collision"]["before"]["interframe_forward_max_depth"] - 1e-4
    )
    assert np.max(np.abs(result[:, int(model.joint("slide").qposadr[0])] - baseline[:, int(model.joint("slide").qposadr[0])])) > 1e-4


def test_moving_correction_keeps_floating_base_and_primary_marker_budget():
    model, baseline = _model_and_trajectory()
    task = _hand_task(model, baseline)
    result, report = correct_whole_body_trajectory(
        model,
        baseline,
        {"left_hand": task},
        _priority(),
        _config(),
        scene_query=_far_scene(),
        scene_objects=[_moving_box()],
    )

    assert report["root_invariant"] is True
    np.testing.assert_array_equal(result[:, :7], baseline[:, :7])
    data = mujoco.MjData(model)
    for frame in range(len(result)):
        data.qpos[:] = result[frame]
        mujoco.mj_forward(model, data)
        point = data.xpos[model.body("hand").id]
        np.testing.assert_allclose(point, task.contact_positions[frame], atol=1e-5)
    assert report["final_fidelity"]["ok"]


def test_rooted_collision_is_reported_as_immovable_and_trajectory_stays_fixed():
    model, baseline = _model_and_trajectory(ROOT_COLLISION_XML)
    result, report = correct_whole_body_trajectory(
        model,
        baseline,
        {},
        {},
        _config(),
        scene_query=_far_scene(),
        scene_objects=[_moving_box(static=True)],
    )

    np.testing.assert_array_equal(result, baseline)
    assert report["root_invariant"] is True
    assert report["max_joint_correction"] == 0.0
    assert report["moving_collision"]["immovable_witnesses"]
    assert any(
        witness["body"] == "root"
        for witness in report["moving_collision"]["immovable_witnesses"]
    )


def test_windowed_moving_correction_checks_composed_intervals(monkeypatch):
    import omniretargeting.whole_body_correction as correction

    monkeypatch.setattr(correction, "WHOLE_BODY_WINDOW_THRESHOLD", 4)
    monkeypatch.setattr(correction, "WHOLE_BODY_WINDOW_FRAMES", 4)
    model, short = _model_and_trajectory()
    baseline = np.tile(short[0], (8, 1))
    obj = _moving_box()
    obj["translations"] = np.c_[np.linspace(-.15, .15, 8), np.zeros((8, 2))]
    obj["rotations"] = np.tile(np.eye(3), (8, 1, 1))
    obj["scales"] = np.ones(8)
    task = _hand_task(model, baseline)
    result, report = correct_whole_body_trajectory(
        model, baseline, {"left_hand": task}, _priority(), _config(iterations=1),
        scene_query=_far_scene(), scene_objects=[obj],
    )
    assert report["window_frames"] == 4
    assert report["final_reasons"] == []
    assert report["root_invariant"]
    np.testing.assert_array_equal(result[:, :7], baseline[:, :7])
    assert report["final_fidelity"]["ok"]
    assert report["moving_collision"]["after"]["complete"]
