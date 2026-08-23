import json

import numpy as np
import pytest

from omniretargeting.contact import ContactMode, SceneContactTask
from omniretargeting.evaluation import (
    distribution,
    evaluate_retargeted_motion,
    metrics_to_json,
)


ROBOT_XML = """
<mujoco>
  <worldbody>
    <body name="root">
      <freejoint/>
      <geom type="sphere" size="0.02" mass="1"/>
      <body name="foot">
        <joint name="foot_joint" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom name="foot_geom" type="sphere" size="0.03"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


ARM_XML = """
<mujoco>
  <worldbody>
    <body name="root">
      <freejoint/>
      <geom type="sphere" size="0.02" mass="1"/>
      <body name="shoulder">
        <joint name="shoulder_joint" axis="0 0 1"/>
        <geom type="sphere" size="0.02" mass="0.1"/>
        <body name="elbow" pos="1 0 0">
          <joint name="elbow_joint" axis="0 0 1"/>
          <geom type="sphere" size="0.02" mass="0.1"/>
          <body name="wrist" pos="1 0 0">
            <geom type="sphere" size="0.02"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _marker_positions(model, trajectory):
    mujoco = pytest.importorskip("mujoco")
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "foot"
    )
    result = []
    for qpos in trajectory:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        result.append(data.xpos[body_id].copy())
    return np.asarray(result)


def test_distribution_reports_metric_percentiles():
    result = distribution([0.0, 1.0, 2.0, 3.0])
    assert result["mean"] == pytest.approx(1.5)
    assert result["p95"] == pytest.approx(2.85)
    assert result["max"] == pytest.approx(3.0)
    assert result["count"] == 4


def test_final_evaluation_separates_tracking_contact_sliding_and_deviation():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(ROBOT_XML)
    trajectory = np.zeros((5, model.nq), dtype=float)
    trajectory[:, 3] = 1.0
    trajectory[:, 2] = 0.5
    trajectory[:, 0] = [0.0, 0.0, 0.01, 0.02, 0.02]
    tracking = _marker_positions(model, trajectory)

    modes = np.asarray(
        [
            ContactMode.NONE,
            ContactMode.SUPPORT,
            ContactMode.SUPPORT,
            ContactMode.SUPPORT,
            ContactMode.NONE,
        ],
        dtype=np.uint8,
    )
    contact_positions = tracking.copy()
    contact_positions[1:4] = tracking[1]
    task = SceneContactTask(
        name="left_foot",
        tracking_positions=tracking,
        contact_positions=contact_positions,
        surface_points=contact_positions.copy(),
        surface_normals=np.tile([0.0, 0.0, 1.0], (5, 1)),
        distances=np.zeros(5),
        relative_speeds=np.zeros(5),
        contact_confidence=np.ones(5),
        contact_modes=modes,
        contact_segment_ids=np.asarray([-1, 0, 0, 0, -1]),
        transition_weights=np.ones(5),
        effective_swing_positions=tracking.copy(),
        active_contact_offset_ids=np.asarray([-1, 0, 0, 0, -1]),
    )
    reference = trajectory.copy()
    reference[:, 7] = 0.05
    metrics = evaluate_retargeted_motion(
        model,
        trajectory,
        {"left_foot": task},
        {
            "effectors": {
                "left_foot": {
                    "robot_body": "foot",
                    "tracking_offset": [0.0, 0.0, 0.0],
                    "contact_offsets": [[0.0, 0.0, 0.0]],
                }
            },
            "base_position_constraint": {"z_offset": 0.0},
        },
        framerate=30.0,
        reference_trajectory=reference,
        source_root_positions=trajectory[:, :3],
    )

    foot = metrics["end_effectors"]["left_foot"]
    assert foot["source_world_error_m"]["all"]["max"] < 1e-12
    assert foot["contact_residual_m"]["max"] == pytest.approx(0.02)
    assert (
        foot["foot_sliding"]["tangential_displacement_m"]["max"]
        == pytest.approx(0.02)
    )
    assert (
        foot["foot_sliding"]["tangential_speed_mps"]["max"]
        == pytest.approx(0.3)
    )
    deviation = metrics["deviation_from_pre_refinement"]
    assert deviation["root_translation_m"]["max"] == pytest.approx(0.0)
    assert deviation["joint_absolute_rad"]["max"] == pytest.approx(0.05)
    assert "whole_body_horizontal_correction_m" in deviation
    assert "whole_body_vertical_correction_m" in deviation
    assert metrics["source_root_target_error_m"]["max"] == pytest.approx(0.0)


def test_effective_target_error_uses_live_limb_anchor():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(ARM_XML)
    trajectory = np.tile(model.qpos0, (4, 1))
    trajectory[:, 0] = np.linspace(0.0, 0.6, len(trajectory))
    trajectory[:, 3] = 1.0
    data = mujoco.MjData(model)
    shoulder_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "shoulder"
    )
    wrist_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "wrist"
    )
    tracking = []
    limb_offsets = []
    for qpos in trajectory:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        tracking.append(data.xpos[wrist_id].copy())
        limb_offsets.append(data.xpos[wrist_id] - data.xpos[shoulder_id])
    tracking = np.asarray(tracking)
    limb_offsets = np.asarray(limb_offsets)
    task = SceneContactTask(
        name="left_hand",
        tracking_positions=tracking,
        contact_positions=tracking.copy(),
        surface_points=tracking.copy(),
        surface_normals=np.tile([0.0, 0.0, 1.0], (len(trajectory), 1)),
        distances=np.zeros(len(trajectory)),
        relative_speeds=np.zeros(len(trajectory)),
        contact_confidence=np.ones(len(trajectory)),
        contact_modes=np.full(
            len(trajectory), int(ContactMode.NONE), dtype=np.uint8
        ),
        contact_segment_ids=np.full(len(trajectory), -1, dtype=np.int32),
        transition_weights=np.ones(len(trajectory)),
        # A stale preview at the origin would report metre-scale error.
        effective_swing_positions=np.zeros_like(tracking),
        swing_limb_offsets=limb_offsets,
        swing_anchor_body="shoulder",
        swing_anchor_offset=np.zeros(3),
        contact_bridge_alpha=np.zeros(len(trajectory)),
        contact_bridge_positions=np.zeros_like(tracking),
    )
    metrics = evaluate_retargeted_motion(
        model,
        trajectory,
        {"left_hand": task},
        {
            "effectors": {
                "left_hand": {
                    "robot_body": "wrist",
                    "tracking_offset": [0.0, 0.0, 0.0],
                    "contact_offsets": [[0.0, 0.0, 0.0]],
                }
            }
        },
        framerate=30.0,
    )
    effective = metrics["end_effectors"]["left_hand"][
        "effective_target_error_m"
    ]
    assert effective["swing"]["max"] < 1e-12

    serialized = json.loads(metrics_to_json(metrics))
    assert serialized["units"]["position"] == "m"


def test_arm_pose_fidelity_reports_both_elbow_branches_against_source():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(ARM_XML)
    trajectory = np.zeros((4, model.nq), dtype=float)
    trajectory[:, 3] = 1.0
    trajectory[:, 8] = np.pi / 2.0
    source_names = [
        "L_Shoulder", "L_Elbow", "L_Wrist",
        "R_Shoulder", "R_Elbow", "R_Wrist",
    ]
    source_arm = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]
    )
    source = np.tile(
        np.concatenate([source_arm, source_arm], axis=0)[None],
        (len(trajectory), 1, 1),
    )
    arm_posture = {
        "enabled": True,
        "shoulder_body": "shoulder",
        "elbow_body": "elbow",
        "wrist_body": "wrist",
    }
    metrics = evaluate_retargeted_motion(
        model,
        trajectory,
        None,
        {
            "effectors": {
                "left_hand": {
                    "arm_posture": {
                        **arm_posture,
                        "source_shoulder": "L_Shoulder",
                        "source_elbow": "L_Elbow",
                        "source_wrist": "L_Wrist",
                    }
                },
                "right_hand": {
                    "arm_posture": {
                        **arm_posture,
                        "source_shoulder": "R_Shoulder",
                        "source_elbow": "R_Elbow",
                        "source_wrist": "R_Wrist",
                    }
                },
            }
        },
        framerate=30.0,
        source_joint_positions=source,
        source_joint_names=source_names,
    )

    assert set(metrics["arm_pose_fidelity"]) == {"left_arm", "right_arm"}
    for arm in metrics["arm_pose_fidelity"].values():
        for key in (
            "upper_arm_direction_error_rad",
            "forearm_direction_error_rad",
            "elbow_bend_plane_error_rad",
        ):
            assert arm[key]["mean"] == pytest.approx(np.pi / 2.0)
            assert arm[key]["p95"] == pytest.approx(np.pi / 2.0)
            assert arm[key]["max"] == pytest.approx(np.pi / 2.0)
        assert arm["elbow_flexion_difference_rad"]["max"] < 1e-12
        assert arm["robot_elbow_flexion_rad"]["mean"] == pytest.approx(
            np.pi / 2.0
        )
        assert arm["source_elbow_flexion_rad"]["mean"] == pytest.approx(
            np.pi / 2.0
        )
