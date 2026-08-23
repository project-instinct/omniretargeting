from __future__ import annotations

from copy import deepcopy

import mujoco
import numpy as np
import pytest

from omniretargeting.core import OmniRetargeter
from omniretargeting.morphology import (
    analyze_robot_morphology,
    per_chain_ratios,
)


_HUMANOID_TEMPLATE = """
<mujoco model="{model_name}">
  <compiler angle="degree"/>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    <body name="pelvis" pos="0 0 {pelvis_height}">
      <freejoint name="floating_base"/>
      <geom name="buttock_shell" type="box" size="{pelvis_x} {pelvis_y} {pelvis_z}"/>
      <body name="torso" pos="0 0 {torso_length}">
        <joint name="waist_yaw" type="hinge" axis="0 0 1" range="{waist_lower} {waist_upper}"/>
        <geom name="torso_shell" type="capsule" size="0.09" fromto="0 0 -0.10 0 0 0.10"/>

        <body name="left_shoulder" pos="0 {shoulder_half_width} 0.08">
          <joint name="left_shoulder_joint" type="hinge" axis="1 0 0" range="{shoulder_lower} {shoulder_upper}"/>
          <geom name="left_shoulder_shell" type="sphere" size="0.045"/>
          <body name="left_elbow" pos="0 {upper_arm} 0" euler="0 0 {elbow_bend}">
            <joint name="left_elbow_joint" type="hinge" axis="0 0 1" range="{elbow_lower} {elbow_upper}"/>
            <geom name="left_forearm_shell" type="capsule" size="0.035" fromto="0 0 0 0 {forearm} 0"/>
            <body name="left_hand" pos="0 {forearm} 0">
              <joint name="left_wrist_joint" type="hinge" axis="1 0 0" range="{wrist_lower} {wrist_upper}"/>
              <geom name="left_palm_shell" type="box" pos="0 {palm_offset} 0" size="0.055 0.070 0.020"/>
            </body>
          </body>
        </body>

        <body name="right_shoulder" pos="0 -{shoulder_half_width} 0.08">
          <joint name="right_shoulder_joint" type="hinge" axis="1 0 0" range="{shoulder_lower} {shoulder_upper}"/>
          <geom name="right_shoulder_shell" type="sphere" size="0.045"/>
          <body name="right_elbow" pos="0 -{upper_arm} 0" euler="0 0 {elbow_bend}">
            <joint name="right_elbow_joint" type="hinge" axis="0 0 1" range="{elbow_lower} {elbow_upper}"/>
            <geom name="right_forearm_shell" type="capsule" size="0.035" fromto="0 0 0 0 -{forearm} 0"/>
            <body name="right_hand" pos="0 -{forearm} 0">
              <joint name="right_wrist_joint" type="hinge" axis="1 0 0" range="{wrist_lower} {wrist_upper}"/>
              <geom name="right_palm_shell" type="capsule" pos="0 -{palm_offset} 0" euler="90 0 0" size="0.025 0.060"/>
            </body>
          </body>
        </body>
      </body>

      <body name="left_hip" pos="0 {hip_half_width} 0">
        <joint name="left_hip_joint" type="hinge" axis="0 1 0" range="{hip_lower} {hip_upper}"/>
        <geom name="left_hip_shell" type="sphere" size="0.055"/>
        <body name="left_knee" pos="0 0 -{thigh}" euler="0 {knee_bend} 0">
          <joint name="left_knee_joint" type="hinge" axis="0 1 0" range="{knee_lower} {knee_upper}"/>
          <geom name="left_shin_shell" type="capsule" size="0.045" fromto="0 0 0 0 0 -{shin}"/>
          <body name="left_foot" pos="0 0 -{shin}">
            <joint name="left_ankle_joint" type="hinge" axis="0 1 0" range="{ankle_lower} {ankle_upper}"/>
            <geom name="left_sole_shell" type="box" pos="{sole_x} 0 -0.035" size="0.12 0.055 0.025"/>
          </body>
        </body>
      </body>

      <body name="right_hip" pos="0 -{hip_half_width} 0">
        <joint name="right_hip_joint" type="hinge" axis="0 1 0" range="{hip_lower} {hip_upper}"/>
        <geom name="right_hip_shell" type="sphere" size="0.055"/>
        <body name="right_knee" pos="0 0 -{thigh}" euler="0 {knee_bend} 0">
          <joint name="right_knee_joint" type="hinge" axis="0 1 0" range="{knee_lower} {knee_upper}"/>
          <geom name="right_shin_shell" type="capsule" size="0.045" fromto="0 0 0 0 0 -{shin}"/>
          <body name="right_foot" pos="0 0 -{shin}">
            <joint name="right_ankle_joint" type="hinge" axis="0 1 0" range="{ankle_lower} {ankle_upper}"/>
            <geom name="right_sole_shell" type="box" pos="{sole_x} 0 -0.035" size="0.12 0.055 0.025"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


# These remain local inline fixtures until tests/robot_fixtures.py lands.
ROBOT_COMPACT_XML = _HUMANOID_TEMPLATE.format(
    model_name="compact",
    pelvis_height=0.88,
    pelvis_x=0.15,
    pelvis_y=0.12,
    pelvis_z=0.105,
    torso_length=0.36,
    waist_lower=-25,
    waist_upper=25,
    shoulder_half_width=0.17,
    shoulder_lower=-80,
    shoulder_upper=65,
    upper_arm=0.28,
    forearm=0.23,
    elbow_bend=42,
    elbow_lower=-65,
    elbow_upper=8,
    wrist_lower=-30,
    wrist_upper=35,
    palm_offset=0.055,
    hip_half_width=0.095,
    hip_lower=-55,
    hip_upper=40,
    thigh=0.39,
    shin=0.34,
    knee_bend=28,
    knee_lower=-48,
    knee_upper=7,
    ankle_lower=-25,
    ankle_upper=20,
    sole_x=0.075,
)

ROBOT_LONG_ARM_XML = _HUMANOID_TEMPLATE.format(
    model_name="long_arm_short_leg",
    pelvis_height=0.79,
    pelvis_x=0.17,
    pelvis_y=0.14,
    pelvis_z=0.125,
    torso_length=0.48,
    waist_lower=-38,
    waist_upper=32,
    shoulder_half_width=0.24,
    shoulder_lower=-115,
    shoulder_upper=95,
    upper_arm=0.43,
    forearm=0.35,
    elbow_bend=58,
    elbow_lower=-92,
    elbow_upper=12,
    wrist_lower=-55,
    wrist_upper=50,
    palm_offset=0.080,
    hip_half_width=0.13,
    hip_lower=-42,
    hip_upper=35,
    thigh=0.33,
    shin=0.28,
    knee_bend=18,
    knee_lower=-32,
    knee_upper=10,
    ankle_lower=-18,
    ankle_upper=16,
    sole_x=0.095,
)

ROBOT_LONG_LEG_XML = _HUMANOID_TEMPLATE.format(
    model_name="short_arm_long_leg",
    pelvis_height=1.14,
    pelvis_x=0.14,
    pelvis_y=0.105,
    pelvis_z=0.095,
    torso_length=0.29,
    waist_lower=-18,
    waist_upper=20,
    shoulder_half_width=0.145,
    shoulder_lower=-62,
    shoulder_upper=58,
    upper_arm=0.235,
    forearm=0.19,
    elbow_bend=27,
    elbow_lower=-38,
    elbow_upper=15,
    wrist_lower=-22,
    wrist_upper=25,
    palm_offset=0.045,
    hip_half_width=0.085,
    hip_lower=-75,
    hip_upper=62,
    thigh=0.55,
    shin=0.48,
    knee_bend=36,
    knee_lower=-72,
    knee_upper=8,
    ankle_lower=-34,
    ankle_upper=28,
    sole_x=0.070,
)


def _semantic_mapping(pelvis_support_depth: float) -> dict:
    return {
        "root_body": "pelvis",
        "chains": {
            "left_hand": {
                "reference_body": "left_shoulder",
                "endpoint_body": "left_hand",
                "endpoint_offset": [0.0, 0.07, 0.0],
            },
            "right_hand": {
                "reference_body": "right_shoulder",
                "endpoint_body": "right_hand",
                "endpoint_offset": [0.0, -0.07, 0.0],
            },
            "left_foot": {
                "reference_body": "left_hip",
                "endpoint_body": "left_foot",
                "endpoint_offset": [0.09, 0.0, -0.04],
            },
            "right_foot": {
                "reference_body": "right_hip",
                "endpoint_body": "right_foot",
                "endpoint_offset": [0.09, 0.0, -0.04],
            },
            "pelvis_support": {
                "reference_body": "pelvis",
                "endpoint_body": "pelvis",
                "endpoint_offset": [0.0, 0.0, -pelvis_support_depth],
            },
        },
        "contact_patches": {
            "left_palm": {
                "body": "left_hand",
                "geom_tokens": ["palm"],
                "normal": [0.0, 0.0, -1.0],
            },
            "right_palm": {
                "body": "right_hand",
                "geom_tokens": ["palm"],
                "normal": [0.0, 0.0, -1.0],
            },
            "left_sole": {
                "body": "left_foot",
                "geom_tokens": ["sole"],
                "normal": [0.0, 0.0, -1.0],
            },
            "right_sole": {
                "body": "right_foot",
                "geom_tokens": ["sole"],
                "normal": [0.0, 0.0, -1.0],
            },
            "pelvis_support": {
                "body": "pelvis",
                "geom_tokens": ["buttock"],
                "normal": [0.0, 0.0, -1.0],
            },
        },
    }


def _reports():
    cases = [
        (ROBOT_COMPACT_XML, 0.105),
        (ROBOT_LONG_ARM_XML, 0.125),
        (ROBOT_LONG_LEG_XML, 0.095),
    ]
    return [
        analyze_robot_morphology(
            mujoco.MjModel.from_xml_string(xml),
            _semantic_mapping(support_depth),
            reach_samples=256,
        )
        for xml, support_depth in cases
    ]


def test_three_humanoids_have_independent_measured_arm_and_leg_morphology():
    compact, long_arm, long_leg = _reports()
    arm_lengths = [
        report.chains["left_hand"].nominal_length
        for report in (compact, long_arm, long_leg)
    ]
    leg_lengths = [
        report.chains["left_foot"].nominal_length
        for report in (compact, long_arm, long_leg)
    ]
    assert len(set(np.round(arm_lengths, 6))) == 3
    assert len(set(np.round(leg_lengths, 6))) == 3

    ratios = per_chain_ratios(long_arm, compact)
    assert ratios["left_hand"] != pytest.approx(ratios["left_foot"], abs=1e-3)
    assert ratios["left_hand"] > 1.3
    assert ratios["left_foot"] < 1.0

    assert compact.chains["left_hand"].body_names == (
        "left_shoulder",
        "left_elbow",
        "left_hand",
    )
    assert compact.chains["left_foot"].body_names == (
        "left_hip",
        "left_knee",
        "left_foot",
    )
    for report in (compact, long_arm, long_leg):
        for chain in report.chains.values():
            assert chain.nominal_length == pytest.approx(
                sum(segment.length for segment in chain.segments)
            )


def test_joint_ranges_are_mechanical_intersected_with_configured_safe_ranges():
    model = mujoco.MjModel.from_xml_string(ROBOT_COMPACT_XML)
    report = analyze_robot_morphology(
        model,
        _semantic_mapping(0.105),
        configured_joint_limits={"left_elbow_joint": [-0.4, 0.05]},
        reach_samples=256,
    )
    joints = {joint.name: joint for joint in report.chains["left_hand"].joints}
    elbow = joints["left_elbow_joint"]
    assert elbow.joint_type == "hinge"
    assert elbow.mechanical_range == pytest.approx(
        (np.deg2rad(-65.0), np.deg2rad(8.0))
    )
    assert elbow.effective_range == pytest.approx((-0.4, 0.05))
    assert elbow.sampled_for_reach

    long_arm_model = mujoco.MjModel.from_xml_string(ROBOT_LONG_ARM_XML)
    assert model.jnt_range[model.joint("left_elbow_joint").id] != pytest.approx(
        long_arm_model.jnt_range[long_arm_model.joint("left_elbow_joint").id]
    )


def test_reach_samples_joint_limits_and_is_not_the_qpos0_proxy():
    model = mujoco.MjModel.from_xml_string(ROBOT_COMPACT_XML)
    mapping = _semantic_mapping(0.105)
    broad = analyze_robot_morphology(model, mapping, reach_samples=512)
    safe = analyze_robot_morphology(
        model,
        mapping,
        configured_joint_limits={"left_elbow_joint": [-0.15, 0.05]},
        reach_samples=512,
    )
    broad_arm = broad.chains["left_hand"]
    safe_arm = safe.chains["left_hand"]
    assert broad_arm.actual_reach > broad_arm.qpos0_reach + 0.02
    assert safe_arm.actual_reach < broad_arm.actual_reach - 0.01
    assert safe_arm.actual_reach >= safe_arm.qpos0_reach


def test_core_reach_adapter_uses_measurement_without_an_explicit_override():
    model = mujoco.MjModel.from_xml_string(ROBOT_COMPACT_XML)
    retargeter = OmniRetargeter.__new__(OmniRetargeter)
    retargeter.robot_model = model
    retargeter.valid_joint_mapping = {}
    config = {
        "morphology_reach_samples": 256,
        "effectors": {
            "left_hand": {
                "robot_body": "left_hand",
                "reach_reference_body": "left_shoulder",
                "tracking_offset": [0.0, 0.07, 0.0],
            }
        },
    }
    profile = retargeter._end_effector_reach_profiles(config)["left_hand"]
    assert profile["max_reach"] == pytest.approx(profile["measured_reach"])
    assert profile["max_reach"] > profile["qpos0_reach"] + 0.02

    config_with_override = deepcopy(config)
    config_with_override["effectors"]["left_hand"]["max_swing_reach"] = 10.0
    overridden = retargeter._end_effector_reach_profiles(config_with_override)[
        "left_hand"
    ]
    assert overridden["max_reach"] == 10.0
    assert overridden["measured_reach"] == pytest.approx(
        profile["measured_reach"]
    )


def _assert_points_on_geom_surface(
    model: mujoco.MjModel, geom_id: int, body_points: np.ndarray
) -> None:
    quat = np.asarray(model.geom_quat[geom_id], dtype=float)
    rotation = np.empty(9, dtype=float)
    mujoco.mju_quat2Mat(rotation, quat)
    rotation = rotation.reshape(3, 3)
    local = (body_points - model.geom_pos[geom_id]) @ rotation
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        assert np.all(np.abs(local) <= size[:3] + 1e-9)
        assert np.all(
            np.any(np.isclose(np.abs(local), size[:3], atol=1e-9), axis=1)
        )
    elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        radial = local.copy()
        radial[:, 2] -= np.clip(radial[:, 2], -size[1], size[1])
        assert np.linalg.norm(radial, axis=1) == pytest.approx(
            np.full(len(local), size[0]), abs=1e-9
        )
    else:
        raise AssertionError(f"Unexpected test geom type {geom_type}")


def test_contact_patch_samples_lie_on_real_palm_sole_and_buttock_geoms():
    model = mujoco.MjModel.from_xml_string(ROBOT_COMPACT_XML)
    report = analyze_robot_morphology(
        model, _semantic_mapping(0.105), reach_samples=64
    )
    for patch_name in (
        "left_palm",
        "right_palm",
        "left_sole",
        "right_sole",
        "pelvis_support",
    ):
        patch = report.contact_patches[patch_name]
        assert len(patch.surface_offsets) > 0
        assert patch.surface_offsets.shape == patch.surface_normals.shape
        assert np.all(np.isfinite(patch.extent))
        assert np.all(patch.extent >= 0.0)
        assert np.all(patch.surface_normals @ patch.preferred_normal >= 0.5 - 1e-9)
        assert len(patch.geom_ids) == 1
        _assert_points_on_geom_surface(
            model, patch.geom_ids[0], patch.surface_offsets
        )
    assert np.all(
        report.contact_patches["pelvis_support"].surface_normals[:, 2] < -0.99
    )


def test_analysis_does_not_modify_model_arrays_or_semantic_mapping():
    model = mujoco.MjModel.from_xml_string(ROBOT_COMPACT_XML)
    mapping = _semantic_mapping(0.105)
    mapping_before = deepcopy(mapping)
    arrays_before = {
        "qpos0": np.asarray(model.qpos0).copy(),
        "jnt_range": np.asarray(model.jnt_range).copy(),
        "body_pos": np.asarray(model.body_pos).copy(),
        "geom_pos": np.asarray(model.geom_pos).copy(),
        "geom_quat": np.asarray(model.geom_quat).copy(),
        "geom_size": np.asarray(model.geom_size).copy(),
    }
    analyze_robot_morphology(
        model,
        mapping,
        configured_joint_limits={"left_elbow_joint": [-0.4, 0.05]},
        reach_samples=64,
    )
    assert mapping == mapping_before
    for name, before in arrays_before.items():
        np.testing.assert_array_equal(np.asarray(getattr(model, name)), before)
