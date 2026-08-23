"""Self-contained primitive humanoids for morphology and pipeline tests.

The three models intentionally share names and topology so one source mapping
can drive all of them.  Their segment proportions, hip heights, collision
geometry, and mechanical joint ranges are independently specified rather than
obtained from a uniform scale factor.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting.retargeting import GenericInteractionRetargeter


@dataclass(frozen=True)
class _RobotShape:
    hip_height: float
    torso_lower: float
    torso_upper: float
    shoulder_half_width: float
    hip_half_width: float
    upper_arm: float
    forearm: float
    palm: float
    thigh: float
    shin: float
    foot: float
    shoulder_range: tuple[float, float]
    elbow_range: tuple[float, float]
    hip_range: tuple[float, float]
    knee_range: tuple[float, float]
    ankle_range: tuple[float, float]


# These are different humanoid proportions, not a common skeleton multiplied
# by three scale factors.  For example, SHORT has relatively long arms and a
# short torso, while TALL has a long torso and legs but narrower arm/height
# ratio.  Joint ranges differ independently as well.
_SHAPES = {
    "short": _RobotShape(
        hip_height=0.68,
        torso_lower=0.19,
        torso_upper=0.22,
        shoulder_half_width=0.19,
        hip_half_width=0.105,
        upper_arm=0.245,
        forearm=0.225,
        palm=0.085,
        thigh=0.315,
        shin=0.285,
        foot=0.185,
        shoulder_range=(-1.75, 1.35),
        elbow_range=(-2.30, 0.20),
        hip_range=(-1.15, 1.05),
        knee_range=(-0.10, 2.35),
        ankle_range=(-0.72, 0.58),
    ),
    "medium": _RobotShape(
        hip_height=0.86,
        torso_lower=0.25,
        torso_upper=0.29,
        shoulder_half_width=0.245,
        hip_half_width=0.12,
        upper_arm=0.285,
        forearm=0.265,
        palm=0.10,
        thigh=0.405,
        shin=0.375,
        foot=0.225,
        shoulder_range=(-1.95, 1.55),
        elbow_range=(-2.55, 0.15),
        hip_range=(-1.32, 1.18),
        knee_range=(-0.08, 2.55),
        ankle_range=(-0.82, 0.66),
    ),
    "tall": _RobotShape(
        hip_height=1.08,
        torso_lower=0.34,
        torso_upper=0.39,
        shoulder_half_width=0.285,
        hip_half_width=0.145,
        upper_arm=0.335,
        forearm=0.305,
        palm=0.115,
        thigh=0.515,
        shin=0.475,
        foot=0.27,
        shoulder_range=(-2.12, 1.72),
        elbow_range=(-2.68, 0.10),
        hip_range=(-1.48, 1.28),
        knee_range=(-0.05, 2.68),
        ankle_range=(-0.91, 0.73),
    ),
}


VALID_JOINT_NAMES = [
    "Pelvis",
    "Spine1",
    "Spine2",
    "Spine3",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Foot",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Foot",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
]

JOINT_MAPPING = {
    "Pelvis": "pelvis",
    "Spine1": "torso_lower",
    "Spine2": "torso_mid",
    "Spine3": "torso",
    "L_Hip": "left_hip",
    "L_Knee": "left_knee",
    "L_Ankle": "left_ankle",
    "L_Foot": "left_foot",
    "R_Hip": "right_hip",
    "R_Knee": "right_knee",
    "R_Ankle": "right_ankle",
    "R_Foot": "right_foot",
    "L_Shoulder": "left_shoulder",
    "L_Elbow": "left_elbow",
    "L_Wrist": "left_hand",
    "R_Shoulder": "right_shoulder",
    "R_Elbow": "right_elbow",
    "R_Wrist": "right_hand",
}


def _pair(values: tuple[float, float]) -> str:
    return f"{values[0]:.6f} {values[1]:.6f}"


def _arm_xml(side: str, sign: int, shape: _RobotShape) -> str:
    y_shoulder = sign * shape.shoulder_half_width
    y_upper = sign * shape.upper_arm
    y_forearm = sign * shape.forearm
    y_palm_end = sign * shape.palm
    return f"""
      <body name="{side}_shoulder" pos="0 {y_shoulder:.6f} 0">
        <joint name="{side}_shoulder_joint" type="hinge" axis="1 0 0"
               range="{_pair(shape.shoulder_range)}" damping="0.08"/>
        <geom name="{side}_upper_arm_geom" type="capsule"
              fromto="0 0 0 0 {y_upper:.6f} 0" size="0.035" mass="0.55"/>
        <body name="{side}_elbow" pos="0 {y_upper:.6f} 0">
          <joint name="{side}_elbow_joint" type="hinge" axis="1 0 0"
                 range="{_pair(shape.elbow_range)}" damping="0.06"/>
          <geom name="{side}_forearm_geom" type="capsule"
                fromto="0 0 0 0 {y_forearm:.6f} 0" size="0.030" mass="0.40"/>
          <body name="{side}_hand" pos="0 {y_forearm:.6f} 0">
            <joint name="{side}_wrist_joint" type="hinge" axis="0 0 1"
                   range="-0.75 0.75" damping="0.03"/>
            <geom name="{side}_palm" type="box"
                  pos="0 {0.5 * y_palm_end:.6f} 0"
                  size="0.035 {0.5 * shape.palm:.6f} 0.018" mass="0.16"/>
          </body>
        </body>
      </body>"""


def _leg_xml(side: str, sign: int, shape: _RobotShape) -> str:
    y_hip = sign * shape.hip_half_width
    return f"""
    <body name="{side}_hip" pos="0 {y_hip:.6f} 0">
      <joint name="{side}_hip_joint" type="hinge" axis="0 1 0"
             range="{_pair(shape.hip_range)}" damping="0.12"/>
      <geom name="{side}_thigh_geom" type="capsule"
            fromto="0 0 0 0 0 {-shape.thigh:.6f}" size="0.050" mass="1.55"/>
      <body name="{side}_knee" pos="0 0 {-shape.thigh:.6f}">
        <joint name="{side}_knee_joint" type="hinge" axis="0 1 0"
               range="{_pair(shape.knee_range)}" damping="0.11"/>
        <geom name="{side}_shin_geom" type="capsule"
              fromto="0 0 0 0 0 {-shape.shin:.6f}" size="0.043" mass="1.15"/>
        <body name="{side}_ankle" pos="0 0 {-shape.shin:.6f}">
          <joint name="{side}_ankle_joint" type="hinge" axis="0 1 0"
                 range="{_pair(shape.ankle_range)}" damping="0.07"/>
          <geom name="{side}_ankle_geom" type="sphere" size="0.045" mass="0.18"/>
          <body name="{side}_foot" pos="{0.42 * shape.foot:.6f} 0 -0.045">
            <geom name="{side}_sole" type="box"
                  pos="{0.08 * shape.foot:.6f} 0 -0.018"
                  size="{0.50 * shape.foot:.6f} 0.065 0.018" mass="0.42"/>
          </body>
        </body>
      </body>
    </body>"""


def _mjcf(shape: _RobotShape, name: str) -> str:
    torso_radius = 0.105 + 0.10 * shape.shoulder_half_width
    return f"""
<mujoco model="primitive_{name}_humanoid">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <default>
    <joint limited="true"/>
    <geom contype="1" conaffinity="1" friction="0.8 0.02 0.01"/>
  </default>
  <worldbody>
    <body name="pelvis" pos="0 0 {shape.hip_height:.6f}">
      <freejoint/>
      <geom name="pelvis_geom" type="box"
            size="0.105 {shape.hip_half_width:.6f} 0.075" mass="3.0"/>
      <body name="torso_lower" pos="0 0 {0.30 * shape.torso_lower:.6f}">
        <joint name="waist_joint" type="hinge" axis="0 0 1" range="-0.55 0.55" damping="0.15"/>
        <geom type="capsule" fromto="0 0 0 0 0 {0.70 * shape.torso_lower:.6f}"
              size="{torso_radius:.6f}" mass="2.2"/>
        <body name="torso_mid" pos="0 0 {0.70 * shape.torso_lower:.6f}">
          <joint name="chest_joint" type="hinge" axis="0 1 0" range="-0.45 0.60" damping="0.12"/>
          <geom type="capsule" fromto="0 0 0 0 0 {shape.torso_upper:.6f}"
                size="{1.08 * torso_radius:.6f}" mass="2.5"/>
          <body name="torso" pos="0 0 {shape.torso_upper:.6f}">
            <geom name="torso_geom" type="box"
                  size="0.11 {0.86 * shape.shoulder_half_width:.6f} 0.085" mass="2.0"/>
            <body name="head" pos="0 0 0.085">
              <geom name="head_geom" type="sphere" size="0.085" mass="0.65"/>
            </body>
            {_arm_xml("left", 1, shape)}
            {_arm_xml("right", -1, shape)}
          </body>
        </body>
      </body>
      {_leg_xml("left", 1, shape)}
      {_leg_xml("right", -1, shape)}
    </body>
  </worldbody>
</mujoco>
"""


_PRIORITY_LEVELS = [
    {"name": "support", "families": ["support"], "absolute_tolerance": 0.02},
    {"name": "contact", "families": ["contact"], "absolute_tolerance": 0.04},
    {"name": "swing", "families": ["swing"], "absolute_tolerance": 0.04},
    {"name": "orientation", "families": ["orientation"], "absolute_tolerance": 0.04},
    {"name": "pose", "families": ["pose_direction"], "absolute_tolerance": 0.04},
    {"name": "interaction", "families": ["interaction"], "absolute_tolerance": 0.05},
    {"name": "regularization", "families": ["regularization"], "absolute_tolerance": 0.04},
]


def _effector(body: str, offset: list[float], anchor: str) -> dict:
    return {
        "robot_body": body,
        "tracking_offset": offset,
        "contact_offsets": [offset],
        "contact_target_space": "world",
        "swing_target_space": "limb_morphology_scaled",
        "reach_reference_body": anchor,
        "track_orientation": False,
    }


def build_robot(size: str) -> tuple[str, dict]:
    """Return an inline MJCF and independent config for ``size``.

    Source/scene contact inference is deliberately absent from the returned
    config: callers may infer it once, then use each robot's own effector and
    morphology config while preparing robot-specific references.
    """
    key = str(size).strip().lower()
    if key not in _SHAPES:
        raise ValueError(f"Unknown robot size {size!r}; expected {sorted(_SHAPES)}.")
    shape = _SHAPES[key]
    config = {
        "name": key,
        "joint_mapping": deepcopy(JOINT_MAPPING),
        "valid_joint_names": list(VALID_JOINT_NAMES),
        "robot_height": shape.hip_height + shape.torso_lower + shape.torso_upper + 0.17,
        "terrain_sample_seed": 173,
        "base_orientation": {
            "root_position": "Pelvis",
            "pelvis": "Pelvis",
            "left_hip": "L_Hip",
            "right_hip": "R_Hip",
            "spine": "Spine1",
        },
        "end_effector_priority": {
            "enabled": True,
            "hierarchy": True,
            "priority_levels": deepcopy(_PRIORITY_LEVELS),
            "pose_direction": {"enabled": True, "post_projection": False},
            "effectors": {
                "left_hand": _effector("left_hand", [0.0, shape.palm, 0.0], "left_shoulder"),
                "right_hand": _effector("right_hand", [0.0, -shape.palm, 0.0], "right_shoulder"),
                "left_foot": _effector("left_foot", [0.5 * shape.foot, 0.0, -0.036], "left_hip"),
                "right_foot": _effector(
                    "right_foot", [0.5 * shape.foot, 0.0, -0.036], "right_hip"
                ),
            },
        },
        "nominal_dimensions": {
            "hip_height": shape.hip_height,
            "torso_length": shape.torso_lower + shape.torso_upper,
            "arm_reach": shape.upper_arm + shape.forearm + shape.palm,
            "leg_length": shape.thigh
            + shape.shin
            + float(np.hypot(0.92 * shape.foot, 0.081)),
        },
    }
    return _mjcf(shape, key), config


def measure_robot_dimensions(xml: str, config: dict) -> dict[str, float]:
    """Measure nominal chain dimensions from the compiled zero pose."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def position(body_name: str) -> np.ndarray:
        return data.xpos[model.body(body_name).id].copy()

    def marker(effector_name: str) -> np.ndarray:
        spec = config["end_effector_priority"]["effectors"][effector_name]
        body_id = model.body(spec["robot_body"]).id
        offset = np.asarray(spec["tracking_offset"], dtype=float)
        return data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ offset

    arm_reach = (
        np.linalg.norm(position("left_elbow") - position("left_shoulder"))
        + np.linalg.norm(position("left_hand") - position("left_elbow"))
        + np.linalg.norm(marker("left_hand") - position("left_hand"))
    )
    leg_length = (
        np.linalg.norm(position("left_knee") - position("left_hip"))
        + np.linalg.norm(position("left_ankle") - position("left_knee"))
        + np.linalg.norm(marker("left_foot") - position("left_ankle"))
    )
    return {
        "robot_height": float(config["robot_height"]),
        "hip_height": float(position("pelvis")[2]),
        "torso_length": float(np.linalg.norm(position("torso") - position("pelvis"))),
        "arm_reach": float(arm_reach),
        "leg_length": float(leg_length),
    }


def _mapped_positions(model: mujoco.MjModel, data: mujoco.MjData, config: dict) -> np.ndarray:
    mujoco.mj_forward(model, data)
    return np.asarray(
        [
            data.xpos[model.body(config["joint_mapping"][name]).id].copy()
            for name in config["valid_joint_names"]
        ]
    )


def _effector_targets(model: mujoco.MjModel, data: mujoco.MjData, config: dict) -> dict:
    targets = {}
    for name, spec in config["end_effector_priority"]["effectors"].items():
        body = model.body(spec["robot_body"]).id
        offset = np.asarray(spec["contact_offsets"][0], dtype=float)
        point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ offset
        targets[name] = {
            "contact_mode": "SUPPORT",
            "contact_segment_id": 0,
            "contact_position": point,
            "surface_normal": [0.0, 0.0, 1.0],
            "confidence": 1.0,
            "transition_weight": 1.0,
        }
    return targets


@pytest.mark.parametrize("size", ["short", "medium", "tall"])
def test_primitive_humanoid_runs_two_framewise_solver_frames(size: str) -> None:
    """Each inline fixture reaches the real framewise optimizer, twice."""
    xml, config = build_robot(size)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    np.testing.assert_allclose(model.qpos0[3:7], [1.0, 0.0, 0.0, 0.0])
    scalar_types = {mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE}
    assert all(joint_type in scalar_types for joint_type in model.jnt_type[1:])
    assert all(model.jnt_limited[1:])

    ground = trimesh.creation.box(extents=[4.0, 4.0, 0.10])
    ground.apply_translation([0.0, 0.0, -0.05])
    retargeter = GenericInteractionRetargeter(
        model,
        data,
        ground,
        config["joint_mapping"],
        robot_height=config["robot_height"],
        terrain_sample_points=12,
        terrain_sample_seed=config["terrain_sample_seed"],
        valid_joint_names=config["valid_joint_names"],
        hard_penetration_constraint=False,
        end_effector_config=config["end_effector_priority"],
    )

    q0 = model.qpos0.copy()
    data.qpos[:] = q0
    human0 = _mapped_positions(model, data, config)
    targets = _effector_targets(model, data, config)
    q1 = retargeter.retarget_frame(
        human0,
        q0,
        max_iter=1,
        target_base_orientation=q0[3:7],
        target_base_position=q0[:3],
        end_effector_targets=targets,
    )
    human1 = human0 + np.array([0.008, 0.0, 0.0])
    q2 = retargeter.retarget_frame(
        human1,
        q1,
        max_iter=1,
        q_last=q1,
        previous_human_joints=human0,
        target_base_orientation=q0[3:7],
        target_base_position=q0[:3] + [0.008, 0.0, 0.0],
        end_effector_targets=targets,
    )
    assert q1.shape == q2.shape == (model.nq,)
    assert np.all(np.isfinite(q1)) and np.all(np.isfinite(q2))


def test_robot_proportions_and_ranges_are_not_uniform_scalings() -> None:
    fixtures = {size: build_robot(size) for size in _SHAPES}
    configs = {size: fixture[1] for size, fixture in fixtures.items()}
    measured = {
        size: measure_robot_dimensions(*fixture) for size, fixture in fixtures.items()
    }
    ratios = {
        size: (
            cfg["nominal_dimensions"]["arm_reach"] / cfg["robot_height"],
            cfg["nominal_dimensions"]["leg_length"] / cfg["robot_height"],
            cfg["nominal_dimensions"]["torso_length"] / cfg["robot_height"],
        )
        for size, cfg in configs.items()
    }
    assert len({tuple(np.round(value, 4)) for value in ratios.values()}) == 3
    assert len({_SHAPES[size].knee_range for size in _SHAPES}) == 3
    assert all(configs[size]["joint_mapping"] == JOINT_MAPPING for size in _SHAPES)
    assert all(configs[size]["valid_joint_names"] == VALID_JOINT_NAMES for size in _SHAPES)
    for size in _SHAPES:
        for dimension in ("hip_height", "torso_length", "arm_reach", "leg_length"):
            assert measured[size][dimension] == pytest.approx(
                configs[size]["nominal_dimensions"][dimension]
            )
