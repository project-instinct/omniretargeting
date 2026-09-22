import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from omniretargeting.contact import (
    SceneContactTask,
    build_world_tracking_tasks,
)
from omniretargeting.core import OmniRetargeter, _partition_end_effector_markers
from omniretargeting.retargeting import (
    GenericInteractionRetargeter,
    _apply_configured_joint_limits,
    _apply_configured_joint_regularization,
    _apply_joint_limit_safety_fraction,
    _base_orientation_step_limit,
    _calibrated_orientation_target,
    _clamp_target_to_reach,
    _limit_relative_orientation_step,
    _relative_orientation_step_magnitude,
    _swing_target_z_offset,
    _transform_swing_target,
)
from omniretargeting.trajectory_refinement import (
    _inset_finite_joint_bounds,
    _joint_coordinate_mask,
    _joint_posture_target_vectors,
    _joint_temporal_weight_vector,
    _local_semantic_rotation,
    _skew_symmetric,
    _transported_arm_segment_targets,
)
from omniretargeting.utils import _densify_smplx_frame_gaps, load_smplx_trajectory
from tests._contact_fixtures import ANKLE_ARM_XML, _task

AUTO_END_EFFECTOR_GEOMETRY_XML = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="root">
      <freejoint/>
      <geom type="sphere" size="0.01" mass="0.01"/>
      <body name="torso">
        <joint name="waist_yaw_joint" type="hinge" axis="0 0 1" range="-2 2"/>
        <geom type="sphere" size="0.01" mass="0.01"/>
        <body name="left_shoulder_pitch_link">
          <joint name="left_shoulder_pitch_joint" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size="0.01" mass="0.01"/>
          <body name="left_elbow_link" pos="0.2 0 0">
            <joint name="left_elbow_joint" type="hinge" axis="0 1 0"/>
            <geom type="sphere" size="0.01" mass="0.01"/>
            <body name="left_wrist_roll_link" pos="0.18 0 0">
              <joint name="left_wrist_roll_joint" type="hinge" axis="1 0 0"/>
              <geom type="sphere" size="0.01" mass="0.01"/>
              <body name="left_wrist_pitch_link">
                <joint name="left_wrist_pitch_joint" type="hinge" axis="0 1 0"/>
                <geom type="sphere" size="0.01" mass="0.01"/>
                <body name="left_wrist_yaw_link">
                  <joint name="left_wrist_yaw_joint" type="hinge" axis="0 0 1"/>
                  <geom type="sphere" size="0.01" mass="0.01"/>
                  <body name="left_hand">
                    <geom type="box" pos="0.08 -0.01 0.005" size="0.05 0.03 0.02"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
        <body name="left_foot">
          <geom type="box" pos="0.04 0 -0.025" size="0.09 0.04 0.01"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_end_effector_offsets_are_inferred_from_robot_geometry():
    mujoco = pytest.importorskip("mujoco")
    retargeter = OmniRetargeter.__new__(OmniRetargeter)
    retargeter.robot_model = mujoco.MjModel.from_xml_string(
        AUTO_END_EFFECTOR_GEOMETRY_XML
    )
    retargeter.retargeting_config = {
        "end_effector_priority": {
            "effectors": {
                "left_hand": {
                    "robot_body": "left_hand",
                    "refine_orientation": True,
                    "arm_posture": {"enabled": True},
                },
                "left_foot": {"robot_body": "left_foot"},
            }
        }
    }

    retargeter._infer_end_effector_geometry()

    effectors = retargeter.retargeting_config["end_effector_priority"][
        "effectors"
    ]
    assert effectors["left_hand"]["tracking_offset"] == [0.08, -0.01, 0.005]
    assert len(effectors["left_hand"]["contact_offsets"]) == 5
    assert effectors["left_hand"]["contact_joint_names"] == [
        "left_shoulder_pitch_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ]
    assert effectors["left_hand"]["orientation_joint_tokens"] == [
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ]
    assert effectors["left_hand"]["orientation_local_forward"] == [1.0, 0.0, 0.0]
    assert effectors["left_hand"]["orientation_local_up"] == [0.0, -1.0, 0.0]
    arm_posture = effectors["left_hand"]["arm_posture"]
    assert arm_posture["shoulder_body"] == "left_shoulder_pitch_link"
    assert arm_posture["elbow_body"] == "left_elbow_link"
    assert arm_posture["wrist_body"] == "left_wrist_yaw_link"
    assert arm_posture["upper_arm_length"] == pytest.approx(0.2)
    assert arm_posture["forearm_length"] == pytest.approx(0.18)
    assert effectors["left_foot"]["tracking_offset"] == [0.04, 0.0, -0.035]
    assert effectors["left_foot"]["contact_offsets"] == [[0.04, 0.0, -0.035]]
    priority = retargeter.retargeting_config["end_effector_priority"]
    assert "joint_posture_limits" not in priority
    assert "joint_regularization_weights" not in priority


def test_explicit_end_effector_offsets_override_geometry_inference():
    mujoco = pytest.importorskip("mujoco")
    retargeter = OmniRetargeter.__new__(OmniRetargeter)
    retargeter.robot_model = mujoco.MjModel.from_xml_string(
        AUTO_END_EFFECTOR_GEOMETRY_XML
    )
    retargeter.retargeting_config = {
        "end_effector_priority": {
            "effectors": {
                "left_hand": {
                    "robot_body": "left_hand",
                    "tracking_offset": [1.0, 2.0, 3.0],
                    "contact_offsets": [[4.0, 5.0, 6.0]],
                }
            }
        }
    }

    retargeter._infer_end_effector_geometry()

    spec = retargeter.retargeting_config["end_effector_priority"]["effectors"][
        "left_hand"
    ]
    assert spec["tracking_offset"] == [1.0, 2.0, 3.0]
    assert spec["contact_offsets"] == [[4.0, 5.0, 6.0]]


def test_calibrated_orientation_target_preserves_reference_frame_offset():
    source_reference = Rotation.from_euler("xyz", [20.0, -10.0, 5.0], degrees=True)
    robot_reference = Rotation.from_euler("xyz", [-30.0, 15.0, 40.0], degrees=True)
    source_delta = Rotation.from_euler("zy", [35.0, -12.0], degrees=True)
    source_current = source_delta.as_matrix() @ source_reference.as_matrix()

    target = _calibrated_orientation_target(
        source_current,
        source_reference.as_matrix(),
        robot_reference.as_matrix(),
    )

    np.testing.assert_allclose(
        target,
        source_delta.as_matrix() @ robot_reference.as_matrix(),
        atol=1e-10,
    )


def test_relative_orientation_step_uses_shortest_quaternion_arc():
    previous = Rotation.from_euler("z", 175.0, degrees=True).as_quat(
        scalar_first=True
    )
    current = Rotation.from_euler("z", -175.0, degrees=True).as_quat(
        scalar_first=True
    )

    step = _relative_orientation_step_magnitude(current, -previous)

    assert np.degrees(step) == pytest.approx(10.0)


def test_relative_orientation_step_projection_is_exact_on_so3():
    previous = Rotation.from_euler("xyz", [20.0, -15.0, 170.0], degrees=True)
    current = previous * Rotation.from_euler(
        "xyz", [25.0, -18.0, 30.0], degrees=True
    )

    projected = _limit_relative_orientation_step(
        -current.as_quat(scalar_first=True),
        previous.as_quat(scalar_first=True),
        np.deg2rad(7.0),
    )

    assert np.degrees(
        _relative_orientation_step_magnitude(
            projected, previous.as_quat(scalar_first=True)
        )
    ) == pytest.approx(7.0)
    assert np.dot(projected, previous.as_quat(scalar_first=True)) > 0.0


def test_base_orientation_step_limit_adds_bounded_catchup():
    previous_robot = Rotation.from_euler("z", -80.0, degrees=True).as_quat(
        scalar_first=True
    )
    previous_target = Rotation.identity().as_quat(scalar_first=True)
    target = Rotation.from_euler("z", 10.0, degrees=True).as_quat(
        scalar_first=True
    )

    allowed = _base_orientation_step_limit(
        target,
        previous_target,
        previous_robot,
        np.deg2rad(5.0),
        catchup_gain=0.35,
        catchup_deadband=np.deg2rad(20.0),
        max_extra_step=np.deg2rad(20.0),
        max_total_step=np.deg2rad(25.0),
    )

    assert np.degrees(allowed) == pytest.approx(25.0)


def test_base_orientation_step_limit_preserves_source_step_above_total_cap():
    previous_target = Rotation.identity().as_quat(scalar_first=True)
    target = Rotation.from_euler("z", 32.0, degrees=True).as_quat(
        scalar_first=True
    )

    allowed = _base_orientation_step_limit(
        target,
        previous_target,
        previous_target,
        np.deg2rad(5.0),
        max_total_step=np.deg2rad(25.0),
    )

    assert np.degrees(allowed) == pytest.approx(32.0)


def test_body_orientation_tasks_auto_follow_generic_joint_mapping():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(ANKLE_ARM_XML)
    retargeter = GenericInteractionRetargeter.__new__(
        GenericInteractionRetargeter
    )
    retargeter.robot_model = model
    retargeter.joint_mapping = {"Pelvis": "root", "L_Ankle": "foot"}
    retargeter.end_effector_config = {
        "body_orientation": {"_source_orientations_available": True}
    }

    retargeter._setup_body_orientations()

    assert retargeter.body_orientation_enabled
    assert set(retargeter.body_orientation_tasks) == {"Pelvis", "L_Ankle"}
    assert retargeter.body_orientation_tasks["Pelvis"]["body_name"] == "root"
    assert retargeter.body_orientation_tasks["L_Ankle"]["body_name"] == "foot"


def test_body_orientation_auto_mode_stays_off_for_positions_only_source():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(ANKLE_ARM_XML)
    retargeter = GenericInteractionRetargeter.__new__(
        GenericInteractionRetargeter
    )
    retargeter.robot_model = model
    retargeter.joint_mapping = {"Pelvis": "root", "L_Ankle": "foot"}
    retargeter.end_effector_config = {"body_orientation": {}}

    retargeter._setup_body_orientations()

    assert not retargeter.body_orientation_enabled
    assert retargeter.body_orientation_tasks == {}


def test_base_orientation_source_joint_is_not_tracked_twice():
    """A torso-base source frame must not also become a body-frame residual."""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(ANKLE_ARM_XML)
    retargeter = GenericInteractionRetargeter.__new__(
        GenericInteractionRetargeter
    )
    retargeter.robot_model = model
    retargeter.joint_mapping = {"Pelvis": "root", "L_Ankle": "foot"}
    retargeter.end_effector_config = {
        "_base_orientation_source_joint": "Pelvis",
        "body_orientation": {"_source_orientations_available": True},
    }

    retargeter._setup_body_orientations()

    assert set(retargeter.body_orientation_tasks) == {"L_Ankle"}


def test_world_tracking_tasks_preserve_positions_without_contact_semantics():
    positions = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, -0.2, 0.3], [0.2, -0.1, 0.4]]
    )

    task = build_world_tracking_tasks(
        {"right_foot": {"tracking_positions": positions}}
    )["right_foot"]

    np.testing.assert_allclose(task.tracking_positions, positions)
    assert not np.any(task.contact_mask)
    assert all(not task.frame(frame)["contact"] for frame in range(len(positions)))


def test_mixed_effectors_keep_feet_out_of_semantic_contact():
    marker = {"tracking_positions": np.zeros((2, 3))}
    markers = {
        "left_hand": marker,
        "right_hand": marker,
        "left_foot": marker,
        "right_foot": marker,
    }

    semantic, world = _partition_end_effector_markers(
        markers,
        {
            "semantic_contact_effectors": ["left_hand", "right_hand"],
            "world_tracking_effectors": ["left_foot", "right_foot"],
        },
        semantic_contact_enabled=True,
        world_tracking_enabled=True,
    )

    assert set(semantic) == {"left_hand", "right_hand"}
    assert set(world) == {"left_foot", "right_foot"}


def test_mixed_effectors_reject_overlapping_contact_modes():
    marker = {"tracking_positions": np.zeros((2, 3))}

    with pytest.raises(ValueError, match="at the same time"):
        _partition_end_effector_markers(
            {"left_foot": marker},
            {
                "semantic_contact_effectors": ["left_foot"],
                "world_tracking_effectors": ["left_foot"],
            },
            semantic_contact_enabled=True,
            world_tracking_enabled=True,
        )


def test_self_collision_objective_skips_queries_when_explicitly_disabled():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(ANKLE_ARM_XML)
    retargeter = GenericInteractionRetargeter.__new__(GenericInteractionRetargeter)
    retargeter.robot_model = model
    retargeter.robot_data = mujoco.MjData(model)
    retargeter.collision_detection_threshold = 0.1
    retargeter.self_collision_enabled = False
    retargeter._get_self_collision_query = lambda: (_ for _ in ()).throw(
        AssertionError("self-collision query must remain disabled")
    )
    retargeter._compute_terrain_penetration_constraints = (
        lambda _q, _dqa, _threshold: []
    )

    constraints = retargeter._compute_penetration_constraints(
        np.asarray(model.qpos0, dtype=float), object()
    )
    penalty = retargeter._self_collision_objective(
        np.asarray(model.qpos0, dtype=float), object(), constraints
    )

    assert constraints == []
    assert penalty.value == 0.0


def test_contact_frame_blends_tracking_position_into_anchor_during_transition():
    task = _task(
        np.asarray([[0.0, 0.0, 0.0]]),
        contact_mask=np.asarray([True]),
    )
    task = SceneContactTask(
        **{
            **task.__dict__,
            "tracking_positions": np.asarray([[0.0, 0.0, 0.0]]),
            "contact_positions": np.asarray([[1.0, 0.0, 0.0]]),
            "transition_weights": np.asarray([0.25]),
        }
    )

    frame = task.frame(0)

    np.testing.assert_allclose(frame["contact_position"], [0.25, 0.0, 0.0])
    assert frame["transition_weight"] == pytest.approx(0.25)


def test_foot_side_matching_uses_name_tokens_not_roll_link_substrings():
    assert OmniRetargeter._body_name_matches_side("left_ankle_roll_link", "left")
    assert not OmniRetargeter._body_name_matches_side(
        "right_ankle_roll_link", "left"
    )
    assert OmniRetargeter._body_name_matches_side("right_ankle_roll_link", "right")
    assert not OmniRetargeter._body_name_matches_side(
        "left_ankle_roll_link", "right"
    )


def test_densify_smplx_frame_gaps_restores_missing_pose_on_shortest_arc():
    frame_ids = np.asarray(["0070", "0072"])
    trans = np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]])
    root = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, np.pi / 2.0]])
    body = np.zeros((2, 63))
    body[1, :3] = [0.0, np.pi / 2.0, 0.0]
    hand = np.zeros((2, 90))
    hand[1, :3] = [np.pi / 2.0, 0.0, 0.0]

    dense = _densify_smplx_frame_gaps(frame_ids, trans, root, body, hand)
    dense_trans, dense_root, dense_body, dense_hand, dense_ids = dense

    assert dense_ids.tolist() == ["0070", "0071", "0072"]
    assert np.allclose(dense_trans[1], [0.1, 0.0, 0.0])
    assert np.allclose(dense_root[1], [0.0, 0.0, np.pi / 4.0])
    assert np.allclose(dense_body[1, :3], [0.0, np.pi / 4.0, 0.0])
    assert np.allclose(dense_hand[1, :3], [np.pi / 4.0, 0.0, 0.0])


def test_densify_smplx_frame_gaps_does_not_guess_across_large_gap():
    frame_ids = np.asarray(["0001", "0020"])
    trans = np.zeros((2, 3))
    root = np.zeros((2, 3))
    body = np.zeros((2, 63))

    dense = _densify_smplx_frame_gaps(frame_ids, trans, root, body)

    assert dense[0].shape[0] == 2
    assert dense[-1].tolist() == frame_ids.tolist()


def test_unit_scale_swing_target_still_follows_robot_root_registration():
    """Unit morphology scale must not suppress the robot root-height offset."""
    transformed = _transform_swing_target(
        tracking_position=np.array([1.2, -0.4, 1.7]),
        source_root_position=np.array([1.0, -0.5, 1.1]),
        robot_root_z_offset=-0.5,
        morphology_scale=1.0,
    )

    np.testing.assert_allclose(transformed, [1.2, -0.4, 1.2])


def test_swing_root_offset_is_measured_from_current_robot_and_source_roots():
    offset = _swing_target_z_offset(
        {},
        robot_root_z=0.82,
        source_root_position=np.array([0.1, -0.2, 1.08]),
    )

    assert offset == pytest.approx(-0.26)


def test_explicit_swing_root_offset_remains_an_override():
    offset = _swing_target_z_offset(
        {"swing_target_z_offset": -0.3},
        robot_root_z=0.82,
        source_root_position=np.array([0.1, -0.2, 1.08]),
    )

    assert offset == -0.3


def test_missing_requested_source_torso_uses_highest_available_spine():
    retargeter = OmniRetargeter.__new__(OmniRetargeter)
    retargeter.base_orientation_config = {"root_position": "Spine3"}
    retargeter.smplx_joint_indices = {"Pelvis": 0, "Spine1": 1, "Head": 2}

    assert retargeter._resolve_source_root_index() == 1


def test_swing_target_scales_root_relative_motion_after_registration():
    transformed = _transform_swing_target(
        tracking_position=np.array([1.4, -0.3, 1.7]),
        source_root_position=np.array([1.0, -0.5, 1.1]),
        robot_root_z_offset=-0.5,
        morphology_scale=0.5,
    )

    np.testing.assert_allclose(transformed, [1.2, -0.4, 0.9])


def test_world_space_swing_target_is_not_morphology_transformed():
    transformed = _transform_swing_target(
        tracking_position=np.array([1.4, -0.3, 1.7]),
        source_root_position=np.array([1.0, -0.5, 1.1]),
        robot_root_z_offset=-0.5,
        morphology_scale=0.5,
        preserve_world_target=True,
    )

    np.testing.assert_allclose(transformed, [1.4, -0.3, 1.7])


def test_swing_reach_clamp_preserves_reachable_target():
    target = np.array([0.3, 0.0, 0.0])
    result = _clamp_target_to_reach(target, np.zeros(3), 0.48)
    np.testing.assert_allclose(result, target)


def test_swing_reach_clamp_projects_only_excess_distance():
    target = np.array([0.6, 0.0, 0.0])
    result = _clamp_target_to_reach(target, np.zeros(3), 0.48)
    np.testing.assert_allclose(result, [0.48, 0.0, 0.0])


def test_joint_temporal_weights_match_robot_joint_name_substrings():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(ANKLE_ARM_XML)
    weights = _joint_temporal_weight_vector(
        model,
        np.array([7]),
        {"ankle": 6.0, "wrist": 3.0},
    )
    np.testing.assert_allclose(weights, [6.0])


def test_joint_posture_targets_match_robot_joint_name_substrings():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(ANKLE_ARM_XML)
    targets, sigmas = _joint_posture_target_vectors(
        model,
        np.array([7]),
        {"ankle": {"target": 0.15, "sigma": 0.2}},
    )
    np.testing.assert_allclose(targets, [0.15])
    np.testing.assert_allclose(sigmas, [0.2])


def test_joint_coordinate_mask_limits_orientation_to_named_joints():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(ANKLE_ARM_XML)
    mask = _joint_coordinate_mask(model, np.array([7]), ["wrist", "ankle"])
    np.testing.assert_array_equal(mask, [True])

    unmatched = _joint_coordinate_mask(model, np.array([7]), ["wrist"])
    np.testing.assert_array_equal(unmatched, [False])


def test_joint_bounds_are_inset_without_collapsing_narrow_ranges():
    lower, upper = _inset_finite_joint_bounds(
        np.array([-1.0, 0.0, -np.inf]),
        np.array([1.0, 0.1, np.inf]),
        margin=0.2,
    )
    np.testing.assert_allclose(lower[:2], [-0.8, 0.025])
    np.testing.assert_allclose(upper[:2], [0.8, 0.075])
    assert np.isneginf(lower[2])
    assert np.isposinf(upper[2])

    ratio_lower, ratio_upper = _inset_finite_joint_bounds(
        np.array([-1.0, 0.0]),
        np.array([1.0, 0.5]),
        margin=0.0,
        margin_ratio=0.08,
    )
    np.testing.assert_allclose(ratio_lower, [-0.84, 0.04])
    np.testing.assert_allclose(ratio_upper, [0.84, 0.46])


def test_local_semantic_rotation_maps_g1_hand_axes():
    rotation = _local_semantic_rotation(
        forward_direction=np.array([1.0, 0.0, 0.0]),
        up_direction=np.array([0.0, -1.0, 0.0]),
    )
    np.testing.assert_allclose(rotation[:, 0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(rotation[:, 1], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(rotation[:, 2], [0.0, -1.0, 0.0])
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_axis_orientation_jacobian_matches_small_world_rotation():
    direction = np.array([0.3, -0.4, 0.5])
    direction /= np.linalg.norm(direction)
    delta = np.array([2e-6, -3e-6, 4e-6])
    predicted = direction - _skew_symmetric(direction) @ delta
    actual = Rotation.from_rotvec(delta).apply(direction)
    np.testing.assert_allclose(predicted, actual, atol=2e-11)


def test_configured_joint_limits_intersect_mechanical_range():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(ANKLE_ARM_XML)
    lower, upper = _apply_configured_joint_limits(
        model,
        np.array([7]),
        np.array([-0.6]),
        np.array([0.6]),
        {"ankle": [-0.25, 0.4]},
    )
    np.testing.assert_allclose(lower, [-0.25])
    np.testing.assert_allclose(upper, [0.4])


def test_configured_joint_regularization_uses_exact_joint_name():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(ANKLE_ARM_XML)
    qpos_idx = int(model.jnt_qposadr[model.joint("ankle").id])
    weights = _apply_configured_joint_regularization(
        model,
        np.asarray([qpos_idx]),
        np.asarray([0.001]),
        {"ankle": 0.5},
    )

    np.testing.assert_allclose(weights, [0.5])


def test_waist_uses_mechanical_range_without_implicit_neutral_bias():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(AUTO_END_EFFECTOR_GEOMETRY_XML)
    solver = GenericInteractionRetargeter(
        model, mujoco.MjData(model), trimesh.creation.box(),
        {"source_root": "root", "source_core": "torso"}, 1.0,
        terrain_sample_seed=0,
    )
    joint = model.joint("waist_yaw_joint")
    coordinate = int(np.flatnonzero(solver.q_a_indices == model.jnt_qposadr[joint.id])[0])
    np.testing.assert_allclose(
        [solver.q_a_lb[coordinate], solver.q_a_ub[coordinate]], model.jnt_range[joint.id]
    )
    assert solver.Q_diag[coordinate] == pytest.approx(0.001)


def test_joint_limit_safety_fraction_keeps_last_twenty_percent_unused():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(ANKLE_ARM_XML)
    qpos_idx = int(model.jnt_qposadr[model.joint("ankle").id])
    lower, upper = _apply_joint_limit_safety_fraction(
        model,
        np.asarray([qpos_idx]),
        np.asarray([-0.6]),
        np.asarray([0.6]),
        0.8,
    )

    np.testing.assert_allclose(lower, [-0.48])
    np.testing.assert_allclose(upper, [0.48])


def test_transported_arm_segments_preserve_shape_and_aim_chord_at_hand():
    upper, forearm = _transported_arm_segment_targets(
        source_shoulder=np.array([0.0, 0.0, 0.0]),
        source_elbow=np.array([1.0, 0.0, 0.0]),
        source_hand=np.array([1.0, 1.0, 0.0]),
        robot_shoulder=np.array([0.0, 0.0, 0.0]),
        robot_hand_target=np.array([0.0, 3.0, 0.0]),
        upper_arm_length=0.4,
        forearm_length=0.6,
    )

    np.testing.assert_allclose(np.linalg.norm(upper), 0.4, atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(forearm), 0.6, atol=1e-8)
    np.testing.assert_allclose(np.dot(upper, forearm), 0.0, atol=1e-8)
    chord = upper + forearm
    np.testing.assert_allclose(chord[0], 0.0, atol=1e-8)
    assert chord[1] > 0.0


def test_position_only_loader_can_return_end_effector_markers(tmp_path):
    joints = np.zeros((3, 22, 3), dtype=float)
    path = tmp_path / "joints.npy"
    np.save(path, joints)

    loaded_joints, orientations, markers = load_smplx_trajectory(
        path, return_end_effectors=True
    )

    np.testing.assert_array_equal(loaded_joints, joints)
    assert orientations is None
    assert set(markers) == {"left_hand", "right_hand", "left_foot", "right_foot"}
    assert markers["left_hand"]["contact_samples"].shape == (3, 1, 3)
