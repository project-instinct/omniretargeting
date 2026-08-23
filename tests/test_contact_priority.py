import json
from pathlib import Path

import cvxpy as cp
import numpy as np
import pytest
import trimesh
from dataclasses import replace
from scipy.spatial.transform import Rotation

from omniretargeting.contact import (
    ContactMode,
    SceneContactTask,
    infer_scene_contact_tasks,
    stack_contact_tasks,
)
from omniretargeting.retargeting import (
    GenericInteractionRetargeter,
    _apply_configured_joint_limits,
    _build_base_orientation_task,
    _bridge_contact_orientation,
    _clamp_target_to_reach,
    _bridge_swing_to_contact,
    _minimum_jerk_blend,
    _resolve_position_sigma,
    _resolve_priority_levels,
    _solve_configured_priority_levels,
    _swivel_direction_and_jacobian,
    _swivel_tangent_delta,
    _transform_swing_target,
)
import omniretargeting.retargeting as retargeting_module
import omniretargeting.trajectory_refinement as trajectory_refinement_module
from omniretargeting.core import (
    _build_per_frame_sigma_scales,
    _build_robot_base_reference,
)
from omniretargeting.trajectory_refinement import (
    _joint_coordinate_mask,
    _inset_finite_joint_bounds,
    _joint_posture_target_vectors,
    _joint_temporal_weight_vector,
    _local_semantic_rotation,
    _skew_symmetric,
    _transported_arm_segment_targets,
    collision_geom_ids,
    geom_surface_offsets,
    measure_scene_penetration,
    refine_joint_trajectory,
    resolve_scene_normal_orientation,
    scene_signed_clearance,
)
from omniretargeting.utils import _densify_smplx_frame_gaps, load_smplx_trajectory


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


def _straight_arm_setup(num_frames=12, amplitude=0.12):
    """Return a model, a chattering trajectory and a fixed endpoint task."""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(ANKLE_ARM_XML)
    trajectory = np.tile(model.qpos0, (num_frames, 1))
    trajectory[:, 3] = 1.0
    trajectory[:, 7] = amplitude * (-1.0) ** np.arange(num_frames)
    targets = np.tile([1.0, 0.0, 0.0], (num_frames, 1))
    return model, trajectory, targets


def _task(targets, contact_modes=None):
    num_frames = len(targets)
    if contact_modes is None:
        contact_modes = np.full(num_frames, int(ContactMode.NONE), dtype=np.uint8)
    contact_modes = np.asarray(contact_modes, dtype=np.uint8)
    assert contact_modes.shape == (num_frames,)
    return SceneContactTask(
        name="left_foot",
        tracking_positions=targets,
        contact_positions=targets,
        surface_points=targets,
        surface_normals=np.tile([0.0, 0.0, 1.0], (num_frames, 1)),
        distances=np.zeros(num_frames),
        relative_speeds=np.zeros(num_frames),
        contact_confidence=np.ones(num_frames),
        forward_directions=np.tile([1.0, 0.0, 0.0], (num_frames, 1)),
        up_directions=np.tile([0.0, 0.0, 1.0], (num_frames, 1)),
        contact_modes=contact_modes,
        contact_segment_ids=np.where(
            contact_modes == int(ContactMode.NONE), -1, 0
        ),
        transition_weights=np.ones(num_frames),
        effective_swing_positions=np.asarray(targets, dtype=float).copy(),
    )


def _effector_config(contact_offsets=None, track_orientation=True):
    return {
        "effectors": {
            "left_foot": {
                "robot_body": "foot",
                "tracking_offset": [1.0, 0.0, 0.0],
                "contact_offsets": contact_offsets or [[1.0, 0.0, 0.0]],
                "contact_target_space": "world",
                "swing_target_space": "root_morphology_scaled",
                "track_orientation": track_orientation,
                "orientation_sigma": 0.1,
            }
        }
    }


def test_unit_scale_swing_target_still_follows_robot_root_registration():
    """Unit morphology scale must not suppress the robot root-height offset."""
    transformed = _transform_swing_target(
        tracking_position=np.array([1.2, -0.4, 1.7]),
        source_anchor_position=np.array([1.0, -0.5, 1.1]),
        robot_anchor_position=np.array([1.0, -0.5, 0.6]),
        morphology_scale=1.0,
        swing_target_space="root_morphology_scaled",
    )

    np.testing.assert_allclose(transformed, [1.2, -0.4, 1.2])


def test_swing_target_scales_root_relative_motion_after_registration():
    transformed = _transform_swing_target(
        tracking_position=np.array([1.4, -0.3, 1.7]),
        source_anchor_position=np.array([1.0, -0.5, 1.1]),
        robot_anchor_position=np.array([1.0, -0.5, 0.6]),
        morphology_scale=0.5,
        swing_target_space="root_morphology_scaled",
    )

    np.testing.assert_allclose(transformed, [1.2, -0.4, 0.9])


def test_world_swing_target_is_not_morphology_scaled_or_root_shifted():
    """Registered source world targets remain immutable even if unreachable."""
    transformed = _transform_swing_target(
        tracking_position=np.array([1.4, -0.3, 1.7]),
        source_anchor_position=np.array([1.0, -0.5, 1.1]),
        robot_anchor_position=np.array([1.0, -0.5, 0.6]),
        morphology_scale=0.5,
        swing_target_space="world",
    )
    np.testing.assert_allclose(transformed, [1.4, -0.3, 1.7])


def test_g1_profile_gates_world_tracking_on_contact_and_on_the_feet():
    """Feet stay world-exact; free hands follow the source arm proportionally."""
    profile_path = (
        Path(__file__).resolve().parents[1]
        / "robot_models"
        / "unitree_g1"
        / "unitree_g1_instinctmj.json"
    )
    with profile_path.open(encoding="utf-8") as profile_file:
        profile = json.load(profile_file)
    priority = profile["retargeting"]["end_effector_priority"]
    assert priority["root_warp"]["enabled"] is False
    assert priority["lock_base_orientation"] is False
    assert ["Spine3", "Pelvis"] in priority["pose_direction"]["edges"]
    for name, spec in priority["effectors"].items():
        # Every contact is a real scene interaction and stays in world metres.
        assert spec["contact_target_space"] == "world"
        if name.endswith("foot"):
            assert spec["swing_target_space"] == "world"
            assert spec["swing_morphology_scale"] == pytest.approx(1.0)
        else:
            assert spec["swing_target_space"] == "limb_morphology_scaled"
            assert spec["swing_morphology_scale"] == "auto"
            # The limb base must resolve, otherwise the anchor is undefined.
            assert spec["reach_reference_body"]


def test_limb_swing_target_keeps_source_direction_and_shortens_the_limb():
    """A limb-anchored swing target only rescales the source limb vector."""
    source_shoulder = np.array([1.0, -0.5, 1.4])
    source_hand = np.array([1.6, -0.5, 1.4])
    robot_shoulder = np.array([1.0, -0.5, 0.9])

    transformed = _transform_swing_target(
        tracking_position=source_hand,
        source_anchor_position=source_shoulder,
        robot_anchor_position=robot_shoulder,
        morphology_scale=0.5,
        swing_target_space="limb_morphology_scaled",
    )

    np.testing.assert_allclose(transformed, [1.3, -0.5, 0.9])
    source_limb = source_hand - source_shoulder
    robot_limb = transformed - robot_shoulder
    # Same direction, half the length.
    np.testing.assert_allclose(
        robot_limb / np.linalg.norm(robot_limb),
        source_limb / np.linalg.norm(source_limb),
    )
    assert np.linalg.norm(robot_limb) == pytest.approx(
        0.5 * np.linalg.norm(source_limb)
    )


def test_unknown_swing_target_space_is_rejected():
    with pytest.raises(ValueError, match="swing_target_space"):
        _transform_swing_target(
            tracking_position=np.zeros(3),
            source_anchor_position=np.zeros(3),
            robot_anchor_position=np.zeros(3),
            morphology_scale=1.0,
            swing_target_space="pelvis_relative",
        )


def test_stage_specific_position_sigmas_override_legacy_keys():
    spec = {
        "contact_sigma": 0.09,
        "contact_position_sigma": 0.08,
        "framewise_contact_position_sigma": 0.02,
        "refinement_contact_position_sigma": 0.012,
    }
    assert _resolve_position_sigma(
        spec, {}, "contact", "framewise", 0.025
    ) == pytest.approx(0.02)
    assert _resolve_position_sigma(
        spec, {}, "contact", "refinement", 0.025
    ) == pytest.approx(0.012)


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


def test_swivel_residual_jacobian_matches_points_and_fades_when_straight():
    shoulder = np.array([0.1, -0.2, 0.3])
    elbow = np.array([0.9, 0.35, 0.55])
    wrist = np.array([1.8, -0.05, 0.15])
    shoulder_jacobian = np.zeros((3, 9))
    elbow_jacobian = np.zeros((3, 9))
    wrist_jacobian = np.zeros((3, 9))
    shoulder_jacobian[:, :3] = np.eye(3)
    elbow_jacobian[:, 3:6] = np.eye(3)
    wrist_jacobian[:, 6:] = np.eye(3)
    direction, analytic, fade, _, axis = _swivel_direction_and_jacobian(
        shoulder,
        elbow,
        wrist,
        shoulder_jacobian,
        elbow_jacobian,
        wrist_jacobian,
        min_sine=0.02,
        full_sine=0.2,
    )
    assert fade == pytest.approx(1.0)
    assert analytic is not None

    points = np.concatenate([shoulder, elbow, wrist])
    finite_difference = np.zeros_like(analytic)
    epsilon = 1e-6
    for coordinate in range(len(points)):
        plus = points.copy()
        minus = points.copy()
        plus[coordinate] += epsilon
        minus[coordinate] -= epsilon
        plus_direction = _swivel_direction_and_jacobian(
            plus[:3], plus[3:6], plus[6:]
        )[0]
        minus_direction = _swivel_direction_and_jacobian(
            minus[:3], minus[3:6], minus[6:]
        )[0]
        finite_difference[:, coordinate] = (
            plus_direction - minus_direction
        ) / (2.0 * epsilon)
    np.testing.assert_allclose(analytic, finite_difference, atol=2e-6)

    target = np.array([0.0, 0.0, 1.0])
    target -= axis * float(np.dot(target, axis))
    target /= np.linalg.norm(target)
    tangent_delta = _swivel_tangent_delta(direction, target, axis)
    expected_angle = np.arccos(
        np.clip(np.dot(direction, target), -1.0, 1.0)
    )
    assert np.linalg.norm(tangent_delta) == pytest.approx(expected_angle)
    assert abs(float(np.dot(tangent_delta, direction))) < 1e-10

    straight_direction, straight_jacobian, straight_fade, _, _ = (
        _swivel_direction_and_jacobian(
            np.zeros(3),
            np.array([1.0, 1e-9, 0.0]),
            np.array([2.0, 0.0, 0.0]),
            shoulder_jacobian,
            elbow_jacobian,
            wrist_jacobian,
            min_sine=0.05,
            full_sine=0.35,
        )
    )
    assert np.all(np.isfinite(straight_direction))
    assert straight_fade == pytest.approx(0.0)
    np.testing.assert_array_equal(straight_jacobian, np.zeros((3, 9)))


def test_support_and_slide_orientation_use_scene_normal_and_source_tangent(
    monkeypatch,
):
    """Contact modes pin the palm normal; source forward supplies heading only."""
    cvxpy = pytest.importorskip("cvxpy")
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(ANKLE_ARM_XML)
    data = mujoco.MjData(model)
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)
    terrain = trimesh.creation.box(extents=[2.0, 2.0, 0.1])
    retargeter = GenericInteractionRetargeter(
        model,
        data,
        terrain,
        {"LeftFoot": "foot"},
        robot_height=1.0,
        terrain_sample_points=4,
        valid_joint_names=["LeftFoot"],
        end_effector_config={
            "enabled": True,
            "hierarchy": True,
            "slide_tangential_weight": 0.0,
            "effectors": {
                "left_foot": {
                    "robot_body": "foot",
                    "tracking_offset": [1.0, 0.0, 0.0],
                    "contact_offsets": [[1.0, 0.0, 0.0]],
                    "contact_target_space": "world",
                    "swing_target_space": "root_morphology_scaled",
                    "track_orientation": True,
                }
            },
        },
    )

    captured = []
    scipy_rotation = retargeting_module.Rotation

    class RotationProbe:
        @staticmethod
        def from_matrix(matrix):
            captured.append(np.asarray(matrix, dtype=float).copy())
            return scipy_rotation.from_matrix(matrix)

    monkeypatch.setattr(retargeting_module, "Rotation", RotationProbe)
    dqa = cvxpy.Variable(retargeter.nq_a)
    expected_heading = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    for mode in (ContactMode.SUPPORT, ContactMode.SLIDE):
        values = retargeter._build_end_effector_objectives(
            dqa,
            {
                "left_foot": {
                    "contact_mode": mode.name,
                    "contact_segment_id": int(mode),
                    "contact_position": [1.0, 0.0, 0.0],
                    "surface_normal": [0.0, 0.0, 1.0],
                    # Deliberately conflicting source up: SUPPORT/SLIDE must
                    # use the scene normal instead.
                    "up_direction": [1.0, 0.0, 0.0],
                    "forward_direction": [1.0, 1.0, 1.0],
                    "confidence": 1.0,
                    "transition_weight": 0.2,
                }
            },
        )
        assert values[3] is not None

    assert len(captured) == 2
    for target_rotation in captured:
        np.testing.assert_allclose(target_rotation[:, 2], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(target_rotation[:, 0], expected_heading)
        np.testing.assert_allclose(
            target_rotation.T @ target_rotation, np.eye(3), atol=1e-12
        )


def test_framewise_and_refinement_consume_the_same_rotation_matrix(monkeypatch):
    """Prepared tasks bypass both solvers' legacy axis reconstruction paths."""
    cvxpy = pytest.importorskip("cvxpy")
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(ANKLE_ARM_XML)
    trajectory = np.tile(model.qpos0, (4, 1))
    trajectory[:, 3] = 1.0
    modes = np.full(4, int(ContactMode.SUPPORT), dtype=np.uint8)
    target_rotation = Rotation.from_euler("y", 0.2).as_matrix()
    target_track = np.repeat(target_rotation[None, :, :], 4, axis=0)
    task = replace(
        _task(np.tile([1.0, 0.0, 0.0], (4, 1)), modes),
        orientation_target_rotations=target_track,
        orientation_bridge_alpha=np.ones(4),
    )
    np.testing.assert_array_equal(
        task.frame(0)["orientation_target_rotation"], target_rotation
    )

    data = mujoco.MjData(model)
    data.qpos[:] = trajectory[0]
    mujoco.mj_forward(model, data)
    retargeter = GenericInteractionRetargeter(
        model,
        data,
        trimesh.creation.box(extents=[2.0, 2.0, 0.1]),
        {"LeftFoot": "foot"},
        robot_height=1.0,
        terrain_sample_points=4,
        valid_joint_names=["LeftFoot"],
        end_effector_config={
            "enabled": True,
            "effectors": {
                "left_foot": {
                    "robot_body": "foot",
                    "tracking_offset": [1.0, 0.0, 0.0],
                    "contact_offsets": [[1.0, 0.0, 0.0]],
                    "contact_target_space": "world",
                    "swing_target_space": "world",
                    "track_orientation": True,
                }
            },
        },
    )

    def legacy_reconstruction_must_not_run(*_args, **_kwargs):
        raise AssertionError("legacy forward/up reconstruction was used")

    monkeypatch.setattr(
        retargeting_module,
        "_orthonormal_target_rotation",
        legacy_reconstruction_must_not_run,
    )
    dqa = cvxpy.Variable(retargeter.nq_a)
    framewise = retargeter._build_end_effector_objectives(
        dqa, {"left_foot": task.frame(0)}
    )
    assert framewise[3] is not None

    refinement_reconstruction = (
        trajectory_refinement_module._orthonormal_target_rotation
    )
    reconstruction_calls = 0

    def allow_local_frame_setup_only(*args, **kwargs):
        nonlocal reconstruction_calls
        reconstruction_calls += 1
        if reconstruction_calls > 1:
            raise AssertionError("task forward/up reconstruction was used")
        return refinement_reconstruction(*args, **kwargs)

    monkeypatch.setattr(
        trajectory_refinement_module,
        "_orthonormal_target_rotation",
        allow_local_frame_setup_only,
    )
    refined = refine_joint_trajectory(
        model,
        trajectory,
        {"left_foot": task},
        _effector_config(track_orientation=True),
        {"iterations": 1, "trust_region": 0.1},
    )
    assert np.all(np.isfinite(refined))
    assert reconstruction_calls == 1


def test_g1_hand_semantic_local_up_axes_are_mirrored_by_side():
    profile_path = (
        Path(__file__).resolve().parents[1]
        / "robot_models"
        / "unitree_g1"
        / "unitree_g1_instinctmj.json"
    )
    with profile_path.open(encoding="utf-8") as profile_file:
        profile = json.load(profile_file)
    effectors = profile["retargeting"]["end_effector_priority"]["effectors"]
    np.testing.assert_allclose(
        effectors["left_hand"]["orientation_local_up"], [0.0, -1.0, 0.0]
    )
    np.testing.assert_allclose(
        effectors["right_hand"]["orientation_local_up"], [0.0, 1.0, 0.0]
    )


def test_framewise_support_position_is_separate_from_orientation_slack():
    """Palm orientation/normal relaxation must not demote SUPPORT position."""
    cvxpy = pytest.importorskip("cvxpy")
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(ANKLE_ARM_XML)
    data = mujoco.MjData(model)
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)
    terrain = trimesh.creation.box(extents=[2.0, 2.0, 0.1])
    retargeter = GenericInteractionRetargeter(
        model,
        data,
        terrain,
        {"LeftFoot": "foot"},
        robot_height=1.0,
        terrain_sample_points=4,
        valid_joint_names=["LeftFoot"],
        end_effector_config={
            "enabled": True,
            "hierarchy": True,
            "slide_tangential_weight": 0.0,
            "effectors": {
                "left_foot": {
                    "robot_body": "foot",
                    "tracking_offset": [1.0, 0.0, 0.0],
                    "contact_offsets": [[1.0, 0.0, 0.0]],
                    "contact_target_space": "world",
                    "swing_target_space": "root_morphology_scaled",
                    "track_orientation": True,
                }
            },
        },
    )
    dqa = cvxpy.Variable(retargeter.nq_a)
    payload = {
        "left_foot": {
            "contact_mode": "SUPPORT",
            "contact_segment_id": 0,
            "contact_position": [1.0, 0.0, 0.0],
            "surface_normal": [0.0, 0.0, 1.0],
            "forward_direction": [1.0, 0.0, 0.0],
            "up_direction": [0.0, 0.0, 1.0],
            "confidence": 0.1,
            "transition_weight": 0.1,
        }
    }
    values = retargeter._build_end_effector_objectives(dqa, payload)
    support, contact, swing, orientation, constraints = values[:5]
    assert support is not None
    assert contact is None
    assert swing is None
    assert orientation is not None
    problem = cvxpy.Problem(
        cvxpy.Minimize(cvxpy.norm(support, 2)),
        constraints + [cvxpy.norm(dqa, 2) <= 0.2],
    )
    problem.solve(solver=cvxpy.CLARABEL)
    assert problem.status in (cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE)
    assert float(cvxpy.norm(support, 2).value) < 1e-6


def test_framewise_modes_split_support_touch_and_swing_objectives():
    """The mode enum selects exactly one positional priority level per frame."""
    cvxpy = pytest.importorskip("cvxpy")
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(ANKLE_ARM_XML)
    data = mujoco.MjData(model)
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)
    terrain = trimesh.creation.box(extents=[2.0, 2.0, 0.1])
    retargeter = GenericInteractionRetargeter(
        model,
        data,
        terrain,
        {"LeftFoot": "foot"},
        robot_height=1.0,
        terrain_sample_points=4,
        valid_joint_names=["LeftFoot"],
        end_effector_config={
            "enabled": True,
            "hierarchy": True,
            "slide_tangential_weight": 0.0,
            "effectors": {
                "left_foot": {
                    "robot_body": "foot",
                    "tracking_offset": [1.0, 0.0, 0.0],
                    "contact_offsets": [[1.0, 0.0, 0.0]],
                    "contact_target_space": "world",
                    "swing_target_space": "root_morphology_scaled",
                }
            },
        },
    )
    dqa = cvxpy.Variable(retargeter.nq_a)
    common = {
        "contact_segment_id": 0,
        "contact_position": [1.0, 0.0, 0.0],
        "tracking_position": [1.0, 0.0, 0.0],
        "surface_normal": [0.0, 0.0, 1.0],
        "confidence": 1.0,
    }
    support = retargeter._build_end_effector_objectives(
        dqa, {"left_foot": {**common, "contact_mode": "SUPPORT"}}
    )
    touch = retargeter._build_end_effector_objectives(
        dqa, {"left_foot": {**common, "contact_mode": "TOUCH"}}
    )
    slide = retargeter._build_end_effector_objectives(
        dqa, {"left_foot": {**common, "contact_mode": "SLIDE"}}
    )
    swing = retargeter._build_end_effector_objectives(
        dqa,
        {
            "left_foot": {
                "tracking_position": [1.0, 0.0, 0.0],
                "contact_mode": "NONE",
                "confidence": 1.0,
                "effective_swing_position": [1.0, 0.0, 0.0],
            }
        },
    )
    assert support[0] is not None and support[1] is None and support[2] is None
    assert touch[0] is None and touch[1] is not None and touch[2] is None
    assert slide[0] is not None and slide[0].size == 1
    assert slide[1] is None and slide[2] is None
    assert swing[0] is None and swing[1] is None and swing[2] is not None


def test_frame_payload_uses_contact_mode_without_legacy_contact_boolean():
    targets = np.tile([0.0, 0.0, 0.01], (2, 1))
    task = _task(targets, np.asarray([ContactMode.TOUCH, ContactMode.NONE]))
    frame = task.frame(0)
    assert frame["contact_mode"] == ContactMode.TOUCH.name
    assert frame["contact_segment_id"] == 0
    assert "contact" not in frame


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


def test_scene_contact_inference_filters_and_anchors_contact_runs():
    scene = trimesh.creation.box(extents=[4.0, 4.0, 0.1])
    scene.apply_translation([0.0, 0.0, -0.05])

    tracking = np.array(
        [
            [0.0, 0.0, 0.30],
            [0.0, 0.0, 0.20],
            [0.0, 0.0, 0.01],
            [0.0, 0.0, 0.01],
            [0.0, 0.0, 0.01],
            [0.0, 0.0, 0.01],
            [0.0, 0.0, 0.01],
            [0.0, 0.0, 0.01],
            [0.0, 0.0, 0.20],
            [0.0, 0.0, 0.30],
        ]
    )
    markers = {
        "left_foot": {
            "tracking_positions": tracking,
            "contact_samples": tracking[:, None, :],
        }
    }

    task = infer_scene_contact_tasks(
        markers,
        scene,
        framerate=30.0,
        config={
            "distance_threshold": 0.03,
            "relative_speed_threshold": 0.1,
            "min_contact_frames": 3,
            "max_gap_frames": 0,
            "effectors": {"left_foot": {"contact_mode": "SUPPORT"}},
        },
    )["left_foot"]

    assert np.array_equal(
        task.contact_modes != int(ContactMode.NONE),
        np.array([False, False, False, True, True, True, True, True, False, False]),
    )
    anchored = task.contact_positions[task.contact_modes != int(ContactMode.NONE)]
    np.testing.assert_allclose(
        anchored, np.repeat(anchored[0:1], len(anchored), axis=0), atol=1e-10
    )
    np.testing.assert_allclose(anchored[:, 2], 0.0, atol=1e-10)


def test_contact_inference_allows_sliding_hand_without_moving_foot_policy():
    scene = trimesh.creation.box(extents=[4.0, 4.0, 0.1])
    scene.apply_translation([0.0, 0.0, -0.05])
    tracking = np.column_stack(
        [np.linspace(0.0, 0.3, 6), np.zeros(6), np.full(6, 0.01)]
    )
    tasks = infer_scene_contact_tasks(
        {
            name: {
                "tracking_positions": tracking,
                "contact_samples": tracking[:, None, :],
            }
            for name in ("left_hand", "left_foot")
        },
        scene,
        framerate=30.0,
        config={
            "distance_threshold": 0.03,
            "relative_speed_threshold": 10.0,
            "min_contact_frames": 1,
            "transition_frames": 2,
            "effectors": {
                "left_hand": {"contact_mode": "SLIDE"},
                "left_foot": {"contact_mode": "SUPPORT"},
            },
        },
    )

    hand_targets = tasks["left_hand"].contact_positions
    foot_targets = tasks["left_foot"].contact_positions
    assert np.all(tasks["left_hand"].contact_modes == int(ContactMode.SLIDE))
    assert np.all(tasks["left_foot"].contact_modes == int(ContactMode.SUPPORT))
    assert tasks["left_hand"].transition_weights[0] < 1.0
    assert tasks["left_hand"].frame(0)["transition_weight"] < 1.0
    np.testing.assert_allclose(tasks["left_foot"].transition_weights, 1.0)
    assert np.ptp(hand_targets[:, 0]) > 0.25
    np.testing.assert_allclose(
        foot_targets,
        np.repeat(foot_targets[0:1], len(foot_targets), axis=0),
        atol=1e-10,
    )


def test_vault_support_anchor_bridge_and_local_robot_root_warp():
    """A vault hand becomes SUPPORT without altering source or scene data."""
    scene = trimesh.creation.box(extents=[4.0, 4.0, 0.1])
    scene.apply_translation([0.0, 0.0, -0.05])
    tracking = np.tile([0.55, 0.0, 0.01], (12, 1)).astype(float)
    tracking[:4, 2] = [0.30, 0.20, 0.10, 0.01]
    tracking[9:, 2] = [0.10, 0.20, 0.30]
    task = infer_scene_contact_tasks(
        {
            "left_hand": {
                "tracking_positions": tracking,
                "contact_samples": tracking[:, None, :],
            }
        },
        scene,
        framerate=30.0,
        config={
            "distance_threshold": 0.03,
            "relative_speed_threshold": 1.0,
            "min_contact_frames": 2,
            "max_gap_frames": 0,
            "transition_frames": 2,
            "effectors": {
                "left_hand": {
                    "contact_mode": "AUTO",
                    "support_speed_threshold": 0.1,
                    "support_max_tangential_travel": 0.02,
                }
            },
        },
    )["left_hand"]

    support = task.contact_modes == int(ContactMode.SUPPORT)
    assert np.count_nonzero(support) >= 4
    np.testing.assert_allclose(
        task.contact_positions[support],
        np.repeat(task.contact_positions[support][0:1], np.count_nonzero(support), axis=0),
        atol=1e-10,
    )
    # SUPPORT cores are never faded; approach/release continuity is supplied
    # by the explicit target bridge instead.
    np.testing.assert_allclose(task.transition_weights[support], 1.0)

    swing = tracking.copy()
    swing[:, 0] -= 0.25
    bridged, bridge_alpha = _bridge_swing_to_contact(
        swing,
        task.contact_positions,
        task.contact_modes,
        task.contact_segment_ids,
        transition_frames=3,
    )
    np.testing.assert_allclose(bridged[support], task.contact_positions[support])
    np.testing.assert_allclose(bridge_alpha[support], 1.0)
    onset = int(np.flatnonzero(support)[0])
    raw_jump = np.linalg.norm(swing[onset - 1] - task.contact_positions[onset])
    bridged_jump = np.linalg.norm(bridged[onset - 1] - bridged[onset])
    assert bridged_jump < raw_jump
    assert 0.0 < bridge_alpha[onset - 1] < 1.0
    np.testing.assert_allclose(bridged[0], swing[0])
    np.testing.assert_allclose(bridged[-1], swing[-1])

    human_root = np.zeros((len(tracking), 3), dtype=float)
    robot_reference, unwarped = _build_robot_base_reference(
        human_root,
        {"left_hand": task},
        {
            "left_hand": {
                "origin_offset": np.zeros(3),
                "max_reach": 0.47,
            }
        },
        {
            "enabled": True,
            "approach_frames": 2,
            "release_frames": 2,
            "max_xyz": [0.08, 0.08, 0.05],
            "max_norm": 0.08,
        },
        root_z_offset=0.0,
    )
    warp = np.linalg.norm(robot_reference - unwarped, axis=1)
    assert 0.0 < np.max(warp) <= 0.080001
    assert warp[0] == pytest.approx(0.0)
    assert warp[-1] == pytest.approx(0.0)
    np.testing.assert_array_equal(human_root, np.zeros_like(human_root))

    slide_task = replace(
        task,
        contact_modes=np.where(
            task.contact_modes == int(ContactMode.NONE),
            int(ContactMode.NONE),
            int(ContactMode.SLIDE),
        ),
    )
    slide_default, slide_unwarped = _build_robot_base_reference(
        human_root,
        {"left_hand": slide_task},
        {
            "left_hand": {
                "origin_offset": np.zeros(3),
                "max_reach": 0.47,
            }
        },
        {"enabled": True},
        root_z_offset=0.0,
    )
    np.testing.assert_allclose(slide_default, slide_unwarped)

    slide_enabled, slide_unwarped = _build_robot_base_reference(
        human_root,
        {"left_hand": slide_task},
        {
            "left_hand": {
                "origin_offset": np.zeros(3),
                "max_reach": 0.47,
            }
        },
        {"enabled": True, "qualifying_contact_modes": ["SUPPORT", "SLIDE"]},
        root_z_offset=0.0,
    )
    assert np.max(np.linalg.norm(slide_enabled - slide_unwarped, axis=1)) > 0.0


_BRIDGE_ANCHOR_A = np.array([0.30, 0.10, 0.95])
_BRIDGE_ANCHOR_B = np.array([0.95, -0.05, 0.80])
_BRIDGE_CORE_FRAMES = 6
_BRIDGE_LEAD_FRAMES = 8


def _two_contact_bridge_case(gap_frames, core_frames=_BRIDGE_CORE_FRAMES):
    """Contact A, a real free gap, then contact B on a clearly different anchor.

    The free swing path is a lifted quintic arc, so the honest free-gap target
    is neither anchor and never lies on the straight A-to-B chord.
    """
    start_a = _BRIDGE_LEAD_FRAMES
    end_a = start_a + core_frames
    start_b = end_a + gap_frames
    end_b = start_b + core_frames
    num_frames = end_b + _BRIDGE_LEAD_FRAMES

    modes = np.full(num_frames, int(ContactMode.NONE), dtype=np.uint8)
    segment_ids = np.full(num_frames, -1, dtype=np.int32)
    contact = np.zeros((num_frames, 3), dtype=float)
    modes[start_a:end_a] = int(ContactMode.SUPPORT)
    segment_ids[start_a:end_a] = 0
    contact[start_a:end_a] = _BRIDGE_ANCHOR_A
    modes[start_b:end_b] = int(ContactMode.SUPPORT)
    segment_ids[start_b:end_b] = 1
    contact[start_b:end_b] = _BRIDGE_ANCHOR_B

    swing = np.zeros((num_frames, 3), dtype=float)
    progress = np.clip(
        (np.arange(num_frames) - (end_a - 1)) / (start_b - (end_a - 1)), 0.0, 1.0
    )
    eased = _minimum_jerk_blend(progress)[:, None]
    swing[:] = (1.0 - eased) * _BRIDGE_ANCHOR_A + eased * _BRIDGE_ANCHOR_B
    swing[:, 2] += 0.18 * np.sin(np.pi * progress)
    return swing, contact, modes, segment_ids, end_a, start_b


def _minimum_jerk_ramp(num_steps):
    """The renormalized quintic a bridge ramp of ``num_steps`` steps must follow."""
    return _minimum_jerk_blend(np.linspace(0.0, 1.0, num_steps + 1))


@pytest.mark.parametrize("gap_frames", [3, 4, 5, 6, 7, 8, 9])
def test_adjacent_contact_bridges_split_the_free_gap_instead_of_overwriting(
    gap_frames,
):
    """Neighbouring contacts must own disjoint halves of the free gap.

    The release ramp of the earlier contact and the approach ramp of the later
    one both reach into the gap.  When they overlap, the later segment must not
    silently overwrite the earlier one: that truncates the release ramp, drops
    the blend weight in a single frame, and can even remove the free state from
    a genuine free gap.
    """
    window = 5
    swing, contact, modes, segment_ids, end_a, start_b = _two_contact_bridge_case(
        gap_frames
    )
    swing_before = swing.copy()
    contact_before = contact.copy()
    modes_before = modes.copy()
    segment_ids_before = segment_ids.copy()

    targets, alpha = _bridge_swing_to_contact(
        swing, contact, modes, segment_ids, window
    )

    # (6) The caller's source arrays stay immutable.
    np.testing.assert_array_equal(swing, swing_before)
    np.testing.assert_array_equal(contact, contact_before)
    np.testing.assert_array_equal(modes, modes_before)
    np.testing.assert_array_equal(segment_ids, segment_ids_before)

    gap = np.arange(end_a, start_b)
    span = np.arange(end_a - 1, start_b + 1)

    # (3) A real free gap keeps a real free state: at least one frame is pure
    # swing, so the hand is never dragged straight from anchor A to anchor B.
    free = gap[alpha[gap] == 0.0]
    assert len(free) >= 1, (
        f"gap={gap_frames}: no pure-swing frame survives, "
        f"alpha={np.array2string(alpha[span], precision=3)}"
    )
    np.testing.assert_allclose(targets[free], swing[free])

    # (1) Neither ramp is truncated by the other: each is a renormalized
    # minimum-jerk curve that reaches exactly zero at its own boundary.
    release_zero = int(free[0])
    approach_zero = int(free[-1])
    release = alpha[end_a - 1 : release_zero + 1]
    np.testing.assert_allclose(
        release,
        1.0 - _minimum_jerk_ramp(len(release) - 1),
        atol=1e-12,
        err_msg=f"gap={gap_frames}: release ramp is not a renormalized quintic",
    )
    approach = alpha[approach_zero : start_b + 1]
    np.testing.assert_allclose(
        approach,
        _minimum_jerk_ramp(len(approach) - 1),
        atol=1e-12,
        err_msg=f"gap={gap_frames}: approach ramp is not a renormalized quintic",
    )

    # (1b) When the gap is too short for both full windows, the two ramps split
    # it at its temporal midpoint instead of one of them claiming everything.
    release_steps = len(release) - 1
    approach_steps = len(approach) - 1
    if gap_frames + 1 <= 2 * window:
        assert release_steps + approach_steps == gap_frames + 1
        assert abs(release_steps - approach_steps) <= 1, (
            f"gap={gap_frames}: release owns {release_steps} steps and approach "
            f"owns {approach_steps}"
        )

    # (5) No blend weight collapses faster than the shortest honest ramp allows.
    # 1.875 is the peak derivative of the quintic blend.
    shortest = min(release_steps, approach_steps)
    assert np.abs(np.diff(alpha[span])).max() <= 1.875 / shortest + 1e-9

    # (2) Every gap target rides the frame's own swing target toward exactly one
    # anchor, never along the anchor-A-to-anchor-B chord.
    weights = alpha[gap, None]
    toward_a = (1.0 - weights) * swing[gap] + weights * _BRIDGE_ANCHOR_A
    toward_b = (1.0 - weights) * swing[gap] + weights * _BRIDGE_ANCHOR_B
    owned = np.minimum(
        np.linalg.norm(targets[gap] - toward_a, axis=1),
        np.linalg.norm(targets[gap] - toward_b, axis=1),
    )
    assert owned.max() < 1e-12

    # (4) The bridge introduces no first-difference spike relative to the free
    # swing it is blending with.
    bridge_step = np.linalg.norm(np.diff(targets[span], axis=0), axis=1).max()
    swing_step = np.linalg.norm(np.diff(swing[span], axis=0), axis=1).max()
    anchor_step = np.linalg.norm(_BRIDGE_ANCHOR_B - _BRIDGE_ANCHOR_A)
    assert bridge_step <= max(swing_step, anchor_step / gap_frames) * 1.6


def test_separated_contact_bridges_keep_the_unclipped_transition_windows():
    """A gap wide enough for both windows must keep the full-length ramps."""
    window = 5
    swing, contact, modes, segment_ids, end_a, start_b = _two_contact_bridge_case(
        2 * window + 4
    )
    _, alpha = _bridge_swing_to_contact(swing, contact, modes, segment_ids, window)

    full = _minimum_jerk_ramp(window)
    np.testing.assert_allclose(alpha[end_a - 1 : end_a - 1 + window + 1], 1.0 - full)
    np.testing.assert_allclose(alpha[start_b - window : start_b + 1], full)
    np.testing.assert_allclose(alpha[end_a - 1 + window : start_b - window + 1], 0.0)


def test_contact_bridge_is_disabled_without_a_transition_window():
    """With the feature off the targets stay swing outside the contact cores."""
    swing, contact, modes, segment_ids, _, _ = _two_contact_bridge_case(5)
    targets, alpha = _bridge_swing_to_contact(swing, contact, modes, segment_ids, 0)

    in_contact = modes != int(ContactMode.NONE)
    np.testing.assert_array_equal(targets[in_contact], contact[in_contact])
    np.testing.assert_array_equal(targets[~in_contact], swing[~in_contact])
    np.testing.assert_array_equal(alpha, in_contact.astype(float))


def test_back_to_back_contact_segments_keep_their_exact_anchors():
    """Anchor-drift splits leave no free frames and must stay hard-anchored."""
    window = 5
    swing, contact, modes, segment_ids, end_a, _ = _two_contact_bridge_case(0)
    targets, alpha = _bridge_swing_to_contact(
        swing, contact, modes, segment_ids, window
    )

    in_contact = modes != int(ContactMode.NONE)
    np.testing.assert_array_equal(targets[in_contact], contact[in_contact])
    np.testing.assert_allclose(alpha[in_contact], 1.0)
    # The shared boundary carries anchor A on one side and anchor B on the next
    # frame; neither leaks a blended target into the other segment.
    np.testing.assert_allclose(targets[end_a - 1], _BRIDGE_ANCHOR_A)
    np.testing.assert_allclose(targets[end_a], _BRIDGE_ANCHOR_B)


def _orientation_bridge_case(
    modes,
    segment_ids,
    normals,
    window,
    forward=None,
    up=None,
):
    num_frames = len(modes)
    forward = (
        np.tile([1.0, 0.0, 0.0], (num_frames, 1))
        if forward is None
        else np.asarray(forward, dtype=float)
    )
    up = (
        np.tile([0.0, 0.0, 1.0], (num_frames, 1))
        if up is None
        else np.asarray(up, dtype=float)
    )
    _, alpha = _bridge_swing_to_contact(
        np.zeros((num_frames, 3)),
        np.ones((num_frames, 3)),
        modes,
        segment_ids,
        window,
    )
    rotations, orientation_alpha = _bridge_contact_orientation(
        forward,
        up,
        normals,
        modes,
        segment_ids,
        alpha,
        window,
    )
    return rotations, orientation_alpha, alpha


def test_orientation_bridge_is_valid_so3_and_continuous_on_both_sides():
    num_frames = 13
    modes = np.full(num_frames, int(ContactMode.NONE), dtype=np.uint8)
    modes[5:8] = int(ContactMode.SUPPORT)
    segment_ids = np.full(num_frames, -1, dtype=np.int32)
    segment_ids[5:8] = 0
    normals = np.tile([0.0, -1.0, 0.0], (num_frames, 1))

    rotations, orientation_alpha, position_alpha = _orientation_bridge_case(
        modes, segment_ids, normals, window=4
    )

    np.testing.assert_allclose(
        np.swapaxes(rotations, 1, 2) @ rotations,
        np.repeat(np.eye(3)[None, :, :], num_frames, axis=0),
        atol=1e-12,
    )
    np.testing.assert_allclose(np.linalg.det(rotations), 1.0, atol=1e-12)
    np.testing.assert_allclose(orientation_alpha, position_alpha)
    single_frame_steps = Rotation.from_matrix(
        rotations[1:] @ np.swapaxes(rotations[:-1], 1, 2)
    ).magnitude()
    assert np.max(single_frame_steps) < 0.65


def test_orientation_bridge_follows_the_shortest_so3_path():
    num_frames = 12
    modes = np.full(num_frames, int(ContactMode.NONE), dtype=np.uint8)
    modes[5:7] = int(ContactMode.SUPPORT)
    segment_ids = np.full(num_frames, -1, dtype=np.int32)
    segment_ids[5:7] = 0
    contact_rotation = Rotation.from_euler("x", 170.0, degrees=True)
    contact_up = contact_rotation.apply([0.0, 0.0, 1.0])
    normals = np.tile(contact_up, (num_frames, 1))

    rotations, orientation_alpha, _ = _orientation_bridge_case(
        modes, segment_ids, normals, window=4
    )

    geodesic_from_source = Rotation.from_matrix(rotations).magnitude()
    np.testing.assert_allclose(
        geodesic_from_source,
        orientation_alpha * np.deg2rad(170.0),
        atol=1e-10,
    )
    assert np.max(geodesic_from_source) < np.pi


def test_orientation_bridge_returns_to_source_between_different_contacts():
    num_frames = 19
    modes = np.full(num_frames, int(ContactMode.NONE), dtype=np.uint8)
    modes[4:6] = int(ContactMode.SUPPORT)
    modes[13:15] = int(ContactMode.SUPPORT)
    segment_ids = np.full(num_frames, -1, dtype=np.int32)
    segment_ids[4:6] = 0
    segment_ids[13:15] = 1
    normals = np.tile([0.0, 0.0, 1.0], (num_frames, 1))
    normals[4:6] = Rotation.from_euler("x", 90, degrees=True).apply(
        [0.0, 0.0, 1.0]
    )
    normals[13:15] = Rotation.from_euler("x", -90, degrees=True).apply(
        [0.0, 0.0, 1.0]
    )

    rotations, orientation_alpha, _ = _orientation_bridge_case(
        modes, segment_ids, normals, window=5
    )

    # The shared midpoint belongs to both zero-alpha boundaries and is the
    # captured source frame, never a contact-A/contact-B interpolation.
    np.testing.assert_allclose(rotations[9], np.eye(3), atol=1e-12)
    assert orientation_alpha[9] == pytest.approx(0.0)
    gap_rotvecs = Rotation.from_matrix(rotations[6:13]).as_rotvec()
    assert np.all(gap_rotvecs[:3, 0] >= -1e-12)
    assert np.all(gap_rotvecs[4:, 0] <= 1e-12)


def test_orientation_contact_cores_respect_contact_mode_semantics():
    modes = np.asarray(
        [
            ContactMode.SUPPORT,
            ContactMode.NONE,
            ContactMode.SLIDE,
            ContactMode.NONE,
            ContactMode.TOUCH,
        ],
        dtype=np.uint8,
    )
    segment_ids = np.asarray([0, -1, 1, -1, 2], dtype=np.int32)
    normals = np.tile([0.0, -1.0, 0.0], (len(modes), 1))
    forward = np.tile([1.0, 0.0, 0.0], (len(modes), 1))
    up = np.tile([0.0, 0.0, 1.0], (len(modes), 1))
    raw_forward = forward.copy()
    raw_up = up.copy()

    rotations, _, _ = _orientation_bridge_case(
        modes, segment_ids, normals, window=0, forward=forward, up=up
    )

    np.testing.assert_allclose(rotations[0, :, 2], normals[0])
    np.testing.assert_allclose(rotations[2, :, 2], normals[2])
    np.testing.assert_allclose(rotations[4, :, 2], up[4])
    np.testing.assert_array_equal(forward, raw_forward)
    np.testing.assert_array_equal(up, raw_up)


def test_per_frame_sigma_scales_tighten_and_release_monotonically():
    num_frames = 13
    modes = np.full(num_frames, int(ContactMode.NONE), dtype=np.uint8)
    modes[5:8] = int(ContactMode.SUPPORT)
    segment_ids = np.full(num_frames, -1, dtype=np.int32)
    segment_ids[5:8] = 0
    _, alpha = _bridge_swing_to_contact(
        np.zeros((num_frames, 3)),
        np.ones((num_frames, 3)),
        modes,
        segment_ids,
        3,
    )
    position_scale, orientation_scale = _build_per_frame_sigma_scales(
        modes,
        segment_ids,
        alpha,
        {
            "contact_bridge_frames": 3,
            "time_varying_tolerance": {
                "enabled": True,
                "loose_position_scale": 4.0,
                "loose_orientation_scale": 3.0,
            },
        },
    )

    assert np.all(np.diff(position_scale[2:6]) <= 1e-12)
    assert np.all(np.diff(orientation_scale[2:6]) <= 1e-12)
    np.testing.assert_allclose(position_scale[5:8], 1.0)
    np.testing.assert_allclose(orientation_scale[5:8], 1.0)
    assert np.all(np.diff(position_scale[7:11]) >= -1e-12)
    assert np.all(np.diff(orientation_scale[7:11]) >= -1e-12)


def test_per_frame_sigma_scales_do_not_full_lock_slide_or_touch_orientation():
    modes = np.asarray(
        [ContactMode.SLIDE, ContactMode.NONE, ContactMode.TOUCH], dtype=np.uint8
    )
    segment_ids = np.asarray([0, -1, 1], dtype=np.int32)
    alpha = (modes != int(ContactMode.NONE)).astype(float)
    position_scale, orientation_scale = _build_per_frame_sigma_scales(
        modes,
        segment_ids,
        alpha,
        {
            "contact_bridge_frames": 0,
            "time_varying_tolerance": {
                "enabled": True,
                "loose_position_scale": 4.0,
                "loose_orientation_scale": 3.0,
            },
        },
    )

    assert position_scale[0] == pytest.approx(1.0)
    assert orientation_scale[0] == pytest.approx(3.0)
    assert position_scale[2] == pytest.approx(4.0)
    assert orientation_scale[2] == pytest.approx(3.0)


@pytest.mark.parametrize(
    "config", [{}, {"time_varying_tolerance": {"enabled": False}}]
)
def test_time_varying_tolerance_feature_off_preserves_old_payload(config):
    task = _task(np.zeros((4, 3)))
    position_scale, orientation_scale = _build_per_frame_sigma_scales(
        task.contact_modes,
        task.contact_segment_ids,
        np.zeros(4),
        config,
    )
    unchanged = replace(
        task,
        position_sigma_scale=position_scale,
        orientation_sigma_scale=orientation_scale,
    )

    assert position_scale is None
    assert orientation_scale is None
    assert "position_sigma_scale" not in unchanged.frame(0)
    assert "orientation_sigma_scale" not in unchanged.frame(0)


def test_contact_tasks_stack_for_downstream_policy_export():
    scene = trimesh.creation.box(extents=[2.0, 2.0, 0.1])
    scene.apply_translation([0.0, 0.0, -0.05])
    positions = np.zeros((4, 3), dtype=float)
    positions[:, 2] = 0.01
    tasks = infer_scene_contact_tasks(
        {
            name: {
                "tracking_positions": positions,
                "contact_samples": positions[:, None, :],
            }
            for name in ("left_hand", "right_foot")
        },
        scene,
        framerate=30.0,
        config={"min_contact_frames": 1},
    )

    tasks = {
        name: replace(
            task,
            effective_swing_positions=task.tracking_positions.copy(),
            morphology_scale=1.0,
            robot_base_reference_w=np.zeros_like(task.tracking_positions),
        )
        for name, task in tasks.items()
    }
    payload = stack_contact_tasks(tasks)
    assert payload["contact_mode"].shape == (4, 2)
    np.testing.assert_array_equal(
        payload["contact_mode_names"], ["NONE", "TOUCH", "SLIDE", "SUPPORT"]
    )
    assert payload["contact_segment_id"].shape == (4, 2)
    assert payload["contact_targets_w"].shape == (4, 2, 3)
    assert payload["end_effector_reference_w"].shape == (4, 2, 3)
    assert payload["effective_swing_targets_w"].shape == (4, 2, 3)
    assert payload["swing_limb_offsets_w"].shape == (4, 2, 3)
    assert payload["contact_bridge_alpha"].shape == (4, 2)
    assert payload["contact_bridge_targets_w"].shape == (4, 2, 3)
    assert payload["active_contact_offset_id"].shape == (4, 2)
    assert set(payload) == {
        "contact_names",
        "contact_mode",
        "contact_mode_names",
        "contact_segment_id",
        "contact_confidence",
        "contact_transition_weight",
        "contact_targets_w",
        "end_effector_reference_w",
        "effective_swing_targets_w",
        "swing_limb_offsets_w",
        "swing_anchor_body",
        "swing_anchor_offset",
        "contact_bridge_alpha",
        "contact_bridge_targets_w",
        "contact_orientation_bridge_alpha",
        "end_effector_bridge_forward_w",
        "end_effector_bridge_up_w",
        "swing_morphology_scale",
        "active_contact_offset_id",
        "contact_surface_normals_w",
        "end_effector_forward_w",
        "end_effector_up_w",
        "robot_base_reference_w",
    }
    assert "contact" not in payload
    assert "anchor_contacts" not in payload
    assert "preserve_world_target" not in payload


def test_foot_reference_axes_form_valid_contact_frames():
    scene = trimesh.creation.box(extents=[2.0, 2.0, 0.1])
    scene.apply_translation([0.0, 0.0, -0.05])
    positions = np.zeros((5, 3), dtype=float)
    positions[:, 2] = 0.01
    task = infer_scene_contact_tasks(
        {
            "left_foot": {
                "tracking_positions": positions,
                "contact_samples": positions[:, None, :],
                "forward_directions": np.tile([1.0, 0.0, -0.3], (5, 1)),
                "up_directions": np.tile([0.0, 0.0, 1.0], (5, 1)),
            }
        },
        scene,
        framerate=30.0,
        config={
            "distance_threshold": 0.03,
            "relative_speed_threshold": 0.1,
            "min_contact_frames": 1,
        },
    )["left_foot"]

    np.testing.assert_allclose(
        np.sum(task.forward_directions * task.up_directions, axis=1),
        0.0,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.linalg.norm(task.forward_directions, axis=1), 1.0, atol=1e-10
    )
    np.testing.assert_allclose(
        np.linalg.norm(task.up_directions, axis=1), 1.0, atol=1e-10
    )


def test_hand_orientation_does_not_force_sole_normal_contact_filter():
    scene = trimesh.creation.box(extents=[2.0, 2.0, 0.1])
    scene.apply_translation([0.0, 0.0, -0.05])
    positions = np.tile([0.0, 0.0, 0.01], (5, 1))
    task = infer_scene_contact_tasks(
        {
            "left_hand": {
                "tracking_positions": positions,
                "contact_samples": positions[:, None, :],
                "forward_directions": np.tile([1.0, 0.0, 0.0], (5, 1)),
                "up_directions": np.tile([0.0, 0.0, -1.0], (5, 1)),
            }
        },
        scene,
        framerate=30.0,
        config={
            "distance_threshold": 0.03,
            "relative_speed_threshold": 0.1,
            "min_contact_frames": 1,
            "effectors": {
                "left_hand": {"require_normal_alignment": False}
            },
        },
    )["left_hand"]

    assert np.all(task.contact_modes != int(ContactMode.NONE))


def test_slide_contact_normal_signs_can_be_stabilized_across_penetration():
    scene = trimesh.creation.box(extents=[2.0, 2.0, 0.1])
    scene.apply_translation([0.0, 0.0, -0.05])
    tracking = np.tile([0.0, 0.0, 0.02], (6, 1))
    samples = tracking[:, None, :].copy()
    samples[:, 0, 2] = [0.01, -0.01, 0.01, -0.01, 0.01, -0.01]
    task = infer_scene_contact_tasks(
        {
            "left_hand": {
                "tracking_positions": tracking,
                "contact_samples": samples,
                "forward_directions": np.tile([1.0, 0.0, 0.0], (6, 1)),
                "up_directions": np.tile([0.0, 0.0, 1.0], (6, 1)),
            }
        },
        scene,
        framerate=30.0,
        config={
            "distance_threshold": 0.03,
            "relative_speed_threshold": 0.1,
            "min_contact_frames": 1,
            "stabilize_contact_normal_signs": True,
            "effectors": {
                "left_hand": {
                    "contact_mode": "SLIDE",
                    "require_normal_alignment": False,
                }
            },
        },
    )["left_hand"]

    assert np.all(task.contact_modes == int(ContactMode.SLIDE))
    assert np.all(task.surface_normals[:, 2] > 0.99)
    assert np.all(
        np.sum(task.surface_normals[1:] * task.surface_normals[:-1], axis=1)
        > 0.99
    )


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


def test_sparse_trajectory_refinement_reduces_jerk_inside_endpoint_task():
    """Trajectory smoothing must be solved together with endpoint tracking."""
    mujoco = pytest.importorskip("mujoco")
    model, trajectory, target_positions = _straight_arm_setup()
    refined = refine_joint_trajectory(
        model,
        trajectory,
        {"left_foot": _task(target_positions)},
        _effector_config(),
        {
            "iterations": 2,
            "reference_sigma": 0.5,
            "acceleration_sigma": 0.05,
            "jerk_sigma": 0.02,
            "trust_region": 0.2,
        },
    )

    initial_jerk = np.diff(trajectory[:, 7], n=3)
    refined_jerk = np.diff(refined[:, 7], n=3)
    assert np.linalg.norm(refined_jerk) < 0.1 * np.linalg.norm(initial_jerk)

    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "foot"
    )
    initial_errors = []
    refined_errors = []
    for initial_qpos, refined_qpos in zip(trajectory, refined):
        for qpos, errors in (
            (initial_qpos, initial_errors),
            (refined_qpos, refined_errors),
        ):
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            rotation = data.xmat[body_id].reshape(3, 3)
            point = data.xpos[body_id] + rotation @ np.array([1.0, 0.0, 0.0])
            errors.append(np.linalg.norm(point - target_positions[0]))
    assert np.mean(refined_errors) < np.mean(initial_errors)


def test_joint_step_avoidance_only_penalizes_steps_above_group_limit():
    """An active speed boundary suppresses IK jumps without global filtering."""
    pytest.importorskip("mujoco")
    model, trajectory, target_positions = _straight_arm_setup(amplitude=0.12)
    common_config = {
        "iterations": 5,
        "reference_sigma": 0.01,
        "swing_position_sigma": 100.0,
        "acceleration_enabled": False,
        "jerk_enabled": False,
        "trust_region": 0.2,
    }
    unconstrained = refine_joint_trajectory(
        model,
        trajectory,
        {"left_foot": _task(target_positions)},
        _effector_config(track_orientation=False),
        common_config,
    )
    constrained = refine_joint_trajectory(
        model,
        trajectory,
        {"left_foot": _task(target_positions)},
        _effector_config(track_orientation=False),
        {
            **common_config,
            "joint_step_avoidance": {
                "enabled": True,
                "groups": {
                    "ankle": {
                        "joint_tokens": ["ankle"],
                        "max_step": 0.05,
                        "sigma": 0.005,
                    }
                },
            },
        },
    )

    assert np.max(np.abs(np.diff(unconstrained[:, 7]))) > 0.2
    assert np.max(np.abs(np.diff(constrained[:, 7]))) < 0.07


def test_refinement_never_moves_the_floating_base():
    """The second stage owns joints only; world registration stays framewise."""
    pytest.importorskip("mujoco")
    model, trajectory, targets = _straight_arm_setup()
    trajectory[:, :3] = np.linspace(0.0, 0.5, len(trajectory))[:, None]

    refined = refine_joint_trajectory(
        model,
        trajectory,
        {"left_foot": _task(targets)},
        _effector_config(),
        {"iterations": 2, "trust_region": 0.2},
    )

    np.testing.assert_array_equal(refined[:, :7], trajectory[:, :7])
    assert not np.array_equal(refined[:, 7], trajectory[:, 7])


def test_refinement_respects_joint_limits():
    """An unreachable target must saturate at the limit, not step past it."""
    pytest.importorskip("mujoco")
    model, trajectory, _ = _straight_arm_setup(amplitude=0.02)
    # Reaching this target needs roughly 1.0 rad of ankle, well past its limit.
    targets = np.tile([0.5, 0.0, -0.8], (len(trajectory), 1))

    refined = refine_joint_trajectory(
        model,
        trajectory,
        {"left_foot": _task(targets)},
        _effector_config(track_orientation=False),
        {
            "iterations": 4,
            "trust_region": 0.3,
            "reference_sigma": 5.0,
            "swing_position_sigma": 0.01,
        },
    )

    lower, upper = model.jnt_range[model.joint("ankle").id]
    assert np.all(refined[:, 7] >= lower - 1e-9)
    assert np.all(refined[:, 7] <= upper + 1e-9)
    # The limit has to be the thing that stopped it, not a weak pull.
    assert np.max(refined[:, 7]) > 0.9 * upper


def test_refinement_bounds_total_deviation_from_framewise_motion():
    """Repeated collision/IK iterations cannot silently redesign the pose."""
    pytest.importorskip("mujoco")
    model, trajectory, _ = _straight_arm_setup(amplitude=0.02)
    targets = np.tile([0.5, 0.0, -0.8], (len(trajectory), 1))
    max_deviation = 0.15

    refined = refine_joint_trajectory(
        model,
        trajectory,
        {"left_foot": _task(targets)},
        _effector_config(track_orientation=False),
        {
            "iterations": 8,
            "trust_region": 0.3,
            "max_joint_deviation": max_deviation,
            "reference_sigma": 5.0,
            "swing_position_sigma": 0.01,
            "acceleration_enabled": False,
            "jerk_enabled": False,
        },
    )

    deviation = np.abs(refined[:, 7] - trajectory[:, 7])
    assert np.max(deviation) <= max_deviation + 1e-8
    assert np.max(deviation) > 0.9 * max_deviation


def test_refinement_accepts_a_trajectory_that_starts_outside_a_joint_limit():
    """A framewise pose past a limit must not make the bounded solve infeasible."""
    pytest.importorskip("mujoco")
    model, trajectory, targets = _straight_arm_setup()
    # Further outside the limit than one trust region, which is exactly the
    # case where the limit bound and the trust-region bound cross over.
    trajectory[5, 7] = -0.95

    refined = refine_joint_trajectory(
        model,
        trajectory,
        {"left_foot": _task(targets)},
        _effector_config(),
        {"iterations": 2, "trust_region": 0.12},
    )

    assert np.all(np.isfinite(refined))
    assert refined[5, 7] > trajectory[5, 7]


def test_contact_offset_selection_is_locked_per_contact_run():
    """A no-slip run keeps one keypoint so the contact residual stays smooth."""
    mujoco = pytest.importorskip("mujoco")
    model, trajectory, targets = _straight_arm_setup(num_frames=14, amplitude=0.02)
    contact_modes = np.full(len(trajectory), int(ContactMode.NONE), dtype=np.uint8)
    contact_modes[3:11] = int(ContactMode.SUPPORT)

    # Two candidate keypoints straddle the target, so a per-frame nearest
    # search would flip between them as the ankle chatters.
    refined = refine_joint_trajectory(
        model,
        trajectory,
        {"left_foot": _task(targets, contact_modes)},
        _effector_config(
            contact_offsets=[[1.0, 0.0, 0.02], [1.0, 0.0, -0.02]]
        ),
        {
            "iterations": 3,
            "contact_position_sigma": 0.01,
            "reference_sigma": 0.5,
            "trust_region": 0.2,
        },
    )

    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot")
    errors = []
    for frame_idx in np.flatnonzero(contact_modes == int(ContactMode.SUPPORT)):
        data.qpos[:] = refined[frame_idx]
        mujoco.mj_forward(model, data)
        rotation = data.xmat[body_id].reshape(3, 3)
        candidates = data.xpos[body_id][None, :] + np.array(
            [[1.0, 0.0, 0.02], [1.0, 0.0, -0.02]]
        ) @ rotation.T
        errors.append(np.min(np.linalg.norm(candidates - targets[0], axis=1)))
    assert np.max(errors) < 0.05
    contact_jerk = np.diff(
        refined[contact_modes == int(ContactMode.SUPPORT), 7], n=3
    )
    assert np.max(np.abs(contact_jerk)) < 0.01


def test_refinement_hierarchy_keeps_vault_support_above_smoothness():
    """Acceleration/jerk may use null space but cannot move a SUPPORT hand."""
    mujoco = pytest.importorskip("mujoco")
    model, trajectory, targets = _straight_arm_setup(num_frames=14, amplitude=0.18)
    contact_modes = np.full(len(trajectory), int(ContactMode.NONE), dtype=np.uint8)
    contact_modes[4:10] = int(ContactMode.SUPPORT)
    task = _task(targets, contact_modes)
    refined = refine_joint_trajectory(
        model,
        trajectory,
        {"left_foot": task},
        {**_effector_config(track_orientation=False), "hierarchy": True},
        {
            "hierarchy": True,
            "iterations": 3,
            "contact_position_sigma": 0.01,
            "support_priority_relative_tolerance": 0.0,
            "support_priority_absolute_tolerance": 0.02,
            "reference_sigma": 2.0,
            "acceleration_sigma": 0.005,
            "jerk_sigma": 0.002,
            "trust_region": 0.3,
        },
    )

    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot")
    errors = []
    markers = []
    for frame_idx in np.flatnonzero(contact_modes == int(ContactMode.SUPPORT)):
        data.qpos[:] = refined[frame_idx]
        mujoco.mj_forward(model, data)
        rotation = data.xmat[body_id].reshape(3, 3)
        marker = data.xpos[body_id] + rotation @ np.array([1.0, 0.0, 0.0])
        markers.append(marker)
        errors.append(np.linalg.norm(marker - targets[frame_idx]))
    assert np.max(errors) < 0.002
    assert np.max(np.linalg.norm(np.asarray(markers) - markers[0], axis=1)) < 0.002


@pytest.mark.parametrize("contact_mode", [ContactMode.SLIDE, ContactMode.TOUCH])
def test_refinement_contact_hierarchy_keeps_contact_above_smoothness(
    contact_mode,
):
    """Temporal smoothing cannot erase SLIDE or TOUCH contact motion."""
    mujoco = pytest.importorskip("mujoco")
    model, trajectory, _ = _straight_arm_setup(num_frames=14, amplitude=0.18)
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot")
    targets = []
    for qpos in trajectory:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        targets.append(
            data.xpos[body_id]
            + data.xmat[body_id].reshape(3, 3) @ np.array([1.0, 0.0, 0.0])
        )
    targets = np.asarray(targets)
    modes = np.full(len(trajectory), int(contact_mode), dtype=np.uint8)
    task = replace(
        _task(targets, modes),
        surface_normals=np.tile([0.0, 1.0, 0.0], (len(trajectory), 1)),
    )
    base_config = {
        "hierarchy": True,
        "iterations": 3,
        "contact_position_sigma": 0.01,
        "support_priority_relative_tolerance": 0.0,
        "support_priority_absolute_tolerance": 0.02,
        "reference_sigma": 2.0,
        "acceleration_sigma": 0.005,
        "jerk_sigma": 0.002,
        "trust_region": 0.3,
    }
    effector_config = {
        **_effector_config(track_orientation=False),
        "hierarchy": True,
        "slide_tangential_weight": 1.0,
    }

    weighted_motion = refine_joint_trajectory(
        model,
        trajectory,
        {"left_foot": task},
        effector_config,
        base_config,
    )
    contact_priority = refine_joint_trajectory(
        model,
        trajectory,
        {"left_foot": task},
        effector_config,
        {
            **base_config,
            "contact_hierarchy": True,
            "contact_priority_relative_tolerance": 0.0,
            "contact_priority_absolute_tolerance": 0.02,
        },
    )

    def marker_errors(result):
        errors = []
        for frame_idx, qpos in enumerate(result):
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            marker = (
                data.xpos[body_id]
                + data.xmat[body_id].reshape(3, 3)
                @ np.array([1.0, 0.0, 0.0])
            )
            errors.append(np.linalg.norm(marker - targets[frame_idx]))
        return np.asarray(errors)

    weighted_errors = marker_errors(weighted_motion)
    priority_errors = marker_errors(contact_priority)
    assert np.max(priority_errors) < 0.002
    assert np.mean(weighted_errors) > 5.0 * np.mean(priority_errors)


def _measure_mixed_touch_slide_contact_refinement(
    absolute_tolerance,
    *,
    slide_tangential_weight=0.1,
):
    """Measure the contact level on jointly achievable TOUCH and SLIDE rows."""
    mujoco = pytest.importorskip("mujoco")
    model, trajectory, _ = _straight_arm_setup(
        num_frames=14, amplitude=0.18
    )
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "foot")
    targets = []
    for qpos in trajectory:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        targets.append(
            data.xpos[body_id]
            + data.xmat[body_id].reshape(3, 3) @ np.array([1.0, 0.0, 0.0])
        )
    targets = np.asarray(targets)
    modes = np.full(len(trajectory), int(ContactMode.TOUCH), dtype=np.uint8)
    modes[len(trajectory) // 2 :] = int(ContactMode.SLIDE)
    normal = np.array([0.0, 1.0, 0.0])
    task = replace(
        _task(targets, modes),
        surface_normals=np.tile(normal, (len(trajectory), 1)),
    )
    effector_config = {
        **_effector_config(track_orientation=False),
        "hierarchy": True,
        "touch_weight": 1.0,
        "slide_tangential_weight": slide_tangential_weight,
    }
    config = {
        "hierarchy": True,
        "contact_hierarchy": True,
        "iterations": 3,
        "contact_position_sigma": 0.012,
        "contact_priority_relative_tolerance": 0.0,
        "support_priority_relative_tolerance": 0.0,
        "support_priority_absolute_tolerance": 0.03,
        "reference_sigma": 2.0,
        "acceleration_sigma": 0.005,
        "jerk_sigma": 0.002,
        "trust_region": 0.3,
        "posture_hierarchy": False,
    }
    if absolute_tolerance is not None:
        config["contact_priority_absolute_tolerance"] = absolute_tolerance
    diagnostics = {}
    refined = refine_joint_trajectory(
        model,
        trajectory,
        {"left_foot": task},
        effector_config,
        config,
        diagnostics=diagnostics,
    )
    markers = []
    for qpos in refined:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        markers.append(
            data.xpos[body_id]
            + data.xmat[body_id].reshape(3, 3)
            @ np.array([1.0, 0.0, 0.0])
        )
    markers = np.asarray(markers)
    physical_errors = np.linalg.norm(markers - targets, axis=1)
    touch = modes == int(ContactMode.TOUCH)
    slide = modes == int(ContactMode.SLIDE)
    tangent_projection = np.eye(3) - np.outer(normal, normal)
    target_slide_path = float(
        np.sum(
            np.linalg.norm(
                np.diff(targets[slide] @ tangent_projection.T, axis=0), axis=1
            )
        )
    )
    refined_slide_path = float(
        np.sum(
            np.linalg.norm(
                np.diff(markers[slide] @ tangent_projection.T, axis=0), axis=1
            )
        )
    )
    result = dict(diagnostics["contact_priority"])
    result.update(
        {
            "physical_error_p50_m": float(np.percentile(physical_errors, 50)),
            "physical_error_p95_m": float(np.percentile(physical_errors, 95)),
            "physical_error_max_m": float(np.max(physical_errors)),
            "touch_error_p95_m": float(
                np.percentile(physical_errors[touch], 95)
            ),
            "slide_error_p95_m": float(
                np.percentile(physical_errors[slide], 95)
            ),
            "slide_tangential_path_preserved": (
                refined_slide_path / target_slide_path
            ),
        }
    )
    return result


def test_default_contact_priority_tolerance_is_physically_small():
    """An unset absolute budget keeps achievable TOUCH/SLIDE below 1 mm."""
    measured = _measure_mixed_touch_slide_contact_refinement(None)

    assert measured["configured_absolute_tolerance"] == pytest.approx(0.03)
    assert measured["active_residual_rows"] == 42
    assert measured["physical_error_p95_m"] < 0.001
    assert measured["slack_consumed"] < 0.031
    assert measured["solver_status"] in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    assert measured["solver_failure_count"] == 0


def test_large_contact_priority_tolerance_clearly_degrades_contact():
    """A huge control budget exposes the regression without pinning its output."""
    tight = _measure_mixed_touch_slide_contact_refinement(0.03)
    loose = _measure_mixed_touch_slide_contact_refinement(10.0)

    assert loose["physical_error_p95_m"] > 0.01
    assert loose["physical_error_p95_m"] > 20.0 * tight["physical_error_p95_m"]
    assert loose["slack_consumed"] > 100.0 * tight["slack_consumed"]
    assert loose["solver_failure_count"] == 0


def test_instinctmj_slide_weight_preserves_tangential_travel_with_small_slack():
    """SLIDE stays softer than TOUCH without sacrificing its source travel."""
    profile_path = (
        Path(__file__).resolve().parents[1]
        / "robot_models"
        / "unitree_g1"
        / "unitree_g1_instinctmj.json"
    )
    with profile_path.open(encoding="utf-8") as profile_file:
        profile = json.load(profile_file)["retargeting"]
    refinement = profile["trajectory_refinement"]
    measured = _measure_mixed_touch_slide_contact_refinement(
        refinement["contact_priority_absolute_tolerance"],
        slide_tangential_weight=profile["end_effector_priority"][
            "slide_tangential_weight"
        ],
    )

    assert measured["slide_tangential_path_preserved"] > 0.99
    assert measured["slide_error_p95_m"] < 0.001
    assert measured["slide_error_p95_m"] > measured["touch_error_p95_m"]


def test_configured_four_level_hierarchy_keeps_support_ahead_of_swing():
    """The all-contact optimum must remain bounded while later levels pull away."""
    variable = cp.Variable(1)
    config = {
        "priority_levels": [
            {
                "name": "all-contact",
                "families": ["support", "contact"],
                "relative_tolerance": 0.0,
                "absolute_tolerance": 0.02,
            },
            {
                "name": "swing",
                "families": ["swing"],
                "relative_tolerance": 0.0,
                "absolute_tolerance": 0.0,
            },
            {
                "name": "posture",
                "families": ["orientation", "pose_direction"],
                "relative_tolerance": 0.0,
                "absolute_tolerance": 0.0,
            },
            {
                "name": "whole-body",
                "families": ["interaction", "regularization"],
                "relative_tolerance": 0.0,
                "absolute_tolerance": 0.0,
            },
        ]
    }
    residuals = {
        "support": variable,
        "contact": None,
        "swing": variable - 1.0,
        "orientation": 0.0 * variable,
        "pose_direction": 0.0 * variable,
        "interaction": 0.0 * variable,
    }

    def solve_problem(objective, _name, constraints):
        problem = cp.Problem(cp.Minimize(objective), list(constraints))
        problem.solve(solver=cp.CLARABEL)
        return problem

    problem, solved_names = _solve_configured_priority_levels(
        variable,
        [variable >= -2.0, variable <= 2.0],
        residuals,
        cp.sum_squares(variable - 1.0),
        config,
        solve_problem,
    )

    assert problem is not None and problem.status == cp.OPTIMAL
    assert solved_names == ("all-contact", "swing", "posture", "whole-body")
    assert abs(float(variable.value[0])) <= 0.0201


def test_base_orientation_hard_cone_bounds_a_conflicting_lower_level():
    delta = cp.Variable(3)
    residual, constraints = _build_base_orientation_task(
        delta,
        np.eye(3),
        np.asarray([0.0, 0.0, 1.0]),
        {
            "enabled": True,
            "hard_constraints": True,
            "max_orientation_error": 0.15,
            "weight": 1.0,
        },
    )
    assert residual is not None
    problem = cp.Problem(cp.Minimize(cp.sum_squares(delta)), constraints)
    problem.solve(solver=cp.CLARABEL)

    assert problem.status == cp.OPTIMAL
    remaining_error = np.linalg.norm(delta.value - [0.0, 0.0, 1.0])
    assert remaining_error <= 0.15001
    assert remaining_error >= 0.149


def test_priority_level_grouping_is_configurable_between_four_and_seven_levels():
    shipped = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "robot_models/unitree_g1/unitree_g1_instinctmj.json"
        ).read_text(encoding="utf-8")
    )["retargeting"]["end_effector_priority"]
    seven_levels = _resolve_priority_levels(shipped)
    assert len(seven_levels) == 7
    assert seven_levels[0]["families"] == ("support",)

    four_config = {
        "priority_levels": [
            {"name": "all-contact", "families": ["support", "contact"]},
            {"name": "swing", "families": ["swing"]},
            {
                "name": "posture",
                "families": ["orientation", "pose_direction"],
            },
            {
                "name": "whole-body",
                "families": ["interaction", "regularization"],
            },
        ]
    }
    four_levels = _resolve_priority_levels(four_config)
    assert len(four_levels) == 4
    assert four_levels[0]["families"] == ("support", "contact")


def test_capsule_surface_offsets_follow_the_local_z_axis():
    """MuJoCo capsules are z-aligned; sampling along x misses both end caps."""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><worldbody><body name="link">
          <geom name="shin" type="capsule" size="0.05 0.2"/>
        </body></worldbody></mujoco>
        """
    )
    offsets = geom_surface_offsets(model, model.geom("shin").id)

    assert np.isclose(np.max(offsets[:, 2]), 0.25)
    assert np.isclose(np.min(offsets[:, 2]), -0.25)
    assert np.isclose(np.max(np.abs(offsets[:, :2])), 0.05)


def test_refinement_penalizes_scene_penetration():
    """An endpoint target below the floor must not be tracked through it."""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <compiler angle="radian"/>
          <worldbody>
            <body name="root">
              <freejoint/>
              <geom type="sphere" size="0.02" mass="1"/>
              <body name="foot">
                <joint name="ankle" type="hinge" axis="0 1 0" range="-2 2"/>
                <geom name="shin" type="capsule" fromto="0 0 0 0.4 0 0" size="0.05"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    # A thick slab keeps the closest surface on top, so the reported depth is
    # measured from the walking surface rather than from the underside.
    floor = trimesh.creation.box(extents=[8.0, 8.0, 2.0])
    floor.apply_translation([0.0, 0.0, -1.0])

    num_frames = 10
    trajectory = np.tile(model.qpos0, (num_frames, 1))
    trajectory[:, 2] = 0.25
    trajectory[:, 3] = 1.0
    trajectory[:, 7] = 0.2
    # The task target sits below the floor, so tracking it alone digs in.
    targets = np.tile([0.15, 0.0, -0.15], (num_frames, 1))

    assert len(collision_geom_ids(model)) == 2
    assert measure_scene_penetration(model, trajectory, floor)["max_depth"] == 0.0

    config = {
        "effectors": {
            "left_foot": {
                "robot_body": "foot",
                "tracking_offset": [0.4, 0.0, 0.0],
                "contact_offsets": [[0.4, 0.0, 0.0]],
                "contact_target_space": "world",
                "swing_target_space": "root_morphology_scaled",
            }
        }
    }
    settings = {
        "iterations": 4,
        "trust_region": 0.4,
        "reference_sigma": 1.0,
        "swing_position_sigma": 0.01,
    }

    unconstrained = refine_joint_trajectory(
        model,
        trajectory,
        {"left_foot": _task(targets)},
        config,
        {**settings, "penetration": {"enabled": False}},
        scene_mesh=floor,
    )
    constrained = refine_joint_trajectory(
        model,
        trajectory,
        {"left_foot": _task(targets)},
        config,
        {**settings, "penetration": {"enabled": True, "clearance": 0.0, "sigma": 0.002}},
        scene_mesh=floor,
    )

    without = measure_scene_penetration(model, unconstrained, floor)
    with_penalty = measure_scene_penetration(model, constrained, floor)
    assert without["max_depth"] > 0.05
    assert without["total_penetrating_samples"] > 0
    assert without["penetrating_body_count"] >= 1
    assert np.sum(without["per_frame_penetrating_samples"]) == without[
        "total_penetrating_samples"
    ]
    assert with_penalty["max_depth"] < 0.5 * without["max_depth"]
    assert without["worst_body"] == "foot"


def test_derivative_sigmas_are_rescaled_to_the_source_framerate():
    """The same config must smooth equally hard at 30 and 60 Hz."""
    pytest.importorskip("mujoco")
    model, trajectory, targets = _straight_arm_setup(num_frames=24)
    config = {
        "iterations": 1,
        "reference_sigma": 0.5,
        "acceleration_sigma": 0.05,
        "jerk_sigma": 0.02,
        "trust_region": 0.2,
        "sigma_reference_framerate": 30.0,
    }
    common = (
        model,
        trajectory,
        {"left_foot": _task(targets)},
        _effector_config(),
        config,
    )

    at_reference = refine_joint_trajectory(*common, framerate=30.0)
    unscaled = refine_joint_trajectory(*common, framerate=None)
    doubled = refine_joint_trajectory(*common, framerate=60.0)

    np.testing.assert_allclose(at_reference[:, 7], unscaled[:, 7], atol=1e-12)
    # Halving dt makes the same physical smoothness a stronger per-frame
    # penalty, so the chatter has to be removed at least as aggressively.
    assert np.linalg.norm(np.diff(doubled[:, 7], n=3)) <= np.linalg.norm(
        np.diff(at_reference[:, 7], n=3)
    )


LIMB_ANCHOR_XML = """
<mujoco>
  <compiler angle="radian"/>
  <worldbody>
    <body name="torso">
      <freejoint/>
      <geom type="sphere" size="0.02" mass="1"/>
      <body name="shoulder" pos="0 0.1 0.25">
        <joint name="shoulder_pitch" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
        <body name="hand" pos="0.2 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0"/>
          <geom type="sphere" size="0.02"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

_LIMB_SOURCE_JOINTS = ("Spine3", "L_Shoulder", "L_Elbow", "L_Wrist")


def _limb_anchor_retargeter(effector_spec):
    """Build the minimal ``OmniRetargeter`` state the swing preparation reads."""
    mujoco = pytest.importorskip("mujoco")
    from omniretargeting.core import OmniRetargeter

    retargeter = object.__new__(OmniRetargeter)
    retargeter.robot_model = mujoco.MjModel.from_xml_string(LIMB_ANCHOR_XML)
    retargeter.valid_joint_mapping = {}
    retargeter.base_orientation_config = {"root_position": "Spine3"}
    retargeter.smplx_joint_indices = {
        name: index for index, name in enumerate(_LIMB_SOURCE_JOINTS)
    }
    retargeter.retargeting_config = {}
    return retargeter, {
        "base_position_constraint": {"z_offset": -0.4},
        "root_warp": {"enabled": False},
        "contact_bridge_frames": 0,
        "effectors": {"left_hand": effector_spec},
    }


def _limb_anchor_source(num_frames=4):
    """A source arm that swings well outside the small robot's reach shell."""
    trajectory = np.zeros((num_frames, len(_LIMB_SOURCE_JOINTS), 3))
    phase = np.linspace(0.0, 1.2, num_frames)
    trajectory[:, 0] = np.stack(
        [phase, np.zeros(num_frames), np.full(num_frames, 1.3)], axis=1
    )
    trajectory[:, 1] = trajectory[:, 0] + [0.0, 0.18, 0.12]
    trajectory[:, 2] = trajectory[:, 1] + [0.3, 0.0, -0.1]
    trajectory[:, 3] = trajectory[:, 2] + [0.3, 0.0, -0.1]
    return trajectory


def test_prepare_orientation_bridge_defaults_to_position_window_and_zero_disables():
    source = _limb_anchor_source(num_frames=9)
    hand = source[:, 3] + [0.05, 0.0, 0.0]
    modes = np.full(len(source), int(ContactMode.NONE), dtype=np.uint8)
    modes[4:6] = int(ContactMode.SUPPORT)
    task = replace(
        _task(hand, modes),
        name="left_hand",
        surface_normals=np.tile([0.0, -1.0, 0.0], (len(source), 1)),
    )
    retargeter, config = _limb_anchor_retargeter(
        {
            "robot_body": "hand",
            "contact_target_space": "world",
            "swing_target_space": "limb_morphology_scaled",
            "swing_morphology_scale": 0.5,
            "reach_reference_body": "shoulder",
            "max_swing_reach": 10.0,
            "contact_offsets": [[0.0, 0.0, 0.0]],
        }
    )
    config["contact_bridge_frames"] = 2

    inherited = retargeter._prepare_contact_aware_references(
        source, {"left_hand": task}, config
    )["left_hand"]
    np.testing.assert_allclose(
        inherited.orientation_bridge_alpha, inherited.contact_bridge_alpha
    )

    disabled = retargeter._prepare_contact_aware_references(
        source,
        {"left_hand": task},
        {**config, "orientation_bridge_frames": 0},
    )["left_hand"]
    free = modes == int(ContactMode.NONE)
    np.testing.assert_allclose(disabled.orientation_bridge_alpha[free], 0.0)
    np.testing.assert_allclose(
        disabled.orientation_target_rotations[free],
        np.repeat(np.eye(3)[None, :, :], np.count_nonzero(free), axis=0),
    )
    assert np.any(disabled.contact_bridge_alpha[free] > 0.0)


def test_limb_anchored_swing_keeps_source_arm_direction_under_torso_rotation():
    """The scaled hand target must hang off the robot shoulder, not the base."""
    source = _limb_anchor_source()
    num_frames = len(source)
    hand = source[:, 3] + [0.05, 0.0, 0.0]
    retargeter, config = _limb_anchor_retargeter(
        {
            "robot_body": "hand",
            "contact_target_space": "world",
            "swing_target_space": "limb_morphology_scaled",
            "swing_morphology_scale": 0.5,
            "reach_reference_body": "shoulder",
            "max_swing_reach": 10.0,
            "contact_offsets": [[0.0, 0.0, 0.0]],
        }
    )
    task = replace(_task(hand), name="left_hand")
    # A pitched torso must carry its shoulder with it.
    yaw = np.linspace(0.0, 0.9, num_frames)
    base_orientations = Rotation.from_euler(
        "z", yaw[:, None]
    ).as_quat(scalar_first=True)

    prepared = retargeter._prepare_contact_aware_references(
        source,
        {"left_hand": task},
        config,
        base_orientations=base_orientations,
    )["left_hand"]

    base_reference = source[:, 0] + [0.0, 0.0, -0.4]
    shoulder_offset = np.array([0.0, 0.1, 0.25])
    robot_shoulder = base_reference + Rotation.from_euler(
        "z", yaw[:, None]
    ).apply(shoulder_offset)
    np.testing.assert_allclose(
        prepared.effective_swing_positions - robot_shoulder,
        0.5 * (hand - source[:, 1]),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        prepared.swing_limb_offsets,
        0.5 * (hand - source[:, 1]),
        atol=1e-9,
    )
    assert prepared.swing_anchor_body == "shoulder"
    np.testing.assert_allclose(prepared.contact_bridge_alpha, 0.0)
    assert prepared.morphology_scale == pytest.approx(0.5)


def test_framewise_limb_swing_uses_live_anchor_and_relative_jacobian():
    """Base motion must move the hand and shoulder together, not expose a ghost."""
    cvxpy = pytest.importorskip("cvxpy")
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(LIMB_ANCHOR_XML)
    data = mujoco.MjData(model)
    data.qpos[:7] = np.r_[
        [0.7, -0.2, 1.1],
        Rotation.from_euler("z", 0.8).as_quat(scalar_first=True),
    ]
    mujoco.mj_forward(model, data)
    shoulder_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "shoulder"
    )
    hand_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "hand"
    )
    tracking_offset = np.array([0.2, 0.0, 0.0])
    hand_position = (
        data.xpos[hand_id]
        + data.xmat[hand_id].reshape(3, 3) @ tracking_offset
    )
    limb_offset = hand_position - data.xpos[shoulder_id]

    retargeter = GenericInteractionRetargeter(
        model,
        data,
        trimesh.creation.box(extents=[2.0, 2.0, 0.1]),
        {"L_Wrist": "hand"},
        robot_height=1.0,
        terrain_sample_points=4,
        valid_joint_names=["L_Wrist"],
        end_effector_config={
            "enabled": True,
            "effectors": {
                "left_hand": {
                    "robot_body": "hand",
                    "tracking_offset": tracking_offset.tolist(),
                    "contact_offsets": [tracking_offset.tolist()],
                    "contact_target_space": "world",
                    "swing_target_space": "limb_morphology_scaled",
                }
            },
        },
    )
    task = replace(
        _task(hand_position[None, :]),
        name="left_hand",
        # This deliberately wrong preview must never enter the objective.
        effective_swing_positions=np.array([[20.0, -10.0, 5.0]]),
        swing_limb_offsets=limb_offset[None, :],
        swing_anchor_body="shoulder",
        swing_anchor_offset=np.zeros(3),
        contact_bridge_alpha=np.zeros(1),
        contact_bridge_positions=np.zeros((1, 3)),
    )
    dqa = cvxpy.Variable(retargeter.nq_a)
    swing = retargeter._build_end_effector_objectives(
        dqa, {"left_hand": task.frame(0)}
    )[2]

    zero_step = np.zeros(retargeter.nq_a)
    dqa.value = zero_step
    assert np.linalg.norm(swing.value) < 1e-10
    base_x = int(np.flatnonzero(retargeter.q_a_indices == 0)[0])
    translated_step = zero_step.copy()
    translated_step[base_x] = 0.03
    dqa.value = translated_step
    # J_hand - J_anchor cancels a common floating-base translation.
    assert np.linalg.norm(swing.value) < 1e-10


def test_contact_ik_can_lock_floating_base_orientation_to_source_reference():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(LIMB_ANCHOR_XML)
    data = mujoco.MjData(model)
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)
    retargeter = GenericInteractionRetargeter(
        model,
        data,
        trimesh.creation.box(extents=[2.0, 2.0, 0.1]),
        {"L_Wrist": "hand"},
        robot_height=1.0,
        terrain_sample_points=4,
        valid_joint_names=["L_Wrist"],
        end_effector_config={
            "enabled": True,
            "lock_base_orientation": True,
            "effectors": {
                "left_hand": {
                    "robot_body": "hand",
                    "tracking_offset": [0.0, 0.0, 0.0],
                    "contact_offsets": [[0.0, 0.0, 0.0]],
                    "contact_target_space": "world",
                    "swing_target_space": "limb_morphology_scaled",
                }
            },
        },
    )
    assert not np.any(np.isin(np.arange(3, 7), retargeter.q_a_indices))


def test_refinement_limb_swing_ignores_reference_pose_world_preview():
    """Trajectory refinement must linearize the same live-shoulder residual."""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(LIMB_ANCHOR_XML)
    trajectory = np.tile(model.qpos0, (4, 1))
    trajectory[:, 3:7] = Rotation.from_euler(
        "z", np.linspace(0.0, 1.0, len(trajectory))[:, None]
    ).as_quat(scalar_first=True)
    data = mujoco.MjData(model)
    shoulder_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "shoulder"
    )
    hand_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "hand"
    )
    tracking_offset = np.array([0.2, 0.0, 0.0])
    hand_positions = []
    limb_offsets = []
    for qpos in trajectory:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        hand = (
            data.xpos[hand_id]
            + data.xmat[hand_id].reshape(3, 3) @ tracking_offset
        )
        hand_positions.append(hand)
        limb_offsets.append(hand - data.xpos[shoulder_id])
    hand_positions = np.asarray(hand_positions)
    limb_offsets = np.asarray(limb_offsets)
    task = replace(
        _task(hand_positions),
        name="left_hand",
        effective_swing_positions=np.full_like(hand_positions, 10.0),
        swing_limb_offsets=limb_offsets,
        swing_anchor_body="shoulder",
        swing_anchor_offset=np.zeros(3),
        contact_bridge_alpha=np.zeros(len(trajectory)),
        contact_bridge_positions=np.zeros_like(hand_positions),
    )
    refined = refine_joint_trajectory(
        model,
        trajectory,
        {"left_hand": task},
        {
            "effectors": {
                "left_hand": {
                    "robot_body": "hand",
                    "tracking_offset": tracking_offset.tolist(),
                    "contact_offsets": [tracking_offset.tolist()],
                    "contact_target_space": "world",
                    "swing_target_space": "limb_morphology_scaled",
                }
            }
        },
        {"iterations": 2, "trust_region": 0.2},
    )
    np.testing.assert_allclose(refined[:, 7:], trajectory[:, 7:], atol=1e-10)


def test_refinement_keeps_swing_endpoints_above_temporal_smoothing():
    """Whole-trajectory smoothing may use null space but not erase free swing."""
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(ANKLE_ARM_XML)
    trajectory = np.tile(model.qpos0, (8, 1))
    trajectory[:, 3] = 1.0
    trajectory[:, 7] = 0.25 * (-1.0) ** np.arange(len(trajectory))
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "foot"
    )
    targets = []
    for qpos in trajectory:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        targets.append(
            data.xpos[body_id]
            + data.xmat[body_id].reshape(3, 3) @ np.array([1.0, 0.0, 0.0])
        )
    targets = np.asarray(targets)
    refined = refine_joint_trajectory(
        model,
        trajectory,
        {"left_foot": _task(targets)},
        _effector_config(track_orientation=False),
        {
            "hierarchy": True,
            "iterations": 2,
            "swing_position_sigma": 0.02,
            "reference_sigma": 100.0,
            "acceleration_sigma": 0.001,
            "jerk_sigma": 0.001,
            "trust_region": 0.3,
        },
    )
    errors = []
    for frame_idx, qpos in enumerate(refined):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        marker = (
            data.xpos[body_id]
            + data.xmat[body_id].reshape(3, 3) @ np.array([1.0, 0.0, 0.0])
        )
        errors.append(np.linalg.norm(marker - targets[frame_idx]))
    assert max(errors) < 0.003


def test_limb_anchored_swing_needs_a_resolvable_robot_limb_base():
    """Silently falling back to the base anchor is the bug this mode fixes."""
    source = _limb_anchor_source()
    retargeter, config = _limb_anchor_retargeter(
        {
            "robot_body": "hand",
            "contact_target_space": "world",
            "swing_target_space": "limb_morphology_scaled",
            "swing_morphology_scale": 0.5,
            "reach_reference_body": "no_such_body",
            "max_swing_reach": 10.0,
            "contact_offsets": [[0.0, 0.0, 0.0]],
        }
    )
    task = replace(_task(source[:, 3]), name="left_hand")

    with pytest.raises(ValueError, match="reach_reference_body"):
        retargeter._prepare_contact_aware_references(
            source, {"left_hand": task}, config
        )


def test_automatic_morphology_scale_matches_the_robot_limb_reach():
    """'auto' must map the source arm length onto the robot arm length."""
    source = _limb_anchor_source()
    hand = source[:, 3] + [0.05, 0.0, 0.0]
    retargeter, config = _limb_anchor_retargeter(
        {
            "robot_body": "hand",
            "contact_target_space": "world",
            "swing_target_space": "limb_morphology_scaled",
            "swing_morphology_scale": "auto",
            "reach_reference_body": "shoulder",
            "max_swing_reach": 0.3,
            "swing_morphology_scale_bounds": [0.05, 1.15],
            "contact_offsets": [[0.0, 0.0, 0.0]],
        }
    )
    task = replace(_task(hand), name="left_hand")

    prepared = retargeter._prepare_contact_aware_references(
        source, {"left_hand": task}, config
    )["left_hand"]

    source_reach = (
        np.linalg.norm(source[:, 2] - source[:, 1], axis=1)
        + np.linalg.norm(source[:, 3] - source[:, 2], axis=1)
        + np.linalg.norm(hand - source[:, 3], axis=1)
    )
    assert prepared.morphology_scale == pytest.approx(
        0.3 / float(np.median(source_reach))
    )
    # Every free-space target now sits inside the robot reach shell.
    base_reference = source[:, 0] + [0.0, 0.0, -0.4]
    robot_shoulder = base_reference + np.array([0.0, 0.1, 0.25])
    reach = np.linalg.norm(
        prepared.effective_swing_positions - robot_shoulder, axis=1
    )
    assert np.all(reach <= 0.3 + 1e-9)


def test_scaled_swing_target_inside_the_scene_fades_back_to_world():
    """A shortened limb must not aim its endpoint into the geometry."""
    from omniretargeting.core import _resolve_swing_targets_against_scene

    # A 1 m block centred on the origin; the world reference clears its face.
    block = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    scaled = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    world = np.array([[1.4, 0.0, 0.0], [2.0, 0.0, 0.0]])

    resolved, blend = _resolve_swing_targets_against_scene(
        scaled, world, block, clearance=0.01
    )

    # The free-space frame is untouched.
    assert blend[1] == pytest.approx(0.0)
    np.testing.assert_allclose(resolved[1], scaled[1])
    # The intersecting frame is faded out of the block, and no further.
    assert 0.0 < blend[0] < 1.0
    assert resolved[0][0] >= 0.51
    orientation_sign = resolve_scene_normal_orientation(block, world)
    clearance = scene_signed_clearance(
        block, resolved, orientation_sign
    )[0]
    assert np.all(clearance >= 0.01 - 1e-9)
    # The fade only moves along the scaled-to-world segment.
    np.testing.assert_allclose(resolved[0][1:], [0.0, 0.0], atol=1e-12)


def test_swing_scene_resolution_is_a_no_op_without_a_scene():
    from omniretargeting.core import _resolve_swing_targets_against_scene

    scaled = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    resolved, blend = _resolve_swing_targets_against_scene(
        scaled, np.zeros_like(scaled), None, clearance=0.01
    )
    np.testing.assert_allclose(resolved, scaled)
    np.testing.assert_allclose(blend, 0.0)


def test_swing_scene_resolution_never_uses_watertight_signed_distance(
    monkeypatch,
):
    """Open reconstructed meshes must use oriented closest-face clearance."""
    from omniretargeting.core import _resolve_swing_targets_against_scene

    scene = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    scene.update_faces(np.arange(len(scene.faces) - 1))
    scene.remove_unreferenced_vertices()
    assert not scene.is_watertight

    def reject_signed_distance(*_args, **_kwargs):
        raise AssertionError("signed_distance is invalid for this open mesh")

    monkeypatch.setattr(
        trimesh.proximity, "signed_distance", reject_signed_distance
    )
    scaled = np.array([[0.0, 0.0, 0.0]])
    world = np.array([[1.4, 0.0, 0.0]])
    resolved, blend = _resolve_swing_targets_against_scene(
        scaled, world, scene, clearance=0.01
    )

    orientation_sign = resolve_scene_normal_orientation(scene, world)
    clearance = scene_signed_clearance(
        scene, resolved, orientation_sign
    )[0]
    assert 0.0 < blend[0] < 1.0
    assert clearance[0] >= 0.01 - 1e-9
