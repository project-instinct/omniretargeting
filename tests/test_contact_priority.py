import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from omniretargeting.contact import (
    SceneContactTask,
    infer_scene_contact_tasks,
    stack_contact_tasks,
)
from omniretargeting.retargeting import (
    _apply_configured_joint_limits,
    _clamp_target_to_reach,
    _transform_swing_target,
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
)
from omniretargeting.utils import _densify_smplx_frame_gaps
from omniretargeting.data_sources.smplx import load_smplx_trajectory


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


def _task(targets, contact_mask=None):
    num_frames = len(targets)
    if contact_mask is None:
        contact_mask = np.zeros(num_frames, dtype=bool)
    return SceneContactTask(
        name="left_foot",
        tracking_positions=targets,
        contact_positions=targets,
        surface_points=targets,
        surface_normals=np.tile([0.0, 0.0, 1.0], (num_frames, 1)),
        distances=np.zeros(num_frames),
        relative_speeds=np.zeros(num_frames),
        contact_confidence=np.ones(num_frames),
        contact_mask=np.asarray(contact_mask, dtype=bool),
        forward_directions=np.tile([1.0, 0.0, 0.0], (num_frames, 1)),
        up_directions=np.tile([0.0, 0.0, 1.0], (num_frames, 1)),
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


def test_unit_scale_swing_target_still_follows_robot_root_registration():
    """Unit morphology scale must not suppress the robot root-height offset."""
    transformed = _transform_swing_target(
        tracking_position=np.array([1.2, -0.4, 1.7]),
        source_root_position=np.array([1.0, -0.5, 1.1]),
        robot_root_z_offset=-0.5,
        morphology_scale=1.0,
    )

    np.testing.assert_allclose(transformed, [1.2, -0.4, 1.2])


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
            "anchor_contacts": True,
        },
    )["left_foot"]

    assert np.array_equal(
        task.contact_mask,
        np.array([False, False, False, True, True, True, True, True, False, False]),
    )
    anchored = task.contact_positions[task.contact_mask]
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
            "anchor_contacts": True,
            "effectors": {
                "left_hand": {"anchor_contacts": False},
            },
        },
    )

    hand_targets = tasks["left_hand"].contact_positions
    foot_targets = tasks["left_foot"].contact_positions
    assert np.ptp(hand_targets[:, 0]) > 0.25
    np.testing.assert_allclose(
        foot_targets,
        np.repeat(foot_targets[0:1], len(foot_targets), axis=0),
        atol=1e-10,
    )


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

    payload = stack_contact_tasks(tasks)
    assert payload["contact_mask"].shape == (4, 2)
    assert payload["contact_targets_w"].shape == (4, 2, 3)
    assert payload["end_effector_reference_w"].shape == (4, 2, 3)


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

    assert np.all(task.contact_mask)


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
    contact_mask = np.zeros(len(trajectory), dtype=bool)
    contact_mask[3:11] = True

    # Two candidate keypoints straddle the target, so a per-frame nearest
    # search would flip between them as the ankle chatters.
    refined = refine_joint_trajectory(
        model,
        trajectory,
        {"left_foot": _task(targets, contact_mask)},
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
    for frame_idx in np.flatnonzero(contact_mask):
        data.qpos[:] = refined[frame_idx]
        mujoco.mj_forward(model, data)
        rotation = data.xmat[body_id].reshape(3, 3)
        candidates = data.xpos[body_id][None, :] + np.array(
            [[1.0, 0.0, 0.02], [1.0, 0.0, -0.02]]
        ) @ rotation.T
        errors.append(np.min(np.linalg.norm(candidates - targets[0], axis=1)))
    assert np.max(errors) < 0.05
    contact_jerk = np.diff(refined[contact_mask, 7], n=3)
    assert np.max(np.abs(contact_jerk)) < 0.01


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
