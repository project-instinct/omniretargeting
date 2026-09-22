"""Whole-trajectory refinement and endpoint fidelity regression tests."""

import numpy as np
import pytest
import trimesh

from omniretargeting.trajectory_refinement import (
    _penetration_depth_weight,
    collision_geom_ids,
    geom_surface_offsets,
    measure_scene_penetration,
    refine_joint_trajectory,
)
from tests._contact_fixtures import (
    _effector_config,
    _straight_arm_setup,
    _task,
)


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


def test_refinement_does_not_lock_transition_contact_tangent():
    """Approach/release frames constrain the surface side, not a false anchor."""
    mujoco = pytest.importorskip("mujoco")
    model, trajectory, _ = _straight_arm_setup(num_frames=10, amplitude=0.0)
    contact_mask = np.ones(len(trajectory), dtype=bool)
    targets = np.tile([1.0, 0.0, -0.5], (len(trajectory), 1))
    normals = np.tile([1.0, 0.0, 0.0], (len(trajectory), 1))
    common = {
        "iterations": 4,
        "contact_position_sigma": 0.01,
        "reference_sigma": 0.5,
        "acceleration_enabled": False,
        "jerk_enabled": False,
        "trust_region": 0.2,
    }

    transition = refine_joint_trajectory(
        model,
        trajectory,
        {
            "left_foot": _task(
                targets,
                contact_mask,
                contact_modes=np.full(len(trajectory), "transition"),
                surface_normals=normals,
            )
        },
        _effector_config(track_orientation=False),
        common,
    )
    sticking = refine_joint_trajectory(
        model,
        trajectory,
        {
            "left_foot": _task(
                targets,
                contact_mask,
                contact_modes=np.full(len(trajectory), "sticking"),
                surface_normals=normals,
            )
        },
        _effector_config(track_orientation=False),
        common,
    )

    np.testing.assert_allclose(transition[:, 7], 0.0, atol=1e-8)
    assert np.mean(sticking[:, 7]) > 0.2


def test_transition_fade_does_not_allow_contact_inside_surface():
    """An uncertain video contact may fade in, but cannot cross the surface."""
    mujoco = pytest.importorskip("mujoco")
    model, trajectory, _ = _straight_arm_setup(num_frames=8, amplitude=0.0)
    targets = np.tile([1.0, 0.0, 0.15], (len(trajectory), 1))
    task = _task(
        targets,
        np.ones(len(trajectory), dtype=bool),
        contact_modes=np.full(len(trajectory), "transition"),
        surface_normals=np.tile([0.0, 0.0, 1.0], (len(trajectory), 1)),
        transition_weights=np.full(len(trajectory), 1e-3),
    )

    refined = refine_joint_trajectory(
        model,
        trajectory,
        {"left_foot": task},
        _effector_config(track_orientation=False),
        {
            "iterations": 4,
            "contact_position_sigma": 0.02,
            "reference_sigma": 0.03,
            "acceleration_enabled": False,
            "jerk_enabled": False,
            "trust_region": 0.2,
        },
    )

    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "foot"
    )
    signed_offsets = []
    for qpos in refined:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        marker = data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ np.array(
            [1.0, 0.0, 0.0]
        )
        signed_offsets.append(marker[2] - targets[0, 2])
    assert np.min(signed_offsets) >= -0.007


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


def test_deep_penetration_samples_receive_bounded_priority_weight():
    config = {"depth_priority_start": 0.05, "max_depth_weight": 9.0}

    assert _penetration_depth_weight(-0.02, config) == pytest.approx(1.0)
    assert _penetration_depth_weight(-0.10, config) == pytest.approx(4.0)
    assert _penetration_depth_weight(-0.20, config) == pytest.approx(9.0)


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

