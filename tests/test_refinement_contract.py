"""Behavior at the framewise/refinement/acceptance boundary."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest
import trimesh
from scipy import sparse

from omniretargeting._refinement.guard import RefinementGuard
from omniretargeting.local_collision_repair import select_safe_postprocessing
from omniretargeting.scene_geometry import SceneQuery
from omniretargeting.trajectory_refinement import refine_joint_trajectory
from omniretargeting.trajectory_validation import (
    TrajectoryValidator,
    local_repair_config,
)
from tests._contact_fixtures import _effector_config, _straight_arm_setup, _task


def settings(**overrides):
    return dict(
        iterations=4,
        self_collision_enabled=False,
        reference_sigma=2.0,
        acceleration_enabled=False,
        jerk_enabled=False,
        trust_region=0.12,
        **overrides,
    )


def test_free_hand_position_switch_keeps_contact_and_orientation_tasks():
    model, q, targets = _straight_arm_setup(amplitude=0.0)
    targets[:, 2] = -0.4
    priority = _effector_config(track_orientation=False)
    priority["effectors"]["left_foot"]["track_swing"] = False
    tasks = {"left_foot": _task(targets)}
    free = refine_joint_trajectory(model, q, tasks, priority, settings())
    np.testing.assert_array_equal(free, q)

    contact = refine_joint_trajectory(
        model,
        q,
        {"left_foot": _task(targets, np.ones(len(q), bool))},
        priority,
        settings(),
    )
    assert np.max(contact[:, 7]) > 0.1

    # With position still disabled, a foot/hand orientation task must survive.
    priority["effectors"]["left_foot"]["track_orientation"] = True
    tilted = replace(
        tasks["left_foot"],
        forward_directions=np.tile([np.cos(0.2), 0, -np.sin(0.2)], (len(q), 1)),
        up_directions=np.tile([np.sin(0.2), 0, np.cos(0.2)], (len(q), 1)),
    )
    oriented = refine_joint_trajectory(
        model, q, {"left_foot": tilted}, priority, settings()
    )
    assert np.mean(oriented[:, 7]) > 0.15


def test_free_hand_position_switch_keeps_arm_shape():
    model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/><worldbody>
      <body name="shoulder"><freejoint/><geom size=".01" mass="1"/>
      <body name="upper"><joint name="shoulder_y" axis="0 1 0" range="-1 1"/>
      <geom type="capsule" fromto="0 0 0 .4 0 0" size=".01"/>
      <body name="elbow" pos=".4 0 0"><joint name="elbow_y" axis="0 1 0" range="-1 1"/>
      <geom type="capsule" fromto="0 0 0 .4 0 0" size=".01"/>
      <body name="wrist" pos=".4 0 0"/></body></body></body></worldbody></mujoco>""")
    q = np.tile(model.qpos0, (6, 1))
    source = np.tile([[0, 0, 0], [0.4, 0, -0.15], [0.65, 0, -0.4]], (6, 1, 1))
    priority = {
        "effectors": {
            "hand": {
                "robot_body": "wrist",
                "track_swing": False,
                "arm_posture": {
                    "enabled": True,
                    "shoulder_body": "shoulder",
                    "elbow_body": "elbow",
                    "wrist_body": "wrist",
                    "source_shoulder": "S",
                    "source_elbow": "E",
                    "source_wrist": "W",
                    "upper_arm_length": 0.4,
                    "forearm_length": 0.4,
                },
            }
        }
    }
    after = refine_joint_trajectory(
        model,
        q,
        {"hand": _task(source[:, 2])},
        priority,
        settings(),
        source_joint_positions=source,
        source_joint_names=["S", "E", "W"],
    )
    assert np.max(np.abs(after[:, 7:] - q[:, 7:])) > 0.1
    np.testing.assert_array_equal(after[:, :7], q[:, :7])


def guarded_setup(*, world_foot=False):
    model, q, targets = _straight_arm_setup(num_frames=6, amplitude=0.0)
    q[:, 2] = 1.0
    targets[:, 2] = 0.6
    task = _task(targets)
    if world_foot:
        task = replace(task, distances=np.full(len(q), np.inf))
    priority = _effector_config(track_orientation=False)
    floor = trimesh.creation.box(extents=[4, 4, 0.2])
    floor.apply_translation([0, 0, -0.1])
    cfg = local_repair_config({"self_collision_enabled": False})
    v = TrajectoryValidator(
        model, q, SceneQuery(floor), {"left_foot": task}, priority, cfg, 30
    )
    return model, q, floor, task, priority, v


def test_guarded_iterations_obey_cumulative_and_world_foot_budgets():
    model, q, floor, task, priority, v = guarded_setup(world_foot=True)
    guard = RefinementGuard(v)
    after = refine_joint_trajectory(
        model,
        q,
        {"left_foot": task},
        priority,
        settings(swing_position_sigma=0.005),
        scene_mesh=floor,
        scene_query=v.query,
        framerate=30,
        refinement_guard=guard,
    )
    assert v.fidelity(after)["ok"]
    np.testing.assert_array_equal(after[:, :7], q[:, :7])
    assert 0 < np.max(np.abs(after[:, 7] - q[:, 7])) <= 0.0010001
    final, _, report = select_safe_postprocessing(
        model,
        q,
        after,
        floor,
        {"left_foot": task},
        priority,
        v.cfg,
        framerate=30,
        scene_query=v.query,
    )
    np.testing.assert_array_equal(final, after)
    assert report["accepted_fraction"] == 1.0


def test_cumulative_budget_is_not_per_iteration_trust_region():
    model, q, floor, task, priority, v = guarded_setup()
    v.cfg["swing_position_budget"] = 1.0
    guard = RefinementGuard(v)
    after = refine_joint_trajectory(
        model,
        q,
        {"left_foot": task},
        priority,
        settings(swing_position_sigma=0.005),
        scene_mesh=floor,
        scene_query=v.query,
        refinement_guard=guard,
    )
    delta = np.max(np.abs(after[:, 7] - q[:, 7]))
    assert 0.19 < delta <= v.cfg["max_joint_correction"] + 1e-8
    assert v.fidelity(after)["ok"]


def test_scene_constraint_changes_direction_before_nonlinear_acceptance():
    model, q, _floor, task, priority, v = guarded_setup()
    q[:, 2] = 0.02  # Capsule rests on floor; positive ankle pitch penetrates it.
    priority["effectors"]["left_foot"]["track_swing"] = False
    v = TrajectoryValidator(model, q, v.query, {"left_foot": task}, priority, v.cfg, 30)
    guard = RefinementGuard(v)
    matrix = sparse.identity(len(q), format="csr")
    proposal = guard.solve(
        matrix, np.full(len(q), 0.1), q, np.full(len(q), -0.12), np.full(len(q), 0.12)
    )
    assert proposal is not None
    assert np.max(proposal.x) < 0.002
    candidate = q.copy()
    candidate[:, 7] += proposal.x
    accepted = guard.accept(q, candidate, matrix, np.full(len(q), 0.1))
    assert v.fidelity(accepted)["ok"]
    assert not v.compare_geometry(
        guard.before, v.geometry(accepted), allow_existing_uncertainty=True
    )


def test_guard_rejects_a_different_reference():
    model, q, _, task, priority, v = guarded_setup()
    other = q.copy()
    other[:, 2] += 0.001
    with pytest.raises(ValueError, match="same model and input trajectory"):
        refine_joint_trajectory(
            model,
            other,
            {"left_foot": task},
            priority,
            settings(),
            refinement_guard=RefinementGuard(v),
        )


def test_bad_smoothing_keeps_the_valid_refinement_instead_of_raw_baseline():
    _model, q, _, _, _, v = guarded_setup(world_foot=True)
    guard = RefinementGuard(v)
    safe = q.copy()
    safe[:, 7] += 0.0003
    assert v.fidelity(safe)["ok"]
    bad = safe.copy()
    bad[3, 7] += 0.05
    after = guard.preserve_safe_smoothing(safe, bad)
    np.testing.assert_array_equal(after, safe)
    assert not np.array_equal(after, q)
    assert guard.report["optional_smoothing"]["accepted"] is False
