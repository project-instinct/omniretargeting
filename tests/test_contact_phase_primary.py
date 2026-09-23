"""Primary hand/foot contacts take roles without dense skin correspondence."""

import copy
from dataclasses import replace

import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting.contact_phase import attach_contact_phase
from omniretargeting.contact_phase_observation import observe_effector_clearances
from tests._contact_fixtures import _task
from tests.test_contact_phase import case
from tests.test_full_body_surfaces import _markers


def hand_case(*, stance=False, hand_environment=0, hand_mode="sticking"):
    """A hand anchor 12 cm from where the world-tracked foot allows."""
    model, q, _, tasks, priority = case(stance=stance)
    frames = len(q)
    tasks["left_hand"] = replace(
        _task(np.tile([0.12, 0, 1.0], (frames, 1)), np.ones(frames, bool)),
        name="left_hand",
        contact_modes=np.full(frames, hand_mode),
        environment_ids=np.full(frames, hand_environment),
    )
    priority["effectors"]["left_hand"] = {
        "robot_body": "abdomen",
        "tracking_offset": [0, 0, -0.1],
        "contact_offsets": [[0, 0, -0.1]],
        "contact_sigma": 0.001,
    }
    policy = attach_contact_phase(
        None,
        None,
        tasks,
        priority,
        model=model,
        source_clearances={"left_foot": np.full(frames, 0.2)},
    )
    return model, q, tasks, priority, policy


def solve_hand_case(**options):
    from omniretargeting.sequence_retargeting import retarget_sequence

    model, q, tasks, priority, policy = hand_case(**options)
    names = ["root", "abdomen", "foot"]
    positions = _markers(model, q, names)
    after, report = retarget_sequence(
        model,
        q,
        positions,
        names,
        {n: n for n in names},
        positions[:, 0],
        tasks,
        priority,
        framerate=30,
        config={
            "iterations": 6,
            "self_collision_enabled": False,
            "lock_root_xy": True,
            "lock_root_z": True,
            "lock_root_orientation": True,
            "world_tracking_sigma_m": 0.005,
        },
    )
    return model, q, tasks, priority, policy, after, report


@pytest.mark.parametrize("hand_environment", [0, 1])
def test_airborne_foot_yields_to_any_hand_contact(hand_environment):
    # Supporting on the box or following a moving object: both outrank a
    # foot observed clear of every surface. No other support is required.
    model, q, tasks, _, policy, after, report = solve_hand_case(
        hand_environment=hand_environment
    )
    reach = model.joint("reach").qposadr[0]
    assert np.max(np.abs(after[:, reach] - 0.12)) < 0.002
    np.testing.assert_array_equal(after[:, :7], q[:, :7])
    assert report["surface_endpoint_guard"]["linear_failures"] == 0
    assert policy.role("left_hand", 0) == ("support" if hand_environment == 0 else "follow")
    assert policy.yields("left_foot", 0)
    assert all(tasks[n].contact_phase is policy for n in tasks)
    np.testing.assert_array_equal(tasks["left_foot"].tracking_positions[:, 0], -0.3)
    np.testing.assert_array_equal(tasks["left_hand"].contact_positions[:, 0], 0.12)
    _, _, _, _, held_policy, held, _ = solve_hand_case(
        stance=True, hand_environment=hand_environment
    )
    assert held_policy.role("left_foot", 0) == "support"
    assert held[:, reach].max() <= 0.001 + 1e-6


def framewise(model, q, tasks, priority, policy):
    from omniretargeting.retargeting import GenericInteractionRetargeter

    scene = trimesh.creation.box()
    scene.apply_translation([10, 10, 10])
    mapping = {n: n for n in ("root", "abdomen", "foot")}
    solver = GenericInteractionRetargeter(
        model,
        mujoco.MjData(model),
        scene,
        mapping,
        1.0,
        q_a_init_idx=0,
        terrain_sample_points=4,
        self_collision_enabled=False,
        hard_penetration_constraint=False,
        end_effector_config=priority,
    )
    solver.contact_phase = policy
    result = solver.retarget_frame(
        _markers(model, q[:1], list(mapping))[0],
        q[0],
        max_iter=8,
        relation_frame=0,
        end_effector_targets={n: t.frame(0) for n, t in tasks.items()},
    )
    data = mujoco.MjData(model)
    data.qpos[:] = result
    mujoco.mj_forward(model, data)
    abdomen = model.body("abdomen").id
    hand = data.xpos[abdomen] + data.xmat[abdomen].reshape(3, 3) @ [0, 0, -0.1]
    return (
        np.linalg.norm(hand - tasks["left_hand"].contact_positions[0]),
        np.linalg.norm(
            data.xpos[model.body("foot").id] - tasks["left_foot"].tracking_positions[0]
        ),
    )


@pytest.mark.parametrize("free_priority", ["balanced", "lexicographic"])
def test_framewise_order_is_support_then_follow_then_free(free_priority):
    # A hand following a moving object, e.g. stroking or carrying it.
    model, q, tasks, priority, policy = hand_case(hand_environment=1)
    priority["free_motion_priority"] = free_priority
    hand, foot = framewise(model, q, tasks, priority, policy)
    assert hand < 0.003 and foot > 0.10
    # A standing foot carries the body: the following hand yields to it.
    model, q, tasks, priority, policy = hand_case(stance=True, hand_environment=1)
    priority["free_motion_priority"] = free_priority
    hand, foot = framewise(model, q, tasks, priority, policy)
    assert foot < 0.003 and hand > 0.10


@pytest.mark.parametrize("mode", ["sliding", "transition"])
def test_surface_only_contact_follows_without_locking_tangent(mode):
    from omniretargeting._sequence.coordinates import Coordinates
    from omniretargeting._sequence.surface_guard import SurfaceSequenceGuard
    from omniretargeting._sequence.tasks import SequenceTasks
    from omniretargeting.sequence_retargeting import SEQUENCE_DEFAULTS

    model, q, tasks, priority, policy = hand_case(hand_mode=mode)
    original = tasks["left_hand"].contact_positions.copy()
    assert policy.role("left_hand", 0) == "follow"
    assert policy.yields("left_foot", 0)
    names = ["root", "abdomen", "foot"]
    positions = _markers(model, q, names)
    task_set = SequenceTasks(
        model,
        Coordinates(model),
        q,
        positions,
        names,
        {n: n for n in names},
        positions[:, 0],
        tasks,
        priority,
        {**SEQUENCE_DEFAULTS, "self_collision_enabled": False},
    )
    guard = SurfaceSequenceGuard(task_set, q, 0.001)
    along_surface = q.copy()
    along_surface[:, model.joint("reach").qposadr[0]] = 0.12
    # The hand travels 12 cm tangentially; both poses satisfy its normal task.
    np.testing.assert_allclose(guard.priority_errors(q)[:, 3], 0, atol=1e-9)
    np.testing.assert_allclose(guard.priority_errors(along_surface)[:, 3], 0, atol=1e-9)
    away_from_surface = along_surface.copy()
    away_from_surface[:, 2] += 0.03
    np.testing.assert_allclose(
        guard.priority_errors(away_from_surface)[:, 3], 0.03, atol=1e-9
    )
    np.testing.assert_allclose(guard.priority_errors(away_from_surface)[:, :3], 0)
    np.testing.assert_array_equal(tasks["left_hand"].contact_positions, original)
    np.testing.assert_array_equal(tasks["left_hand"].contact_modes, mode)


def test_clearance_requires_all_samples_and_all_surfaces_and_preserves_source():
    scene = trimesh.creation.box(extents=[2, 2, 1])
    scene.apply_translation([0, 0, 0.5])
    points = np.array(
        [
            [[0, 0, 1.2], [0.1, 0, 1.2]],
            [[0, 0, 1.2], [0.1, 0, 1.01]],
            [[0, 0, 1.2], [0.1, 0, 0.8]],
            [[0, 0, np.nan], [0.1, 0, 1.2]],
        ]
    )
    markers = {"left_foot": {"contact_samples": points}}
    original = copy.deepcopy(markers)
    priority = {"effectors": {"left_foot": {"preserve_world_target": True}}}
    distances = observe_effector_clearances(markers, scene, priority)["left_foot"]
    np.testing.assert_allclose(distances[:2], [0.2, 0.01])
    assert np.isnan(distances[2:]).all()
    np.testing.assert_array_equal(points, original["left_foot"]["contact_samples"])
    obstacle = trimesh.creation.box(extents=[0.4, 0.4, 0.4])
    obstacle.apply_translation([0, 0, 1.2])
    combined = observe_effector_clearances(
        markers, scene, priority, scene_objects=[{"mesh": obstacle}]
    )["left_foot"]
    assert np.isnan(combined).all()
    assert np.isnan(
        observe_effector_clearances(markers, None, priority)["left_foot"]
    ).all()


def test_inward_wound_closed_room_has_free_interior_clearance():
    box = trimesh.creation.box(extents=[2, 2, 2])
    room = trimesh.Trimesh(
        vertices=box.vertices, faces=box.faces[:, ::-1], process=False
    )
    markers = {"left_foot": {"contact_samples": np.array([
        [[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.8]],
    ])}}
    priority = {"effectors": {"left_foot": {"preserve_world_target": True}}}
    observed = observe_effector_clearances(markers, room, priority)["left_foot"]
    np.testing.assert_allclose(observed, [1.0, 0.2])


def test_unknown_clearance_keeps_the_foot_guarded():
    model, _, tasks, priority, _ = hand_case()
    unknown = attach_contact_phase(None, None, tasks, priority, model=model)
    assert unknown.role("left_foot", 0) == "free"
    assert not unknown.yields("left_foot", 0)
    attach_contact_phase(None, None, tasks, priority, {"enabled": False}, model=model)
    assert all(t.contact_phase is None for t in tasks.values())


def test_roles_survive_refinement_and_whole_body_guards():
    from scipy import sparse

    from omniretargeting._refinement.guard import RefinementGuard
    from omniretargeting.scene_geometry import SceneQuery
    from omniretargeting.trajectory_validation import (
        TrajectoryValidator,
        local_repair_config,
    )
    from omniretargeting.whole_body_correction import whole_body_config
    from omniretargeting.whole_body_guard import WholeBodyGuard, WholeBodyValidator

    model, q, tasks, priority, policy, after, _ = solve_hand_case()
    scene = trimesh.creation.box()
    scene.apply_translation([10, 10, 10])
    args = (
        model,
        after,
        SceneQuery.from_tasks(scene, tasks),
        tasks,
        priority,
        local_repair_config({"self_collision_enabled": False}),
        30,
    )
    validators = [
        TrajectoryValidator(*args),
        WholeBodyValidator(
            *args, whole_body_config=whole_body_config({"trust_region": 0.15})
        ),
    ]
    reach = model.joint("reach").qposadr[0]
    for validator in validators:
        assert validator.contact_phase is policy
        assert np.isfinite(validator.swing_budget("left_foot", 0))
        assert validator.fidelity(after)["ok"]
        with validator.segment(2, 5) as window:
            assert window.contact_phase is policy
            assert window.fidelity(after[2:5])["ok"]
        # A repair request that pulls the hand off its anchor is bounded.
        target = after.copy()
        target[:, reach] = 0.0
        matrix = sparse.eye(len(q) * len(validator.qi), format="csr")
        rhs = (target[:, validator.qi] - after[:, validator.qi]).ravel()
        if isinstance(validator, WholeBodyValidator):
            step, _ = WholeBodyGuard(validator).solve(
                matrix, rhs, after, np.ones(len(validator.qi), bool)
            )
        else:
            proposal = RefinementGuard(validator).solve(
                matrix, rhs, after, np.full(len(rhs), -0.15), np.full(len(rhs), 0.15)
            )
            step = None if proposal is None else proposal.x.reshape(len(q), -1)
        assert step is not None
        candidate = after.copy()
        candidate[:, validator.qi] += step
        assert candidate[:, reach].min() > 0.10
        assert validator.fidelity(candidate)["ok"]


def test_core_attaches_primary_roles_without_dense_correspondence():
    from types import SimpleNamespace

    from omniretargeting._core.surfaces import prepare_surface_tasks

    model, q, tasks, priority, _ = hand_case()
    tasks = {name: replace(task, contact_phase=None) for name, task in tasks.items()}
    scene = trimesh.creation.box()
    scene.apply_translation([0.62, 0, 0.5])
    owner = SimpleNamespace(
        robot_model=model,
        valid_joint_mapping={"root": "root"},
        retargeting_config={"end_effector_priority": priority},
        last_end_effector_tasks=tasks,
        _surface_tasks_prepared=True,
        _surface_contacts_built=True,
        _surface_pose_built=True,
        _learned_surface_motion=None,
        last_surface_pose_tasks=None,
        _contact_phase_markers={
            "left_foot": {"contact_samples": tasks["left_foot"].tracking_positions}
        },
    )
    solver = SimpleNamespace(robot_reference_qpos=q[0])
    positions = np.zeros((len(q), 1, 3))
    prepare_surface_tasks(
        owner, solver, positions, scene, None, apply_scene_contacts=False
    )
    assert solver.contact_phase is None
    assert all(t.contact_phase is None for t in tasks.values())
    prepare_surface_tasks(owner, solver, positions, scene, None)
    assert solver.contact_phase is owner.last_contact_phase
    assert solver.contact_phase.role("left_hand", 0) == "support"
    assert solver.contact_phase.yields("left_foot", 0)
    assert all(t.contact_phase is solver.contact_phase for t in tasks.values())


def test_free_hand_is_handed_over_to_its_anchor_without_an_onset_jump(monkeypatch):
    from omniretargeting.contact import SceneContactTask
    from omniretargeting.retargeting import GenericInteractionRetargeter

    model, _, _, _, _ = case()
    frames = 8
    q = np.tile(model.qpos0, (frames, 1))
    # The source hand reaches the anchor at x=.12 on frame 5 and stays there.
    source = np.zeros((frames, 3))
    source[:, 0] = [0, 0, 0.03, 0.06, 0.09, 0.12, 0.12, 0.12]
    source[:, 2] = 1.0
    hand = replace(_task(source, np.arange(frames) >= 5), name="left_hand")
    priority = {
        "enabled": True,
        "hierarchy": True,
        "pose_direction": {"enabled": False},
        "effectors": {
            "left_hand": {
                "robot_body": "abdomen",
                "tracking_offset": [0, 0, -0.1],
                "contact_offsets": [[0, 0, -0.1]],
                "contact_sigma": 0.001,
                "track_swing": True,
                # Morphology: the free target is not the source world path.
                "swing_morphology_scale": 0.5,
            }
        },
    }
    mapping = {n: n for n in ("root", "abdomen", "foot")}
    markers = _markers(model, q, list(mapping))
    scene = trimesh.creation.box()
    scene.apply_translation([10, 10, 10])
    reach = model.joint("reach").qposadr[0]

    def solve():
        solver = GenericInteractionRetargeter(
            model, mujoco.MjData(model), scene, mapping, 1.0, q_a_init_idx=0,
            terrain_sample_points=4, self_collision_enabled=False,
            hard_penetration_constraint=False, end_effector_config=priority,
        )
        pose, result = q[0].copy(), []
        for f in range(frames):
            payload = {**hand.frame(f), "source_root_position": np.zeros(3)}
            pose = solver.retarget_frame(
                markers[f], pose, max_iter=8, relation_frame=f,
                end_effector_targets={"left_hand": payload},
            )
            result.append(pose[reach])
        return np.array(result)

    handed_over = solve()
    monkeypatch.setattr(
        SceneContactTask, "approach",
        property(lambda task: (np.zeros(len(task.contact_mask)), task.tracking_positions)),
    )
    switched = solve()
    # Both hold the anchor from the first contact frame.
    np.testing.assert_allclose(handed_over[5:], 0.12, atol=2e-3)
    np.testing.assert_allclose(switched[5:], 0.12, atol=2e-3)
    # A hard switch jumps the whole 12 cm at the onset frame; the hand-over
    # follows the source's own approach (0, .25, .5, .75 of the way).
    assert np.diff(switched)[4] > 0.1
    assert np.all(np.diff(handed_over)[1:4] > 0)
    assert np.max(np.diff(handed_over)) < 0.07


def test_non_penetration_outranks_a_support_target_inside_the_scene():
    # Like SMPL-X feet through the floor: the source anchor lies inside a
    # wall. The body stops at the wall and reports the hand residual.
    from omniretargeting.scene_geometry import SceneQuery
    from omniretargeting.sequence_retargeting import retarget_sequence

    model, q, tasks, priority, policy = hand_case()
    wall = trimesh.creation.box(extents=[0.8, 2, 3])
    wall.apply_translation([0.60, 0, 1.0])  # face at x=.2; skin reaches it at .08
    names = ["root", "abdomen", "foot"]
    positions = _markers(model, q, names)
    after, report = retarget_sequence(
        model, q, positions, names, {n: n for n in names}, positions[:, 0],
        tasks, priority, framerate=30, scene_query=SceneQuery.from_tasks(wall, tasks),
        config={"iterations": 6, "self_collision_enabled": False, "lock_root_xy": True,
                "lock_root_z": True, "lock_root_orientation": True,
                "world_tracking_sigma_m": 0.005},
    )
    assert policy.role("left_hand", 0) == "support"
    assert report["final_residuals"]["scene_depth_m"] < 1e-6
    reach = after[:, model.joint("reach").qposadr[0]]
    np.testing.assert_allclose(reach, 0.08, atol=2e-3)
    np.testing.assert_array_equal(tasks["left_hand"].contact_positions[:, 0], 0.12)
