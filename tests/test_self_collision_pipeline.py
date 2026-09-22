"""Regression cases for warm starts, playback paths and independent repair."""

import mujoco
import numpy as np
import trimesh
from types import SimpleNamespace

from omniretargeting.local_collision_repair import repair_local_collisions
from omniretargeting.retargeting import GenericInteractionRetargeter
from omniretargeting.self_collision import SelfCollisionQuery, summarize_self_collisions
from omniretargeting.self_collision_projection import project_self_collision_trajectory
from omniretargeting.trajectory_refinement import refine_joint_trajectory
from omniretargeting.trajectory_validation import TrajectoryValidator, local_repair_config
from omniretargeting.scene_geometry import SceneQuery
from tests.test_self_collision import XML
from tests._contact_fixtures import _task


def motion_driver(model):
    from omniretargeting._core.motion import FramewiseMotionMixin

    driver = FramewiseMotionMixin()
    driver.robot_model, driver.robot_data = model, mujoco.MjData(model)
    driver.robot_height = 1.0
    driver.retargeting_config = {}
    driver.valid_joint_mapping = {"root": "root"}
    driver.valid_joint_names = ["root"]
    driver.mapped_joint_indices = [0]
    driver.smplx_joint_indices = {"root": 0}
    driver._resolve_source_root_index = lambda: 0
    driver._estimate_base_orientation_from_joints = lambda *_args: None
    return driver


def test_contact_transition_does_not_blend_a_valid_solve_into_collision(monkeypatch):
    model = mujoco.MjModel.from_xml_string(XML)
    baseline = model.qpos0.copy()
    baseline[7] = -0.2
    solved = baseline.copy()
    solved[7] = 0.38
    query = SelfCollisionQuery(model)
    assert max(query.depths(0.5 * (baseline + solved)).values()) > 0.1
    fake = SimpleNamespace(retarget_frame=lambda *_args, **_kwargs: solved.copy())
    monkeypatch.setattr("omniretargeting.retargeting.GenericInteractionRetargeter",
                        lambda *_args, **_kwargs: fake)
    task = _task(np.zeros((1, 3)), np.ones(1, bool), ["transition"],
                 transition_weights=np.array([0.5]))
    result = motion_driver(model)._perform_retargeting(
        np.zeros((1, 1, 3)), trimesh.creation.box(),
        end_effector_tasks={"left_hand": task}, reference_trajectory=baseline[None],
        overlay_contact_names={"left_hand"},
    )
    np.testing.assert_array_equal(result[0], solved)
    assert max(query.depths(result[0]).values(), default=0) == 0


def test_contact_release_checks_the_splice_before_reusing_baseline(monkeypatch):
    model = mujoco.MjModel.from_xml_string(XML)
    baseline = np.tile(model.qpos0, (2, 1))
    baseline[:, 7] = [-0.2, 0.38]
    safe = baseline[0].copy()
    safe[7] = -0.045
    fake = GenericInteractionRetargeter.__new__(GenericInteractionRetargeter)
    fake.robot_model, fake.self_collision_enabled = model, True
    calls = []

    def solve(*_args, **kwargs):
        calls.append(kwargs)
        return baseline[0].copy() if len(calls) == 1 else safe.copy()

    fake.retarget_frame = solve
    monkeypatch.setattr("omniretargeting.retargeting.GenericInteractionRetargeter",
                        lambda *_args, **_kwargs: fake)
    task = _task(np.zeros((2, 3)), np.array([True, False]))
    result = motion_driver(model)._perform_retargeting(
        np.zeros((2, 1, 3)), trimesh.creation.box(),
        end_effector_tasks={"left_hand": task}, reference_trajectory=baseline,
        overlay_contact_names={"left_hand"},
    )
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[1]["q_last"], result[0])
    audit = summarize_self_collisions(fake._get_self_collision_query(), result, include_interframes=True)
    assert audit["acceptance_status"] == "passed"


def crossing_motion(n=9):
    xml = XML.replace('<geom name="left_collision"',
                      '<joint name="left_y" type="slide" axis="0 1 0" range="-0.5 0.5"/>'
                      '<geom name="left_collision"')
    xml = xml.replace('<geom name="right_collision"',
                      '<joint name="right_y" type="slide" axis="0 1 0" range="-0.5 0.5"/>'
                      '<geom name="right_collision"')
    model = mujoco.MjModel.from_xml_string(xml)
    q = np.tile(model.qpos0, (n, 1))
    q[n // 2 + 1:, 7] = 0.32
    q[:, 8] = 0.195
    return model, q


def test_invalid_warm_start_cannot_become_allowed_path_penetration():
    model = mujoco.MjModel.from_xml_string(XML)
    solver = GenericInteractionRetargeter.__new__(GenericInteractionRetargeter)
    solver.robot_model = model
    solver.self_collision_enabled = True
    query = solver._get_self_collision_query()
    previous = model.qpos0.copy()
    previous[7] = -0.2
    initial = previous.copy()
    initial[7] = 0.38
    initial[:3] = [2, 3, 4]
    assert max(query.segment(previous, initial)["depths"].values()) > 0.1
    seed = solver._self_collision_initial_guess(initial, previous)
    assert max(query.segment(previous, seed)["depths"].values(), default=0) <= 0.001001
    np.testing.assert_array_equal(seed[:7], initial[:7])
    # Also protect callers that invoke a single step with an invalid baseline.
    guarded = solver._guard_self_collision_step(initial, initial, previous)
    assert max(query.segment(previous, guarded)["depths"].values(), default=0) <= 0.001001


def test_trajectory_refinement_actively_repairs_between_frame_only_collision():
    model, before = crossing_motion()
    query = SelfCollisionQuery(model)
    old = summarize_self_collisions(query, before, include_interframes=True)
    assert old["max_depth"] == 0
    assert old["interframe_max_depth"] > 0.004
    after = refine_joint_trajectory(model, before, {}, {}, {
        "iterations": 8, "self_collision_enabled": True, "trust_region": 0.02,
    })
    checked = summarize_self_collisions(query, after, include_interframes=True)
    assert checked["acceptance_status"] == "passed"
    np.testing.assert_array_equal(after[:, :7], before[:, :7])


def test_unknown_scene_and_contact_switch_do_not_skip_self_repair():
    model, before = crossing_motion()
    # Open, unregistered scene sides are explicitly unknown. They are retained
    # as such while robot self distances are fully known.
    scene = trimesh.Trimesh(vertices=[[10, 0, 0], [10, 1, 0], [10, 0, 1]],
                            faces=[[0, 1, 2]], process=False)
    data = mujoco.MjData(model)
    targets = []
    for q in before:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        targets.append(data.xpos[model.body("left").id].copy())
    task = _task(np.asarray(targets), np.arange(len(before)) >= 4)
    priority = {"effectors": {"left_hand": {"robot_body": "left"}}}
    cfg = {"self_collision_enabled": True, "max_query_points": None, "max_substeps": 256}
    after, report = repair_local_collisions(
        model, before, scene, {"left_hand": task}, priority, cfg, framerate=30,
    )
    events = [e for e in report["events"] if e.get("focus") == "self"]
    assert events and any(e["iterations"] > 0 for e in events)
    assert report["before"]["uncertain_frames"]
    assert report["after"]["uncertain_frames"]  # Never claim the scene was repaired.
    assert report["fidelity"]["ok"]
    audit = summarize_self_collisions(SelfCollisionQuery(model), after, include_interframes=True)
    assert audit["acceptance_status"] == "passed"
    np.testing.assert_array_equal(after[:, :7], before[:, :7])
    np.testing.assert_array_equal(after[[0, -1]], before[[0, -1]])
    np.testing.assert_array_equal(task.contact_mask, np.arange(len(before)) >= 4)


def test_self_repair_cannot_hide_new_scene_uncertainty_or_reliable_penetration():
    model, before = crossing_motion()
    scene = trimesh.creation.box(extents=[0.01] * 3)
    scene.apply_translation([10, 0, 0])
    v = TrajectoryValidator(model, before, SceneQuery(scene), {}, {},
                            local_repair_config({"self_collision_enabled": True}), 30)
    stats = v.geometry(before)
    changed = dict(stats)
    changed["uncertain_geoms"] = stats["uncertain_geoms"].copy()
    changed["uncertain_geoms"][4, 1] = True
    assert "new_uncertain_scene_side" in v.compare_geometry(stats, changed, focus_self=True)
    changed = dict(stats)
    changed["reliable_scene_depth"] = stats["reliable_scene_depth"].copy()
    changed["reliable_scene_depth"][4, 1] = 0.02
    assert "collision_regression" in v.compare_geometry(stats, changed, focus_self=True)


def test_final_audit_distinguishes_residual_from_incomplete_sampling():
    model, q = crossing_motion()
    residual = summarize_self_collisions(SelfCollisionQuery(model), q, include_interframes=True)
    assert residual["acceptance_status"] == "residual"
    assert residual["intervals_over_tolerance"] == [4]
    incomplete = summarize_self_collisions(SelfCollisionQuery(model, {"max_substeps": 1}),
                                          q, include_interframes=True)
    assert incomplete["acceptance_status"] == "incomplete"
    assert incomplete["subdivision_limit_intervals"] == [4]


def test_clear_coupled_proposal_is_exact_noop_without_querying_scene():
    model, q = crossing_motion()
    q[:, 8] = 0.21
    result, report = project_self_collision_trajectory(model, q, object(), {}, {}, {})
    np.testing.assert_array_equal(result, q)
    assert report["status"] == "passed"
    assert report["events"] == []


def test_full_framewise_pass_projects_its_proposals_before_returning(monkeypatch):
    model, proposal = crossing_motion()
    fake = GenericInteractionRetargeter.__new__(GenericInteractionRetargeter)
    fake.robot_model, fake.self_collision_enabled = model, True
    frames = iter(proposal)
    fake.retarget_frame = lambda *_args, **_kwargs: next(frames).copy()
    monkeypatch.setattr("omniretargeting.retargeting.GenericInteractionRetargeter",
                        lambda *_args, **_kwargs: fake)
    driver = motion_driver(model)
    driver._coupled_self_collision_projection = True
    driver._projection_framerate = 30.0
    driver._self_collision_projection_config = {"local_collision_repair": {"max_substeps": 256}}
    driver.last_framewise_self_collision_reports = []
    driver._validation_foot_references = lambda _q, tasks, priority: (tasks, priority, [])
    scene = trimesh.creation.box(extents=[0.01] * 3)
    scene.apply_translation([10, 0, 0])
    result = driver._perform_retargeting(np.zeros((len(proposal), 1, 3)), scene)
    audit = summarize_self_collisions(SelfCollisionQuery(model), result, include_interframes=True)
    assert audit["acceptance_status"] == "passed"
    assert driver.last_framewise_self_collision_reports[0]["before_max_depth"] > 0.004
    assert driver.last_framewise_self_collision_reports[0]["status"] == "passed"
    np.testing.assert_array_equal(result[:, :7], proposal[:, :7])


def test_self_projection_does_not_spend_slack_on_an_already_acceptable_scene_gap():
    model, before = crossing_motion()
    # Right sphere is at the accepted 1 mm wall boundary. Separating the two
    # bodies must use the left hand's free tangential direction, rather than
    # repeatedly proposing a right-hand move through that wall.
    scene = trimesh.creation.box(extents=[2.0, 0.2, 2.0])
    scene.apply_translation([0, -0.199, 0])
    positions = np.column_stack([before[:, 7] - 0.08, before[:, 8], np.zeros(len(before))])
    task = _task(positions, np.ones(len(before), bool), ["sliding"] * len(before))
    after, report = project_self_collision_trajectory(
        model, before, scene, {"left_hand": task},
        {"effectors": {"left_hand": {"robot_body": "left"}}},
        {"local_collision_repair": {"max_substeps": 256, "max_query_points": None}},
        scene_query=SceneQuery(scene),
    )
    assert report["status"] == "passed"
    assert report["fidelity"]["ok"]
    assert np.min(after[:, 10]) >= -1e-6
    assert np.max(after[:, 8] - before[:, 8]) > 0.004
