"""Small metric translation chains make scene and interpolation truth explicit."""

import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting.collision_window import _window_stats
from omniretargeting.local_collision_repair import (
    TrajectoryValidator,
    local_repair_config,
    repair_local_collisions,
)
from omniretargeting.scene_geometry import SceneQuery
from tests.test_contact_priority import _task


def setup_motion(n=9):
    model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
    <worldbody><body name="root"><freejoint/>
    <geom size="0.01" mass="1" contype="0" conaffinity="0"/>
    <body name="foot"><joint name="x" type="slide" axis="1 0 0" range="-1 1"/>
    <joint name="y" type="slide" axis="0 1 0" range="-1 1"/>
    <geom name="toe" type="sphere" size="0.02"/></body>
    <body name="other"><joint name="unrelated"/><geom size="0.01" pos="0 0 1"/>
    </body></body></worldbody></mujoco>""")
    q = np.tile(model.qpos0, (n, 1))
    q[:, 7] = np.linspace(-0.2, 0.2, n)
    q[:, 8] = 0.1
    box = trimesh.creation.box(extents=[0.12] * 3)
    return model, q, box


def test_short_collision_interpolates_and_leaves_root_other_joints_and_outside_exact():
    model, q, box = setup_motion()
    q[3:6, 8] = 0.04
    before = q.copy()
    result, report = repair_local_collisions(model, q, box, {}, {}, {}, framerate=30)
    assert report["events"][0]["method"] == "interpolation"
    assert report["after"]["max_depth"] <= 1e-6
    assert report["before"]["max_depth"] > 0.02
    np.testing.assert_array_equal(q, before)
    np.testing.assert_array_equal(result[:, :7], q[:, :7])
    np.testing.assert_array_equal(result[:, 9:], q[:, 9:])
    np.testing.assert_array_equal(result[[0, 1, 2, 6, 7, 8]], q[[0, 1, 2, 6, 7, 8]])


def test_local_repair_rejects_update_outside_cumulative_budget():
    model, q, box = setup_motion()
    q[3:6, 8] = 0.04
    result, report = repair_local_collisions(
        model, q, box, {}, {}, {}, framerate=30,
        candidate_acceptance=lambda candidate: np.array_equal(candidate, q),
    )
    np.testing.assert_array_equal(result, q)
    assert report["events"][0]["method"] == "fallback"
    assert report["events"][0]["reason"] == "fixed_sequence_budget"


def test_safe_motion_is_exact_noop_and_disabled_is_noop():
    model, q, box = setup_motion()
    result, report = repair_local_collisions(model, q, box, {}, {}, {}, framerate=30)
    np.testing.assert_array_equal(q, result)
    assert report["events"] == []
    q[4, 8] = 0
    result, report = repair_local_collisions(
        model, q, box, {}, {}, {"enabled": False}, framerate=30
    )
    np.testing.assert_array_equal(q, result)
    assert report["status"] == "disabled"


def test_saved_frames_safe_but_intermediate_pose_collides():
    model, q, box = setup_motion(2)
    q[:, 8] = 0
    v = TrajectoryValidator(
        model, q, SceneQuery(box), {}, {}, local_repair_config({}), 30
    )
    check = v.geometry(q)
    assert check["max_depth"] == 0
    assert check["interframe_max_depth"] > 0.04
    assert check["interframe_samples"] > 0
    assert check["method"] == "adaptive_discrete"


def test_window_geometry_matches_full_timeline_with_boundary_intervals():
    model, q, box = setup_motion(12)
    q[4:7, 8] = 0.04
    v = TrajectoryValidator(model, q, SceneQuery(box), {}, {}, local_repair_config({}), 30)
    full = _window_stats(v.geometry(q), 3, 7, model)
    with v.segment(2, 9) as view:
        local = _window_stats(view.geometry(q[2:9]), 1, 5, model)
    for key in (
        "per_geom_depth", "interframe_depth", "self_depth", "interframe_self_depth",
        "reliable_scene_depth", "interframe_reliable_scene_depth",
    ):
        np.testing.assert_allclose(local[key], full[key])
    assert local["max_depth"] == full["max_depth"]
    assert local["interframe_max_depth"] == full["interframe_max_depth"]


def test_reused_intervals_match_fresh_validation_after_one_pose_changes():
    model, q, box = setup_motion(12)
    q[4:7, 8] = 0.04
    v = TrajectoryValidator(model, q, SceneQuery(box), {}, {}, local_repair_config({}), 30)
    v.geometry(q)
    changed = q.copy()
    changed[5, 8] += 0.01
    reused = v.geometry(changed)
    fresh = TrajectoryValidator(
        model, q, SceneQuery(box), {}, {}, local_repair_config({}), 30
    ).geometry(changed)
    for key in (
        "per_geom_depth", "interframe_depth", "self_depth", "interframe_self_depth",
        "reliable_scene_depth", "interframe_reliable_scene_depth",
    ):
        np.testing.assert_allclose(reused[key], fresh[key])
    assert reused["interframe_samples"] == fresh["interframe_samples"]
    assert reused["records"] == fresh["records"]


def test_interpolation_crossing_corner_is_rejected():
    model, q, box = setup_motion(3)
    q[:, 7:9] = [[-0.12, 0], [0, 0], [0, -0.12]]
    result, report = repair_local_collisions(
        model, q, box, {}, {}, {"iterations": 0}, framerate=15
    )
    assert report["events"][0]["method"] == "fallback"
    assert any(
        "collision" in reason
        for reason in report["events"][0]["interpolation_rejected"]
    )
    np.testing.assert_array_equal(q, result)


def test_sticking_is_protected_but_sliding_tangent_is_available():
    model, q, box = setup_motion()
    targets = np.column_stack([q[:, 7], q[:, 8], np.zeros(len(q))])
    cfg = local_repair_config({})
    spec = {
        "effectors": {"left_foot": {"robot_body": "foot", "track_orientation": False}}
    }
    trial = q.copy()
    trial[3:6, 7] += 0.02
    for mode, expected in [
        ("sticking", False),
        ("sliding", True),
        ("transition", True),
    ]:
        task = _task(targets, np.ones(len(q), bool), [mode] * len(q))
        v = TrajectoryValidator(
            model, q, SceneQuery(box), {"left_foot": task}, spec, cfg, 30
        )
        assert v.fidelity(trial)["ok"] is expected


def test_contact_switch_is_not_interpolated_across():
    model, q, box = setup_motion()
    q[4, 8] = 0.04
    targets = np.column_stack([q[:, 7], q[:, 8], np.zeros(len(q))])
    task = _task(targets, np.arange(9) >= 4)
    spec = {"effectors": {"left_foot": {"robot_body": "foot"}}}
    result, report = repair_local_collisions(
        model, q, box, {"left_foot": task}, spec, {"iterations": 0}, framerate=30
    )
    assert report["events"][0]["method"] == "fallback"
    np.testing.assert_array_equal(result, q)
    np.testing.assert_array_equal(task.contact_mask, np.arange(9) >= 4)


@pytest.mark.parametrize("fps", [30, 60])
def test_window_seconds_and_endpoint_fallback(fps):
    model, q, box = setup_motion(fps + 1)
    q[:, 7] = 0
    q[: fps // 2, 8] = 0.04
    result, report = repair_local_collisions(
        model,
        q,
        box,
        {},
        {},
        {"iterations": 0, "max_window_seconds": 0.2},
        framerate=fps,
    )
    assert all(e["method"] == "fallback" for e in report["events"])
    assert all(
        e["window"][1] - e["window"][0] <= int(0.2 * fps) for e in report["events"]
    )
    np.testing.assert_array_equal(result, q)


def test_candidate_joint_limits_are_checked_without_clipping():
    model, q, box = setup_motion()
    trial = q.copy()
    trial[4, 7] = 1.2
    v = TrajectoryValidator(
        model, q, SceneQuery(box), {}, {}, local_repair_config({}), 30
    )
    assert "joint_limits" in v.fidelity(trial)["reasons"]


def test_postprocess_guard_rejects_new_shallow_collision_under_global_target():
    from omniretargeting.core import _select_penetration_safe_refinement
    from omniretargeting.scene_geometry import measure_scene_penetration

    model, baseline, scene = setup_motion()
    candidate = baseline.copy()
    candidate[4, 8] = 0.03
    old = measure_scene_penetration(model, baseline, scene)
    new = measure_scene_penetration(model, candidate, scene)
    assert 0.01 < new["max_depth"] < 0.1
    selected, stats, fraction = _select_penetration_safe_refinement(
        model, baseline, candidate, scene, {"target_max_depth": 0.1}, old, new
    )
    assert stats["max_depth"] <= 0.001
    np.testing.assert_array_equal(selected, baseline)
    assert fraction == 0


@pytest.mark.parametrize("point_limit", [2_000_000, None])
def test_failed_interpolation_uses_bounded_optimization_and_reports_residual(point_limit):
    model, q, scene = setup_motion(5)
    q[:, 7:9] = [[-0.12, 0], [0, 0], [0, 0], [0, 0], [0, -0.12]]
    result, report = repair_local_collisions(
        model, q, scene, {}, {}, {"max_query_points": point_limit}, framerate=15
    )
    event = report["events"][0]
    assert event["method"] == "local_optimization"
    assert event["iterations"] <= 8
    assert report["after"]["max_depth"] < report["before"]["max_depth"] - 0.01
    assert report["fidelity"]["ok"]
    np.testing.assert_array_equal(result[:, :7], q[:, :7])
    np.testing.assert_array_equal(result[[0, -1]], q[[0, -1]])


def test_world_foot_reference_and_contact_arrays_are_immutable():
    model, q, scene = setup_motion()
    q[3:6, 8] = 0.04
    target = np.column_stack([q[:, 7], q[:, 8], np.zeros(len(q))])
    task = _task(target)
    spec = {
        "effectors": {
            "left_foot": {"robot_body": "foot", "preserve_world_target": True}
        }
    }
    saved_targets, saved_normals = (
        task.tracking_positions.copy(),
        task.surface_normals.copy(),
    )
    _result, report = repair_local_collisions(
        model, q, scene, {"left_foot": task}, spec, {}, framerate=30
    )
    assert (
        report["fidelity"]["metrics"]["left_foot"]["swing_world_deviation"] <= 0.020001
    )
    np.testing.assert_array_equal(task.tracking_positions, saved_targets)
    np.testing.assert_array_equal(task.surface_normals, saved_normals)
    assert report["events"][0]["method"] != "interpolation"


def test_overlapping_geoms_form_one_event_and_repair_is_deterministic():
    model, q, scene = setup_motion()
    q[3:6, 8] = 0.04
    first, report1 = repair_local_collisions(model, q, scene, {}, {}, {}, framerate=30)
    second, report2 = repair_local_collisions(model, q, scene, {}, {}, {}, framerate=30)
    assert len(report1["events"]) == len(report2["events"]) == 1
    np.testing.assert_array_equal(first, second)
    assert report1["query_points"] == report2["query_points"]


def test_subdivision_and_query_budgets_are_explicit_unresolved():
    model, q, scene = setup_motion(2)
    q[:, 8] = 0
    _, report = repair_local_collisions(
        model, q, scene, {}, {}, {"max_substeps": 1, "iterations": 0}, framerate=30
    )
    assert report["unresolved"]
    assert report["after"]["subdivision_limit_intervals"] == [0]
    result, report = repair_local_collisions(
        model, q, scene, {}, {}, {"max_query_points": 1}, framerate=30
    )
    assert report["status"] == "unvalidated_budget"
    assert report["failed_stage"] == "initial_validation"
    assert report["geometry_calls"] == 1
    assert report["query_points"] == 0
    np.testing.assert_array_equal(q, result)


def test_offline_unlimited_point_queries_keep_same_collision_and_motion_checks():
    model, q, scene = setup_motion()
    q[3:6, 8] = 0.04
    expected, limited_report = repair_local_collisions(model, q, scene, {}, {}, {})
    actual, report = repair_local_collisions(
        model, q, scene, {}, {}, {"max_query_points": None}
    )
    np.testing.assert_array_equal(actual, expected)
    assert report["events"][0]["method"] == "interpolation"
    assert report["after"]["max_depth"] <= 1e-6
    assert report["query_points"] == limited_report["query_points"] > 0
    np.testing.assert_array_equal(actual[:, :7], q[:, :7])
    np.testing.assert_array_equal(actual[:, 9:], q[:, 9:])
    # Disabling point counts does not disable the independent iteration/call cap.
    validator = TrajectoryValidator(
        model, q, SceneQuery(scene), {}, {},
        local_repair_config({"max_query_points": None, "max_geometry_calls": 1}), 30,
    )
    validator.geometry(q)
    with pytest.raises(RuntimeError, match="geometry call budget"):
        validator.geometry(q)


def test_shared_geometry_cache_reuses_poses_but_separates_scene_and_sampling():
    model, q, scene = setup_motion(3)
    q[1, 8] = 0
    query, cache = SceneQuery(scene), {}
    cfg = local_repair_config({"max_query_points": None})
    first = TrajectoryValidator(model, q, query, {}, {}, cfg, 30, geometry_cache=cache)
    expected = first.geometry(q)
    points = query.query_points
    assert points > 0
    task = _task(np.zeros((len(q), 3)), np.ones(len(q), dtype=bool))
    priority = {"effectors": {"left_foot": {"robot_body": "foot"}}}
    second = TrajectoryValidator(
        model, q, query, {"left_foot": task}, priority, cfg, 30, geometry_cache=cache
    )
    actual = second.geometry(q)
    assert query.query_points == points
    np.testing.assert_array_equal(actual["per_geom_depth"], expected["per_geom_depth"])
    assert actual["records"]
    for record in actual["records"]:
        assert record["contact_modes"] == list(second.phase(record["frame"]))
    # New qpos must run FK/query; source and earlier cache entries stay intact.
    changed = q.copy()
    changed[1, 7] += 0.005
    second.geometry(changed)
    assert query.query_points > points
    points = query.query_points
    dense = TrajectoryValidator(
        model, q, query, {}, {}, {**cfg, "sample_spacing": 0.01}, 30,
        geometry_cache=cache,
    )
    dense.geometry(q)
    assert query.query_points > points
    points = query.query_points
    stricter = TrajectoryValidator(
        model, q, query, {}, {}, {**cfg, "penetration_tolerance": 0.0005}, 30,
        geometry_cache=cache,
    )
    stricter.geometry(q)
    assert query.query_points > points
    remote_scene = scene.copy()
    remote_scene.apply_translation([10, 0, 0])
    other_query = SceneQuery(remote_scene)
    other = TrajectoryValidator(model, q, other_query, {}, {}, cfg, 30, geometry_cache=cache)
    assert other.geometry(q)["max_depth"] == 0
    assert other_query.query_points > 0


def test_postprocessing_rejects_bad_fidelity_before_extra_geometry_and_reuses_initial_check():
    from omniretargeting.local_collision_repair import select_safe_postprocessing

    model, q, scene = setup_motion()
    bad = q.copy()
    bad[:, 0] += 0.1
    query, cache = SceneQuery(scene), {}
    cfg = local_repair_config({"max_query_points": None})
    result, _, guard = select_safe_postprocessing(
        model, q, bad, scene, {}, {}, cfg, scene_query=query, geometry_cache=cache
    )
    np.testing.assert_array_equal(result, q)
    assert guard["decisions"]
    assert all(not item["geometry_evaluated"] for item in guard["decisions"])
    points = query.query_points
    repaired, report = repair_local_collisions(
        model, result, scene, {}, {}, cfg, scene_query=query, geometry_cache=cache
    )
    np.testing.assert_array_equal(repaired, q)
    assert query.query_points == points
    assert report["query_points"] == 0
    assert report["after"]["max_depth"] == 0


def test_postprocessing_respects_fixed_sequence_candidate_guard():
    from omniretargeting.local_collision_repair import select_safe_postprocessing

    model, baseline, scene = setup_motion()
    candidate = baseline.copy()
    candidate[:, 8] += 0.003
    selected, _, report = select_safe_postprocessing(
        model, baseline, candidate, scene, {}, {}, {},
        candidate_acceptance=lambda poses: np.array_equal(poses, baseline),
    )
    np.testing.assert_array_equal(selected, baseline)
    assert all(not decision["accepted"] for decision in report["decisions"])


def test_many_postprocessing_blocks_keep_safe_changes_and_reject_bad_window():
    from omniretargeting.local_collision_repair import select_safe_postprocessing

    model, baseline, scene = setup_motion(300)
    candidate = baseline.copy()
    candidate[:, 8] += 0.003
    candidate[150:155, 8] = 0.0
    selected, checked, report = select_safe_postprocessing(
        model, baseline, candidate, scene, {}, {}, {"max_query_points": None},
        framerate=30,
    )
    assert len(report["decisions"]) > 20
    assert 0.8 < report["accepted_fraction"] < 1
    np.testing.assert_array_equal(selected[150:155], baseline[150:155])
    np.testing.assert_array_equal(selected[25:30], candidate[25:30])
    assert checked["max_depth"] <= 0.001


def test_uncertain_scene_keeps_unknown_report_while_known_window_is_repaired():
    model, q, box = setup_motion()
    q[3:6, 8] = 0.04
    open_triangle = trimesh.Trimesh(
        vertices=[[10, 0, 0], [10, 1, 0], [10, 0, 1]], faces=[[0, 1, 2]],
        process=False,
    )
    scene = trimesh.util.concatenate([box, open_triangle])
    result, report = repair_local_collisions(
        model, q, scene, {}, {}, {"max_query_points": None}
    )
    assert report["before"]["max_depth"] > 0.01
    assert report["after"]["max_depth"] < report["before"]["max_depth"]
    assert report["unresolved"]
    assert report["events"]
    assert report["after"]["uncertain_frames"]
    assert "uncertain_scene_side" in report["unresolved_reasons"]
    event = report["events"][0]
    assert event["method"] in {"local_optimization", "bidirectional_dp", "interpolation"}
    # The scene repair may reduce a known collision, but it cannot alter root
    # alignment or endpoints while the unrelated open patch remains unknown.
    np.testing.assert_array_equal(result[:, :7], q[:, :7])
    np.testing.assert_array_equal(result[[0, -1]], q[[0, -1]])
    assert report["geometry_calls"] > 1  # The changed candidate must be checked.


def test_actual_support_foot_stays_put_during_other_chain_repair():
    model, q, scene = setup_motion()
    q[3:6, 8] = 0.04
    task = _task(np.tile([0, 0, 1.0], (len(q), 1)), np.ones(len(q), dtype=bool))
    spec = {
        "effectors": {
            "right_foot": {
                "robot_body": "other",
                "tracking_offset": [0, 0, 1],
                "track_orientation": True,
            }
        }
    }
    result, report = repair_local_collisions(
        model, q, scene, {"right_foot": task}, spec, {}, framerate=30
    )
    assert report["events"][0]["method"] == "interpolation"
    metrics = report["fidelity"]["metrics"]["right_foot"]
    assert metrics["contact_position_error"] == 0
    assert metrics["support_slip"] == 0
    np.testing.assert_array_equal(result[:, 9], q[:, 9])


def test_sliding_repair_preserves_deliberate_tangential_motion_and_timing():
    model, q, scene = setup_motion()
    q[3:6, 8] = 0.04
    task = _task(
        np.column_stack([q[:, 7], q[:, 8], np.zeros(len(q))]),
        np.ones(len(q), bool),
        ["sliding"] * len(q),
    )
    spec = {"effectors": {"left_foot": {"robot_body": "foot"}}}
    result, report = repair_local_collisions(
        model, q, scene, {"left_foot": task}, spec, {}, framerate=30
    )
    assert report["events"][0]["method"] == "interpolation"
    np.testing.assert_allclose(result[:, 7], q[:, 7], atol=1e-15)
    assert np.ptp(result[:, 7]) == pytest.approx(0.4)
    assert np.all(task.contact_modes == "sliding")
    assert report["fidelity"]["metrics"]["left_foot"]["contact_position_error"] == 0


def test_self_collision_policy_is_unchanged_and_enabled_validator_checks_pairs():
    model, q, _ = setup_motion()
    # Move the unrelated sibling collider into the toe path; MuJoCo owns masks
    # and excluded pairs, matching the enabled framewise collision policy.
    model.geom_pos[2] = [0, 0, 0]
    q[:, 8] = 0
    scene = trimesh.creation.box(extents=[0.1] * 3)
    scene.apply_translation([10, 10, 10])
    off = TrajectoryValidator(
        model, q, SceneQuery(scene), {}, {}, local_repair_config({}), 30
    ).geometry(q)
    on = TrajectoryValidator(
        model,
        q,
        SceneQuery(scene),
        {},
        {},
        local_repair_config({"self_collision_enabled": True}),
        30,
    ).geometry(q)
    assert np.max(off["self_depth"]) == 0
    assert np.max(on["self_depth"]) > 0.01
    assert np.max(on["interframe_self_depth"]) > 0.01


def test_core_preserves_root_before_repair_and_hashes_exact_qpos(
    monkeypatch,
):
    from omniretargeting.core import OmniRetargeter
    from omniretargeting.local_collision_repair import trajectory_digest

    model, q, scene = setup_motion()
    q[3:6, 8] = 0.04
    retargeter = OmniRetargeter.__new__(OmniRetargeter)
    retargeter.robot_model = model
    retargeter.terrain_mesh = scene
    retargeter.retargeting_config = {"foot_stabilization": {"enabled": True}}
    retargeter.smplx_joint_names = ["root"]
    retargeter._process_smplx_trajectory = lambda source, scale: source.copy()
    retargeter._perform_retargeting = lambda *args, **kwargs: q.copy()
    calls = []

    def stabilize(motion, *args, **kwargs):
        calls.append("stabilize")
        return motion.copy()

    original_clamp = retargeter._clamp_final_joint_safety_limits

    def clamp(motion):
        calls.append("clamp")
        return original_clamp(motion)

    retargeter._apply_foot_stabilization = stabilize
    retargeter._clamp_final_joint_safety_limits = clamp
    source = np.zeros((len(q), 1, 3))
    source_copy = source.copy()
    scale, output = retargeter.retarget_motion(
        source, framerate=30, visualize_trajectory=False
    )
    assert scale == 1
    assert calls == ["clamp", "clamp"]
    np.testing.assert_array_equal(output[:, :7], q[:, :7])
    report = retargeter.last_collision_diagnostics
    assert report["effective_config"]["active_stages"]["foot_stabilization"] is False
    assert report["local_repair"]["events"][0]["method"] == "interpolation"
    assert report["accepted_qpos_sha256"] == trajectory_digest(output)
    assert report["local_repair"]["after"]["max_depth"] <= 0.001
    np.testing.assert_array_equal(source, source_copy)
    changed = output.copy()
    changed[4, 8] -= 0.1
    assert trajectory_digest(changed) != report["accepted_qpos_sha256"]


def test_contact_drift_between_exact_saved_contact_poses_is_rejected():
    model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
    <worldbody><body><freejoint/><geom size=".01" contype="0" conaffinity="0"/>
    <body name="upper"><joint name="shoulder" axis="0 0 1" range="-3 3"/><geom fromto="0 0 0 1 0 0" size=".01" type="capsule"/>
    <body name="hand" pos="1 0 0"><joint name="elbow" axis="0 0 1" range="-3 3"/>
    <geom fromto="0 0 0 1 0 0" size=".01" type="capsule"/></body></body></body></worldbody></mujoco>""")
    reference = np.tile(model.qpos0, (2, 1))
    reference[:, 7:] = [-np.pi / 4, np.pi / 2]
    candidate = reference.copy()
    candidate[1, 7:] = [np.pi / 4, -np.pi / 2]
    task = _task(np.tile([np.sqrt(2), 0, 0], (2, 1)), np.ones(2, bool))
    priority = {
        "effectors": {"hand": {"robot_body": "hand", "tracking_offset": [1, 0, 0]}}
    }
    scene = trimesh.creation.box()
    scene.apply_translation([10, 10, 10])
    cfg = local_repair_config({"max_substeps": 1024})
    validator = TrajectoryValidator(
        model, reference, SceneQuery(scene), {"hand": task}, priority, cfg, 30
    )
    poses = validator.effectors(candidate)["hand"]["position"]
    np.testing.assert_allclose(poses, task.contact_positions, atol=1e-12)
    check = validator.geometry(candidate)
    assert check["interframe_contact_failures"]
    assert check["interframe_contact_change"] > 0.5


def test_guard_allows_contact_improvement_toward_immutable_anchor():
    from omniretargeting.local_collision_repair import select_safe_postprocessing

    model, reference, scene = setup_motion(3)
    reference[:, 7:9] = [0.4, 0.4]
    targets = np.tile([0.38, 0.4, 0], (3, 1))
    task = _task(targets, np.ones(3, bool))
    priority = {"effectors": {"left_foot": {"robot_body": "foot"}}}
    candidate = reference.copy()
    candidate[:, 7] -= 0.01
    selected, _, report = select_safe_postprocessing(
        model,
        reference,
        candidate,
        scene,
        {"left_foot": task},
        priority,
        {},
        framerate=30,
    )
    np.testing.assert_array_equal(selected, candidate)
    assert report["accepted_fraction"] == 1
    np.testing.assert_array_equal(task.contact_positions, targets)


def test_unclassified_world_foot_does_not_spend_free_swing_budget():
    from omniretargeting.contact import build_world_tracking_tasks

    model, q, scene = setup_motion()
    targets = np.column_stack([q[:, 7], q[:, 8], np.zeros(len(q))])
    tasks = build_world_tracking_tasks({"left_foot": {"tracking_positions": targets}})
    spec = {
        "effectors": {
            "left_foot": {"robot_body": "foot", "preserve_world_target": True}
        }
    }
    validator = TrajectoryValidator(
        model, q, SceneQuery(scene), tasks, spec, local_repair_config({}), 30
    )
    trial = q.copy()
    trial[3:6, 8] += 0.01
    assert not validator.fidelity(trial)["ok"]
    assert validator.swing_budget("left_foot", 4) == 0.001
    assert not np.any(tasks["left_foot"].contact_mask)


def test_local_overrides_cannot_disable_global_self_collision_or_relax_limits():
    cfg = local_repair_config(
        {"self_collision_enabled": False, "joint_limit_safety_fraction": 1.0},
        self_collision_enabled=True,
        joint_limit_safety_fraction=0.8,
    )
    assert cfg["self_collision_enabled"]
    assert cfg["joint_limit_safety_fraction"] == 0.8


def test_unlimited_hinge_matches_saved_qpos_playback_without_erasing_turns():
    from omniretargeting.trajectory_validation import _pose_velocity

    model, q, _ = setup_motion(2)
    joint = model.joint("unrelated").id
    qi, di = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
    assert not model.jnt_limited[joint]
    q[:, qi] = [3.1, -3.1]
    assert _pose_velocity(model, q[0], q[1])[di] == pytest.approx(-6.2)
    q[:, qi] = [0, 2 * np.pi]
    assert _pose_velocity(model, q[0], q[1])[di] == pytest.approx(2 * np.pi)


def test_legacy_profile_keeps_robot_feet_without_inventing_source_contacts():
    from omniretargeting.core import OmniRetargeter

    model, q, scene = setup_motion()
    retargeter = OmniRetargeter.__new__(OmniRetargeter)
    retargeter.robot_model = model
    retargeter.retargeting_config = {}
    retargeter.valid_joint_mapping = {"L_Ankle": "foot", "R_Ankle": "other"}
    tasks, priority, names = retargeter._validation_foot_references(q, {}, {})
    assert names == ["left_foot", "right_foot"]
    assert all(not np.any(task.contact_mask) for task in tasks.values())
    assert all(np.all(np.isinf(task.distances)) for task in tasks.values())
    v = TrajectoryValidator(
        model, q, SceneQuery(scene), tasks, priority, local_repair_config({}), 30
    )
    candidate = q.copy()
    candidate[4, 8] += 0.01
    assert not v.fidelity(candidate)["ok"]
    assert "swing_budget:left_foot" in v.fidelity(candidate)["reasons"]
