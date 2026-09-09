import mujoco
import numpy as np
import pytest

from omniretargeting.local_candidates import select_candidate_path
from omniretargeting.local_collision_repair import repair_local_collisions
from omniretargeting.scene_geometry import SceneQuery
from omniretargeting.trajectory_validation import (
    TrajectoryValidator,
    local_repair_config,
)
from tests.test_contact_priority import _task
from tests.test_local_collision_repair import setup_motion


def directional_case():
    model, q, scene = setup_motion(5)
    q[:, 7:9] = [
        [-0.12, 0],
        [-0.07, -0.07],
        [-0.07, -0.07],
        [-0.07, -0.07],
        [-0.02, -0.12],
    ]
    return model, q, scene


def test_interpolation_first_does_not_generate_directional_candidates(monkeypatch):
    import omniretargeting.local_candidates as module

    def unexpected(*args):
        raise AssertionError("interpolation already solves this event")

    monkeypatch.setattr(module, "bidirectional_window", unexpected)
    model, q, scene = setup_motion()
    q[3:6, 8] = 0.04
    _, report = repair_local_collisions(model, q, scene, {}, {}, {})
    assert report["events"][0]["method"] == "interpolation"


def test_backward_candidate_repairs_failed_interpolation_and_is_reproducible():
    model, q, scene = directional_case()
    results = []
    for _ in range(2):
        result, report = repair_local_collisions(
            model, q, scene, {}, {}, {}, framerate=15
        )
        event = report["events"][0]
        assert event["method"] == "bidirectional_dp"
        assert event["interpolation_rejected"]
        selection = event["bidirectional"]["selection"]
        assert 3 in selection["states"][1:-1]
        assert event["optimization"]["iterations"] <= 2
        assert report["after"]["interframe_max_depth"] <= 0.001
        assert report["fidelity"]["ok"]
        assert report["invariants"]["fixed_coordinates_unchanged"]
        assert report["invariants"]["outside_windows_and_unrelated_joints_unchanged"]
        np.testing.assert_array_equal(result[:, :7], q[:, :7])
        np.testing.assert_array_equal(result[[0, -1]], q[[0, -1]])
        np.testing.assert_array_equal(result[:, 9:], q[:, 9:])
        results.append(result)
    np.testing.assert_array_equal(*results)


def test_safe_nodes_with_obstacle_crossing_edge_cannot_form_dp_path():
    model, q, scene = setup_motion(2)
    q[:, 8] = 0
    v = TrajectoryValidator(
        model, q, SceneQuery(scene), {}, {}, local_repair_config({}), 30
    )
    path, log = select_candidate_path(v, q[:, None, :], 0, [7, 8])
    assert not log["node_rejections"]
    assert path is None
    assert "collision_budget" in log["edge_rejections"][0]["reasons"]
    assert log["status"] == "no_admissible_path"


def test_dp_edges_respect_existing_self_collision_policy():
    model, q, scene = setup_motion(2)
    model.geom_pos[2] = 0
    q[:, 8] = 0
    scene.apply_translation([10, 10, 10])
    for enabled in (False, True):
        cfg = local_repair_config({"self_collision_enabled": enabled})
        v = TrajectoryValidator(model, q, SceneQuery(scene), {}, {}, cfg, 30)
        path, log = select_candidate_path(v, q[:, None, :], 0, [7, 8])
        assert (path is None) == enabled
        if enabled:
            assert not log["node_rejections"]
            assert "collision_budget" in log["edge_rejections"][0]["reasons"]


def test_directions_use_identical_original_time_contact_payloads_and_locked_markers(
    monkeypatch,
):
    import omniretargeting.local_candidates as module

    model, q, scene = directional_case()
    task = _task(np.tile([0, 0, 1.0], (len(q), 1)), np.ones(len(q), bool))
    tasks = {"right_foot": task}
    spec = {
        "effectors": {
            "right_foot": {"robot_body": "other", "tracking_offset": [0, 0, 1]}
        }
    }
    snapshots = []
    original = module._solve_pose

    def observe(v, frame, *args):
        snapshots.append(
            (
                frame,
                v.tasks["right_foot"].frame(frame).copy(),
                v.offset_choice["right_foot"][frame],
            )
        )
        return original(v, frame, *args)

    monkeypatch.setattr(module, "_solve_pose", observe)
    arrays = {
        key: value.copy()
        for key, value in vars(task).items()
        if isinstance(value, np.ndarray)
    }
    result, report = repair_local_collisions(
        model, q, scene, tasks, spec, {}, framerate=15
    )
    assert [f for f, _, _ in snapshots] == [1, 2, 3, 3, 2, 1]
    for forward, backward in zip(snapshots[:3], reversed(snapshots[3:])):
        assert forward[0] == backward[0] and forward[2] == backward[2]
        for key in forward[1]:
            np.testing.assert_array_equal(forward[1][key], backward[1][key])
    for key, value in arrays.items():
        np.testing.assert_array_equal(value, getattr(task, key))
    assert report["fidelity"]["metrics"]["right_foot"]["contact_position_error"] == 0
    np.testing.assert_array_equal(result[:, 9], q[:, 9])


def test_hard_budget_rejects_low_cost_node_and_acceleration_rejects_triple():
    model, q, scene = setup_motion(3)
    q[:, 7] = -0.3
    q[:, 8] = 0.3
    config = local_repair_config({"correction_acceleration": 1.0})
    v = TrajectoryValidator(model, q, SceneQuery(scene), {}, {}, config, 30)
    candidates = q[:, None, :].copy()
    candidates[1, 0, 7] += 0.01
    path, log = select_candidate_path(v, candidates, 0, [7, 8])
    assert path is None
    assert any(
        "correction_acceleration" in x["reasons"] for x in log["edge_rejections"]
    )
    candidates[1, 0, 7] += 0.3
    path, log = select_candidate_path(v, candidates, 0, [7, 8])
    assert path is None
    assert any(
        "joint_correction_budget" in x["reasons"] for x in log["node_rejections"]
    )


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_seconds_budget_fallback_is_explicit_and_noop_without_event(fps):
    model, q, scene = setup_motion(fps + 1)
    result, report = repair_local_collisions(model, q, scene, {}, {}, {}, framerate=fps)
    np.testing.assert_array_equal(result, q)
    assert not report["events"]
    q[:3, 8] = 0.01
    q[:3, 7] = 0
    result, report = repair_local_collisions(
        model, q, scene, {}, {}, {"iterations": 0}, framerate=fps
    )
    assert all(e["method"] == "fallback" for e in report["events"])
    assert all(
        e["window"][1] - e["window"][0] <= int(0.4 * fps) for e in report["events"]
    )


def test_fixed_base_scalar_joints_use_their_actual_nq_nv_columns():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body>
    <joint name="slide" type="slide" axis="1 0 0"/><geom size=".02"/>
    </body></worldbody></mujoco>""")
    _, _, scene = setup_motion(1)
    q = np.array([[-0.2], [-0.2]])
    v = TrajectoryValidator(
        model, q, SceneQuery(scene), {}, {}, local_repair_config({}), 30
    )
    assert v.qi.tolist() == v.di.tolist() == [0]
    assert not v.fixed_qi.size
    assert v.geometry(q)["max_depth"] == 0
