"""Observed contact roles order support > follow > free, without force optimization."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting.contact_phase import attach_contact_phase, contact_phase_config
from omniretargeting._sequence.surface_guard import SurfaceSequenceGuard
from omniretargeting.surface_contact import SurfaceContactTasks
from omniretargeting.surface_tasks import SurfaceBinding
from tests._contact_fixtures import _task


def case(frames=6, stance=False, sole_mode=None):
    """Abdomen skin on a 1 m box edge 4 cm from where the world foot allows."""
    model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
    <worldbody><body name="root"><freejoint/><geom size=".02"/>
      <body name="abdomen" pos="0 0 1.1">
        <joint name="reach" type="slide" axis="1 0 0" range="-.2 .2"/>
        <geom name="skin" type="box" size=".12 .12 .1"/>
        <body name="foot" pos="-.3 0 -.8">
          <joint name="ankle" type="slide" axis="0 1 0" range="-.01 .01"/>
          <geom name="sole" type="box" size=".08 .04 .02"/>
        </body>
      </body>
    </body></worldbody></mujoco>""")
    q = np.tile(model.qpos0, (frames, 1))
    environment = np.tile([[0.04, 0, 1.0], [-0.3, 0, 0.28]], (frames, 1, 1))
    active = np.ones((frames, 2), bool)
    active[:, 1] = sole_mode is not None
    modes = np.tile(["sticking", sole_mode or "none"], (frames, 1))
    surface = SurfaceContactTasks(
        model,
        [
            SurfaceBinding("abdomen", 0, (1, 0, 0), "skin", (0, 0, -0.1), (0, 0, -1)),
            SurfaceBinding("left_foot", 1, (1, 0, 0), "sole", (0, 0, -0.02), (0, 0, -1)),
        ],
        environment,
        np.zeros_like(environment),
        active,
        np.tile([0, 0, 1], (frames, 2, 1)),
        modes=modes,
    )
    surface.static_surface_id = 0
    surface.source_clearances = {"left_foot": np.full(frames, 0.2)}
    surface.protect_primary_endpoints = True
    foot = _task(np.tile([-0.3, 0, 0.3], (frames, 1)), np.full(frames, stance))
    tasks = {"left_foot": foot}
    priority = {
        "enabled": True,
        "hierarchy": True,
        "free_motion_priority": "balanced",
        "pose_direction": {"enabled": False},
        "effectors": {
            "left_foot": {
                "robot_body": "foot",
                "preserve_world_target": True,
                "swing_sigma": 0.005,
            }
        },
    }
    return model, q, surface, tasks, priority


def test_roles_follow_contact_timing_and_leave_source_unchanged():
    _, _, surface, tasks, priority = case(frames=9)
    tasks["left_foot"] = replace(
        tasks["left_foot"], contact_mask=np.array([1, 1, 1, 0, 0, 0, 1, 1, 1], bool)
    )
    surface.source_clearances["left_foot"] = np.array([0, 0, 0, 0.2, 0.2, 0.2, 0, 0, 0])
    anchors = surface.targets.copy()
    policy = attach_contact_phase(surface, None, tasks, priority)
    assert policy.diagnostics["role_frames"]["left_foot"] == {
        "support": [0, 1, 2, 6, 7, 8],
        "free": [3, 4, 5],
    }
    assert [policy.yields("left_foot", f) for f in range(9)] == [False] * 3 + [
        True
    ] * 3 + [False] * 3
    assert [(p["start"], p["stop"]) for p in policy.diagnostics["phases"]] == [
        (0, 3),
        (3, 6),
        (6, 9),
    ]
    np.testing.assert_array_equal(policy.support[:, 0], True)
    np.testing.assert_array_equal(surface.targets, anchors)
    assert not policy.diagnostics["physical_support_verified"]
    assert not policy.diagnostics["force_or_torque_optimization"]


@pytest.mark.parametrize(
    ("change", "supports"),
    [
        ("wall", True),
        ("low_confidence", True),
        ("moving_object", False),
        ("sliding", False),
        ("touch", False),
    ],
)
def test_surface_support_is_still_contact_with_the_static_scene(change, supports):
    _, _, surface, tasks, priority = case()
    if change == "wall":
        surface.normals = np.tile([1, 0, 0], surface.normals.shape[:2] + (1,))
    elif change == "low_confidence":
        surface.confidence = np.full(surface.active.shape, 0.1)
    elif change == "moving_object":
        surface.static_surface_id = None
    else:
        surface.modes = np.tile([change, "none"], (len(surface.active), 1))
    policy = attach_contact_phase(surface, None, tasks, priority)
    np.testing.assert_array_equal(policy.support[:, 0], supports)
    np.testing.assert_array_equal(policy.follow_surface[:, 0], not supports)
    if not supports:
        assert len(list(policy.rows(np.asarray(surface.model.qpos0), 0, [],
                                    role="follow"))) == 1


def test_sliding_body_contact_enters_sequence_follow_priority():
    model, q, surface, tasks, priority = case(frames=2)
    surface.modes = np.tile(["sliding", "none"], (2, 1))
    policy = attach_contact_phase(surface, None, tasks, priority)
    task_set = SimpleNamespace(
        model=model, data=mujoco.MjData(model), effectors={}, extra_tasks=(),
        coords=SimpleNamespace(cols=np.arange(model.nv)), contact_phase=policy,
    )
    guard = SurfaceSequenceGuard.__new__(SurfaceSequenceGuard)
    guard.tasks = task_set
    rows = list(guard.priority_rows(q[0], 0, 3))
    assert len(rows) == 1
    assert np.linalg.norm(rows[0][1]) > 0


@pytest.mark.parametrize(
    ("sole_mode", "stance", "role", "yields"),
    [
        ("sticking", False, "support", False),
        ("sliding", False, "follow", False),
        (None, True, "support", False),
        (None, False, "free", True),
    ],
)
def test_feet_take_the_role_of_their_own_skin(sole_mode, stance, role, yields):
    _, _, surface, tasks, priority = case(stance=stance, sole_mode=sole_mode)
    policy = attach_contact_phase(surface, None, tasks, priority)
    assert policy.role("left_foot", 0) == role
    assert policy.yields("left_foot", 0) is yields
    # A foot's approximate learned skin never becomes a second foot target.
    np.testing.assert_array_equal(policy.support[:, 1], False)


def test_unknown_or_moving_contacts_keep_protection():
    _, _, surface, tasks, priority = case()
    surface.source_clearances = {}
    policy = attach_contact_phase(surface, None, tasks, priority)
    assert policy.role("left_foot", 0) == "free"
    assert not policy.yields("left_foot", 0)
    _, _, surface, tasks, priority = case(stance=True)
    tasks["left_foot"] = replace(tasks["left_foot"], environment_ids=np.ones(6, int))
    assert attach_contact_phase(surface, None, tasks, priority).role("left_foot", 0) == "follow"


def solve_case(stance, enabled):
    from omniretargeting.sequence_retargeting import retarget_sequence
    from tests.test_full_body_surfaces import _markers

    model, q, surface, tasks, priority = case(stance=stance)
    attach_contact_phase(surface, None, tasks, priority, {"enabled": enabled})
    names = ["root", "abdomen", "foot"]
    positions = _markers(model, q, names)
    result, report = retarget_sequence(
        model,
        q,
        positions,
        names,
        {n: n for n in names},
        positions[:, 0],
        tasks,
        priority,
        framerate=30,
        extra_tasks=(surface,),
        config={
            "iterations": 4,
            "self_collision_enabled": False,
            "lock_root_xy": True,
            "lock_root_z": True,
            "lock_root_orientation": True,
            "world_tracking_sigma_m": 0.005,
        },
    )
    return model, q, surface, tasks, priority, result, report


def abdomen_error(surface, poses):
    return float(np.nanmean(surface.errors(poses)[:, 0]))


def test_sequence_airborne_foot_yields_to_abdomen_support():
    _, _, surface0, _, _, before, _ = solve_case(False, False)
    model, q, surface, _, _, after, report = solve_case(False, True)
    assert abdomen_error(surface0, before) > 0.038
    assert abdomen_error(surface, after) < 0.002
    assert 0.038 < after[:, model.joint("reach").qposadr[0]].mean() <= 0.05
    np.testing.assert_array_equal(after[:, :7], q[:, :7])
    assert report["surface_endpoint_guard"]["linear_failures"] == 0
    assert report["contact_phase_priority"]["role_frames"]["left_foot"] == {
        "free": list(range(6))
    }
    # A standing foot supports first; the abdomen cannot pull it away.
    _, _, held_surface, _, _, held, _ = solve_case(True, True)
    assert abdomen_error(held_surface, held) > 0.038


def test_framewise_airborne_foot_yields_to_abdomen_support():
    from omniretargeting.retargeting import GenericInteractionRetargeter
    from tests.test_full_body_surfaces import _markers

    model, q, surface, tasks, priority = case()
    attach_contact_phase(surface, None, tasks, priority)
    scene = trimesh.creation.box()
    scene.apply_translation([10, 10, 10])
    mapping = {n: n for n in ("root", "abdomen", "foot")}
    solver = GenericInteractionRetargeter(
        model,
        mujoco.MjData(model),
        scene,
        mapping,
        1.0,
        terrain_sample_points=4,
        self_collision_enabled=False,
        hard_penetration_constraint=False,
        end_effector_config=priority,
    )
    solver.surface_contact_tasks = surface
    result = solver.retarget_frame(
        _markers(model, q[:1], list(mapping))[0],
        q[0],
        max_iter=6,
        relation_frame=0,
        end_effector_targets={"left_foot": tasks["left_foot"].frame(0)},
    )
    assert abdomen_error(surface, result[None]) < 0.003


def test_repair_keeps_ordinary_budgets_and_splits_windows_on_roles():
    from omniretargeting.scene_geometry import SceneQuery
    from omniretargeting.trajectory_validation import (
        TrajectoryValidator,
        local_repair_config,
    )
    from omniretargeting.whole_body_correction import whole_body_config
    from omniretargeting.whole_body_guard import WholeBodyValidator

    model, q, surface, tasks, priority, after, _ = solve_case(False, True)
    scene = trimesh.creation.box()
    scene.apply_translation([0.54, 0, 0.5])  # Actual one-metre box; edge x=.04, z=1.
    cfg = local_repair_config({"self_collision_enabled": False})
    args = (model, after, SceneQuery.from_tasks(scene, tasks), tasks, priority, cfg, 30)
    v = TrajectoryValidator(*args, surface_contact_tasks=surface)
    # Roles reorder the main solve; repair still bounds every free endpoint.
    assert v.swing_budget("left_foot", 0) == cfg["swing_position_budget"]
    assert v.fidelity(after)["ok"]
    assert ("contact_roles", (0,), ("free",)) in v.phase(0)
    wb_cfg = whole_body_config()
    w = WholeBodyValidator(*args, surface_contact_tasks=surface, whole_body_config=wb_cfg)
    assert w.swing_budget("left_foot", 0) == wb_cfg["foot_position_budget"]
    assert w.fidelity(after)["ok"]
    geometry = w.geometry(after)
    assert geometry["max_depth"] < 1e-6
    assert geometry["interframe_max_depth"] < 1e-6
    with w.segment(2, 5) as window:
        assert window.contact_phase is w.contact_phase
        assert window.fidelity(after[2:5])["ok"]


def test_source_clearance_comes_from_all_original_effector_samples(monkeypatch):
    from omniretargeting import surface_learning
    from omniretargeting.surface_contact import build_surface_contact_tasks
    from tests.test_surface_contact import source_case

    model, _, scene, source = source_case()
    full = source["full_body"]
    full["sample_effectors"] = np.full(len(full["face_ids"]), "left_foot")
    original = copy.deepcopy(source)
    monkeypatch.setattr(
        surface_learning,
        "learn_indexed_points",
        lambda human, robot, edges, config, **kw: (human.copy(), {"loss": 0.0}),
    )
    task, _ = build_surface_contact_tasks(
        model, source, model.qpos0, scene, {"correspondence": {"cache_dir": None}}
    )
    assert task is not None
    assert task.source_clearances["left_foot"][-1] > 0.1
    assert task.source_clearances["left_foot"][0] < 0.04
    np.testing.assert_array_equal(full["triangles"], original["full_body"]["triangles"])


def test_only_enabled_is_configurable():
    _, _, surface, tasks, priority = case()
    assert (
        attach_contact_phase(surface, None, tasks, priority, {"enabled": False}) is None
    )
    for cfg in ({"free_weight": 0.01}, {"release_distance": 0.04}, {"confidence": 0.5}):
        with pytest.raises(ValueError):
            contact_phase_config(cfg)
    with pytest.raises(TypeError):
        contact_phase_config({"enabled": 1})
