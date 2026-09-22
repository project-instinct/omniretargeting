"""Surface contact semantics, independent of collision avoidance."""

import copy

import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting.scene_geometry import SceneQuery
from omniretargeting.self_collision import SelfCollisionQuery
from omniretargeting.self_contact import build_self_contact_tasks
from omniretargeting.trajectory_refinement import refine_joint_trajectory
from omniretargeting.trajectory_validation import (
    TrajectoryValidator,
    local_repair_config,
)
from tests._contact_fixtures import _task


def surface_case(frames=8):
    model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
    <worldbody><body name="root"><freejoint/><geom size=".02"/>
      <body name="thigh" pos="0 0 .5"><joint name="hip" axis="0 1 0" range="-.5 .5"/>
        <geom name="thigh_skin" type="box" size=".1 .1 .2"/>
        <body name="knee" pos="0 0 -.2"><joint axis="0 1 0"/><geom size=".01"/></body>
      </body>
      <body name="hand" pos=".3 0 .72">
        <joint name="hand_x" type="slide" axis="1 0 0" range="-.5 .2"/>
        <joint name="hand_z" type="slide" axis="0 0 1" range="-.2 .2"/>
        <geom name="hand_skin" type="box" size=".06 .04 .02"/>
      </body>
    </body></worldbody></mujoco>""")
    hand = np.array([[[0.24, -0.04, 0.7], [0.36, 0.04, 0.7], [0.36, -0.04, 0.7]]])
    thigh = np.array([[[-0.1, -0.1, 0.7], [0.1, -0.1, 0.7], [0.1, 0.1, 0.7]]])
    parts = {}
    for name, triangle, bary, joint, end, bounds in (
        (
            "hand",
            hand,
            [0.5, 0.5, 0],
            "wrist",
            None,
            [[0.24, -0.04, 0.70], [0.36, 0.04, 0.74]],
        ),
        (
            "thigh",
            thigh,
            [0.5, 0, 0.5],
            "hip",
            "knee",
            [[-0.1, -0.1, 0.3], [0.1, 0.1, 0.7]],
        ),
    ):
        motion = np.tile(triangle, (frames, 1, 1, 1))
        if name == "hand":
            motion += [-0.3, 0, 0.005]
        parts[name] = {
            "source_joint": joint,
            "source_end_joint": end,
            "effector": "hand" if name == "hand" else None,
            "reference_triangles": triangle,
            "triangles": motion,
            "barycentric": np.array([bary]),
            "face_ids": np.array([0]),
            "reference_bounds": np.array(bounds),
        }
    source = {"kind": "barycentric_skin", "metric_scale": 1.0, "parts": parts}
    mapping = {"wrist": "hand", "hip": "thigh", "knee": "knee"}
    q = np.tile(model.qpos0, (frames, 1))
    q[:, model.joint("hand_x").qposadr] = -0.3
    q[:, model.joint("hand_z").qposadr] = 0.005
    return model, source, mapping, q


def build_case():
    model, source, mapping, q = surface_case()
    tasks, report = build_self_contact_tasks(model, source, mapping, model.qpos0)
    assert tasks is not None, report
    return model, source, mapping, q, tasks


def test_samples_create_persistent_surface_pair_and_do_not_edit_source():
    model, source, mapping, q = surface_case()
    before = copy.deepcopy(source)
    tasks, report = build_self_contact_tasks(model, source, mapping, model.qpos0)
    assert report["event_count"] == 1
    np.testing.assert_allclose(tasks.errors(q), 0, atol=1e-12)
    assert tasks.bindings[0].robot_geom == model.geom("hand_skin").id
    assert abs(tasks.bindings[0].local_point[2]) == pytest.approx(0.02)
    for name in source["parts"]:
        np.testing.assert_array_equal(
            source["parts"][name]["triangles"], before["parts"][name]["triangles"]
        )


def test_both_moving_surfaces_jacobian_and_common_root_invariance():
    model, _, _, q, tasks = build_case()
    pose = q[0].copy()
    pose[model.joint("hip").qposadr] = 0.2
    _value, jac, _normal, jnormal = tasks.at(pose, 0)
    np.testing.assert_allclose(jac[:, :6], 0, atol=1e-14)
    assert np.linalg.norm(jac[:, model.joint("hip").dofadr]) > 0.1
    for d in range(model.nv):
        delta = np.eye(model.nv)[d] * 1e-6
        first, second = pose.copy(), pose.copy()
        mujoco.mj_integratePos(model, first, delta, -1)
        mujoco.mj_integratePos(model, second, delta, 1)
        a, b = tasks.at(first, 0), tasks.at(second, 0)
        np.testing.assert_allclose(jac[:, d], (b[0] - a[0]) / 2e-6, atol=2e-9)
        np.testing.assert_allclose(jnormal[:, d], (b[2] - a[2]) / 2e-6, atol=2e-9)


def test_release_short_pass_and_scene_contact_do_not_attract_hand():
    model, source, mapping, q = surface_case()
    source["parts"]["hand"]["triangles"][4:] += [0, 0, 0.2]
    tasks, report = build_self_contact_tasks(model, source, mapping, model.qpos0)
    assert report["events"][0]["stop"] == 4
    assert not list(tasks.rows(q[4], 4, np.arange(model.nv)))
    scene = {"hand": _task(np.zeros((8, 3)), contact_mask=np.ones(8, bool))}
    assert (
        build_self_contact_tasks(model, source, mapping, model.qpos0, scene)[0] is None
    )
    source["parts"]["hand"]["triangles"][2:] += [0, 0, 0.2]
    assert build_self_contact_tasks(model, source, mapping, model.qpos0)[0] is None


def test_validator_preserves_contact_and_original_timeline_for_windows():
    model, _, _, q, tasks = build_case()
    scene = trimesh.creation.box(extents=[2, 2, 0.1])
    scene.apply_translation([0, 0, -0.3])
    cfg = local_repair_config({"self_collision_enabled": True})
    v = TrajectoryValidator(
        model, q, SceneQuery(scene), {}, {}, cfg, 30, self_contact_tasks=tasks
    )
    assert v.fidelity(q)["ok"]
    changed = q.copy()
    changed[:, model.joint("hand_z").qposadr] += 0.03
    assert "self_contact_regression" in v.fidelity(changed)["reasons"]
    with v.segment(2, 6) as window:
        assert "self_contact_regression" in window.fidelity(changed[2:6])["reasons"]
        assert window.fidelity(q[2:6])["ok"]


def test_refinement_brings_hovering_hand_to_surface_without_penetration():
    model, _, _, q, tasks = build_case()
    q[:, model.joint("hand_z").qposadr] += 0.04
    result = refine_joint_trajectory(
        model,
        q,
        {},
        {},
        {
            "iterations": 6,
            "self_collision_enabled": True,
            "joint_regularization_sigma": 1.0,
        },
        self_contact_tasks=tasks,
        framerate=30,
    )
    assert tasks.metrics(result)["pair_vector_error_mean_m"] < 0.01
    np.testing.assert_array_equal(result[:, :7], q[:, :7])
    query = SelfCollisionQuery(model)
    assert (
        max(
            (max(-r["clearance"], 0) for pose in result for r in query.query(pose)),
            default=0,
        )
        <= 0.001
    )


def test_missing_surfaces_disable_and_mismatched_timeline_fails():
    model, source, mapping, _q, _tasks = build_case()
    assert build_self_contact_tasks(model, None, mapping, model.qpos0)[0] is None
    source["parts"]["hand"]["triangles"] = source["parts"]["hand"]["triangles"][:-1]
    with pytest.raises(ValueError, match="timeline"):
        build_self_contact_tasks(model, source, mapping, model.qpos0)


def test_framewise_solver_preserves_scene_contact_before_self_contact():
    from omniretargeting.retargeting import GenericInteractionRetargeter

    model, _, _, q, tasks = build_case()
    q[:, model.joint("hand_z").qposadr] += 0.04
    scene = trimesh.creation.box()
    scene.apply_translation([0, 0, -5])
    priority = {
        "enabled": True,
        "hierarchy": True,
        "free_motion_priority": "balanced",
        "pose_direction": {"enabled": False},
        "effectors": {"hand": {"robot_body": "hand", "track_orientation": False}},
    }
    solver = GenericInteractionRetargeter(
        model,
        mujoco.MjData(model),
        scene,
        {"root": "root", "hip": "thigh", "hand": "hand"},
        1.0,
        terrain_sample_points=4,
        terrain_sample_seed=0,
        hard_penetration_constraint=False,
        end_effector_config=priority,
    )
    solver.self_contact_tasks = tasks
    data = mujoco.MjData(model)
    data.qpos[:] = q[0]
    mujoco.mj_forward(model, data)
    points = np.array(
        [data.xpos[model.body(b).id] for b in solver.joint_mapping.values()]
    )
    result = solver.retarget_frame(points, q[0], max_iter=10, relation_frame=0)
    assert tasks.metrics([result])["pair_vector_error_mean_m"] < 0.01
    # A conflicting world contact wins even with this self-contact task present.
    target = np.tile(data.xpos[model.body("hand").id].copy(), (len(q), 1))
    world = _task(target, np.ones(len(q), bool))
    result = solver.retarget_frame(
        points,
        q[0],
        max_iter=10,
        relation_frame=0,
        end_effector_targets={"hand": world.frame(0)},
    )
    data.qpos[:] = result
    mujoco.mj_forward(model, data)
    assert np.linalg.norm(data.xpos[model.body("hand").id] - target[0]) < 0.001


def test_surface_resampling_preserves_barycentric_identity():
    from omniretargeting.data_sources.base import MotionData

    _, source, _, _ = surface_case(4)
    motion = MotionData(
        np.zeros((4, 1, 3)), framerate=10, metadata={"body_surface_samples": source}
    )
    resized = motion.resample(20)
    surface = resized.metadata["body_surface_samples"]
    for name, part in surface["parts"].items():
        assert len(part["triangles"]) == len(resized.positions)
        np.testing.assert_array_equal(
            part["barycentric"], source["parts"][name]["barycentric"]
        )
        np.testing.assert_array_equal(
            part["face_ids"], source["parts"][name]["face_ids"]
        )
    assert len(source["parts"]["hand"]["triangles"]) == 4


def test_relative_surface_angle_preserves_source_tilt():
    from scipy.spatial.transform import Rotation

    model, source, mapping, _ = surface_case()
    triangles = source["parts"]["hand"]["triangles"]
    pivot = np.array([0, 0, 0.705])
    rotation = Rotation.from_euler("y", 20, degrees=True).as_matrix()
    source["parts"]["hand"]["triangles"] = (triangles - pivot) @ rotation.T + pivot
    tasks, report = build_self_contact_tasks(model, source, mapping, model.qpos0)
    assert tasks is not None, report
    np.testing.assert_allclose(
        tasks.normal_targets[0, 0], rotation @ [0, 0, -1], atol=1e-12
    )


def test_contact_confidence_really_changes_its_influence_on_pose():
    from omniretargeting.retargeting import GenericInteractionRetargeter
    from omniretargeting.self_contact import SelfContactTasks

    model, _, _, q, tasks = build_case()
    q[:, model.joint("hand_z").qposadr] += 0.04
    scene = trimesh.creation.box()
    scene.apply_translation([0, 0, -5])
    solver = GenericInteractionRetargeter(
        model,
        mujoco.MjData(model),
        scene,
        {"root": "root", "hip": "thigh", "hand": "hand"},
        1.0,
        terrain_sample_points=4,
        terrain_sample_seed=0,
        hard_penetration_constraint=False,
        end_effector_config={"enabled": False, "free_motion_priority": "balanced"},
    )
    data = mujoco.MjData(model)
    data.qpos[:] = q[0]
    mujoco.mj_forward(model, data)
    points = np.array(
        [data.xpos[model.body(b).id] for b in solver.joint_mapping.values()]
    )
    errors = []
    for scale in (0.001, 1.0):
        solver.self_contact_tasks = SelfContactTasks(
            model,
            tasks.bindings,
            tasks.targets,
            tasks.confidence * scale,
            tasks.events,
            tasks.cfg,
            tasks.diagnostics,
            tasks.normal_targets,
        )
        result = solver.retarget_frame(points, q[0], max_iter=10, relation_frame=0)
        errors.append(tasks.metrics([result])["pair_vector_error_mean_m"])
    # Scaling an isolated lexicographic objective leaves its minimizer unchanged
    # and used to make the source contact ramp ineffective.
    assert errors[0] > errors[1] + 0.001
