"""Whole-body learned correspondence and contact preservation across stages."""

import copy

import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting.data_sources.base import MotionData
from omniretargeting.scene_geometry import SceneQuery
from omniretargeting.surface_contact import (
    SurfaceContactTasks,
    build_surface_contact_tasks,
)
from omniretargeting.surface_learning import (
    bind_whole_body,
    geodesic_edges,
    learn_indexed_points,
    robot_surface_mesh,
)
from omniretargeting.surface_tasks import SurfaceBinding
from omniretargeting.trajectory_refinement import refine_joint_trajectory
from omniretargeting.trajectory_validation import (
    TrajectoryValidator,
    local_repair_config,
)


def contact_case(frames=6):
    model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
    <worldbody><body name="root"><freejoint/>
      <geom type="sphere" size=".02" pos="0 0 .7"/>
      <body name="forearm" pos="0 0 .5">
        <joint name="x" type="slide" axis="1 0 0" range="-.3 .3"/>
        <joint name="z" type="slide" axis="0 0 1" range="-.2 .2"/>
        <geom name="skin" type="box" size=".2 .1 .1"/>
      </body>
    </body></worldbody></mujoco>""")
    q = np.tile(model.qpos0, (frames, 1))
    q[:, model.joint("z").qposadr] = 0.005
    environment = np.tile([0, 0, 0.4], (frames, 1, 1))
    vectors = np.tile([0, 0, 0.005], (frames, 1, 1))
    binding = SurfaceBinding(
        "left_forearm", 0, (1, 0, 0), "skin", (0, 0, -0.1), (0, 0, -1)
    )
    tasks = SurfaceContactTasks(
        model,
        [binding],
        environment,
        vectors,
        np.ones((frames, 1), bool),
        np.tile([0, 0, 1], (frames, 1, 1)),
    )
    scene = trimesh.creation.box(extents=[2, 2, 0.1])
    scene.apply_translation([0, 0, 0.35])
    return model, q, tasks, scene


def source_case(frames=6, samples=96):
    model, q, _, scene = contact_case(frames)
    mesh, _, _ = robot_surface_mesh(model, model.qpos0)
    points, face_ids = trimesh.sample.sample_surface(mesh, samples, seed=3)
    bary = trimesh.triangles.points_to_barycentric(mesh.triangles[face_ids], points)
    triangles = np.tile(mesh.triangles[face_ids], (frames, 1, 1, 1))
    triangles += [0, 0, 0.005]
    triangles[4:] += [0, 0, 0.15]
    source = {
        "kind": "barycentric_skin",
        "metric_scale": 1.0,
        "parts": {},
        "full_body": {
            "reference_vertices": np.asarray(mesh.vertices),
            "faces": np.asarray(mesh.faces),
            "face_ids": face_ids,
            "barycentric": bary,
            "triangles": triangles,
            "labels": np.full(samples, "forearm"),
        },
    }
    return model, q, scene, source


def test_learned_correspondence_reduces_coverage_error_and_preserves_indexing():
    source = np.random.default_rng(1).uniform(-0.3, 0.3, (48, 3))
    target = source + [0.07, -0.04, 0.02]
    edges = np.column_stack([np.arange(47), np.arange(1, 48)])
    predicted, report = learn_indexed_points(source, target, edges, {"iterations": 90})
    assert report["chamfer"] < report["initial_chamfer"] * 0.1
    np.testing.assert_allclose(predicted, target, atol=0.014)


def test_geodesic_graph_does_not_bridge_disconnected_nearby_surfaces():
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0.001], [1, 0, 0.001], [0, 1, 0.001]]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    ids = np.array([0, 0, 1, 1])
    bary = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]] * 2)
    edges = geodesic_edges(vertices, faces, ids, bary, 3)
    assert len(edges) == 2
    assert all((a < 2) == (b < 2) for a, b in edges)


def test_learned_binding_cannot_transfer_left_foot_contact_to_right_foot(monkeypatch):
    from omniretargeting import surface_learning

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody>\n      <body name="root"><freejoint/><geom size=".02" pos="0 0 .5"/>\n        <body name="left" pos="0 .1 .025"><joint type="slide" axis="0 0 1"/>\n          <geom type="box" size=".07 .025 .025"/></body>\n        <body name="right" pos="0 -.1 .025"><joint type="slide" axis="0 0 1"/>\n          <geom type="box" size=".07 .025 .025"/></body>\n      </body></worldbody></mujoco>'
    )
    mesh, _, _ = robot_surface_mesh(model, model.qpos0)
    points, ids = trimesh.sample.sample_surface(mesh, 192, seed=4)
    bary = trimesh.triangles.points_to_barycentric(mesh.triangles[ids], points)
    source = {
        "kind": "barycentric_skin",
        "metric_scale": 1.0,
        "parts": {},
        "full_body": {
            "reference_vertices": mesh.vertices,
            "faces": mesh.faces,
            "face_ids": ids,
            "barycentric": bary,
            "triangles": np.tile(mesh.triangles[ids], (3, 1, 1, 1)),
        },
    }

    # A low-loss learned solution can swap two geometrically identical feet.
    # It must be rejected before contact targets are transferred to those feet.
    def swapped(human, robot, edges, config):
        return human * [1, -1, 1], {"loss": 0.0}

    monkeypatch.setattr(surface_learning, "learn_indexed_points", swapped)
    scene = trimesh.creation.box(extents=[2, 2, 0.1])
    scene.apply_translation([0, 0, -0.05])
    task, report = build_surface_contact_tasks(
        model, source, model.qpos0, scene, {"correspondence": {"cache_dir": None}}
    )
    assert task is None
    assert report["reason"] == "no_valid_contact_correspondences"
    assert report["source_active_frame_points"] > 0


def test_full_surface_binding_is_learned_cached_and_source_immutable(tmp_path):
    model, _, _, source = source_case()
    before = copy.deepcopy(source)
    cfg = {"iterations": 30, "cache_dir": str(tmp_path)}
    a, valid, first = bind_whole_body(model, source, model.qpos0, cfg)
    b, valid_again, second = bind_whole_body(model, source, model.qpos0, cfg)
    assert first["method"] == "learned_indexed_surface_correspondence"
    assert not first["cache_hit"] and second["cache_hit"]
    assert valid.sum() > 0.75 * len(valid)
    assert a == b
    np.testing.assert_array_equal(valid, valid_again)
    for key in source["full_body"]:
        np.testing.assert_array_equal(
            source["full_body"][key], before["full_body"][key]
        )
    source["full_body"]["reference_vertices"] = (
        source["full_body"]["reference_vertices"] * 1.01
    )
    _, _, changed = bind_whole_body(model, source, model.qpos0, cfg)
    assert changed["cache_key"] != first["cache_key"]


def test_full_body_scene_map_retains_source_vectors_and_releases(tmp_path):
    model, q, scene, source = source_case()
    tasks, report = build_surface_contact_tasks(
        model,
        source,
        model.qpos0,
        scene,
        {"correspondence": {"iterations": 25, "cache_dir": str(tmp_path)}},
    )
    assert tasks is not None, report
    assert report["contact_regions"] == ["forearm"]
    assert tasks.active[:4].any() and not tasks.active[4:].any()
    assert not list(tasks.rows(q[4], 4, []))
    selected = report["source_sample_indices"]
    full = source["full_body"]
    points = np.einsum("tnij,ni->tnj", full["triangles"], full["barycentric"])
    np.testing.assert_allclose(tasks.source_vectors, points[:, selected] - tasks.environment_points)
    # Contact anchors use the actual surface; an observed 5 mm gap remains in
    # the immutable source vectors instead of being copied into the robot.
    np.testing.assert_allclose(tasks.targets[tasks.active, 2], 0.4, atol=1e-9)
    np.testing.assert_allclose(tasks.targets[~tasks.active], points[:, selected][~tasks.active])
    np.testing.assert_allclose(tasks.weights[:4].sum(1), 5)
    with pytest.raises(ValueError):
        tasks.targets[0, 0, 0] = 1


def test_world_contact_jacobian_includes_robot_base_and_ancestor_joints():
    model, poses, tasks, _ = contact_case()
    pose = poses[0]
    _, _, jacobian = next(tasks.position_rows(pose, 0, np.arange(model.nv)))
    for d in range(model.nv):
        a, b = pose.copy(), pose.copy()
        velocity = np.eye(model.nv)[d]
        mujoco.mj_integratePos(model, a, velocity, -1e-6)
        mujoco.mj_integratePos(model, b, velocity, 1e-6)
        ra, rb = (
            next(tasks.position_rows(a, 0, []))[1],
            next(tasks.position_rows(b, 0, []))[1],
        )
        np.testing.assert_allclose((rb - ra) / 2e-6, jacobian[:, d], atol=1e-9)
    np.testing.assert_allclose(jacobian[:, :3], np.eye(3))


def test_shared_environment_assignment_preserves_sliding_vector():
    model, q, tasks, _ = contact_case()
    environment = tasks.environment_points.copy()
    environment[:, 0, 0] = np.arange(len(q)) * 0.02
    sliding = SurfaceContactTasks(
        model,
        tasks.bindings,
        environment,
        tasks.source_vectors,
        tasks.active,
        tasks.normals,
    )
    q[:, model.joint("x").qposadr] = environment[:, 0, 0, None]
    np.testing.assert_allclose(sliding.errors(q), 0, atol=1e-12)
    assert sliding.errors(np.tile(q[0], (len(q), 1)))[-1, 0] > 0.09


def test_refinement_and_local_window_validation_protect_forearm_contact():
    model, q, tasks, scene = contact_case()
    cfg = local_repair_config({"self_collision_enabled": False})
    validator = TrajectoryValidator(
        model, q, SceneQuery(scene), {}, {}, cfg, 30, surface_contact_tasks=tasks
    )
    bad = q.copy()
    bad[:, model.joint("z").qposadr] += 0.04
    assert "surface_contact_regression" in validator.fidelity(bad)["reasons"]
    with validator.segment(2, 5) as segment:
        assert segment.surface_contact_tasks is tasks
        assert "surface_contact_regression" in segment.fidelity(bad[2:5])["reasons"]
    result = refine_joint_trajectory(
        model,
        bad,
        {},
        {},
        {"iterations": 5, "self_collision_enabled": False},
        surface_contact_tasks=tasks,
    )
    assert tasks.metrics(result)["contact_vector_error_mean_m"] < 0.005
    np.testing.assert_array_equal(result[:, :7], bad[:, :7])


def test_surface_resampling_and_export_keep_fixed_identities():
    _, _, _, source = source_case()
    motion = MotionData(
        np.zeros((6, 1, 3)), framerate=10, metadata={"body_surface_samples": source}
    )
    full = motion.resample(20).metadata["body_surface_samples"]["full_body"]
    assert len(full["triangles"]) == 11
    for key in ("face_ids", "barycentric", "reference_vertices", "faces", "labels"):
        np.testing.assert_array_equal(full[key], source["full_body"][key])
    _, _, tasks, _ = contact_case()
    assert tasks.export()["surface_contact_robot_local_points"].shape == (1, 3)


def test_missing_source_and_opt_out_are_explicit():
    model, _, _, scene = contact_case()
    assert (
        build_surface_contact_tasks(model, None, None, scene)[1]["reason"]
        == "missing_whole_body_skin_samples"
    )
    assert (
        build_surface_contact_tasks(model, None, None, scene, {"enabled": False})[1][
            "reason"
        ]
        == "configured_off"
    )


def test_guarded_smoothing_cannot_pull_body_contact_off_the_scene():
    from omniretargeting._refinement.guard import RefinementGuard

    model, q, tasks, scene = contact_case()
    cfg = local_repair_config({"self_collision_enabled": False})
    validator = TrajectoryValidator(
        model, q, SceneQuery(scene), {}, {}, cfg, 30, surface_contact_tasks=tasks
    )
    guard = RefinementGuard(validator)
    bad = q.copy()
    bad[2:4, model.joint("z").qposadr] += 0.025
    accepted = guard.preserve_safe_smoothing(q, bad)
    np.testing.assert_array_equal(accepted, q)


def test_surface_contact_timeline_and_metric_contracts_fail_explicitly():
    model, q, tasks, scene = contact_case()
    with pytest.raises(ValueError, match="timeline"):
        refine_joint_trajectory(model, q[:-1], {}, {}, {}, surface_contact_tasks=tasks)
    model, _, scene, source = source_case()
    source["metric_scale"] = 0.9
    with pytest.raises(ValueError, match="unscaled"):
        build_surface_contact_tasks(model, source, model.qpos0, scene)


def test_framewise_surface_contact_improves_forearm_and_preserves_hand_contact():
    from omniretargeting.retargeting import GenericInteractionRetargeter
    from tests._contact_fixtures import _task

    model, q, tasks, _ = contact_case()
    q[:, model.joint("z").qposadr] += 0.04
    scene = trimesh.creation.box()
    scene.apply_translation([0, 0, -5])
    solver = GenericInteractionRetargeter(
        model,
        mujoco.MjData(model),
        scene,
        {"root": "root", "forearm": "forearm"},
        1.0,
        terrain_sample_points=4,
        terrain_sample_seed=0,
        hard_penetration_constraint=False,
        self_collision_enabled=False,
        end_effector_config={
            "enabled": True,
            "free_motion_priority": "balanced",
            "pose_direction": {"enabled": False},
            "effectors": {"hand": {"robot_body": "forearm"}},
        },
    )
    solver.surface_contact_tasks = tasks
    data = mujoco.MjData(model)
    data.qpos[:] = q[0]
    mujoco.mj_forward(model, data)
    points = np.array(
        [data.xpos[model.body(b).id] for b in solver.joint_mapping.values()]
    )
    result = solver.retarget_frame(points, q[0], max_iter=8, relation_frame=0)
    assert tasks.metrics([result])["contact_vector_error_mean_m"] < 0.002
    target = np.tile(data.xpos[model.body("forearm").id], (len(q), 1))
    hand = _task(target, np.ones(len(q), bool))
    result = solver.retarget_frame(
        points,
        q[0],
        max_iter=8,
        relation_frame=0,
        end_effector_targets={"hand": hand.frame(0)},
    )
    data.qpos[:] = result
    mujoco.mj_forward(model, data)
    assert np.linalg.norm(data.xpos[model.body("forearm").id] - target[0]) < 0.002
