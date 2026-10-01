"""Whole-body contacts and dense pose objectives on real articulated surfaces."""

import copy

import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting import surface_learning
from omniretargeting.self_contact import build_self_contact_tasks
from omniretargeting.surface_learning import robot_surface_mesh
from omniretargeting.surface_motion import SurfaceMotion
from omniretargeting.surface_pose import build_surface_pose_tasks
from omniretargeting.surface_contact import calibrate_contact_bindings
from omniretargeting.surface_tasks import SurfaceBinding
from tests._contact_fixtures import _task


def test_contact_seed_calibrates_persistent_point_on_facing_sphere_side():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="pelvis"><freejoint/><geom name="skin" type="sphere" size=".1"/>
      </body></worldbody></mujoco>""")
    seed = np.tile(model.qpos0, (3, 1))
    bindings = (SurfaceBinding(
        "pelvis", 0, (1, 0, 0), "skin", (0, -0.1, 0), (0, -1, 0)
    ),)
    targets = np.tile([0.0, 0.15, 0.0], (3, 1, 1))
    original = targets.copy()
    calibrated, report = calibrate_contact_bindings(
        model, bindings, np.array([0]), np.ones((3, 1), bool), targets, seed
    )
    np.testing.assert_allclose(calibrated[0].local_point, [0, 0.1, 0])
    np.testing.assert_allclose(calibrated[0].local_normal, [0, 1, 0])
    np.testing.assert_array_equal(targets, original)
    assert bindings[0].local_point == (0, -0.1, 0)
    assert report["calibrated_samples"] == 1


def rig(
    monkeypatch, labels=("left_hand", "right_hand", "torso"), frames=8, release=True
):
    model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
      <worldbody><body name="root"><freejoint/><geom size=".01" pos="0 0 1"/>
        <body name="a" pos="-.3 0 .5"><joint name="ax" type="slide" axis="1 0 0" range="-.4 .4"/>
          <joint name="ar" axis="0 1 0" range="-1 1"/><geom name="a_skin" type="box" size=".05 .07 .1"/></body>
        <body name="b" pos=".3 0 .5"><joint name="bx" type="slide" axis="1 0 0" range="-.4 .4"/>
          <joint name="br" axis="0 1 0" range="-1 1"/><geom name="b_skin" type="box" size=".05 .07 .1"/></body>
        <body name="c" pos="0 .4 .5"><joint name="cr" axis="1 0 0" range="-1 1"/>
          <geom name="c_skin" type="box" size=".1 .05 .15"/></body>
      </body></worldbody></mujoco>""")
    mesh, face_geoms, data = robot_surface_mesh(model, model.qpos0)
    gids = [model.geom(name + "_skin").id for name in ("a", "b", "c")]
    ids = np.flatnonzero(np.isin(face_geoms, gids))
    # Sample planar face centres, including the facing contact patches.
    points = []
    for f in ids:
        centre = data.geom_xpos[face_geoms[f]]
        triangle = mesh.triangles[f]
        normal = mesh.face_normals[f]
        points.append(centre + normal * ((triangle[0] - centre) @ normal))
    points = np.array(points)
    bary = trimesh.triangles.points_to_barycentric(mesh.triangles[ids], points)
    bary = np.maximum(bary, 0)
    bary /= bary.sum(axis=1, keepdims=True)
    q = np.tile(model.qpos0, (frames, 1))
    q[:, model.joint("ax").qposadr[0]] = 0.2475
    q[:, model.joint("bx").qposadr[0]] = -0.2475
    if release:
        q[-2:, model.joint("ax").qposadr[0]] = 0  # Exact source release.
    motion = []
    ref_centres = data.geom_xpos.copy()
    for pose in q:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        motion.append(
            np.array(
                [
                    (mesh.triangles[f] - ref_centres[face_geoms[f]])
                    @ data.geom_xmat[face_geoms[f]].reshape(3, 3).T
                    + data.geom_xpos[face_geoms[f]]
                    for f in ids
                ]
            )
        )
    segment = np.array([gids.index(int(face_geoms[f])) + 1 for f in ids])
    source = {
        "kind": "barycentric_skin",
        "metric_scale": 1.0,
        "parts": {},
        "full_body": {
            "reference_vertices": np.array(mesh.vertices),
            "faces": np.array(mesh.faces),
            "face_ids": ids,
            "barycentric": bary,
            "triangles": np.array(motion),
            "labels": np.array(labels)[segment - 1],
            "sample_joint_indices": segment,
            "joint_parents": np.array([-1, 0, 0, 0]),
            "joint_names": ["root", "a", "b", "c"],
            "sample_effectors": np.array(["left_hand", "right_hand", ""])[segment - 1],
            "reference_root_position": np.zeros(3),
        },
    }
    # Exact same-geometry correspondence isolates motion transfer from training
    # noise; the independent learning/convergence tests exercise the network.
    monkeypatch.setattr(
        surface_learning,
        "learn_indexed_points",
        lambda human, robot, edges, config, **kw: (human.copy(), {"loss": 0.0}),
    )
    mapping = {x: x for x in ("root", "a", "b", "c")}
    shared = SurfaceMotion(model, source, model.qpos0, {"cache_dir": None}, mapping)
    return model, source, mapping, q, shared


@pytest.mark.parametrize(
    "labels",
    [
        ("left_hand", "right_hand", "chest"),
        ("hand", "abdomen", "leg"),
        ("left_forearm", "right_upper_arm", "torso"),
    ],
)
def test_arbitrary_nonadjacent_surface_pairs_and_exact_release(monkeypatch, labels):
    model, source, mapping, q, shared = rig(monkeypatch, labels)
    before = copy.deepcopy(source)
    task, report = build_self_contact_tasks(
        model, source, mapping, model.qpos0, surface_motion=shared
    )
    assert report["status"] == "active"
    assert [(e["part_a"], e["part_b"]) for e in task.events] == [labels[:2]]
    assert task.events[0]["stop"] == len(q) - 2
    np.testing.assert_allclose(task.errors(q)[:-2], 0, atol=1e-12)
    assert list(task.rows(q[-1], len(q) - 1, np.arange(model.nv))) == []
    for name, value in before["full_body"].items():
        np.testing.assert_array_equal(source["full_body"][name], value)


def test_excludes_adjacent_segments_and_both_primary_contact_endpoints(monkeypatch):
    model, source, mapping, q, shared = rig(monkeypatch)
    for hand in ("left_hand", "right_hand"):
        primary = {hand: _task(np.zeros((len(q), 3)), np.ones(len(q), bool))}
        task, report = build_self_contact_tasks(
            model, source, mapping, model.qpos0, primary, surface_motion=shared
        )
        assert task is None, report
        assert report["scene_excluded_frame_samples"] > 0
    source["full_body"]["joint_parents"][2] = 1
    task, _ = build_self_contact_tasks(
        model, source, mapping, model.qpos0, surface_motion=shared
    )
    assert task is None


def test_geodesic_neighbours_are_not_self_contact_but_disconnected_skin_can_touch():
    from omniretargeting.full_body_self_contact import SurfaceAdjacency

    vertices = np.array([[0, 0, 0], [0.04, 0, 0], [0, 0.04, 0], [0.04, 0.04, 0]])
    full = {
        "reference_vertices": vertices,
        "faces": np.array([[0, 1, 2], [1, 3, 2]]),
        "face_ids": np.array([0, 1]),
        "barycentric": np.full((2, 3), 1 / 3),
    }
    assert not SurfaceAdjacency(full, 0.12).separate(0, 1)
    full.update(
        reference_vertices=np.concatenate([vertices[:3], vertices[:3] + [0, 0, 0.002]]),
        faces=np.array([[0, 1, 2], [3, 4, 5]]),
    )
    assert SurfaceAdjacency(full, 0.12).separate(0, 1)


def test_general_contact_keeps_single_frame_touch(monkeypatch):
    model, source, mapping, _q, shared = rig(monkeypatch, frames=3)
    task, report = build_self_contact_tasks(
        model, source, mapping, model.qpos0, surface_motion=shared
    )
    assert report["active_frame_pairs"] == 1
    np.testing.assert_array_equal(task.confidence[:, 0], [1, 0, 0])


def test_changing_nearest_rank_does_not_replace_a_valid_material_pair(monkeypatch):
    model, source, mapping, _q, shared = rig(monkeypatch)
    original, _ = build_self_contact_tasks(
        model, source, mapping, model.qpos0, surface_motion=shared
    )
    selected = original.events[0]["sample_a"]
    # Move just this sampled patch by 1 mm, while other equally near patches
    # retain their old rank. The selected pair is still a valid contact.
    source["full_body"]["triangles"][2:4, selected] += [0, 0.001, 0]
    updated = SurfaceMotion(model, source, model.qpos0, {"cache_dir": None}, mapping)
    task, report = build_self_contact_tasks(
        model, source, mapping, model.qpos0, surface_motion=updated
    )
    assert report["event_count"] == 1
    assert task.events[0]["sample_a"] == selected
    assert task.events[0]["sample_b"] == original.events[0]["sample_b"]


def test_general_self_contact_jacobian_includes_both_bodies(monkeypatch):
    model, source, mapping, q, shared = rig(monkeypatch)
    task, _ = build_self_contact_tasks(
        model, source, mapping, model.qpos0, surface_motion=shared
    )
    pose = q[0].copy()
    pose[model.joint("br").qposadr[0]] = 0.15
    value, jac, normal, jnormal = task.at(pose, 0)
    np.testing.assert_allclose(jac[:, :6], 0, atol=1e-12)
    for d in range(model.nv):
        moved = pose.copy()
        velocity = np.eye(model.nv)[d]
        mujoco.mj_integratePos(model, moved, velocity, 1e-6)
        v2, _, n2, _ = task.at(moved, 0)
        np.testing.assert_allclose((v2 - value) / 1e-6, jac[:, d], atol=2e-6)
        np.testing.assert_allclose((n2 - normal) / 1e-6, jnormal[:, d], atol=2e-6)


def pose_case(monkeypatch, *, release=True, **kwargs):
    model, source, mapping, q, shared = rig(monkeypatch, release=release)
    task, report = build_surface_pose_tasks(
        model,
        source,
        model.qpos0,
        np.zeros((len(q), 3)),
        np.zeros(3),
        mapping=mapping,
        surface_motion=shared,
        **kwargs,
    )
    assert report["status"] == "active", report
    return model, source, mapping, q, shared, task


def test_dense_pose_without_scene_tracks_surface_and_free_root_translation(monkeypatch):
    model, _, _, q, shared, task = pose_case(monkeypatch)
    # One stacked position block and one normal block cover every sample.
    rows = list(task.rows(q[0], 0, np.arange(model.nv)))
    assert sum(len(residual) for _, residual, _ in rows) == 6 * shared.valid[0].sum()
    assert task.metrics(q)["position_error_m"]["max"] < 1e-12
    assert task.metrics(q)["normal_error_rad"]["max"] < 1e-12
    translated = q.copy()
    translated[:, :3] += [1, -2, 0.8]
    assert task.metrics(translated)["position_error_m"]["max"] < 1e-12
    # Posture normals are active on a body that never contacts anything.
    rotated = q.copy()
    rotated[:, model.joint("cr").qposadr[0]] = 0.3
    assert task.metrics(rotated)["normal_error_rad"]["max"] > 0.25


def test_surface_pose_jacobian_and_world_feet_semantics(monkeypatch):
    priority = {
        "effectors": {"foot": {"robot_body": "b", "preserve_world_target": True}}
    }
    model, _, _, q, shared, task = pose_case(monkeypatch, priority=priority)
    for index in [0, int(np.flatnonzero(shared.body_ids == model.body("b").id)[0])]:
        pose = q[0].copy()
        pose[model.joint("ar").qposadr[0]] = 0.1
        residual, jac, normal, jnormal = task.at(pose, 0, index)
        for d in range(model.nv):
            moved = pose.copy()
            mujoco.mj_integratePos(model, moved, np.eye(model.nv)[d], 1e-6)
            r2, _, n2, _ = task.at(moved, 0, index)
            np.testing.assert_allclose((r2 - residual) / 1e-6, jac[:, d], atol=2e-6)
            np.testing.assert_allclose((n2 - normal) / 1e-6, jnormal[:, d], atol=2e-6)
    moved = q.copy()
    moved[:, 0] += 0.1
    assert task.metrics(moved)["position_error_m"]["max"] == pytest.approx(0.1)
    # Stacked rows equal the per-sample rows scaled by sqrt(weight)/sigma.
    pose = moved[0].copy()
    pose[model.joint("ar").qposadr[0]] = 0.1
    expected_r, expected_j = [], []
    for index in np.flatnonzero(task.valid[0]):
        residual, jac, normal, jnormal = task.at(pose, 0, int(index))
        w = np.sqrt(task.weights[index])
        if task.position_active[0, index]:
            expected_r.append(w * residual / task.cfg["position_sigma"])
            expected_j.append(w * jac / task.cfg["position_sigma"])
        if task.normal_active[0, index]:
            expected_r.append(w * normal / task.cfg["normal_sigma"])
            expected_j.append(w * jnormal / task.cfg["normal_sigma"])
    expected_r, expected_j = np.concatenate(expected_r), np.vstack(expected_j)
    rows = list(task.rows(pose, 0, np.arange(model.nv)))
    stacked_r = np.concatenate([np.sqrt(w) * r for w, r, _ in rows])
    stacked_j = np.vstack([np.sqrt(w) * j for w, _, j in rows])
    np.testing.assert_allclose(stacked_r @ stacked_r, expected_r @ expected_r)
    np.testing.assert_allclose(stacked_j.T @ stacked_r, expected_j.T @ expected_r, atol=1e-10)
    np.testing.assert_allclose(stacked_j.T @ stacked_j, expected_j.T @ expected_j, atol=1e-10)


def test_missing_capabilities_disable_explicitly_and_invalid_topology_fails(
    monkeypatch,
):
    model, source, mapping, q, shared = rig(monkeypatch)
    task, report = build_surface_pose_tasks(model, source, model.qpos0, q[:, :3], None)
    assert task is None and report["reason"] == "missing_source_root_reference"
    source["full_body"]["joint_parents"] = np.array([-1, 2, 1, 0])
    with pytest.raises(ValueError, match="Cyclic"):
        build_self_contact_tasks(
            model, source, mapping, model.qpos0, surface_motion=shared
        )


def _markers(model, poses, names=("root", "a", "b", "c")):
    data = mujoco.MjData(model)
    result = []
    for pose in poses:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        result.append([data.xpos[model.body(n).id].copy() for n in names])
    return np.asarray(result)


def test_sequence_surface_normals_resolve_pose_invisible_to_joint_positions(
    monkeypatch,
):
    from omniretargeting.sequence_retargeting import retarget_sequence

    model, source, mapping, q, shared = rig(monkeypatch)
    positions = _markers(model, q)
    contacts = {"hand": _task(positions[:, 2], np.ones(len(q), bool))}
    priority = {
        "effectors": {"hand": {"robot_body": "b", "preserve_world_target": True}}
    }
    task, _ = build_surface_pose_tasks(
        model,
        source,
        model.qpos0,
        positions[:, 0],
        np.zeros(3),
        mapping=mapping,
        surface_motion=shared,
        scene_tasks=contacts,
        priority=priority,
    )
    initial = q.copy()
    index = model.joint("cr").qposadr[0]
    initial[:, index] = 0.3
    common = {
        "framerate": 30,
        "config": {
            "iterations": 6,
            "self_collision_enabled": False,
            "lock_root_xy": True,
            "lock_root_z": True,
            "lock_root_orientation": True,
        },
    }
    args = (
        model,
        initial,
        positions,
        list(mapping),
        mapping,
        positions[:, 0],
        contacts,
        priority,
    )
    baseline, _ = retarget_sequence(*args, **common)
    result, report = retarget_sequence(*args, extra_tasks=(task,), **common)
    assert report["surface_endpoint_guard"]["enabled"]
    assert report["surface_endpoint_guard"]["linear_failures"] == 0
    np.testing.assert_allclose(baseline[:, index], 0.3, atol=1e-8)
    # Finite reference regularization intentionally leaves a small compromise.
    assert abs(result[:, index]).max() < 0.06
    assert (
        np.linalg.norm(_markers(model, result)[:, 2] - positions[:, 2], axis=1).max()
        <= 0.00101
    )


def test_dense_surface_pose_flows_through_refinement_and_validation(monkeypatch):
    from omniretargeting.scene_geometry import SceneQuery
    from omniretargeting.trajectory_refinement import refine_joint_trajectory
    from omniretargeting.trajectory_validation import (
        TrajectoryValidator,
        local_repair_config,
    )

    model, _, _, q, _, task = pose_case(monkeypatch, release=False)
    initial = q.copy()
    index = model.joint("cr").qposadr[0]
    initial[:, index] = 0.3
    result = refine_joint_trajectory(
        model,
        initial,
        {},
        {},
        {"iterations": 4, "self_collision_enabled": False, "reference_sigma": 1.0},
        surface_pose_tasks=task,
    )
    assert abs(result[:, index]).max() < 0.25
    np.testing.assert_array_equal(result[:, :7], initial[:, :7])
    scene = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    scene.apply_translation([10, 10, 10])
    validator = TrajectoryValidator(
        model,
        initial,
        SceneQuery(scene),
        {},
        {},
        local_repair_config({"self_collision_enabled": False}),
        30,
        surface_pose_tasks=task,
    )
    assert validator.relation_score(result) < validator.relation_score(initial)
    assert "surface_pose" in validator.fidelity(result)["metrics"]


def test_added_dense_pose_preserves_framewise_primary_endpoint(monkeypatch):
    from omniretargeting.retargeting import GenericInteractionRetargeter

    model, _, mapping, _q, _, task = pose_case(monkeypatch, config={"weight": 10.0})
    initial = model.qpos0.copy()
    positions = _markers(model, initial[None])[0]
    scene = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
    scene.apply_translation([10, 10, 10])
    solver = GenericInteractionRetargeter(
        model,
        mujoco.MjData(model),
        scene,
        mapping,
        1.0,
        terrain_sample_points=4,
        self_collision_enabled=False,
        hard_penetration_constraint=False,
        end_effector_config={
            "enabled": True,
            "hierarchy": True,
            "free_motion_priority": "balanced",
            "pose_direction": {"enabled": False},
            "effectors": {"hand": {"robot_body": "b", "preserve_world_target": True}},
        },
    )
    solver.surface_pose_tasks = task
    target = _task(positions[None, 2], np.ones(1, bool)).frame(0)
    result = solver.retarget_frame(
        positions,
        initial,
        max_iter=3,
        relation_frame=0,
        end_effector_targets={"hand": target},
    )
    error = np.linalg.norm(_markers(model, result[None])[0, 2] - positions[2])
    assert error < 0.002
    assert np.linalg.norm(result - initial) > 1e-4


def test_surface_target_frame_follows_per_frame_clearance_not_adapter_part_labels(monkeypatch):
    from dataclasses import replace
    model, source, mapping, q, _ = rig(monkeypatch)
    shared = SurfaceMotion(model, source, model.qpos0, {"cache_dir": None}, mapping)
    group = model.body_weldid[shared.body_ids] == model.body_weldid[model.body("b").id]
    world = np.arange(len(q)) % 2 == 0
    endpoint = replace(_task(np.zeros((len(q), 3)), np.zeros(len(q), bool)), swing_world_mask=world)
    task, _ = build_surface_pose_tasks(
        model, source, model.qpos0, q[:, :3], np.zeros(3),
        config={"morphology_scale": 0.5}, mapping=mapping, surface_motion=shared,
        scene_tasks={"any_region": endpoint}, priority={"effectors": {"any_region": {"robot_body": "b"}}},
    )
    np.testing.assert_array_equal(task.targets[world][:, group], shared.points[world][:, group])
    assert not task.root_relative[world][:, group].any()
    assert task.root_relative[~world][:, group].all()


def test_core_uses_selected_root_in_surface_reference_coordinates(monkeypatch):
    from types import SimpleNamespace

    from omniretargeting._core.surfaces import prepare_surface_tasks

    model, source, mapping, q, _ = rig(monkeypatch)
    source["full_body"]["reference_joint_positions"] = _markers(
        model, model.qpos0[None]
    )[0]
    owner = SimpleNamespace(
        robot_model=model,
        _body_surface_samples=source,
        valid_joint_mapping=mapping,
        smplx_joint_names=list(mapping),
        _resolve_source_root_index=lambda: 3,
        retargeting_config={"surface_pose": {"correspondence": {"cache_dir": None}}},
    )
    solver = SimpleNamespace(robot_reference_qpos=model.qpos0)
    prepare_surface_tasks(owner, solver, _markers(model, q), None, None)
    assert solver.surface_pose_tasks is not None
    assert solver.surface_pose_tasks.metrics(q)["position_error_m"]["max"] < 1e-12
