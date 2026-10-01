"""Clearance transfer is about actual surfaces and does not invent contact."""

import igl
import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting._sequence.proximity import SurfaceProximityTasks, support_points
from omniretargeting.scene_geometry import ScenePenetrationProbe, SceneQuery


def test_gap_preservation_uses_collision_shell_and_is_sampling_invariant():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="body" pos="0 0 .2"><freejoint/><geom name="shell" size=".1"/>
      </body></worldbody></mujoco>""")
    pose = model.qpos0.copy()
    costs = []
    for copies in (1, 9):
        task = SurfaceProximityTasks(
            model,
            [0] * copies,
            ["body"] * copies,
            np.zeros((1, copies, 3)),
            np.tile([0.0, 0.0, 0.04], (1, copies, 1)),
            np.tile([0.0, 0.0, 1.0], (1, copies, 1)),
            np.ones((1, copies), bool),
            distance_scale=0.03,
            sigma=0.015,
            weight=5.0,
        )
        weight, residual, jac = next(task.rows(pose, 0, np.arange(model.nv)))
        costs.append(weight * float(residual @ residual))
        correct = pose.copy()
        correct[2] = 0.14
        assert (
            np.linalg.norm(next(task.rows(correct, 0, np.arange(model.nv)))[1]) < 1e-12
        )
        delta = np.zeros(model.nv)
        delta[2] = 1e-6
        plus, minus = pose.copy(), pose.copy()
        mujoco.mj_integratePos(model, plus, delta, 1)
        mujoco.mj_integratePos(model, minus, delta, -1)
        fplus = np.linalg.norm(next(task.rows(plus, 0, np.arange(model.nv)))[1]) ** 2
        fminus = np.linalg.norm(next(task.rows(minus, 0, np.arange(model.nv)))[1]) ** 2
        np.testing.assert_allclose(
            (fplus - fminus) / 2e-6, 2 * residual @ jac[:, 2], rtol=1e-6
        )
    np.testing.assert_allclose(costs[0], costs[1])


def test_surface_extent_preserves_two_orthogonal_nearby_surfaces():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body pos=".14 0 .14"><freejoint/><geom size=".1"/></body>
      </worldbody></mujoco>""")
    environment = np.array([[[0.0, 0.0, 0.2], [0.2, 0.0, 0.0]]])
    vectors = np.array([[[0.04, 0.0, 0.0], [0.0, 0.0, 0.04]]])
    normals = np.array([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    task = SurfaceProximityTasks(
        model,
        [0, 0],
        ["region", "region"],
        environment,
        vectors,
        normals,
        np.ones((1, 2), bool),
        distance_scale=0.03,
        sigma=0.015,
        weight=5.0,
    )
    assert np.linalg.norm(next(task.rows(model.qpos0, 0, np.arange(6)))[1]) < 1e-12
    for axis in (0, 2):
        pose = model.qpos0.copy()
        pose[axis] += 0.1
        assert np.linalg.norm(next(task.rows(pose, 0, np.arange(6)))[1]) > 1


def test_far_side_source_samples_do_not_raise_the_closest_surface_target():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body pos="0 0 .14"><freejoint/><geom size=".1"/></body>
      </worldbody></mujoco>""")
    task = SurfaceProximityTasks(
        model,
        [0, 0],
        ["region", "region"],
        np.zeros((1, 2, 3)),
        np.array([[[0.0, 0.0, 0.04], [0.0, 0.0, 0.2]]]),
        np.tile([0.0, 0.0, 1.0], (1, 2, 1)),
        np.ones((1, 2), bool),
        distance_scale=0.03,
        sigma=0.015,
        weight=5.0,
    )
    np.testing.assert_allclose(task.support_gaps, 0.04)
    assert np.linalg.norm(next(task.rows(model.qpos0, 0, np.arange(6)))[1]) < 1e-12


@pytest.mark.parametrize(
    "kind,size",
    [
        ("box", ".1 .2 .3"),
        ("capsule", ".1 .2"),
        ("sphere", ".1"),
        ("ellipsoid", ".1 .2 .3"),
        ("cylinder", ".1 .2"),
    ],
)
def test_support_points_agree_with_mujoco_extents(kind, size):
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody><body>
      <geom type="{kind}" size="{size}"/></body></worldbody></mujoco>''')
    points = support_points(model, 0, np.r_[np.eye(3), -np.eye(3)])
    np.testing.assert_allclose(points[:3], -points[3:])
    assert np.all(np.isfinite(points))
    assert np.all(np.diag(points[:3]) > 0)


def test_enclosed_free_space_keeps_room_interior_and_obstacle_holes_separate():
    room = trimesh.creation.box(extents=[4, 4, 4])
    room.invert()
    obstacle = trimesh.creation.box(extents=[0.4, 0.4, 0.4])
    scene = trimesh.util.concatenate([room, obstacle])
    points = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.1, 0.0, 0.0]])
    query = SceneQuery(scene, enclosed_free_space=True)
    clearance = query.query(points)["clearance"]
    assert clearance[0] > 0
    assert clearance[1] < 0
    assert clearance[2] < 0
    solid = SceneQuery(room)
    assert solid.query(points[:1])["clearance"][0] < 0


def test_convex_free_room_is_not_sent_to_the_solid_collision_backend():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body>
      <freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>""")
    room = trimesh.creation.box(extents=[4, 4, 4])
    room.invert()
    query = SceneQuery(room, enclosed_free_space=True)
    probe = ScenePenetrationProbe(model, room, [0], 0.01, query=query)
    inside, crossing, outside = np.tile(model.qpos0, (3, 1))
    crossing[0], outside[0] = 1.95, 2.3
    frames = probe.probe_trajectory([inside, crossing, outside])
    assert np.all(frames[0]["clearance"] > 0)
    assert frames[1]["clearance"].min() < 0
    assert frames[2]["clearance"].min() < 0
    assert not probe.geometry_backend.parts


def test_approximate_winding_ambiguity_uses_exact_containment(monkeypatch):
    mesh = trimesh.creation.box()
    points = np.array([[0.0, 0.0, 0.0], [0.6, 0.0, 0.0]])
    monkeypatch.setattr(
        igl, "fast_winding_number", lambda v, f, p: np.full(len(p), 0.4)
    )
    result = SceneQuery(mesh).query(points)
    assert not result["uncertain"].any()
    np.testing.assert_allclose(result["clearance"], [-0.5, 0.1])


def test_free_space_contact_normals_do_not_flip_when_skin_crosses_boundary():
    from omniretargeting.contact import infer_scene_contact_tasks

    room = trimesh.creation.box(extents=[4, 4, 2])
    room.apply_translation([0, 0, 1])
    room.invert()
    for gap in (-0.02, 0.02):
        points = np.tile([0.0, 0.0, gap], (10, 1))
        tasks = infer_scene_contact_tasks(
            {
                "primary": {
                    "tracking_positions": points,
                    "contact_samples": points[:, None],
                    "up_directions": np.tile([0.0, 0.0, 1.0], (10, 1)),
                }
            },
            room,
            30.0,
            {"scene_enclosed_free_space": True, "min_contact_frames": 1},
        )
        assert tasks["primary"].contact_mask.all()
        np.testing.assert_allclose(
            tasks["primary"].surface_normals, np.tile([0.0, 0.0, 1.0], (10, 1))
        )


@pytest.mark.parametrize("reverse", [False, True])
def test_cached_pseudonormals_match_independent_libigl_and_winding(reverse):
    mesh = trimesh.creation.icosphere(subdivisions=2)
    if reverse:
        mesh.invert()
    rng = np.random.default_rng(30)
    points = np.r_[
        rng.normal(size=(600, 3)),
        np.asarray(mesh.vertices) * 1.01,
        np.asarray(mesh.vertices) * 0.99,
    ]
    actual = SceneQuery(mesh, closed_sign_method="pseudonormal").query(points)
    winding = SceneQuery(mesh).query(points)
    distance = igl.signed_distance(
        points,
        np.asarray(mesh.vertices),
        np.asarray(mesh.faces),
        igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL,
    )[0]
    if reverse:
        distance = -distance
    np.testing.assert_allclose(actual["clearance"], distance, atol=1e-8)
    np.testing.assert_allclose(actual["clearance"], winding["clearance"], atol=1e-8)
    assert not actual["uncertain"].any()
