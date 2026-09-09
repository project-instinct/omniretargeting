"""Analytic boxes: regressions must not depend on a reconstructed scan."""

import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting.trajectory_refinement import (
    ScenePenetrationProbe,
    collision_geom_ids,
    geom_surface_offsets,
    measure_scene_penetration,
)


def model_with_geom(kind="sphere", size="0.02"):
    return mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody>
      <body name="probe"><freejoint/><geom type="{kind}" size="{size}"/>
      </body></worldbody></mujoco>''')


def test_deep_inside_box_is_not_culled_or_reinterpreted_as_free():
    model = model_with_geom()
    scene = trimesh.creation.box(extents=[4, 4, 4])
    q = model.qpos0[None].copy()
    stats = measure_scene_penetration(model, q, scene, {"margin": 0.01})
    assert stats["max_depth"] >= 1.97
    assert stats["frames_penetrating"] == 1


@pytest.mark.parametrize(
    "point,direction",
    [
        ([0.95, 0, 0], [1, 0, 0]),
        ([0, 0, -0.95], [0, 0, -1]),
    ],
)
def test_closed_box_exit_direction_never_forced_up(point, direction):
    model = model_with_geom()
    scene = trimesh.creation.box(extents=[2, 2, 2])
    probe = ScenePenetrationProbe(model, scene, collision_geom_ids(model), 0.01)
    q = model.qpos0[None].copy()
    q[0, :3] = point
    frame = probe.probe_trajectory(q)[0]
    assert np.all(frame["clearance"] < 0)
    assert np.min(frame["directions"] @ direction) > 0.99


def test_box_corner_coverage_detects_toe_corner():
    model = model_with_geom("box", "0.1 0.1 0.1")
    scene = trimesh.creation.box(extents=[0.08] * 3)
    scene.apply_translation([0.12, 0.12, 0.12])
    stats = measure_scene_penetration(model, model.qpos0[None], scene)
    assert stats["max_depth"] == pytest.approx(0.02)
    offsets = geom_surface_offsets(model, 0)
    assert np.any(np.all(np.isclose(offsets, [0.1, 0.1, 0.1]), axis=1))


def test_tilted_box_bottom_uses_full_exit_vector():
    model = model_with_geom()
    transform = trimesh.transformations.rotation_matrix(0.6, [0, 1, 0])
    scene = trimesh.creation.box(extents=[2, 2, 2], transform=transform)
    q = model.qpos0[None].copy()
    q[0, :3] = transform[:3, :3] @ [0, 0, -0.95]
    probe = ScenePenetrationProbe(model, scene, collision_geom_ids(model), 0.1)
    frame = probe.probe_trajectory(q)[0]
    expected = transform[:3, :3] @ [0, 0, -1]
    assert np.all(frame["clearance"] < 0)
    assert np.min(frame["directions"] @ expected) > 0.99


def test_open_mesh_without_evidence_is_unknown_even_far_from_surface():
    from omniretargeting.scene_geometry import SceneQuery

    mesh = trimesh.Trimesh(
        vertices=[[0, -2, -2], [0, 2, -2], [0, 2, 2], [0, -2, 2]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    query = SceneQuery(mesh)
    result = query.query([[2, 0, 0], [-2, 0, 0]])
    assert np.all(result["uncertain"])
    assert np.all(np.isnan(result["clearance"]))
    model = model_with_geom()
    q = model.qpos0[None].copy()
    q[0, 0] = -2
    frame = ScenePenetrationProbe(
        model, mesh, collision_geom_ids(model), 0.01, query=query
    ).probe_trajectory(q)[0]
    assert len(frame["points"]) > 0  # Unknown deep points survive broad phase.


def test_open_wall_uses_fixed_local_evidence_not_current_robot_side():
    from omniretargeting.scene_geometry import SceneQuery

    mesh = trimesh.Trimesh(
        vertices=[[0, -2, -2], [0, 2, -2], [0, 2, 2], [0, -2, 2]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    query = SceneQuery(mesh, [[0, 0, 0]], [[1, 0, 0]])
    for x in (0.2, -0.2, -2, 0.1):
        result = query.query([[x, 0, 0]])
        assert result["clearance"][0] == pytest.approx(x)
        np.testing.assert_allclose(result["directions"][0], [1, 0, 0])
        assert result["confidence"][0] == 0.5
    assert query.query([[0.2, 1, 1]])["uncertain"][0]


def test_mixed_winding_is_uncertain_without_evidence():
    from omniretargeting.scene_geometry import SceneQuery

    mesh = trimesh.creation.box()
    mesh.faces[0] = mesh.faces[0, ::-1]
    query = SceneQuery(mesh)
    assert query.query([[0, 0, 0]])["uncertain"][0]


def test_capsule_barrel_diagonal_and_mesh_hull_are_sampled():
    model = model_with_geom("capsule", "0.05 0.2")
    points = geom_surface_offsets(model, 0, spacing=0.02)
    assert np.any(
        np.linalg.norm(points - [0.05 / np.sqrt(2), 0.05 / np.sqrt(2), 0.12], axis=1)
        < 0.02
    )
    meshmodel = mujoco.MjModel.from_xml_string("""<mujoco><asset>
      <mesh name="tetra" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
      </asset><worldbody><body><freejoint/><geom type="mesh" mesh="tetra"/></body></worldbody></mujoco>""")
    offsets = geom_surface_offsets(meshmodel, 0, spacing=0.1)
    vertices = meshmodel.mesh_vert
    assert all(np.min(np.linalg.norm(offsets - v, axis=1)) < 1e-6 for v in vertices)
    assert len(offsets) > len(vertices)
