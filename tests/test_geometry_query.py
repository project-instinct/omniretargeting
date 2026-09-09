import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting.geometry_query import (
    GeometryQueryError,
    RigidGeometryQuery,
    distance_direction,
    relative_point_jacobian,
)
from omniretargeting.scene_geometry import (
    ScenePenetrationProbe,
    SceneQuery,
    collision_geom_ids,
)
from tests.test_local_collision_repair import setup_motion


def test_full_geometry_detects_small_obstacle_enclosed_between_robot_samples():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body><freejoint/>
    <geom type="box" size=".5 .5 .5"/></body></worldbody></mujoco>""")
    scene = trimesh.creation.box(extents=[0.02] * 3)
    scene.apply_translation([0.23, 0.17, 0.11])
    depths = []
    for full in (False, True):
        query = SceneQuery(scene, full_geometry=full)
        probe = ScenePenetrationProbe(model, scene, [0], 0.05, query=query)
        frame = probe.probe_trajectory(model.qpos0[None])[0]
        depths.append(np.max(-frame["clearance"], initial=0))
    assert depths[0] == 0
    assert depths[1] > 0.25


@pytest.mark.parametrize("x", [-0.09, -0.07, -0.01])
def test_signed_distance_jacobian_matches_nv_tangent_finite_difference(x):
    model, q, scene = setup_motion(1)
    q[0, 7:9] = [x, 0.013]
    query = RigidGeometryQuery(model, collision_geom_ids(model), SceneQuery(scene))
    data = mujoco.MjData(model)
    data.qpos[:] = q[0]
    mujoco.mj_forward(model, data)
    row = next(
        row for row in query.query(data, 0.2) if row["geom_ids"] == model.geom("toe").id
    )
    jac = relative_point_jacobian(
        model, data, row["points"], row["body_ids"], row["directions"]
    )
    assert jac.shape == (model.nv,) and model.nv != model.nq
    for dof in [0, 1, 2, 6, 7]:
        distances = []
        for sign in [-1, 1]:
            tangent = np.zeros(model.nv)
            tangent[dof] = sign * 1e-6
            data.qpos[:] = q[0]
            mujoco.mj_integratePos(model, data.qpos, tangent, 1)
            mujoco.mj_forward(model, data)
            distances.append(
                next(
                    r["clearance"]
                    for r in query.query(data, 0.2)
                    if r["geom_ids"] == row["geom_ids"]
                )
            )
        assert jac[dof] == pytest.approx((distances[1] - distances[0]) / 2e-6, abs=2e-4)


def test_query_scene_fixed_and_named_mapping_does_not_assume_prefix():
    model, q, scene = setup_motion(3)
    extra = scene.copy()
    extra.apply_translation([0, 0, 2])
    scene = trimesh.util.concatenate([scene, extra])
    query = RigidGeometryQuery(model, collision_geom_ids(model), SceneQuery(scene))
    original = [model.body_pos.copy(), model.geom_size.copy(), scene.vertices.copy()]
    scene_positions = query.model.body_pos.copy()
    data = mujoco.MjData(model)
    for pose in q:
        data.qpos[:] = pose
        mujoco.mj_forward(model, data)
        query.query(data, 0.1)
    assert query.model.nq == query.model.nv == query.model.njnt == 0
    assert any(gid != mapped[0] for gid, mapped in query.mapping.items())
    for gid in query.scene_ids.values():
        assert query.model.body_mocapid[query.model.geom_bodyid[gid]] == -1
    np.testing.assert_array_equal(scene_positions, query.model.body_pos)
    for before, after in zip(
        original, [model.body_pos, model.geom_size, scene.vertices]
    ):
        np.testing.assert_array_equal(before, after)
    assert query.max_vertex_deviation < 1e-6


def test_compound_scene_keeps_free_gap_and_concave_component_uses_triangle_fallback():
    model, q, _ = setup_motion(1)
    q[0, 7:9] = 0
    left = trimesh.creation.box(extents=[0.1, 0.4, 0.4])
    left.apply_translation([-0.2, 0, 0])
    right = left.copy()
    right.apply_translation([0.4, 0, 0])
    scene = trimesh.util.concatenate([left, right])
    query = RigidGeometryQuery(model, collision_geom_ids(model), SceneQuery(scene))
    assert len(query.scene_ids) == 2
    data = mujoco.MjData(model)
    data.qpos[:] = q[0]
    mujoco.mj_forward(model, data)
    assert all(r["clearance"] > 0 for r in query.query(data, 0.3))
    # A single closed concave extruded L. Do not replace it with a convex hull.
    vertices = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], float)
    from trimesh.creation import extrude_triangulation

    concave = extrude_triangulation(
        vertices, np.array([[0, 1, 3], [1, 2, 3], [0, 3, 5], [3, 4, 5]]), 1
    )
    backend = RigidGeometryQuery(model, collision_geom_ids(model), SceneQuery(concave))
    assert not backend.scene_ids
    assert backend.describe()["triangle_fallback_components"] == [0]


def test_missing_geometry_and_invalid_direction_are_explicit():
    with pytest.raises(ValueError, match="nonempty"):
        SceneQuery(trimesh.Trimesh())
    model, _q, scene = setup_motion(1)
    for enabled in (False, True):
        with pytest.raises(ValueError, match="active collision geoms"):
            ScenePenetrationProbe(
                model, scene, [], 0.05, query=SceneQuery(scene, full_geometry=enabled)
            )
    with pytest.raises(GeometryQueryError, match="coincident"):
        distance_direction(0, np.zeros(6))
    np.testing.assert_array_equal(
        distance_direction(0, np.zeros(6), [0, 1, 0]), [0, -1, 0]
    )
    with pytest.raises(GeometryQueryError, match="nonfinite"):
        distance_direction(np.nan, np.zeros(6))


def test_failed_query_is_uncertain_and_logged(monkeypatch):
    model, q, scene = setup_motion(1)
    q[0, 7:9] = 0
    query = RigidGeometryQuery(model, collision_geom_ids(model), SceneQuery(scene))
    data = mujoco.MjData(model)
    data.qpos[:] = q[0]
    mujoco.mj_forward(model, data)

    def fail(*args):
        raise ValueError("injected distance failure")

    monkeypatch.setattr(mujoco, "mj_geomDistance", fail)
    rows = query.query(data, 0.1)
    assert rows and all(row["uncertain"] for row in rows)
    assert query.describe()["failures"]


def test_asymmetric_compiled_robot_mesh_preserves_geometry_after_query_recompile():
    from scipy.spatial import cKDTree

    model = mujoco.MjModel.from_xml_string("""<mujoco><asset>
    <mesh name="shape" vertex="0 0 0 .1 0 0 0 .2 0 0 0 .3"/>
    </asset><worldbody><body><freejoint/><geom type="mesh" mesh="shape" pos=".2 .3 .4"/>
    </body></worldbody></mujoco>""")
    scene = trimesh.creation.box(extents=[0.1] * 3)
    query = RigidGeometryQuery(model, [0], SceneQuery(scene))
    data = mujoco.MjData(model)
    data.qpos[:] = [0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5]
    mujoco.mj_forward(model, data)
    query.query(data, 0.5)
    qg = query.mapping[0][0]

    def vertices(m, d, g):
        mid = int(m.geom_dataid[g])
        start = int(m.mesh_vertadr[mid])
        count = int(m.mesh_vertnum[mid])
        return (
            m.mesh_vert[start : start + count] @ d.geom_xmat[g].reshape(3, 3).T
            + d.geom_xpos[g]
        )

    original = vertices(model, data, 0)
    copied = vertices(query.model, query.data, qg)
    assert cKDTree(original).query(copied)[0].max() < 1e-6
    assert query.robot_max_vertex_deviation < 1e-6
