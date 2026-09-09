import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting.geometry_query import RigidGeometryQuery, relative_point_jacobian
from omniretargeting.scene_geometry import SceneQuery, ScenePenetrationProbe
from omniretargeting.triangle_capsule import segment_triangle_witnesses


@pytest.mark.parametrize("flip", [False, True])
def test_open_finite_triangle_intersection_independent_of_winding(flip):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <freejoint/><geom type="capsule" size=".05 .2"/>
      </body></worldbody></mujoco>''')
    # Tiny horizontal triangle is inside the capsule between sparse shell points.
    scene = trimesh.Trimesh([[.005, 0, .113], [.008, 0, .113], [.005, .003, .113]],
                            [[0, 2, 1] if flip else [0, 1, 2]], process=False)
    query = SceneQuery(scene)
    probe = ScenePenetrationProbe(model, scene, [0], .1, query=query)
    frame = probe.probe_trajectory(model.qpos0[None])[0]
    assert np.max(-frame["clearance"][np.isfinite(frame["clearance"])]) == pytest.approx(.045)
    # Open scan uncertainty is retained; intersection is not a solid-side oracle.
    assert np.any(frame["uncertain"])
    row = probe.geometry_backend.query(_data(model), .1)[-1]
    np.testing.assert_allclose(row["directions"], [-1, 0, 0], atol=1e-10)


def _data(model):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return data


def test_axis_crossing_retains_depth_without_inventing_exit_direction():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><freejoint/>
      <geom type="capsule" size=".05 .2"/></body></worldbody></mujoco>''')
    scene = trimesh.Trimesh([[-1,-1,0], [1,-1,0], [0,1,0]], [[0,1,2]], process=False)
    query = RigidGeometryQuery(model, [0], SceneQuery(scene))
    row = query.query(_data(model), .1)[0]
    assert row["clearance"] == pytest.approx(-.25)
    assert row["uncertain"]
    np.testing.assert_array_equal(row["directions"], np.zeros(3))


@pytest.mark.parametrize('center,expected_depth,expected_direction',[
    ([0,0,.15],.1,[0,0,1]),([0,0,-.15],.1,[0,0,-1]),
    ([.49,0,0],.06,[1,0,0]),
])
def test_deep_axis_intersection_uses_exact_nearest_exit_including_finite_edges(center,expected_depth,expected_direction):
    from omniretargeting.triangle_capsule import intersecting_capsule_exit
    # This finite triangle has a vertical edge at x=.5; the near-edge case
    # exits sideways, not along an arbitrarily chosen face normal.
    tri=np.array([[-.5,-.5,0],[.5,-.5,0],[.5,.5,0]])
    center=np.asarray(center);start=center-[0,0,.2];end=center+[0,0,.2]
    # For the first two cases move inside the triangle, away from diagonal edge.
    if center[0]==0:
        start[0]=end[0]=.2
    depth,direction=intersecting_capsule_exit(start,end,.05,tri)
    assert depth==pytest.approx(expected_depth)
    np.testing.assert_allclose(direction,expected_direction,atol=1e-12)
    for fraction in [.999,1.001]:
        offset=fraction*depth*direction
        p,s=segment_triangle_witnesses(start+offset,end+offset,tri[None])
        distance=float(np.linalg.norm(p-s))
        assert (distance>.05)==(fraction>1)


def test_open_triangle_does_not_extend_into_an_infinite_obstacle():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body pos="2 0 0"><freejoint/>
      <geom type="sphere" size=".05"/></body></worldbody></mujoco>''')
    scene = trimesh.Trimesh([[-.1,-.1,0], [.1,-.1,0], [0,.1,0]], [[0,1,2]], process=False)
    query = RigidGeometryQuery(model, [0], SceneQuery(scene))
    assert not query.query(_data(model), .1)


def test_capsule_surface_jacobian_matches_nv_finite_difference():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body pos="0 0 .3"><freejoint/>
      <inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>
      <body><joint axis="0 1 0"/><geom type="capsule" pos=".2 0 0" size=".05 .2"/>
      </body></body></worldbody></mujoco>''')
    scene = trimesh.Trimesh([[-2,-2,0], [2,-2,0], [0,2,0]], [[0,1,2]], process=False)
    query = RigidGeometryQuery(model, [0], SceneQuery(scene))
    data = _data(model)
    row = query.query(data, .2)[0]
    jac = relative_point_jacobian(model, data, row["points"], row["body_ids"], row["directions"])
    assert jac.shape == (model.nv,) and model.nq != model.nv
    for dof in range(model.nv):
        values = []
        for sign in [-1, 1]:
            data.qpos[:] = model.qpos0
            velocity = np.zeros(model.nv); velocity[dof] = sign
            mujoco.mj_integratePos(model, data.qpos, velocity, 1e-6)
            mujoco.mj_forward(model, data)
            values.append(query.query(data, .2)[0]["clearance"])
        assert jac[dof] == pytest.approx((values[1] - values[0]) / 2e-6, abs=1e-7)


def test_degenerate_triangle_is_its_finite_edges_not_nan():
    p, s = segment_triangle_witnesses([0,0,0], [0,0,1], [[[.2,0,.5], [.2,0,.6], [.2,0,.7]]])
    np.testing.assert_allclose(np.linalg.norm(p-s, axis=1), [.2])


def test_finite_segment_to_triangle_interior_and_edge_truth():
    tri = np.array([[[-1,-1,0], [1,-1,0], [0,1,0]]])
    for a,b,d in [([0,0,-1],[0,0,1],0), ([0,0,.1],[0,.3,.1],.1),
                  ([3,-1,0],[3,1,0],2)]:
        p,s = segment_triangle_witnesses(a,b,tri)
        np.testing.assert_allclose(np.linalg.norm(p-s,axis=1), [d])
