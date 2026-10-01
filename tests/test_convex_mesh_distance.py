"""Physical convex-mesh exits do not depend on detection margins."""
import mujoco
import numpy as np
import pytest

from omniretargeting.convex_mesh_distance import ConvexMeshDistance
from omniretargeting.self_collision import SelfCollisionQuery


def mesh_pair():
    vertices = ' '.join(map(str, np.array(np.meshgrid(*[[-.1, .1]] * 3)).reshape(3, -1).T.ravel()))
    return mujoco.MjModel.from_xml_string(f'''<mujoco><asset>
      <mesh name="cube" vertex="{vertices}"/></asset><worldbody>
      <body name="a"><freejoint/><geom type="mesh" mesh="cube"/></body>
      <body name="b" pos=".15 0 0"><freejoint/><geom type="mesh" mesh="cube"/></body>
      </worldbody></mujoco>''')


@pytest.mark.parametrize('shift', [.15, .05, 0., .23])
def test_full_exit_matches_boxes_including_containment_and_coplanar_facets(shift):
    model = mesh_pair()
    data = mujoco.MjData(model)
    data.qpos[7] = shift
    mujoco.mj_forward(model, data)
    query = ConvexMeshDistance(model)
    phi, points, direction = query.query(0, 1, data)
    assert phi == pytest.approx(shift - .2, abs=1e-8)
    assert np.linalg.norm(direction) == pytest.approx(1.)
    if phi < 0:
        np.testing.assert_allclose(points[0] - points[1], phi * direction, atol=1e-8)
        data.qpos[:3] += (-phi + 1e-5) * direction
        mujoco.mj_forward(model, data)
        assert query.query(0, 1, data)[0] >= 0


def test_mesh_distance_and_jacobian_are_margin_independent():
    model = mesh_pair()
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD)
    query = SelfCollisionQuery(model)
    original = model.opt.disableflags
    q = model.qpos0.copy()
    values = [query.query(q, distance=cutoff)[0]['clearance'] for cutoff in (.01, .03, .2)]
    np.testing.assert_allclose(values, -.05, atol=1e-8)
    row = query.query(q)[0]
    jacobian = query.jacobian(row, query.data)
    samples = []
    for sign in (-1, 1):
        moved = q.copy()
        moved[0] += sign * 1e-6
        samples.append(query.query(moved)[0]['clearance'])
    assert (samples[1] - samples[0]) / 2e-6 == pytest.approx(jacobian[0], abs=1e-8)
    assert model.opt.disableflags == original


def test_rotated_mesh_exit_witness_has_correct_relative_pose_derivative():
    model = mesh_pair()
    query = SelfCollisionQuery(model)
    q = model.qpos0.copy()
    q[7:10] = [.16, .035, .022]
    q[10:14] = [np.cos(.17), 0., 0., np.sin(.17)]
    row = query.query(q)[0]
    assert row['clearance'] < 0
    jacobian = query.jacobian(row, query.data)
    for column in (0, 1, 5, 6, 7, 11):
        samples = []
        for sign in (-1, 1):
            moved = q.copy()
            tangent = np.zeros(model.nv)
            tangent[column] = sign * 1e-5
            mujoco.mj_integratePos(model, moved, tangent, 1.)
            samples.append(query.query(moved)[0]['clearance'])
        assert (samples[1] - samples[0]) / 2e-5 == pytest.approx(jacobian[column], abs=2e-6)


def test_unclosed_mesh_bound_is_an_error_not_a_collision_free_answer(monkeypatch):
    import omniretargeting.convex_mesh_distance as mesh
    from omniretargeting.geometry_query import GeometryQueryError
    model = mesh_pair()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    monkeypatch.setattr(mesh, 'MAX_ITERATIONS', 0)
    with pytest.raises(GeometryQueryError, match='iteration limit'):
        mesh.ConvexMeshDistance(model).query(0, 1, data)
