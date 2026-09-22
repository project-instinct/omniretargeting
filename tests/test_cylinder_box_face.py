"""Analytic face distances for the exact cylinder and finite box solids."""

import mujoco
import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from omniretargeting.geometry_query import RigidGeometryQuery, _cylinder_box_face_distance
from omniretargeting.scene_geometry import SceneQuery


@pytest.mark.parametrize("height", [.040144298, .02, -.13])
@pytest.mark.parametrize("angle", [0., .7])
def test_tilted_cylinder_face_distance_and_witness(height, angle):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <freejoint/><geom type="cylinder" size=".01 .093"/>
    </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    rotation = Rotation.from_rotvec(np.array([.4, -.2, .1]) * angle)
    cylinder = Rotation.from_quat([.56107763, -.41153334, -.43520171, -.57134199], scalar_first=True)
    terrain = trimesh.creation.box(extents=[10, 10, .1])
    terrain.apply_translation([0, 0, -.05])
    # The native exact-box representation is generated before rotating the
    # query scene, so the test also covers a face that is not world vertical.
    query = RigidGeometryQuery(model, [0], SceneQuery(terrain))
    scene = query.scene_ids[0]
    scene_body = query.model.geom_bodyid[scene]
    query.model.body_pos[scene_body] = rotation.apply([0, 0, -.05])
    query.model.body_quat[scene_body] = rotation.as_quat(scalar_first=True)
    data.qpos[:3] = rotation.apply([.41671326, .73094973, height])
    data.qpos[3:7] = (rotation * cylinder).as_quat(scalar_first=True)
    mujoco.mj_forward(model, data)
    extent = .093 * abs(cylinder.as_matrix()[2, 2]) + .01 * np.sqrt(1 - cylinder.as_matrix()[2, 2]**2)
    expected = max(height - extent, -height - extent - .1)
    rows = query.query(data, 1.)
    assert len(rows) == 1 and not rows[0]['uncertain']
    assert rows[0]['clearance'] == pytest.approx(expected, abs=1e-9)
    normal = rotation.apply([0, 0, 1 if height >= -.05 else -1])
    np.testing.assert_allclose(rows[0]['directions'], normal, atol=1e-9)
    np.testing.assert_allclose(rows[0]['points'] - rows[0]['surface_points'], expected * normal, atol=1e-9)
    assert query.analytic_face_queries == 1


@pytest.mark.parametrize("center,expected", [([.96, 0, 0], -.14), ([1.1, 0, 0], 0.)])
def test_shortest_exit_can_be_horizontal_and_exact_contact_has_normal(center, expected):
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <freejoint/><geom type="cylinder" size=".1 .2"/>
    </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.qpos[:3] = center
    mujoco.mj_forward(model, data)
    query = RigidGeometryQuery(model, [0], SceneQuery(trimesh.creation.box(extents=[2]*3)))
    row, = query.query(data, .5)
    assert row['clearance'] == pytest.approx(expected, abs=1e-9)
    np.testing.assert_allclose(row['directions'], [1, 0, 0], atol=1e-9)


def test_corner_overlap_requires_general_geometry_query():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <geom type="box" size="1 1 1"/>
      <body pos=".95 .95 0"><geom type="cylinder" size=".1 .2"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert _cylinder_box_face_distance(model, data, 1, 0) is None
