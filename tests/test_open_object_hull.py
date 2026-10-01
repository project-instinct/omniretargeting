"""A recorded object lies within the convex hull of its vertices.

Free-side evidence of an open patch extends the patch's local side only near
it; beyond the object's hull no closure of the recorded surface can contain a
point, whatever that evidence votes.
"""

import numpy as np
import pytest
import trimesh

from omniretargeting._fast.sdf import nearest_component_query
from omniretargeting.scene_geometry import SceneQuery

# Two open quads meeting at the y axis: a V whose concave side is observed free.
V_SHEET = trimesh.Trimesh(
    vertices=[[0, -1, 0], [0, 1, 0], [1, 1, 1], [1, -1, 1], [-1, 1, 1], [-1, -1, 1]],
    faces=[[0, 1, 2], [0, 2, 3], [0, 4, 1], [0, 5, 4]],
    process=False,
)
EVIDENCE = dict(points=[[0.5, 0, 0.5], [-0.5, 0, 0.5]],
                normals=np.array([[-1, 0, 1], [1, 0, 1]]) / np.sqrt(2), radius=1.0)
BEHIND_APEX = [[0, 0, -0.5]]
IN_OPENING = [[0, 0, 0.5]]


def _query(**kwargs):
    return SceneQuery(V_SHEET, EVIDENCE["points"], EVIDENCE["normals"], evidence_radius=EVIDENCE["radius"],
                      **kwargs)


def test_evidence_alone_extends_the_patch_side_beyond_the_object():
    result = _query().query(BEHIND_APEX)
    assert not result["uncertain"][0]
    assert result["clearance"][0] == pytest.approx(-0.5)


def test_a_point_outside_the_hull_is_outside_the_object():
    query = _query(bounded_object=True)
    result = query.query(BEHIND_APEX)
    assert not result["uncertain"][0]
    assert result["clearance"][0] == pytest.approx(0.5)
    np.testing.assert_allclose(result["directions"][0], [0, 0, -1], atol=1e-9)
    # Inside the hull the evidence still decides.
    inside = query.query(IN_OPENING)
    assert inside["clearance"][0] == pytest.approx(_query().query(IN_OPENING)["clearance"][0])
    assert query.outside_hull(BEHIND_APEX + IN_OPENING).tolist() == [True, False]


def test_no_certificate_for_flat_sheets_or_enclosed_free_space():
    wall = trimesh.Trimesh(vertices=[[0, -2, -2], [0, 2, -2], [0, 2, 2], [0, -2, 2]],
                           faces=[[0, 1, 2], [0, 2, 3]], process=False)
    query = SceneQuery(wall, [[0, 0, 0]], [[1, 0, 0]], bounded_object=True)
    # A flat sheet has no interior hull: evidence keeps deciding.
    assert query.query([[-2, 0, 0]])["clearance"][0] == pytest.approx(-2)
    room = trimesh.creation.box(extents=[4, 4, 4])
    room.invert()
    enclosed = SceneQuery(room, enclosed_free_space=True, bounded_object=True)
    assert not enclosed.outside_hull([[5, 0, 0]])[0]
    assert enclosed.query([[5, 0, 0]])["clearance"][0] < 0


def test_object_grid_query_uses_the_same_certificate():
    run = nearest_component_query(V_SHEET, EVIDENCE)
    result = run(BEHIND_APEX + IN_OPENING)
    assert not result["uncertain"].any()
    assert result["clearance"][0] == pytest.approx(0.5)
    assert result["clearance"][1] > 0


def _sphere_model():
    import mujoco

    return mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="probe"><freejoint/><geom type="sphere" size="0.05"/></body>
      </worldbody></mujoco>""")


def test_moving_query_knows_samples_far_outside_the_hull_without_rows():
    from omniretargeting.moving_scene import MovingSceneQuery

    model = _sphere_model()
    obj = {"mesh": V_SHEET, "translations": np.zeros((1, 3)), "rotations": np.eye(3)[None],
           "collision_evidence": EVIDENCE}
    query = MovingSceneQuery(model, [obj], 1)
    pose = model.qpos0.copy()
    pose[:3] = [0, 0, -0.5]  # behind the apex: outside the hull, beyond the activation band
    far = query.query(pose, 0)
    assert not far["rows"] and far["forward_depth"].max() == 0
    assert far["open_known_samples"] and not far["open_penetrating_samples"]
    pose[:3] = [0, 0, 0.5]  # in the observed opening, inside the hull: evidence decides
    inside = query.query(pose, 0)
    assert inside["forward_depth"].max() == 0 and inside["open_known_samples"]


def _open_l_solid():
    """A double-sided L prism with a distant 1.6% hole: open, not nearly closed."""
    from shapely.geometry import Polygon

    solid = trimesh.creation.extrude_polygon(
        Polygon([(0, 0), (2, 0), (2, .5), (.5, .5), (.5, 2), (0, 2)]), 1.0).subdivide().subdivide().subdivide()
    centres = solid.triangles_center
    hole = np.flatnonzero((centres[:, 0] > 1.999) & (centres[:, 2] > .6))
    return trimesh.Trimesh(solid.vertices, np.delete(solid.faces, hole, axis=0), process=False)


def test_evidence_cannot_enclose_a_point_the_winding_number_leaves_free():
    mesh = _open_l_solid()
    # Evidence on the concave faces claims the corner's free side is material.
    points, normals = [[.5, 1., .5], [1., .5, .5]], [[-1, 0, 0], [0, -1, 0]]
    corner, material = [[1., 1., .5]], [[.25, 1., .5]]
    plain = SceneQuery(mesh, points, normals, evidence_radius=1.0)
    bounded = SceneQuery(mesh, points, normals, evidence_radius=1.0, bounded_object=True)
    assert not bounded.nearly_closed(0) and not bounded.outside_hull(corner)[0]
    assert plain.query(corner, component_ids=[0])["clearance"][0] == pytest.approx(-.5)
    assert bounded.query(corner, component_ids=[0])["clearance"][0] == pytest.approx(.5)
    # Inside the material the winding number is near 1: evidence still decides.
    assert bounded.query(material, component_ids=[0])["clearance"][0] == pytest.approx(-.25)


def test_bounding_box_prefilter_keeps_the_hull_decisions():
    query = _query(bounded_object=True)
    points = np.random.default_rng(2).uniform(-3, 3, size=(2000, 3))
    exact = query.hull_margin(points)
    for band in (0.0, 0.05, 0.5):
        fast = query.hull_margin(points, beyond=band)
        # Same inside/outside decision; the box distance only certifies more
        # points as far, and like the plane margin it never exceeds the distance.
        np.testing.assert_array_equal(fast > 0, exact > 0)
        assert np.all(fast[exact > band] > band)
        nearest = np.linalg.norm(points[:, None] - V_SHEET.vertices[None], axis=2).min(1)
        assert np.all(fast[fast > 0] <= nearest[fast > 0] + 1e-12)


def test_points_beyond_a_small_components_winding_grid_are_vetted_directly():
    mesh = _open_l_solid()
    bounded = SceneQuery(mesh, [[.5, 1., .5], [1., .5, .5]], [[-1, 0, 0], [0, -1, 0]],
                         evidence_radius=1.0, bounded_object=True)
    corner = np.array([[1., 1., .5]])
    dist = np.array([.5])
    enclosed = np.array([True])
    bounded._winding_free(0, corner, dist, enclosed)  # builds the grid
    lower, spacing, winding = bounded._winding_grids[0]
    # Shrink the grid so the corner lies beyond it: the direct evaluation decides.
    bounded._winding_grids[0] = (lower, spacing, winding[:4, :4, :4])
    assert bounded._winding_free(0, corner, dist, enclosed).tolist() == [0]
