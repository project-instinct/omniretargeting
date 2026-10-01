"""Nearly closed open components are signed by a decisive winding number."""

import numpy as np
import trimesh

from omniretargeting.scene_geometry import SceneQuery


def _cracked_box():
    """A finely subdivided unit box with one small triangle removed."""
    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    for _ in range(4):
        box = box.subdivide()
    centers = box.triangles_center
    # Remove the triangle closest to the +X face centre.
    crack = int(np.argmin(np.linalg.norm(centers - [0.5, 0.0, 0.0], axis=1)))
    keep = np.ones(len(box.faces), bool)
    keep[crack] = False
    mesh = trimesh.Trimesh(box.vertices, box.faces[keep], process=False)
    assert not mesh.is_watertight
    return mesh


def _query(mesh):
    # Free-side evidence on the -Z face only, facing outward.
    evidence = np.array([[0.0, 0.0, -0.52]])
    return SceneQuery(mesh, evidence, [[0.0, 0.0, -1.0]], evidence_radius=0.3)


def test_interior_points_of_a_cracked_solid_are_known_inside():
    query = _query(_cracked_box())
    assert not query.components[0][2]
    # Deep points whose nearest surface is an edge or corner (feature votes are
    # unstable there) and ordinary interior points.
    points = np.array([[0.3, 0.3, 0.3], [-0.35, 0.4, 0.2], [0.0, 0.0, 0.1], [0.44, -0.44, 0.44]])
    result = query.query(points)
    assert not result["uncertain"].any()
    np.testing.assert_array_less(result["clearance"], 0.0)
    outside = query.query(np.array([[0.0, 0.0, 0.9], [-0.9, 0.2, 0.0]]))
    assert not outside["uncertain"].any()
    assert np.all(outside["clearance"] > 0)


def test_points_at_the_crack_stay_undecided_by_winding():
    mesh = _cracked_box()
    query = _query(mesh)
    (center, radius, _), = query._completions[0][0]
    # At the missing triangle the side depends on how the crack is closed.
    points = center + np.array([[-.3, 0, 0], [.3, 0, 0]]) * radius
    decided, _ = query._decisive_winding(0, points, np.full(2, 0.01))
    assert not decided.any()
    # Just beyond the crack's sphere, every closure agrees.
    far = center + np.array([[-3., 0, 0], [3., 0, 0]]) * radius
    decided, inside = query._decisive_winding(0, far, np.full(2, 0.01))
    assert decided.all() and inside.tolist() == [True, False]


def test_small_fragments_are_not_closed_by_winding():
    # Closing a lone 2 mm patch would add as much area as the patch itself.
    patch = trimesh.Trimesh(vertices=np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]) * 1e-3,
                            faces=[[0, 1, 2], [0, 2, 3]], process=False)
    query = SceneQuery(patch, [[0, 0, 1e-3]], [[0, 0, 1]], evidence_radius=3.)
    assert not query.nearly_closed(0)
    decided, _ = query._decisive_winding(0, np.array([[.05, .05, -.1]]), np.full(1, .1))
    assert not decided.any()


def test_a_true_open_sheet_is_not_signed_by_winding_near_it():
    sheet = trimesh.Trimesh(vertices=[[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
                            faces=[[0, 1, 2], [0, 2, 3]], process=False)
    query = SceneQuery(sheet, [[0, 0, 0.1]], [[0, 0, 1]], evidence_radius=0.3)
    points = np.array([[0.0, 0.0, 0.05], [0.0, 0.0, -0.05]])
    decided, _ = query._decisive_winding(0, points, np.full(2, 0.05))
    assert not decided.any()
    # The evidence answer is unchanged: free above, inside below.
    result = query.query(points)
    assert not result["uncertain"].any()
    assert result["clearance"][0] > 0 > result["clearance"][1]


def test_winding_only_fills_evidence_gaps():
    mesh = _cracked_box()
    # Evidence covering the top face answers there as before, even where the
    # winding number would decide; winding only decides where evidence cannot.
    query = SceneQuery(mesh, [[0.0, 0.0, 0.55]], [[0.0, 0.0, 1.0]], evidence_radius=0.3)
    point = np.array([[0.0, 0.0, 0.3]])
    before = query.query(point)
    query._completions[0] = None
    unchanged = query.query(point)
    assert not before["uncertain"].any() and not unchanged["uncertain"].any()
    np.testing.assert_array_equal(before["clearance"], unchanged["clearance"])


def test_winding_side_matches_the_query_where_evidence_is_absent():
    mesh = _cracked_box()
    query = SceneQuery(mesh, [[5.0, 5.0, 5.0]], [[0.0, 0.0, 1.0]], evidence_radius=0.1)
    points = np.array([[0.3, 0.3, 0.3], [0.0, 0.0, 0.9], [0.49, 0.0, 0.0]])
    full, side = query.query(points), query.winding_side(points, 0)
    np.testing.assert_array_equal(full["uncertain"], side["uncertain"])
    known = ~side["uncertain"]
    np.testing.assert_allclose(full["clearance"][known], side["clearance"][known])
    np.testing.assert_allclose(full["directions"][known], side["directions"][known])
