"""Validation-only regressions for open scene crease evidence.

These tests cover the geometric contract without using benchmark assets or
changing production scene/source data.  An open floor/wall crease has no
unique signed normal, while coplanar triangulation seams do.
"""

import numpy as np
import pytest
import trimesh

from omniretargeting.scene_geometry import SceneQuery


def floor_wall_crease():
    # Floor faces point up.  Wall faces point toward +X.  The floor/wall edge
    # is shared by both surfaces, so a point in the outside lower-right
    # quadrant has equal distance to perpendicular faces with opposite signed
    # projections.
    vertices = np.array([
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [1, 5, 4], [1, 2, 5],
    ])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def closed_extrusion(points, height):
    """Create a watertight prism from a counter-clockwise 2-D polygon."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    vertices = np.vstack([
        np.column_stack([points, np.zeros(n)]),
        np.column_stack([points, np.full(n, height)]),
    ])
    bottom = np.array([[0, i + 1, i] for i in range(1, n - 1)])
    top = np.array([[n, n + i, n + i + 1] for i in range(1, n - 1)])
    sides = []
    for i in range(n):
        j = (i + 1) % n
        sides.extend([[i, j, n + j], [i, n + j, n + i]])
    return trimesh.Trimesh(
        vertices=vertices,
        faces=np.vstack([bottom, top, np.asarray(sides)]),
        process=False,
    )


def test_perpendicular_open_crease_is_unknown_under_rounding_and_face_order():
    point = np.array([1.5, 0.0, -0.5])
    for order in (
        np.arange(4),
        np.array([2, 0, 3, 1]),
        np.array([1, 3, 0, 2]),
    ):
        mesh = floor_wall_crease()
        mesh.faces = mesh.faces[order]
        query = SceneQuery(mesh, [[0.0, 0.0, 0.0]], [[0.0, 0.0, 1.0]], evidence_radius=0.2)
        for rounded in (point, np.round(point, 8), point + [1e-10, -1e-10, -1e-10]):
            result = query.query(rounded[None])
            assert result["uncertain"][0]
            assert np.isnan(result["clearance"][0])
            assert result["unsigned_distance"][0] == pytest.approx(np.sqrt(0.5), abs=1e-7)


def test_coplanar_internal_triangulation_seam_keeps_reliable_signed_side():
    mesh = trimesh.Trimesh(
        vertices=[[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    query = SceneQuery(mesh, [[0, 0, 0]], [[0, 0, 1]], evidence_radius=0.15)
    result = query.query([[0, 0, 0.2]])
    assert not result["uncertain"][0]
    assert result["clearance"][0] == pytest.approx(0.2)
    np.testing.assert_allclose(result["directions"][0], [0, 0, 1], atol=1e-7)


def test_closed_corner_keeps_negative_containment_and_full_exit_gradient():
    # A closed L-shaped extrusion supplies a genuine re-entrant corner.  A
    # point in that corner is inside the solid, and its closest witness is the
    # corner itself rather than one arbitrarily selected boundary face.
    mesh = closed_extrusion(
        [(-1, -1), (1, -1), (1, 0), (0, 0), (0, 1), (-1, 1)], 2.0
    )
    query = SceneQuery(mesh)
    inside = query.query([[-0.1, -0.1, 0.2]])
    assert inside["clearance"][0] < 0
    assert not inside["uncertain"][0]

    # The nearest re-entrant corner is itself inside the closed solid, so the
    # negative clearance and exit direction must retain both components.
    expected = np.array([0.1, 0.1, 0.0])
    expected /= np.linalg.norm(expected)
    assert inside["clearance"][0] == pytest.approx(-np.linalg.norm([0.1, 0.1]))
    assert not inside["uncertain"][0]
    np.testing.assert_allclose(inside["directions"][0], expected, atol=1e-6)


def test_distant_evidence_backed_interior_normal_is_not_dropped():
    # This point is two metres behind an evidence-backed open wall.  Its
    # signed value is still meaningful locally; distance alone cannot turn it
    # into an unexamined free-space result.
    mesh = trimesh.Trimesh(
        vertices=[[0, -2, -2], [0, 2, -2], [0, 2, 2], [0, -2, 2]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    query = SceneQuery(mesh, [[0, 0, 0]], [[1, 0, 0]], evidence_radius=0.15)
    result = query.query([[-2.0, 0.0, 0.0]])
    assert not result["uncertain"][0]
    assert result["clearance"][0] == pytest.approx(-2.0)
    np.testing.assert_allclose(result["directions"][0], [1, 0, 0], atol=1e-7)


@pytest.mark.parametrize('size', [.001, 1.])
@pytest.mark.parametrize('reverse_faces', [False, True])
def test_boundary_vertex_shared_by_triangles_is_not_an_infinite_patch(size, reverse_faces):
    from scipy.spatial.transform import Rotation

    mesh = trimesh.Trimesh(
        vertices=np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]) * size,
        faces=[[0, 1, 2], [0, 2, 3]], process=False,
    )
    rotation = Rotation.from_rotvec([.4, -.2, .7]).as_matrix()
    mesh.vertices = np.asarray(mesh.vertices) @ rotation.T
    if reverse_faces:
        mesh.faces = mesh.faces[::-1, ::-1]
    corner = np.array([size, size, 0.]) @ rotation.T
    query = SceneQuery(mesh, [corner], [rotation[:, 2]], evidence_radius=3.)
    points = np.array([[size+.04, size+.04, -.1], [0., 0., -.1]]) @ rotation.T
    result = query.query(points)
    assert result['uncertain'][0]
    assert np.isnan(result['clearance'][0])
    assert not result['uncertain'][1]
    assert result['clearance'][1] == pytest.approx(-.1)


def test_finite_crease_endpoint_does_not_extrude_past_its_boundary():
    query = SceneQuery(floor_wall_crease(), [[1, -1, 0], [1, -1, 0]],
                       [[0, 0, 1], [1, 0, 0]], evidence_radius=2.)
    result = query.query([[.9, -1.1, -.1]])
    assert result['uncertain'][0]
    assert np.isnan(result['clearance'][0])
