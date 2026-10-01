import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from omniretargeting.mesh_inertia import segment_inertias


def test_open_anatomical_cut_is_capped_with_analytic_mass_properties():
    mesh = trimesh.creation.box(extents=[1, 2, 3])
    faces = mesh.faces[mesh.face_normals[:, 2] != 1]
    vertices = mesh.vertices.copy()
    result = segment_inertias(vertices, faces, np.zeros(len(faces)), total_mass=5)
    np.testing.assert_allclose(result["centers"], [[0, 0, 0]], atol=1e-12)
    np.testing.assert_allclose(result["volumes"], [6])
    np.testing.assert_allclose(result["inertias"][0], np.diag([13, 10, 5]) * 5 / 12)
    np.testing.assert_array_equal(vertices, mesh.vertices)


def test_covariance_and_relative_masses_follow_source_geometry():
    small = trimesh.creation.box(extents=[1, 1, 1])
    large = trimesh.creation.box(extents=[2, 2, 2])
    large.apply_translation([4, 0, 0])
    mesh = trimesh.util.concatenate([small, large])
    labels = np.repeat([0, 1], 12)
    result = segment_inertias(mesh.vertices, mesh.faces, labels)
    np.testing.assert_allclose(result["masses"], [1 / 9, 8 / 9])
    rotation = Rotation.from_rotvec([0.2, -0.3, 0.5]).as_matrix()
    shift = [0.5, -2, 1]
    moved = segment_inertias(mesh.vertices @ rotation.T + shift, mesh.faces, labels)
    np.testing.assert_allclose(
        moved["centers"], result["centers"] @ rotation.T + shift, atol=1e-12
    )
    np.testing.assert_allclose(
        moved["inertias"], rotation @ result["inertias"] @ rotation.T, atol=1e-12
    )


@pytest.mark.parametrize("mass", [0, -1, np.inf, np.nan])
def test_invalid_mass_is_not_silently_inferred(mass):
    mesh = trimesh.creation.box()
    with pytest.raises(ValueError):
        segment_inertias(
            mesh.vertices, mesh.faces, np.zeros(len(mesh.faces)), total_mass=mass
        )
