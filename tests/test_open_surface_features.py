"""Grouped/batched crease handling equals the per-feature reference."""

import numpy as np
import pytest
import trimesh
from scipy.spatial import cKDTree

from omniretargeting.open_surface_features import OpenSurfaceFeatures


def _reference(features, points, nearest, face_ids, reliable, tree, evidence, normals_e, radius):
    """Per-feature loop of the original implementation."""
    reliable = np.array(reliable, bool, copy=True)
    candidates = np.flatnonzero(reliable)
    if not len(candidates):
        return reliable
    triangles = features.triangles[face_ids[candidates]]
    bary = trimesh.triangles.points_to_barycentric(triangles, nearest[candidates])
    rows_ = np.flatnonzero(np.any(bary <= 1e-7, axis=1))
    members = candidates[rows_]
    keys = np.sort(np.where(bary[rows_] > 1e-7, features.faces[face_ids[members]], -1), axis=1)
    degenerate = keys[:, -1] < 0
    reliable[members[degenerate]] = False
    members, keys = members[~degenerate], keys[~degenerate]
    unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    for index, feature in enumerate(unique):
        rows = members[inverse == index]
        faces = features._incident_faces(tuple(int(v) for v in feature if v >= 0))
        normals = features.face_normals[faces]
        offset = points[rows] - nearest[rows]
        coplanar = np.all(np.abs(normals @ normals[0]) >= 1 - 1e-6)
        if len(faces) == 1 or coplanar:
            tangent = offset - (offset @ normals[0])[:, None] * normals[0]
            reliable[rows] &= np.linalg.norm(tangent, axis=1) <= 1e-7 * (1 + np.linalg.norm(offset, axis=1))
            continue
        distances, indices = tree.query(nearest[rows], k=min(4, len(evidence)),
                                        distance_upper_bound=radius * (1 + 1e-9) + 1e-12)
        distances = np.asarray(distances).reshape(len(rows), -1)
        indices = np.minimum(np.asarray(indices).reshape(len(rows), -1), len(evidence) - 1)
        alignment = np.einsum("nkj,fj->nkf", normals_e[indices], normals)
        eligible = (distances[..., None] <= radius) & (np.abs(alignment) >= 0.8)
        count = eligible.sum(axis=1)
        vote = np.where(eligible, np.sign(alignment), 0).sum(axis=1)
        supported = (count > 0) & (np.abs(vote) == count)
        oriented = normals[None] * np.where(vote < 0, -1, 1)[..., None]
        projections = np.einsum("nj,nfj->nf", offset, oriented)
        consistent = np.all(projections >= -1e-8, axis=1) | np.all(projections <= 1e-8, axis=1)
        reliable[rows] &= supported.all(axis=1) & consistent
    return reliable


def _folded_sheet():
    """An open, creased height field: interior seams, external edges, corners."""
    x, y = np.meshgrid(np.linspace(0, 1, 9), np.linspace(0, 1, 7), indexing="ij")
    z = 0.3 * np.abs(x - 0.5) + 0.1 * (y > 0.5) * y
    vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    faces = []
    for i in range(8):
        for j in range(6):
            a, b, c, d = i * 7 + j, (i + 1) * 7 + j, (i + 1) * 7 + j + 1, i * 7 + j + 1
            faces += [(a, b, c), (a, c, d)]
    return trimesh.Trimesh(vertices, np.array(faces), process=False)


@pytest.mark.parametrize("seed", range(4))
def test_reliable_witnesses_equal_the_reference(seed):
    mesh = _folded_sheet()
    rng = np.random.default_rng(seed)
    points = rng.uniform([-0.2, -0.2, -0.3], [1.2, 1.2, 0.6], size=(3000, 3))
    closest, _, face_ids = trimesh.proximity.closest_point(mesh, points)
    # Snap some witnesses exactly onto vertices and edges (features).
    snap = rng.random(len(points)) < 0.3
    closest[snap] = mesh.vertices[mesh.faces[face_ids[snap], rng.integers(0, 3, snap.sum())]]
    evidence = mesh.vertices[rng.integers(0, len(mesh.vertices), 40)] + rng.normal(scale=0.01, size=(40, 3))
    evidence_normals = rng.normal(size=(40, 3))
    evidence_normals /= np.linalg.norm(evidence_normals, axis=1, keepdims=True)
    tree = cKDTree(evidence)
    reliable = rng.random(len(points)) < 0.8
    features = OpenSurfaceFeatures(mesh)
    expected = _reference(OpenSurfaceFeatures(mesh), points, closest, face_ids, reliable, tree,
                          evidence, evidence_normals, 0.25)
    actual = features.reliable_witnesses(points, closest, face_ids, reliable, tree, evidence,
                                         evidence_normals, 0.25)
    assert expected.sum() and (~expected & reliable).sum()
    np.testing.assert_array_equal(actual, expected)
