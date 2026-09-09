"""Compare spatial pruning with exhaustive queries of the same triangle trees."""

import numpy as np
import pytest
import trimesh

from omniretargeting.scene_geometry import SceneQuery


def wall(x, y=0.0, extent=1.0):
    return trimesh.Trimesh(
        vertices=[[x, y - extent, -extent], [x, y + extent, -extent],
                  [x, y + extent, extent], [x, y - extent, extent]],
        faces=[[0, 1, 2], [0, 2, 3]], process=False,
    )


def exhaustive(query):
    # Force every component through the original all-points narrow phase. This
    # does not add evidence: the actual per-point evidence/radius/normal voting
    # still determines whether an open component has a signed distance.
    query._component_query_data = [
        (vertices, faces, bounds, True)
        for vertices, faces, bounds, _ in query._component_query_data
    ]
    return query


def assert_same(actual, reference):
    assert actual.keys() == reference.keys()
    for key in reference:
        np.testing.assert_array_equal(actual[key], reference[key], err_msg=key)


@pytest.mark.parametrize("evidence", ["none", "consistent", "conflicting"])
def test_pruning_matches_exhaustive_for_disconnected_scans_and_solids(evidence):
    parts = [wall(0), wall(2)] + [wall(i, 10 + i) for i in range(3, 40)]
    box = trimesh.creation.box(extents=[8, 8, 8])
    box.apply_translation([20, 0, 0])
    parts.append(box)
    mesh = trimesh.util.concatenate(parts)
    points, normals = [], []
    if evidence != "none":
        points, normals = [[0, 0, 0], [0, 0.05, 0]], [[1, 0, 0], [1, 0, 0]]
    if evidence == "conflicting":
        normals[1] = [-1, 0, 0]
    fast = SceneQuery(mesh, points, normals)
    slow = exhaustive(SceneQuery(mesh, points, normals))
    rng = np.random.default_rng(721)
    samples = np.vstack([
        rng.uniform([-5, -1, -1], [40, 45, 1], (300, 3)),
        [[-10, 0, 0], [1, 0, 0], [0, 0, 0], [20, 0, 0], [100, 0, 0]],
    ])
    assert_same(fast.query(samples), slow.query(samples))
    # Queries range across the whole scan, not just one local cluster. Measure
    # actual narrow-phase work, not a machine-dependent timing threshold.
    assert fast.component_query_points < slow.component_query_points / 2


def test_unknown_witness_and_ties_keep_original_component_and_face_identity():
    mesh = trimesh.util.concatenate([wall(0), wall(0), wall(2), wall(20)])
    samples = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 1], [-100, 0, 0]])
    fast = SceneQuery(mesh)
    actual = fast.query(samples)
    assert_same(actual, exhaustive(SceneQuery(mesh)).query(samples))
    assert np.all(actual["uncertain"])
    assert np.all(np.isnan(actual["clearance"]))
    assert np.all(actual["surface_ids"] >= 0)
    assert np.all(actual["obstacle_ids"] == 0)


def test_near_unknown_surface_cannot_hide_far_signed_penetration():
    # The nearest unknown sheet is 1 cm away, but a distant evidence-backed
    # sheet still reports a 2 m penetration. A closed solid also remains signed.
    box = trimesh.creation.box(extents=[20, 20, 20])
    mesh = trimesh.util.concatenate([wall(1.99), wall(0), box])
    args = (mesh, [[0, 0, 0]], [[-1, 0, 0]])
    samples = [[2, 0, 0], [2, 0.1, 0]]
    fast = SceneQuery(*args)
    actual = fast.query(samples)
    assert_same(actual, exhaustive(SceneQuery(*args)).query(samples))
    np.testing.assert_allclose(actual["unsigned_distance"], 0.01)
    np.testing.assert_allclose(actual["clearance"], -8)
    assert np.all(actual["uncertain"])
    # The same pruning restriction applies without the enclosing solid.
    open_args = (trimesh.util.concatenate([wall(1.99), wall(0)]), *args[1:])
    open_result = SceneQuery(*open_args).query(samples)
    np.testing.assert_allclose(open_result["clearance"], -2)
    assert_same(open_result, exhaustive(SceneQuery(*open_args)).query(samples))


@pytest.mark.parametrize("distance", [0.15, 0.15 - 1e-12, 0.15 + 1e-12])
def test_evidence_radius_boundary_keeps_exact_eligibility(distance):
    args = (wall(0), [[distance, 0, 0]], [[1, 0, 0]])
    actual = SceneQuery(*args).query([[-2, 0, 0]])
    assert_same(actual, exhaustive(SceneQuery(*args)).query([[-2, 0, 0]]))
    assert actual["uncertain"][0] == (distance > 0.15)


def test_pruning_keeps_unknown_flag_even_when_all_unsigned_queries_are_skipped():
    box = trimesh.creation.box(extents=[2, 2, 2])
    mesh = trimesh.util.concatenate([box, wall(100)])
    query = SceneQuery(mesh)
    actual = query.query([[0.99, 0, 0]])
    assert query.component_query_calls == 1  # The far sheet is still unknown.
    assert actual["uncertain"][0]
    assert actual["confidence"][0] == 0
    assert_same(actual, exhaustive(SceneQuery(mesh)).query([[0.99, 0, 0]]))


def test_empty_batch_has_no_component_queries():
    mesh = trimesh.util.concatenate([wall(0), wall(100)])
    query = SceneQuery(mesh)
    assert_same(query.query([]), exhaustive(SceneQuery(mesh)).query([]))
    assert query.component_query_calls == query.component_query_points == 0
