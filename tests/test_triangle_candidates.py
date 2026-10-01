"""Compare conservative capsule pruning against every original triangle."""

from itertools import pairwise

import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting.triangle_candidates import TriangleCandidates
from omniretargeting.triangle_capsule import (
    intersecting_capsule_exit,
    segment_triangle_witnesses,
)


def exhaustive_ids(mesh, start, end, band):
    bounds = np.r_[np.minimum(start, end) - band, np.maximum(start, end) + band]
    return np.asarray(sorted(mesh.triangles_tree.intersection(bounds)), int)


def witness(triangles, ids, start, end, radius):
    axis, surface = segment_triangle_witnesses(start, end, triangles[ids])
    distances = np.linalg.norm(axis - surface, axis=1)
    clearance = distances - radius
    exits = {}
    for hit in np.flatnonzero(distances <= 1e-10):
        depth, direction = intersecting_capsule_exit(start, end, radius, triangles[ids[hit]])
        clearance[hit] = -depth
        exits[hit] = direction
    k = int(np.argmin(clearance))
    return ids[k], clearance[k], axis[k], surface[k], exits.get(k)


@pytest.mark.parametrize("sphere", [False, True])
def test_random_witnesses_match_exhaustive_faces(sphere):
    rng = np.random.default_rng(420)
    mesh = trimesh.creation.icosphere(subdivisions=3)
    # Include degenerate and duplicate faces and separate intersecting sheets.
    vertices = np.r_[mesh.vertices, [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]]
    i = len(mesh.vertices)
    faces = np.r_[mesh.faces, mesh.faces[:3], [[i, i, i], [i, i+1, i+1],
                                            [i, i+1, i+2], [i, i+2, i+3]]]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    candidates = TriangleCandidates(mesh)
    for _ in range(80):
        start = rng.uniform(-1.3, 1.3, 3)
        end = start.copy() if sphere else start + rng.uniform(-.6, .6, 3)
        radius = rng.uniform(.01, .4)
        all_ids = exhaustive_ids(mesh, start, end, radius + .1)
        ids = candidates.query(start, end, radius + .1)
        if not len(all_ids):
            assert not len(ids)
            continue
        expected = witness(candidates.triangles, all_ids, start, end, radius)
        actual = witness(candidates.triangles, ids, start, end, radius)
        for old, new in zip(expected, actual):
            if old is None:
                assert new is None
            else:
                np.testing.assert_array_equal(new, old)
    assert candidates.retained_faces < candidates.candidate_faces


def test_intersections_ties_and_numerical_axis_hits_survive():
    triangles = np.array([
        [[-1, -1, 0], [1, -1, 0], [0, 1, 0]],
        [[0, -1, -1], [0, 1, -1], [0, 0, 1]],
        [[-1, -1, 8e-11], [1, -1, 8e-11], [0, 1, 8e-11]],
    ])
    triangles = np.concatenate([triangles, triangles])
    mesh = trimesh.Trimesh(triangles.reshape(-1, 3), np.arange(18).reshape(-1, 3), process=False)
    candidates = TriangleCandidates(mesh)
    for end in (np.zeros(3), np.array([.4, .2, .3])):
        ids = candidates.query(np.zeros(3), end, .1)
        np.testing.assert_array_equal(ids, np.arange(6))


def test_dense_plane_prunes_work_without_changing_the_winning_face():
    grid = np.linspace(-1, 1, 81)
    x, y = np.meshgrid(grid, grid)
    vertices = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
    faces = []
    for row in range(80):
        for col in range(80):
            a = row * 81 + col
            faces.extend([[a, a + 1, a + 81], [a + 1, a + 82, a + 81]])
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    candidates = TriangleCandidates(mesh)
    start = np.array([.001, .002, .01])
    end = start + [0, 0, .15]
    all_ids = exhaustive_ids(mesh, start, end, .2)
    ids = candidates.query(start, end, .2)
    assert len(ids) < len(all_ids) / 20
    for actual, expected in zip(witness(candidates.triangles, ids, start, end, .05),
                                witness(candidates.triangles, all_ids, start, end, .05)):
        if expected is not None:
            np.testing.assert_array_equal(actual, expected)


def test_batched_axes_preserve_independent_sphere_capsule_and_degenerate_queries():
    rng = np.random.default_rng(87)
    triangles = rng.normal(size=(150, 3, 3))
    triangles[::5, 1] = triangles[::5, 0]
    triangles[::7, 2] = triangles[::7, 0]
    starts = rng.normal(size=(150, 3))
    ends = starts + rng.normal(size=(150, 3))
    ends[::3] = starts[::3]
    ends[1::3] = starts[1::3] + 1e-14
    axis, surface = segment_triangle_witnesses(starts, ends, triangles)
    for index in range(len(triangles)):
        expected = segment_triangle_witnesses(starts[index], ends[index], triangles[index:index+1])
        np.testing.assert_allclose(axis[index], expected[0][0], atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(surface[index], expected[1][0], atol=1e-13, rtol=1e-13)


def test_grouped_dot_products_keep_witnesses_bitwise_equal():
    rng = np.random.default_rng(24)
    starts, ends, triangles, expected = [], [], [], []
    for index in range(20):
        start, end = rng.normal(size=(2, 3))
        if index % 3 == 0:
            end = start.copy()
        faces = rng.normal(size=(index + 2, 3, 3))
        faces[0] = faces[-1]  # tied candidates keep their original ordering
        expected.append(segment_triangle_witnesses(start, end, faces))
        triangles.append(faces)
        starts.append(np.broadcast_to(start, (len(faces), 3)))
        ends.append(np.broadcast_to(end, (len(faces), 3)))
    actual = segment_triangle_witnesses(
        np.concatenate(starts), np.concatenate(ends), np.concatenate(triangles),
        groups=np.cumsum([0, *map(len, triangles)]),
    )
    for side in range(2):
        np.testing.assert_array_equal(actual[side], np.concatenate([row[side] for row in expected]))


@pytest.mark.parametrize("active", [None, [0, 2]])
def test_complete_rigid_rows_match_unpruned_individual_axis_queries(monkeypatch, active):
    from omniretargeting import triangle_capsule
    from omniretargeting.geometry_query import RigidGeometryQuery
    from omniretargeting.scene_geometry import SceneQuery

    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body><freejoint/>
      <geom type="capsule" size=".05 .2"/>
      <geom type="sphere" size=".08" pos=".2 .03 .1"/>
      <geom type="capsule" size=".035 .12" pos="-.12 0 .05" euler="25 15 10"/>
      </body></worldbody></mujoco>''')
    shell = trimesh.creation.icosphere(subdivisions=2, radius=.3)
    shell.update_faces(np.arange(len(shell.faces) - 1))
    mesh = trimesh.util.concatenate([
        shell,
        trimesh.Trimesh([[-1, -1, -.2], [1, -1, -.2], [0, 1, -.2]], [[0, 1, 2]]),
    ])
    query = RigidGeometryQuery(model, [0, 1, 2], SceneQuery(mesh))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(98)

    def all_candidates(self, start, end, band):
        bounds = np.r_[np.minimum(start, end) - band, np.maximum(start, end) + band]
        return np.asarray(sorted(self.tree.intersection(bounds)), int)

    def individual_axes(start, end, triangles, *, groups=None):
        if groups is None:
            return segment_triangle_witnesses(start, end, triangles)
        rows = [segment_triangle_witnesses(start[a], end[a], triangles[a:b])
                for a, b in pairwise(groups)]
        return tuple(np.concatenate([row[side] for row in rows]) for side in (0, 1))

    compared = 0
    for _ in range(20):
        data.qpos[:3] = rng.uniform(-.35, .35, 3)
        quat = rng.normal(size=4)
        data.qpos[3:7] = quat / np.linalg.norm(quat)
        mujoco.mj_forward(model, data)
        actual = query.query(data, .12, active_geom_ids=active)
        with monkeypatch.context() as legacy:
            legacy.setattr(TriangleCandidates, "query", all_candidates)
            legacy.setattr(triangle_capsule, "segment_triangle_witnesses", individual_axes)
            expected = query.query(data, .12, active_geom_ids=active)
        assert len(actual) == len(expected)
        for after, before in zip(actual, expected):
            assert after.keys() == before.keys()
            for key in before:
                np.testing.assert_array_equal(after[key], before[key], err_msg=key)
            compared += 1
    assert compared > 30
