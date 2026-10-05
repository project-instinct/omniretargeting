"""Batching original-scene queries preserves samples, witnesses and Jacobians."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting._fast.mesh_repair import OriginalGeometry
import omniretargeting._fast.original_rigid_fallback as original_rigid_fallback
from omniretargeting.scene_geometry import SceneQuery


class _RecordingQuery:
    def __init__(self, scene):
        self.scene = scene
        self.points = []

    def query(self, points):
        self.points.append(points.copy())
        return self.scene.query(points)


class _FramewiseGeometry(OriginalGeometry):
    """Reference: the original per-frame world transform and scene query."""

    def _static_frames(self, poses, frames=None):
        ids = self.objective.shell_sample_geom
        for frame in range(len(poses)) if frames is None else frames:
            pose = poses[frame]
            self.data.qpos[:] = pose
            mujoco.mj_forward(self.model, self.data)
            points = self.data.geom_xpos[ids] + np.einsum(
                "nij,nj->ni", self.data.geom_xmat[ids].reshape(-1, 3, 3),
                self.objective.shell_local)
            measured = self.scene.query(points)
            summaries = []
            for geom in self.objective.shell_geoms:
                selected = np.flatnonzero(ids == geom)
                value = measured["clearance"][selected]
                valid = np.isfinite(value) & ~measured["uncertain"][selected]
                row, clearance = None, np.inf
                if valid.any():
                    index = selected[np.argmin(np.where(valid, value, np.inf))]
                    clearance = float(measured["clearance"][index])
                    if clearance < .008:
                        row = {"clearance": clearance, "sample_id": int(index), "point": points[index].copy(),
                               "direction": measured["directions"][index].copy()}
                summaries.append((int(geom), {"row": row, "clearance": clearance,
                                              "unknown": not valid.all()}))
            yield summaries


def _problem():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="base" pos="0 0 .5"><freejoint/>
        <geom type="sphere" size=".04"/>
        <body name="left" pos=".15 0 0"><joint axis="0 1 0"/>
          <geom type="sphere" pos=".04 0 0" size=".06"/></body>
        <body name="right" pos=".25 .04 0"><joint axis="1 0 0"/>
          <geom type="sphere" pos=".02 0 0" size=".06"/></body>
      </body></worldbody></mujoco>''')
    poses = np.tile(model.qpos0, (8, 1))
    poses[:, 0] += np.arange(len(poses)) * .007
    poses[:, 7] = np.arange(len(poses)) * .04
    poses[:, 8] = np.arange(len(poses)) * -.03
    axes = np.concatenate([np.eye(3), -np.eye(3)])
    objective = SimpleNamespace(
        model=model, frames=len(poses),
        coords=SimpleNamespace(qi=np.array([7, 8]), di=np.array([6, 7])),
        shell_geoms=np.arange(model.ngeom),
        shell_sample_geom=np.repeat(np.arange(model.ngeom), len(axes)),
        shell_local=np.concatenate([axes * model.geom_size[g, 0]
                                    for g in range(model.ngeom)]))
    return objective, poses


def _scene(kind):
    box = trimesh.creation.box(extents=[.7, .7, 1.4])
    box.apply_translation([.1, 0, .6])
    sheet = trimesh.Trimesh(
        vertices=[[.2, -.3, .2], [.2, .3, .2], [.2, .3, .8], [.2, -.3, .8]],
        faces=[[0, 1, 2], [0, 2, 3]], process=False)
    if kind == "closed":
        return SceneQuery(box)
    if kind == "evidence":
        return SceneQuery(sheet, [[.2, 0, .5]], [[1, 0, 0]], evidence_radius=.09)
    return SceneQuery(trimesh.util.concatenate([box, sheet]))


def _assert_same(actual, expected):
    assert actual["depths"] == expected["depths"]
    assert actual["failures"] == expected["failures"]
    assert [key for key, _ in actual["rows"]] == [key for key, _ in expected["rows"]]
    for (_, row), (_, reference) in zip(actual["rows"], expected["rows"]):
        assert row.keys() == reference.keys()
        for name in row:
            np.testing.assert_array_equal(row[name], reference[name])


@pytest.mark.parametrize("kind", ["closed", "evidence", "mixed"])
@pytest.mark.parametrize("jacobians", [False, True])
def test_batches_keep_original_geometry_samples_results_and_frame_jacobians(kind, jacobians):
    objective, poses = _problem()
    original = poses.copy()
    serial_query, batch_query = _RecordingQuery(_scene(kind)), _RecordingQuery(_scene(kind))
    serial = _FramewiseGeometry(objective, [], serial_query)
    batched = OriginalGeometry(objective, [], batch_query)
    batched.SCENE_BATCH_POINTS = 3 * len(objective.shell_local) + 5
    expected = serial.measure(poses, jacobians=jacobians)
    actual = batched.measure(poses, jacobians=jacobians)
    _assert_same(actual, expected)
    np.testing.assert_array_equal(poses, original)
    np.testing.assert_array_equal(np.concatenate(batch_query.points), np.concatenate(serial_query.points))
    assert len(batch_query.points) == 3
    assert max(map(len, batch_query.points)) <= batched.SCENE_BATCH_POINTS
    assert any(key[1] == "self" for key, _ in actual["rows"])
    if kind == "closed":
        assert any(key[1] == "static" for key, _ in actual["rows"])
    if kind != "closed":
        assert actual["failures"]  # Unknown regions were never treated as free.
    for key, row in actual["rows"]:
        if key[1] == "static":
            assert row["point"].base is None
        assert row["direction"].base is None


def test_unsigned_static_samples_use_exact_rigid_fallback_without_zero_depth():
    objective, poses = _problem()
    geometry = OriginalGeometry(objective, [], _scene("mixed"))
    measured = geometry.measure(poses, jacobians=False)
    assert not measured["failures"]
    coverage = geometry.summary(measured)["query_coverage"]
    assert coverage["sample_unknown"] > 0
    assert coverage["rigid_resolved"] == coverage["sample_unknown"]
    assert coverage["rigid_unknown"] == 0


def test_uncertain_rigid_fallback_remains_unknown(monkeypatch):
    objective, poses = _problem()
    geometry = OriginalGeometry(objective, [], _scene("mixed"))

    def uncertain(*args, **kwargs):
        return [{
            "clearance": -0.01, "directions": np.zeros(3), "points": np.zeros(3),
            "surface_points": np.zeros(3), "geom_ids": 0, "body_ids": 0,
            "obstacle_ids": 0, "surface_ids": 0, "uncertain": True,
            "unsigned_distance": np.nan,
        }]

    monkeypatch.setattr(original_rigid_fallback, "_backend", lambda _geometry: SimpleNamespace(
        open_parts=[object()], failures=[], query=uncertain))
    monkeypatch.setattr(original_rigid_fallback, "uncovered_components", lambda *_: [])
    measured = geometry.measure(poses[:1], jacobians=False)
    assert measured["failures"]
    assert any("rigid backend" in failure and "rigid_ambiguous_exit" in failure
               for failure in measured["failures"])
    summary = geometry.summary(measured)
    assert summary["query_coverage"]["rigid_unknown"] > 0
    assert summary["query_coverage"]["rigid_ambiguous_exit"] == summary["query_coverage"]["rigid_unknown"]
    assert {row["category"] for row in summary["query_failure_examples"]} >= {
        "shell_sample_unknown_side", "rigid_ambiguous_exit"}


def test_signed_component_beside_open_scan_is_never_resolved_by_crossing_test():
    objective, poses = _problem()
    sheet = trimesh.Trimesh([[2, -.3, .2], [2, .3, .2], [2, .3, .8]], [[0, 1, 2]], process=False)
    # A closed, non-convex L block: its winding side is not a MuJoCo convex pair.
    block = trimesh.creation.box(extents=[.3, .1, .1]).union(
        trimesh.creation.box(extents=[.1, .3, .1], transform=trimesh.transformations.translation_matrix([.1, .1, 0])))
    block.apply_translation([3, 0, 0])
    geometry = OriginalGeometry(objective, [], SceneQuery(trimesh.util.concatenate([block, sheet])))
    measured = geometry.measure(poses[:1], jacobians=False)
    coverage = geometry.summary(measured)["query_coverage"]
    assert measured["failures"]
    assert coverage["rigid_resolved"] == 0
    assert coverage["rigid_uncovered_scene_components"] == coverage["sample_unknown"]


def test_unsupported_geom_type_on_open_scan_stays_unknown():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="base" pos="0 0 .5"><freejoint/><geom type="box" size=".04 .04 .04"/></body>
      </worldbody></mujoco>''')
    poses = model.qpos0[None].copy()
    objective = SimpleNamespace(
        model=model, frames=1, coords=SimpleNamespace(qi=np.array([], int), di=np.array([], int)),
        shell_geoms=np.arange(1), shell_sample_geom=np.zeros(6, int),
        shell_local=np.concatenate([np.eye(3), -np.eye(3)]) * .04)
    sheet = trimesh.Trimesh([[-1, -1, .45], [1, -1, .45], [0, 1, .45]], [[0, 1, 2]], process=False)
    geometry = OriginalGeometry(objective, [], SceneQuery(sheet))
    measured = geometry.measure(poses, jacobians=False)
    coverage = geometry.summary(measured)["query_coverage"]
    assert measured["failures"]
    assert coverage["rigid_unsupported_geom_type"] == 1


def test_single_dense_frame_still_respects_query_point_bound():
    objective, poses = _problem()
    serial = _FramewiseGeometry(objective, [], _scene("closed"))
    query = _RecordingQuery(_scene("closed"))
    batched = OriginalGeometry(objective, [], query)
    batched.SCENE_BATCH_POINTS = 5
    actual = batched.measure(poses, jacobians=True)
    _assert_same(actual, serial.measure(poses, jacobians=True))
    assert max(map(len, query.points)) <= 5
    assert sum(map(len, query.points)) == len(poses) * len(objective.shell_local)


def test_no_scene_does_not_require_or_allocate_static_samples():
    objective, poses = _problem()
    del objective.shell_local, objective.shell_sample_geom, objective.shell_geoms
    result = OriginalGeometry(objective, [], None).measure(poses, jacobians=True)
    assert result["rows"]
    assert all(key[1] == "self" for key, _ in result["rows"])


@pytest.mark.parametrize("kind", ["closed", "evidence", "mixed"])
def test_identical_transforms_reuse_witnesses_and_unknowns_but_recompute_jacobians(kind, monkeypatch):
    objective, poses = _problem()
    scene = _RecordingQuery(_scene(kind))
    geometry = OriginalGeometry(objective, [], scene)
    original_jacobian = mujoco.mj_jac
    calls = []

    def counted_jacobian(*args, **kwargs):
        calls.append(1)
        return original_jacobian(*args, **kwargs)

    monkeypatch.setattr(mujoco, "mj_jac", counted_jacobian)
    first = geometry.measure(poses, jacobians=True)
    query_count, jacobian_count = len(scene.points), len(calls)
    second = geometry.measure(poses, jacobians=True)
    _assert_same(second, first)
    assert len(scene.points) == query_count
    assert jacobian_count > 0
    assert len(calls) == 2 * jacobian_count
    assert len(geometry._static_cache) == len(poses) * objective.model.ngeom


def test_only_changed_geometry_is_requeried_and_cached_outputs_match_fresh_queries():
    objective, poses = _problem()
    scene = _RecordingQuery(_scene("closed"))
    geometry = OriginalGeometry(objective, [], scene)
    original = geometry.measure(poses, jacobians=True)
    scene.points.clear()
    changed = poses.copy()
    changed[4, 7] += .031
    actual = geometry.measure(changed, jacobians=True)
    expected = _FramewiseGeometry(objective, [], _scene("closed")).measure(changed, jacobians=True)
    _assert_same(actual, expected)
    left_geom = int(np.flatnonzero(objective.model.geom_bodyid == objective.model.body("left").id)[0])
    assert sum(map(len, scene.points)) == np.count_nonzero(objective.shell_sample_geom == left_geom)
    # Returned witness arrays cannot let a caller corrupt the immutable cache.
    static = next(row for key, row in actual["rows"] if key[1] == "static")
    static["point"][:] = 100.
    static["direction"][:] = 100.
    _assert_same(geometry.measure(changed, jacobians=True), expected)
    restored = geometry.measure(poses, jacobians=True)
    _assert_same(restored, original)


@pytest.mark.parametrize("kind", ["closed", "evidence", "mixed"])
@pytest.mark.parametrize("jacobians", [False, True])
def test_frame_parallel_measurement_returns_the_serial_result(kind, jacobians, monkeypatch):
    from omniretargeting._fast import parallel_geometry

    monkeypatch.setattr(parallel_geometry, "MIN_FRAMES_PER_WORKER", 2)
    monkeypatch.setattr(parallel_geometry, "WORKERS", 3)
    objective, poses = _problem()
    serial = OriginalGeometry(objective, [], _scene(kind))
    parallel = OriginalGeometry(objective, [], _scene(kind))
    parallel.start_workers(len(poses))
    try:
        assert parallel.parallel is not None and len(parallel.parallel.workers) == 3
        expected = serial.measure(poses, jacobians=jacobians)
        _assert_same(parallel.measure(poses, jacobians=jacobians), expected)
        # Accepted candidates are measured whole. A rejected one gets the serial
        # decision; its reported frame is a genuinely rejecting changed frame.
        cuts = []
        for delta in (.05, -.2):
            changed = poses.copy()
            changed[4:, 7] += delta
            changed[4:, 8] -= delta
            protected = dict(expected["depths"])
            cut = serial.measure(changed, jacobians=jacobians, protected_depths=protected)
            actual = parallel.measure(changed, jacobians=jacobians, protected_depths=protected)
            assert actual["complete"] == cut["complete"]
            if cut["complete"]:
                _assert_same(actual, cut)
            else:
                frame = actual["rejection_frame"]
                assert frame >= 4
                alone = serial.measure_frames(changed, [frame], jacobians=jacobians,
                                              protected_depths=protected)
                assert alone["rejection"] == frame
            cuts.append(cut.get("rejection_frame"))
        if kind == "closed":
            assert cuts == [None, 4]
    finally:
        parallel.stop_workers()
    assert parallel.parallel is None


def test_far_geoms_on_closed_scenes_reuse_exact_summaries():
    objective, poses = _problem()
    floor = trimesh.creation.box(extents=[4, 4, .1])
    floor.apply_translation([0, 0, -.05])
    geometry = OriginalGeometry(objective, [], SceneQuery(floor))
    assert geometry._static_bound_reuse
    geometry.measure(poses, jacobians=True)
    changed = poses.copy()
    changed[:, 7] += .01
    changed[:, 8] -= .02
    reference = _FramewiseGeometry(objective, [], SceneQuery(floor))
    _assert_same(geometry.measure(changed, jacobians=True), reference.measure(changed, jacobians=True))
    assert geometry.static_reused == len(poses) * 2  # both moved geoms, every frame
    # Next to the floor the displacement bound fails: geoms are queried again.
    low = changed.copy()
    low[:, 2] = .03
    reused = geometry.static_reused
    actual = geometry.measure(low, jacobians=True)
    _assert_same(actual, reference.measure(low, jacobians=True))
    assert geometry.static_reused == reused
    assert any(key[1] == "static" for key, _ in actual["rows"])


@pytest.mark.parametrize("kind", ["evidence", "mixed"])
def test_open_scene_components_are_always_requeried(kind):
    objective, _ = _problem()
    assert not OriginalGeometry(objective, [], _scene(kind))._static_bound_reuse
