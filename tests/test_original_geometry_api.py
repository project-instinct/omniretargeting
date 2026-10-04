"""Explicit tangent columns and immutable reference evidence across workers."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting._fast.mesh_repair import OriginalGeometry
from tests.test_original_geometry_batches import _assert_same, _problem, _scene


@pytest.fixture(autouse=True)
def cpu_queries(monkeypatch):
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "0")


def _rotated_problem():
    objective, poses = _problem()
    objective.coords.cols = np.arange(objective.model.nv)
    tangent = np.zeros(objective.model.nv)
    tangent[:6] = [.03, -.02, .04, .17, -.22, .11]
    for pose in poses:
        mujoco.mj_integratePos(objective.model, pose, tangent, 1.)
    return objective, poses


def _objects():
    return [dict(mesh=trimesh.creation.box(extents=[.22, .18, .2]),
                 translations=np.tile([.21, 0, .53], (8, 1)))]


def _reference_jacobian(model, data, key, row, columns):
    first = np.zeros((3, model.nv))
    if key[1] == "self":
        second = np.zeros_like(first)
        mujoco.mj_jac(model, data, first, None, row["points"][0], row["body_ids"][0])
        mujoco.mj_jac(model, data, second, None, row["points"][1], row["body_ids"][1])
        # Preserve the original self path: full relative row, then select.
        return (row["direction"] @ (first - second))[columns]
    mujoco.mj_jac(model, data, first, None, row["point"], int(model.geom_bodyid[row["geom_id"]]))
    return row["direction"] @ first[:, columns]


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("manifold", [False, True])
def test_explicit_columns_preserve_scene_moving_self_rows_and_linearization(parallel, manifold, monkeypatch):
    from omniretargeting._fast import parallel_geometry

    objective, poses = _rotated_problem()
    columns = np.array([5, 0, 7, 3, 2, 6, 1, 4])
    geometry = OriginalGeometry(objective, _objects(), _scene("closed"),
                                manifold=manifold, columns=columns, reference_poses=poses)
    assert geometry.width == len(columns) and geometry.di is geometry.columns
    columns[0] = 1  # The constructor owns its column vector.
    assert geometry.columns[0] == 5 and not geometry.columns.flags.writeable
    if parallel:
        monkeypatch.setattr(parallel_geometry, "WORKERS", 3)
        monkeypatch.setattr(parallel_geometry, "MIN_FRAMES_PER_WORKER", 2)
        geometry.start_workers(len(poses))
    try:
        measured = geometry.measure(poses, jacobians=True)
        assert measured["complete"] and not measured["failures"]
        assert {key[1] for key, _ in measured["rows"]} == {"static", "moving", "self"}
        data = mujoco.MjData(objective.model)
        for key, row in measured["rows"]:
            data.qpos[:] = poses[key[0]]
            mujoco.mj_forward(objective.model, data)
            expected = _reference_jacobian(objective.model, data, key, row, geometry.columns)
            assert row["jacobian"].tobytes() == expected.tobytes()
            if key[1] == "self":
                np.testing.assert_allclose(row["jacobian"][[0, 1, 3, 4, 6, 7]], 0., atol=2e-16)
        without_rows = geometry.measure(poses)
        _assert_same(geometry.linearize(poses, without_rows), measured)
        matrix, phi = geometry.linear_rows(measured)
        assert matrix.shape == (len(phi), objective.frames * geometry.width)
        for index, (key, row) in enumerate(measured["rows"]):
            block = matrix[index].toarray().reshape(objective.frames, geometry.width)
            np.testing.assert_array_equal(block[key[0]], row["jacobian"])
            assert not np.any(np.delete(block, key[0], axis=0))
            assert phi[index] == row["clearance"]
    finally:
        geometry.stop_workers()


def test_static_root_tangent_derivative_uses_material_point_and_freejoint_basis():
    objective, poses = _rotated_problem()
    geometry = OriginalGeometry(objective, [], _scene("closed"), columns=objective.coords.cols)
    measured = geometry.measure(poses, jacobians=True)
    key, row = next((k, r) for k, r in measured["rows"] if k[1] == "static")
    sample, geom = row["sample_id"], row["geom_id"]
    data = mujoco.MjData(objective.model)
    finite = []
    for column in range(6):
        values = []
        tangent = np.eye(objective.model.nv)[column]
        for sign in (-1, 1):
            data.qpos[:] = poses[key[0]]
            mujoco.mj_integratePos(objective.model, data.qpos, tangent, sign * 1e-6)
            mujoco.mj_forward(objective.model, data)
            point = data.geom_xpos[geom] + data.geom_xmat[geom].reshape(3, 3) @ objective.shell_local[sample]
            values.append(geometry.scene.query(point[None])["clearance"][0])
        finite.append((values[1] - values[0]) / 2e-6)
    np.testing.assert_allclose(row["jacobian"][:6], finite, rtol=0, atol=1e-8)
    assert np.any(np.abs(row["jacobian"][3:6]) > .001)


def test_default_scalar_map_remains_bitwise_compatible_and_empty_map_keeps_constant_rows():
    objective, poses = _rotated_problem()
    original = OriginalGeometry(objective, [], _scene("closed"))
    explicit = OriginalGeometry(objective, [], _scene("closed"), columns=objective.coords.di)
    assert original.width == len(objective.coords.qi)
    _assert_same(original.measure(poses, jacobians=True), explicit.measure(poses, jacobians=True))
    constant = OriginalGeometry(objective, [], _scene("closed"), columns=[])
    measured = constant.measure(poses, jacobians=True)
    matrix, phi = constant.linear_rows(measured)
    assert matrix.shape == (len(measured["rows"]), 0) and np.any(phi < 0)
    matrix, phi = constant.linear_rows(dict(rows=[]))
    assert matrix.shape == (0, 0) and phi.shape == (0,)


@pytest.mark.parametrize("columns", [[-1], [8], [0, 0], [[0, 1]], [1.5], [True]])
def test_bad_tangent_columns_are_rejected(columns):
    objective, _ = _problem()
    with pytest.raises(ValueError, match="columns"):
        OriginalGeometry(objective, [], None, columns=columns)


class _EvidenceMoving:
    """Controlled source observation loss, independent of geometric depths."""
    def __init__(self):
        self.query_count = 0

    def query(self, pose, frame, dofs):
        self.query_count += 1
        sample = ("trial" if pose[0] > 1 else "reference", int(frame))
        return dict(rows=[], failures=["source-side evidence unknown"] if pose[0] < -1 else [],
                    open_known_samples={sample}, open_penetrating_samples={sample})


@pytest.mark.parametrize("when", ["serial", "before_workers", "after_workers"])
def test_reference_registry_stays_frozen_across_trials_worker_sync_and_restart(when, monkeypatch):
    from omniretargeting._fast import parallel_geometry

    objective, poses = _problem()
    monkeypatch.setattr(parallel_geometry, "WORKERS", 3)
    monkeypatch.setattr(parallel_geometry, "MIN_FRAMES_PER_WORKER", 2)
    geometry = OriginalGeometry(objective, [], None)
    geometry.moving = _EvidenceMoving()
    if when == "after_workers":
        geometry.start_workers(len(poses))
    evidence_id = geometry.freeze_reference_observations(poses)
    if when == "before_workers":
        geometry.start_workers(len(poses))
    registry = dict(geometry._moving_evidence_baseline)
    try:
        if when != "serial":
            assert geometry.parallel.reference_observation_id == evidence_id
        expected = geometry.measure(poses)
        assert expected["complete"] and not expected["failures"]
        trial = poses.copy()
        trial[:, 0] += 2.
        for _ in range(2):
            rejected = geometry.measure(trial, protected_depths=expected["depths"])
            assert not rejected["complete"]
            assert any("lost original source-side observation" in f for f in rejected["failures"])
            assert geometry.reference_observation_id == evidence_id
            assert dict(geometry._moving_evidence_baseline) == registry
        _assert_same(geometry.measure(poses), expected)
        count = geometry.moving.query_count
        assert geometry.freeze_reference_observations(poses.copy()) == evidence_id
        assert geometry.moving.query_count == count
        changed = poses.copy()
        changed[0, 0] = np.nextafter(changed[0, 0], 1.)
        with pytest.raises(ValueError, match="rebased"):
            geometry.freeze_reference_observations(changed)
        with pytest.raises(TypeError):
            geometry._moving_evidence_baseline[0] = frozenset()
        geometry.stop_workers()
        geometry.start_workers(len(poses))
        assert geometry.parallel.reference_observation_id == evidence_id
        assert not geometry.measure(trial, protected_depths={})["complete"]
        _assert_same(geometry.measure(poses), expected)
    finally:
        geometry.stop_workers()


def test_unknown_reference_cannot_become_clear_by_querying_a_new_trial():
    objective, poses = _problem()
    reference = poses.copy()
    reference[2, 0] = -2.
    geometry = OriginalGeometry(objective, [], None)
    geometry.moving = _EvidenceMoving()
    geometry.freeze_reference_observations(reference)
    measured = geometry.measure(poses)
    assert any("unknown reference moving frame 2" in failure for failure in measured["failures"])
    assert geometry.reference_observation_failures[2]


def test_missing_frozen_registry_entry_is_unknown_instead_of_trial_seeded():
    objective, poses = _problem()
    geometry = OriginalGeometry(objective, [], None)
    geometry.moving = _EvidenceMoving()
    geometry._install_reference_observations({}, {}, None, "missing-registry")
    measured = geometry.measure_frames(poses, [2])
    assert measured["frames"][0][3] == ["missing frozen source-side registry at frame 2"]
    assert not geometry._moving_evidence_baseline


def test_actual_open_object_preserves_legacy_rows_when_reference_is_known():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><freejoint/><geom type="sphere" size=".03"/></body>
    </worldbody></mujoco>''')
    poses = np.tile(model.qpos0, (2, 1))
    poses[:, 0], poses[:, 2] = -.04, .5
    objective = SimpleNamespace(model=model, frames=2, coords=SimpleNamespace(qi=[], di=np.array([], int)))
    wall = trimesh.Trimesh(vertices=[[0, -.5, 0], [0, .5, 0], [0, .5, 1], [0, -.5, 1]],
                           faces=[[0, 1, 2], [0, 2, 3]], process=False)
    objects = [dict(mesh=wall, collision_evidence=dict(
        points=[[0, 0, .5]], normals=[[1, 0, 0]], radius=.3))]
    legacy = OriginalGeometry(objective, objects, None, columns=np.arange(model.nv))
    frozen = OriginalGeometry(objective, objects, None, columns=np.arange(model.nv), reference_poses=poses)
    assert any(frozen._moving_evidence_baseline.values())
    _assert_same(frozen.measure(poses, jacobians=True), legacy.measure(poses, jacobians=True))
