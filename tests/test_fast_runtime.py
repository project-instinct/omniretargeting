"""Runtime changes must preserve linearizations across trials and reweighting."""

import numpy as np
import pytest
from scipy import sparse

import omniretargeting._fast.objective as fast_objective
import omniretargeting._fast.pipeline as fast_pipeline
from omniretargeting._fast.rows import _block_rows
from tests.test_fast_backend import _objective


def _same_evaluation(first, second):
    assert first[0].cost == pytest.approx(second[0].cost, rel=1e-12)
    np.testing.assert_allclose(first[1], second[1], rtol=0, atol=0)
    if first[0].jacobian:
        np.testing.assert_array_equal(first[0].hessian, second[0].hessian)
        np.testing.assert_array_equal(first[0].gradient, second[0].gradient)
        np.testing.assert_array_equal(first[2].toarray(), second[2].toarray())
    assert first[3] == second[3]


def test_parallel_results_survive_later_trials_and_worker_restart(monkeypatch):
    objective, poses, _ = _objective(temporal=True, extra=True)
    monkeypatch.setattr(fast_objective, "WORKERS", 3)
    monkeypatch.setattr(fast_objective, "MIN_CHUNK_FRAMES", 8)
    initial = objective.evaluate(poses)
    moved = poses.copy()
    moved[:, objective.coords.qi] += 0.01
    expected = objective.evaluate(moved, with_jacobian=False)
    objective.start_workers()
    try:
        retained = objective.evaluate(poses)
        trial = objective.evaluate(moved, with_jacobian=False)
        _same_evaluation(expected, trial)
        objective.evaluate(moved)
        _same_evaluation(initial, retained)
    finally:
        objective.stop_workers()
    # A later penetration stage changes held contacts before restarting.
    objective.surfaces[0]["hold"].fill(0.01)
    objective.envelope_weight[::3] *= 10.0
    expected = objective.evaluate(moved)
    objective.start_workers()
    try:
        _same_evaluation(expected, objective.evaluate(moved))
    finally:
        objective.stop_workers()


def test_values_do_not_depend_on_the_chunking(monkeypatch):
    """Self-collision rows differ in count between frames; neither they nor
    the cost summation may depend on how frames are grouped into chunks."""
    from omniretargeting.self_collision import SelfCollisionQuery

    objective, poses, _ = _objective(temporal=True, extra=True)
    objective.self_collision = SelfCollisionQuery(objective.model)
    objective.cfg = dict(objective.cfg, self_clearance_m=0.35)
    rng = np.random.default_rng(1)
    moved = objective.coords.integrate(poses, 0.2 * rng.normal(size=(len(poses), objective.width)), 1.0)
    monkeypatch.setattr(fast_objective, "WORKERS", 1)
    serial = objective.evaluate(moved)
    counts = []
    for frame in range(len(moved)):
        counts.append(sum(row["clearance"] < 0.35 for row in objective.self_collision.query(moved[frame])))
    assert len(set(counts)) > 1 and max(counts) > 0
    for workers, frames, per_worker in ((3, 8, 1), (2, 5, 2), (4, 7, 2)):
        monkeypatch.setattr(fast_objective, "WORKERS", workers)
        monkeypatch.setattr(fast_objective, "MIN_CHUNK_FRAMES", frames)
        monkeypatch.setattr(fast_objective, "CHUNKS_PER_WORKER", per_worker)
        objective.start_workers()
        try:
            assert objective.pool is not None
            parallel = objective.evaluate(moved)
        finally:
            objective.stop_workers()
        assert parallel[0].cost == serial[0].cost
        np.testing.assert_array_equal(parallel[0].frame_cost, serial[0].frame_cost)
        _same_evaluation(serial, parallel)


def test_deepest_fields_in_worker_chunks_equal_the_serial_pass(monkeypatch):
    from omniretargeting._fast.sdf import GridSDF

    objective, poses, _ = _objective(temporal=True)
    lower, spacing = np.array([-1.0, -1.0, -0.2]), 0.02
    shape = (120, 120, 90)
    rng = np.random.default_rng(2)
    z = lower[2] + spacing * np.arange(shape[2])
    clearance = (np.broadcast_to(z - 0.02, shape) + 0.002 * rng.normal(size=shape)).astype(np.float32)
    clearance[rng.random(shape) < 0.01] = np.nan
    objective.scene_sdf = GridSDF(lower, spacing, clearance, np.zeros(shape + (3,), np.float32))
    moved = poses.copy()
    moved[:, objective.coords.root_q + 2] -= np.linspace(0.0, 0.9, len(poses))
    serial = objective.deepest_fields(moved)
    assert len(serial[0][1]) and np.isfinite(serial[0][0]).any()
    monkeypatch.setattr(fast_objective, "WORKERS", 3)
    monkeypatch.setattr(fast_objective, "MIN_CHUNK_FRAMES", 5)
    objective.start_workers()
    try:
        assert objective.pool is not None
        parallel = objective.deepest_fields(moved)
    finally:
        objective.stop_workers()
    for a, b in zip(serial, parallel):
        for x, y in zip(a, b):
            assert x.dtype == y.dtype
            np.testing.assert_array_equal(x, y)


def test_temporal_cache_tracks_in_place_weights_and_stage_scales():
    objective, poses, _ = _objective(temporal=True)
    for scale, multiplier in ((1.0, 1.0), (100.0, 10.0), (10.0, 0.5)):
        objective.correction_temporal_scale = scale
        objective.temporal_weight[1][::2] *= multiplier
        objective.evaluate(poses)
        for root in (False, True):
            cached = objective._temporal_matrices(scale, root=root)
            if root:
                operators = [
                    (op, size, i)
                    for ops, size in ((objective.xy_ops, 2), (objective.height_ops, 1),
                                      (objective.angular_ops, 4))
                    for i, op in enumerate(ops)
                ]
            else:
                operators = [(op, len(objective.coords.qi), i)
                             for i, op in enumerate(objective.joint_ops)]
            for matrix, (op, size, i) in zip(cached, operators):
                # Dense Kronecker construction is independent of the cached
                # sparse factors and includes locks through the joint selector.
                selector = np.eye(size) if root else objective.selector
                expected = np.kron(op.toarray(), selector)
                expected *= np.repeat(objective.temporal_weight[i] * scale, size)[:, None]
                np.testing.assert_array_equal(matrix.toarray(), expected)
            assert cached is objective._temporal_matrices(scale, root=root)


def test_block_rows_store_only_nonzero_entries_without_changing_values():
    blocks = np.zeros((4, 3, 10))
    blocks[:, 0, 2], blocks[:, 1, 5], blocks[:, 2, 7] = 2.0, -3.0, 4.0
    matrix = _block_rows(blocks, 10)
    assert matrix.nnz == np.count_nonzero(blocks)
    np.testing.assert_array_equal(matrix.toarray(), sparse.block_diag(list(blocks)).toarray())


@pytest.mark.parametrize("fail", [False, True])
def test_fast_pipeline_restores_callers_blas_threads(monkeypatch, fail):
    threadpoolctl = pytest.importorskip("threadpoolctl")

    def counts():
        return [info["num_threads"] for info in threadpoolctl.threadpool_info()
                if info["user_api"] == "blas"]

    from types import SimpleNamespace
    import mujoco
    model = mujoco.MjModel.from_xml_string('<mujoco/>')
    owner = SimpleNamespace()
    inputs = (owner, model, model.qpos0, np.zeros((4, 1, 3)), [], {}, None,
              {}, {}, {}, {'framerate': 30.}, [])
    released = []

    def solve(*args, resources):
        assert counts() and set(counts()) == {1}
        # Geometry workers belong to this solve even when it raises. Check
        # their cleanup as well as restoration of the caller's BLAS settings.
        resources.callback(released.append, "geometry")
        assert not released
        if fail:
            raise RuntimeError("test solve failure")
        return np.empty((4, 0)), {"backend": "fast", "status": "test"}

    monkeypatch.setattr(fast_pipeline, "_solve_fast_sequence", solve)
    with threadpoolctl.threadpool_limits(2, user_api="blas"):
        before = counts()
        if fail:
            with pytest.raises(RuntimeError, match="test solve failure"):
                fast_pipeline.solve_fast_sequence(*inputs)
        else:
            poses, report = fast_pipeline.solve_fast_sequence(*inputs)
            assert poses.shape == (4, 0) and report['status'] == 'test'
            assert not report['deadline']['expired']
        assert counts() == before
        assert released == ["geometry"]
