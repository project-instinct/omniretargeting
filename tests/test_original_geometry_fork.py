"""Resident serial geometry preparation and fixed-witness nonlinear queries."""

import os
from pathlib import Path

import igl
import numpy as np
import pytest

from tests.test_original_geometry_window import _geometry, _restricted
from tests.test_original_geometry_batches import _assert_same

WINDING_HEADERS = Path(os.environ.get("OMNIRETARGETING_WINDING_INCLUDE",
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) /
    "omniretargeting/native/r4_winding_headers"))


@pytest.fixture(autouse=True)
def cpu_queries(monkeypatch):
    monkeypatch.setenv("OMNIRETARGETING_GPU_GEOMETRY", "0")


@pytest.mark.skipif(not (WINDING_HEADERS / "FastWindingNumberForSoups.h").exists(),
                    reason="optional native winding headers unavailable")
def test_cached_winding_and_serial_scene_equal_original_values_and_flags():
    from omniretargeting._fast.original_geometry_cpu import CachedWinding, serial_scene
    geometry, _ = _geometry()
    scene = geometry.scene
    vertices, faces, bounds, _ = scene._component_query_data[0]
    points = np.random.default_rng(12).uniform(bounds[0] - .01, bounds[1] + .01, size=(2047, 3))
    cached = CachedWinding(vertices, faces)
    assert cached(points).tobytes() == igl.fast_winding_number(vertices, faces, points).tobytes()
    original = scene.query(points)
    actual = serial_scene(scene).query(points)
    for field in original:
        assert actual[field].tobytes() == original[field].tobytes(), field
    shifted = points + np.array([1e-4, 0., 0.])
    hints = np.where(np.abs(original["clearance"]) > 1e-4 + 1e-5,
                     original["clearance"], np.nan)
    hinted = serial_scene(scene).query_with_sign_hints(shifted, hints)
    reference = scene.query(shifted)
    for field in reference:
        assert hinted[field].tobytes() == reference[field].tobytes(), field


@pytest.mark.skipif(not (WINDING_HEADERS / "FastWindingNumberForSoups.h").exists(),
                    reason="optional native winding headers unavailable")
def test_prepare_is_parent_only_idempotent_closes_pool_and_does_not_measure_reference(monkeypatch):
    from omniretargeting._fast import parallel_geometry
    monkeypatch.setattr(parallel_geometry, "WORKERS", 3)
    monkeypatch.setattr(parallel_geometry, "MIN_FRAMES_PER_WORKER", 2)
    geometry, poses = _geometry()
    geometry.start_workers(len(poses))
    expected = geometry.measure(poses, jacobians=True)
    original_id, original_registry = geometry.reference_observation_id, geometry._moving_evidence_baseline
    moving_count = geometry.moving.query_count
    geometry.prepare_fork_queries()
    assert geometry.parallel is None and not geometry._gpu_allowed
    assert geometry.reference_observation_id == original_id
    assert geometry._moving_evidence_baseline is original_registry
    assert geometry.moving.query_count == moving_count
    assert geometry.scene._r4_serial
    assert all(obj["query"]._r4_serial for obj in geometry.moving.objects)
    geometry.prepare_fork_queries()
    client = geometry.fork_private_client()
    _assert_same(client.measure(poses, frames=range(2, 6), jacobians=True), _restricted(expected, range(2, 6)))
    with pytest.raises(RuntimeError, match="worker pools"):
        client.start_workers(len(poses))
    altered = poses.copy()
    altered[2:6, 7] += .002
    oracle, _ = _geometry()
    reference = oracle.measure(altered, jacobians=True)
    actual = client.measure(altered, frames=range(2, 6), jacobians=True)
    _assert_same(actual, _restricted(reference, range(2, 6)))


def test_same_witness_query_retains_alignment_bits_and_reference_purity():
    geometry, poses = _geometry()
    frames = range(2, 6)
    measured = geometry.measure(poses, frames=frames, jacobians=True)
    reference_id = geometry.reference_observation_id
    baseline = geometry.reevaluate_witnesses(poses, measured, frames=frames)
    assert baseline["complete"] and baseline["reliable"].all()
    expected = np.array([row["clearance"] for _, row in measured["rows"]])
    assert baseline["clearance"].tobytes() == expected.tobytes()
    assert baseline["keys"] == tuple(key for key, _ in measured["rows"])
    candidate = poses.copy()
    candidate[2:6, 0] += 1e-7
    current = geometry.reevaluate_witnesses(candidate, measured, frames=frames)
    assert current["complete"] and current["reliable"].all()
    assert current["keys"] == baseline["keys"]
    # Directional derivative agrees with the frozen full-root Jacobians.
    derivatives = np.array([row["jacobian"][0] for _, row in measured["rows"]])
    np.testing.assert_allclose((current["clearance"] - expected) / 1e-7,
                               derivatives, atol=2e-5, rtol=2e-5)
    assert geometry.reference_observation_id == reference_id
    assert not measured["failures"]


def test_same_witness_rejects_different_reference():
    geometry, poses = _geometry()
    measured = geometry.measure(poses, frames=range(2, 6))
    measured["evidence_id"] = "unrelated"
    with pytest.raises(ValueError, match="reference"):
        geometry.reevaluate_witnesses(poses, measured, frames=range(2, 6))


def test_coarse_distance_cones_are_lower_bounds_and_unknown_outside_stays_unbounded():
    from omniretargeting._fast.original_geometry_bounds import ClearanceBounds
    geometry, _ = _geometry()
    origin, spacing = np.array([-1., -1., -1.]), .1
    shape = (21, 21, 21)
    points = np.stack(np.meshgrid(*[origin[j] + spacing * np.arange(shape[j]) for j in range(3)],
                                  indexing="ij"), axis=-1).reshape(-1, 3)
    nodes = geometry.scene.query(points)
    values = np.where(nodes["uncertain"], np.nan, nodes["clearance"]).reshape(shape)
    bounds = ClearanceBounds(origin, spacing, values)
    requested = np.random.default_rng(21).uniform(-.999, .999, (501, 3))
    exact = geometry.scene.query(requested)
    lower = bounds.query(requested)
    assert np.all(lower <= exact["clearance"] + 1e-12)
    assert np.isneginf(bounds.query([[2., 0., 0.]])).all()
    unknown = ClearanceBounds(origin, spacing, np.full(shape, np.nan))
    assert np.isneginf(unknown.query(requested)).all()


def test_window_parent_auto_gpu_attempt_ignores_child_env_and_closes_temporary_pools(monkeypatch):
    from dataclasses import replace
    from omniretargeting._fast import gpu_geometry, mesh_repair
    monkeypatch.setattr(mesh_repair, "OPTIONS", replace(mesh_repair.OPTIONS, windowed_priority=True))
    attempts = []
    monkeypatch.setattr(gpu_geometry, "initialize_static", lambda geometry: attempts.append(os.getpid()) or False)
    geometry, poses = _geometry()
    assert attempts == [os.getpid()] and geometry.scene._r4_serial
    initial = geometry.measure(poses)
    assert geometry.parallel is None
    identity = geometry.reference_observation_id
    geometry.prepare_fork_queries()
    assert geometry.fork_self_seed_stats["skipped"] is None
    assert geometry.fork_self_seed_stats["seeded_pairs"]
    client = geometry.fork_private_client()
    assert not client._gpu_allowed and client.parallel is None
    audit = geometry.measure(poses)
    _assert_same(audit, initial)
    assert attempts == [os.getpid(), os.getpid()]
    assert geometry._fork_queries_prepared and geometry.parallel is None and not geometry._gpu_allowed
    assert geometry.reference_observation_id == identity
