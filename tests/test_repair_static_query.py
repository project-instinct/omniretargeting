"""Resident center screening retains original query values and dispatch."""
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from omniretargeting.scene_geometry import SceneQuery
from omniretargeting._fast import repair_static_query as centers


def problem():
    scene = SceneQuery(trimesh.creation.box(extents=[2., 2., .1]))
    geometry = SimpleNamespace(scene=scene)
    points = np.random.default_rng(14).uniform([-1.2, -1.2, -.06], [1.2, 1.2, .06], (2047, 3))
    points = np.vstack([points, [0., 0., .05], [0., 0., -.05], [0., 0., 0.]])
    return geometry, points


def assert_bytes(first, second):
    assert first.keys() == second.keys()
    for key in first:
        assert first[key].dtype == second[key].dtype
        assert first[key].shape == second[key].shape
        assert first[key].tobytes() == second[key].tobytes(), key


def test_values_reuse_and_changed_scene_match_original(monkeypatch):
    geometry, points = problem()
    build, calls = centers.serial_scene, []

    def counted(scene):
        calls.append(scene)
        return build(scene)

    monkeypatch.setattr(centers, 'serial_scene', counted)
    for offset in (0., 1e-6, -.001):
        actual = centers.query_scene(geometry, points + offset)
        assert_bytes(actual, geometry.scene.query(points + offset))
    assert len(calls) == 1
    geometry.scene = SceneQuery(trimesh.creation.box(extents=[1., 1., .2]))
    assert_bytes(centers.query_scene(geometry, points), geometry.scene.query(points))
    assert len(calls) == 2


def test_replaced_query_keeps_custom_dispatch_after_cache_exists(monkeypatch):
    geometry, points = problem()
    centers.query_scene(geometry, points)
    original, calls = geometry.scene.query, []

    def changed(points):
        calls.append(1)
        result = original(points)
        result['uncertain'][:] = True
        return result

    monkeypatch.setattr(geometry.scene, 'query', changed)
    actual = centers.query_scene(geometry, points)
    assert actual['uncertain'].all() and len(calls) == 1


def test_missing_native_backend_falls_back_once(monkeypatch):
    geometry, points = problem()
    calls = []

    def unavailable(scene):
        calls.append(scene)
        raise FileNotFoundError('optional winding headers')

    monkeypatch.setattr(centers, 'serial_scene', unavailable)
    for _ in range(2):
        assert_bytes(centers.query_scene(geometry, points), geometry.scene.query(points))
    assert len(calls) == 1


def test_query_error_is_propagated(monkeypatch):
    geometry, points = problem()

    def unavailable(points):
        raise ValueError('unknown query')

    monkeypatch.setattr(geometry.scene, 'query', unavailable)
    with pytest.raises(ValueError, match='unknown query'):
        centers.query_scene(geometry, points)
