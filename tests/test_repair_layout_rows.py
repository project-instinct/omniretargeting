"""Layout reuse preserves all coefficients, margins and envelope row order."""

import numpy as np
import pytest

from omniretargeting._fast.repair_layout_rows import layout_bounds
from tests.test_fast_legacy_window import problem


@pytest.fixture(autouse=True)
def small_chunks(monkeypatch):
    from omniretargeting._fast import repair_layout_rows
    monkeypatch.setattr(repair_layout_rows, "CHUNK_FRAMES", 16)


def _problem():
    from omniretargeting._fast.reference import ReferenceShape
    objective, poses = problem(extra=True)
    objective.source_shape = ReferenceShape(objective, poses)
    return objective, poses


def _equal(actual, expected):
    assert len(actual[0]) == len(expected[0])
    assert actual[1] == expected[1] == []
    for a, b in zip(actual[0], expected[0]):
        assert a[0].shape == b[0].shape
        np.testing.assert_array_equal(a[0].indptr, b[0].indptr)
        np.testing.assert_array_equal(a[0].indices, b[0].indices)
        for x, y in zip((a[0].data, a[1], a[2]), (b[0].data, b[1], b[2])):
            assert x.tobytes() == y.tobytes()


@pytest.mark.parametrize("restore", [False, True])
def test_layout_chunk_rows_equal_original_full_rows_across_two_references(restore):
    objective, poses = _problem()
    guard = objective.source_shape
    acquired = poses.copy()
    acquired[:, objective.coords.qi[0]] += .001
    guard.freeze_reference(acquired)
    for trial in (poses, acquired, poses.copy()):
        _equal(layout_bounds(guard, trial, restore=restore), guard.repair_bounds(trial, restore=restore))


def test_layout_chunk_cache_detects_reference_content_mutation():
    objective, poses = _problem()
    guard = objective.source_shape
    layout_bounds(guard, poses)
    guard.envelopes[0].reference[25, 0, 0] += .01
    _equal(layout_bounds(guard, poses), guard.repair_bounds(poses))


def test_one_changed_pose_reuses_unmodified_layout_chunks():
    objective, poses = _problem()
    guard = objective.source_shape
    layout_bounds(guard, poses)
    previous = {start: next(iter(cache.values())) for start, cache in guard._repair_layout_cache[1].items()}
    changed = poses.copy()
    changed[20, objective.coords.qi[0]] += .001
    _equal(layout_bounds(guard, changed), guard.repair_bounds(changed))
    assert [len(cache) for cache in guard._repair_layout_cache[1].values()] == [1, 2, 1]
    assert next(iter(guard._repair_layout_cache[1][0].values())) is previous[0]


def test_custom_layout_method_keeps_its_original_callback(monkeypatch):
    objective, poses = _problem()
    guard = objective.source_shape
    monkeypatch.setattr(guard, "repair_bounds", lambda *args, **kwargs: ([], []))
    assert layout_bounds(guard, poses) is None
