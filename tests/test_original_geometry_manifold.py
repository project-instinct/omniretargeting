"""The production repair keeps complete depth auditing with bounded witnesses."""
from collections import Counter

import numpy as np
import pytest

from omniretargeting._fast.mesh_repair import OriginalGeometry
from tests.test_original_geometry_batches import (
    _assert_same,
    _problem,
    _RecordingQuery,
    _scene,
)


@pytest.mark.parametrize('kind', ['closed', 'evidence', 'mixed'])
def test_manifold_depths_unknowns_and_witnesses_survive_batching_and_cache(kind):
    objective, poses = _problem()
    # One query semantics for all three: the recording wrapper is not the
    # repository SceneQuery, so none of them takes the open-scan rigid fallback.
    reference = OriginalGeometry(objective, [], _RecordingQuery(_scene(kind))).measure(poses)
    fresh = OriginalGeometry(objective, [], _RecordingQuery(_scene(kind)), manifold=True)
    expected = fresh.measure(poses, jacobians=True)
    query = _RecordingQuery(_scene(kind))
    split = OriginalGeometry(objective, [], query, manifold=True)
    split.SCENE_BATCH_POINTS = 5
    actual = split.measure(poses, jacobians=True)
    assert actual['depths'] == reference['depths']
    assert actual['failures'] == reference['failures']
    _assert_same(actual, expected)
    counts = Counter(key for key, _ in actual['rows'] if key[1] == 'static')
    assert max(counts.values(), default=0) <= 15
    query_count = len(query.points)
    _assert_same(split.measure(poses, jacobians=True), expected)
    assert len(query.points) == query_count
    # Cached arrays never escape by reference.
    row = next((row for key, row in actual['rows'] if key[1] == 'static'), None)
    if row is not None:
        row['point'][:] = 1e5
        row['direction'][:] = 0
        _assert_same(split.measure(poses, jacobians=True), expected)


def test_manifold_changed_geometry_uses_current_frame_jacobians():
    objective, poses = _problem()
    query = _RecordingQuery(_scene('closed'))
    geometry = OriginalGeometry(objective, [], query, manifold=True)
    geometry.measure(poses, jacobians=True)
    query.points.clear()
    changed = poses.copy()
    changed[4, 7] += .03
    actual = geometry.measure(changed, jacobians=True)
    expected = OriginalGeometry(objective, [], _scene('closed'), manifold=True).measure(changed, jacobians=True)
    _assert_same(actual, expected)
    assert sum(map(len, query.points)) == 6
    assert np.isfinite(geometry.linear_rows(actual)[0].data).all()
