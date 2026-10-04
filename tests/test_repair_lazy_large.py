"""Keep productive small restoration order; defer expensive large work."""
from contextlib import closing
from types import SimpleNamespace

import numpy as np
import pytest

from omniretargeting._fast import mesh_repair, repair_proposals


@pytest.mark.parametrize('moving_frames,expect_restore', [(40, True), (80, False)])
def test_projected_program_size_controls_order_before_geometry_commit(monkeypatch, moving_frames, expect_restore):
    monkeypatch.setattr(repair_proposals, 'LAZY_LARGE_RESTORATION', True)
    monkeypatch.setattr(repair_proposals, 'LAZY_MIN_VARIABLES', 48)
    monkeypatch.setattr(repair_proposals, 'LAZY_RESTORATION', False)
    monkeypatch.setattr(mesh_repair, 'OPTIONS', SimpleNamespace(
        lazy_repair_restoration=False, parallel_repair_restore=False))
    current = np.zeros((80, 1))
    step = np.zeros_like(current)
    step[:moving_frames] = .04
    mask = np.arange(80) < moving_frames
    limits = SimpleNamespace(coords=SimpleNamespace(qi=np.array([0])), width=1, frames=80,
        frame_mask=mask, movable_body=None, limb_scopes=(), objective=SimpleNamespace())
    limits.check = lambda q: (bool(np.max(np.abs(q)) <= .02), {})
    restored = []

    def restore(limits, origin, candidate, geometry, measured, protected, projection):
        restored.append(projection.shape[1])
        return candidate * .25, {'status': 'restored'}
    monkeypatch.setattr(mesh_repair, 'restore_repair', restore)
    iterator = repair_proposals._candidate_iterator(limits, SimpleNamespace(model=None), current,
        {}, {}, None, .005, step, {})
    with closing(iterator):
        alpha, q = next(iterator)
    assert limits.check(q)[0]
    if expect_restore:
        # A small frame-local program stays restoration-first even though the
        # whole trajectory exceeds the threshold. Its full step can commit.
        assert restored == [40] and alpha == 1.
        np.testing.assert_array_equal(q, .25 * step)
    else:
        # Caller commits the ordinary admissible fraction, so closing the
        # stream does not perform any unconsumed restoration.
        assert restored == [] and alpha == .5
        np.testing.assert_array_equal(q, .5 * step)


@pytest.mark.parametrize('local_only', [False, True])
def test_full_domain_fallback_is_preserved_by_default_and_explicitly_experimental(monkeypatch, local_only):
    monkeypatch.setattr(repair_proposals, 'LOCAL_RESTORATION_ONLY', local_only)
    monkeypatch.setattr(repair_proposals, 'LAZY_LARGE_RESTORATION', False)
    monkeypatch.setattr(mesh_repair, 'OPTIONS', SimpleNamespace(
        lazy_repair_restoration=False, parallel_repair_restore=False))
    current = np.zeros((80, 1))
    step = np.zeros_like(current)
    step[:40] = .04
    limits = SimpleNamespace(coords=SimpleNamespace(qi=np.array([0])), width=1, frames=80,
        frame_mask=np.arange(80) < 40, movable_body=None, limb_scopes=(), objective=SimpleNamespace())
    limits.check = lambda q: (bool(np.max(np.abs(q)) <= .02), {})
    calls = []

    def restore(limits, origin, candidate, geometry, measured, protected, projection):
        calls.append(projection.shape[1])
        return ((None, {'status': 'unresolved'}) if projection.shape[1] == 40 else
                (candidate * .25, {'status': 'restored'}))
    monkeypatch.setattr(mesh_repair, 'restore_repair', restore)
    iterator = repair_proposals._candidate_iterator(limits, SimpleNamespace(model=None), current,
        {}, {}, None, .005, step, {})
    with closing(iterator):
        alpha, q = next(iterator)
    assert limits.check(q)[0]
    assert calls == ([40] if local_only else [40, 80])
    assert alpha == (.5 if local_only else 1.)
