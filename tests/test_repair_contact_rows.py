"""Chunked contact assembly preserves every original family and coefficient."""

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast import repair_contact_rows
from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
from omniretargeting._fast.contact_guard import ContactGuard
from omniretargeting._fast.repair_limits import RepairLimits
from tests.test_fast_legacy_window import problem
from tests.test_fast_repair_projected import assert_program_equal


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("chunk", [5, 16, 64])
def test_contact_chunk_cache_exact_full_bounds_and_invalidation(monkeypatch, preserve, chunk):
    monkeypatch.setattr(repair_contact_rows, "CHUNK_FRAMES", chunk)
    objective, poses = problem(extra=True)
    guards = [AcquisitionNormalGuard(objective, poses),
              ContactGuard(objective, poses, preserve_positions=preserve)]
    for guard in guards:
        for frame in (None, 12, 35, None):
            trial = poses.copy()
            if frame is not None:
                trial[frame, objective.coords.qi[0]] += .01
            expected = guard.repair_bounds(trial)
            actual = repair_contact_rows.contact_bounds(guard, trial)
            for full, cached in zip(expected, actual):
                assert len(full) == len(cached)
                for (a, lo, hi), (b, low, high) in zip(full, cached):
                    np.testing.assert_array_equal(a.toarray(), b.toarray())
                    np.testing.assert_array_equal(lo, low)
                    np.testing.assert_array_equal(hi, high)


def test_projected_program_with_every_immutable_guard(monkeypatch):
    monkeypatch.setattr(repair_contact_rows, "CHUNK_FRAMES", 8)
    objective, poses = problem(extra=True)
    objective.candidate_guards.append(ContactGuard(objective, poses, preserve_positions=True))
    limits = RepairLimits(objective, poses)
    size = len(poses) * limits.width
    selector = sparse.eye(size, format="csc")[:, np.arange(size).reshape(len(poses), -1)[10:17, :2].ravel()]
    for frame in (12, 25):
        candidate = poses.copy()
        candidate[frame, limits.coords.qi[-1]] += .0001
        for restore in (False, True):
            full = limits.constraints(candidate, restore_layout=restore)
            reduced = limits.constraints(candidate, restore_layout=restore, projection=selector)
            assert_program_equal(full, reduced, selector)
