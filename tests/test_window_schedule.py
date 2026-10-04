"""Scheduling cannot let simultaneous windows share a boundary stencil."""

from types import SimpleNamespace

import numpy as np
import pytest

from omniretargeting._fast.window_schedule import (dependency_window, independent_batches,
                                                  overlaps, sweep_windows)


@pytest.mark.parametrize('count', [1, 17, 32, 80, 957])
def test_two_shifted_sweeps_cover_all_frames_without_blending(count):
    for sweep in (0, 1):
        covered = np.zeros(count, bool)
        for window in sweep_windows(count, sweep):
            assert 0 < len(window) <= 32
            covered[list(window)] = True
        assert covered.all()


def test_half_overlap_two_colors_would_share_jerk_but_batches_do_not():
    count = 160
    component = SimpleNamespace(window_dependencies=lambda w: range(max(0, w.start - 3), min(count, w.stop + 3)))
    windows = [dependency_window(w, count, [component]) for w in sweep_windows(count, 0)]
    assert overlaps(windows[0].footprint, windows[2].footprint)
    batches = independent_batches(windows)
    assert len(batches) >= 3
    for batch in batches:
        for i, window in enumerate(batch):
            assert all(not overlaps(window.footprint, other.footprint) for other in batch[i + 1:])


def test_unknown_or_global_dependency_forces_serial_batches():
    windows = [dependency_window(w, 80, [object()]) for w in sweep_windows(80, 0)]
    assert all(len(batch) == 1 for batch in independent_batches(windows))
