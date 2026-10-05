"""Long layout flips stay sustained, including former int8 wrap boundaries."""

import numpy as np
import pytest

from omniretargeting.pose_relations import _sustained


@pytest.mark.parametrize('length', [127, 128, 129, 256, 300, 600])
@pytest.mark.parametrize('window', [1, 7, 129, 257])
def test_long_true_runs_do_not_lose_sustained_layout(length, window):
    # Independent components start at different frames; false resets each
    # counter before a second long run. Expected history never wraps.
    frames = 2 * length + 17
    mask = np.zeros((frames, 2, 2), bool)
    expected = np.zeros_like(mask)
    segments = [(3, 3 + length, 0, 0),
                (7, 7 + length, 1, 0),
                (length + 8, 2 * length + 8, 0, 0),
                (0, frames, 1, 1)]
    for first, stop, row, column in segments:
        mask[first:stop, row, column] = True
        if stop - first >= window:
            expected[first + window - 1:stop, row, column] = True
    np.testing.assert_array_equal(_sustained(mask, window), expected)


def test_layout_sustained_counter_handles_a_scalar_component():
    mask = np.r_[np.ones(300, bool), False, np.ones(300, bool)]
    expected = mask.copy()
    expected[:6] = expected[301:307] = False
    np.testing.assert_array_equal(_sustained(mask, 7), expected)
