"""Short-clip stencil safety and sparse equivalence to the production audit."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from omniretargeting._fast.local_chatter import JointChatter, KERNEL_RADIUS, high_pass_matrix
from omniretargeting.motion_budgets import high_frequency


@pytest.mark.parametrize("frames", [1, 2, 7, 16, 17, 40])
@pytest.mark.parametrize("position", ["left", "middle", "right", "both", "none"])
def test_stencil_clips_to_mask_length_and_can_index_audit(frames, position):
    chatter = JointChatter(SimpleNamespace(qi=np.array([0])), np.zeros((frames, 1)))
    mask = np.zeros(frames, dtype=bool)
    changed = {"left": [0], "middle": [frames // 2], "right": [frames - 1],
               "both": [0, frames - 1], "none": []}[position]
    mask[changed] = True
    stencil = chatter.stencil_frames(mask)
    expected = [frame for frame in range(frames)
                if any(abs(frame - point) <= KERNEL_RADIUS for point in changed)]
    np.testing.assert_array_equal(stencil, expected)
    assert np.all((stencil >= 0) & (stencil < frames))
    assert chatter.excess(np.zeros((frames, 1)))[stencil].shape == (len(expected), 1)


def test_empty_stencil_and_high_pass():
    chatter = JointChatter(SimpleNamespace(qi=np.array([0])), np.empty((0, 1)))
    assert chatter.stencil_frames(np.zeros(0, bool)).size == 0
    assert chatter.operator.shape == (0, 0)


@pytest.mark.parametrize("frames", [*range(1, 61), 500])
def test_sparse_high_pass_matches_scipy_kernel_and_nearest_boundaries(frames):
    rng = np.random.default_rng(frames)
    signals = np.column_stack((rng.normal(size=(frames, 3)), np.ones(frames),
                               np.linspace(-2., 2., frames)))
    impulses = np.zeros((frames, 3))
    impulses[0, 0] = impulses[frames // 2, 1] = impulses[-1, 2] = 1.
    signals = np.column_stack((signals, impulses))
    operator = high_pass_matrix(frames)
    difference = operator @ signals - high_frequency(signals)
    assert np.max(np.abs(difference)) <= 1e-12


@pytest.mark.parametrize("frames", [1, 2, 7, 17, 500, 30000])
def test_high_pass_storage_has_fixed_bandwidth(frames):
    operator = high_pass_matrix(frames)
    assert sparse.isspmatrix_csr(operator)
    assert operator.nnz <= frames * (2 * KERNEL_RADIUS + 1)
    rows, columns = operator.nonzero()
    assert np.all(np.abs(rows - columns) <= KERNEL_RADIUS)
