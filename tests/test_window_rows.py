"""Continuity duplicate compression cannot weaken any original halfspace."""

import numpy as np
from scipy import sparse

from omniretargeting._fast.qp_guard_rows import HardBlock
from omniretargeting._fast.window_rows import merge_temporal


def test_identical_rows_keep_tightest_bound_and_switching_nonlinear_witness():
    A = sparse.csr_matrix([[1., -1.], [-1., 1.]])
    first = HardBlock(A, np.array([1., 2.]), evaluate=lambda p: np.array([1. - p[0, 0] ** 2, 2.]))
    second = HardBlock(A, np.array([2., 1.]), evaluate=lambda p: np.array([2. - 4 * p[0, 0] ** 2, 1.]))
    merged = merge_temporal([first, second])
    assert len(merged) == 1 and len(merged[0].b) == 2
    np.testing.assert_array_equal(merged[0].A.toarray(), A.toarray())
    np.testing.assert_array_equal(merged[0].b, [1., 1.])
    np.testing.assert_array_equal(merged[0].evaluate(np.array([[1.]])), [-2., 1.])
    for delta in ([0., 0.], [1.5, 0.], [0., 1.5]):
        expected = all(np.all(block.A @ delta <= block.b) for block in [first, second])
        assert np.all(merged[0].A @ delta <= merged[0].b) == expected


def test_one_ulp_distinct_jacobian_rows_are_not_merged():
    a = HardBlock(sparse.csr_matrix([[1., -1.]]), np.ones(1), evaluate=lambda p: np.ones(1))
    b = HardBlock(sparse.csr_matrix([[np.nextafter(1., 2.), -1.]]), np.ones(1), evaluate=lambda p: np.ones(1))
    assert len(merge_temporal([a, b])) == 2
