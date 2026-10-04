"""Vectorized true-run extraction must preserve the temporal boundaries."""

import inspect

import numpy as np
import pytest

from omniretargeting.contact_schedule import _iter_true_runs


def _reference_true_runs(mask):
    """The original scalar implementation, retained as an exact oracle."""
    start = None
    for idx, value in enumerate(np.r_[mask, False]):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            yield start, idx
            start = None


def _assert_same_runs(mask):
    expected = list(_reference_true_runs(mask))
    actual = list(_iter_true_runs(mask))
    assert actual == expected
    assert all(type(start) is int and type(stop) is int for start, stop in actual)


def test_function_remains_a_generator():
    assert inspect.isgeneratorfunction(_iter_true_runs)


@pytest.mark.parametrize(
    "mask",
    [
        np.array([], dtype=bool),
        np.array([], dtype=np.int16),
        np.array([], dtype=np.float32),
        np.array(False),
        np.array(True),
        np.array(0),
        np.array(2),
        np.array([False]),
        np.array([True]),
        np.ones(37, dtype=bool),
        np.ones(37, dtype=np.uint8),
        np.array([True, True, False, True, False, False, True]),
        np.array([2, 0, -3, 0.0, np.nan, 1e-300]),
        np.array([None, "", 2, 0.0], dtype=object),
        [False, True, False, True, True],
    ],
)
def test_edge_cases_match_the_scalar_reference(mask):
    _assert_same_runs(mask)


@pytest.mark.parametrize("seed", range(24))
def test_random_numeric_masks_match_the_scalar_reference(seed):
    rng = np.random.default_rng(seed)
    for _ in range(8):
        length = int(rng.integers(0, 101))
        probability = float(rng.random())
        values = rng.random(length) < probability
        for dtype in (np.bool_, np.int8, np.uint16, np.int32, np.float64):
            if dtype == np.bool_:
                mask = values
            elif np.issubdtype(dtype, np.integer):
                mask = values.astype(dtype) * int(rng.integers(1, 5))
            else:
                mask = values.astype(dtype) * rng.random(length)
            _assert_same_runs(mask)


def test_multidimensional_masks_continue_to_be_rejected():
    mask = np.array([[True, False], [False, True]])
    with pytest.raises(ValueError):
        list(_reference_true_runs(mask))
    with pytest.raises(ValueError):
        list(_iter_true_runs(mask))
