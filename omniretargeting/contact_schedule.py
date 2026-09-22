"""Temporal contact hysteresis and run filtering."""

from __future__ import annotations

import numpy as np
from scipy.special import expit


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return expit(np.clip(value, -60.0, 60.0))


def _iter_true_runs(mask: np.ndarray):
    start = None
    for idx, value in enumerate(np.r_[mask, False]):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            yield start, idx
            start = None


def _close_short_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    if max_gap_frames <= 0:
        return result
    false_runs = list(_iter_true_runs(~result))
    for start, end in false_runs:
        if start == 0 or end == len(result):
            continue
        if end - start <= max_gap_frames:
            result[start:end] = True
    return result


def _remove_short_runs(mask: np.ndarray, min_contact_frames: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    if min_contact_frames <= 1:
        return result
    for start, end in list(_iter_true_runs(result)):
        if end - start < min_contact_frames:
            result[start:end] = False
    return result


def _apply_contact_hysteresis(
    enter_mask: np.ndarray,
    stay_mask: np.ndarray,
) -> np.ndarray:
    """Keep an active contact until the looser exit condition is violated."""
    enter = np.asarray(enter_mask, dtype=bool)
    stay = np.asarray(stay_mask, dtype=bool)
    if enter.shape != stay.shape:
        raise ValueError(
            "enter_mask and stay_mask must have the same shape, got "
            f"{enter.shape} and {stay.shape}."
        )
    result = np.zeros_like(enter)
    active = False
    for frame_idx in range(len(result)):
        if active:
            active = bool(stay[frame_idx])
        elif enter[frame_idx]:
            active = True
        result[frame_idx] = active
    return result
