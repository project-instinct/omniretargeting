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


def _progress(distance, far, near):
    """Share of the source's own travel from ``far`` to the anchor at ``near``."""
    span = distance[far] - distance[near]
    if not np.isfinite(span) or span <= 1e-9:
        return None
    return np.clip((distance[far] - distance) / span, 0.0, 1.0)


def approach_blend(tracking, anchors, contact_mask, *, return_contact_frames=False):
    """Hand a free target to its contact anchor by the source's own progress.

    For a run starting at ``s`` with anchor ``A``, the source endpoint's
    distance to ``A`` (its contact vector length) measures progress: from the
    farthest frame since the previous run it rises to exactly 1 at the onset.
    The world reference is the source path shifted to land on ``A`` there, so
    fast source motion hands over quickly and slow motion slowly, with no
    duration or distance threshold. Departures mirror this after each run;
    a gap between two runs splits where the source is farthest from both.
    Contact intervals, anchors and source samples are not changed.
    """
    tracking = np.asarray(tracking, dtype=float)
    anchors = np.asarray(anchors, dtype=float)
    frames = len(tracking)
    weights = np.zeros(frames)
    references = tracking.copy()
    contact_frames = np.arange(frames)
    runs = list(_iter_true_runs(np.asarray(contact_mask, dtype=bool)))
    if not runs or not np.isfinite(tracking).all():
        result = (weights, references, contact_frames)
        return result if return_contact_frames else result[:2]
    bounds = [(None, runs[0])] + list(zip(runs, runs[1:])) + [(runs[-1], None)]
    for before, after in bounds:
        start = 0 if before is None else before[1]
        stop = frames if after is None else after[0]
        if stop <= start:
            continue
        span = np.arange(start, stop)
        if before is not None:
            last = before[1] - 1
            leave = np.linalg.norm(tracking - anchors[last], axis=1)
        if after is not None:
            first = after[0]
            reach = np.linalg.norm(tracking - anchors[first], axis=1)
        if before is not None and after is not None:
            split = span[np.argmax(np.minimum(leave[span], reach[span]))]
        elif before is not None:
            split = span[np.argmax(leave[span])]
        else:
            split = span[np.argmax(reach[span])]
        if before is not None:
            share = _progress(leave, split, last)
            if share is not None:
                window = np.arange(start, split + 1)
                weights[window] = share[window]
                references[window] = tracking[window] + anchors[last] - tracking[last]
                contact_frames[window] = last
        if after is not None:
            share = _progress(reach, split, first)
            if share is not None:
                window = np.arange(split, stop)
                use = share[window] > weights[window]
                weights[window[use]] = share[window][use]
                references[window[use]] = (
                    tracking[window[use]] + anchors[first] - tracking[first]
                )
                contact_frames[window[use]] = first
    result = (weights, references, contact_frames)
    return result if return_contact_frames else result[:2]
