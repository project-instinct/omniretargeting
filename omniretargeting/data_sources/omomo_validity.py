"""OMOMO-only empty object-frame handling, before body/contact preparation.

Original valid rows are never rewritten. Ranges use inclusive zero-based source
indices; retained_stop is exclusive. Short internal gaps are bridged before the
longest usable interval is selected (ties keep the earliest interval).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


# Explicit OMOMO time fields: shape-only slicing could truncate static betas or
# rest_offsets when their dimensions happen to equal the sequence length.
_TIME_FIELDS = (
    "trans", "root_orient", "pose_body", "pose_hand", "obj_scale", "obj_trans",
    "obj_rot", "obj_com_pos", "frame_ids",
)


def frame_ranges(mask: np.ndarray) -> list[list[int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(int)
    return [[int(a), int(b - 1)] for a, b in zip(
        np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1)
    )]


def object_frame_validity(sequence: dict) -> np.ndarray:
    """Zero translation alone is valid: the official scale is per frame."""
    scales = np.asarray(sequence["obj_scale"])
    if scales.ndim != 1 or len(scales) == 0:
        raise ValueError("OMOMO obj_scale must have nonempty shape (T,)")
    return np.isfinite(scales) & (scales > 0)


def object_gap_plan(sequence: dict, framerate: float = 30.0) -> tuple[np.ndarray, dict]:
    if not np.isfinite(framerate) or framerate <= 0:
        raise ValueError("OMOMO framerate must be finite and positive")
    valid = object_frame_validity(sequence)
    usable = valid.copy()
    gaps = []
    for first, last in frame_ranges(~valid):
        count = last - first + 1
        bounded = first > 0 and last < len(valid) - 1
        bridgeable = bounded and count / framerate <= 0.5
        if bridgeable:
            usable[first:last + 1] = True
        gaps.append(dict(first=first, last=last, frames=count,
                         seconds=count / framerate, bounded=bounded,
                         bridgeable=bridgeable))
    intervals = frame_ranges(usable)
    if not intervals:
        raise ValueError(f"OMOMO {sequence.get('seq_name', '<unknown>')}: no valid object frames")
    first, last = max(intervals, key=lambda span: span[1] - span[0])
    retained = np.zeros(len(valid), dtype=bool)
    retained[first:last + 1] = True
    filled = retained & ~valid
    for gap in gaps:
        gap["action"] = "interpolate" if filled[gap["first"]] else "discard"
    report = dict(
        seq_name=str(sequence.get("seq_name", "<unknown>")),
        framerate=float(framerate), original_frames=len(valid),
        invalid_frames=np.flatnonzero(~valid).tolist(), invalid_frame_count=int((~valid).sum()),
        gaps=gaps, interpolated_frames=np.flatnonzero(filled).tolist(),
        interpolated_frame_count=int(filled.sum()), retained_start=first,
        retained_stop=last + 1, retained_frames=last - first + 1,
        truncated_frames=len(valid) - (last - first + 1),
        truncated_invalid_frame_count=int((~valid & ~retained).sum()),
        truncated_valid_frame_count=int((valid & ~retained).sum()),
        truncated_ranges=frame_ranges(~retained),
    )
    return valid, report


def prepare_object_frames(sequence: dict, framerate: float = 30.0) -> tuple[dict, np.ndarray, dict]:
    """Repair only retained empty poses and synchronously slice raw body rows."""
    valid, report = object_gap_plan(sequence, framerate)
    first, stop = report["retained_start"], report["retained_stop"]
    if not report["invalid_frame_count"]:
        return sequence, valid, report
    repaired = dict(sequence)
    for key in _TIME_FIELDS:
        if key in sequence:
            rows = np.asarray(sequence[key])
            if len(rows) != len(valid):
                raise ValueError(f"OMOMO {key} must share the obj_scale timeline")
            repaired[key] = rows[first:stop].copy()
    for gap in report["gaps"]:
        if gap["action"] != "interpolate":
            continue
        a, b = gap["first"] - 1, gap["last"] + 1
        ids = np.arange(a + 1, b)
        alpha = (ids - a) / (b - a)
        for key in ("obj_scale", "obj_trans", "obj_com_pos"):
            if key not in sequence:
                continue
            rows = np.asarray(sequence[key])
            weight = alpha.reshape((-1,) + (1,) * (rows.ndim - 1))
            repaired[key][ids - first] = (1 - weight) * rows[a] + weight * rows[b]
        rotations = Rotation.from_matrix(np.asarray(sequence["obj_rot"])[[a, b]])
        repaired["obj_rot"][ids - first] = Slerp([a, b], rotations)(ids).as_matrix()
    # The SMPL-X raw adapter treats these IDs as provenance, not missing time to
    # synthesize; all downstream surfaces/soles follow the already sliced body.
    if report["truncated_frames"] and "frame_ids" not in repaired:
        repaired["frame_ids"] = np.arange(first, stop)
    return repaired, valid[first:stop], report
