"""Export source validity independently of solver/geometry acceptance."""
from __future__ import annotations

import json

import numpy as np

from .quality_policy import reset_states


def export_object_validity(metadata: dict, frames: int) -> dict:
    """Unknown geometry at export; source-invalid frames already forbid reset.

    reset_invalid is a source exclusion mask, not a complete collision audit.
    The independent pair audit later supplies geometry candidate/reset states.
    """
    if "object_valid" not in metadata:
        return {}
    valid = np.asarray(metadata["object_valid"], dtype=bool)
    zeros = np.zeros(frames)
    states = reset_states(zeros, zeros, zeros, np.zeros(frames, bool),
                          np.ones(frames, bool), object_valid=valid)
    result = dict(object_valid=valid, reset_invalid=~valid, reset_states=states)
    if "source_frame_indices" in metadata:
        indices = np.asarray(metadata["source_frame_indices"])
        if indices.shape != (frames,):
            raise ValueError("source_frame_indices must share the output timeline")
        result["source_frame_indices"] = indices
    if "object_gap_report" in metadata:
        result["object_gap_report_json"] = np.asarray(json.dumps(metadata["object_gap_report"]))
    return result
