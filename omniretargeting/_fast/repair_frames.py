"""Frame scope of one repair group: its penetrating frames plus a temporal halo.

A group's subproblem only moves the frames where one of its collision keys
penetrates, widened by a halo long enough to ramp the correction in and out
within the per-joint continuity budgets. Every other frame keeps its pose, so
its constraints become constants (dropped only when already satisfied) and its
original-geometry results come from the exact per-pose frame cache. Guards,
budgets and acceptance are unchanged; only the variable set shrinks.
"""

import numpy as np
from scipy import sparse

# Halo on each side of the penetrating frames, in seconds of motion.
HALO_SECONDS = 0.5


def halo_frames(framerate):
    return max(4, int(np.ceil(HALO_SECONDS * float(framerate))))


def group_frames(measured, model, group, frames, halo, *, activation=None):
    """Boolean frame mask of the group's penetrating keys, dilated by ``halo``."""
    hit = np.zeros(frames, bool)
    if activation is not None:
        activation = float(activation)
    else:
        activation = 0.
    for key, depth in measured["depths"].items():
        if depth <= activation:
            continue
        geoms = key[2:] if key[1] == "self" else key[-1:]
        if group is None or any(model.body_weldid[model.geom_bodyid[g]] == group for g in geoms):
            hit[key[0]] = True
    if not hit.any():
        return None
    padded = np.convolve(np.r_[np.zeros(halo), hit, np.zeros(halo)], np.ones(2 * halo + 1), "valid")
    return padded[:frames] > 0


def restrict(projection, mask):
    """Keep the projection columns of the masked frames (frame-major columns)."""
    if mask is None or mask.all():
        return projection
    per_frame, remainder = divmod(projection.shape[1], len(mask))
    assert not remainder, "projection columns must be whole frames"
    keep = np.flatnonzero(np.repeat(mask, per_frame))
    return sparse.csc_matrix(projection)[:, keep]
