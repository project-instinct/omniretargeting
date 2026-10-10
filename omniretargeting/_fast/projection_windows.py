"""Temporal working sets for the long-clip direct whole-body QP.

The body scope stays full. Variables around current policy defects include
12 neighboring frames; gaps shorter than the high-pass stencil are merged.
All proposed windows still undergo one global nonlinear acceptance check.
"""
import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy import sparse

from . import deadline_state


def focus_frames(quality, frames):
    mask = np.zeros(frames, bool)
    for pair in quality['pairs'].values():
        for event in pair.get('events', ()):
            mask |= event['frames']
    return maximum_filter1d(mask, 25, mode='constant') if mask.any() else mask


def intervals(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False])).reshape(-1, 2)
    result = []
    for start, stop in edges:
        if result and start - result[-1][1] <= 16:
            result[-1][1] = int(stop)
        else:
            result.append([int(start), int(stop)])
    return result


def solve_windows(projection, hessian, rhs, matrix, lower, upper, solve):
    """Exact subproblems of the frozen linearization, not limb restrictions.

    Non-moving rows impose no new condition on the already accepted motion.
    In particular the numerical inward reserve of the force linearization is
    not a request to change unrelated frames. Coupling stencils do not cross
    the merged windows; global exact checks remain authoritative.
    """
    step = np.zeros((projection.frames, projection.width))
    records = []
    for start, stop in intervals(projection.focus):
        deadline_state.check('projection_temporal_window')
        cols = np.arange(start * projection.width, stop * projection.width)
        part = sparse.csr_matrix(matrix[:, cols])
        keep = np.diff(part.indptr) > 0
        direction, record = solve(hessian[cols][:, cols], rhs[cols], part[keep].tocsc(),
                                  lower[keep], upper[keep])
        record.update(frames=[start, stop], variables=len(cols))
        records.append(record)
        if direction is not None:
            step[start:stop] = direction.reshape(stop - start, projection.width) * projection.scales
    projection.step_status = dict(solver='clarabel_temporal_windows', windows=records,
                                 seconds=sum(r['seconds'] for r in records))
    return step if np.any(step) else None
