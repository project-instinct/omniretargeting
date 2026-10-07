"""Sticking-contact slip added by a local correction, as a soft objective term.

The contact audit measures the frame-to-frame change of a sticking sample's
error to its target. A correction that lifts a body out of a surface may keep
that change zero by moving the whole contact consistently; one that varies
from frame to frame adds slip. This term prefers the former. It is an
objective term (contact ranks below non-penetration), never a constraint.
"""

from copy import copy

import numpy as np
from scipy import sparse

# Added sticking-error change (metres per frame) weighted like a unit
# deviation of the other objective terms; each frame's samples of one surface
# group share one unit of weight. 1 mm removed the hand-slip increase of
# chair150 but doubled interior-point iterations on sit180's leg windows;
# 3 mm keeps both (out/unified_local_20261006, S1 vs S2).
SLIP_SCALE = 3e-3


def entry_points(limits):
    """World positions of every surface sample at the repair entry."""
    return [limits.kin.geom_points(surface["geoms"], surface["local"]).copy()
            for surface in limits.objective.surfaces]


def slip_term(limits, entry, mask, projection):
    """(S, c): added slip of the sticking pairs touching ``mask`` is S x + c.

    ``limits.kin`` must hold the current poses with Jacobians. ``x`` is the
    projected joint step. Returns None without sticking pairs in reach.
    """
    kin, frames, width = limits.kin, limits.frames, limits.width
    touched = mask[:-1] | mask[1:]
    blocks, offsets = [], []
    for surface, start in zip(limits.objective.surfaces, entry):
        sticking = np.asarray(surface["sticking"], bool) & np.asarray(surface["active"], bool)
        pairs = sticking[:-1] & sticking[1:] & touched[:, None]
        if not pairs.any():
            continue
        points = kin.geom_points(surface["geoms"], surface["local"])
        moved = points - start
        frame, sample = np.nonzero(pairs)
        count = np.bincount(frame, minlength=frames - 1)[frame]
        weight = 1. / (SLIP_SCALE * np.sqrt(count))
        needed = np.unique(np.r_[frame, frame + 1])
        local = copy(kin)
        for name in ("xpos", "xmat", "geom_xpos", "geom_xmat", "jacp", "jacr"):
            setattr(local, name, getattr(kin, name)[needed])
        jacobian = local.point_jacobians(np.asarray(surface["bodies"]), points[needed])
        position = np.searchsorted(needed, frame)
        after, before = jacobian[position + 1, sample], jacobian[position, sample]  # (n, 3, width)
        rows = np.repeat(np.arange(len(frame)) * 3, 3) + np.tile(np.arange(3), len(frame))
        rows = np.repeat(rows, width)
        data = np.concatenate([(after * weight[:, None, None]).reshape(-1),
                               (-before * weight[:, None, None]).reshape(-1)])
        columns_after = ((frame + 1)[:, None, None] * width + np.arange(width)).repeat(3, axis=1)
        columns_before = (frame[:, None, None] * width + np.arange(width)).repeat(3, axis=1)
        columns = np.concatenate([columns_after.reshape(-1), columns_before.reshape(-1)])
        matrix = sparse.csr_matrix((data, (np.concatenate([rows, rows]), columns)),
                                   shape=(3 * len(frame), frames * width))
        blocks.append(matrix)
        offsets.append(((moved[frame + 1, sample] - moved[frame, sample]) * weight[:, None]).ravel())
    if not blocks:
        return None
    matrix = (sparse.vstack(blocks, format="csr") @ projection).tocsr()
    return matrix, np.concatenate(offsets)
