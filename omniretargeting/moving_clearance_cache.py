"""Conservative whole-shell exclusion for bounded, open moving meshes.

The maximum outward hull-plane value is a lower bound on distance to every
original triangle. Subtract a sphere containing both forward samples and the
reverse search hull. A positive remainder beyond the activation band proves
that the original query has no witness and that every open sample is known.
The hull never supplies a near-field distance, sign or collision witness.
"""

from collections import Counter, OrderedDict

import numpy as np


class MovingClearanceCache:
    """Reuse separation bounds in object coordinates, with bounded storage.

    Geometry and sample arrays are immutable within a query owner. Replacing
    them invalidates a bound; references are retained to prevent id reuse.
    The containing sphere permits *any* shell rotation, so only centre motion
    consumes its old margin. Scale changes require a fresh certificate.
    """

    def __init__(self):
        self.entries = OrderedDict()
        self.counts = Counter()
        self._radii = {}

    def separated(self, owner, oid, obj, time, translation, rotation, scale, band, shells):
        query = obj["query"]
        if (not shells or not (obj["observed_open"] or obj["nearly_closed"])
                or query._hull is None):
            return set()
        # SceneQuery hull normals are normalized by Qhull. Include their
        # actual norm and an outward rounding cushion in the certificate.
        hull = query._hull
        lipschitz = max(1., float(np.linalg.norm(hull[:, :3], axis=1).max()))
        separated, pending = set(), []
        for shell in shells:
            gid = shell["geom_id"]
            key = (oid, float(time), gid)
            center = (owner.data.geom_xpos[gid] - translation) @ rotation / scale
            offsets = shell["offsets"]
            radii = self._radii.get(gid)
            if radii is None or radii[0] is not offsets:
                radii = self._radii[gid] = (offsets, float(np.linalg.norm(offsets, axis=1).max()))
            radius = max(shell["radius"], radii[1]) / scale
            identity = (query, hull, query.evidence_points, query.evidence_normals,
                        shell["offsets"], shell["planes"], shell["bounds"])
            previous = self.entries.get(key)
            # Forward samples transform to world before subtracting the
            # object translation. Cover cancellation there as well as the
            # local hull dot products, including large world translations.
            epsilon = 1e-9 * max(1., float(np.linalg.norm(center)), radius,
                float(np.linalg.norm(owner.data.geom_xpos[gid])) / scale,
                float(np.linalg.norm(translation)) / scale, abs(band / scale))
            self.counts["tested"] += 1
            if previous is not None:
                old_identity, pos, lower, old_radius, old_scale = previous
                if (old_scale == scale and old_radius == radius
                        and all(a is b for a, b in zip(identity, old_identity))):
                    moved = lipschitz * np.linalg.norm(center - pos)
                    if lower > band / scale + moved + epsilon:
                        separated.add(gid)
                        self.entries.move_to_end(key)
                        self.counts["motion_bound_hits"] += 1
                        continue
            pending.append((key, identity, center, radius, epsilon))
        if pending:
            margins = query.hull_margin(np.array([p[2] for p in pending]))
            for (key, identity, center, radius, epsilon), margin in zip(pending, margins):
                lower = float(margin - lipschitz * radius)
                if lower > band / scale + epsilon:
                    separated.add(key[2])
                    self.entries[key] = (identity, center.copy(), lower, radius, scale)
                    self.entries.move_to_end(key)
                    self.counts["fresh_bound_hits"] += 1
                else:
                    self.counts["exact_required"] += 1
        while len(self.entries) > 8192:
            self.entries.popitem(last=False)
        return separated


def empty_summary(owner, oid, shell):
    """Preserve the original per-material-sample hull-far known identities."""
    key = oid, shell["geom_id"]
    offsets = shell["offsets"]
    previous = owner._separated_summaries.get(key)
    if previous is None or previous[0] is not offsets:
        keys = tuple((oid, key[1], i) for i in range(len(offsets)))
        summary = dict(known=frozenset(keys), penetrating=frozenset(), unknown=0,
                       failures=[], rows=[], depths=[])
        previous = owner._separated_summaries[key] = (offsets, summary)
    return previous[1]
