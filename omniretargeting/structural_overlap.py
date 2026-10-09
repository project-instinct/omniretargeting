"""Measured model defects, separate from trajectory quality and reset policy.

The compiled collision model is unchanged: training still needs the proposed
MJCF excludes. Classification uses deterministic model samples, never the clip.
"""

import numpy as np


class StructuralOverlap:
    def __init__(self, query, frames, labels=None):
        self.model = getattr(query, "model", None)
        self.evidence = getattr(query, "structural_evidence", {})
        self.frames = frames
        self.labels = np.arange(frames) if labels is None else np.asarray(labels)
        self.depths = {pair: np.zeros(frames) for pair in self.evidence}
        self.unknown = np.zeros(frames, bool)

    def collect(self, rows, frame):
        """Retain measurement of model defects; return trajectory rows only."""
        result = []
        for row in rows:
            pair = tuple(sorted(map(int, row["pair"])))
            if pair in self.depths:
                self.depths[pair][frame] = max(self.depths[pair][frame], -float(row["clearance"]))
            else:
                result.append(row)
        return result

    def describe(self):
        pairs, excludes = {}, []
        for (a, b), depth in self.depths.items():
            bodies = [self.model.body(int(self.model.geom_bodyid[g])).name for g in (a, b)]
            key = f"self:{a}:{b}"
            pairs[key] = dict(geom_ids=[a, b], bodies=bodies, depth_m=depth.tolist(),
                frames=self.labels.tolist(), unknown=self.unknown.tolist(), max_depth_m=float(depth.max(initial=0)),
                always_overlap_evidence=self.evidence[a, b])
            excludes.append(dict(body1=bodies[0], body2=bodies[1], pair=key,
                xml=f'<exclude body1="{bodies[0]}" body2="{bodies[1]}"/>'))
        return dict(classification="model_structural_overlap", pairs=pairs, pair_count=len(pairs),
            trajectory_defect=False, reset_invalid=False, measured=True,
            training_mjcf_recommendation=dict(action="add_contact_excludes", excludes=excludes,
                applied=False, reason="These pairs remain active in the compiled model and can produce persistent training contact forces."))
