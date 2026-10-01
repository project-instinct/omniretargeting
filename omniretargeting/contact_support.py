"""Carry source wrench explanations to any anatomically mapped contact patch.

This bridge has no privileged body-part names. It preserves alternative support
sets; an optional contact is still a following contact, never declared unloaded.
Missing or uncertain mechanics cannot become a negative contact observation.
"""

from dataclasses import replace

import numpy as np

from .contact_mechanics import explain_contacts
from .contact_temporal import explain_support_timeline


class SourceSupportEvidence:
    def __init__(self, explanations, regions, *, temporal=True, candidates=()):
        """regions maps stable candidate names to (source region, surface id)."""
        self.explanations = tuple(explanations)
        self.regions = {str(k): (str(v[0]), int(v[1])) for k, v in regions.items()}
        self.frames = len(self.explanations)

        def snapshot(candidate):
            fields = {}
            for name in (
                "observed_points",
                "surface_points",
                "normals",
                "sliding_velocity",
                "sample_indices",
            ):
                value = getattr(candidate, name)
                if value is not None:
                    fields[name] = np.array(value, copy=True)
                    fields[name].setflags(write=False)
            return replace(candidate, **fields)

        self.candidates = tuple(
            tuple(snapshot(c) for c in frame) for frame in candidates
        )
        self.temporal = (
            explain_support_timeline(self.explanations) if temporal else None
        )
        self.decisions = []
        for frame, explanation in enumerate(self.explanations):
            alternatives = (
                explanation.get("support_sets", explanation["minimal_feasible_sets"])
                if self.temporal is None
                else self.temporal["alternatives"][frame]
            )
            needed = set().union(*map(set, alternatives))
            states = {}
            for name, role in explanation["roles"].items():
                if name not in self.regions:
                    raise ValueError(f"Missing source-region correspondence for {name}")
                key = self.regions[name]
                state = (
                    "support"
                    if name in needed
                    else "follow"
                    if role in {"optional_contact", "geometry_rejected"}
                    else None
                )
                # Multiple patches of one region/surface can jointly support.
                if state == "support" or key not in states:
                    states[key] = state
                elif state is None and states[key] != "support":
                    states[key] = None
            self.decisions.append(states)
        self.diagnostics = {
            "method": "source_net_wrench_feasibility",
            "frames": self.frames,
            "candidate_regions": sorted({r for r, _ in self.regions.values()}),
            "unexplained_frames": [
                f
                for f, x in enumerate(self.explanations)
                if x["full"]["status"] != "feasible"
            ],
            "alternative_support_sets_preserved": True,
            "force_distribution_inferred": False,
            "physical_execution_verified": False,
            "temporal": self.temporal,
        }

    def role(self, region, surface_id, frame):
        return self.decisions[frame].get((str(region), int(surface_id)))


def explain_source_support(
    candidate_frames,
    wrench,
    regions,
    *,
    origin=(0.0, 0.0, 0.0),
    uncertainty=0.0,
    temporal=True,
    method="dual",
    max_candidates=24,
):
    """Use source-only candidates and inertia demand before robot reachability.

    The caller supplies the inertia/friction assumptions and bounded geometric
    observations. No robot configuration is accepted or inspected here.
    """
    frames = tuple(tuple(frame) for frame in candidate_frames)
    wrench = np.asarray(wrench, float)
    if wrench.shape != (len(frames), 6):
        raise ValueError(
            "Source wrench and candidates must share the original timeline"
        )
    uncertainty = np.broadcast_to(np.asarray(uncertainty, float), wrench.shape)
    explanations = [
        explain_contacts(candidates, demand, origin=origin, uncertainty=radius,
                         method=method, max_candidates=max_candidates)
        for candidates, demand, radius in zip(frames, wrench, uncertainty)
    ]
    return SourceSupportEvidence(
        explanations, regions, temporal=temporal, candidates=frames
    )
