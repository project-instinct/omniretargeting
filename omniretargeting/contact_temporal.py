"""Keep all least-switching paths through mechanically feasible support sets.

This is a continuity prior, not a load estimator or a minimum-duration filter.
An infeasible state is never restored to smooth an event. Unknown mechanics
splits the timeline rather than manufacturing a contact through the gap.
"""

import itertools

import numpy as np


def explain_support_timeline(explanations):
    states = [
        [frozenset(s) for s in row.get("support_sets", row["minimal_feasible_sets"])] for row in explanations
    ]
    alternatives = [[] for _ in states]
    segments = []
    begin = 0
    while begin < len(states):
        if not states[begin]:
            begin += 1
            continue
        end = begin + 1
        while end < len(states) and states[end]:
            end += 1
        current = states[begin:end]
        edges = [
            np.array([[len(a ^ b) for b in right] for a in left], dtype=int)
            for left, right in itertools.pairwise(current)
        ]
        forward = [np.zeros(len(current[0]), dtype=int)]
        for edge in edges:
            forward.append((forward[-1][:, None] + edge).min(axis=0))
        backward = [None] * len(current)
        backward[-1] = np.zeros(len(current[-1]), dtype=int)
        for i in range(len(edges) - 1, -1, -1):
            backward[i] = (edges[i] + backward[i + 1][None]).min(axis=1)
        optimum = int(forward[-1].min())
        for i, choices in enumerate(current):
            alternatives[begin + i] = [
                sorted(state)
                for j, state in enumerate(choices)
                if forward[i][j] + backward[i][j] == optimum
            ]
        segments.append({"start": begin, "stop": end, "minimum_switch_events": optimum})
        begin = end
    return {
        "method": "minimum_contact_changes_over_feasible_sets",
        "alternatives": alternatives,
        "segments": segments,
        "unresolved_frames": [i for i, row in enumerate(states) if not row],
        "all_tied_paths_preserved": True,
        "short_events_filtered": False,
        "force_distribution_inferred": False,
    }
