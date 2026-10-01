"""Temporal preference cannot override source mechanics or fabricate certainty."""

import itertools

from omniretargeting.contact_temporal import explain_support_timeline


def explain(states):
    return explain_support_timeline([{"minimal_feasible_sets": x} for x in states])


def test_preserve_brief_required_support_and_flight_without_delay():
    states = [[["foot"]], [["hand"]], [[]], [["foot"]]]
    assert explain(states)["alternatives"] == states


def test_redundant_support_remains_ambiguous_and_handoff_time_is_not_invented():
    states = [[["A"]], [["A"], ["B"]], [["A"], ["B"]], [["B"]]]
    result = explain(states)
    assert result["alternatives"] == states
    assert result["segments"][0]["minimum_switch_events"] == 2
    assert explain([[["A"], ["B"]]] * 3)["alternatives"] == [[["A"], ["B"]]] * 3


def test_no_chatter_inside_ambiguity_and_no_contact_across_missing_evidence():
    states = [[["A"]], [["A"], ["B"]], [["A"]], [], [["B"]]]
    result = explain(states)
    assert result["alternatives"] == [[["A"]]] * 3 + [[], [["B"]]]
    assert result["unresolved_frames"] == [3]
    assert len(result["segments"]) == 2


def test_dynamic_program_matches_exhaustive_paths_and_is_resampling_invariant():
    states = [[["A"], ["B"]], [["B"], ["C"]], [["A", "C"]], [[]]]
    paths = list(itertools.product(*states))
    costs = [sum(len(set(a) ^ set(b)) for a, b in itertools.pairwise(p)) for p in paths]
    best = [p for p, c in zip(paths, costs) if c == min(costs)]
    expected = [sorted({tuple(p[f]) for p in best}) for f in range(len(states))]
    actual = explain(states)["alternatives"]
    assert [sorted(map(tuple, x)) for x in actual] == expected
    repeated = [frame for frame in states for _ in range(2)]
    assert explain(repeated)["alternatives"][::2] == actual
