import numpy as np
from omniretargeting.contact_mechanics import ContactCandidate
from omniretargeting.support_removal import removal_evidence
from omniretargeting.contact_support import explain_source_support
from omniretargeting.support_sensitivity import combine_support_scenarios


def patch(name, friction=0.5):
    return ContactCandidate(name, [[0, 0, 0]], [[0, 0, 0]], [0, 0, 1], friction, 0)


def test_redundant_individual_contacts_do_not_remove_the_needed_group():
    contacts = [patch("a"), patch("b")]
    result = removal_evidence(
        contacts, [0, 0, 10, 0, 0, 0], {"whole_surface": ["a", "b"], "one_patch": ["a"]}
    )
    assert result["groups"]["whole_surface"]["role"] == "necessary_support_group"
    assert result["groups"]["one_patch"]["role"] == "optional_explanation"


def test_uncertain_flight_demand_does_not_prove_support_necessity():
    result = removal_evidence(
        [patch("a")], [0, 0, 1, 0, 0, 0], {"all": ["a"]}, uncertainty=[0, 0, 2, 0, 0, 0]
    )
    assert result["groups"]["all"]["role"] == "unexplained"


def test_scenario_consensus_requires_box_feasibility_and_retains_alternatives():
    wrench = np.array([[0, 0, 10, 0, 0, 0], [4, 0, 10, 0, 0, 0]])
    scenarios = []
    for friction in [0.3, 0.8]:
        cs = [patch("a", friction), patch("b", friction)]
        scenarios.append(
            explain_source_support(
                [cs, cs],
                wrench,
                {"a": ("new_region", 0), "b": ("other", 0)},
                method="dual",
            )
        )
    result = combine_support_scenarios(scenarios, wrench, origin=np.zeros(3))
    assert result.role("new_region", 0, 0) == "support"
    assert result.role("other", 0, 0) == "support"
    assert result.role("new_region", 0, 1) is None
    assert result.explanations[0]["support_sets"] == [["a", "b"]]
    assert not result.diagnostics["support_sets_are_minimal"]


def test_consensus_does_not_restore_a_passing_contact_removed_by_temporal_evidence():
    wrench = np.tile([0.0, 0.0, 10.0, 0.0, 0.0, 0.0], (3, 1))
    stable, passing = patch("stable"), patch("passing")
    evidence = explain_source_support(
        [[stable], [stable, passing], [stable]],
        wrench,
        {"stable": ("region_a", 0), "passing": ("region_b", 1)},
    )
    assert evidence.role("region_b", 1, 1) == "follow"
    result = combine_support_scenarios([evidence], wrench, origin=np.zeros(3))
    assert result.explanations[1]["support_sets"] == [["stable"]]
    assert result.role("region_a", 0, 1) == "support"
    assert result.role("region_b", 1, 1) != "support"


def test_consensus_marks_an_unneeded_touch_optional_not_unexplained():
    # Two feet carry the body; a hand touches a wall with no load needed.
    def foot(name, x):
        return ContactCandidate(name, [[x, 0, 0], [x, 0.1, 0]], [[x, 0, 0], [x, 0.1, 0]], [0, 0, 1], 0.6, 0)

    touch = ContactCandidate("hand", [[0.1, 0.3, 1.0]], [[0.1, 0.3, 1.0]], [0, -1, 0], 0.6, 0)
    wrench = np.tile([0.0, 0.0, 10.0, 0.5, -1.0, 0.0], (2, 1))
    frames = [[foot("left", 0.0), foot("right", 0.2), touch], [foot("left", 0.0), foot("right", 0.2)]]
    regions = {"left": ("left_foot", 0), "right": ("right_foot", 0), "hand": ("left_hand", 0)}
    evidence = explain_source_support(frames, wrench, regions)
    result = combine_support_scenarios([evidence], wrench, origin=np.zeros(3))
    assert result.role("left_foot", 0, 0) == "support"
    assert result.role("right_foot", 0, 0) == "support"
    assert result.role("left_hand", 0, 0) == "follow"
    # An identity absent from a frame gets no role there.
    assert "hand" not in result.explanations[1]["roles"]


def test_absent_mode_does_not_override_a_present_contact():
    stuck = ContactCandidate("hand@0:sticking", [[0.1, 0.3, 1.0]], [[0.1, 0.3, 1.0]], [0, -1, 0], 0.6, 0)
    feet = ContactCandidate("feet@0:sticking", [[0, 0, 0], [0.2, 0, 0], [0.1, 0.2, 0]],
                            [[0, 0, 0], [0.2, 0, 0], [0.1, 0.2, 0]], [0, 0, 1], 0.6, 0)
    wrench = np.tile([0.0, 0.0, 10.0, 1.0, -1.0, 0.0], (1, 1))
    regions = {"hand@0:sticking": ("left_hand", 0), "hand@0:sliding": ("left_hand", 0),
               "feet@0:sticking": ("feet", 0)}
    evidence = explain_source_support([[feet, stuck]], wrench, regions)
    result = combine_support_scenarios([evidence], wrench, origin=np.zeros(3))
    assert result.role("left_hand", 0, 0) == "follow"
