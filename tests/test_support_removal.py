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


def _demotion_fixture(pattern, min_demotion_frames=0):
    """A linked seat patch (A1, A2) versus a floor foot X, per-frame wrench types.

    The seat contacts sit at height 0.5 with a 0.15 m gap, so a skin spacing
    of 0.2 m links them into one patch; X lies on the floor on the x-axis at
    (0.4, 0, 0), a separate patch whose tangential force can stay radial and
    balance tau_z. Wrench types (force [0, 0, 10]):
    - "U" (tau=(5, -5)): no feasible support set; an unexplained frame that
      splits the timeline into independent segments.
    - "D" (tau=(0, -0.7)): the seat patch alone is minimal and an X-pair is
      minimal too, so X joins the union while the patch is necessary and
      sufficient: X is demoted.
    - "S" (tau=(0, -2.0)): beyond the seat patch's torque span; X belongs to
      every minimal set, so X is necessary support.
    """
    def seat(name, x):
        return ContactCandidate(name, [[x, 0, 0.5]], [[x, 0, 0.5]], [0, 0, 1], 0.6, 0)

    foot = ContactCandidate("X", [[0.4, 0, 0]], [[0.4, 0, 0]], [0, 0, 1], 0.6, 0)
    wrenches = {
        "U": [0.0, 0, 10, 5.0, -5.0, 0.0],
        "D": [0.0, 0, 10, 0.0, -0.7, 0.0],
        "S": [0.0, 0, 10, 0.0, -2.0, 0.0],
    }
    frames = [[seat("A1", 0.0), seat("A2", 0.15), foot] for _ in pattern]
    wrench = np.asarray([wrenches[p] for p in pattern], float)
    regions = {"A1": ("seat_a", 0), "A2": ("seat_b", 0), "X": ("foot", 1)}
    evidence = explain_source_support(frames, wrench, regions)
    return combine_support_scenarios(
        [evidence], wrench, origin=np.zeros(3), skin_spacing=0.2,
        min_demotion_frames=min_demotion_frames)


def test_short_demotion_run_is_restored_to_support():
    # Frames 1-2 demote X for only two frames, shorter than the source
    # dynamics' derivative window: the run is restored to support.
    pattern = ["U", "D", "D", "U", "U"]
    unguarded = _demotion_fixture(pattern)
    assert [unguarded.role("foot", 1, f) for f in range(5)] == [
        None, "follow", "follow", None, None]
    guarded = _demotion_fixture(pattern, min_demotion_frames=4)
    assert [guarded.role("foot", 1, f) for f in range(5)] == [
        None, "support", "support", None, None]
    assert guarded.explanations[1]["support_sets"] == [["A1", "A2", "X"]]
    assert guarded.explanations[1]["roles"]["X"] == "necessary_support_candidate"
    patches = guarded.diagnostics["support_patches"]
    assert patches["restored_candidate_frames"] == 2
    assert patches["demoted_candidate_frames"] == 0
    assert patches["per_frame"]["1"]["restored"] == ["X"]
    assert patches["per_frame"]["1"]["demoted"] == []
    assert patches["min_demotion_frames"] == 4


def test_demotion_run_at_least_the_window_stays_optional():
    # A four-frame demotion run meets the derivative window: it is resolved
    # evidence, not chatter, and stays optional.
    pattern = ["U", "D", "D", "D", "D"]
    guarded = _demotion_fixture(pattern, min_demotion_frames=4)
    assert [guarded.role("foot", 1, f) for f in range(5)] == [
        None, "follow", "follow", "follow", "follow"]
    patches = guarded.diagnostics["support_patches"]
    assert patches["restored_candidate_frames"] == 0
    assert patches["per_frame"]["1"]["restored"] == []
    assert patches["per_frame"]["1"]["demoted"] == ["X"]


def test_short_support_segment_is_not_reverse_filled():
    # X is necessary in frames 1-2 only. The rule restores demotions; it never
    # releases a support segment, however short, and the S segment keeps X.
    pattern = ["U", "S", "S", "U", "D", "D", "D", "D"]
    guarded = _demotion_fixture(pattern, min_demotion_frames=4)
    assert [guarded.role("foot", 1, f) for f in range(8)] == [
        None, "support", "support", None, "follow", "follow", "follow", "follow"]


def test_explain_and_combine_passes_the_source_derivative_window(monkeypatch):
    from omniretargeting import source_support_evidence as module

    captured = {}
    real = module.combine_support_scenarios

    def spy(scenarios, wrench, **kwargs):
        captured.update(kwargs)
        return real(scenarios, wrench, **kwargs)

    monkeypatch.setattr(module, "combine_support_scenarios", spy)
    foot = ContactCandidate("foot", [[0, 0, 0]], [[0, 0, 0]], [0, 0, 1], 0.6, 0)
    demand = {
        "wrench": np.asarray([[0.0, 0, 10, 0, 0, 0]]),
        "origin": np.zeros(3),
        "derivative_disagreement": np.zeros((1, 6)),
        "derivative_window": 7,
    }
    module._explain_and_combine((((foot,),),), {"foot": ("left_foot", 0)}, demand)
    assert captured["min_demotion_frames"] == 7
