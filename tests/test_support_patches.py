"""Support patches: surface grouping and necessary-patch contraction."""

import numpy as np

from omniretargeting.contact_mechanics import ContactCandidate
from omniretargeting.contact_support import explain_source_support
from omniretargeting.support_patches import support_patches, source_skin_spacing
from omniretargeting.support_sensitivity import combine_support_scenarios


def contact(name, points, friction=0.6, bound=0.03):
    points = np.asarray(points, float).reshape(-1, 3)
    return ContactCandidate(name, points, points.copy(), np.tile([0, 0, 1], (len(points), 1)), friction, bound)


def region_map(names_surfaces):
    return {name: (name.split("@")[0], surface) for name, surface in names_surfaces.items()}


def test_environment_points_within_the_observation_resolution_join_one_patch():
    points = np.array([[0.0, 0, 0], [0.06, 0, 0], [0.12, 0, 0], [0.5, 0, 0]])
    contacts = {f"p{i}@0:sticking": contact(f"p{i}@0:sticking", [p]) for i, p in enumerate(points)}
    surfaces = {name: 0 for name in contacts}
    patches = support_patches(contacts, surfaces, link_distance=0.07)
    assert patches == [["p0@0:sticking", "p1@0:sticking", "p2@0:sticking"], ["p3@0:sticking"]]


def test_candidates_on_different_surfaces_never_share_a_patch():
    near = {"a@0:sticking": contact("a@0:sticking", [[0, 0, 0]]),
            "b@1:sticking": contact("b@1:sticking", [[0.01, 0, 0]])}
    patches = support_patches(near, {"a@0:sticking": 0, "b@1:sticking": 1}, link_distance=0.5)
    assert patches == [["a@0:sticking"], ["b@1:sticking"]]


def test_skin_spacing_measures_the_median_nearest_neighbour_distance():
    points = np.tile(np.array([[0.0, 0, 0], [0.1, 0, 0], [1.0, 0, 0]]), (3, 1, 1))
    assert abs(source_skin_spacing(points) - 0.1) < 1e-9


def test_seated_frame_keeps_the_seat_patch_and_demotes_the_foot_patch():
    # A seated frame: pelvis and thighs rest on a seat (z ~ 0.42, surface 2)
    # and the feet stand on the floor ahead (surface 0). The demand is a
    # downward force over the seat, so the seat patch alone is necessary and
    # sufficient; the feet are optional in the union.
    seat = {"pelvis@2:sticking": contact("pelvis@2:sticking", [[0, 0, 0.42]]),
            "left_thigh@2:sticking": contact("left_thigh@2:sticking", [[0.04, 0, 0.42]]),
            "right_thigh@2:sticking": contact("right_thigh@2:sticking", [[-0.04, 0, 0.42]])}
    feet = {"left_foot@0:sticking": contact("left_foot@0:sticking", [[0.1, 0.2, 0.0]]),
            "right_foot@0:sticking": contact("right_foot@0:sticking", [[-0.1, 0.2, 0.0]])}
    frames = [list(seat.values()) + list(feet.values())]
    names = {**{n: (n.split("@")[0], 2) for n in seat}, **{n: (n.split("@")[0], 0) for n in feet}}
    wrench = np.array([[0.0, 0.0, 10.0, 0.0, 0.0, 0.0]])
    evidence = explain_source_support(frames, wrench, names)
    result = combine_support_scenarios([evidence], wrench, origin=(0, 0, 0), skin_spacing=0.014)
    assert result.role("pelvis", 2, 0) == "support"
    assert result.role("left_thigh", 2, 0) == "support"
    assert result.role("right_thigh", 2, 0) == "support"
    assert result.role("left_foot", 0, 0) == "follow"
    assert result.role("right_foot", 0, 0) == "follow"
    plan = result.diagnostics["support_patches"]["per_frame"]["0"]
    assert plan["necessary"] == [sorted(seat)] or sorted(plan["necessary"][0]) == sorted(seat)


def test_standing_frame_with_one_foot_patch_is_unchanged():
    # Standing: both feet on the floor, one connected patch; the demand is
    # over the feet, both are necessary, nothing is demoted.
    feet = {"left_foot@0:sticking": contact("left_foot@0:sticking", [[0.1, 0.1, 0], [0.1, -0.1, 0]]),
            "right_foot@0:sticking": contact("right_foot@0:sticking", [[-0.1, 0.1, 0], [-0.1, -0.1, 0]])}
    names = {n: (n.split("@")[0], 0) for n in feet}
    wrench = np.array([[0.0, 0.0, 10.0, 0.0, 0.0, 0.0]])
    evidence = explain_source_support([list(feet.values())], wrench, names)
    result = combine_support_scenarios([evidence], wrench, origin=(0, 0, 0), skin_spacing=0.014)
    assert result.role("left_foot", 0, 0) == "support"
    assert result.role("right_foot", 0, 0) == "support"
    assert result.diagnostics["support_patches"]["demoted_candidate_frames"] == 0


def test_two_independently_feasible_patches_keep_the_union_semantics():
    # True ambiguity: either the seat patch or the foot patch alone explains
    # the demand. No patch is necessary; the union protects both.
    seat = {"pelvis@2:sticking": contact("pelvis@2:sticking", [[0, 0, 0.42], [0.08, 0, 0.42]])}
    feet = {"left_foot@0:sticking": contact("left_foot@0:sticking", [[0.1, 0.1, 0], [0.1, -0.1, 0]]),
            "right_foot@0:sticking": contact("right_foot@0:sticking", [[-0.1, 0.1, 0], [-0.1, -0.1, 0]])}
    names = {**{n: (n.split("@")[0], 2) for n in seat}, **{n: (n.split("@")[0], 0) for n in feet}}
    wrench = np.array([[0.0, 0.0, 10.0, 0.0, 0.0, 0.0]])
    evidence = explain_source_support([list(seat.values()) + list(feet.values())], wrench, names)
    result = combine_support_scenarios([evidence], wrench, origin=(0, 0, 0), skin_spacing=0.014)
    assert result.role("pelvis", 2, 0) == "support"
    assert result.role("left_foot", 0, 0) == "support"
    assert result.role("right_foot", 0, 0) == "support"
    plan = result.diagnostics["support_patches"]["per_frame"]["0"]
    assert plan["necessary"] == []
    assert not plan["contracted"]
    assert result.explanations[0]["support_set_semantics"] == (
        "certified_union_of_alternatives_not_a_minimal_set")


def test_patch_link_distance_uses_the_observation_bound_plus_skin_spacing():
    # Two contacts jointly necessary (either alone fails the moment demand),
    # 0.12 m apart on one surface: separate patches with skin spacing 0.014
    # (link 0.03 + 0.014 = 0.044), one patch with spacing 0.1 (link 0.13).
    def pair(name, x):
        return contact(name, [[x, 0.1, 0], [x, -0.1, 0]])

    groups = {"left@0:sticking": pair("left@0:sticking", 0.06),
              "right@0:sticking": pair("right@0:sticking", -0.06)}
    names = {n: (n.split("@")[0], 0) for n in groups}
    wrench = np.array([[0.0, 0.0, 10.0, 0.0, 0.0, 0.0]])
    evidence = explain_source_support([list(groups.values())], wrench, names)

    separated = combine_support_scenarios([evidence], wrench, origin=(0, 0, 0), skin_spacing=0.014)
    plan = separated.diagnostics["support_patches"]["per_frame"]["0"]
    assert plan["patches"] == [["left@0:sticking"], ["right@0:sticking"]]
    assert plan["link_distance_m"] == 0.044

    joined = combine_support_scenarios([evidence], wrench, origin=(0, 0, 0), skin_spacing=0.1)
    plan = joined.diagnostics["support_patches"]["per_frame"]["0"]
    assert plan["patches"] == [["left@0:sticking", "right@0:sticking"]]
