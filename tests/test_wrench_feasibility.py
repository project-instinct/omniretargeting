import numpy as np
import pytest

from omniretargeting.contact_mechanics import (
    ContactCandidate,
    WrenchCone,
    explain_contacts,
)
from omniretargeting.wrench_feasibility import DualWrenchCone


@pytest.mark.parametrize("seed", range(4))
def test_dual_separator_matches_converted_cone_without_force_outputs(seed):
    rng = np.random.default_rng(seed)
    candidates = []
    for i in range(3):
        p, n = rng.normal(size=(2, 3))
        candidates.append(ContactCandidate(str(i), p[None], p[None], n, 0.5, 0))
    dual, cone = DualWrenchCone(candidates), WrenchCone(candidates)
    rays = np.concatenate([c.rays(np.zeros(3), 1) for c in candidates])
    for wrench in [rng.uniform(size=len(rays)) @ rays, *rng.normal(size=(10, 6))]:
        assert dual.check(wrench)["status"] == cone.check(wrench)["status"]


def test_pruned_subset_search_retains_all_redundant_supports_and_flight():
    c = [
        ContactCandidate(str(i), np.zeros((1, 3)), np.zeros((1, 3)), [0, 0, 1], 0.5, 0)
        for i in range(4)
    ]
    for wrench in [[0, 0, 100, 0, 0, 0], np.zeros(6), [0, 0, -10, 0, 0, 0]]:
        reference = explain_contacts(c, wrench)
        actual = explain_contacts(c, wrench, method="dual")
        assert actual["roles"] == reference["roles"]
        assert sorted(actual["minimal_feasible_sets"]) == sorted(
            reference["minimal_feasible_sets"]
        )


def test_uncertainty_never_becomes_a_false_impossibility():
    c = ContactCandidate("point", np.zeros((1, 3)), np.zeros((1, 3)), [0, 0, 1], 0.5, 0)
    dual = DualWrenchCone([c])
    assert (
        dual.check([51, 0, 100, 0, 0, 0], [2, 0, 0, 0, 0, 0])["status"] == "uncertain"
    )
    assert (
        dual.check([60, 0, 100, 0, 0, 0], [2, 0, 0, 0, 0, 0])["status"] == "infeasible"
    )
    assert dual.check([0, 0, 100, 0, 0, 0], [2, 2, 0, 0, 0, 0])["status"] == "feasible"


def test_minimal_support_search_does_not_enumerate_irrelevant_subsets():
    from omniretargeting.contact_mechanics import _minimal_dual_subsets

    required = {1, 5, 11, 18}
    checked = set()

    def check(chosen):
        checked.add(chosen)
        return {"status": "feasible" if required <= set(chosen) else "infeasible"}

    assert _minimal_dual_subsets(range(24), check) == [tuple(sorted(required))]
    assert len(checked) < 60


def test_minimal_support_search_retains_all_overlapping_alternatives():
    from itertools import combinations

    from omniretargeting.contact_mechanics import _minimal_dual_subsets

    expected = [{0, 2}, {1, 2}, {0, 3, 4}, {1, 3, 5}, {4, 5}]

    def check(chosen):
        feasible = any(s <= set(chosen) for s in expected)
        return {"status": "feasible" if feasible else "infeasible"}

    actual = _minimal_dual_subsets(range(7), check)
    exhaustive = [
        c
        for n in range(8)
        for c in combinations(range(7), n)
        if check(c)["status"] == "feasible"
        and all(check(c[:i] + c[i + 1 :])["status"] != "feasible" for i in range(n))
    ]
    assert sorted(actual) == sorted(exhaustive)
