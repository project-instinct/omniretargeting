"""Decision-only mechanics retains the full explanation's support semantics."""

import numpy as np
import pytest

import omniretargeting.contact_mechanics as mechanics
from omniretargeting.contact_mechanics import ContactCandidate, explain_contacts
from omniretargeting.contact_support import SourceSupportEvidence
from omniretargeting.object_support_evidence import _decide_task


def contact(name, x, normal=(0., 0., 1.)):
    point = np.array([[x, 0., 0.]])
    return ContactCandidate(name, point, point, np.asarray(normal), .6, .01)


def test_unique_previous_set_is_certified_without_enumeration(monkeypatch):
    contacts = [contact("up", 0.), contact("down", 0., (0., 0., -1.))]
    wrench = [0., 0., 1., 0., 0., 0.]
    expected = explain_contacts(contacts, wrench, method="dual")

    def forbidden(*args, **kwargs):
        raise AssertionError("A certified unique set should skip enumeration")

    monkeypatch.setattr(mechanics, "_minimal_dual_subsets", forbidden)
    result = explain_contacts(contacts, wrench, method="dual", detail="source",
                              prior_minimal=[["up"]])
    assert result["minimal_feasible_sets"] == expected["minimal_feasible_sets"]
    assert result["roles"] == expected["roles"]


def test_previous_set_that_loses_feasibility_falls_back_to_current_enumeration(monkeypatch):
    contacts = [contact("up", 0.), contact("down", 0., (0., 0., -1.))]
    downward = [0., 0., -1., 0., 0., 0.]
    expected = explain_contacts(contacts, downward, method="dual")
    original = mechanics._minimal_dual_subsets
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(mechanics, "_minimal_dual_subsets", counted)
    result = explain_contacts(contacts, downward, method="dual", detail="source",
                              prior_minimal=[["up"]])
    assert calls
    assert result["minimal_feasible_sets"] == expected["minimal_feasible_sets"]
    assert result["roles"] == expected["roles"]


def test_uncertain_full_set_cannot_have_feasible_subset(monkeypatch):
    contacts = [contact("up", 0.), contact("also_up", 0.)]
    wrench = [0., 0., 1., 0., 0., 0.]
    uncertainty = [0., 0., 2., 0., 0., 0.]
    expected = explain_contacts(contacts, wrench, uncertainty=uncertainty, method="dual")
    assert expected["full"]["status"] == "uncertain"
    assert expected["minimal_feasible_sets"] == []

    def forbidden(*args, **kwargs):
        raise AssertionError("An uncertain full set cannot contain a feasible subset")

    monkeypatch.setattr(mechanics, "_minimal_dual_subsets", forbidden)
    result = explain_contacts(contacts, wrench, uncertainty=uncertainty,
                              method="dual", detail="source")
    assert result["minimal_feasible_sets"] == expected["minimal_feasible_sets"]
    assert result["roles"] == expected["roles"]


@pytest.mark.parametrize("seed", range(8))
def test_temporal_decisions_and_sets_equal_full_explanations(seed):
    rng = np.random.default_rng(seed)
    contacts = [contact("left", -.2), contact("right", .2),
                contact("reverse", 0., (0., 0., -1.))]
    wrenches = np.tile([0., 0., 1., 0., 0., 0.], (8, 1))
    wrenches[2:4, 4] = -.25
    wrenches[4:6, 4] = .25
    wrenches[6:, 2] = -1.
    wrenches += rng.normal(scale=1e-5, size=wrenches.shape)
    full, compact, prior = [], [], ()
    for wrench in wrenches:
        full.append(explain_contacts(contacts, wrench, method="dual"))
        row = explain_contacts(contacts, wrench, method="dual", detail="source",
                               prior_minimal=prior)
        compact.append(row)
        prior = row["minimal_feasible_sets"]
        assert row["minimal_feasible_sets"] == full[-1]["minimal_feasible_sets"]
        assert row["roles"] == full[-1]["roles"]
        assert row["full"]["status"] == full[-1]["full"]["status"]
    regions = {name: (name, 0) for name in ("left", "right", "reverse")}
    detailed = SourceSupportEvidence(full, regions)
    reduced = SourceSupportEvidence(compact, regions)
    assert detailed.decisions == reduced.decisions
    assert detailed.temporal["alternatives"] == reduced.temporal["alternatives"]
    assert detailed.diagnostics["unexplained_frames"] == reduced.diagnostics["unexplained_frames"]


def test_object_system_warm_jobs_keep_decisions_and_counters():
    names = {"left": ("left", 1), "right": ("right", 1)}
    skin = [contact("left", -.2), contact("right", .2)]
    wrenches = [[0., 0., 1., 0., 0., moment] for moment in (0., 0., -.3, .3, 0.)] * 7
    warm = [(f, (1,), wrench, np.zeros(3), [], skin, names)
            for f, wrench in enumerate(wrenches)]
    cold = [(wrench, np.zeros(3), [], skin, names) for wrench in wrenches]
    assert list(map(_decide_task, warm)) == list(map(_decide_task, cold))


def test_many_previous_alternatives_do_not_hide_current_minimal_sets():
    contacts = [contact(str(i), 0.) for i in range(10)]
    demand = [0., 0., 1., 0., 0., 0.]
    full = explain_contacts(contacts, demand, method="dual", max_candidates=10)
    assert len(full["minimal_feasible_sets"]) == 10
    compact = explain_contacts(contacts, demand, method="dual", detail="source",
                               prior_minimal=full["minimal_feasible_sets"], max_candidates=10)
    assert compact["minimal_feasible_sets"] == full["minimal_feasible_sets"]
    assert compact["roles"] == full["roles"]
