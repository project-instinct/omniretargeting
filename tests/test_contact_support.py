"""Support priority depends on source mechanics, never a body-part allowlist."""

from dataclasses import replace

import numpy as np
import pytest

from omniretargeting.contact_mechanics import ContactCandidate
from omniretargeting.contact_phase import attach_contact_phase
from omniretargeting.contact_support import explain_source_support
from tests.test_contact_phase import case


@pytest.mark.parametrize(
    "part", ["abdomen", "back", "forearm", "knee", "unseen_region"]
)
@pytest.mark.parametrize("mode", ["sticking", "sliding"])
def test_any_source_body_region_can_support_on_a_moving_surface(part, mode):
    _, _, surface, tasks, priority = case(frames=3)
    surface.bindings = [replace(surface.bindings[0], part=part), surface.bindings[1]]
    surface.modes = surface.modes.copy()
    surface.surface_ids = surface.surface_ids.copy()
    surface.modes[:, 0] = mode
    surface.surface_ids[:, 0] = (
        1  # Prescribed moving surface; not static-scene heuristic.
    )
    candidate = ContactCandidate(
        "patch",
        [[0, 0, 0]],
        [[0, 0, 0]],
        [0, 0, 1],
        friction=0.0,
        reconstruction_bound_m=0.0,
        mode=mode,
        sliding_velocity=[1, 0, 0],
    )
    evidence = explain_source_support(
        [[candidate]] * 3,
        np.tile([0.0, 0.0, 10.0, 0.0, 0.0, 0.0], (3, 1)),
        {"patch": (part, 1)},
    )
    assert all(
        x["roles"]["patch"] == "necessary_support_candidate"
        for x in evidence.explanations
    )
    policy = attach_contact_phase(
        surface, None, tasks, priority, support_evidence=evidence
    )
    assert policy.support[:, 0].all()
    assert not policy.follow_surface[:, 0].any()


def test_redundant_support_regions_remain_a_protected_alternative_pool():
    patches = [
        ContactCandidate(n, [[0, 0, 0]], [[0, 0, 0]], [0, 0, 1], 0.0, 0.0)
        for n in ["A", "B"]
    ]
    evidence = explain_source_support(
        [patches],
        np.array([[0.0, 0.0, 10.0, 0.0, 0.0, 0.0]]),
        {"A": ("region_A", 0), "B": ("region_B", 0)},
    )
    assert evidence.role("region_A", 0, 0) == "support"
    assert evidence.role("region_B", 0, 0) == "support"
    assert evidence.explanations[0]["roles"] == {
        "A": "optional_contact",
        "B": "optional_contact",
    }


def test_flight_is_not_support_and_uncertain_evidence_is_not_a_negative_contact():
    candidate = ContactCandidate("patch", [[0, 0, 0]], [[0, 0, 0]], [0, 0, 1], 0.0, 0.0)
    evidence = explain_source_support(
        [[candidate], [candidate]],
        np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 10.0, 0.0, 0.0, 0.0]]),
        {"patch": ("arbitrary_region", 0)},
    )
    assert evidence.role("arbitrary_region", 0, 0) == "follow"
    assert evidence.role("arbitrary_region", 0, 1) is None
    assert evidence.diagnostics["unexplained_frames"] == [1]
