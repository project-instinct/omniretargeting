"""Source dynamics may justify bounded reconstruction gaps, never remote forces."""

from dataclasses import replace

import numpy as np
import pytest

from omniretargeting.contact_mechanics import ContactCandidate
from omniretargeting.contact_support import explain_source_support
from omniretargeting.support_observation import apply_support_observations
from omniretargeting.surface_contact import SurfaceContactTasks
from tests.test_surface_contact import contact_case


def fixture(bound=0.06, demand=10.0):
    points = np.array([[[0.0, 0.0, 0.05]]])
    candidate = ContactCandidate(
        "patch",
        points[0],
        [[0, 0, 0]],
        [0, 0, 1],
        0.0,
        bound,
        sample_indices=np.array([0]),
    )
    evidence = explain_source_support(
        [[candidate]], np.array([[0, 0, demand, 0, 0, 0]]), {"patch": ("region", 0)}
    )
    observation = {
        "active": np.zeros((1, 1), bool),
        "environment": np.zeros_like(points),
        "normals": np.tile([0.0, 0.0, 1.0], (1, 1, 1)),
        "surface_ids": np.zeros((1, 1), int),
        "modes": np.full((1, 1), "none", dtype="U8"),
        "confidence": np.zeros((1, 1)),
    }
    return points, observation, evidence


def test_gap_confirmation_separates_raw_skin_from_real_surface_target():
    points, observation, evidence = fixture()
    updated, targets = apply_support_observations(
        observation, points, np.ones((1, 1), bool), evidence
    )
    assert updated["active"].all()
    np.testing.assert_array_equal(targets, 0)
    np.testing.assert_array_equal(points, [[[0, 0, 0.05]]])
    assert not observation["active"].any()
    model, _, surface, _ = contact_case(frames=1)
    tasks = SurfaceContactTasks(
        model,
        surface.bindings,
        updated["environment"],
        points - updated["environment"],
        updated["active"],
        updated["normals"],
        contact_targets=targets,
    )
    assert tasks.source_vectors[0, 0, 2] == 0.05
    assert tasks.targets[0, 0, 2] == 0
    assert tasks.export()["surface_contact_source_vectors_w"][0, 0, 2] == 0.05


@pytest.mark.parametrize("bound,demand", [(0.04, 10), (0.06, 0)])
def test_impossible_gap_or_flight_does_not_create_support(bound, demand):
    points, observation, evidence = fixture(bound, demand)
    updated, targets = apply_support_observations(
        observation, points, np.ones((1, 1), bool), evidence
    )
    assert not updated["active"].any()
    np.testing.assert_array_equal(targets, points)


def test_unmatched_identity_and_mutation_cannot_retarget_the_source_evidence():
    points, observation, evidence = fixture()
    with pytest.raises(ValueError, match="immutable source"):
        apply_support_observations(
            observation, points + 0.01, np.ones((1, 1), bool), evidence
        )
    with pytest.raises(ValueError):
        evidence.candidates[0][0].surface_points[:] = 5
    evidence.candidates = ((replace(evidence.candidates[0][0], sample_indices=None),),)
    updated, _ = apply_support_observations(
        observation, points, np.ones((1, 1), bool), evidence
    )
    assert updated["support_inference"]["unmapped"] == [
        {"frame": 0, "candidate": "patch"}
    ]


def test_surface_builder_uses_necessary_support_beyond_geometric_touch_threshold(
    monkeypatch,
):
    from omniretargeting import surface_contact

    points, _, evidence = fixture()
    model, _, old_task, scene = contact_case(frames=1)
    vertices = np.array([[0.0, 0.0, 0.05], [0.1, 0.0, 0.05], [0.0, 0.1, 0.05]])
    source = {
        "full_body": {
            "reference_vertices": vertices,
            "faces": [[0, 1, 2]],
            "face_ids": [0],
            "barycentric": [[1.0, 0.0, 0.0]],
            "triangles": vertices[None, None],
        }
    }
    monkeypatch.setattr(
        "omniretargeting.surface_motion.bind_whole_body",
        lambda *a, **k: (old_task.bindings, np.ones(1, bool), {}),
    )
    # Use an actual box surface at z=0; raw skin is five centimetres above it.
    scene.apply_translation([0, 0, -0.4])
    task, report = surface_contact.build_surface_contact_tasks(
        model, source, model.qpos0, scene, support_evidence=evidence
    )
    assert task is not None
    assert task.active.all()
    assert report["support_inference"]["confirmed"][0]["candidate"] == "patch"
    np.testing.assert_array_equal(task.targets, 0)
    np.testing.assert_array_equal(task.source_vectors, points)
