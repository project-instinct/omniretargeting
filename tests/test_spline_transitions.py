"""Semantic, body-level transition allocation for adaptive spline knots."""

from types import SimpleNamespace

import numpy as np

from omniretargeting._fast.spline_solver import transition_times
from omniretargeting._fast.spline_transitions import semantic_transition_times


def _surface_objective():
    frames, framerate = 10, 10
    active = np.zeros((frames, 4), bool)
    modes = np.full((frames, 4), "none", object)
    surface_ids = np.zeros((frames, 4), int)
    roles = np.full((frames, 4), "none", object)

    # Two left-foot samples exchange while the same welded body remains in
    # contact with the same mode, material and role.
    active[1:4, 0] = [True, True, False]
    active[1:, 1] = [False, True, True, True, True, True,
                     False, False, False]
    modes[:, :2][active[:, :2]] = "sticking"
    modes[4:, 1] = "sliding"
    surface_ids[:, :2][active[:, :2]] = 10
    surface_ids[5:7, 1] = 11
    roles[:, :2][active[:, :2]] = "support"

    # A one-binding right-foot onset is a real transition because that body
    # was not in contact in the preceding frame.
    active[6:10, 2] = True
    active[8:10, 3] = True
    modes[:, 2:][active[:, 2:]] = "sticking"
    surface_ids[:, 2:][active[:, 2:]] = 20
    roles[:, 2:][active[:, 2:]] = "support"
    roles[8:10, 2] = "control"

    task = SimpleNamespace(
        effector_names=np.array(["left_foot", "left_foot",
                                 "right_foot", "right_foot"]),
        modes=modes,
        surface_ids=surface_ids,
    )
    policy = SimpleNamespace(
        surface=task,
        surface_roles={
            "support": roles == "support",
            "control": roles == "control",
        },
    )
    surface = {
        "task": task,
        "active": active,
        "bodies": np.array([2, 2, 4, 4]),
    }
    return SimpleNamespace(
        model=SimpleNamespace(body_weldid=np.arange(6)),
        tasks=SimpleNamespace(effectors={}, contact_phase=policy),
        frames=frames,
        framerate=framerate,
        effectors=[],
        surfaces=[surface],
    )


def test_semantic_allocation_keeps_body_contact_and_material_handoffs():
    objective = _surface_objective()
    np.testing.assert_array_equal(
        semantic_transition_times(objective),
        np.array([.1, .4, .5, .6, .7, .8]),
    )
    # Legacy function is retained only as the over-fine oracle: it sees the
    # frame-2/3 sample exchange but cannot see mode, material or role changes.
    np.testing.assert_array_equal(
        transition_times(objective),
        np.array([.1, .2, .3, .6, .7, .8]),
    )


def test_endpoint_mode_environment_role_and_release_are_transitions():
    frames, framerate = 10, 10
    endpoint_active = np.zeros(frames, bool)
    managed = np.zeros(frames, bool)
    modes = np.full(frames, "none", object)
    environment = np.zeros(frames, int)
    roles = np.full(frames, "free", object)

    endpoint_active[2:8] = True
    managed[4:8] = True
    modes[endpoint_active] = "sticking"
    modes[6:8] = "sliding"
    environment[2:7] = 0
    environment[7:8] = 1
    roles[2:6] = "support"
    roles[6:8] = "control"

    task = SimpleNamespace(contact_modes=modes, environment_ids=environment)
    objective = SimpleNamespace(
        model=SimpleNamespace(body_weldid=np.arange(4)),
        tasks=SimpleNamespace(
            effectors={"hand": {"task": task}},
            contact_phase=SimpleNamespace(
                surface=None,
                surface_roles={},
                roles={"hand": roles},
            ),
        ),
        frames=frames,
        framerate=framerate,
        effectors=[{
            "name": "hand",
            "body": 3,
            "active": endpoint_active & ~managed,
            "managed": managed,
            "role": np.ones(frames),  # policy role must take precedence
        }],
        surfaces=[],
    )
    np.testing.assert_array_equal(
        semantic_transition_times(objective),
        np.array([.2, .6, .7, .8]),
    )


def test_endpoint_fallback_role_changes_are_visible_without_contact_phase():
    frames, framerate = 8, 4
    active = np.zeros(frames, bool)
    active[1:5] = True
    role = np.ones(frames)
    role[3:5] = 2.
    task = SimpleNamespace(contact_modes=None, environment_ids=None)
    objective = SimpleNamespace(
        model=SimpleNamespace(body_weldid=np.arange(3)),
        tasks=SimpleNamespace(effectors={"hand": {"task": task}}),
        frames=frames,
        framerate=framerate,
        effectors=[{
            "name": "hand", "body": 2, "active": active,
            "managed": np.zeros(frames, bool), "role": role,
        }],
        surfaces=[],
    )
    np.testing.assert_array_equal(
        semantic_transition_times(objective),
        np.array([.25, .75, 1.25]),
    )


def test_semantic_groups_do_not_merge_different_body_ids():
    frames, framerate = 10, 10
    active = np.zeros((frames, 2), bool)
    active[1:4, 0] = True       # body 2
    active[2:5, 1] = True       # body 3
    modes = np.where(active, "sticking", "none").astype(object)
    surface_ids = np.where(active, 10, -1)
    roles = np.where(active, "support", "none").astype(object)
    task = SimpleNamespace(
        effector_names=np.array(["foot", "foot"]),
        modes=modes,
        surface_ids=surface_ids,
    )
    objective = SimpleNamespace(
        model=SimpleNamespace(body_weldid=np.arange(4)),
        tasks=SimpleNamespace(
            effectors={},
            contact_phase=SimpleNamespace(
                surface=task,
                surface_roles={"support": roles == "support"},
            ),
        ),
        frames=frames,
        framerate=framerate,
        effectors=[],
        surfaces=[{"task": task, "active": active, "bodies": np.array([2, 3])}],
    )
    # Merging both bindings under only the effector name would hide body 2's
    # release at frame 4 while body 3 is still in contact.
    np.testing.assert_array_equal(
        semantic_transition_times(objective),
        np.array([.1, .2, .4, .5]),
    )


def test_sample_exchange_semantics_without_mutation():
    objective = _surface_objective()
    surface = objective.surfaces[0]
    task = surface["task"]
    before = semantic_transition_times(objective)

    # Same body, effector, mode, material and role: a pure sample exchange has
    # no new semantic state.  Change either mode or material and the handoff
    # must be retained.
    surface["active"][:, 0] = False
    surface["active"][1:6, 1] = True
    task.modes[1, 1] = "sticking"
    task.surface_ids[1, 1] = 10
    objective.tasks.contact_phase.surface_roles["support"][1, 1] = True
    np.testing.assert_array_equal(semantic_transition_times(objective), before)

    task.modes[5, 1] = "sticking"
    np.testing.assert_array_equal(semantic_transition_times(objective), before)
    task.modes[5, 1] = "sliding"
    task.surface_ids[5, 1] = 99
    # The inputs as edited above; the query itself must not mutate them.
    saved = {
        "active": surface["active"].copy(),
        "modes": task.modes.copy(),
        "ids": task.surface_ids.copy(),
        "bodies": surface["bodies"].copy(),
    }
    changed = semantic_transition_times(objective)
    # Frame .5 was already a material handoff in this fixture; the important
    # contract is that cross-semantic sample exchange is retained there, not
    # collapsed with the pure exchange case.
    assert .5 in changed

    np.testing.assert_array_equal(surface["active"], saved["active"])
    np.testing.assert_array_equal(task.modes, saved["modes"])
    np.testing.assert_array_equal(task.surface_ids, saved["ids"])
    np.testing.assert_array_equal(surface["bodies"], saved["bodies"])


def test_single_binding_transients_are_not_lost():
    objective = _surface_objective()
    surface = objective.surfaces[0]
    surface["active"][:] = False
    surface["active"][3, 0] = True
    task = surface["task"]
    task.modes[:] = "none"
    task.modes[3, 0] = "sticking"
    task.surface_ids[:] = -1
    task.surface_ids[3, 0] = 12
    policy = objective.tasks.contact_phase
    policy.surface_roles = {
        "support": np.zeros_like(surface["active"]),
        "control": np.zeros_like(surface["active"]),
    }
    policy.surface_roles["support"][3, 0] = True
    np.testing.assert_array_equal(
        semantic_transition_times(objective),
        np.array([.3, .4]),
    )


def test_semantic_transitions_exclude_clip_boundaries():
    objective = _surface_objective()
    surface = objective.surfaces[0]
    surface["active"][:2, 0] = [True, True]
    surface["active"][9, 2] = True
    times = semantic_transition_times(objective)
    assert len(times) and times[0] > 0.
    assert times[-1] < (objective.frames - 1) / objective.framerate
