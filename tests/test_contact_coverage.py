"""Dropped surface samples cannot suppress a surviving source observation."""
from dataclasses import replace

import numpy as np
import pytest

from omniretargeting.contact_phase import attach_contact_phase
from omniretargeting.contact_windows import slice_contact_task
from omniretargeting.surface_contact import SurfaceContactTasks
from omniretargeting.surface_tasks import SurfaceBinding
from tests.test_fast_backend import _problem


@pytest.mark.parametrize("roles_enabled", [True, False])
def test_surface_coverage_is_per_frame_and_preserves_source_targets(roles_enabled):
    model, _, _, poses, task_set, _ = _problem(frames=8)
    task = replace(task_set.effectors["foot"]["task"], surface_managed=True,
                   contact_mask=np.ones(8, bool), contact_modes=np.full(8, "sticking"))
    tasks = {"foot": task}
    priority = {"effectors": {"foot": {"robot_body": "foot"}}}
    active = np.ones((8, 1), bool)
    active[3:5] = False
    ids = np.zeros((8, 1), int)
    ids[5] = 1  # A different object cannot replace a ground observation.
    modes = np.full((8, 1), "sticking")
    modes[6] = "sliding"  # Normal-only rows cannot hold a sticking anchor.
    surface = SurfaceContactTasks(
        model, [SurfaceBinding("toes", 0, (1, 0, 0), "foot_geom",
                               (.05, 0, -.02), (0, 0, -1))],
        np.zeros((8, 1, 3)), np.zeros((8, 1, 3)), active,
        np.broadcast_to([0, 0, 1], (8, 1, 3)),
        surface_ids=ids, modes=modes, effector_names=["foot"],
    )
    policy = attach_contact_phase(surface, None, tasks, priority,
                                  {"enabled": roles_enabled})
    covered = np.array([True, True, True, False, False, False, False, True])
    np.testing.assert_array_equal(tasks["foot"].managed_frames, covered)
    for frame in range(8):
        assert tasks["foot"].frame(frame)["surface_managed"] == covered[frame]
    if policy is not None:
        assert np.all(policy.roles["foot"][3:7] == "hold")
    np.testing.assert_array_equal(tasks["foot"].contact_positions, task.contact_positions)
    np.testing.assert_array_equal(tasks["foot"].contact_mask, task.contact_mask)
    np.testing.assert_array_equal(slice_contact_task(tasks["foot"], 2, 6).managed_frames,
                                  covered[2:6])
    np.testing.assert_array_equal(surface.export()["surface_contact_effector_names"], ["foot"])


def test_same_rigid_body_different_source_region_does_not_cover_primary():
    model, _, _, _, task_set, _ = _problem(frames=8)
    tasks = {"foot": replace(task_set.effectors["foot"]["task"], surface_managed=True)}
    surface = SurfaceContactTasks(
        model, [SurfaceBinding("other", 0, (1, 0, 0), "foot_geom",
                               (.05, 0, -.02), (0, 0, -1))],
        np.zeros((8, 1, 3)), np.zeros((8, 1, 3)), np.ones((8, 1), bool),
        np.broadcast_to([0, 0, 1], (8, 1, 3)), modes=np.full((8, 1), "sticking"),
    )
    policy = attach_contact_phase(surface, None, tasks,
                                  {"effectors": {"foot": {"robot_body": "foot"}}})
    assert not tasks["foot"].managed_frames.any()
    assert policy.surface_roles["hold"].all()
    assert np.all(policy.roles["foot"][4:] == "hold")


def test_missing_surface_keeps_primary_active_and_explicit_disable_is_respected():
    model, _, _, _, task_set, _ = _problem(frames=8)
    task = replace(task_set.effectors["foot"]["task"], surface_managed=True)
    tasks = {"foot": task}
    attach_contact_phase(None, None, tasks,
                         {"effectors": {"foot": {"robot_body": "foot"}}}, model=model)
    assert not tasks["foot"].managed_frames.any()
    disabled = replace(task, surface_managed=False, surface_managed_mask=np.ones(8, bool))
    assert not disabled.managed_frames.any()
