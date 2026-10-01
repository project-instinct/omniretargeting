"""Authored penetration remains source evidence, never a robot surface goal."""
import copy

import numpy as np

from omniretargeting.surface_contact import build_surface_contact_tasks
from tests.test_surface_contact import source_case


def test_penetrating_source_keeps_skin_unchanged_and_robot_goal_on_scene(tmp_path):
    model, _, scene, source = source_case()
    source['full_body']['triangles'] -= [0, 0, .01]
    original = copy.deepcopy(source)
    task, report = build_surface_contact_tasks(
        model, source, model.qpos0, scene,
        {'correspondence': {'cache_dir': str(tmp_path), 'iterations': 30}},
    )
    assert task is not None and task.active.any()
    # The original skin is 5 mm inside the top of the original scene box.
    assert task.source_vectors[task.active, 2].min() < -.004
    np.testing.assert_allclose(task.targets[task.active, 2], .4, atol=1e-9)
    for key, value in original['full_body'].items():
        np.testing.assert_array_equal(source['full_body'][key], value)
    assert report['anchors']['targets_on_observed_surface']
