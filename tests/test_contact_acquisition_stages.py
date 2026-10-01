"""Whole-body contacts participate before the correction reference is frozen."""

from types import SimpleNamespace

import numpy as np
import pytest

from omniretargeting._core.sequence import solve_core_sequence
from omniretargeting.contact_phase import attach_contact_phase
from tests.test_contact_phase import case, support_on
from tests.test_full_body_surfaces import _markers


@pytest.mark.parametrize("primary_contacts", [True, False])
def test_all_body_contacts_and_roles_survive_acquisition(primary_contacts, monkeypatch):
    from omniretargeting._core import sequence

    model, q, surface, tasks, priority = case()
    if not primary_contacts:
        tasks, priority = {}, {"effectors": {}}
    policy = attach_contact_phase(
        surface, None, tasks, priority, support_evidence=support_on("abdomen", len(q))
    )
    names = ["root", "abdomen", "foot"]
    positions = _markers(model, q, names)
    owner = SimpleNamespace(
        robot_model=model,
        _source_framerate=30,
        _resolve_source_root_index=lambda: 0,
        retargeting_config={"end_effector_priority": priority},
        smplx_joint_names=names,
        valid_joint_mapping={n: n for n in names},
        last_surface_contact_tasks=surface,
    )
    solver = SimpleNamespace(
        robot_reference_qpos=q[0], terrain_points=np.zeros((0, 3)), scene_query=None
    )
    calls, outputs = [], []

    def solve(model, initial, *args, **kwargs):
        calls.append((args[4], kwargs))  # Endpoint task mapping.
        result = initial.copy()
        result[:, -1] += 0.001
        outputs.append(result.copy())
        return result, {
            "contact_offset_choice": {name: [0] * len(initial) for name in args[4]}
        }

    monkeypatch.setattr(sequence, "retarget_sequence", solve)
    monkeypatch.setattr(sequence, "sequence_body_orientation_targets", lambda *args: ())
    source = positions.copy()
    solve_core_sequence(owner, solver, positions, tasks, q[:, 3:7], None)
    assert len(calls) == 3
    assert not calls[0][1]["extra_tasks"]
    assert calls[0][1]["scene_query"] is None
    assert calls[0][1]["config"]["self_collision_enabled"] is False
    for endpoint_tasks, kwargs in calls[1:]:
        assert surface in kwargs["extra_tasks"]
        assert surface.contact_phase is policy
        assert all(task.contact_phase is policy for task in endpoint_tasks.values())
        np.testing.assert_array_equal(
            kwargs["correction_reference_trajectory"], outputs[0]
        )
    np.testing.assert_array_equal(
        calls[2][1]["endpoint_reference_trajectory"], outputs[1]
    )
    np.testing.assert_array_equal(positions, source)
    from omniretargeting._core.sequence_guard import preserve_sequence_budgets

    valid = owner._sequence_accepted_motion.copy()
    accepted, status = preserve_sequence_budgets(owner, valid)
    assert status["candidate_retained"]
    np.testing.assert_array_equal(accepted, valid)
    invalid = valid.copy()
    invalid[:, 0] += 0.1
    accepted, status = preserve_sequence_budgets(owner, invalid)
    assert not status["candidate_retained"]
    np.testing.assert_array_equal(accepted, valid)


def test_source_fit_residual_does_not_abort_bounded_contact_optimization(monkeypatch):
    from omniretargeting._core import sequence

    model, q, _, tasks, priority = case()
    names = ["root", "abdomen", "foot"]
    positions = _markers(model, q, names)
    owner = SimpleNamespace(
        robot_model=model,
        _source_framerate=30,
        _resolve_source_root_index=lambda: 0,
        retargeting_config={"end_effector_priority": priority},
        smplx_joint_names=names,
        valid_joint_mapping={n: n for n in names},
    )
    solver = SimpleNamespace(
        robot_reference_qpos=q[0], terrain_points=np.empty((0, 3)), scene_query=None
    )
    calls = []

    def unresolved_reference(*args, **kwargs):
        calls.append(kwargs)
        return q.copy(), {
            "source_shape": {"angular_envelope_rad": None, "maximum_source_axis_angle_rad": 0.7},
            "contact_offset_choice": {name: [0] * len(q) for name in tasks},
        }

    monkeypatch.setattr(sequence, "retarget_sequence", unresolved_reference)
    monkeypatch.setattr(sequence, "sequence_body_orientation_targets", lambda *args: ())
    result = solve_core_sequence(owner, solver, positions, tasks, q[:, 3:7], None)
    # This fixture has no active contact: source fit, final solve.
    assert len(calls) == 2
    assert not calls[0]["extra_tasks"]
    for kwargs in calls[1:]:
        np.testing.assert_array_equal(kwargs["correction_reference_trajectory"], q)
    diagnostics = owner.last_sequence_report["initialization"]["source_pose"]
    assert diagnostics["source_shape"]["maximum_source_axis_angle_rad"] == 0.7
    assert diagnostics["source_shape"]["angular_envelope_rad"] is None
    np.testing.assert_array_equal(result, q)
