"""Losing a mechanically needed body contact cannot buy a prettier hand task."""

from dataclasses import replace

import numpy as np
import pytest

from omniretargeting.contact_phase import attach_contact_phase, priority_progress
from omniretargeting.object_control import SourceObjectControlEvidence
from omniretargeting.sequence_retargeting import retarget_sequence
from tests.test_contact_phase import abdomen_error, case, support_on
from tests.test_full_body_surfaces import _markers


@pytest.mark.parametrize("part", ["abdomen", "pelvis", "forearm", "new_region"])
@pytest.mark.parametrize("lower_role", ["control", "hold", "follow"])
def test_supporting_body_patch_outranks_endpoint_tracking(part, lower_role):
    model, q, surface, tasks, priority = case(stance=True)
    surface.bindings = [replace(surface.bindings[0], part=part), surface.bindings[1]]
    if lower_role == "follow":
        tasks["left_foot"] = replace(
            tasks["left_foot"],
            contact_modes=np.full(len(q), "touch"),
            surface_normals=np.tile([1.0, 0, 0], (len(q), 1)),
        )
    control = (
        SourceObjectControlEvidence([{("left_foot", 0)}] * len(q), [])
        if lower_role == "control"
        else None
    )
    policy = attach_contact_phase(
        surface,
        None,
        tasks,
        priority,
        support_evidence=support_on(part, len(q)),
        control_evidence=control,
    )
    assert policy.role("left_foot", 0) == lower_role
    positions = _markers(model, q, ["root", "abdomen", "foot"])
    result, report = retarget_sequence(
        model,
        q,
        positions,
        ["root", "abdomen", "foot"],
        {n: n for n in ("root", "abdomen", "foot")},
        positions[:, 0],
        tasks,
        priority,
        framerate=30,
        extra_tasks=(surface,),
        config={
            "iterations": 5,
            "self_collision_enabled": False,
            "lock_root_xy": True,
            "lock_root_z": True,
            "lock_root_orientation": True,
        },
    )
    assert abdomen_error(surface, result) < 0.002
    assert (
        report["contact_role_residuals"]["support"]["weighted_position_norm_max_m"]
        < 0.002
    )
    assert result[:, model.joint("reach").qposadr[0]].min() > 0.038
    np.testing.assert_array_equal(result[:, :7], q[:, :7])


def test_object_control_yields_to_support_evidence_on_the_same_region():
    _, q, surface, tasks, priority = case(stance=True)
    control = SourceObjectControlEvidence([{("left_foot", 0)}] * len(q), [])
    evidence = support_on("left_foot", len(q))
    policy = attach_contact_phase(
        surface,
        None,
        tasks,
        priority,
        support_evidence=evidence,
        control_evidence=control,
    )
    assert policy.role("left_foot", 0) == "support"
    assert policy.surface_roles["hold"][:, 0].all()
    assert not policy.support[:, 0].any()


def test_optional_touch_retains_object_control_evidence():
    from omniretargeting.contact_mechanics import ContactCandidate
    from omniretargeting.contact_support import explain_source_support

    _, q, surface, tasks, priority = case(stance=True)
    patch = ContactCandidate("patch", [[0, 0, 0]], [[0, 0, 0]], [0, 0, 1], 0.0, 0.0)
    evidence = explain_source_support(
        [[patch]] * len(q), np.zeros((len(q), 6)), {"patch": ("left_foot", 0)}
    )
    assert evidence.role("left_foot", 0, 0) == "follow"
    control = SourceObjectControlEvidence([{("left_foot", 0)}] * len(q), [])
    policy = attach_contact_phase(
        surface,
        None,
        tasks,
        priority,
        support_evidence=evidence,
        control_evidence=control,
    )
    assert policy.role("left_foot", 0) == "control"


def test_endpoint_on_same_robot_body_cannot_erase_a_different_support_patch():
    from tests._contact_fixtures import _task

    model, q, surface, tasks, priority = case()
    tasks["hand"] = replace(
        _task(np.tile([0, 0, 1.0], (len(q), 1)), np.ones(len(q), bool)), name="hand"
    )
    priority["effectors"]["hand"] = {
        "robot_body": "abdomen",
        "source_contact_part": "hand",
    }
    policy = attach_contact_phase(
        surface, None, tasks, priority, support_evidence=support_on("abdomen", len(q))
    )
    assert policy.role("hand", 0) == "hold"
    assert policy.support[:, 0].all()
    assert len(list(policy.rows(q[0], 0, np.arange(model.nv)))) == 1


def test_lower_role_can_yield_but_other_frames_cannot_lose_support():
    before = np.array([[0, 0.04, 0], [0, 0, 0]])
    better = np.array([[0, 0, 0.04], [0, 0, 0]])
    assert priority_progress(before, better, 0.001) == (True, True)
    better[1, 1] = 0.01
    assert priority_progress(before, better, 0.001) == (False, False)


def test_temporally_coupled_primary_solve_preserves_each_frames_contacts():
    """A large error elsewhere must not strand the solve at a good contact."""
    from types import SimpleNamespace

    from scipy import sparse

    from omniretargeting._sequence.surface_guard import SurfaceSequenceGuard

    class CoupledFrames:
        def constraints(self, poses, flat):
            return [flat[0] == flat[1]]

    class Contacts(SurfaceSequenceGuard):
        def priority_rows(self, pose, frame, level):
            if level == 0:
                yield 1.0, np.array([pose[0] - (0.01, 0.5)[frame]]), np.ones((1, 1))

    poses = np.zeros((2, 1))
    guard = Contacts(
        SimpleNamespace(contact_phase=SimpleNamespace(tolerance=0.001)),
        poses,
        0.001,
        preserve_contacts=False,
    )
    guard.continuity = CoupledFrames()
    result = guard.solve(
        sparse.eye(2, format="csr"),
        np.ones(2),
        -np.ones_like(poses),
        np.ones_like(poses),
        poses,
    )
    assert result.status == 1
    np.testing.assert_allclose(result.x[0], result.x[1], atol=1e-7)
    assert 0.005 < result.x[0] < 0.021001
    # This checks the real per-frame acceptance rule at full and reduced steps.
    before = guard.priority_errors(poses)
    for fraction in (1.0, 0.25, 1 / 128):
        after = guard.priority_errors(poses + fraction * result.x[:, None])
        assert priority_progress(before, after, fraction * 0.001001) == (True, True)


def test_new_achieved_reference_does_not_reset_cumulative_correction_budget():
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        '<mujoco><worldbody><body name="tool">'
        '<joint type="slide" axis="1 0 0" range="-1 1"/>'
        '<geom size=".02"/></body></worldbody></mujoco>'
    )
    initial = np.full((5, 1), 0.025)
    source = np.zeros((5, 1, 3))

    class Objective:
        def rows(self, pose, frame, dofs):
            yield 1e8, np.array([pose[0] - 0.1]), np.ones((1, len(dofs)))

    result, _ = retarget_sequence(
        model,
        initial,
        source,
        ["tool"],
        {"tool": "tool"},
        source[:, 0],
        {},
        {},
        framerate=30,
        extra_tasks=(Objective(),),
        config={"iterations": 3, "self_collision_enabled": False},
        endpoint_reference_trajectory=initial,
        correction_reference_trajectory=np.zeros_like(initial),
    )
    assert result.min() > 0.025
    assert result.max() <= 0.0300001


def test_starting_outside_fixed_correction_budget_is_not_silently_accepted():
    model, q, _, tasks, priority = case()
    positions = _markers(model, q, ["root", "abdomen", "foot"])
    invalid = q.copy()
    invalid[:, 0] += 0.1
    with pytest.raises(ValueError, match="fixed correction reference"):
        retarget_sequence(
            model,
            invalid,
            positions,
            ["root", "abdomen", "foot"],
            {n: n for n in ("root", "abdomen", "foot")},
            positions[:, 0],
            tasks,
            priority,
            framerate=30,
            config={"self_collision_enabled": False},
            correction_reference_trajectory=q,
        )


@pytest.mark.parametrize("backend", ["framewise", "sequence"])
def test_control_cannot_discard_an_unexplained_geometric_hold(backend):
    from tests.test_contact_phase_primary import framewise, hand_case

    model, q, tasks, priority, _ = hand_case(stance=True)
    control = SourceObjectControlEvidence([{("left_hand", 0)}] * len(q), [])
    policy = attach_contact_phase(
        None, None, tasks, priority, model=model, control_evidence=control
    )
    assert policy.role("left_foot", 0) == "hold"
    assert policy.role("left_hand", 0) == "control"
    if backend == "framewise":
        hand, foot = framewise(model, q, tasks, priority, policy)
        assert foot < 0.003 and hand > 0.10
    else:
        positions = _markers(model, q, ["root", "abdomen", "foot"])
        result, _ = retarget_sequence(
            model,
            q,
            positions,
            ["root", "abdomen", "foot"],
            {n: n for n in ("root", "abdomen", "foot")},
            positions[:, 0],
            tasks,
            priority,
            framerate=30,
            config={
                "iterations": 5,
                "self_collision_enabled": False,
                "lock_root_xy": True,
                "lock_root_z": True,
                "lock_root_orientation": True,
            },
        )
        assert result[:, model.joint("reach").qposadr[0]].max() < 0.003
