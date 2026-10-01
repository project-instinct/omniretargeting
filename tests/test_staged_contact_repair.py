"""Other supports stay fixed while a colliding contact is corrected in stages."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest

from omniretargeting._sequence.contact_hold import ContactHoldAnchors, ContactHoldStage
from omniretargeting._sequence.surface_guard import SurfaceSequenceGuard
from omniretargeting.contact import build_world_tracking_tasks
from omniretargeting.sequence_retargeting import (
    prepare_sequence,
    retarget_sequence,
    sequence_config,
)
from omniretargeting.staged_contact_repair import repair_contact_stages


def fixture():
    model = mujoco.MjModel.from_xml_string("""<mujoco><compiler angle="radian"/>
    <worldbody><body name="seat_contact" pos="0 0 .49"><freejoint/>
      <geom name="seat_skin" type="sphere" size=".02"/>
      <body name="support_a" pos="-.15 0 -.4">
        <joint type="slide" axis="0 0 1" range="-.1 .1"/>
        <geom size=".02"/>
      </body>
      <body name="support_b" pos=".15 0 -.4">
        <joint type="slide" axis="0 0 1" range="-.1 .1"/>
        <geom size=".02"/>
      </body>
    </body></worldbody></mujoco>""")
    frames = 5
    poses = np.tile(model.qpos0, (frames, 1))
    names = ["seat_contact", "support_a", "support_b"]
    source = np.tile([[0, 0, 0.49], [-0.15, 0, 0.09], [0.15, 0, 0.09]], (frames, 1, 1))
    tasks = build_world_tracking_tasks(
        {name: {"tracking_positions": source[:, k]} for k, name in enumerate(names)}
    )
    for k, name in enumerate(names):
        target = source[:, k].copy()
        if k == 0:
            target[:, 2] += 0.02
        tasks[name] = replace(
            tasks[name],
            contact_mask=np.ones(frames, bool),
            contact_modes=np.full(frames, "sticking"),
            contact_positions=target,
        )
    priority = {"effectors": {name: {"robot_body": name} for name in names}}
    config = sequence_config({"iterations": 4, "self_collision_enabled": False})
    args = (
        model,
        poses,
        source,
        names,
        {n: n for n in names},
        source[:, 0],
        tasks,
        priority,
    )
    _, _, task_set, _ = prepare_sequence(*args, config)
    guard = SurfaceSequenceGuard(task_set, poses, 0.001)
    return args, config, guard


def test_hold_rejects_equal_error_motion_around_a_contact_target():
    args, cfg, guard = fixture()
    model, poses, *rest = args
    # Give both supports a residual. Rotating this error about the unchanged
    # target preserves the old norm budget while moving the actual support.
    tasks = args[-2]
    for name in ("support_a", "support_b"):
        target = tasks[name].contact_positions.copy()
        target[:, 2] += 0.02
        tasks[name] = replace(tasks[name], contact_positions=target)
    _, _, task_set, _ = prepare_sequence(model, poses, *rest, cfg)
    guard = SurfaceSequenceGuard(task_set, poses, 0.001)
    stage = ContactHoldStage(
        ContactHoldAnchors(guard, poses), model.body("seat_contact").id, 0.0001
    ).bind(guard)
    candidate = poses.copy()
    candidate[:, 7:] += 0.04
    assert guard.accepts(candidate)  # Same 2 cm distance, opposite side.
    assert not stage.accepts(candidate)


def test_lifting_one_contact_keeps_both_other_supports_in_world():
    args, cfg, guard = fixture()
    model, poses = args[:2]
    body = model.body("seat_contact").id
    stage = ContactHoldStage(ContactHoldAnchors(guard, poses), body, 0.0001)
    result, report = retarget_sequence(
        *args,
        config=cfg,
        framerate=30.0,
        contact_stage=stage,
        endpoint_reference_trajectory=poses,
        correction_reference_trajectory=poses,
    )
    assert (result[:, 2] - poses[:, 2]).min() > 0.015
    assert (
        np.max(np.abs(result[:, 2, None] + result[:, 7:] - poses[:, 2, None]))
        < 0.000101
    )
    assert report["contact_stage"]["within_hold_budget"]
    assert report["posture_envelope"]["within_translation_bound"]


def test_acquiring_one_group_does_not_reset_other_hold_anchors():
    args, _, guard = fixture()
    poses = args[1]
    anchors = ContactHoldAnchors(guard, poses)
    body = args[0].body("seat_contact").id
    moved = poses.copy()
    moved[:, 2] += 0.01
    updated = anchors.acquired(guard, moved, body)
    for frame, values in enumerate(anchors.values):
        for key, value in values.items():
            if anchors.bodies[frame][key] != body:
                np.testing.assert_array_equal(updated.values[frame][key], value)
    assert not anchors.values[0][("endpoint", "seat_contact")].flags.writeable


def test_stages_require_original_fixed_references():
    args, cfg, guard = fixture()
    stage = ContactHoldStage(ContactHoldAnchors(guard, args[1]), 1, 0.0001)
    with pytest.raises(ValueError, match="fixed contact and correction references"):
        retarget_sequence(*args, config=cfg, framerate=30.0, contact_stage=stage)


class SeatPlane:
    def query(self, points):
        # Finite seat: supports at x=+/- .15 are outside it. The root sphere's
        # bottom is 1 cm below the .48 m seat surface at the initial pose.
        clearance = points[:, 2] - 0.48
        clearance[np.abs(points[:, 0]) > 0.08] = 1.0
        return {
            "clearance": clearance,
            "directions": np.tile([0.0, 0.0, 1.0], (len(points), 1)),
            "uncertain": np.zeros(len(points), bool),
        }


def test_joint_attempt_then_small_root_lift_with_two_fixed_supports():
    args, cfg, _ = fixture()
    poses = args[1]
    result, report = repair_contact_stages(
        *args,
        config=cfg,
        framerate=30.0,
        scene_query=SeatPlane(),
        endpoint_reference_trajectory=poses,
        correction_reference_trajectory=poses,
    )
    assert [s["mode"] for s in report["stages"]] == ["fixed_base", "bounded_root_lift"]
    assert not report["stages"][0]["accepted"]
    assert report["stages"][1]["accepted"]
    assert report["after"]["max_depth_m"] < 0.001
    assert 0.009 < (result[:, 2] - poses[:, 2]).min() < 0.012
    assert (
        np.max(np.abs(result[:, 2, None] + result[:, 7:] - poses[:, 2, None]))
        < 0.000101
    )
    np.testing.assert_allclose(np.diff(result[:, 7:], axis=0), 0, atol=2e-7)


def test_preexisting_chatter_is_not_certified_with_a_new_reference():
    args, cfg, _ = fixture()
    model, poses, *rest = args
    bad = poses.copy()
    bad[:, 7] += 0.005 * (-1.0) ** np.arange(len(bad))
    result, report = repair_contact_stages(
        model,
        bad,
        *rest,
        config=cfg,
        framerate=30.0,
        endpoint_reference_trajectory=bad,
        correction_reference_trajectory=poses,
    )
    assert report["status"] == "input_exceeds_fixed_budgets"
    assert report["input_checks"]["continuity"] is False
    np.testing.assert_array_equal(result, bad)


@pytest.mark.parametrize("lock_height", [True, False])
def test_no_root_lift_without_vertical_exit_evidence_or_when_locked(lock_height):
    args, cfg, _ = fixture()

    class LateralScene(SeatPlane):
        def query(self, points):
            measured = super().query(points)
            if not lock_height:
                measured["directions"][:] = [1.0, 0.0, 0.0]
            return measured

    _, report = repair_contact_stages(
        *args,
        config={**cfg, "lock_root_z": lock_height},
        framerate=30.0,
        scene_query=LateralScene(),
        endpoint_reference_trajectory=args[1],
        correction_reference_trajectory=args[1],
    )
    assert all(s["mode"] == "fixed_base" for s in report["stages"])


@pytest.mark.parametrize("mode", ["sticking", "sliding", "touch"])
def test_surface_contacts_use_the_same_holds_as_endpoint_contacts(mode):
    from tests.test_contact_phase import case
    from tests.test_full_body_surfaces import _markers

    model, q, surface, tasks, priority = case(stance=True, sole_mode=mode)
    names = ["root", "abdomen", "foot"]
    positions = _markers(model, q, names)
    _, _, task_set, _ = prepare_sequence(
        model,
        q,
        positions,
        names,
        {n: n for n in names},
        positions[:, 0],
        tasks,
        priority,
        sequence_config(),
        extra_tasks=(surface,),
    )
    guard = SurfaceSequenceGuard(task_set, q, 0.001)
    anchors = ContactHoldAnchors(guard, q)
    stage = ContactHoldStage(anchors, model.body("abdomen").id, 0.0001).bind(guard)
    candidate = q.copy()
    candidate[:, -2] += 0.005  # Moving abdomen also drags its contacting foot.
    assert not stage.accepts(candidate)
    surface_keys = [k for k in anchors.values[0] if k[0] == "surface_contact"]
    assert len(surface_keys) == 2


def test_replay_preserves_exported_support_roles():
    from omniretargeting.contact_phase import attach_contact_phase
    from omniretargeting.contact_windows import restore_contact_roles
    from tests.test_contact_phase import case, support_on

    _, q, surface, tasks, priority = case(stance=True)
    policy = attach_contact_phase(
        surface, None, tasks, priority, support_evidence=support_on("abdomen", len(q))
    )
    saved = policy.export()
    restored = restore_contact_roles(surface, tasks, priority, saved)
    for key, value in saved.items():
        np.testing.assert_array_equal(restored.export()[key], value)
    assert restored.diagnostics["phases"] == policy.diagnostics["phases"]
    assert restored.diagnostics["surface_support_bodies"] == ["abdomen"]
    assert all(task.contact_phase is restored for task in tasks.values())


def test_downstream_pass_cannot_move_any_frozen_final_contact():
    args, _, guard = fixture()
    poses = args[1]
    guard.contact_stage = ContactHoldStage(
        ContactHoldAnchors(guard, poses), None, 0.0001
    ).bind(guard)
    assert guard.accepts(poses)
    candidate = poses.copy()
    candidate[:, 2] += 0.0005
    assert not guard.accepts(candidate)
    assert guard.contact_stage.diagnostics(poses)["movable_body"] is None


def test_clearance_stage_does_not_refit_an_unrelated_free_limb():
    args, cfg, _ = fixture()
    model, poses, source, names, mapping, roots, tasks, priority = args
    free = tasks["support_b"]
    desired = free.tracking_positions.copy()
    desired[:, 2] += 0.05
    tasks["support_b"] = replace(
        free,
        contact_mask=np.zeros(len(poses), bool),
        contact_modes=np.full(len(poses), "none"),
        tracking_positions=desired,
    )
    result, report = repair_contact_stages(
        model,
        poses,
        source,
        names,
        mapping,
        roots,
        tasks,
        priority,
        config=cfg,
        framerate=30.0,
        scene_query=SeatPlane(),
        endpoint_reference_trajectory=poses,
        correction_reference_trajectory=poses,
    )
    assert report["after"]["max_depth_m"] < 0.001
    # This body has no hold constraint. Clearance-only minimal movement keeps
    # it near its input pose instead of using repair to refit the 5 cm target.
    assert np.max(np.abs(result[:, 2] + result[:, -1] - poses[:, 2])) < 0.001


def test_stage_preserves_acquired_smoothness_as_well_as_source_budget():
    from scipy.spatial.transform import Rotation

    from omniretargeting._sequence.continuity_guard import SequenceContinuityGuard

    args, _, guard = fixture()
    acquired = args[1]
    reference = acquired.copy()
    reference[:, 7] += 0.005 * (-1.0) ** np.arange(len(reference))
    rotations = Rotation.from_quat(acquired[:, 3:7], scalar_first=True)
    guard.continuity = SequenceContinuityGuard(guard.tasks, reference, rotations)
    stage = ContactHoldStage(
        ContactHoldAnchors(guard, acquired), 1, 0.0001, continuity_reference=acquired
    ).bind(guard)
    candidate = acquired.copy()
    candidate[:, 7] += 0.00001 * (-1.0) ** np.arange(len(reference))
    assert guard.accepts(candidate)  # Inside source and contact budgets.
    assert not stage.accepts(candidate)  # Still adds chatter to acquired motion.
