from dataclasses import replace

import numpy as np
import pytest

from omniretargeting.contact import ContactMode, SceneContactTask
from omniretargeting.core import (
    _SupportHandoffGuard,
    _build_robot_base_reference,
)


def _contact_task(
    name: str,
    targets: np.ndarray,
    support_start: int,
    support_end: int,
) -> SceneContactTask:
    targets = np.asarray(targets, dtype=float)
    num_frames = len(targets)
    modes = np.full(num_frames, int(ContactMode.NONE), dtype=np.uint8)
    modes[support_start:support_end] = int(ContactMode.SUPPORT)
    segment_ids = np.full(num_frames, -1, dtype=np.int32)
    segment_ids[support_start:support_end] = 0
    alpha = np.zeros(num_frames, dtype=float)
    alpha[support_start:support_end] = 1.0
    return SceneContactTask(
        name=name,
        tracking_positions=targets.copy(),
        contact_positions=targets.copy(),
        surface_points=targets.copy(),
        surface_normals=np.tile([0.0, 0.0, 1.0], (num_frames, 1)),
        distances=np.zeros(num_frames),
        relative_speeds=np.zeros(num_frames),
        contact_confidence=np.ones(num_frames),
        contact_modes=modes,
        contact_segment_ids=segment_ids,
        transition_weights=np.ones(num_frames),
        forward_directions=np.tile([1.0, 0.0, 0.0], (num_frames, 1)),
        up_directions=np.tile([0.0, 0.0, 1.0], (num_frames, 1)),
        effective_swing_positions=targets.copy(),
        contact_bridge_alpha=alpha,
        contact_bridge_positions=targets.copy(),
    )


def _root_warp_case(num_frames=100, support_start=40, support_end=55):
    targets = np.tile([0.0, 0.9, 0.0], (num_frames, 1))
    task = _contact_task(
        "left_hand", targets, support_start=support_start, support_end=support_end
    )
    roots = np.zeros((num_frames, 3), dtype=float)
    orientations = np.tile([1.0, 0.0, 0.0, 0.0], (num_frames, 1))
    profile = {"origin_offset": np.array([0.35, 0.0, 0.0]), "max_reach": 0.2}
    return roots, orientations, task, profile


def test_root_yaw_warp_is_local_bounded_and_zero_for_reachable_contact():
    roots, orientations, task, profile = _root_warp_case()
    roots_before = roots.copy()
    orientations_before = orientations.copy()
    config = {
        "enabled": True,
        "approach_frames": 6,
        "release_frames": 7,
        "max_xyz": [0.0, 0.0, 0.0],
        "max_translation_norm": 0.0,
        "yaw_enabled": True,
        "max_yaw": 0.35,
    }

    warped, unwarped, yaw, diagnostics = _build_robot_base_reference(
        roots,
        {"left_hand": task},
        {"left_hand": profile},
        config,
        root_z_offset=np.array([0.01, -0.02, 0.3]),
        base_orientations=orientations,
        framerate=30.0,
        return_diagnostics=True,
    )

    np.testing.assert_array_equal(roots, roots_before)
    np.testing.assert_array_equal(orientations, orientations_before)
    np.testing.assert_allclose(warped, unwarped)
    assert np.max(np.abs(yaw)) == pytest.approx(0.35)
    assert diagnostics["max_yaw"] <= 0.35 + 1e-12
    # The minimum-jerk endpoints are exactly zero and no yaw leaks outside the
    # segment's owning approach/core/release window.
    np.testing.assert_allclose(yaw[: 40 - 6 + 1], 0.0, atol=1e-14)
    np.testing.assert_allclose(yaw[55 + 7 - 1 :], 0.0, atol=1e-14)

    reachable_targets = np.tile([0.50, -0.02, 0.30], (len(roots), 1))
    reachable = _contact_task("left_hand", reachable_targets, 40, 55)
    reachable_warped, reachable_unwarped, reachable_yaw, _ = (
        _build_robot_base_reference(
            roots,
            {"left_hand": reachable},
            {"left_hand": profile},
            config,
            root_z_offset=np.array([0.01, -0.02, 0.3]),
            base_orientations=orientations,
            framerate=30.0,
            return_diagnostics=True,
        )
    )
    np.testing.assert_allclose(reachable_warped, reachable_unwarped, atol=1e-14)
    np.testing.assert_allclose(reachable_yaw, 0.0, atol=1e-14)


def test_root_warp_derivative_caps_widen_the_smooth_minimum_jerk_windows():
    fps = 30.0
    roots, orientations, task, profile = _root_warp_case(
        num_frames=180, support_start=80, support_end=100
    )
    caps = {
        "max_linear_velocity": 0.30,
        "max_linear_acceleration": 2.0,
        "max_yaw_rate": 1.2,
        "max_yaw_acceleration": 8.0,
    }
    warped, unwarped, yaw, diagnostics = _build_robot_base_reference(
        roots,
        {"left_hand": task},
        {"left_hand": profile},
        {
            "enabled": True,
            "approach_frames": 2,
            "release_frames": 2,
            "max_xyz": [0.06, 0.06, 0.04],
            "max_translation_norm": 0.08,
            "yaw_enabled": True,
            "max_yaw": 0.35,
            **caps,
        },
        root_z_offset=0.0,
        base_orientations=orientations,
        framerate=fps,
        return_diagnostics=True,
    )
    correction = warped - unwarped
    speed = np.linalg.norm(np.diff(correction, axis=0) * fps, axis=1)
    acceleration = np.linalg.norm(
        np.diff(correction, n=2, axis=0) * fps**2, axis=1
    )
    yaw_rate = np.abs(np.diff(yaw) * fps)
    yaw_acceleration = np.abs(np.diff(yaw, n=2) * fps**2)

    assert np.max(speed) <= caps["max_linear_velocity"] + 1e-9
    assert np.max(acceleration) <= caps["max_linear_acceleration"] + 1e-9
    assert np.max(yaw_rate) <= caps["max_yaw_rate"] + 1e-9
    assert np.max(yaw_acceleration) <= caps["max_yaw_acceleration"] + 1e-9
    assert diagnostics["derivative_slack"] == {
        "linear_velocity": 0.0,
        "linear_acceleration": 0.0,
        "yaw_rate": 0.0,
        "yaw_acceleration": 0.0,
    }
    window = diagnostics["segment_windows"][0]
    assert window["approach_frames"] > 2
    assert window["release_frames"] > 2
    # A clipped correction would concentrate motion in one frame.  The widened
    # quintic has many active samples and a broad peak on both tracks.
    assert np.count_nonzero(speed > 1e-6) >= 12
    assert np.count_nonzero(yaw_rate > 1e-6) >= 12
    assert np.partition(speed, -2)[-2] >= 0.8 * np.max(speed)
    assert np.partition(yaw_rate, -2)[-2] >= 0.8 * np.max(yaw_rate)


def test_disabled_root_warp_is_exact_and_does_not_alias_source_or_unwarped():
    roots, orientations, task, profile = _root_warp_case(num_frames=12)
    roots[:] = np.arange(36, dtype=float).reshape(12, 3) / 10.0
    roots_before = roots.copy()
    orientations_before = orientations.copy()
    offset = np.array([0.1, -0.2, 0.3])

    warped, unwarped, yaw, diagnostics = _build_robot_base_reference(
        roots,
        {"left_hand": task},
        {"left_hand": profile},
        {
            "enabled": False,
            "yaw_enabled": True,
            "max_yaw": 1.0,
            "max_linear_velocity": 0.0,
        },
        root_z_offset=offset,
        base_orientations=orientations,
        return_diagnostics=True,
    )

    expected = roots_before + offset
    np.testing.assert_array_equal(warped, expected)
    np.testing.assert_array_equal(unwarped, expected)
    np.testing.assert_array_equal(yaw, np.zeros(len(roots)))
    np.testing.assert_array_equal(roots, roots_before)
    np.testing.assert_array_equal(orientations, orientations_before)
    assert not np.shares_memory(warped, roots)
    assert not np.shares_memory(unwarped, roots)
    assert not np.shares_memory(warped, unwarped)
    warped[0, 0] += 10.0
    np.testing.assert_array_equal(unwarped, expected)
    assert diagnostics["max_translation"] == 0.0


def _handoff_tasks(num_frames=14, new_support_start=4):
    targets = np.zeros((num_frames, 3), dtype=float)
    old = _contact_task("left_foot", targets, 0, 5)
    new = _contact_task(
        "left_hand", targets + [0.4, 0.0, 0.2], new_support_start, num_frames
    )
    # Make the read-only phase view realistic on the two transition shoulders.
    old_alpha = old.contact_bridge_alpha.copy()
    old_alpha[5:7] = [0.8, 0.3]
    new_alpha = new.contact_bridge_alpha.copy()
    if new_support_start > 0:
        new_alpha[new_support_start - 1] = 0.5
    old = replace(old, contact_bridge_alpha=old_alpha)
    new = replace(new, contact_bridge_alpha=new_alpha)
    return {old.name: old, new.name: new}


def test_support_handoff_holds_old_support_until_new_support_is_stable():
    tasks = _handoff_tasks()
    old_modes_before = tasks["left_foot"].contact_modes.copy()
    guard = _SupportHandoffGuard(
        tasks,
        {
            "enabled": True,
            "position_error_threshold": 0.02,
            "hold_frames": 3,
            "max_penetration": 0.03,
            "events": [
                {
                    "type": "support_transfer",
                    "old_supports": ["left_foot"],
                    "new_supports": ["left_hand"],
                    "release_frame": 5,
                }
            ],
        },
    )

    overrides = []
    for frame_idx in range(14):
        overrides.append(guard.overrides_for_frame(frame_idx))
        error = 0.01 if frame_idx >= 5 else 0.05
        # Frame 6 has a serious intersection, so the consecutive hold restarts.
        penetration = 0.04 if frame_idx == 6 else 0.0
        guard.observe(frame_idx, {"left_hand": error}, penetration)

    assert all("left_foot" in overrides[frame] for frame in range(5, 10))
    assert "left_foot" not in overrides[10]
    diagnostics = guard.diagnostics()[0]
    assert diagnostics["requested_release_frame"] == 5
    assert diagnostics["established_frame"] == 9
    assert diagnostics["actual_release_frame"] == 10
    assert diagnostics["held_extra_frames"] == 5

    guarded = guard.materialize(tasks)
    np.testing.assert_array_equal(
        guarded["left_foot"].contact_modes[5:10],
        np.full(5, int(ContactMode.SUPPORT), dtype=np.uint8),
    )
    assert guarded["left_foot"].contact_modes[10] == int(ContactMode.NONE)
    # Materialization is robot-policy data; the source-derived input schedule
    # remains independently traceable and is never overwritten in place.
    np.testing.assert_array_equal(tasks["left_foot"].contact_modes, old_modes_before)


def test_explicit_flight_event_permits_full_release():
    tasks = _handoff_tasks(new_support_start=8)
    guard = _SupportHandoffGuard(
        tasks,
        {
            "enabled": True,
            "hold_frames": 3,
            "events": [
                {
                    "type": "support_transfer",
                    "old_supports": ["left_foot"],
                    "new_supports": ["left_hand"],
                    "release_frame": 5,
                },
                {"type": "flight", "start_frame": 5, "end_frame": 8},
            ],
        },
    )

    assert guard.overrides_for_frame(5) == {}
    assert tasks["left_foot"].contact_modes[5] == int(ContactMode.NONE)
    assert tasks["left_hand"].contact_modes[5] == int(ContactMode.NONE)
    assert guard.diagnostics()[0]["flight_override"] is True
