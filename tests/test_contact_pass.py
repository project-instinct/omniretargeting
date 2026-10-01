"""Near-pass review must never be a short-contact or body-name filter."""

from dataclasses import replace

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting.contact import build_world_tracking_tasks
from omniretargeting.contact_pass import review_contact_passes, without_pass_attraction
from omniretargeting.contact_schedule import approach_blend
from omniretargeting.contact_windows import slice_contact_task
from omniretargeting.sequence_retargeting import retarget_sequence


def example(fps=30):
    time = np.arange(fps + 1) / fps
    positions = np.c_[0.3 * time, np.full(len(time), 0.025), np.zeros(len(time))]
    mask = (time >= 0.4) & (time <= 0.6)
    return positions, np.full(len(time), 0.025), mask, time


def review(points, gap, mask, time, **kwargs):
    return review_contact_passes(
        points, gap, mask, times=time, position_uncertainty_m=0.005, **kwargs
    )


@pytest.mark.parametrize("fps", [15, 30, 60, 120])
def test_separated_pass_is_invariant_to_frame_rate_and_rigid_coordinates(fps):
    points, gap, mask, time = example(fps)
    original = points.copy()
    first = review(points, gap, mask, time)
    rotation = Rotation.from_rotvec([0.6, -0.4, 0.9]).as_matrix()
    moved = review(points @ rotation.T + [3, -1, 2], gap, mask, time)
    np.testing.assert_array_equal(first.pass_mask, mask)
    np.testing.assert_array_equal(moved.pass_mask, first.pass_mask)
    np.testing.assert_array_equal(points, original)


@pytest.mark.parametrize(
    "reason",
    [
        "touch",
        "label",
        "support",
        "unknown",
        "penetration",
        "stationary",
        "turn",
        "environment",
    ],
)
def test_same_duration_events_with_contact_or_conflicting_evidence_are_retained(reason):
    points, gap, mask, time = example()
    extra = {}
    if reason == "touch":
        gap[np.flatnonzero(mask)[0]] = 0.0  # Even one frame matters.
    elif reason in {"label", "support"}:
        extra["protected_mask"] = mask.copy()
    elif reason == "unknown":
        gap[mask] = np.nan
    elif reason == "penetration":
        gap[mask] = -0.02
    elif reason == "stationary":
        points[:] = points[0]
    elif reason == "turn":
        points[np.flatnonzero(mask)[len(np.flatnonzero(mask)) // 2], 2] += 0.04
    else:
        extra["environment_ids"] = mask.astype(int)
    assert not review(points, gap, mask, time, **extra).pass_mask.any()


@pytest.mark.parametrize("which", [0, -1])
def test_partial_events_at_recording_boundaries_are_not_dropped(which):
    points, gap, mask, time = example()
    mask[:] = False
    mask[which] = True
    assert not review(points, gap, mask, time).pass_mask.any()


def test_irregular_timestamps_are_used_and_invalid_uncertainty_fails():
    time = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
    points = np.c_[time, np.zeros((5, 2))]
    mask = np.array([False, True, True, True, False])
    assert review(points, np.full(5, 0.02), mask, time).pass_mask.sum() == 3
    with pytest.raises(ValueError):
        review_contact_passes(
            points, np.full(5, 0.02), mask, times=time, position_uncertainty_m=0
        )


def make_task(points, mask):
    task = build_world_tracking_tasks(
        {"arbitrary_region": {"tracking_positions": points}}
    )["arbitrary_region"]
    anchors = points.copy()
    anchors[:, 1] = 0
    return replace(
        task,
        contact_mask=mask.copy(),
        contact_positions=anchors,
        contact_modes=np.where(mask, "sticking", "none"),
    )


def test_pass_removes_its_approach_without_extending_or_changing_confirmed_neighbour():
    points, gap, mask, time = example()
    mask[24:27] = True
    gap[24:27] = 0
    task = make_task(points, mask)
    decision = review(points, gap, mask, time)
    trial = without_pass_attraction(task, decision)
    assert trial.contact_mask.sum() == 3
    _, _, assigned = approach_blend(
        points, task.contact_positions, mask, return_contact_frames=True
    )
    affected = decision.pass_mask | decision.pass_mask[assigned]
    np.testing.assert_array_equal(trial.approach[0][affected], 0)
    np.testing.assert_array_equal(
        trial.approach[0][~affected], task.approach[0][~affected]
    )
    np.testing.assert_array_equal(trial.contact_positions, task.contact_positions)
    np.testing.assert_array_equal(task.contact_mask, mask)
    cropped = slice_contact_task(trial, 5, 29)
    np.testing.assert_array_equal(cropped.approach[0], trial.approach[0][5:29])


def test_source_pass_does_not_drag_a_short_robot_root_to_the_surface():
    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <body name="root"><freejoint/><geom size=".02" mass="1"/>
        <body name="tip" pos=".2 .05 0"><geom size=".01" mass=".1"/></body>
      </body></worldbody></mujoco>""")
    points, gap, mask, time = example()
    roots = np.c_[points[:, 0], np.zeros((len(time), 2))]
    points[:, 0] += 0.3
    task = make_task(points, mask)
    poses = np.tile(model.qpos0, (len(time), 1))
    poses[:, :3] = roots
    config = {
        "iterations": 6,
        "self_collision_enabled": False,
        "lock_root_orientation": True,
    }
    effectors = {
        "effectors": {
            "arbitrary_region": {"robot_body": "tip", "swing_morphology_scale": 1.0}
        }
    }

    def solve(target):
        return retarget_sequence(
            model,
            poses,
            roots[:, None],
            ["source"],
            {"source": "root"},
            roots,
            {"arbitrary_region": target},
            effectors,
            framerate=30,
            config=config,
        )[0]

    baseline = solve(task)
    trial = solve(without_pass_attraction(task, review(points, gap, mask, time)))
    assert np.max(np.linalg.norm(baseline[:, :3] - roots, axis=1)) > 0.04
    assert np.max(np.linalg.norm(trial[:, :3] - roots, axis=1)) < 1e-6
