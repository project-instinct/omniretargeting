"""Object following is distinct from support, proximity and passive stillness."""

import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting.object_control import observe_object_control


def recording(frames=13, duration=1.0):
    time = np.linspace(0, duration, frames)
    phase = time / duration
    pose = np.tile(np.eye(4), (frames, 1, 1))
    pose[:, :3, :3] = Rotation.from_euler("z", phase[:, None]).as_matrix()
    pose[:, 0, 3] = 0.3 * phase
    observations = {}
    for name, point in (("controller_A", [-0.2, 0, 0]), ("unseen_region", [0.2, 0, 0])):
        world = np.einsum("fij,j->fi", pose[:, :3, :3], point) + pose[:, :3, 3]
        observations[name] = {
            "positions": world.copy(),
            "surface_points": world.copy(),
            "contact_mask": np.ones(frames, bool),
            "surface_ids": np.full(frames, 7),
            "controller": name,
        }
    return time, observations, {7: pose}


def infer(time, observations, objects):
    return observe_object_control(
        observations, objects, times=time, position_uncertainty_m=0.005
    )


@pytest.mark.parametrize("frames,duration", [(3, 0.01), (31, 1), (121, 1)])
def test_shared_object_follow_is_independent_of_rate_duration_and_region_names(
    frames, duration
):
    time, source, objects = recording(frames, duration)
    original = copy.deepcopy((source, objects))
    evidence = infer(time, source, objects)
    assert all(
        evidence.role(name, 7, f) == "control" for name in source for f in range(frames)
    )
    assert not evidence.diagnostics["grasp_stability_verified"]
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [0.7, -0.4, 0.3]).as_matrix()
    transform[:3, 3] = [4, 2, -3]
    for observation in source.values():
        for key in ("positions", "surface_points"):
            observation[key] = observation[key] @ transform[:3, :3].T + transform[:3, 3]
    transformed = infer(time, source, {7: transform @ objects[7]})
    assert transformed.decisions == evidence.decisions
    # The evidence owns decisions, not live references to observations.
    source["controller_A"]["contact_mask"][:] = False
    assert evidence.decisions == infer(time, *original).decisions


@pytest.mark.parametrize(
    "change",
    [
        "same_controller",
        "different_objects",
        "separated",
        "slipping",
        "stationary_object",
        "one_frame",
        "unknown_pose",
        "unknown_point",
    ],
)
def test_close_or_still_points_do_not_prove_object_control(change):
    time, source, objects = recording()
    a, b = source.values()
    if change == "same_controller":
        b["controller"] = a["controller"]
    elif change == "different_objects":
        b["surface_ids"][:] = 8
        objects[8] = objects[7].copy()
    elif change == "separated":
        b["positions"][:, 2] += 0.05
    elif change == "slipping":
        b["positions"][:, 2] += np.linspace(0, 0.1, len(time))
        b["surface_points"] = b["positions"].copy()
    elif change == "stationary_object":
        objects[7][:] = objects[7][0]
        for observation in source.values():
            observation["positions"][:] = observation["positions"][0]
            observation["surface_points"][:] = observation["positions"]
    elif change == "one_frame":
        b["contact_mask"][:] = False
        b["contact_mask"][6] = True
    elif change == "unknown_pose":
        objects[7][5] = np.nan
    else:
        b["positions"][5] = np.nan
    evidence = infer(time, source, objects)
    assert not any(evidence.decisions)


def test_handoffs_do_not_extend_control_past_observed_overlap():
    time, source, objects = recording()
    a, b = source.values()
    a["contact_mask"][:2] = False
    b["contact_mask"][9:] = False
    evidence = infer(time, source, objects)
    assert [bool(frame) for frame in evidence.decisions] == [False] * 2 + [True] * 7 + [
        False
    ] * 4


def test_nonrigid_object_pose_is_rejected():
    time, source, objects = recording()
    objects[7][:, :3, :3] *= 2
    with pytest.raises(ValueError, match="rigid"):
        infer(time, source, objects)
