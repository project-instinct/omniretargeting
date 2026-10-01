"""Object-control evidence derived from the surface contact observations."""

from types import SimpleNamespace

import numpy as np
import trimesh

from omniretargeting.object_control_evidence import build_object_control_evidence


def _task(parts, travel, frames=20):
    box = trimesh.creation.box(extents=[.4, .4, .4])
    translations = np.zeros((frames, 3))
    translations[:, 0] = np.linspace(0., travel, frames)
    local = np.array([[-.2, 0., 0.], [-.2, .05, 0.], [.2, 0., 0.], [.2, .05, 0.]])[:len(parts)]
    environment = translations[:, None] + local[None]
    task = SimpleNamespace(
        environment_points=environment, source_vectors=np.zeros_like(environment),
        active=np.ones((frames, len(parts)), bool), surface_ids=np.ones((frames, len(parts)), int),
        bindings=[SimpleNamespace(part=p) for p in parts], cfg={"distance": .03})
    objects = [{"mesh": box, "translations": translations,
                "rotations": np.tile(np.eye(3), (frames, 1, 1))}]
    return task, objects, frames


def test_two_hands_following_a_moving_object_are_control():
    task, objects, frames = _task(["left_hand", "left_hand", "right_hand", "right_hand"], .5)
    evidence = build_object_control_evidence(task, objects, frames, 30., scene_present=True)
    assert all(evidence.role(part, 1, f) == "control"
               for part in ("left_hand", "right_hand") for f in range(frames))
    assert evidence.role("left_hand", 0, 0) is None  # the static scene is never control
    assert evidence.diagnostics["control_region_frames"] == {"left_hand": frames, "right_hand": frames}


def test_patches_of_one_hand_are_not_independent_observations():
    task, objects, frames = _task(["left_hand", "left_hand"], .5)
    assert build_object_control_evidence(task, objects, frames, 30., scene_present=True) is None


def test_stationary_object_stays_undecided():
    task, objects, frames = _task(["left_hand", "left_hand", "right_hand", "right_hand"], 0.)
    evidence = build_object_control_evidence(task, objects, frames, 30., scene_present=True)
    assert not any(evidence.decisions)
