"""Explicit sole-material approximation for skeleton-only source adapters.

BVH carries no measured skin. Heel height uses the existing skeleton-source
approximation, 0.039 of source height; neutral pitch is inferred from the
stillest source frames. This is source geometry metadata, not a source motion
correction, and never consults the scene or the target robot.
"""

import numpy as np

ANKLE_HEIGHT_PER_BODY_HEIGHT = 0.039
STILL_FRACTION = 0.1


def _unit(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-10)


def _speed(points, framerate):
    speed = np.linalg.norm(np.diff(points, axis=0), axis=1) * framerate
    return np.concatenate([speed[:1], speed]) if len(speed) else np.zeros(len(points))


def sole_frame(knee, ankle, toe, framerate):
    """Rigid sole axes following the source leg plane and neutral foot pitch."""
    line = _unit(toe - ankle)
    leg = knee - ankle
    normal = _unit(leg - np.sum(leg * line, axis=1, keepdims=True) * line)
    stillness = np.maximum(_speed(ankle, framerate), _speed(toe, framerate))
    count = max(1, int(np.ceil(STILL_FRACTION * len(ankle))))
    still = np.argsort(stillness, kind="stable")[:count]
    pitch = float(np.median(np.arcsin(np.clip(-line[still, 2], -1.0, 1.0))))
    forward = np.cos(pitch) * line + np.sin(pitch) * normal
    up = -np.sin(pitch) * line + np.cos(pitch) * normal
    return forward, up, pitch


def skeleton_sole_markers(positions, names, parents, feet, body_height, framerate):
    """Heel and ball material samples; no invented whole-body contact skin."""
    positions = np.asarray(positions, dtype=float)
    parents = np.asarray(parents, dtype=int)
    ankle_height = ANKLE_HEIGHT_PER_BODY_HEIGHT * float(body_height)
    markers = {}
    for effector, joint in feet.items():
        if joint not in names:
            continue
        index = names.index(joint)
        children = np.flatnonzero(parents == index)
        if parents[index] < 0 or not len(children):
            continue
        knee, ankle, toe = (
            positions[:, parents[index]],
            positions[:, index],
            positions[:, children[0]],
        )
        forward, up, _ = sole_frame(knee, ankle, toe, framerate)
        heel = ankle - ankle_height * up
        ball = toe - np.sum((toe - heel) * up, axis=1, keepdims=True) * up
        samples = np.stack([heel, ball], axis=1)
        markers[effector] = {
            "tracking_positions": samples.mean(axis=1),
            "contact_samples": samples,
            "contact_normals": np.repeat(-up[:, None], 2, axis=1),
            "forward_directions": forward,
            "up_directions": up,
            "world_target": True,
        }
    return markers
