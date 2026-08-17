"""Infer where the source touches the scene, and anchor those contacts.

Preserving an interaction means the robot has to touch what the source touched,
where the source touched it.  A Laplacian interaction mesh preserves relative
shape, not contact: it is free to place a whole limb a few centimetres off the
surface at no cost, because every edge around it stretches by the same small
amount.  Non-penetration constraints do not close that gap either, since
hovering above a surface violates nothing.

So contact has to be an explicit objective.  This module answers two questions
from the source motion alone:

* when is an end effector in contact?  Proximity by itself is not enough: a
  foot passing an inch above the floor is close but not planted.  Contact
  additionally requires the effector to be moving slowly, and to stay that way
  for a few consecutive frames, which is what separates a plant from a
  fly-by.
* where is that contact?  Each contact interval is anchored to a single point
  on the scene surface, taken from the opening frames of the interval, and
  every frame in the interval targets that same point.  Anchoring per interval
  rather than per frame is what stops the effector sliding: a per-frame
  projection would happily track a foot skating across the floor, since each
  individual frame is close to the surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "ContactSpec",
    "infer_contact_targets",
    "contact_runs",
    "estimate_float_bias",
    "hysteretic_state",
]


def _blend_ramp(length: int, blend: int) -> np.ndarray:
    """Weights that ease in and out of a contact interval.

    A contact task that switches on at full strength between one frame and the
    next asks the solver for a different pose than it asked for a frame
    earlier, and the jump lands in the output as an acceleration spike. Ramping
    the weight over a few frames removes the discontinuity, which is how foot
    IK in animation blends its own contact constraints.

    Off by default, because measured here the trade is bad: ramping over five
    frames buys about a tenth off the ninety-fifth percentile acceleration and
    nothing at all off the peak, while foot slide within a contact goes from
    0.5 cm to 3.2 cm. Slide is visible and acceleration at that margin is not.
    Temporal smoothness is the better lever for the same goal.
    """
    w = np.ones(length)
    if blend <= 0 or length == 0:
        return w
    # Leave at least one frame at full weight in the middle. Ramping both ends
    # over half the interval each would consume a short plant entirely and hold
    # the foot at reduced weight for its whole duration.
    span = min(blend, (length - 1) // 2)
    if span <= 0:
        return w
    ease = 0.5 * (1.0 - np.cos(np.pi * (np.arange(1, span + 1) / (span + 1))))
    w[:span] = ease
    w[-span:] = ease[::-1]
    return w


GRAVITY = 9.80665


@dataclass(frozen=True)
class ContactSpec:
    """When a frame counts as contact, in units that survive a change of scale.

    The absolute thresholds this replaces meant different things in different
    worlds.  A 5 cm proximity gate is 23 percent of this subject's foot length;
    on a scene scaled to 0.74 it became 31 percent, and that difference alone
    moved detection from 27 and 40 contact frames to 4 and 15.  A speed gate in
    metres per second has the same problem across subject sizes, and a gate in
    body-lengths-per-sample additionally changes when the same continuous motion
    is resampled.

    So distance is expressed as a fraction of the effector's own length plus an
    absolute floor for scene noise, and speed as a fraction of the gravitational
    speed scale of the source's leg.  Both are dimensionless and both transfer.

    Separate enter and leave thresholds give hysteresis, so a frame sitting on
    the boundary does not flicker in and out of contact and drag the solver with
    it.
    """

    # Fractions of effector length.  Defaults reproduce roughly what the
    # previous absolute 5 cm gate did on this subject once its float bias is
    # removed, with the leave gate twice the enter gate.
    enter_gap_fraction: float = 0.10
    leave_gap_fraction: float = 0.20
    # Absolute floor, for scenes whose reconstruction noise does not scale with
    # the subject.
    gap_floor: float = 0.005

    # Fractions of sqrt(g * leg_length).
    enter_speed_fraction: float = 0.19
    leave_speed_fraction: float = 0.35

    min_frames: int = 3
    anchor_frames: int = 3
    blend_frames: int = 0

    # Percentile of the slow-frame distance distribution taken as the source's
    # systematic float above its own scan.  A monocular reconstruction sits a
    # few centimetres off its own scene mesh even when planted; measured here
    # that is 2.6 to 3.2 cm, and pretending it is contact at 5 cm only works
    # because 5 cm happens to be larger.
    bias_percentile: float = 10.0
    estimate_bias: bool = True

    def __post_init__(self) -> None:
        if self.enter_gap_fraction <= 0.0:
            raise ValueError("enter_gap_fraction must be positive")
        if self.leave_gap_fraction < self.enter_gap_fraction:
            raise ValueError("leave_gap_fraction must not be below enter_gap_fraction")
        if self.enter_speed_fraction <= 0.0:
            raise ValueError("enter_speed_fraction must be positive")
        if self.leave_speed_fraction < self.enter_speed_fraction:
            raise ValueError("leave_speed_fraction must not be below enter_speed_fraction")
        if self.gap_floor < 0.0:
            raise ValueError("gap_floor must not be negative")
        if self.min_frames < 1:
            raise ValueError("min_frames must be at least 1")
        if self.anchor_frames < 1:
            raise ValueError("anchor_frames must be at least 1")
        if self.blend_frames < 0:
            raise ValueError("blend_frames must not be negative")
        if not 0.0 < self.bias_percentile < 50.0:
            raise ValueError("bias_percentile must lie in (0, 50)")


def estimate_float_bias(distance: np.ndarray, slow: np.ndarray, percentile: float) -> float:
    """How far the source sits off its own scan even when it is planted.

    Taken from the lower envelope of the distance distribution over slow frames
    rather than from its centre: the aim is the offset present when the effector
    is as close as it ever gets, not the average over a clip that includes
    airborne stretches.
    """
    sample = distance[slow] if np.any(slow) else distance
    if sample.size == 0:
        return 0.0
    return float(max(0.0, np.percentile(sample, percentile)))


def hysteretic_state(enter: np.ndarray, leave: np.ndarray) -> np.ndarray:
    """Latch contact on ``enter`` and hold it until ``leave`` says otherwise."""
    state = np.zeros(len(enter), dtype=bool)
    active = False
    for i in range(len(enter)):
        if active:
            active = not leave[i]
        else:
            active = bool(enter[i])
        state[i] = active
    return state


def contact_runs(flags: np.ndarray, min_frames: int) -> List[Tuple[int, int]]:
    """Half-open ``[start, end)`` runs of consecutive True at least ``min_frames`` long."""
    padded = np.concatenate(([0], np.asarray(flags, dtype=np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded) != 0).reshape(-1, 2)
    return [(int(a), int(b)) for a, b in edges if b - a >= min_frames]


def infer_contact_targets(
    effector_points: Dict[str, np.ndarray],
    terrain,
    framerate: Optional[float],
    spec: Optional[ContactSpec] = None,
    effector_lengths: Optional[Dict[str, float]] = None,
    leg_length: Optional[float] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], dict]:
    """Anchor every contact interval of every effector to the scene.

    Args:
        effector_points: ``{name: (T, 3)}`` world positions of the point that
            would touch the scene, in the same frame as ``terrain``.
        terrain: a trimesh surface.
        framerate: used to convert per-frame displacement into speed. When it
            is unknown the speed gate is skipped and proximity alone decides.
        effector_lengths: ``{name: metres}`` the effector's own size, used to
            express the proximity gate as a fraction of it. Falls back to the
            leg length, and then to a bare absolute floor.
        leg_length: source leg length in metres, used for the gravitational
            speed scale. Without it the speed gate is skipped rather than
            applied in units that do not transfer.

    Returns:
        ``(targets, normals, weights, stats)``. ``targets[name]`` is ``(T, 3)``
        with NaN on frames that are not in contact; ``normals[name]`` is the
        anchored surface normal, also NaN outside contact; ``weights[name]``
        carries the detector's confidence, eased across interval boundaries when
        ``blend_frames`` asks for it.
    """
    spec = spec or ContactSpec()
    targets: Dict[str, np.ndarray] = {}
    normals: Dict[str, np.ndarray] = {}
    weights: Dict[str, np.ndarray] = {}
    stats: dict = {}

    speed_scale = (
        float(np.sqrt(GRAVITY * leg_length))
        if leg_length and leg_length > 1e-6
        else None
    )

    for name, points in effector_points.items():
        points = np.asarray(points, dtype=float)
        n_frames = len(points)
        target = np.full((n_frames, 3), np.nan)
        normal = np.full((n_frames, 3), np.nan)
        weight = np.zeros(n_frames)

        closest, distance, face_ids = terrain.nearest.on_surface(points)
        distance = np.asarray(distance, dtype=float)

        # Speed, made dimensionless by the gravitational scale of the source's
        # own leg.  Per-sample displacement would change under resampling of the
        # same continuous motion; this does not.
        if framerate and speed_scale:
            step = np.linalg.norm(np.diff(points, axis=0), axis=1) * float(framerate)
            speed = np.concatenate(([step[0] if len(step) else 0.0], step)) / speed_scale
            slow_enter = speed < spec.enter_speed_fraction
            slow_leave = speed > spec.leave_speed_fraction
        else:
            speed = np.zeros(n_frames)
            slow_enter = np.ones(n_frames, dtype=bool)
            slow_leave = np.zeros(n_frames, dtype=bool)

        # Remove the source's systematic float before deciding anything.  A
        # reconstruction that sits three centimetres off its own scan while
        # planted would otherwise need a gate loose enough to also admit a foot
        # genuinely passing overhead.
        bias = (
            estimate_float_bias(distance, slow_enter, spec.bias_percentile)
            if spec.estimate_bias
            else 0.0
        )
        gap = np.maximum(distance - bias, 0.0)

        reference = None
        if effector_lengths and effector_lengths.get(name):
            reference = float(effector_lengths[name])
        elif leg_length:
            reference = float(leg_length)
        enter_gap = spec.gap_floor + (
            spec.enter_gap_fraction * reference if reference else 0.0
        )
        leave_gap = spec.gap_floor + (
            spec.leave_gap_fraction * reference if reference else 0.0
        )

        state = hysteretic_state(
            (gap < enter_gap) & slow_enter,
            (gap > leave_gap) | slow_leave,
        )
        runs = contact_runs(state, spec.min_frames)
        face_normals = terrain.face_normals[np.asarray(face_ids, dtype=int)]

        # Confidence, rather than a boolean: how far inside the gates the frame
        # sits.  It is carried through as the residual's per-frame weight, so a
        # marginal frame contributes proportionally rather than either fully or
        # not at all.
        confidence = np.clip(
            (leave_gap - gap) / max(leave_gap - enter_gap, 1e-9), 0.0, 1.0
        )
        if framerate and speed_scale:
            speed_conf = np.clip(
                (spec.leave_speed_fraction - speed)
                / max(spec.leave_speed_fraction - spec.enter_speed_fraction, 1e-9),
                0.0,
                1.0,
            )
            confidence = np.minimum(confidence, speed_conf)

        for start, end in runs:
            # Anchor on the most confident frames of the interval rather than
            # its opening ones: the opening frames are where the effector is
            # still settling, and on a noisy scan they are the least reliable
            # samples in the run.
            span = np.arange(start, end)
            order = span[np.argsort(-confidence[span])][: spec.anchor_frames]
            anchor = np.median(closest[order], axis=0)
            anchor_normal = np.median(face_normals[order], axis=0)
            norm = np.linalg.norm(anchor_normal)
            anchor_normal = anchor_normal / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0])
            target[start:end] = anchor
            normal[start:end] = anchor_normal
            weight[start:end] = confidence[start:end] * _blend_ramp(
                end - start, spec.blend_frames
            )

        targets[name] = target
        normals[name] = normal
        weights[name] = weight
        stats[name] = {
            "contact_frames": int(np.isfinite(target[:, 0]).sum()),
            "runs": len(runs),
            "median_distance": float(np.median(distance)),
            "float_bias": bias,
            "enter_gap": enter_gap,
            "leave_gap": leave_gap,
            "median_confidence": float(np.median(weight[weight > 0])) if np.any(weight > 0) else 0.0,
        }

    return targets, normals, weights, stats
