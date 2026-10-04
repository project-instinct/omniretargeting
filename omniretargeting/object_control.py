"""Source-only evidence for coordinated material-relative object following.

Co-motion is evidence for a task worth preserving, not proof of grasp stability,
applied forces, intent, or which agent caused the object's motion. No robot or
body-part names, minimum duration, or reachability parameters enter this module.
"""

from itertools import combinations

import numpy as np

from .contact_schedule import _iter_true_runs


class SourceObjectControlEvidence:
    def __init__(self, decisions, events):
        self.decisions = tuple(frozenset(frame) for frame in decisions)
        self.frames = len(self.decisions)
        self.diagnostics = {
            "method": "source_material_relative_co_motion",
            "events": events,
            "control_frames": sum(bool(frame) for frame in decisions),
            "grasp_stability_verified": False,
            "causal_control_verified": False,
            "force_distribution_inferred": False,
        }

    def role(self, region, surface_id, frame):
        return (
            "control"
            if (str(region), int(surface_id)) in self.decisions[frame]
            else None
        )


def observe_object_control(
    observations, object_poses, *, times, position_uncertainty_m
):
    """Recognize pairs following the same object within declared source error.

    Each region supplies ``positions`` of a stable source material point,
    corresponding actual ``surface_points``, ``contact_mask``, ``surface_ids``
    and a ``controller`` identity. Two skin patches on the same controller do
    not count as two independent observations. Poses are object-to-world 4x4
    transforms, keyed by the same surface IDs used by the contact detector.

    A candidate run is promoted only if both points fit fixed object-local
    anchors within the observation budget, the observed gaps fit that budget,
    and object-induced travel at EACH anchor exceeds combined endpoint error.
    Stationary objects, missing observations, slipping, and one-frame glimpses
    stay undecided. They are not declarations of absence of interaction.
    """
    time = np.asarray(times, float)
    budget = float(position_uncertainty_m)
    if (
        time.ndim != 1
        or not np.isfinite(time).all()
        or np.any(np.diff(time) <= 0)
        or not np.isfinite(budget)
        or budget <= 0
    ):
        raise ValueError(
            "Object-control observations need increasing times and a positive metric uncertainty"
        )
    frames = len(time)
    poses = {}
    for key, value in object_poses.items():
        pose = np.asarray(value, float)
        if pose.shape != (frames, 4, 4):
            raise ValueError("Object poses must share the complete source timeline")
        finite = np.isfinite(pose).all(axis=(1, 2))
        rotation = pose[finite, :3, :3]
        if (
            not np.allclose(pose[finite, 3], [0, 0, 0, 1])
            or not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3))
            or not np.allclose(np.linalg.det(rotation), 1)
        ):
            raise ValueError("Object-control poses must be rigid, unscaled transforms")
        poses[int(key)] = (pose, finite)
    source = {}
    for region, observation in observations.items():
        point = np.asarray(observation["positions"], float)
        surface = np.asarray(observation["surface_points"], float)
        mask = np.asarray(observation["contact_mask"], bool)
        ids = np.asarray(observation["surface_ids"])
        if (
            point.shape != (frames, 3)
            or surface.shape != point.shape
            or mask.shape != (frames,)
            or ids.shape != (frames,)
        ):
            raise ValueError(
                "Object-control observations must share the complete source timeline"
            )
        source[str(region)] = (point, surface, mask, ids, observation["controller"])
    decisions, events = [set() for _ in range(frames)], []
    for a, b in combinations(source, 2):
        pa, sa, ma, ia, ca = source[a]
        pb, sb, mb, ib, cb = source[b]
        if ca == cb:
            continue
        overlap = ma & mb & (ia == ib)
        pair_points = pair_surfaces = None
        for object_id, (pose, finite_pose) in poses.items():
            mask = overlap & (ia == object_id)
            if not mask.any():
                continue
            for start, stop in _iter_true_runs(mask):
                event = {
                    "regions": [a, b],
                    "surface_id": object_id,
                    "start": start,
                    "stop": stop,
                    "decision": "undecided",
                }
                events.append(event)
                if pair_points is None:
                    pair_points = np.stack([pa, pb], axis=1)
                    pair_surfaces = np.stack([sa, sb], axis=1)
                points = pair_points[start:stop]
                surfaces = pair_surfaces[start:stop]
                if (
                    not finite_pose[start:stop].all()
                    or not np.isfinite(points).all()
                    or not np.isfinite(surfaces).all()
                ):
                    event["reason"] = "unknown_observation"
                    continue
                gap = np.linalg.norm(points - surfaces, axis=-1)
                event["max_observed_gap_m"] = float(gap.max())
                if np.any(gap > budget):
                    event["reason"] = "gap_exceeds_source_uncertainty"
                    continue
                rotation, translation = (
                    pose[start:stop, :3, :3],
                    pose[start:stop, :3, 3],
                )
                local = np.einsum(
                    "fji,fcj->fci", rotation, points - translation[:, None]
                )
                anchor = local.mean(axis=0)
                drift = np.linalg.norm(local - anchor, axis=-1).max(axis=0)
                event["material_anchor_error_m"] = drift.tolist()
                if np.any(drift > budget):
                    event["reason"] = "relative_slip_or_changing_material_point"
                    continue
                tracked = (
                    np.einsum("fij,cj->fci", rotation, anchor) + translation[:, None]
                )
                travel = np.linalg.norm(tracked - tracked[0], axis=-1).max(axis=0)
                event["object_induced_travel_m"] = travel.tolist()
                if np.any(travel <= 2 * budget):
                    event["reason"] = (
                        "motion_not_distinguishable_from_observation_error"
                    )
                    continue
                event.update(decision="control", reason="coordinated_object_follow")
                for frame in range(start, stop):
                    decisions[frame].update(((a, object_id), (b, object_id)))
    return SourceObjectControlEvidence(decisions, events)
