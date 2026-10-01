"""Source object-control evidence from whole-body skin contact observations.

Every retained skin sample of the surface contact task is a fixed material
point of the source body, with its observed nearest object point, contact mask
and surface ID. Samples of one body region share a controller, so two patches
of the same hand never count as independent observations. Object poses are the
prescribed rigid transforms of the moving objects. The observation budget is
the detector's own contact distance: no new threshold enters. Decisions are
reported per body region, as the contact roles look them up.

Scale changes of a prescribed object enter as material drift of its anchors;
they are not removed, so a scaled object can only fail to be promoted.
"""

from collections import Counter

import numpy as np
from scipy.spatial.transform import Rotation

from .object_control import SourceObjectControlEvidence, observe_object_control
from .surface_observation import ContactSurface


def build_object_control_evidence(surface_task, scene_objects, frames, framerate, *, scene_present):
    """Evidence keyed by (region, surface ID), or None without moving objects.

    Surface IDs follow the contact observation: the static scene is 0 when it
    exists and moving objects follow in order, as endpoint environment IDs do.
    """
    if surface_task is None or not scene_objects:
        return None
    offset = 1 if scene_present else 0
    poses = {}
    for index, spec in enumerate(scene_objects):
        surface = ContactSurface(
            spec["mesh"], frames,
            translations=spec.get("translations"), rotations=spec.get("rotations"),
            scales=spec.get("scales"), name=spec.get("name", "object"),
        )
        pose = np.tile(np.eye(4), (frames, 1, 1))
        # Single-precision recordings are orthonormal only to ~1e-7; use the
        # nearest exact rotation (a change far below the observation budget).
        pose[:, :3, :3] = Rotation.from_matrix(surface.rotations).as_matrix()
        pose[:, :3, 3] = surface.translations
        poses[index + offset] = pose
    skin = surface_task.environment_points + surface_task.source_vectors
    moving = np.isin(surface_task.surface_ids, list(poses))
    observations, regions = {}, {}
    for index, binding in enumerate(surface_task.bindings):
        mask = surface_task.active[:, index] & moving[:, index]
        if not mask.any():
            continue
        key = f"{binding.part}#{index}"
        regions[key] = binding.part
        observations[key] = {
            "positions": skin[:, index],
            "surface_points": surface_task.environment_points[:, index],
            "contact_mask": mask,
            "surface_ids": surface_task.surface_ids[:, index],
            "controller": binding.part,
        }
    if len({regions[key] for key in observations}) < 2:
        return None
    samples = observe_object_control(
        observations, poses,
        times=np.arange(frames) / float(framerate),
        position_uncertainty_m=float(surface_task.cfg["distance"]),
    )
    decisions = [
        {(regions[key], int(surface)) for key, surface in frame}
        for frame in samples.decisions
    ]
    # Sample-pair events are many; keep per region pair outcomes and reasons.
    summary = Counter()
    for event in samples.diagnostics["events"]:
        pair = tuple(sorted(regions[key] for key in event["regions"]))
        summary[(pair, int(event["surface_id"]), event["decision"], event.get("reason"))] += 1
    events = [
        {"regions": list(pair), "surface_id": surface, "decision": decision,
         "reason": reason, "sample_pair_runs": count}
        for (pair, surface, decision, reason), count in sorted(summary.items(), key=str)
    ]
    evidence = SourceObjectControlEvidence(decisions, events)
    evidence.diagnostics.update(
        observation_budget_m=float(surface_task.cfg["distance"]),
        observation_budget_source="surface_contact.distance",
        material_samples=len(observations),
        control_region_frames={
            region: sum(any(r == region for r, _ in frame) for frame in decisions)
            for region in sorted(set(regions.values()))
        },
    )
    return evidence
