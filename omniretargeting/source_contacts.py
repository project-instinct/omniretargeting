"""One source-only contact table for all observed skin regions.

Raw observations, temporally accepted geometry and mechanical hypotheses stay
separate. Stable sample indices address immutable source skin, never robot
vertices. Region names are opaque identifiers; there are no limb exceptions.
"""

from dataclasses import dataclass

import numpy as np

from .contact_mechanics import ContactCandidate
from .contact_schedule import (
    _apply_contact_hysteresis,
    _close_short_gaps,
    _remove_short_runs,
)
from .surface_observation import observe_contacts
from .surface_tasks import _face_frames


@dataclass(frozen=True)
class SourceContactTable:
    points: np.ndarray
    skin_normals: np.ndarray
    valid: np.ndarray
    regions: np.ndarray
    observation: dict
    active: np.ndarray
    events: tuple
    diagnostics: dict

    def candidates(
        self,
        *,
        friction,
        reconstruction_bound_m,
        normal_speed_m_s,
        touch_distance_m,
        normal_opposition,
        sticking_speed_m_s,
    ):
        """Propose bounded hypotheses, without asserting that a gap transmits force.

        Fast normal motion is retained only in the close-touch band. Tangential
        slip has its measured friction direction; it does not get an arbitrary
        static friction cone. Short hypotheses are not duration-filtered here.
        """
        values = [
            friction,
            reconstruction_bound_m,
            normal_speed_m_s,
            touch_distance_m,
            normal_opposition,
            sticking_speed_m_s,
        ]
        if not np.isfinite(values).all() or min(values) < 0:
            raise ValueError(
                "Mechanical candidate assumptions must be finite and nonnegative"
            )
        obs = self.observation
        eligible = self.valid & (obs["distance"] <= reconstruction_bound_m)
        eligible &= obs["normal_opposition"] >= normal_opposition
        eligible &= (obs["normal_speed"] <= normal_speed_m_s) | (
            obs["distance"] <= touch_distance_m
        )
        frames, correspondence = [], {}
        for frame in range(len(self.points)):
            candidates = []
            for region in np.unique(self.regions[eligible[frame]]):
                indices = np.flatnonzero(eligible[frame] & (self.regions == region))
                for surface in np.unique(obs["surface_ids"][frame, indices]):
                    selected = indices[obs["surface_ids"][frame, indices] == surface]
                    slip = obs["tangent_speed"][frame, selected] > sticking_speed_m_s
                    for mode, mask in (("sticking", ~slip), ("sliding", slip)):
                        ids = selected[mask]
                        if not len(ids):
                            continue
                        name = f"{region}@{surface}:{mode}"
                        correspondence[name] = (str(region), int(surface))
                        candidates.append(
                            ContactCandidate(
                                name,
                                self.points[frame, ids],
                                obs["environment"][frame, ids],
                                obs["normals"][frame, ids],
                                friction,
                                reconstruction_bound_m,
                                mode=mode,
                                sliding_velocity=obs["relative_velocity"][frame, ids],
                                sample_indices=ids,
                            )
                        )
            frames.append(tuple(candidates))
        return tuple(frames), correspondence


def build_source_contacts(
    full_body,
    scene_mesh,
    config,
    *,
    framerate,
    scene_objects=(),
    scene_enclosed_free_space=False,
    exit_distance_m=None,
    max_gap_s=0.0,
    min_duration_s=0.0,
    authored_mask=None,
):
    """Observe once, then filter region × surface intervals with bounded geometry.

    Hysteresis/gap filling cannot bridge invalid samples or a real geometric
    departure. Explicit source contacts survive short-run rejection, but still
    require a valid observation within the exit bound. Their original labels
    remain available to later inference; the skin itself is never changed.
    """
    if "points" in full_body:
        points = np.array(full_body["points"], float, copy=True)
        normals = np.array(full_body.get("skin_normals", np.zeros_like(points)), float, copy=True)
        valid = np.array(full_body.get("valid", np.isfinite(points).all(-1)), bool, copy=True)
        if points.ndim != 3 or points.shape[-1] != 3 or normals.shape != points.shape or valid.shape != points.shape[:2]:
            raise ValueError("Source material markers require (frames, samples, 3) points and normals")
        samples = points.shape[1]
    else:
        triangles = np.asarray(full_body["triangles"], float)
        bary = np.asarray(full_body["barycentric"], float)
        if (triangles.ndim != 4 or triangles.shape[1:] != (len(bary), 3, 3)
                or bary.shape != (len(bary), 3) or not np.isfinite(bary).all()
                or not np.allclose(bary.sum(1), 1)):
            raise ValueError("Source skin requires finite triangle and barycentric identities")
        rotations, valid = _face_frames(triangles)
        points = np.einsum("tnij,ni->tnj", triangles, bary)
        normals = rotations[..., 2].copy()
        samples = len(bary)
    regions = np.array(
        full_body.get("labels", ["unsegmented"] * samples), str, copy=True
    )
    if regions.shape != (samples,):
        raise ValueError("Source region labels must address every material sample")
    obs = observe_contacts(
        np.where(valid[..., None], points, 0),
        normals,
        valid,
        scene_mesh,
        {**config, "min_contact_frames": 1},
        framerate=framerate,
        scene_objects=scene_objects,
        scene_enclosed_free_space=scene_enclosed_free_space,
    )
    exit_distance = (
        config["distance"] if exit_distance_m is None else float(exit_distance_m)
    )
    if (
        not np.isfinite([exit_distance, max_gap_s, min_duration_s]).all()
        or exit_distance < config["distance"]
        or min(max_gap_s, min_duration_s) < 0
    ):
        raise ValueError(
            "Temporal contact bounds must be finite and contain the entry band"
        )
    authored = (
        np.zeros(valid.shape, bool)
        if authored_mask is None
        else np.asarray(authored_mask, bool)
    )
    if authored.shape != valid.shape:
        raise ValueError(
            "Authored contact labels must follow the same samples and frames"
        )
    stay = valid & (obs["distance"] <= exit_distance)
    stay &= obs["normal_opposition"] >= config["normal_opposition"]
    stay &= (obs["normal_speed"] <= config["normal_speed"]) | (
        obs["distance"] <= config["touch_distance"]
    )
    denoise = full_body.get("source_denoise")
    debounced = bool(denoise and denoise.get("applied"))
    if debounced:
        from .data_sources.source_denoise import debounce_contacts
        # Geometry still bounds gap filling; speed threshold flicker alone may
        # be debounced, but invalid/unknown surfaces and departures may not.
        geometric_stay = valid & (obs["distance"] <= exit_distance)
        geometric_stay &= obs["normal_opposition"] >= config["normal_opposition"]
    active = np.zeros_like(valid)
    sets = [set() for _ in range(len(points))]
    for region in np.unique(regions):
        group = regions == region
        for surface in np.unique(obs["surface_ids"][:, group]):
            membership = group[None] & (obs["surface_ids"] == surface)
            enter = (obs["active"] & membership).any(1)
            remain = (stay & membership).any(1)
            timeline = _apply_contact_hysteresis(enter, remain)
            timeline = (
                _close_short_gaps(timeline, int(np.floor(max_gap_s * framerate)))
                & remain
            )
            if debounced:
                velocity = np.sum(obs["relative_velocity"] * obs["normals"], axis=-1)
                normal_velocity = np.median(velocity[:, group], axis=1)
                timeline, merges = debounce_contacts(
                    timeline, framerate, normal_velocity=normal_velocity, eligible=(geometric_stay & membership).any(1),
                )
                denoise["contact_merges"].extend(
                    dict(region=str(region), surface=int(surface), stage="scene_detection",
                         fps=float(framerate), frame_space="prepared", **e)
                    for e in merges
                )
            else:
                timeline = _remove_short_runs(
                    timeline, max(1, int(np.ceil(min_duration_s * framerate)))
                )
            timeline |= (authored & membership & stay).any(1)
            permitted = geometric_stay if debounced else stay
            active |= timeline[:, None] & membership & permitted
            for frame in np.flatnonzero(timeline):
                sets[frame].add((str(region), int(surface)))
    if debounced:
        # Keep raw observations intact. Accepted short gaps are bounded touches,
        # not invented sticking anchors with a raw mode of "none" and zero weight.
        bridged = active & ~obs["active"]
        accepted_modes = obs["modes"].copy()
        accepted_modes[bridged] = "touch"
        confidence = obs["confidence"].copy()
        strength = np.clip((config["distance"]-obs["distance"]) /
                           (config["distance"]-config["touch_distance"]), 0, 1)
        geometric_confidence = strength**2*(3-2*strength)*np.clip(obs["normal_opposition"],0,1)
        confidence[bridged] = geometric_confidence[bridged]
        obs.update(accepted_modes=accepted_modes, accepted_confidence=confidence)
    events, previous = [], set()
    for frame, current in enumerate(sets):
        if current != previous:
            events.append(
                {
                    "frame": frame,
                    "enter": sorted(current - previous),
                    "leave": sorted(previous - current),
                    "held": sorted(current & previous),
                }
            )
        previous = current
    diagnostics = {
        "source_only": True,
        "frames": len(points),
        "samples": len(regions),
        "regions": sorted(set(regions)),
        "events": len(events),
        "raw_active_frame_points": int(obs["active"].sum()),
        "active_frame_points": int(active.sum()),
        "observations_modified": False,
        "exit_distance_m": exit_distance,
        "max_gap_s": max_gap_s,
        "min_duration_s": min_duration_s,
        "observation": obs["diagnostics"],
    }
    for array in (points, normals, valid, regions, active, *obs.values()):
        if isinstance(array, np.ndarray):
            array.setflags(write=False)
    return SourceContactTable(
        points, normals, valid, regions, obs, active, tuple(events), diagnostics
    )
