"""Primary end-effector contacts on objects in their original metric poses."""

from dataclasses import replace

import numpy as np

from .contact_correspondence import coherent_sample_indices
from .contact_schedule import (
    _apply_contact_hysteresis,
    _iter_true_runs,
    _remove_short_runs,
)
from .surface_observation import ContactSurface


def _anchor_sticking_contacts(values, surfaces, config):
    """Keep one material point for each uninterrupted sticking interval.

    Nearest skin samples can exchange rank despite very little hand motion.
    Reprojecting the selected sample every frame then moves the target around
    the object. Use the interval's observed onset point, transported by the
    original object transform, just as static sticking uses a fixed anchor.
    """
    if not config.get("anchor_contacts", True):
        return
    for environment, surface in enumerate(surfaces, 1):
        mask = (
            values["contact_mask"]
            & (values["environment_ids"] == environment)
            & (values["contact_modes"] == "sticking")
        )
        for start, end in _iter_true_runs(mask):
            rotation = surface.rotations[start]
            anchor = (
                (values["surface_points"][start] - surface.translations[start])
                @ rotation / surface.scales[start]
            )
            normal = values["surface_normals"][start] @ rotation
            normals = np.einsum("i,tji->tj", normal, surface.rotations[start:end])
            points = (
                np.einsum("i,tji->tj", anchor, surface.rotations[start:end])
                * surface.scales[start:end, None] + surface.translations[start:end]
            )
            values["surface_points"][start:end] = points
            values["surface_normals"][start:end] = normals
            values["contact_positions"][start:end] = (
                points + float(config.get("surface_clearance", 0)) * normals
            )
            if config.get("anchor_contact_orientation", True):
                for key in ("forward_directions", "up_directions"):
                    if key in values:
                        local = values[key][start] @ rotation
                        values[key][start:end] = np.einsum(
                            "i,tji->tj", local, surface.rotations[start:end]
                        )


def merge_object_contacts(tasks, markers, objects, *, framerate, config=None):
    """Observe in material coordinates, then keep targets in world coordinates.

    A hand carried by an object can have high world speed and zero relative
    speed. Its target follows the recorded object pose on every frame; freezing
    a world anchor would detach that hand. Scene and object candidates compete
    only when active, using their actual surface distance in metres.
    """
    cfg = dict(config or {})
    if not np.isfinite(framerate) or framerate <= 0:
        raise ValueError("Object contacts require a positive source framerate")
    result = dict(tasks)
    if not objects or not tasks:
        return result
    frames = len(next(iter(tasks.values())).tracking_positions)
    surfaces = [
        ContactSurface(
            spec["mesh"],
            frames,
            translations=spec.get("translations"),
            rotations=spec.get("rotations"),
            scales=spec.get("scales"),
            name=spec.get("name", "object"),
        )
        for spec in objects
    ]
    for name, original in tasks.items():
        if name not in markers:
            continue
        marker = markers[name]
        samples = np.asarray(
            marker.get("contact_samples", original.tracking_positions[:, None]), float
        )
        if (
            samples.ndim != 3
            or samples.shape[0] != frames
            or samples.shape[-1] != 3
            or not np.isfinite(samples).all()
        ):
            raise ValueError(
                "Object contact samples must be finite and follow the source timeline"
            )
        eff = {**cfg, **cfg.get("effectors", {}).get(name, {})}
        distance = float(eff.get("distance_threshold", 0.06))
        exit_distance = max(
            distance, float(eff.get("distance_exit_threshold", 1.25 * distance))
        )
        normal_limit = float(eff.get("sliding_normal_speed_threshold", 0.25))
        sticking_limit = float(eff.get("sticking_tangential_speed_threshold", 0.12))
        touch_distance = min(float(eff.get("touch_distance", 0.008)), distance)
        if min(distance, normal_limit, sticking_limit, touch_distance) <= 0:
            raise ValueError("Object contact thresholds must be positive")
        values = {
            key: np.array(getattr(original, key), copy=True)
            for key in (
                "contact_positions",
                "surface_points",
                "surface_normals",
                "distances",
                "relative_speeds",
                "contact_confidence",
                "contact_mask",
            )
        }
        values["contact_modes"] = np.array(
            original.contact_modes
            if original.contact_modes is not None
            else np.where(original.contact_mask, "sticking", "none"),
            dtype="U10",
            copy=True,
        )
        values["transition_weights"] = np.array(
            original.transition_weights
            if original.transition_weights is not None
            else np.ones(frames),
            copy=True,
        )
        values["environment_ids"] = np.array(
            original.environment_ids
            if original.environment_ids is not None
            else np.zeros(frames),
            dtype=int,
            copy=True,
        )
        for key in ("forward_directions", "up_directions"):
            if getattr(original, key) is not None:
                values[key] = np.array(getattr(original, key), copy=True)
        for surface_id, surface in enumerate(surfaces, 1):
            distances, points, normals, velocity, local = surface.observe(
                samples, framerate
            )
            material_points = np.einsum('tni,tij->tnj', points - surface.translations[:, None],
                                         surface.rotations) / surface.scales[:, None, None]
            selected = coherent_sample_indices(
                distances / surface.scales[:, None], material_points, local,
                max_distance=exit_distance / surface.scales[:, None],
                switch_weight=eff.get('contact_sample_switch_weight', 1.))
            rows = np.arange(frames)
            d, p, n, v = (
                value[rows, selected]
                for value in (distances, points, normals, velocity)
            )
            normal_speed = np.abs(np.sum(v * n, axis=1))
            tangent_speed = np.linalg.norm(
                v - np.sum(v * n, axis=1)[:, None] * n, axis=1
            )
            enter = (d <= distance) & (
                (normal_speed <= normal_limit) | (d <= touch_distance)
            )
            stay = (d <= exit_distance) & (
                (normal_speed <= normal_limit) | (d <= touch_distance)
            )
            if marker.get("up_directions") is not None and eff.get(
                "require_normal_alignment", True
            ):
                alignment = np.sum(np.asarray(marker["up_directions"]) * n, axis=1)
                facing = alignment >= np.cos(
                    float(eff.get("normal_alignment_threshold", np.pi / 6))
                )
                enter &= facing
                stay &= facing
            active = _apply_contact_hysteresis(enter, stay)
            active = _remove_short_runs(
                active, max(int(eff.get("min_contact_frames", 1)), 1)
            )
            take = active & (~values["contact_mask"] | (d < values["distances"]))
            modes = np.full(frames, "transition", dtype="U10")
            modes[(normal_speed <= normal_limit) & (tangent_speed > sticking_limit)] = (
                "sliding"
            )
            modes[
                (normal_speed <= normal_limit) & (tangent_speed <= sticking_limit)
            ] = "sticking"
            confidence = np.clip(
                (exit_distance - d) / max(exit_distance - touch_distance, 1e-8), 0, 1
            )
            replacements = {
                "contact_positions": p + float(eff.get("surface_clearance", 0)) * n,
                "surface_points": p,
                "surface_normals": n,
                "distances": d,
                "relative_speeds": np.linalg.norm(v, axis=1),
                "contact_confidence": np.maximum(confidence, 0.5),
                "contact_mask": active,
                "contact_modes": modes,
                "transition_weights": np.ones(frames),
                "environment_ids": np.full(frames, surface_id),
            }
            for key, value in replacements.items():
                values[key][take] = value[take]
            for key in ("forward_directions", "up_directions"):
                if key in values and marker.get(key) is not None:
                    values[key][take] = np.asarray(marker[key])[take]
        _anchor_sticking_contacts(values, surfaces, eff)
        result[name] = replace(original, **values)
    return result
