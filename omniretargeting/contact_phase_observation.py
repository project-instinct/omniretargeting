"""Clearance evidence from original endpoint samples, independent of skin fitting."""

import numpy as np

from .surface_observation import ContactSurface


def observe_effector_clearances(
    markers, scene, priority, *, framerate=30.0, scene_objects=()
):
    """Minimum distance only when every sample is outside every known surface.

    Unknown observations remain NaN. In particular, an inactive contact mask or
    an infinite distance from a world-tracking-only task is not airborne evidence.
    No source coordinates, contact anchors or contact intervals are changed.
    """
    selected = {
        name: marker
        for name, marker in markers.items()
        if priority.get("effectors", {})
        .get(name, {})
        .get("preserve_world_target", False)
        and "contact_samples" in marker
    }
    if not selected:
        return {}
    frames = len(next(iter(selected.values()))["contact_samples"])
    surfaces = []
    if scene is not None and len(scene.faces):
        surfaces.append(ContactSurface(scene, frames))
    for spec in scene_objects or ():
        surfaces.append(
            ContactSurface(
                spec["mesh"],
                frames,
                translations=spec.get("translations"),
                rotations=spec.get("rotations"),
                scales=spec.get("scales"),
                name=spec.get("name", "object"),
            )
        )
    result = {}
    for name, marker in selected.items():
        points = np.asarray(marker["contact_samples"], float)
        if points.ndim == 2:
            points = points[:, None]
        if points.ndim != 3 or points.shape[0] != frames or points.shape[-1] != 3:
            raise ValueError(
                "Endpoint samples must share the original (frames, samples, 3) timeline"
            )
        if not points.shape[1]:
            continue
        valid = np.isfinite(points).all(axis=(1, 2))
        distance = np.full(frames, np.inf)
        observed = np.where(np.isfinite(points), points, 0)
        for surface in surfaces:
            d, _, _, _, _, outside = surface.observe(
                observed, framerate, return_clearance_side=True
            )
            valid &= (outside & np.isfinite(d)).all(axis=1)
            distance = np.minimum(distance, d.min(axis=1))
        result[name] = np.where(valid & np.isfinite(distance), distance, np.nan)
        result[name].setflags(write=False)
    return result
