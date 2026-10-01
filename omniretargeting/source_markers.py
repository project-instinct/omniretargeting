"""Adapt material contact markers to the same source-only observation schema."""

import numpy as np


def marker_observations(markers):
    points, normals, regions, authored = [], [], [], []
    for name, marker in markers.items():
        # Tracking joint centers are not observed skin. Missing material
        # samples stays unknown; it must not create source contact evidence.
        if "contact_samples" not in marker:
            continue
        value = np.asarray(marker["contact_samples"], float)
        if value.ndim == 2:
            value = value[:, None]
        if value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError("Contact markers must have shape (frames, samples, 3)")
        points.append(value)
        normal = marker.get("contact_normals")
        normals.append(np.zeros_like(value) if normal is None else np.broadcast_to(normal, value.shape))
        regions.extend([name] * value.shape[1])
        mask = np.asarray(marker.get("source_contact_mask", np.zeros(len(value))), bool)
        authored.append(np.broadcast_to(mask[:, None], value.shape[:2]))
    if not points:
        return None, None
    source = dict(points=np.concatenate(points, axis=1), skin_normals=np.concatenate(normals, axis=1),
                  labels=np.asarray(regions), sample_effectors=np.asarray(regions))
    return source, np.concatenate(authored, axis=1)
