"""Immutable, source-observed local free sides of prescribed object surfaces."""

import numpy as np


def source_object_evidence(objects, table, radius_m=0.15):
    """Attach local contact-side witnesses without changing object geometry.

    Use observed, confidently facing skin on the mesh's clear side. Penetrating
    or conflicted source samples cannot reverse that side. This evidence has
    no support/control meaning, and applies equally to every source region.
    """
    objects = tuple(objects or ())
    if table is None or not objects:
        return objects
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError('Moving surface evidence radius must be finite and positive')
    observation = table.observation
    offset = len(observation['surface_poses']) - len(objects)
    if offset not in (0, 1):
        raise ValueError('Object evidence must share the source surface identities')
    result = []
    for oid, spec in enumerate(objects):
        if spec["mesh"].is_watertight and spec["mesh"].is_winding_consistent:
            result.append(spec)
            continue
        surface_id = oid + offset
        rotation, _, scale = observation['surface_poses'][surface_id]
        valid = (table.valid & table.active & observation['clear_side']
                 & (observation['confidence'] >= 0.5)
                 & (observation['surface_ids'] == surface_id))
        points = observation['material_environment'][valid]
        normals = np.einsum('tni,tij->tnj', observation['normals'], rotation)[valid]
        values = np.unique(np.column_stack((points, normals)), axis=0)
        values.setflags(write=False)
        result.append({**spec, 'collision_evidence': {
            'points': values[:, :3], 'normals': values[:, 3:],
            'radius': float(radius_m / np.max(scale)),
            'source': 'immutable clear-side source contact observations',
        }})
    return tuple(result)
