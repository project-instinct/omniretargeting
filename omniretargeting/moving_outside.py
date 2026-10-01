"""Recheck mandatory material samples that have lost local side observations.

Only an actual outside certificate can replace an earlier penetrating
observation. Samples on geoms skipped by broad phase are still transformed
and checked here. Evidence radii and the set of mandatory samples never grow
or shrink as a side effect of this check.
"""

from collections import defaultdict

import mujoco
import numpy as np


def certified_outside(owner, pose, time, required):
    from .moving_scene import interpolate_object

    if not required:
        return set()
    owner.data.qpos[:] = pose
    mujoco.mj_forward(owner.model, owner.data)
    groups = defaultdict(list)
    for oid, gid, sample in required:
        groups[int(oid), int(gid)].append(int(sample))
    shells = {s['geom_id']: s for s in owner.shells}
    proven = set()
    for (oid, gid), ids in sorted(groups.items()):
        obj = owner.objects[oid]
        ids = np.array(sorted(ids), dtype=int)
        shell = shells[gid]
        translation, rotation, scale = interpolate_object(obj['surface'], time)
        world = (shell['offsets'][ids] @ owner.data.geom_xmat[gid].reshape(3, 3).T
                 + owner.data.geom_xpos[gid])
        local = (world - translation) @ rotation / scale
        query = obj['query']
        outside = query.outside_hull(local)
        remaining = np.flatnonzero(~outside)
        # SceneQuery marks an unsigned-only component uncertain at EVERY
        # distance, even if all other components are known. Inside the hull
        # such a whole-object query therefore cannot prove outside. Keep the
        # required samples unknown without repeatedly querying every fragment.
        if len(remaining) and not query.has_unsigned_only_components:
            # All components must be reliable. Being outside just one of two
            # components is not a certificate of leaving the whole object.
            signed = query.query(local[remaining])
            outside[remaining] = (~signed['uncertain'] & np.isfinite(signed['clearance'])
                                  & (signed['clearance'] >= 0.))
        proven.update((oid, gid, int(sample)) for sample in ids[outside])
    return proven
