"""Conservative optimization rows for formerly penetrating, now unknown samples.

These are NOT signed-distance certificates. A lost side vote must neither pin
the entire trajectory nor erase a penetration residual. Until the original
query determines the side again, minimize distance to the actual triangles
from the presumed interior and keep the observation explicitly unknown.
"""

from collections import defaultdict

import mujoco
import numpy as np

from ..moving_scene import interpolate_object


def material_rows(moving, pose, frame, missing, columns):
    """One worst unknown material point per original object/robot geom.

    The negative unsigned distance is a conservative optimization surrogate:
    no unknown point receives positive clearance. Its gradient leads to the
    actual nearest triangle, not an unbounded tangent-plane extension.
    Source observations and all original query results remain unchanged.
    """
    if not missing:
        return []
    moving.data.qpos[:] = pose
    mujoco.mj_forward(moving.model, moving.data)
    shells = {s['geom_id']: s for s in moving.shells}
    groups = defaultdict(list)
    for oid, gid, sample in missing:
        groups[int(oid), int(gid)].append(int(sample))
    result = []
    for (oid, gid), samples in sorted(groups.items()):
        obj = moving.objects[oid]
        samples = np.array(sorted(samples), dtype=int)
        translation, rotation, scale = interpolate_object(obj['surface'], frame)
        robot_rotation = moving.data.geom_xmat[gid].reshape(3, 3)
        points = shells[gid]['offsets'][samples] @ robot_rotation.T + moving.data.geom_xpos[gid]
        local = (points - translation) @ rotation / scale
        surface = obj['surface']
        squared, triangle, nearest = surface.tree.squared_distance(
            surface.vertices, surface.faces, local)
        worst = int(np.argmax(squared))
        distance = float(np.sqrt(max(0., squared[worst])))
        if distance <= 1e-10:
            continue
        direction = rotation @ ((nearest[worst] - local[worst]) / distance)
        row = moving._row(points[worst], direction, -distance * scale, gid, oid,
                          int(samples[worst]), 'unresolved_material', columns,
                          component_id=int(obj['face_components'][triangle[worst]]))
        row.update(uncertain=True, surrogate='negative_unsigned_distance',
                   missing_material_samples=len(samples))
        result.append(row)
    return result
