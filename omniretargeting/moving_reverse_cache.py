"""Exact-transform reuse of original moving-mesh reverse witnesses."""

import numpy as np


def reverse_witnesses(
    obj, shell, center, robot_rotation, translation, rotation, scale, band, max_rows
):
    """Check every original sample, retaining the same deepest facet witnesses."""
    indices = np.asarray(
        obj["tree"].query_ball_point(
            (center - translation) @ rotation / scale, (shell["radius"] + band) / scale
        ),
        dtype=int,
    )
    empty = {"depth": 0.0, "count": 0, "available": 0, "rows": []}
    if not len(indices):
        return empty
    points = obj["samples"][indices] @ rotation.T * scale + translation
    local = (points - center) @ robot_rotation
    bounds = shell["bounds"]
    keep = np.all((local >= bounds[0] - band) & (local <= bounds[1] + band), axis=1)
    indices, points, local = indices[keep], points[keep], local[keep]
    if not len(indices):
        return empty
    values = local @ shell["planes"][:, :3].T + shell["planes"][:, 3]
    facets = np.argmax(values, axis=1)
    phi = values[np.arange(len(values)), facets]
    selected = [
        int(np.flatnonzero(facets == face)[np.argmin(phi[facets == face])])
        for face in np.unique(facets[phi < band])
    ]
    selected.sort(key=lambda k: phi[k])
    return {
        "depth": max(0.0, -float(phi.min())),
        "count": int(np.count_nonzero(phi < -1e-9)),
        "available": len(selected),
        "rows": [
            (
                points[k].copy(),
                -(robot_rotation @ shell["planes"][facets[k], :3]),
                float(phi[k]),
                int(indices[k]),
                int(facets[k]),
            )
            for k in selected[:max_rows]
        ],
    }


def cached_reverse(
    owner,
    oid,
    obj,
    shell,
    time,
    center,
    robot_rotation,
    translation,
    rotation,
    scale,
    band,
):
    """Cache geometry only; point Jacobians always use the caller's current FK."""
    key = (oid, float(time), shell["geom_id"],
        center.tobytes(),
        robot_rotation.tobytes(),
        translation.tobytes(),
        rotation.tobytes(),
        float(scale),
        float(band),
        int(owner.cfg["max_rows_per_geom"]),
    )
    previous = owner.reverse_cache.get(key)
    if previous is not None:
        owner.reverse_cache.move_to_end(key)
        return previous
    result = reverse_witnesses(
        obj,
        shell,
        center,
        robot_rotation,
        translation,
        rotation,
        scale,
        band,
        int(owner.cfg["max_rows_per_geom"]),
    )
    owner.reverse_cache[key] = result
    owner.reverse_cache.move_to_end(key)
    while len(owner.reverse_cache) > 8192:
        owner.reverse_cache.popitem(last=False)
    return result
