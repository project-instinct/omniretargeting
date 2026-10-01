"""Support patches: connected contact regions a frame's union certifies.

A support patch is one physically connected contact region on one surface:
buttocks and thighs resting on a seat, one foot on the floor. Candidates of
the same surface whose environment contact points lie within the
observation's own resolution form one patch; that resolution is the contact
reconstruction bound plus the source skin's measured sample spacing, so the
link distance is derived from existing observation constants, never from a
body-part rule. Necessity itself is decided by the same dual-cone
certification as the union (``support_sensitivity``); this module only
groups and measures.
"""

import numpy as np


def _min_gap(first, second):
    """Minimum distance between two finite point sets."""
    first = np.asarray(first, float).reshape(-1, 3)
    second = np.asarray(second, float).reshape(-1, 3)
    if not len(first) or not len(second):
        return np.inf
    if len(first) * len(second) <= 4096:
        return float(np.sqrt(((first[:, None] - second[None]) ** 2).sum(-1)).min())
    from scipy.spatial import cKDTree

    distance, _ = cKDTree(first).query(second, k=1)
    return float(np.asarray(distance).min())


def support_patches(contacts, surfaces, link_distance):
    """Sorted name lists: connected candidate groups within one surface.

    ``contacts`` maps candidate name to a contact (any scenario; the
    geometry observations are shared). ``surfaces`` maps name to surface id.
    Two candidates of one surface whose environment contact points come
    within ``link_distance`` join one patch, transitively. Candidates of
    different surfaces never join, however close: they are different
    physical contact regions.
    """
    names = sorted(contacts)
    parent = {name: name for name in names}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    by_surface = {}
    for name in names:
        by_surface.setdefault(int(surfaces[name]), []).append(name)
    for surface, members in sorted(by_surface.items()):
        for index, first in enumerate(members):
            for second in members[index + 1:]:
                if find(first) == find(second):
                    continue
                gap = _min_gap(
                    contacts[first].surface_points, contacts[second].surface_points)
                if gap <= link_distance:
                    parent[find(first)] = find(second)
    patches = {}
    for name in names:
        patches.setdefault(find(name), []).append(name)
    return sorted(sorted(members) for members in patches.values())


def source_skin_spacing(points, valid=None):
    """Median nearest-neighbour distance between the source skin samples.

    The source's own sampling resolution, measured per frame and summarized
    as a median over frames. Two contact observations whose environment
    points are closer than this spacing plus the contact reconstruction
    bound cannot be certified as separate contact regions by this
    observation. It is a measurement of immutable source data, not a
    configured threshold.
    """
    points = np.asarray(points, float)
    if points.ndim != 3 or points.shape[-1] != 3:
        return 0.0
    from scipy.spatial import cKDTree

    per_frame = []
    for frame in range(len(points)):
        samples = points[frame]
        mask = (
            np.isfinite(samples).all(-1) if valid is None
            else np.asarray(valid, bool)[frame] & np.isfinite(samples).all(-1)
        )
        if int(mask.sum()) < 2:
            continue
        selected = samples[mask]
        distance, _ = cKDTree(selected).query(selected, k=2)
        per_frame.append(float(np.median(distance[:, 1])))
    return float(np.median(per_frame)) if per_frame else 0.0
