"""Bounded collision witnesses for a single rigid geometry and scene state.

This is a selection heuristic, not a certificate for omitted samples. Callers
must keep their full-geometry nonlinear acceptance and all existing pose,
contact, root and temporal limits. No dataset or body-region rules are used.
"""

from operator import index

import numpy as np

_TETRAHEDRON = np.asarray([(1., 1., 1.), (1., -1., -1.),
                           (-1., 1., -1.), (-1., -1., 1.)])
# Four moment probes first cover the four corners of axis-aligned planar
# rectangles. Six force probes cover signed normal axes; four opposite moment
# probes complete the cube directions. With the deepest point: at most 15 rows.
_PROBES = np.vstack([
    np.column_stack([np.zeros((4, 3)), _TETRAHEDRON]),
    np.column_stack([np.vstack([np.eye(3), -np.eye(3)]), np.zeros((6, 3))]),
    np.column_stack([np.zeros((4, 3)), -_TETRAHEDRON]),
])


def _support_indices(features, limit):
    """Finite directional support probes; ties follow canonical input order."""
    selected = {0}
    # A fixed prefix, not candidate-dependent pruning, keeps summaries mergeable
    # even at small caps. Smaller caps necessarily cover fewer directions.
    for probe in _PROBES[:limit - 1]:
        score = np.einsum("ij,j->i", features, probe)
        selected.add(int(np.argmax(score)))
    return np.asarray(sorted(selected), dtype=int)


def select_collision_manifold(
    points,
    clearance,
    directions,
    *,
    valid=None,
    max_points=16,
    activation_distance=.008,
    origin=(0., 0., 0.),
):
    """Return original sample indices for a bounded collision manifold.

    ``clearance`` is signed (negative means penetration); ``directions`` point
    toward increasing clearance. Pass ``valid=~uncertain`` to exclude unreliable
    observations. Nonfinite data and zero normals are independently excluded.
    An empty result means no usable active witness, never collision-free proof.

    Samples with clearance below ``activation_distance`` are candidates. This
    fixed world-distance threshold can include near-contact samples that a
    trust step might push into penetration; set it to zero for penetrations
    only. It is not a joint-angle bound or a guarantee about the proposed step.
    The threshold must stay fixed across batches. A moving near-deepest cutoff
    would invalidate incremental summaries when a later batch is deeper.

    If any sample is eligible, its deepest reliable witness is always first.
    Force normals and moments ``(point - origin) cross normal`` distinguish
    equal-normal points with different rotational lever arms. A fixed set of
    signed force-axis and moment-diagonal support probes retains physical
    extremes without sample-density weighting. ``origin`` can be the current
    geom center, but must stay fixed throughout the same geometry query.

    Repetition and reordering do not change the selected physical witnesses;
    an exact duplicate may return a different original index. With fixed
    parameters, merging already selected witnesses gives the same physical
    witnesses as selecting their full union, including arbitrary batch splits.
    The default returns at most 15 rows; a smaller cap uses fewer fixed probes.
    The finite directions can omit an important witness, especially for many
    patches, curved surfaces or a small cap. Full-geometry acceptance remains
    mandatory; this does not prove the omitted samples redundant.

    This function supplies no per-point weights. Use a shared maximum-depth
    residual/epigraph per geometry pair, so adding witnesses cannot multiply
    that pair's objective weight.
    """
    try:
        limit = index(max_points)
    except TypeError as error:
        raise ValueError("max_points must be a positive integer") from error
    if limit < 1:
        raise ValueError("max_points must be a positive integer")
    activation = float(activation_distance)
    if not np.isfinite(activation) or activation < 0:
        raise ValueError("activation_distance must be finite and nonnegative")
    origin = np.asarray(origin, dtype=float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("origin must be a finite three-vector")
    points = np.asarray(points, dtype=float)
    clearance = np.asarray(clearance, dtype=float)
    directions = np.asarray(directions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or directions.shape != points.shape:
        raise ValueError("points and directions must have shape (n, 3)")
    if clearance.shape != (len(points),):
        raise ValueError("clearance must have shape (n,)")
    reliable = np.ones(len(points), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if reliable.shape != clearance.shape:
        raise ValueError("valid must have shape (n,)")
    # Scale normals before normalization so finite, very large values cannot
    # overflow their squared norm and silently discard usable observations.
    normal_scale = np.max(np.abs(directions), axis=1)
    reliable = (reliable & np.isfinite(clearance) & np.isfinite(points).all(axis=1)
                & np.isfinite(directions).all(axis=1) & (normal_scale > 0))
    indices = np.flatnonzero(reliable)
    if not len(indices):
        return np.empty(0, dtype=int)
    indices = indices[clearance[indices] < activation]
    if not len(indices):
        return np.empty(0, dtype=int)
    p, phi = points[indices], clearance[indices]
    normals = directions[indices] / normal_scale[indices, None]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    # Canonical depth/position/normal ordering makes argmax tie handling stable
    # under point-cloud permutation and exact sample duplication.
    order = np.lexsort((normals[:, 2], normals[:, 1], normals[:, 0],
                        p[:, 2], p[:, 1], p[:, 0], phi))
    indices, p, normals = indices[order], p[order], normals[order]
    if limit == 1:
        return indices[:1]
    with np.errstate(over="ignore", invalid="ignore"):
        moments = np.cross(p - origin, normals)
    if not np.isfinite(moments).all():
        raise ValueError("points relative to origin produce nonfinite moments")
    features = np.concatenate([normals, moments], axis=1)
    return indices[_support_indices(features, limit)]
