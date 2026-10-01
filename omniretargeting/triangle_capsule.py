"""Finite triangle / capsule surface queries, including open scans.

This tests intersection with the actual triangles, independent of their winding.
It does not infer which side of an open scan is solid. Clearance is the distance
from a capsule axis to the triangle minus its radius. Axis intersections use the
exact pairwise configuration obstacle for their minimum translation. Ambiguous
equal exits retain the intersection without inventing a gradient.
No scene thickening, convexification, surface sampling, or asset copying is used.
"""

from itertools import pairwise

import numpy as np
import trimesh


def _dot_axis(values, axis):
    return values @ axis if axis.ndim == 1 else np.einsum("ij,ij->i", values, axis)


def intersecting_capsule_exit(start, end, radius, triangle):
    """Minimum translation when the capsule axis intersects a triangle.

    Triangle minus segment is an oblique triangular prism. Its Minkowski sum
    with the radius ball is the *exact* capsule/triangle configuration obstacle.
    An origin inside the prism exits through its nearest supporting face plus
    radius. Only this single-pair difference is convex; the scene is untouched.
    Symmetric minimum exits retain their ambiguity instead of choosing a side.
    """
    triangle = np.asarray(triangle, dtype=float)
    start, end = np.asarray(start, float), np.asarray(end, float)
    edges = np.roll(triangle, -1, axis=0) - triangle
    face = np.cross(edges[0], edges[1])
    normals = np.vstack([face, np.cross(end - start, edges)])
    norms = np.linalg.norm(normals, axis=1)
    normals = normals[norms > 1e-14] / norms[norms > 1e-14, None]
    if not len(normals):
        return float(radius), None
    normals = np.vstack([normals, -normals])
    depths = float(radius) + np.max((triangle - start) @ normals.T, axis=0) - np.minimum(0, (end - start) @ normals.T)
    minimum = float(np.min(depths))
    tied = normals[depths <= minimum + 1e-9]
    direction = tied[0]
    if np.any(tied @ direction < 1 - 1e-8):
        return minimum, None
    return minimum, direction


def segment_triangle_witnesses(start, end, triangles, *, groups=None):
    """Exact closest witnesses for segments and finite triangles.

    Enumerate endpoint/face, segment/edge, and segment/face intersections.
    Degenerate triangles are covered by their edges and vertices.
    ``start/end`` may be (3,) for a shared axis or (N, 3) for one per face.
    Contiguous shared-axis ``groups`` retain the scalar-axis dot-product order
    so rounding does not select a different face at a coplanar witness tie.
    """
    triangles = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
    start, end = np.asarray(start, float), np.asarray(end, float)
    count = len(triangles)
    if not count:
        return np.empty((0, 3)), np.empty((0, 3))
    axis = end - start

    def dot_axis(values):
        if groups is None:
            return _dot_axis(values, axis)
        return np.concatenate([
            values[a:b] @ axis[a] for a, b in pairwise(groups)
        ])

    length2 = (
        _dot_axis(axis, axis) if groups is None else
        np.repeat([float(axis[a] @ axis[a]) for a in groups[:-1]], np.diff(groups))
    )
    candidates = []
    normals = np.cross(triangles[:, 1] - triangles[:, 0],
                       triangles[:, 2] - triangles[:, 0])
    regular = np.einsum("ij,ij->i", normals, normals) > 1e-24
    # Avoid undefined barycentric divisions on degenerate faces.
    for endpoint in (start, end):
        surface = triangles[:, 0].copy()
        if np.any(regular):
            surface[regular] = trimesh.triangles.closest_point(
                triangles[regular], np.broadcast_to(endpoint, (count, 3))[regular]
            )
        candidates.append((np.broadcast_to(endpoint, (count, 3)), surface))
    for i in range(3):
        a, b = triangles[:, i], triangles[:, (i + 1) % 3]
        edge = b - a
        edge2 = np.einsum("ij,ij->i", edge, edge)
        safe_edge2 = np.maximum(edge2, 1e-30)
        for endpoint in (start, end):
            t = np.clip(np.einsum("ij,ij->i", endpoint - a, edge) / safe_edge2, 0, 1)
            candidates.append((np.broadcast_to(endpoint, (count, 3)), a + t[:, None] * edge))
        for vertex in (a, b):
            s = np.clip(dot_axis(vertex - start) / np.maximum(length2, 1e-30), 0, 1)
            candidates.append((start + s[:, None] * axis, vertex))
        cross = dot_axis(edge)
        w = start - a
        d, e = dot_axis(w), np.einsum("ij,ij->i", w, edge)
        det = length2 * edge2 - cross**2
        usable = det > 1e-14 * np.maximum(length2 * edge2, 1e-30)
        denominator = np.where(usable, det, 1)
        s = (cross * e - edge2 * d) / denominator
        t = (length2 * e - cross * d) / denominator
        valid = usable & (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
        p, surface = start + s[:, None] * axis, a + t[:, None] * edge
        p[~valid] = np.inf
        candidates.append((p, surface))
    denominator = dot_axis(normals)
    crossing = regular & (np.abs(denominator) > 1e-15)
    s = np.einsum("ij,ij->i", triangles[:, 0] - start, normals) / np.where(crossing, denominator, 1)
    point = start + s[:, None] * axis
    # A finite-face test prevents extending a scan triangle into an infinite wall.
    inside = crossing & (s >= 0) & (s <= 1)
    for i in range(3):
        edge = triangles[:, (i + 1) % 3] - triangles[:, i]
        side = np.einsum("ij,ij->i", np.cross(edge, point - triangles[:, i]), normals)
        inside &= side >= -1e-12 * np.einsum("ij,ij->i", normals, normals)
    p = point.copy()
    p[~inside] = np.inf
    candidates.append((p, point))
    axis_points = np.stack([pair[0] for pair in candidates])
    surface_points = np.stack([pair[1] for pair in candidates])
    distance2 = np.sum((axis_points - surface_points)**2, axis=2)
    chosen = np.argmin(distance2, axis=0)
    return axis_points[chosen, np.arange(count)], surface_points[chosen, np.arange(count)]
