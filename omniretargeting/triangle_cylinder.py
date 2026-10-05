"""Finite triangle / solid cylinder queries, including open scans.

Same contract as ``triangle_capsule``: the actual triangles, independent of
their winding, without inferring which side of an open scan is solid.

For a cylinder C and a triangle T the configuration obstacle D = C - T is
convex: the oblique prism (axis segment - T) plus the radius disk. Its signed
clearance is ``-min_{|u|=1} h_D(u)`` for separated and intersecting pairs
alike (h_D is the support function). Every direction gives a *lower* bound on
the clearance, so a missed direction can only overstate a depth, never hide an
overlap. The minimum lies in one of D's face families, all enumerated here:
prism face normals, the two cap normals, prism-vertex/disk closest points, and
the great circle normal to each prism edge (triangle edges and the axis, each
plus the disk rim), sampled whole and refined. Separated pairs are certified by an explicit witness
pair; a remaining gap or a tied intersecting exit is reported as uncertain.
No scene thickening, convexification, or surface sampling is used.
"""

import numpy as np
import trimesh

ARC_SAMPLES = 720
ARC_REFINED = 3
GOLDEN_STEPS = 64
# Equal minimum exits closer than this (metres) are a tie of distinct exits.
TIE_M = 1e-9
# A separated witness pair must reproduce the lower bound within this gap.
CERTIFIED_GAP_M = 1e-7
# Support-face membership: within this distance of the supporting plane.
FACE_M = 1e-9
_GOLDEN = (np.sqrt(5.) - 1.) / 2.


def _unit(values):
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, norm, out=np.zeros_like(values), where=norm > 1e-14), norm[..., 0]


def _support(center, axis, radius, half, triangles, u):
    """h_D(u) for unit directions ``u`` (N, K, 3) and triangles (N, 3, 3)."""
    along = u @ axis
    cylinder = u @ center + half * np.abs(along) + radius * np.sqrt(np.maximum(1. - along**2, 0.))
    return cylinder + np.max(-np.einsum("nkj,nij->nki", u, triangles), axis=2)


def _closest_on_face(face, point):
    """Closest point of a support face given by one, two or three vertices."""
    if len(face) == 1:
        return face[0]
    if len(face) == 2:
        edge = face[1] - face[0]
        t = np.clip((point - face[0]) @ edge / max(float(edge @ edge), 1e-300), 0., 1.)
        return face[0] + t * edge
    return trimesh.triangles.closest_point(face[None], point[None])[0]


def witness_pair(center, axis, radius, half, triangle, u, h):
    """Cylinder/triangle witnesses of the exit ``u`` with support value ``h``.

    The cylinder's support face in ``u`` (point, side segment or cap disk) and
    the triangle's in ``-u`` (vertex, edge or face) lie in parallel planes ``h``
    apart along ``u``; a face is everything within ``FACE_M`` of its plane.
    Returns the witnesses and their excess distance over a separated
    clearance (an upper minus the lower bound), or their in-plane offset when
    the pair intersects.
    """
    from .triangle_capsule import segment_triangle_witnesses
    offset = h * u
    values = -(triangle @ u)
    face = triangle[values >= values.max() - FACE_M] + offset
    along = float(u @ axis)
    radial = u - along * axis
    across = float(np.linalg.norm(radial))
    if radius * across <= FACE_M:
        cap = center + half * (np.sign(along) or 1.) * axis
        target = _closest_on_face(face, cap)
        reach = target - cap
        reach -= (reach @ axis) * axis
        distance = float(np.linalg.norm(reach))
        point = cap + reach * min(1., radius / distance) if distance > 0. else cap
    elif half * abs(along) <= FACE_M:
        side = center + radius * radial / across
        padded = np.vstack([face, np.repeat(face[-1:], 3 - len(face), axis=0)])
        points, targets = segment_triangle_witnesses(side - half * axis, side + half * axis, padded)
        point, target = points[0], targets[0]
    else:
        point = center + half * np.sign(along) * axis + radius * radial / across
        target = _closest_on_face(face, point)
    surface = target - offset
    if h < 0.:
        return point, surface, float(np.linalg.norm(point - surface) + h)
    return point, surface, float(np.linalg.norm(point - target))


def _circle_basis(edges, axis):
    normal, length = _unit(edges)
    reference = np.where((np.abs(normal @ axis) < .9)[..., None], axis, np.array([0., 0., 1.]))
    reference = np.where((np.abs(np.einsum("...j,...j->...", normal, reference)) < .9)[..., None],
                         reference, np.array([1., 0., 0.]))
    first, _ = _unit(np.cross(normal, reference))
    return first, np.cross(normal, first), length > 1e-14


def _arc_candidates(center, axis, radius, half, triangles):
    """Refined minima of h_D on each prism-edge normal great circle.

    Prism edges are the triangle edges and three copies of the axis.
    """
    count = len(triangles)
    edges = np.concatenate([np.roll(triangles, -1, axis=1) - triangles,
                            np.broadcast_to(axis, (count, 1, 3))], axis=1)
    first, second, valid = _circle_basis(edges, axis)
    theta = np.linspace(0., 2. * np.pi, ARC_SAMPLES, endpoint=False)
    cos, sin = np.cos(theta), np.sin(theta)

    def directions(angle):
        # angle (N, 4, R) -> unit directions (N, 4, R, 3)
        return (np.cos(angle)[..., None] * first[:, :, None]
                + np.sin(angle)[..., None] * second[:, :, None])

    sampled = (cos[None, None, :, None] * first[:, :, None]
               + sin[None, None, :, None] * second[:, :, None])
    values = _support(center, axis, radius, half, triangles,
                      sampled.reshape(count, -1, 3)).reshape(count, 4, ARC_SAMPLES)
    minimum = (values <= np.roll(values, 1, axis=2)) & (values <= np.roll(values, -1, axis=2))
    ranked = np.argsort(np.where(minimum, values, np.inf), axis=2)[:, :, :ARC_REFINED]
    step = 2. * np.pi / ARC_SAMPLES
    low, high = theta[ranked] - step, theta[ranked] + step

    def evaluate(angle):
        u = directions(angle)
        return _support(center, axis, radius, half, triangles,
                        u.reshape(count, -1, 3)).reshape(angle.shape)

    # Golden-section search inside each bracketing sample interval.
    a = high - _GOLDEN * (high - low)
    b = low + _GOLDEN * (high - low)
    fa, fb = evaluate(a), evaluate(b)
    for _ in range(GOLDEN_STEPS):
        left = fa <= fb
        low, high = np.where(left, low, a), np.where(left, b, high)
        fresh = np.where(left, high - _GOLDEN * (high - low), low + _GOLDEN * (high - low))
        value = evaluate(fresh)
        a, b = np.where(left, fresh, b), np.where(left, a, fresh)
        fa, fb = np.where(left, value, fb), np.where(left, fa, value)
    refined = directions(.5 * (low + high))
    keep = np.broadcast_to(valid[:, :, None], ranked.shape)
    return np.where(keep[..., None], refined, np.nan).reshape(count, -1, 3)


def _face_candidates(center, axis, radius, half, triangles):
    edges = np.roll(triangles, -1, axis=1) - triangles
    face, _ = _unit(np.cross(edges[:, 0], edges[:, 1]))
    sides, _ = _unit(np.cross(axis, edges))
    caps = np.broadcast_to(axis, (len(triangles), 1, 3))
    # Closest point of each translated disk (prism vertex plus radius disk).
    vertices = np.concatenate([center + half * axis - triangles, center - half * axis - triangles], axis=1)
    radial = vertices - (vertices @ axis)[..., None] * axis
    unit, length = _unit(-radial)
    closest = vertices + np.minimum(length, radius)[..., None] * unit
    disk, _ = _unit(-closest)
    fixed = np.concatenate([face[:, None], -face[:, None], sides, -sides, caps, -caps, disk], axis=1)
    norm = np.linalg.norm(fixed, axis=2)
    return np.where((norm > .5)[..., None], fixed, np.nan)


def cylinder_triangle_rows(center, axis, radius, half, triangles, *, with_witnesses=True):
    """Signed clearance, outward direction, witnesses and certificate per triangle.

    ``direction`` is the gradient of the clearance with respect to the
    cylinder's translation. ``ambiguous`` marks tied intersecting exits;
    ``gap`` is the in-plane distance of the support-face witnesses.
    """
    triangles = np.asarray(triangles, dtype=float).reshape(-1, 3, 3)
    center, axis = np.asarray(center, float), np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    radius, half = float(radius), float(half)
    count = len(triangles)
    if not count:
        empty = np.empty((0, 3))
        return dict(clearance=np.empty(0), direction=empty, ambiguous=np.empty(0, bool),
                    exit=empty, support=np.empty(0), point=empty, surface=empty, gap=np.empty(0))
    candidates = np.concatenate([
        _face_candidates(center, axis, radius, half, triangles),
        _arc_candidates(center, axis, radius, half, triangles),
    ], axis=1)
    finite = np.isfinite(candidates).all(axis=2)
    safe = np.where(finite[..., None], candidates, np.array([0., 0., 1.]))
    values = np.where(finite, _support(center, axis, radius, half, triangles, safe), np.inf)
    best = np.argmin(values, axis=1)
    rows = np.arange(count)
    u = safe[rows, best]
    h = values[rows, best]
    clearance = -h
    direction = -u
    # A smooth curved minimum has close candidates; only distinct exits tie.
    tied = (values <= h[:, None] + TIE_M) & (np.einsum("nkj,nj->nk", safe, u) < 1. - 1e-6)
    ambiguous = (h > 0.) & tied.any(axis=1)
    result = dict(clearance=clearance, direction=direction, ambiguous=ambiguous,
                  exit=u, support=h)
    if with_witnesses:
        pairs = [witness_pair(center, axis, radius, half, triangles[i], u[i], h[i])
                 for i in range(count)]
        result.update(point=np.array([p[0] for p in pairs]),
                      surface=np.array([p[1] for p in pairs]),
                      gap=np.array([p[2] for p in pairs]))
    return result
