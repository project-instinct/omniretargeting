"""Source-conditioned geometric contact regions on registered scene meshes.

The source human decides *which* connected scene patch is relevant.  The
resulting region lets a differently proportioned robot choose another point
inside that same patch without changing either the scene or source motion.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial import ConvexHull, QhullError
import trimesh


_TOP_NORMAL_Z = 0.82
_VERTICAL_NORMAL_Z = 0.28


def _unit(vector: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(3).copy()
    length = float(np.linalg.norm(value))
    if not np.isfinite(length) or length < 1e-10:
        raise ValueError(f"{name} contains a zero or non-finite vector.")
    return value / length


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=float).copy()
    result.setflags(write=False)
    return result


def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    return transform[:3, :3] @ point + transform[:3, 3]


def _polygon_area(polygon: np.ndarray) -> float:
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _point_in_convex_polygon(point: np.ndarray, polygon: np.ndarray, tolerance: float) -> bool:
    edges = np.roll(polygon, -1, axis=0) - polygon
    relative = point - polygon
    cross = edges[:, 0] * relative[:, 1] - edges[:, 1] * relative[:, 0]
    return bool(np.all(cross >= -tolerance) or np.all(cross <= tolerance))


def _closest_polygon_point(point: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    if _point_in_convex_polygon(point, polygon, 1e-12):
        return point.copy()
    starts = polygon
    segments = np.roll(polygon, -1, axis=0) - starts
    lengths_squared = np.sum(segments * segments, axis=1)
    alphas = np.divide(
        np.sum((point - starts) * segments, axis=1),
        lengths_squared,
        out=np.zeros(len(segments), dtype=float),
        where=lengths_squared > 1e-16,
    )
    alphas = np.clip(alphas, 0.0, 1.0)
    candidates = starts + alphas[:, None] * segments
    return candidates[int(np.argmin(np.sum((candidates - point) ** 2, axis=1)))]


def _clip_axis(
    polygon: np.ndarray, axis: int, bound: float, keep_greater: bool
) -> np.ndarray:
    if len(polygon) == 0:
        return polygon

    def inside(point: np.ndarray) -> bool:
        return bool(point[axis] >= bound - 1e-12) if keep_greater else bool(
            point[axis] <= bound + 1e-12
        )

    output: list[np.ndarray] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside != previous_inside:
            denominator = current[axis] - previous[axis]
            alpha = 0.0 if abs(denominator) < 1e-14 else (bound - previous[axis]) / denominator
            output.append(previous + np.clip(alpha, 0.0, 1.0) * (current - previous))
        if current_inside:
            output.append(current)
        previous = current
        previous_inside = current_inside
    return np.asarray(output, dtype=float).reshape(-1, 2)


def _clip_to_square(polygon: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    clipped = polygon.copy()
    for axis in (0, 1):
        clipped = _clip_axis(clipped, axis, center[axis] - radius, True)
        clipped = _clip_axis(clipped, axis, center[axis] + radius, False)
    return clipped


def _face_components(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return deterministic connected-component labels for every face."""
    count = len(mesh.faces)
    parent = np.arange(count, dtype=int)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for first, second in np.asarray(mesh.face_adjacency, dtype=int):
        root_first = find(int(first))
        root_second = find(int(second))
        if root_first != root_second:
            parent[root_second] = root_first
    roots = np.asarray([find(index) for index in range(count)], dtype=int)
    unique = sorted(set(roots.tolist()), key=lambda root: int(np.flatnonzero(roots == root)[0]))
    labels_by_root = {root: label for label, root in enumerate(unique)}
    return np.asarray([labels_by_root[root] for root in roots], dtype=int)


def _closest_scene_points(
    mesh: trimesh.Trimesh, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        closest, distances, triangle_ids = trimesh.proximity.closest_point(mesh, points)
    except (ImportError, ModuleNotFoundError):
        closest, distances, triangle_ids = trimesh.proximity.closest_point_naive(mesh, points)
    return (
        np.asarray(closest, dtype=float),
        np.asarray(distances, dtype=float),
        np.asarray(triangle_ids, dtype=int),
    )


def _distances_to_faces(triangles: np.ndarray, points: np.ndarray) -> np.ndarray:
    distances = np.full(len(triangles), np.inf, dtype=float)
    for point in points:
        repeated = np.repeat(point[None, :], len(triangles), axis=0)
        closest = trimesh.triangles.closest_point(triangles, repeated)
        distances = np.minimum(distances, np.linalg.norm(closest - point, axis=1))
    return distances


def _connected_neighborhood(
    mesh: trimesh.Trimesh,
    component_faces: np.ndarray,
    seed_faces: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    radius: float,
    ledge_evidence: bool,
) -> np.ndarray:
    component_set = set(int(face) for face in component_faces)
    triangles = np.asarray(mesh.triangles[component_faces], dtype=float)
    distances = _distances_to_faces(triangles, points)
    spatial = {
        int(face)
        for face, distance in zip(component_faces, distances)
        if distance <= radius + 1e-10
    }

    face_normals = np.asarray(mesh.face_normals, dtype=float)
    if ledge_evidence:
        similarity = np.max(np.abs(face_normals @ normals.T), axis=1)
        normal_ok = similarity >= 0.45
    else:
        mean_normal = _unit(np.mean(normals, axis=0), name="inferred_normals")
        normal_ok = np.abs(face_normals @ mean_normal) >= np.cos(np.deg2rad(38.0))
    eligible = spatial.intersection(np.flatnonzero(normal_ok).tolist())
    eligible.update(int(face) for face in seed_faces)

    adjacency: dict[int, list[int]] = {int(face): [] for face in component_faces}
    for first, second in np.asarray(mesh.face_adjacency, dtype=int):
        first = int(first)
        second = int(second)
        if first in component_set and second in component_set:
            adjacency[first].append(second)
            adjacency[second].append(first)

    queue = [int(face) for face in seed_faces if int(face) in component_set]
    selected = set(queue)
    while queue:
        face = queue.pop(0)
        for neighbor in adjacency[face]:
            if neighbor in eligible and neighbor not in selected:
                selected.add(neighbor)
                queue.append(neighbor)
    return np.asarray(sorted(selected), dtype=int)


def _classify_patch(normals: np.ndarray) -> tuple[str, bool]:
    absolute_z = np.abs(normals[:, 2])
    mean = _unit(np.mean(normals, axis=0), name="inferred_normals")
    mean_z = abs(float(mean[2]))
    has_top = bool(np.any(absolute_z >= 0.65))
    has_vertical = bool(np.any(absolute_z <= 0.45))
    ledge = (has_top and has_vertical) or (_VERTICAL_NORMAL_Z < mean_z < _TOP_NORMAL_Z)
    if ledge:
        return "ledge", True
    if mean_z >= _TOP_NORMAL_Z:
        return "top", False
    if mean_z <= _VERTICAL_NORMAL_Z:
        return "vertical", False
    return "ledge", True


def _plane_frame(
    points: np.ndarray,
    source_normals: np.ndarray,
    patch_face_normals: np.ndarray,
    patch_face_areas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source_mean = _unit(np.mean(source_normals, axis=0), name="inferred_normals")
    oriented = patch_face_normals.copy()
    signs = np.sign(oriented @ source_mean)
    signs[signs == 0.0] = 1.0
    oriented *= signs[:, None]
    weighted_patch = np.sum(oriented * patch_face_areas[:, None], axis=0)
    normal = _unit(weighted_patch + source_mean * np.sum(patch_face_areas), name="patch normal")
    if np.dot(normal, source_mean) < 0.0:
        normal = -normal
    origin = np.mean(points, axis=0)

    reference = np.array([1.0, 0.0, 0.0])
    forward = reference - normal * float(np.dot(reference, normal))
    if np.linalg.norm(forward) < 0.2:
        reference = np.array([0.0, 1.0, 0.0])
        forward = reference - normal * float(np.dot(reference, normal))
    forward = _unit(forward, name="plane forward")
    lateral = _unit(np.cross(normal, forward), name="plane lateral")
    forward = _unit(np.cross(lateral, normal), name="plane forward")
    return origin, normal, np.column_stack([forward, lateral])


def _convex_boundary(points_2d: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    rounded = np.unique(np.round(points_2d, decimals=12), axis=0)
    try:
        if len(rounded) < 3:
            raise QhullError("fewer than three polygon points")
        hull = ConvexHull(rounded)
        polygon = rounded[hull.vertices]
    except QhullError:
        minimum = np.min(rounded, axis=0) if len(rounded) else center - radius
        maximum = np.max(rounded, axis=0) if len(rounded) else center + radius
        minimum = np.minimum(minimum, center - 0.02)
        maximum = np.maximum(maximum, center + 0.02)
        polygon = np.asarray(
            [minimum, [maximum[0], minimum[1]], maximum, [minimum[0], maximum[1]]]
        )
    clipped = _clip_to_square(polygon, center, radius)
    if len(clipped) < 3 or _polygon_area(clipped) < 1e-10:
        clipped = np.asarray(
            [
                center + [-radius, -radius],
                center + [radius, -radius],
                center + [radius, radius],
                center + [-radius, radius],
            ]
        )
    return clipped


@dataclass(frozen=True)
class RegionQuery:
    """Result of testing and projecting one world-space candidate point."""

    inside: bool
    nearest_point: np.ndarray
    plane_distance: float
    boundary_distance: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "nearest_point", _readonly(self.nearest_point))


@dataclass(frozen=True)
class ContactRegion:
    """A convex plane-local region tied to one connected scene object."""

    region_id: str
    region_type: str
    object_id: str
    world_from_object: np.ndarray
    plane_origin_local: np.ndarray
    normal_local: np.ndarray
    tangent_basis_local: np.ndarray
    polygon_local_2d: np.ndarray
    boundary_local: np.ndarray
    face_ids: tuple[int, ...]
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        transform = np.asarray(self.world_from_object, dtype=float).copy()
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("world_from_object must be a finite 4x4 transform.")
        if abs(float(np.linalg.det(transform[:3, :3]))) < 1e-10:
            raise ValueError("world_from_object must have an invertible rotation block.")
        transform.setflags(write=False)
        object.__setattr__(self, "world_from_object", transform)
        object.__setattr__(self, "plane_origin_local", _readonly(self.plane_origin_local))
        object.__setattr__(self, "normal_local", _readonly(self.normal_local))
        tangent = _readonly(self.tangent_basis_local)
        if tangent.shape != (3, 2):
            raise ValueError("tangent_basis_local must have shape (3, 2).")
        object.__setattr__(self, "tangent_basis_local", tangent)
        polygon = _readonly(self.polygon_local_2d)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
            raise ValueError("polygon_local_2d must have shape (N>=3, 2).")
        object.__setattr__(self, "polygon_local_2d", polygon)
        object.__setattr__(self, "boundary_local", _readonly(self.boundary_local))

    def _transform(self, world_from_object: np.ndarray | None) -> np.ndarray:
        transform = (
            self.world_from_object
            if world_from_object is None
            else np.asarray(world_from_object, dtype=float)
        )
        if transform.shape != (4, 4):
            raise ValueError("world_from_object must have shape (4, 4).")
        return transform

    def query(
        self,
        candidate_world: Sequence[float],
        *,
        world_from_object: np.ndarray | None = None,
        plane_tolerance: float = 0.025,
        boundary_tolerance: float = 1e-9,
    ) -> RegionQuery:
        """Test containment and return the nearest point in the region."""
        transform = self._transform(world_from_object)
        object_from_world = np.linalg.inv(transform)
        candidate = _transform_point(
            object_from_world,
            np.asarray(candidate_world, dtype=float).reshape(3),
        )
        relative = candidate - self.plane_origin_local
        plane_distance = float(np.dot(relative, self.normal_local))
        plane_point = candidate - plane_distance * self.normal_local
        coordinates = self.tangent_basis_local.T @ (plane_point - self.plane_origin_local)
        polygon_point = _closest_polygon_point(coordinates, self.polygon_local_2d)
        boundary_distance = float(np.linalg.norm(polygon_point - coordinates))
        nearest_local = self.plane_origin_local + self.tangent_basis_local @ polygon_point
        nearest_world = _transform_point(transform, nearest_local)
        inside = abs(plane_distance) <= plane_tolerance and _point_in_convex_polygon(
            coordinates, self.polygon_local_2d, boundary_tolerance
        )
        return RegionQuery(inside, nearest_world, plane_distance, boundary_distance)

    def contains(self, candidate_world: Sequence[float], **kwargs: Any) -> bool:
        return self.query(candidate_world, **kwargs).inside

    def project(self, candidate_world: Sequence[float], **kwargs: Any) -> np.ndarray:
        return self.query(candidate_world, **kwargs).nearest_point.copy()

    @property
    def center(self) -> np.ndarray:
        center_2d = np.mean(self.polygon_local_2d, axis=0)
        local = self.plane_origin_local + self.tangent_basis_local @ center_2d
        return _transform_point(self.world_from_object, local)

    @property
    def normal(self) -> np.ndarray:
        return _unit(self.world_from_object[:3, :3] @ self.normal_local, name="world normal")

    @property
    def forward(self) -> np.ndarray:
        return _unit(
            self.world_from_object[:3, :3] @ self.tangent_basis_local[:, 0],
            name="world forward",
        )

    @property
    def boundary(self) -> np.ndarray:
        return (
            self.boundary_local @ self.world_from_object[:3, :3].T
            + self.world_from_object[:3, 3]
        )


def build_contact_region(
    scene_mesh: trimesh.Trimesh,
    surface_points: np.ndarray,
    inferred_normals: np.ndarray,
    *,
    object_id: str | None = None,
    world_from_object: np.ndarray | None = None,
    neighborhood_radius: float | None = None,
    seat: bool = False,
) -> ContactRegion:
    """Fit a connected contact region conditioned on one source contact run.

    ``scene_mesh`` and both caller arrays are read only.  Connectivity is a
    hard object boundary: even a geometrically nearby disconnected mesh cannot
    enter the returned patch.
    """
    if not isinstance(scene_mesh, trimesh.Trimesh) or len(scene_mesh.faces) == 0:
        raise ValueError("scene_mesh must be a non-empty trimesh.Trimesh.")
    points = np.asarray(surface_points, dtype=float).copy()
    normals = np.asarray(inferred_normals, dtype=float).copy()
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("surface_points must have shape (N, 3) with N > 0.")
    if normals.ndim == 1:
        normals = np.repeat(normals.reshape(1, 3), len(points), axis=0)
    if normals.shape != points.shape:
        raise ValueError("inferred_normals must have shape (N, 3) or (3,).")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(normals)):
        raise ValueError("surface_points and inferred_normals must be finite.")
    normals = np.asarray([_unit(normal, name="inferred_normals") for normal in normals])

    closest, _, nearest_faces = _closest_scene_points(scene_mesh, points)
    components = _face_components(scene_mesh)
    nearest_components = components[nearest_faces]
    labels, counts = np.unique(nearest_components, return_counts=True)
    component_label = int(labels[int(np.argmax(counts))])
    component_faces = np.flatnonzero(components == component_label)
    seed_faces = np.unique(nearest_faces[nearest_components == component_label])
    patch_type, ledge_evidence = _classify_patch(normals)

    centered = closest - np.mean(closest, axis=0)
    run_spread = float(np.max(np.linalg.norm(centered, axis=1))) if len(points) > 1 else 0.0
    radius = (
        max(0.12, 2.0 * run_spread + 0.06)
        if neighborhood_radius is None
        else float(neighborhood_radius)
    )
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("neighborhood_radius must be positive and finite.")
    selected_faces = _connected_neighborhood(
        scene_mesh,
        component_faces,
        seed_faces,
        closest,
        normals,
        radius,
        ledge_evidence,
    )
    if len(selected_faces) == 0:
        selected_faces = seed_faces.copy()

    patch_normals = np.asarray(scene_mesh.face_normals[selected_faces], dtype=float)
    patch_areas = np.asarray(scene_mesh.area_faces[selected_faces], dtype=float)
    origin, normal, tangent = _plane_frame(closest, normals, patch_normals, patch_areas)
    if seat:
        if normal[2] < 0.0:
            normal = -normal
            tangent[:, 1] = -tangent[:, 1]
        if normal[2] < 0.65 or patch_type == "vertical":
            raise ValueError("A seat affordance requires a roughly upward-facing patch.")
        region_type = "seat"
    else:
        region_type = patch_type

    vertices = np.asarray(
        scene_mesh.vertices[np.unique(scene_mesh.faces[selected_faces])],
        dtype=float,
    )
    vertex_2d = (vertices - origin) @ tangent
    source_2d = (closest - origin) @ tangent
    polygon = _convex_boundary(
        np.vstack([vertex_2d, source_2d]),
        np.mean(source_2d, axis=0),
        radius,
    )

    transform = (
        np.eye(4)
        if world_from_object is None
        else np.asarray(world_from_object, dtype=float).copy()
    )
    if transform.shape != (4, 4):
        raise ValueError("world_from_object must have shape (4, 4).")
    object_from_world = np.linalg.inv(transform)
    origin_local = _transform_point(object_from_world, origin)
    rotation_local = object_from_world[:3, :3]
    normal_local = _unit(rotation_local @ normal, name="local normal")
    tangent_local = rotation_local @ tangent
    tangent_local[:, 0] = _unit(tangent_local[:, 0], name="local forward")
    tangent_local[:, 1] = _unit(tangent_local[:, 1], name="local lateral")
    boundary_local = origin_local + polygon @ tangent_local.T

    resolved_object_id = object_id or f"component:{component_label}"
    fingerprint = np.r_[origin_local, normal_local, float(component_label)]
    digest = hashlib.sha1(np.round(fingerprint, 6).tobytes()).hexdigest()[:12]
    plane_residuals = (vertices - origin) @ normal
    parameters = {
        "support_type": patch_type,
        "neighborhood_radius": radius,
        "plane_rms": float(np.sqrt(np.mean(plane_residuals**2))),
        "normal_horizontal_component": float(np.linalg.norm(normal[:2])),
        "normal_vertical_component": float(abs(normal[2])),
        "polygon_area": _polygon_area(polygon),
        "source_sample_count": int(len(points)),
        "face_count": int(len(selected_faces)),
        "component_face_count": int(len(component_faces)),
        "component_id": component_label,
    }
    return ContactRegion(
        region_id=f"{resolved_object_id}:{region_type}:{digest}",
        region_type=region_type,
        object_id=str(resolved_object_id),
        world_from_object=transform,
        plane_origin_local=origin_local,
        normal_local=normal_local,
        tangent_basis_local=tangent_local,
        polygon_local_2d=polygon,
        boundary_local=boundary_local,
        face_ids=tuple(int(face) for face in selected_faces),
        parameters=parameters,
    )


def build_seat_affordance(
    scene_mesh: trimesh.Trimesh,
    buttock_surface_points: np.ndarray,
    inferred_normals: np.ndarray,
    **kwargs: Any,
) -> ContactRegion:
    """Build an upward seat plane and boundary from a buttock contact run."""
    return build_contact_region(
        scene_mesh,
        buttock_surface_points,
        inferred_normals,
        seat=True,
        **kwargs,
    )


# Descriptive alias for callers that think of this operation as extraction.
extract_contact_region = build_contact_region


__all__ = [
    "ContactRegion",
    "RegionQuery",
    "build_contact_region",
    "build_seat_affordance",
    "extract_contact_region",
]
