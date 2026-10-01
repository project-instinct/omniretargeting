"""Continuous surface clearance transfer, independently of contact detection.

Source gaps stay gaps. This task never creates a support label or moves an
observation onto a surface. The closest extent of the mapped collision geom is
compared with the observed skin-to-scene normal distance, so a sphere's far-side
learned material point cannot masquerade as the robot's contact-facing surface.
"""

import weakref

import mujoco
import numpy as np
from scipy.linalg import qr
from scipy.spatial import ConvexHull, QhullError

from ..surface_contact import surface_contact_config
from ..surface_observation import observe_contacts
from ..surface_tasks import _face_frames, _skew


# Hull vertices of each model's collision meshes, in original vertex order.
_HULL_VERTICES = weakref.WeakKeyDictionary()


def _hull_vertices(model, mesh):
    """Mesh vertices that can be a support point, in their original order.

    A linear function's maximum over the vertices is attained at a convex
    hull vertex; the dropped vertices lie inside or on the hull and can at
    most tie with it (a support point on the same face). ``argmax`` keeps
    the first maximizing index among the retained, ascending indices.
    """
    cache = _HULL_VERTICES.setdefault(model, {})
    if mesh not in cache:
        begin, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = np.asarray(model.mesh_vert[begin : begin + count], float)
        try:
            vertices = vertices[np.sort(ConvexHull(vertices).vertices)]
        except QhullError:
            pass  # degenerate (e.g. planar) mesh: every vertex stays a candidate
        cache[mesh] = vertices
    return cache[mesh]


def support_points(model, geom, directions):
    """Exact support of MuJoCo convex collision geometry in local coordinates."""
    directions = np.asarray(directions, float)
    directions = directions / np.maximum(
        np.linalg.norm(directions, axis=1, keepdims=True), 1e-12
    )
    size = model.geom_size[geom]
    kind = model.geom_type[geom]
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return size[0] * directions
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        points = size[0] * directions
        points[:, 2] += size[1] * np.sign(directions[:, 2])
        return points
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return np.sign(directions) * size
    if kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        return (
            size**2
            * directions
            / np.linalg.norm(size * directions, axis=1, keepdims=True)
        )
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        points = np.zeros_like(directions)
        radius = np.linalg.norm(directions[:, :2], axis=1, keepdims=True)
        points[:, :2] = size[0] * directions[:, :2] / np.maximum(radius, 1e-12)
        points[:, 2] = size[1] * np.sign(directions[:, 2])
        return points
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        vertices = _hull_vertices(model, int(model.geom_dataid[geom]))
        return vertices[np.argmax(directions @ vertices.T, axis=1)]
    raise ValueError("Surface proximity requires a finite robot collision geom")


class SurfaceProximityTasks:
    """Soft geometric relationship, never a contact or force certificate."""

    def __init__(
        self,
        model,
        geoms,
        parts,
        environment,
        source_vectors,
        normals,
        valid,
        *,
        distance_scale,
        sigma,
        weight,
    ):
        self.model, self.data = model, mujoco.MjData(model)
        self.geoms = np.asarray(geoms, int)
        self.environment = np.array(environment, float, copy=True)
        self.source_vectors = np.array(source_vectors, float, copy=True)
        self.normals = np.array(normals, float, copy=True)
        valid = np.asarray(valid, bool)
        if (
            self.environment.shape != self.source_vectors.shape
            or self.normals.shape != self.environment.shape
            or self.environment.shape[1:] != (len(self.geoms), 3)
            or valid.shape != self.environment.shape[:2]
            or not all(
                np.isfinite(a).all()
                for a in (self.environment, self.source_vectors, self.normals)
            )
            or any(
                not np.isfinite(v) or v <= 0 for v in (distance_scale, sigma, weight)
            )
        ):
            raise ValueError(
                "Surface proximity needs consistent finite metric observations"
            )
        lengths = np.linalg.norm(self.normals, axis=-1, keepdims=True)
        if np.any(valid & (lengths[..., 0] < 1e-10)):
            raise ValueError("Valid proximity observations require a surface normal")
        self.normals /= np.maximum(lengths, 1e-12)
        # Smooth relevance, using the existing observation resolution. No new
        # distance cutoff and no privileged body part or action class.
        distance = np.linalg.norm(self.source_vectors, axis=-1)
        relevance = valid * weight / (1 + (distance / distance_scale) ** 2) / sigma**2
        self.weights = np.zeros_like(relevance)
        self.support_gaps = np.zeros_like(relevance)
        skin = self.environment + self.source_vectors
        labels = np.asarray(parts)
        for geom in np.unique(self.geoms):
            group = np.flatnonzero(self.geoms == geom)
            # Keep all observed surface directions (e.g. a seat AND its back).
            # For each tangent plane, compare source and robot SUPPORT extents,
            # not the average skin point or a single winning surface direction.
            projected = np.einsum(
                "tki,tji->tkj", self.normals[:, group], skin[:, group]
            )
            projected = np.where(valid[:, None, group], projected, np.inf)
            minimum = projected.min(axis=2) - np.einsum(
                "tki,tki->tk", self.normals[:, group], self.environment[:, group]
            )
            self.support_gaps[:, group] = np.where(np.isfinite(minimum), minimum, 0.0)
            for index in group:
                count = len(np.unique(self.geoms[labels == labels[index]]))
                self.weights[:, index] = relevance[:, index] / (count * len(group))
        self.parts = labels.copy()
        self.valid = valid.copy()
        self.distance_scale, self.sigma, self.weight = distance_scale, sigma, weight

    def rows(self, pose, frame, dofs):
        self.data.qpos[:] = pose
        mujoco.mj_forward(self.model, self.data)
        residuals, derivatives = [], []
        for geom in np.unique(self.geoms):
            indices = np.flatnonzero((self.geoms == geom) & (self.weights[frame] > 0))
            if not len(indices):
                continue
            normal = self.normals[frame, indices]
            rotation = self.data.geom_xmat[geom].reshape(3, 3)
            local = support_points(self.model, geom, -normal @ rotation)
            points = self.data.geom_xpos[geom] + local @ rotation.T
            error = np.einsum(
                "ij,ij->i", points - self.environment[frame, indices], normal
            )
            error -= self.support_gaps[frame, indices]
            body = int(self.model.geom_bodyid[geom])
            linear, angular = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, linear, angular, body)
            jac = linear - _skew(points - self.data.xpos[body]) @ angular
            gradient = np.einsum("ki,kij->kj", normal, jac)[:, dofs]
            scale = np.sqrt(self.weights[frame, indices])
            residuals.extend(error * scale)
            derivatives.extend(gradient * scale[:, None])
        if residuals:
            # Exact least-squares compression, retaining the constant residual
            # row. Dense sampling does not enlarge the trajectory solve.
            augmented = np.column_stack((np.asarray(derivatives), residuals))
            factor = qr(augmented, mode="r")[0][: min(augmented.shape)]
            yield 1.0, factor[:, -1], factor[:, :-1]

    def export(self):
        return {
            "geoms": self.geoms,
            "parts": self.parts,
            "environment": self.environment,
            "source_vectors": self.source_vectors,
            "normals": self.normals,
            "valid": self.valid,
            "distance_scale": self.distance_scale,
            "sigma": self.sigma,
            "weight": self.weight,
        }


def build_surface_proximity(
    model,
    source,
    surface_motion,
    scene,
    objects,
    config,
    fps,
    *,
    source_contact_table=None,
):
    if source is None or surface_motion is None:
        return None
    if (scene is None or not len(scene.faces)) and not objects:
        return None
    cfg = surface_contact_config(config)
    if not cfg["enabled"]:
        return None
    full = source["full_body"]
    triangles = np.asarray(full["triangles"])
    frames, valid = _face_frames(triangles)
    points = np.einsum("tnij,ni->tnj", triangles, full["barycentric"])
    observed = (
        source_contact_table.observation
        if source_contact_table is not None
        else observe_contacts(
            points,
            frames[..., 2],
            valid,
            scene,
            cfg,
            framerate=fps,
            scene_objects=objects,
        )
    )
    bindings, binding_valid, _ = surface_motion.binding_result
    keep = np.flatnonzero(binding_valid)
    if not len(keep):
        return None
    return SurfaceProximityTasks(
        model,
        [model.geom(bindings[i].robot_geom).id for i in keep],
        [bindings[i].part for i in keep],
        observed["environment"][:, keep],
        points[:, keep] - observed["environment"][:, keep],
        observed["normals"][:, keep],
        valid[:, keep],
        distance_scale=cfg["distance"],
        sigma=cfg["sigma"],
        weight=cfg["weight"],
    )
