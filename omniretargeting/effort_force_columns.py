"""Exact cone-generator reduction for dense contact patches.

For a fixed body and force ray, generalized wrench is affine in p cross f.
An interior moment is a convex combination of extreme moments, so deleting
its force column preserves the complete nonnegative force cone, not just a
working subset. No normal, point or friction coefficient is modified.
"""

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .effort_dynamics import CONE_SIDES, friction_rays


def force_column_indices(contacts, friction, *, outer=False):
    groups = {}
    for index, contact in enumerate(contacts):
        # Do not round different normals into a common friction cone.
        key = (contact["body"], *np.asarray(contact["normal"], float))
        groups.setdefault(key, []).append(index)
    keep, rays_per_contact = [], CONE_SIDES
    for group in groups.values():
        rays = friction_rays(contacts[group[0]]["normal"], friction, outer=outer)
        rays_per_contact = rays.shape[1]
        points = np.asarray([contacts[index]["point"] for index in group])
        for ray_index, ray in enumerate(rays.T):
            first = np.cross(ray, np.eye(3)[np.argmin(np.abs(ray))])
            first /= np.linalg.norm(first)
            second = np.cross(ray, first); second /= np.linalg.norm(second)
            coordinate = (points - points[0]) @ np.c_[first, second]
            if len(group) >= 3:
                try:
                    hull = ConvexHull(coordinate)
                    selected = hull.vertices
                    outside = coordinate @ hull.equations[:, :2].T + hull.equations[:, 2]
                    if outside.max(initial=0) > 1e-12:
                        selected = np.arange(len(group))
                except QhullError:
                    selected = np.arange(len(group))
            else:
                selected = np.arange(len(group))
            keep.extend(group[int(index)] * rays_per_contact + ray_index for index in selected)
    return np.asarray(sorted(keep), int), len(contacts) * rays_per_contact


def reduced_contact_matrix(model, data, contacts, friction, *, outer=False):
    import mujoco

    indices, count = force_column_indices(contacts, friction, outer=outer)
    matrix = np.empty((model.nv, len(indices)))
    rays, groups = {}, {}
    for column, index in enumerate(indices):
        groups.setdefault(contacts[index // CONE_SIDES]["body"], []).append(column)
    for body, columns in groups.items():
        linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, linear, angular, body)
        forces, positions = [], []
        for column in columns:
            index = indices[column]
            contact = contacts[index // CONE_SIDES]
            normal = tuple(np.asarray(contact["normal"], float))
            if normal not in rays:
                rays[normal] = friction_rays(normal, friction, outer=outer)
            forces.append(rays[normal][:, index % CONE_SIDES])
            positions.append(contact["point"])
        forces = np.asarray(forces)
        moments = np.cross(np.asarray(positions) - data.xpos[body], forces)
        matrix[:, columns] = linear.T @ forces.T + angular.T @ moments.T
    return matrix, indices, count
