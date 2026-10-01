"""Cached angle-weighted pseudonormals for explicitly manifold closed meshes.

This is the Baerentzen/Aanaes signed-distance test also provided by libigl.
It is not valid for open or self-intersecting scans; callers opt in only when
that mesh contract is known, and final winding-number audits stay available.
"""

import igl
import numpy as np
import trimesh


class ClosedSurfaceNormals:
    def __init__(self, mesh):
        if not (mesh.is_watertight and mesh.is_winding_consistent):
            raise ValueError(
                "Pseudonormals require a watertight oriented manifold mesh"
            )
        self.vertices = np.asarray(mesh.vertices)
        self.faces = np.asarray(mesh.faces)
        self.face_normals, valid = trimesh.triangles.normals(mesh.triangles)
        if not valid.all():
            raise ValueError("Pseudonormals require nondegenerate triangles")
        self.vertex_normals = igl.per_vertex_normals(
            self.vertices, self.faces, igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE
        )
        # Edge j is opposite barycentric coordinate j.
        edges = np.stack(
            (self.faces[:, [1, 2]], self.faces[:, [2, 0]], self.faces[:, [0, 1]]),
            axis=1,
        )
        _, inverse = np.unique(
            np.sort(edges.reshape(-1, 2), axis=1), axis=0, return_inverse=True
        )
        self.edge_ids = inverse.reshape(-1, 3)
        self.edge_normals = np.zeros((inverse.max() + 1, 3))
        np.add.at(self.edge_normals, inverse, np.repeat(self.face_normals, 3, axis=0))

    def at(self, face_ids, closest):
        bary = trimesh.triangles.points_to_barycentric(
            self.vertices[self.faces[face_ids]], closest
        )
        boundary = bary <= 1e-8  # numerical feature identity, not a distance tolerance
        count = boundary.sum(axis=1)
        normals = self.face_normals[face_ids].copy()
        vertex = count >= 2
        vertex_ids = self.faces[face_ids[vertex], np.argmax(bary[vertex], axis=1)]
        normals[vertex] = self.vertex_normals[vertex_ids]
        edge = count == 1
        edge_ids = self.edge_ids[face_ids[edge], np.argmin(bary[edge], axis=1)]
        normals[edge] = self.edge_normals[edge_ids]
        length = np.linalg.norm(normals, axis=1, keepdims=True)
        return normals / np.maximum(length, 1e-12), np.isfinite(bary).all(axis=1) & (
            length[:, 0] > 1e-10
        )
