"""Resolve signed-side ambiguity at finite open-mesh edges and vertices."""

import numpy as np
import trimesh


class OpenSurfaceFeatures:
    def __init__(self, mesh):
        self.mesh = mesh
        self.incident = [set() for _ in mesh.vertices]
        for face, vertices in enumerate(mesh.faces):
            for vertex in vertices:
                self.incident[int(vertex)].add(face)

    def reliable_witnesses(
        self,
        points,
        nearest,
        face_ids,
        reliable,
        evidence_tree,
        evidence_points,
        evidence_normals,
        evidence_radius,
    ):
        """A nearest triangle is not a unique normal at a crease.

        Check all incident faces at a feature. Open external edges additionally
        require an orthogonal projection on the observed finite surface. The
        test has no penetration-depth cutoff: deep interior face projections
        remain measurable when their side is supported by evidence.
        """
        reliable = np.array(reliable, bool, copy=True)
        candidates = np.flatnonzero(reliable)
        if not len(candidates):
            return reliable
        triangles = self.mesh.triangles[face_ids[candidates]]
        bary = trimesh.triangles.points_to_barycentric(triangles, nearest[candidates])
        feature_rows = np.flatnonzero(np.any(bary <= 1e-7, axis=1))
        groups = {}
        for row in feature_rows:
            face = int(face_ids[candidates[row]])
            vertices = tuple(sorted(map(int, self.mesh.faces[face][bary[row] > 1e-7])))
            if not vertices:
                reliable[candidates[row]] = False
                continue
            groups.setdefault(vertices, []).append(int(candidates[row]))
        for vertices, rows in groups.items():
            rows = np.asarray(rows, int)
            incident = set.intersection(*(self.incident[v] for v in vertices))
            faces = np.asarray(sorted(incident), int)
            normals = np.asarray(self.mesh.face_normals)[faces]
            offset = points[rows] - nearest[rows]
            if len(faces) == 1:
                tangent = offset - (offset @ normals[0])[:, None] * normals[0]
                reliable[rows] &= np.linalg.norm(tangent, axis=1) <= 1e-7 * (
                    1 + np.linalg.norm(offset, axis=1)
                )
                continue
            if np.all(normals @ normals[0] >= 1 - 1e-6):
                continue  # A triangulation seam on the same plane.
            distances, indices = evidence_tree.query(
                nearest[rows], k=min(4, len(evidence_points))
            )
            distances = np.asarray(distances).reshape(len(rows), -1)
            indices = np.asarray(indices).reshape(len(rows), -1)
            alignment = np.einsum("nkj,fj->nkf", evidence_normals[indices], normals)
            eligible = (distances[..., None] <= evidence_radius) & (
                np.abs(alignment) >= 0.8
            )
            count = eligible.sum(axis=1)
            vote = np.where(eligible, np.sign(alignment), 0).sum(axis=1)
            supported = (count > 0) & (np.abs(vote) == count)
            oriented = normals[None] * np.where(vote < 0, -1, 1)[..., None]
            projections = np.einsum("nj,nfj->nf", offset, oriented)
            # A tiny deadzone covers floating-point ties, not physical depth.
            consistent = np.all(projections >= -1e-8, axis=1) | np.all(
                projections <= 1e-8, axis=1
            )
            reliable[rows] &= supported.all(axis=1) & consistent
        return reliable
