"""Resolve signed-side ambiguity at finite open-mesh edges and vertices."""

import numpy as np
import trimesh


class OpenSurfaceFeatures:
    def __init__(self, mesh):
        self.mesh = mesh
        # Plain copies of the immutable mesh arrays: trimesh's tracked
        # properties validate their cache on every access.
        self.faces = np.array(mesh.faces, dtype=int)
        self.face_normals = np.array(mesh.face_normals, dtype=float)
        self.triangles = np.array(mesh.triangles, dtype=float)
        self.incident = [set() for _ in mesh.vertices]
        for face, vertices in enumerate(self.faces):
            for vertex in vertices:
                self.incident[int(vertex)].add(face)
        self._feature_faces = {}

    def _incident_faces(self, vertices):
        faces = self._feature_faces.get(vertices)
        if faces is None:
            incident = set.intersection(*(self.incident[v] for v in vertices))
            faces = self._feature_faces[vertices] = np.asarray(sorted(incident), int)
        return faces

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
        triangles = self.triangles[face_ids[candidates]]
        bary = trimesh.triangles.points_to_barycentric(triangles, nearest[candidates])
        feature_rows = np.flatnonzero(np.any(bary <= 1e-7, axis=1))
        # Group feature samples by the sorted vertices of their edge/vertex.
        members = candidates[feature_rows]
        keys = np.sort(np.where(bary[feature_rows] > 1e-7, self.faces[face_ids[members]], -1), axis=1)
        degenerate = keys[:, -1] < 0
        reliable[members[degenerate]] = False
        members, keys = members[~degenerate], keys[~degenerate]
        # Sorted vertex triples (entries in [-1, V)) as one integer: integer
        # order is their row order, so features and groups match np.unique.
        base = len(self.incident) + 1
        if base**3 < 2**62:
            codes = ((keys[:, 0] + 1) * base + keys[:, 1] + 1) * base + keys[:, 2] + 1
            unique, first, inverse = np.unique(codes, return_index=True, return_inverse=True)
            features = keys[first]
        else:
            features, inverse = np.unique(keys, axis=0, return_inverse=True)
            unique = features
        order = np.argsort(inverse.ravel(), kind="stable")
        groups = np.split(members[order], np.cumsum(np.bincount(inverse.ravel(), minlength=len(unique)))[:-1])
        crease = []
        for feature, rows in zip(features, groups):
            faces = self._incident_faces(tuple(int(v) for v in feature if v >= 0))
            normals = self.face_normals[faces]
            offset = points[rows] - nearest[rows]
            coplanar = np.all(np.abs(normals @ normals[0]) >= 1 - 1e-6)
            if len(faces) == 1 or coplanar:
                # Multiple coplanar faces can meet at an external corner.
                # Their shared vertex is not necessarily an internal seam:
                # a tangential offset lies beyond the finite observed patch.
                tangent = offset - (offset @ normals[0])[:, None] * normals[0]
                reliable[rows] &= np.linalg.norm(tangent, axis=1) <= 1e-7 * (
                    1 + np.linalg.norm(offset, axis=1)
                )
                continue
            crease.append((rows, normals, offset))
        if not crease:
            return reliable
        # Pruned as in ``SceneQuery.query``: missing neighbors are never eligible.
        # One query for all crease samples; neighbours are found per point.
        every = np.concatenate([rows for rows, _, _ in crease])
        all_distances, all_indices = evidence_tree.query(
            nearest[every], k=min(4, len(evidence_points)),
            distance_upper_bound=evidence_radius * (1 + 1e-9) + 1e-12,
        )
        all_distances = np.asarray(all_distances).reshape(len(every), -1)
        all_indices = np.minimum(np.asarray(all_indices).reshape(len(every), -1), len(evidence_points) - 1)
        start = 0
        for rows, normals, offset in crease:
            distances = all_distances[start:start + len(rows)]
            indices = all_indices[start:start + len(rows)]
            start += len(rows)
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
