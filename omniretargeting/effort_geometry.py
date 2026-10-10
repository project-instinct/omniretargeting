"""Original-triangle contact queries for the independent dynamics audit.

Correspondence drift is reported separately from actual surface distance.
Collision hull vertices, shell samples and exact support extrema are queried;
as with the scene audit, this is not an exact collision-free certificate.
"""

from collections import OrderedDict

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

from ._sequence.proximity import support_points
from .scene_geometry import collision_geom_ids, geom_surface_offsets
from .surface_observation import ContactSurface

ACTUAL_GAP_M = 0.005
SENSITIVITY_GAP_M = 0.015
SAMPLE_SPACING_M = 0.02


def _extremes(points):
    """Keep every extremal point of a contact patch (including its footprint)."""
    points = np.unique(np.round(points, 10), axis=0)
    if len(points) <= 4:
        return points
    centered = points - points.mean(0)
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.count_nonzero(singular > 1e-8))
    if rank == 0:
        return points[:1]
    if rank == 1:
        coordinate = centered @ axes[0]
        return points[[coordinate.argmin(), coordinate.argmax()]]
    try:
        return points[ConvexHull(centered @ axes[:rank].T).vertices]
    except QhullError:
        return points


class EffortGeometry:
    def __init__(self, model, scene, objects, frames, evidence, *, enclosed=None):
        self.model, self.evidence = model, evidence
        self.geoms = collision_geom_ids(model)
        self.shells = {int(g): geom_surface_offsets(model, int(g), SAMPLE_SPACING_M)
                       for g in self.geoms}
        self.surfaces = [ContactSurface(scene, frames, enclosed_free_space=enclosed)]
        self.surfaces += [ContactSurface(o["mesh"], frames, name=o.get("name", "object"),
                                        translations=o.get("translations"), rotations=o.get("rotations"),
                                        scales=o.get("scales")) for o in objects]
        self.query_arrays = [(np.asfortranarray(s.vertices), np.asfortranarray(s.faces)) for s in self.surfaces]
        self.orientation = []
        for sid, surface in enumerate(self.surfaces):
            selected = evidence["mask"] & (evidence["surface_ids"] == sid)
            ff, slots = np.nonzero(selected)
            world = evidence["environment_w"][ff, slots]
            local = np.einsum("ni,nij->nj", world - surface.translations[ff], surface.rotations[ff])
            local /= surface.scales[ff, None]
            normal = np.einsum("ni,nij->nj", evidence["normals_w"][ff, slots], surface.rotations[ff])
            valid = np.isfinite(local).all(1) & np.isfinite(normal).all(1)
            self.orientation.append((cKDTree(local[valid]) if valid.any() else None, normal[valid]))
        self.cache = OrderedDict()

    def _query(self, surface, sid, points, frame):
        rotation, scale = surface.rotations[frame], surface.scales[frame]
        local = (points - surface.translations[frame]) @ rotation / scale
        vertices, triangles = self.query_arrays[sid]
        squared, faces, near = surface.tree.squared_distance(vertices, triangles, np.asfortranarray(local))
        distance = np.sqrt(np.maximum(squared, 0)) * scale
        normals = surface.normals[faces].copy()
        known = np.full(len(local), surface.oriented and surface.known_side, bool)
        evidence_distance = np.full(len(local), np.inf)
        evidence_alignment = np.zeros(len(local))
        tree, evidence_normals = self.orientation[sid]
        if not known.all() and tree is not None:
            dd, ii = tree.query(near)
            align = np.einsum("ij,ij->i", normals, evidence_normals[ii])
            evidence_distance, evidence_alignment = dd * scale, np.abs(align)
            oriented = (dd * scale <= 0.15) & (np.abs(align) >= 0.8)
            normals[oriented] *= np.where(align[oriented] < 0, -1., 1.)[:, None]
            known |= oriented
        signed = np.where(np.einsum("ij,ij->i", local - near, normals) >= -1e-9, distance, -distance)
        return {"gap": signed, "distance": distance, "known": known,
                "orientation_evidence_distance": evidence_distance,
                "orientation_evidence_alignment": evidence_alignment,
                "normal": normals @ rotation.T,
                "near": near @ rotation.T * scale + surface.translations[frame], "face": faces}

    def _extent_filter(self, data, frame, sid, contacts, diagnostics):
        """Reject deep intrusion even through a thin slab's opposite side.

        A signed shell-to-solid distance can be bounded by slab thickness.
        Use exact convex support toward the contact plane, but only when its
        projection lies on the actual finite original surface (not its extension).
        """
        if not contacts:
            return contacts
        keys, projections, extents = {}, [], []
        for contact in contacts:
            key = (contact["geom"], *np.round(contact["normal"], 12), contact["plane_offset_m"])
            if key in keys:
                continue
            gid, normal, plane = contact["geom"], contact["normal"], contact["plane_offset_m"]
            rotation = data.geom_xmat[gid].reshape(3, 3)
            point = data.geom_xpos[gid] + support_points(self.model, gid, (-normal @ rotation)[None])[0] @ rotation.T
            gap = float(normal @ point - plane)
            keys[key] = len(projections)
            projections.append(point - gap * normal)
            extents.append(gap)
        measured = self._query(self.surfaces[sid], sid, np.asarray(projections), frame)
        by_geom = {d["geom"]: d for d in diagnostics if d["surface"] == sid}
        kept = []
        for contact in contacts:
            key = (contact["geom"], *np.round(contact["normal"], 12), contact["plane_offset_m"])
            index = keys[key]
            known = measured["known"][index] and measured["distance"][index] < 1e-7
            known &= measured["normal"][index] @ contact["normal"] > 1 - 1e-7
            gap = extents[index]
            if known:
                contact["convex_extent_gap_m"] = gap
                diag = by_geom[contact["geom"]]
                if diag["source_candidates"]:
                    ev = self.evidence
                    slots = np.flatnonzero(ev["mask"][frame]
                                          & (ev["robot_geoms"] == contact["geom"])
                                          & (ev["surface_ids"][frame] == sid))
                    if diag["source_support"]:
                        slots = slots[ev["support"][frame, slots]]
                    normal = ev["normals_w"][frame, slots]
                    point = ev["environment_w"][frame, slots]
                    matching = (normal @ contact["normal"] > .8) & (
                        np.abs(point @ contact["normal"] - contact["plane_offset_m"]) <= SENSITIVITY_GAP_M)
                    if matching.any() and gap < diag["gap_m"]:
                        diag.update(gap_m=gap, gap_sign_known=True,
                                    actual_contact=abs(gap) <= ACTUAL_GAP_M,
                                    tolerance_contact=abs(gap) <= SENSITIVITY_GAP_M)
                if abs(gap) > contact["tolerance_m"] + 1e-9:
                    continue
            kept.append(contact)
        return kept

    def frame(self, data, frame):
        token = (int(frame), data.qpos.tobytes())
        if token in self.cache:
            return self.cache[token]
        contacts, diagnostics = [], []
        unknown = {ACTUAL_GAP_M: 0, SENSITIVITY_GAP_M: 0}
        ev = self.evidence
        for sid, surface in enumerate(self.surfaces):
            contact_begin = len(contacts)
            jobs, begin = [], 0
            for gid in self.geoms:
                gid = int(gid)
                rotation = data.geom_xmat[gid].reshape(3, 3)
                slots = np.flatnonzero(ev["mask"][frame] & (ev["surface_ids"][frame] == sid)
                                      & (ev["robot_geoms"] == gid))
                local = self.shells[gid]
                if len(slots):
                    normals = ev["normals_w"][frame, slots]
                    local = np.vstack([local, ev["robot_local_points"][slots],
                                       support_points(self.model, gid, -normals @ rotation)])
                points = data.geom_xpos[gid] + local @ rotation.T
                jobs.append((gid, rotation, slots, points, begin))
                begin += len(points)
            primary = self._query(surface, sid, np.concatenate([job[3] for job in jobs]), frame)
            extras = []
            for gid, rotation, slots, points, begin in jobs:
                measured = {k: v[begin:begin+len(points)] for k, v in primary.items()}
                # Query exact convex support extrema toward the nearest original
                # faces too, so a mesh's far-side learned point cannot hide its sole.
                closest = np.argsort(measured["distance"])[:4]
                extra = support_points(self.model, gid, -measured["normal"][closest] @ rotation)
                extra = data.geom_xpos[gid] + extra @ rotation.T
                extras.append(extra)
            secondary = self._query(surface, sid, np.concatenate(extras), frame)
            extra_begin = 0
            for (gid, rotation, slots, points, begin), extra in zip(jobs, extras):
                measured = {k: v[begin:begin+len(points)] for k, v in primary.items()}
                added = {k: v[extra_begin:extra_begin+len(extra)] for k, v in secondary.items()}
                extra_begin += len(extra)
                points = np.vstack([points, extra])
                measured = {k: np.concatenate([v, added[k]]) for k, v in measured.items()}
                near = measured["distance"] <= SENSITIVITY_GAP_M + 1e-9
                for tolerance in unknown:
                    unknown[tolerance] += int(np.count_nonzero((measured["distance"] <= tolerance) & ~measured["known"]))
                known = np.flatnonzero(measured["known"])
                nearest = int(np.argmin(measured["distance"]))
                deepest = int(known[np.argmin(measured["gap"][known])]) if len(known) else nearest
                gap = float(measured["gap"][deepest] if measured["gap"][deepest] < 0
                            else measured["gap"][nearest])
                sign_known = bool(measured["known"][deepest] if gap < 0 else measured["known"][nearest])
                if not sign_known:
                    gap = float(measured["distance"][nearest])
                body = int(self.model.geom_bodyid[gid])
                source_support = bool(ev["support"][frame, slots].any())
                parts = sorted(set(map(str, ev["parts"][slots])))
                # Source support diagnostics concern its observed surface
                # direction, not an unrelated wall that the same foot clips.
                source_gap = gap
                if len(slots):
                    supports = slots[ev["support"][frame, slots]] if source_support else slots
                    normals = ev["normals_w"][frame, supports]
                    environments = ev["environment_w"][frame, supports]
                    alignment = measured["normal"] @ normals.T > .8
                    plane_error = np.abs(np.einsum("nki,ki->nk", measured["near"][:, None] - environments, normals))
                    # Associate the observed finite surface without penalizing
                    # along-surface drift. Do not confuse parallel floor/seat planes.
                    aligned = np.flatnonzero((alignment & (plane_error <= SENSITIVITY_GAP_M)).any(axis=1))
                    if len(aligned):
                        nearest = int(aligned[np.argmin(measured["distance"][aligned])])
                        signed_indices = aligned[measured["known"][aligned]]
                        deepest = int(signed_indices[np.argmin(measured["gap"][signed_indices])]) if len(signed_indices) else nearest
                        source_gap = float(measured["gap"][deepest] if measured["gap"][deepest] < 0
                                           else measured["gap"][nearest])
                        sign_known = bool(measured["known"][deepest] if source_gap < 0 else measured["known"][nearest])
                        if not sign_known:
                            source_gap = float(measured["distance"][nearest])
                    else:
                        source_gap = float(measured["distance"].min())
                diagnostics.append({"geom": gid, "body": self.model.body(body).name,
                                    "surface": sid, "surface_name": surface.name,
                                    "parts": parts, "source_candidates": int(len(slots)),
                                    "source_support": source_support, "gap_m": source_gap,
                                    "minimum_any_surface_gap_m": gap,
                                    "gap_sign_known": sign_known,
                                    "nearest_unsigned_m": float(measured["distance"][nearest]),
                                    "normal_known": bool(measured["known"][nearest]),
                                    "actual_contact": bool(near.any() and abs(source_gap) <= ACTUAL_GAP_M and measured["known"][nearest]),
                                    "tolerance_contact": bool(near.any() and abs(source_gap) <= SENSITIVITY_GAP_M and measured["known"][nearest])})
                uncertain = np.flatnonzero(near & ~measured["known"])
                if len(uncertain):
                    sample = int(uncertain[np.argmin(measured["distance"][uncertain])])
                    observed_distance = measured["orientation_evidence_distance"][sample]
                    diagnostics[-1].update(
                        uncertain_points_5mm=int(np.count_nonzero(measured["distance"][uncertain] <= ACTUAL_GAP_M)),
                        uncertain_points_15mm=int(len(uncertain)),
                        uncertain_closest_unsigned_m=float(measured["distance"][sample]),
                        orientation_evidence_distance_m=float(observed_distance) if np.isfinite(observed_distance) else None,
                        orientation_evidence_abs_alignment=float(measured["orientation_evidence_alignment"][sample]),
                        surface_watertight_oriented=bool(surface.oriented),
                        surface_free_side_specified=bool(surface.known_side),
                    )
                if len(slots):
                    material = data.geom_xpos[gid] + ev["robot_local_points"][slots] @ rotation.T
                    displacement = material - ev["environment_w"][frame, slots]
                    normals = ev["normals_w"][frame, slots]
                    normal_gap = np.einsum("ij,ij->i", displacement, normals)
                    diagnostics[-1].update(
                        source_correspondence_error_range_m=[float(np.linalg.norm(displacement, axis=1).min()),
                                                             float(np.linalg.norm(displacement, axis=1).max())],
                        source_correspondence_normal_gap_range_m=[float(normal_gap.min()), float(normal_gap.max())])
                groups = {}
                # Far free-side points cannot change a near plane's minimum
                # signed gap or carry a reaction. Keep every negative gap so
                # intrusion cannot be hidden by a shell/surface crossing ring.
                candidates = measured["known"] & (measured["gap"] <= SENSITIVITY_GAP_M)
                for i in np.flatnonzero(candidates):
                    normal = measured["normal"][i]
                    key = (*np.round(normal, 12), round(float(normal @ measured["near"][i]), 10))
                    groups.setdefault(key, []).append(int(i))
                for key, group in groups.items():
                    # Gate each original-face direction independently. A side
                    # penetration must not erase an attained sole/floor patch.
                    ii = min(group, key=lambda i: measured["gap"][i])
                    group_gap = float(measured["gap"][ii])
                    if abs(group_gap) > SENSITIVITY_GAP_M:
                        continue
                    for tolerance in (ACTUAL_GAP_M, SENSITIVITY_GAP_M):
                        if abs(group_gap) > tolerance:
                            continue
                        selected = [i for i in group if measured["distance"][i] <= tolerance]
                        if not selected:
                            continue
                        normal = measured["normal"][selected[0]]
                        for point in _extremes(points[selected]):
                            contacts.append({"body": body, "point": point, "normal": normal,
                                             "geom": gid, "surface": sid, "slot": -1,
                                             "part": ",".join(parts) or self.model.body(body).name,
                                             "source_support": source_support, "gap_m": group_gap,
                                             "plane_offset_m": key[-1],
                                             "tolerance_m": tolerance,
                                             "correspondence_error_m": None})
            contacts[contact_begin:] = self._extent_filter(data, frame, sid, contacts[contact_begin:], diagnostics)
        value = contacts, diagnostics, unknown
        self.cache[token] = value
        if len(self.cache) > 6000:
            self.cache.popitem(last=False)
        return value


def load_geometry(prepared_path, objects_path, model, frames, evidence):
    if prepared_path is None or objects_path is None or evidence is None:
        return None
    import trimesh
    prepared = np.load(prepared_path, allow_pickle=True).item()
    objects = np.load(objects_path, allow_pickle=True).tolist()
    scene = trimesh.load_mesh(prepared_path.parent / "static_scene.obj", process=False)
    return EffortGeometry(model, scene, objects, frames, evidence,
                          enclosed=prepared.get("scene_enclosed_free_space"))
