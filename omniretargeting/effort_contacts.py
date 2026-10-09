"""Source-observed contact candidates for the independent effort audit.

Source roles rank importance, not force magnitudes. An optional geometric
contact can carry force too. Carried objects are never treated as world support.
"""

import numpy as np

CONTACT_GAP_M = 0.005


def contact_evidence(saved, model, fixed_objects=None):
    """Load dense, collision-geom material correspondences without new labels."""
    required = (
        "surface_contact_mask", "surface_contact_robot_geoms",
        "surface_contact_robot_local_points", "surface_contact_environment_w",
        "surface_contact_normals_w", "surface_contact_surface_ids",
    )
    if any(name not in saved for name in required):
        return None
    values = {name.removeprefix("surface_contact_"): np.asarray(saved[name]) for name in required}
    frames, count = values["mask"].shape
    for key, shape in {
        "robot_geoms": (count,), "robot_local_points": (count, 3),
        "environment_w": (frames, count, 3), "normals_w": (frames, count, 3),
        "surface_ids": (frames, count),
    }.items():
        if values[key].shape != shape:
            raise ValueError(f"Invalid effort contact shape: {key}")
    geoms = values["robot_geoms"]
    if np.any((geoms < 0) | (geoms >= model.ngeom)):
        raise ValueError("Effort contact geometry is outside the dynamics model")
    values["collision"] = (model.geom_contype[geoms] != 0) | (model.geom_conaffinity[geoms] != 0)
    values["parts"] = np.asarray(saved.get("surface_contact_parts", ["unknown"] * count))
    values["support"] = np.asarray(saved.get("contact_role_surface_support", np.zeros((frames, count), bool)))
    values["fixed_objects"] = fixed_objects
    return values


def frame_contacts(evidence, model, data, frame, *, tolerance=CONTACT_GAP_M):
    """Actual triangle contacts, and optimistic source-correspondence patches.

    With original geometry, scan all collision geoms, not just source-active
    patches. Without it, correspondence gating is a conservative fallback and
    cannot prove missing support. The optimistic set is diagnostic only.
    """
    if evidence is None:
        return [], [], {"missing_evidence": True, "unmodelled_object_contacts": 0}
    attained, optimistic = [], []
    stats = {"missing_evidence": False, "unmodelled_object_contacts": 0,
             "rejected_geometry": 0, "invalid_normals": 0}
    fixed = evidence["fixed_objects"]
    for slot in np.flatnonzero(evidence["mask"][frame] & evidence["collision"]):
        sid = int(evidence["surface_ids"][frame, slot])
        if sid < 0:
            continue
        if sid > 0 and (fixed is None or sid > len(fixed) or not fixed[sid - 1, frame]):
            stats["unmodelled_object_contacts"] += 1
            continue
        geom = int(evidence["robot_geoms"][slot])
        point = data.geom_xpos[geom] + data.geom_xmat[geom].reshape(3, 3) @ evidence["robot_local_points"][slot]
        normal = evidence["normals_w"][frame, slot]
        environment = evidence["environment_w"][frame, slot]
        length = np.linalg.norm(normal)
        if not np.isfinite(np.r_[normal, point, environment]).all() or length < 1e-8:
            stats["invalid_normals"] += 1
            continue
        contact = {"body": int(model.geom_bodyid[geom]), "point": point,
                   "normal": normal / length, "geom": geom, "surface": sid,
                   "slot": int(slot), "source_support": bool(evidence["support"][frame, slot]),
                   "part": str(evidence["parts"][slot]),
                   "correspondence_error_m": float(np.linalg.norm(point - environment))}
        optimistic.append(contact)
        # The full vector gate avoids crediting an infinite extension of a
        # finite seat/table triangle. No invented wrench at a floating foot.
        if contact["correspondence_error_m"] <= CONTACT_GAP_M:
            attained.append(contact)
        else:
            stats["rejected_geometry"] += 1
    stats.update(attained_candidates=len(attained), source_candidates=len(optimistic))
    geometry = evidence.get("geometry")
    stats["original_triangle_geometry_available"] = geometry is not None
    if geometry is not None:
        measured, diagnostics, unknown = geometry.frame(data, frame)
        measured = [c for c in measured if c["tolerance_m"] == tolerance]
        # Missing object inertials matter only when the ROBOT actually touches
        # that object. A source-only label at a floating hand is not a load.
        fixed = evidence.get("fixed_objects_15mm", fixed) if tolerance > CONTACT_GAP_M else fixed
        unmodelled = [c for c in measured if c["surface"] > 0 and (
            fixed is None or c["surface"] > len(fixed) or not fixed[c["surface"] - 1, frame])]
        object_pairs = {}
        for contact in unmodelled:
            key = contact["geom"], contact["surface"]
            gap = contact.get("convex_extent_gap_m", contact["gap_m"])
            if key not in object_pairs or abs(gap) < abs(object_pairs[key]["gap_m"]):
                object_pairs[key] = {
                    "geom": key[0], "surface": key[1],
                    "body": model.body(contact["body"]).name,
                    "gap_m": gap, "gap_sign_known": True,
                    "source_support": contact["source_support"],
                    "point_w_m": contact["point"].tolist(),
                }
        excluded = {id(c) for c in unmodelled}
        attained = [c for c in measured if id(c) not in excluded]
        stats.update(original_triangle_pairs=diagnostics, uncertain_near_surface_points=unknown[tolerance],
                     invalid_source_normals=stats["invalid_normals"], invalid_normals=0,
                     source_only_unmodelled_object_candidates=stats["unmodelled_object_contacts"],
                     unmodelled_object_contacts=len(unmodelled), attained_candidates=len(attained),
                     unmodelled_object_contact_pairs=list(object_pairs.values()),
                     unmodelled_object_surfaces=sorted({c["surface"] for c in unmodelled}),
                     missing_source_support_pairs=[d for d in diagnostics if d["source_support"]
                                                   and not d["actual_contact"]
                                                   and (d["gap_sign_known"] or d["nearest_unsigned_m"] > CONTACT_GAP_M)],
                     missing_source_contact_pairs=[d for d in diagnostics if d["source_candidates"]
                                                   and not d["actual_contact"]
                                                   and (d["gap_sign_known"] or d["nearest_unsigned_m"] > CONTACT_GAP_M)],
                     unresolved_source_support_pairs=[d for d in diagnostics if d["source_support"]
                                                       and not d["normal_known"]
                                                       and d["nearest_unsigned_m"] <= CONTACT_GAP_M],
                     tolerance_m=tolerance)
    return attained, optimistic, stats


def load_fixed_objects(prepared_path, objects_path, frames, fps, *, distance=CONTACT_GAP_M):
    """Prove a moving seat is still and grounded using existing source logic."""
    if prepared_path is None or objects_path is None:
        return None
    import trimesh
    from .source_support_evidence import object_fixed

    prepared = np.load(prepared_path, allow_pickle=True).item()
    objects = np.load(objects_path, allow_pickle=True).tolist()
    if not objects:
        return np.empty((0, frames), bool)
    scene = trimesh.load_mesh(prepared_path.parent / "static_scene.obj", process=False)
    # Reuse original geometry and stillness/transitive-grounding mechanics, but
    # source detection's 30 mm evidence radius is not an attained support gate.
    return object_fixed(scene, objects, frames, distance,
                        prepared["surface_config"]["sticking_speed"], fps)
