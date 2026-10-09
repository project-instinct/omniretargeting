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


def frame_contacts(evidence, model, data, frame):
    """Two sets: attained patches, and optimistic source-correspondence patches.

    The latter only diagnoses overload that persists even granting the source
    support plan. It cannot certify attained contact or mask a floating robot.
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
    return attained, optimistic, stats


def load_fixed_objects(prepared_path, objects_path, frames, fps):
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
    # This is the same original geometry and grounding/stillness rule as the
    # source support preparation, not an object mass inferred from its mesh.
    return object_fixed(scene, objects, frames, prepared["surface_config"]["distance"],
                        prepared["surface_config"]["sticking_speed"], fps)
