"""Source net-wrench support evidence for the main retargeting pipeline.

A contact that must carry the body (a stance foot, a hand braced on a box, an
abdomen resting on an edge) and a contact that merely touches (a hand brushing
a wall) look the same geometrically. This module asks the source motion which
contacts its centroidal dynamics need: SMPL-X skin segments give an explicit
uniform-density inertia estimate (``smplx_inertia``), every skin contact
becomes a bounded candidate (``source_contacts``), and each frame keeps the
contact sets that can explain gravity and the observed momentum rates
(``contact_mechanics``, ``contact_temporal``). Only contacts needed in every
friction scenario become support; ``contact_phase`` turns the evidence into
roles, and frames the mechanics cannot explain keep their geometric roles.

A moving object can support the body only in frames where it is part of the
momentarily fixed environment: still (every vertex slower than the sticking
speed) and resting on the static scene, or on another object that is itself
fixed in that frame (a seat stacked on a chair base). Grounding is
transitive within the frame and iterated to a fixed point. A box carried
through the air cannot support its carrier, and a pushed or dragged object,
or one riding a moving support, is being controlled; those contacts keep
their object-control or following roles.
"""

import multiprocessing
import os

import numpy as np
from scipy.spatial.transform import Rotation

from .contact_mechanics import explain_contacts
from .contact_support import SourceSupportEvidence
from .source_contacts import build_source_contacts
from .support_patches import source_skin_spacing
from .support_sensitivity import combine_support_scenarios

# Friction coefficients of the explicit physical scenarios (as in the TRUMANS
# source review). Support must be necessary in every one of them.
FRICTION_SCENARIOS = (0.3, 0.6, 0.9)
MAX_CANDIDATES = 24
_WORKER = {}


def _object_vertices(spec, frames):
    """(frames, vertices, 3) world vertices of a prescribed object."""
    local = np.asarray(spec["mesh"].vertices, float)
    translations = np.broadcast_to(
        np.asarray(spec.get("translations", np.zeros(3)), float), (frames, 3))
    rotations = np.broadcast_to(
        np.asarray(spec.get("rotations", np.eye(3)), float), (frames, 3, 3))
    scales = spec.get("scales")
    scales = np.ones(frames) if scales is None else np.broadcast_to(np.asarray(scales, float), (frames,))
    return translations[:, None] + scales[:, None, None] * np.einsum("fij,vj->fvi", rotations, local)


def object_fixed(scene_mesh, scene_objects, frames, distance, speed, framerate):
    """(objects, frames): the object rests on the fixed environment and is still.

    Fixed environment membership is transitive within a frame: an object is
    fixed when it is still and touches the static scene, or touches another
    object that is fixed in the same frame, iterated to a fixed point. An
    object in the air, being pushed, or resting on a moving object is not
    fixed.
    """
    fixed = object_grounded(scene_mesh, scene_objects, frames, distance)
    still = np.ones((len(scene_objects), frames), bool)
    worlds = []
    for index, spec in enumerate(scene_objects):
        world = _object_vertices(spec, frames)
        velocity = np.gradient(world, 1.0 / framerate, axis=0) if frames > 1 else np.zeros_like(world)
        still[index] = np.linalg.norm(velocity, axis=-1).max(axis=1) <= speed
        worlds.append(world)
    fixed &= still
    _stack_fixed(fixed, still, worlds, scene_objects, distance)
    return fixed


def _stack_fixed(fixed, still, worlds, scene_objects, distance):
    """Complete grounding through object stacks, frame by frame.

    Object-to-object distance queries run only where both objects are still,
    after an axis-aligned bounding-box prefilter; the touching relation is
    then closed transitively, so a seat on a grounded base becomes fixed
    while a seat on a pushed (not fixed) base does not.
    """
    if len(worlds) < 2:
        return
    import igl

    faces = [np.asarray(spec["mesh"].faces, np.int64) for spec in scene_objects]
    touching = {}
    for i in range(len(worlds)):
        for j in range(i + 1, len(worlds)):
            both = np.flatnonzero(still[i] & still[j])
            linked = np.zeros(len(fixed[0]), bool)
            touching[(i, j)] = linked
            if not len(both) or not len(faces[j]) or not len(faces[i]):
                continue
            for frame in both:
                first, second = worlds[i][frame], worlds[j][frame]
                low = np.maximum(first.min(0), second.min(0))
                high = np.minimum(first.max(0), second.max(0))
                separation = np.sqrt((np.maximum(low - high, 0.0) ** 2).sum())
                if separation > distance:
                    continue  # boxes already farther apart than the bound
                gap_ij = np.sqrt(igl.point_mesh_squared_distance(first, worlds[j][frame], faces[j])[0].min())
                gap_ji = np.sqrt(igl.point_mesh_squared_distance(second, worlds[i][frame], faces[i])[0].min())
                linked[frame] = min(gap_ij, gap_ji) <= distance
    changed = True
    while changed:
        changed = False
        for i in range(len(fixed)):
            for j in range(len(fixed)):
                if i == j:
                    continue
                key = (i, j) if i < j else (j, i)
                newly = ~fixed[i] & fixed[j] & touching[key]
                if newly.any():
                    fixed[i] |= newly
                    changed = True


def object_grounded(scene_mesh, scene_objects, frames, distance):
    """(objects, frames): the object touches the static scene within ``distance``."""
    grounded = np.zeros((len(scene_objects), frames), bool)
    if scene_mesh is None or not len(scene_mesh.faces):
        return grounded
    import igl

    vertices = np.asarray(scene_mesh.vertices, float)
    faces = np.asarray(scene_mesh.faces, np.int64)
    for index, spec in enumerate(scene_objects):
        local = np.asarray(spec["mesh"].vertices, float)
        translations = np.broadcast_to(
            np.asarray(spec.get("translations", np.zeros(3)), float), (frames, 3))
        rotations = np.broadcast_to(
            np.asarray(spec.get("rotations", np.eye(3)), float), (frames, 3, 3))
        scales = spec.get("scales")
        scales = np.ones(frames) if scales is None else np.broadcast_to(np.asarray(scales, float), (frames,))
        for frame in range(frames):
            world = translations[frame] + scales[frame] * local @ rotations[frame].T
            squared = igl.point_mesh_squared_distance(world, vertices, faces)[0]
            grounded[index, frame] = np.sqrt(squared.min()) <= distance
    return grounded


def _explain(task):
    scenario, frame = task
    candidates = _WORKER["candidates"][scenario][frame]
    demand = _WORKER["demand"]
    return explain_contacts(candidates, demand["wrench"][frame], origin=demand["origin"],
                            method="dual", max_candidates=MAX_CANDIDATES)


def _explain_and_combine(candidate_sets, regions, demand, skin_spacing=0.0):
    """Per-frame explanations of every scenario, then their conservative union."""
    frames = len(demand["wrench"])
    tasks = [(s, f) for s in range(len(candidate_sets)) for f in range(frames)]
    workers = max(1, min(len(tasks), (os.cpu_count() or 2) // 2))
    _WORKER.update(candidates=candidate_sets, demand=demand)
    pool = None
    try:
        if workers > 1 and hasattr(os, "fork"):
            pool = multiprocessing.get_context("fork").Pool(workers)
        mapper = map if pool is None else (
            lambda function, items: pool.map(function, items, chunksize=max(1, len(items) // (4 * workers))))
        results = list(mapper(_explain, tasks))
        scenarios = [SourceSupportEvidence(results[s * frames:(s + 1) * frames], regions, candidates=sets)
                     for s, sets in enumerate(candidate_sets)]
        return combine_support_scenarios(
            scenarios, demand["wrench"], origin=demand["origin"],
            uncertainty=demand["derivative_disagreement"], map_function=mapper,
            skin_spacing=skin_spacing)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        _WORKER.clear()


def build_source_support_evidence(
    full_body,
    positions,
    joint_names,
    orientations,
    scene_mesh,
    config,
    *,
    framerate,
    scene_objects=(),
    scene_enclosed_free_space=False,
    table=None,
):
    """Return ``(evidence or None, contact table or None, diagnostics)``.

    ``config`` is the surface-contact configuration. ``table`` is the source
    contact table the pipeline already built from the same skin; without it
    the same table is built here.
    """
    from .data_sources.smplx_inertia import source_inertia

    diagnostics = {"status": "disabled", "method": "source_net_wrench_feasibility"}
    if full_body is None or "triangles" not in full_body:
        return None, None, {**diagnostics, "reason": "missing_whole_body_skin"}
    if full_body.get("face_regions") is None:
        return None, None, {**diagnostics, "reason": "missing_skin_segment_labels"}
    if orientations is None:
        return None, None, {**diagnostics, "reason": "missing_source_joint_orientations"}
    if (scene_mesh is None or not len(scene_mesh.faces)) and not scene_objects:
        return None, None, {**diagnostics, "reason": "missing_scene_surface"}
    names = [str(n) for n in full_body["joint_names"]]
    joint_names = [str(n) for n in joint_names]
    if any(n not in joint_names for n in names):
        return None, None, {**diagnostics, "reason": "source_joints_do_not_cover_skin_segments"}
    order = [joint_names.index(n) for n in names]
    positions = np.asarray(positions, float)[:, order]
    orientations = np.asarray(orientations, float)[:, order]
    frames = len(positions)
    if np.asarray(full_body["triangles"]).shape[0] != frames:
        return None, None, {**diagnostics, "reason": "skin_and_joints_on_different_timelines"}
    if table is None:
        table = build_source_contacts(
            full_body, scene_mesh, config, framerate=framerate, scene_objects=scene_objects,
            scene_enclosed_free_space=scene_enclosed_free_space,
            min_duration_s=config["min_contact_frames"] / framerate,
        )
    reference = Rotation.from_quat(
        np.asarray(full_body["reference_joint_orientations"])[0], scalar_first=True).as_matrix()
    try:
        inertia = source_inertia(full_body, positions, orientations, full_body["face_regions"],
                                 reference_rotation=reference, framerate=framerate)
    except ValueError as error:
        return None, table, {**diagnostics, "reason": f"inertia_estimate_failed: {error}"}
    demand = inertia["demand"]
    fixed_env = object_fixed(scene_mesh, scene_objects, frames, config["distance"],
                             config["sticking_speed"], framerate)
    spacing = source_skin_spacing(table.points, table.valid)
    first_object = 1 if scene_mesh is not None and len(scene_mesh.faces) else 0
    candidate_sets, regions, excluded = [], {}, 0
    for friction in FRICTION_SCENARIOS:
        per_frame, names_map = table.candidates(
            friction=friction,
            reconstruction_bound_m=config["distance"],
            normal_speed_m_s=config["normal_speed"],
            touch_distance_m=config["touch_distance"],
            normal_opposition=config["normal_opposition"],
            sticking_speed_m_s=config["sticking_speed"],
        )
        admissible = []
        for frame, candidates in enumerate(per_frame):
            kept = []
            for candidate in candidates:
                surface = names_map[candidate.name][1] - first_object
                if 0 <= surface < len(fixed_env) and not fixed_env[surface, frame]:
                    excluded += 1
                    continue
                kept.append(candidate)
            admissible.append(tuple(kept))
        candidate_sets.append(tuple(admissible))
        regions.update(names_map)
    evidence = _explain_and_combine(candidate_sets, regions, demand, skin_spacing=spacing)
    patches = evidence.diagnostics.get("support_patches") or {}
    support_frames = {}
    for frame, decision in enumerate(evidence.decisions):
        for (region, surface), state in decision.items():
            if state == "support":
                key = f"{region}@{surface}"
                support_frames[key] = support_frames.get(key, 0) + 1
    diagnostics.update(
        status="active",
        friction_scenarios=list(FRICTION_SCENARIOS),
        reconstruction_bound_m=float(config["distance"]),
        inertia_source=demand.get("inertia_source"),
        derivative_window_s=demand.get("derivative_window_seconds"),
        frames=frames,
        explained_frames=frames - len(evidence.diagnostics["unexplained_frames"]),
        support_region_frames=support_frames,
        moving_object_candidates_excluded=excluded,
        fixed_object_frames=[int(g.sum()) for g in fixed_env],
        skin_sample_spacing_m=float(spacing),
        support_patch_frames=patches.get("frames_contracted", 0),
        support_patch_demoted_candidates=patches.get("demoted_candidate_frames", 0),
        physical_support_verified=False,
        force_distribution_inferred=False,
    )
    return evidence, table, diagnostics
