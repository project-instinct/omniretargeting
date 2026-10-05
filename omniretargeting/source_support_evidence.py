"""Source net-wrench support evidence for the main retargeting pipeline.

A contact that must carry the body (a stance foot, a hand braced on a box, an
abdomen resting on an edge) and a contact that merely touches (a hand brushing
a wall) look the same geometrically. This module asks the source motion which
contacts its centroidal dynamics need: source adapters declare segment inertia
or a labelled reference mesh supplies a geometric estimate. Observed material
contacts become bounded candidates (``source_contacts``), and each frame keeps the
contact sets that can explain gravity and the observed momentum rates
(``contact_mechanics``, ``contact_temporal``). Only contacts needed in every
friction scenario become support; ``contact_phase`` turns the evidence into
roles, and frames the mechanics cannot explain protect every detected contact.

A moving object can support the body only in frames where it is part of the
momentarily fixed environment: still (every vertex slower than the sticking
speed) and resting on the static scene, or on another object that is itself
fixed in that frame (a seat stacked on a chair base). Grounding is
transitive within the frame and iterated to a fixed point. A box carried
through the air cannot support its carrier, and a pushed or dragged object,
or one riding a moving support, is being controlled; those contacts keep
their object-control or following roles.
"""

from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from .contact_mechanics import explain_contacts
from .contact_support import SourceSupportEvidence
from .source_contacts import SourceContactTable, build_source_contacts
from .support_patches import source_skin_spacing
from .support_sensitivity import combine_support_scenarios
from .support_parallel import SupportPool

# Friction coefficients of the explicit physical scenarios (as in the TRUMANS
# source review). Support must be necessary in every one of them.
FRICTION_SCENARIOS = (0.3, 0.6, 0.9)
MAX_CANDIDATES = 24
_SOURCE_PRIOR = {}


def _object_transforms(spec, frames):
    """Per-frame translations, rotations and uniform scales of an object."""
    translations = np.broadcast_to(
        np.asarray(spec.get("translations", np.zeros(3)), float), (frames, 3))
    rotations = np.broadcast_to(
        np.asarray(spec.get("rotations", np.eye(3)), float), (frames, 3, 3))
    scales = spec.get("scales")
    scales = np.ones(frames) if scales is None else np.broadcast_to(np.asarray(scales, float), (frames,))
    return translations, rotations, scales


def _object_vertices(spec, frames, select=slice(None)):
    """(frames, vertices, 3) world vertices of a prescribed object.

    ``select`` restricts the evaluation to some frames; each frame's values
    do not depend on which other frames are evaluated with it.
    """
    local = np.asarray(spec["mesh"].vertices, float)
    translations, rotations, scales = (a[select] for a in _object_transforms(spec, frames))
    return translations[:, None] + scales[:, None, None] * np.einsum("fij,vj->fvi", rotations, local)


# Lipschitz shortcuts only accept a touching answer this far inside the bound,
# far above the rounding of the bound itself.
_BOUND_MARGIN = 1e-9
# Frames evaluated together when testing stillness: bounds the temporary
# world-vertex arrays of long clips without changing any frame's result.
_STILL_CHUNK = 64


def _object_still(spec, frames, speed, framerate):
    """(frames,) every vertex is slower than ``speed`` (``np.gradient`` speeds).

    Evaluated in frame chunks with a one-frame halo, using exactly the
    one-sided edge and central interior differences of ``np.gradient``.
    """
    still = np.ones(frames, bool)
    if frames < 2:
        return still
    step = 1.0 / framerate
    for start in range(0, frames, _STILL_CHUNK):
        stop = min(frames, start + _STILL_CHUNK)
        low, high = max(0, start - 1), min(frames, stop + 1)
        world = _object_vertices(spec, frames, slice(low, high))
        velocity = np.empty((stop - start,) + world.shape[1:])
        for frame in range(start, stop):
            if frame == 0:
                value = (world[1 - low] - world[0 - low]) / step
            elif frame == frames - 1:
                value = (world[frame - low] - world[frame - 1 - low]) / step
            else:
                value = (world[frame + 1 - low] - world[frame - 1 - low]) / (2.0 * step)
            velocity[frame - start] = value
        still[start:stop] = np.linalg.norm(velocity, axis=-1).max(axis=1) <= speed
    return still


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
    for index, spec in enumerate(scene_objects):
        still[index] = _object_still(spec, frames, speed, framerate)
    fixed &= still
    _stack_fixed(fixed, still, scene_objects, distance)
    return fixed


def _stack_fixed(fixed, still, scene_objects, distance):
    """Complete grounding through object stacks, frame by frame.

    Object-to-object distance queries run only where both objects are still,
    after an axis-aligned bounding-box prefilter; the touching relation is
    then closed transitively, so a seat on a grounded base becomes fixed
    while a seat on a pushed (not fixed) base does not.

    Point-to-mesh distance is invariant under the object's rigid motion and
    scales with its uniform scale, so each object's triangle tree is built
    once in its own frame; the other object's world vertices are mapped
    into that frame for every query.
    """
    if len(scene_objects) < 2:
        return
    import igl

    frames = len(fixed[0])
    local = [np.asarray(spec["mesh"].vertices, float) for spec in scene_objects]
    faces = [np.asarray(spec["mesh"].faces, np.int64) for spec in scene_objects]
    transforms = [_object_transforms(spec, frames) for spec in scene_objects]
    trees = {}

    def gap(points, index, frame):
        """World distance from ``points`` to object ``index`` at ``frame`` and its nearest point."""
        if index not in trees:
            trees[index] = igl.AABB()
            trees[index].init(local[index], faces[index])
        translation, rotation, scale = (a[frame] for a in transforms[index])
        mapped = (points - translation) @ rotation / scale
        squared = trees[index].squared_distance(local[index], faces[index], mapped)[0]
        nearest = int(np.argmin(squared))
        return scale * np.sqrt(squared[nearest]), nearest

    def mapped_point(owner, vertex, other, frame):
        """Vertex of ``owner`` in the frame of ``other`` at ``frame``, and that object's scale."""
        t_o, r_o, s_o = (a[frame] for a in transforms[owner])
        t_x, r_x, s_x = (a[frame] for a in transforms[other])
        world = t_o + s_o * r_o @ local[owner][vertex]
        return (world - t_x) @ r_x / s_x, s_x

    touching = {}
    for i in range(len(scene_objects)):
        for j in range(i + 1, len(scene_objects)):
            both = np.flatnonzero(still[i] & still[j])
            linked = np.zeros(frames, bool)
            touching[(i, j)] = linked
            if not len(both) or not len(faces[j]) or not len(faces[i]):
                continue
            # Lipschitz shortcut, as in ``object_grounded``: the last touching
            # witness vertex, followed in the other object's own frame.
            witness = None
            for frame in both:
                if witness is not None:
                    owner, other, vertex, value, position = witness
                    point, scale = mapped_point(owner, vertex, other, frame)
                    if value + scale * np.linalg.norm(point - position) <= distance - _BOUND_MARGIN:
                        linked[frame] = True
                        continue
                first = _object_vertices(scene_objects[i], frames, slice(frame, frame + 1))[0]
                second = _object_vertices(scene_objects[j], frames, slice(frame, frame + 1))[0]
                low = np.maximum(first.min(0), second.min(0))
                high = np.minimum(first.max(0), second.max(0))
                separation = np.sqrt((np.maximum(low - high, 0.0) ** 2).sum())
                if separation > distance:
                    continue  # boxes already farther apart than the bound
                gap_ij, vertex_i = gap(first, j, frame)
                gap_ji, vertex_j = gap(second, i, frame)
                linked[frame] = min(gap_ij, gap_ji) <= distance
                owner, other, vertex, value = (i, j, vertex_i, gap_ij) if gap_ij <= gap_ji else (j, i, vertex_j, gap_ji)
                witness = (owner, other, vertex, value, mapped_point(owner, vertex, other, frame)[0])
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
    """(objects, frames): the object touches the static scene within ``distance``.

    The scene's triangle tree is built once and queried every frame, with
    the same exact distances as a per-query ``point_mesh_squared_distance``.
    A frame needs no query when the last queried nearest vertex, moved by
    the object's motion since, is provably still within ``distance``
    (point-to-mesh distance is 1-Lipschitz): only a touching answer is ever
    taken from this bound.
    """
    grounded = np.zeros((len(scene_objects), frames), bool)
    if scene_mesh is None or not len(scene_mesh.faces):
        return grounded
    import igl

    vertices = np.asarray(scene_mesh.vertices, float)
    faces = np.asarray(scene_mesh.faces, np.int64)
    tree = igl.AABB()
    tree.init(vertices, faces)
    for index, spec in enumerate(scene_objects):
        local = np.asarray(spec["mesh"].vertices, float)
        radius = float(np.linalg.norm(local, axis=1).max(initial=0.0))
        translations, rotations, scales = _object_transforms(spec, frames)
        witness = queried = None
        for frame in range(frames):
            if witness is not None:
                vertex, gap, position = witness
                moved = translations[frame] + scales[frame] * rotations[frame] @ local[vertex]
                if gap + np.linalg.norm(moved - position) <= distance - _BOUND_MARGIN:
                    grounded[index, frame] = True
                    continue
            if queried is not None:
                # Every vertex moved at most |Δt| + ‖ΔsR‖₂·max|v| since the last
                # complete query, so the nearest one is still that far away.
                last, nearest_gap = queried
                shift = np.linalg.norm(translations[frame] - translations[last]) + radius * np.linalg.norm(
                    scales[frame] * rotations[frame] - scales[last] * rotations[last], 2)
                if nearest_gap - shift > distance + _BOUND_MARGIN:
                    continue
            world = translations[frame] + scales[frame] * local @ rotations[frame].T
            squared = tree.squared_distance(vertices, faces, world)[0]
            nearest = int(np.argmin(squared))
            gap = float(np.sqrt(squared[nearest]))
            grounded[index, frame] = gap <= distance
            witness = (nearest, gap, world[nearest])
            queried = (frame, gap)
    return grounded


def _explain(task):
    scenario, frame, candidates, wrench, origin = task
    previous = _SOURCE_PRIOR.get(scenario)
    prior = previous[1] if previous is not None and previous[0] == frame - 1 else ()
    result = explain_contacts(candidates, wrench, origin=origin, method="dual",
                              max_candidates=MAX_CANDIDATES, detail="source",
                              prior_minimal=prior)
    _SOURCE_PRIOR[scenario] = (frame, result["minimal_feasible_sets"])
    return result


def _explain_and_combine(candidate_sets, regions, demand, skin_spacing=0.0, map_function=None):
    """Per-frame explanations of every scenario, then their conservative union."""
    if map_function is None:
        with SupportPool() as pool:
            return _explain_and_combine(candidate_sets, regions, demand, skin_spacing, pool.map)
    frames = len(demand["wrench"])
    _SOURCE_PRIOR.clear()
    tasks = [(s, f, sets[f], demand["wrench"][f], demand["origin"])
             for s, sets in enumerate(candidate_sets) for f in range(frames)]
    results = list(map_function(_explain, tasks))
    scenarios = [SourceSupportEvidence(results[s * frames:(s + 1) * frames], regions, candidates=sets)
                 for s, sets in enumerate(candidate_sets)]
    from ._core.preprocess_cache import CertificateMap
    from .support_sensitivity import _certify

    certificates = CertificateMap(map_function, _certify)
    return combine_support_scenarios(
        scenarios, demand["wrench"], origin=demand["origin"],
        uncertainty=demand["derivative_disagreement"], map_function=certificates,
        skin_spacing=skin_spacing,
        min_demotion_frames=int(demand.get("derivative_window") or 0))


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
    inertia_model=None,
    map_function=None,
):
    """Return ``(evidence or None, contact table or None, diagnostics)``.

    ``table`` contains the existing geometric observations, including material
    markers without a mesh. An adapter's explicit ``inertia_model`` supplies
    segment COMs/inertias independently of contact-surface representation.
    Missing physical input is reported unresolved, not as a disabled feature.
    """
    from .source_inertia import mesh_source_inertia, segment_source_inertia

    diagnostics = {"status": "unresolved", "enabled": True,
                   "method": "source_net_wrench_feasibility"}
    if orientations is None:
        return None, table, {**diagnostics, "reason": "missing_source_joint_orientations"}
    if (scene_mesh is None or not len(scene_mesh.faces)) and not scene_objects:
        return None, table, {**diagnostics, "reason": "missing_scene_surface"}
    frames = len(positions)
    if table is None:
        if full_body is None or not {"points", "triangles"}.intersection(full_body):
            return None, None, {**diagnostics, "reason": "missing_source_material_observations"}
        table = build_source_contacts(
            full_body, scene_mesh, config, framerate=framerate, scene_objects=scene_objects,
            scene_enclosed_free_space=scene_enclosed_free_space,
            min_duration_s=config["min_contact_frames"] / framerate,
        )
    if len(table.points) != frames:
        raise ValueError("Source contacts and segment dynamics must share the original timeline")
    try:
        if inertia_model is not None:
            inertia = segment_source_inertia(
                inertia_model, positions, orientations, joint_names, framerate=framerate
            )
        else:
            if full_body is None or full_body.get("face_regions") is None:
                return None, table, {**diagnostics, "reason": "missing_source_inertia_model"}
            names = [str(n) for n in full_body["joint_names"]]
            joint_names = [str(n) for n in joint_names]
            if any(n not in joint_names for n in names):
                return None, table, {**diagnostics, "reason": "source_joints_do_not_cover_skin_segments"}
            order = [joint_names.index(n) for n in names]
            reference = Rotation.from_quat(
                np.asarray(full_body["reference_joint_orientations"])[0], scalar_first=True
            ).as_matrix()
            inertia = mesh_source_inertia(
                full_body, np.asarray(positions)[:, order], np.asarray(orientations)[:, order],
                full_body["face_regions"], reference_rotation=reference, framerate=framerate,
            )
    except ValueError as error:
        return None, table, {**diagnostics, "reason": f"inertia_estimate_failed: {error}"}
    demand = inertia["demand"]
    fixed_env = object_fixed(scene_mesh, scene_objects, frames, config["distance"],
                             config["sticking_speed"], framerate)
    spacing = source_skin_spacing(table.points, table.valid)
    first_object = 1 if scene_mesh is not None and len(scene_mesh.faces) else 0
    candidate_sets, regions, excluded = [], {}, 0
    geometric_candidates = None
    for friction in FRICTION_SCENARIOS:
        if geometric_candidates is None or type(table) is not SourceContactTable:
            per_frame, names_map = table.candidates(
                friction=friction,
                reconstruction_bound_m=config["distance"],
                normal_speed_m_s=config["normal_speed"],
                touch_distance_m=config["touch_distance"],
                normal_opposition=config["normal_opposition"],
                sticking_speed_m_s=config["sticking_speed"],
            )
            geometric_candidates = per_frame
        else:
            if not np.isfinite(friction) or friction < 0:
                raise ValueError("Mechanical candidate assumptions must be finite and nonnegative")
            # Friction changes wrench rays, never the geometric observations
            # or slip grouping. Each scenario retains its own ray cache.
            per_frame = tuple(tuple(replace(c, friction=friction) for c in row)
                              for row in geometric_candidates)
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
    evidence = _explain_and_combine(candidate_sets, regions, demand, skin_spacing=spacing,
                                    map_function=map_function)
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
        inertia_assumptions=demand.get("inertia_assumptions"),
        derivative_window_s=demand.get("derivative_window_seconds"),
        frames=frames,
        explained_frames=frames - len(evidence.diagnostics["unexplained_frames"]),
        unexplained_frames=evidence.diagnostics["unexplained_frames"],
        support_region_frames=support_frames,
        moving_object_candidates_excluded=excluded,
        fixed_object_frames=[int(g.sum()) for g in fixed_env],
        skin_sample_spacing_m=float(spacing),
        support_patch_frames=patches.get("frames_contracted", 0),
        support_patch_demoted_candidates=patches.get("demoted_candidate_frames", 0),
        support_patch_restored_candidates=patches.get("restored_candidate_frames", 0),
        support_demotion_window_frames=int(demand.get("derivative_window") or 0),
        physical_support_verified=False,
        force_distribution_inferred=False,
    )
    return evidence, table, diagnostics
