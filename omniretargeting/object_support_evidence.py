"""Approximate net-wrench support evidence for moving objects.

The human's contacts are ranked by the human's own centroidal demand
(``source_support_evidence``). A carried object needs the same question asked
of the object: the hands under a box carried through the air hold it up, yet
the human's demand never needs them (they only add load to the human). This
module asks it approximately:

* objects touching each other in a frame form one system, as the human's
  segments do: a seat on its swivel base, a box on a moving cart. Their joint
  and resting forces are internal and cancel; only the system's contacts with
  the static scene and the skin are tested;
* each object is a uniform-density rigid body (its convex hull when the mesh
  is not a closed volume); masses follow volumes, and the wrench-cone test is
  homogeneous in mass, so no absolute mass is needed;
* the prescribed poses give the nominal demand ``F = m (c'' - g)``,
  ``tau = L'`` (``centroidal_demand``); no uncertainty box;
* the skin candidates are the human's own observations
  (``SourceContactTable.candidates``) with the force reversed onto the object;
  scene contacts are the object's vertices within the same detector bounds.
  Every contact gets a full friction cone, without a slip direction: a pushed
  or rolled object stays carried by the floor, so the pushing hands keep
  their object-control role. One candidate per skin region or scene, on a few
  spread-out points;
* environment first: in a frame where the scene alone can carry the system,
  no skin contact is its support. Otherwise every skin contact in a minimal
  feasible set is support, in any friction scenario.

The result only promotes a skin contact to ``support``; it never demotes the
human's own roles, and frames it cannot explain change nothing.
"""

from dataclasses import replace

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from .centroidal_demand import estimate_centroidal_demand
from .contact_mechanics import ContactCandidate, explain_contacts
from .contact_schedule import _iter_true_runs
from .source_support_evidence import FRICTION_SCENARIOS, MAX_CANDIDATES
from .surface_observation import ContactSurface
from .support_parallel import SupportPool
from .wrench_feasibility import DualWrenchCone

# Spread-out points kept per contact candidate (enough to keep a patch's
# extent, i.e. its moment capability) and object vertices probed against
# other surfaces (dense enough to find a small resting patch on a large part).
POINTS_PER_CANDIDATE = 8
OBJECT_SAMPLES = 2048
_OBJECT_PRIOR = {}


def farthest_points(points, count):
    """Indices of up to ``count`` points spread over the set, extremes first."""
    points = np.asarray(points, float).reshape(-1, 3)
    if len(points) <= count:
        return np.arange(len(points))
    chosen = [int(np.argmax(np.linalg.norm(points - points.mean(0), axis=1)))]
    distance = np.linalg.norm(points - points[chosen[0]], axis=1)
    while len(chosen) < count:
        chosen.append(int(np.argmax(distance)))
        distance = np.minimum(distance, np.linalg.norm(points - points[chosen[-1]], axis=1))
    return np.asarray(chosen)


def _candidate(name, observed, surface, normals, bound, sample_indices=None):
    """One sticking candidate on its spread-out points."""
    keep = farthest_points(surface, POINTS_PER_CANDIDATE)
    return ContactCandidate(
        name, observed[keep], surface[keep], normals[keep], FRICTION_SCENARIOS[0], float(bound),
        sample_indices=None if sample_indices is None else sample_indices[keep])


def _onto_object(candidates, name):
    """One region's skin candidates on one object, seen from the object.

    The force reverses onto the object; sticking and sliding samples merge
    into one full friction cone.
    """
    def stack(field):
        return np.concatenate([np.broadcast_to(
            np.asarray(getattr(c, field), float), np.shape(c.surface_points)) for c in candidates])

    indices = [c.sample_indices for c in candidates]
    return _candidate(
        name, stack("observed_points"), stack("surface_points"), -stack("normals"),
        candidates[0].reconstruction_bound_m,
        None if any(i is None for i in indices) else np.concatenate(indices))


def object_inertia(mesh):
    """Unit-mass COM, COM inertia (mesh axes) and volume, at uniform density."""
    body = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    source = "closed_mesh"
    if not (body.is_watertight and body.is_winding_consistent) or abs(body.volume) < 1e-12:
        try:
            body, source = body.convex_hull, "convex_hull"
        except Exception:  # flat or degenerate geometry has no hull volume
            body = None
    if body is not None and abs(body.volume) > 1e-12:
        if body.volume < 0:
            body.invert()
        props = body.mass_properties
        return props["center_mass"], props["inertia"] / props["mass"], props["volume"], source
    # Last resort: a thin sheet, unit mass spread evenly over the vertices.
    vertices = np.asarray(mesh.vertices, float)
    offset = vertices - vertices.mean(0)
    covariance = offset.T @ offset / len(vertices)
    volume = float(mesh.area) * 0.01
    return vertices.mean(0), np.trace(covariance) * np.eye(3) - covariance, volume, "vertex_cloud"


def object_demand(surface, mesh, framerate):
    """Nominal net-wrench demand of one prescribed rigid object, per unit mass."""
    center, inertia, volume, source = object_inertia(mesh)
    rotations = Rotation.from_matrix(surface.rotations).as_matrix()
    centers = surface.translations + surface.scales[:, None] * (rotations @ center)
    scale = float(np.median(surface.scales))
    demand = estimate_centroidal_demand(
        centers[:, None], rotations[:, None], np.ones(1), (inertia * scale ** 2)[None],
        framerate=framerate)
    demand["inertia_source"] = f"uniform_density_{source}"
    demand["volume"] = volume * scale ** 3
    return demand


def _world_bounds(surface, margin):
    """(frames, 2, 3) world AABB of a surface, inflated by ``margin``."""
    low, high = surface.vertices.min(0), surface.vertices.max(0)
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1])
                        for z in (low[2], high[2])])
    world = surface.translations[:, None] + surface.scales[:, None, None] * np.einsum(
        "tij,nj->tni", surface.rotations, corners)
    return np.stack((world.min(1) - margin, world.max(1) + margin), axis=1)


def environment_contacts(index, surfaces, others, com, config, framerate):
    """Per frame ``{surface id: candidate}`` of the object touching ``others``.

    The other surface pushes the object toward the object's own COM side,
    which also settles open scans whose faces carry no inside/outside.
    """
    own = surfaces[index]
    local = np.asarray(own.vertices, float)[farthest_points(own.vertices, OBJECT_SAMPLES)]
    points = own.translations[:, None] + own.scales[:, None, None] * np.einsum(
        "tij,nj->tni", own.rotations, local)
    frames, count = points.shape[:2]
    distance = np.full((frames, count), np.inf)
    nearest, normals, velocity = (np.zeros_like(points) for _ in range(3))
    ids = np.full((frames, count), -1)
    for other in others:
        d, p, n, v, _ = surfaces[other].observe(points, framerate)
        take = d < distance
        distance[take], nearest[take], normals[take], velocity[take] = d[take], p[take], n[take], v[take]
        ids[take] = other
    toward = np.sum((com[:, None] - nearest) * normals, axis=-1) < 0
    normals[toward] *= -1
    normal_speed = np.abs(np.sum(velocity * normals, axis=-1))
    eligible = (distance <= config["distance"]) & (
        (normal_speed <= config["normal_speed"]) | (distance <= config["touch_distance"]))
    result = []
    for f in range(frames):
        result.append({
            int(other): _candidate(
                f"<environment {other}>@{index}", points[f, chosen], nearest[f, chosen],
                normals[f, chosen], config["distance"])
            for other in np.unique(ids[f, eligible[f]])
            for chosen in [np.flatnonzero(eligible[f] & (ids[f] == other))]
        })
    return result


class ObjectSupportEvidence:
    """Skin contacts some moving object needs as support, per frame."""

    def __init__(self, decisions, diagnostics):
        self.decisions = tuple(frozenset(frame) for frame in decisions)
        self.frames = len(self.decisions)
        self.diagnostics = diagnostics

    def role(self, region, surface_id, frame):
        return "support" if (str(region), int(surface_id)) in self.decisions[frame] else None


class CombinedSupportEvidence:
    """Role view: support when the source human or a moving object needs it.

    Object evidence only promotes; every other answer is the human's own.
    """

    def __init__(self, source, objects):
        self.source, self.objects = source, objects
        self.frames = objects.frames
        self.diagnostics = {
            **(source.diagnostics if source is not None else {}),
            "object_support": objects.diagnostics,
        }

    def role(self, region, surface_id, frame):
        if self.objects.role(region, surface_id, frame) == "support":
            return "support"
        return None if self.source is None else self.source.role(region, surface_id, frame)


def combine_support_roles(source, objects):
    return source if objects is None else CombinedSupportEvidence(source, objects)


def _systems(objects, contacts, frame):
    """Objects touching each other in ``frame``, as connected groups."""
    parent = {k: k for k in objects}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for k in objects:
        for other in contacts[k][frame]:
            if other in parent:
                parent[find(k)] = find(other)
    groups = {}
    for k in objects:
        groups.setdefault(find(k), []).append(k)
    return list(groups.values())


def _decide(wrench, origin, scene, skin, names, counts, prior=None, updated=None):
    """Skin (region, object) keys the system needs in one frame."""
    needed, carried, explained = set(), True, False
    prior = {} if prior is None else prior
    updated = {} if updated is None else updated
    for friction in FRICTION_SCENARIOS:
        env = [replace(c, friction=friction) for c in scene]
        if env and DualWrenchCone(env, origin=origin).check(wrench)["status"] == "feasible":
            explained = True
            continue
        carried = False
        candidates = env + [replace(c, friction=friction) for c in skin]
        if len(candidates) > MAX_CANDIDATES:
            counts["skipped_frames"] += 1
            return set()
        result = explain_contacts(candidates, wrench, origin=origin, method="dual",
                                  max_candidates=MAX_CANDIDATES, detail="object",
                                  prior_minimal=prior.get(friction, ()))
        updated[friction] = result["minimal_feasible_sets"]
        explained |= result["full"]["status"] == "feasible"
        needed |= {names[n] for n in set().union(*map(set, result["minimal_feasible_sets"]))
                   if n in names}
    counts["environment_carried_frames"] += carried
    counts["unexplained_frames"] += not explained
    return needed


def _restore_short_gaps(decisions, touching, window):
    """A support gap shorter than the derivative window sits below the
    demand's own temporal resolution (as in ``support_sensitivity``): while
    the skin keeps touching, it stays support."""
    restored = 0
    for key, mask in touching.items():
        on = np.array([key in frame for frame in decisions])
        for start, stop in _iter_true_runs(~on):
            if 0 < start and stop < len(on) and stop - start < window and mask[start:stop].all():
                for f in range(start, stop):
                    decisions[f].add(key)
                restored += stop - start
    return restored


def _decide_task(task):
    """Independent mechanics plus local counters, reduced in timeline order."""
    counts = dict(environment_carried_frames=0, unexplained_frames=0, skipped_frames=0)
    if len(task) == 5:  # Small direct callers have no temporal hint.
        return _decide(*task, counts), counts
    frame, system, *values = task
    previous = _OBJECT_PRIOR.get(system)
    prior = previous[1] if previous is not None and previous[0] == frame - 1 else {}
    updated = {}
    needed = _decide(*values, counts, prior=prior, updated=updated)
    _OBJECT_PRIOR[system] = (frame, updated)
    return needed, counts


def build_object_support_evidence(
    table,
    scene_mesh,
    scene_objects,
    config,
    *,
    framerate,
    scene_enclosed_free_space=None,
    map_function=None,
):
    """Return ``(ObjectSupportEvidence or None, diagnostics)``.

    ``table`` is the human's source contact table; ``config`` the surface
    contact configuration whose detector bounds it was built with.
    """
    diagnostics = {"status": "disabled", "method": "approximate_object_net_wrench"}
    if table is None:
        return None, {**diagnostics, "reason": "missing_source_contact_table"}
    if not scene_objects:
        return None, {**diagnostics, "reason": "no_moving_objects"}
    frames = len(table.points)
    surfaces = []
    if scene_mesh is not None and len(scene_mesh.faces):
        surfaces.append(ContactSurface(scene_mesh, frames, enclosed_free_space=scene_enclosed_free_space))
    first = len(surfaces)
    for spec in scene_objects:
        surfaces.append(ContactSurface(
            spec["mesh"], frames, translations=spec.get("translations"), rotations=spec.get("rotations"),
            scales=spec.get("scales"), name=spec.get("name", "object"), enclosed_free_space=False))
    per_frame, regions = table.candidates(
        friction=FRICTION_SCENARIOS[0],
        reconstruction_bound_m=config["distance"],
        normal_speed_m_s=config["normal_speed"],
        touch_distance_m=config["touch_distance"],
        normal_opposition=config["normal_opposition"],
        sticking_speed_m_s=config["sticking_speed"],
    )
    skin, names = {k: [] for k in range(first, len(surfaces))}, {}
    for frame in per_frame:
        groups = {}
        for c in frame:
            region, surface = regions[c.name]
            if surface >= first:
                groups.setdefault((region, surface), []).append(c)
        for k in skin:
            skin[k].append(tuple(_onto_object(group, f"{region}@{k}")
                                 for (region, k_), group in sorted(groups.items()) if k_ == k))
        names.update({f"{region}@{k}": (region, k) for region, k in groups})
    touched = [k for k in skin if any(skin[k])]
    reports = {k: {"surface_id": k, "name": surfaces[k].name} for k in skin}
    # Only objects reachable from a touched object through possible contact
    # need their demand and scene contacts. Both sides of a possible contact
    # probe the other, so a coarse face resting on a fine post is found.
    bounds = {k: _world_bounds(surfaces[k], config["distance"]) for k in skin}
    demands, contacts, queue = {}, {}, list(touched)
    try:
        while queue:
            k = queue.pop()
            if k in demands:
                continue
            demands[k] = object_demand(surfaces[k], scene_objects[k - first]["mesh"], framerate)
            near = [j for j in skin if j != k and np.any(
                (bounds[k][:, 0] <= bounds[j][:, 1]).all(1) & (bounds[j][:, 0] <= bounds[k][:, 1]).all(1))]
            contacts[k] = environment_contacts(
                k, surfaces, list(range(first)) + near, demands[k]["com"], config, framerate)
            queue += near
            reports[k].update(inertia_source=demands[k]["inertia_source"],
                              volume_m3=float(demands[k]["volume"]))
    except ValueError as error:
        return None, {**diagnostics, "reason": f"demand_estimate_failed: {error}"}
    counts = dict(skin_contact_frames=0, environment_carried_frames=0,
                  unexplained_frames=0, skipped_frames=0)
    decisions = [set() for _ in range(frames)]
    _OBJECT_PRIOR.clear()
    jobs, job_frames = [], []
    for f in range(frames):
        for system in _systems(sorted(demands), contacts, f):
            touching = [c for k in system for c in skin[k][f]]
            if not touching:
                continue
            counts["skin_contact_frames"] += 1
            wrench = sum(demands[k]["volume"] * demands[k]["wrench"][f] for k in system)
            scene = [c for k in system for other, c in contacts[k][f].items() if other < first]
            local_names = {c.name: names[c.name] for c in touching}
            jobs.append((f, tuple(system), wrench, demands[system[0]]["origin"],
                         scene, touching, local_names))
            job_frames.append(f)
    if map_function is None:
        with SupportPool() as pool:
            results = pool.map(_decide_task, jobs)
    else:
        results = map_function(_decide_task, jobs)
    for f, (needed, local_counts) in zip(job_frames, results):
        decisions[f] |= needed
        for key, value in local_counts.items():
            counts[key] += value
    touching = {key: np.array([any(names[c.name] == key for c in frame) for frame in skin[key[1]]])
                for key in sorted(set(names.values()))}
    window = int(next(iter(demands.values()))["derivative_window"]) if demands else 0
    counts["restored_gap_frames"] = _restore_short_gaps(decisions, touching, window)
    counts["skin_support_frames"] = sum(bool(frame) for frame in decisions)
    support_frames = {}
    for frame in decisions:
        for region, surface in frame:
            key = f"{region}@{surface}"
            support_frames[key] = support_frames.get(key, 0) + 1
    diagnostics.update(
        status="active",
        friction_scenarios=list(FRICTION_SCENARIOS),
        points_per_candidate=POINTS_PER_CANDIDATE,
        object_samples=OBJECT_SAMPLES,
        demand_uncertainty="nominal_only",
        objects=[reports[k] for k in sorted(reports)],
        support_region_frames=support_frames,
        physical_support_verified=False,
        force_distribution_inferred=False,
        **counts,
    )
    return ObjectSupportEvidence(decisions, diagnostics), diagnostics
