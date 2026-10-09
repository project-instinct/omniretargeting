"""Moving original meshes, reliable signed samples and reverse intersections.

Objects have prescribed poses, never optimization coordinates. Open components
remain globally unknown; fixed source observations can orient local patches,
and nearly closed ones are signed where their winding number is decisive.
Reverse witnesses prove sampled surface/robot intersections;
their absence is not a certificate of collision-free geometry. In particular,
we never replace a chair and its gaps by a single convex hull.
"""

from collections import OrderedDict

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation

from .collision_motion import CollisionMotionBounds
from .evidence_scope import EvidenceScope
from .moving_clearance_cache import MovingClearanceCache, empty_summary
from .moving_reverse_cache import cached_reverse
from .moving_sample_summary import summarize
from .scene_geometry import HULL_TOLERANCE, SceneQuery, collision_geom_ids, geom_surface_offsets
from .surface_observation import ContactSurface


class MovingSceneBudgetError(RuntimeError):
    """The requested temporal coverage exceeds the explicit query budget."""


def interpolate_object(surface, time):
    """Original translation/scale and shortest-arc rotation at a frame time."""
    if not np.isfinite(time) or not 0 <= time <= len(surface.scales) - 1:
        raise ValueError("Object query time is outside the original timeline")
    lo = int(np.floor(time))
    hi = min(lo + 1, len(surface.scales) - 1)
    alpha = float(time - lo)
    first, second = surface.rotations[lo], surface.rotations[hi]
    relative = Rotation.from_matrix(first.T @ second).as_rotvec()
    rotation = first @ Rotation.from_rotvec(alpha * relative).as_matrix()
    translation = (1 - alpha) * surface.translations[lo] + alpha * surface.translations[
        hi
    ]
    scale = (1 - alpha) * surface.scales[lo] + alpha * surface.scales[hi]
    return translation, rotation, float(scale)


class MovingSceneQuery:
    """Cache immutable local geometry; return collision rows in robot nv space.

    Closed components use original triangle distances/winding at fixed robot
    material samples. Reverse queries use ALL original vertices and triangle
    centroids inside robot collision hulls. Mesh hulls match MuJoCo's compiled
    representation; curved primitive hulls are inscribed approximations.
    Linearization selects the worst forward sample per geom/component and the
    deepest reverse sample per exit facet. All samples are checked for acceptance.
    """

    def __init__(self, model, scene_objects, frames, config=None):
        self.model, self.frames = model, int(frames)
        self.cfg = {
            "sample_spacing": 0.02,
            "reverse_spacing": 0.008,
            "activation_distance": 0.008,
            "max_rows_per_geom": 24,
            **dict(config or {}),
        }
        for name in ("sample_spacing", "reverse_spacing", "activation_distance"):
            if not np.isfinite(self.cfg[name]) or self.cfg[name] <= 0:
                raise ValueError(f"moving_scene.{name} must be finite and positive")
        if int(self.cfg["max_rows_per_geom"]) < 1:
            raise ValueError("moving_scene.max_rows_per_geom must be positive")
        self.geom_ids = collision_geom_ids(model)
        self.motion_bounds = CollisionMotionBounds(model, self.geom_ids)
        self.data = mujoco.MjData(model)
        self.shells = []
        for gid in self.geom_ids:
            points = geom_surface_offsets(model, int(gid), self.cfg["reverse_spacing"])
            hull = ConvexHull(points)
            equations = hull.equations.copy()
            equations /= np.linalg.norm(equations[:, :3], axis=1)[:, None]
            # Coplanar triangulation facets have the same exit plane.
            _, indices = np.unique(np.round(equations, 10), axis=0, return_index=True)
            offsets = geom_surface_offsets(model, int(gid), self.cfg["sample_spacing"])
            offsets = np.unique(np.vstack([offsets, np.zeros((1, 3))]), axis=0)
            self.shells.append(
                {
                    "geom_id": int(gid),
                    "offsets": offsets,
                    "planes": equations[np.sort(indices)],
                    "radius": float(np.max(np.linalg.norm(points, axis=1))),
                    "bounds": np.array([points.min(0), points.max(0)]),
                }
            )
        self.objects = []
        for spec in scene_objects:
            surface = ContactSurface(
                spec["mesh"],
                frames,
                translations=spec.get("translations"),
                rotations=spec.get("rotations"),
                scales=spec.get("scales"),
                name=spec.get("name", "object"),
            )
            evidence = spec.get("collision_evidence", {})
            query = SceneQuery(
                spec["mesh"], evidence.get("points"), evidence.get("normals"),
                evidence_radius=evidence.get("radius", 0.15), full_geometry=False,
                bounded_object=True,
            )
            face_components = np.empty(len(surface.faces), dtype=int)
            for cid, (_, faces, _, _) in enumerate(query.components):
                face_components[faces] = cid
            observed_open = [cid for cid, component in enumerate(query.components)
                             if not component[2] and query._component_query_data[cid][3]]
            # Nearly closed components with volume: a sample their winding
            # number puts inside penetrates them, whichever surface is nearest
            # (``SceneQuery``). Zero-volume pieces cannot contain a sample.
            nearly_closed = [cid for cid, component in enumerate(query.components)
                             if not component[2] and query.nearly_closed(cid)
                             and abs(component[0].volume) > 1e-12]
            samples = np.vstack(
                [surface.vertices, surface.vertices[surface.faces].mean(axis=1)]
            )
            samples.setflags(write=False)
            self.objects.append(
                {
                    "surface": surface,
                    "query": query,
                    "samples": samples,
                    "tree": cKDTree(samples),
                    "closed": [i for i, part in enumerate(query.components) if part[2]],
                    "observed_open": observed_open,
                    "nearly_closed": nearly_closed,
                    "nearly_bounds": [np.asarray(query.components[cid][0].bounds) for cid in nearly_closed],
                    "face_components": face_components,
                    "radius": float(np.max(np.linalg.norm(surface.vertices, axis=1))),
                }
            )
        self.unknown_components = sum(
            len(obj["query"].components) - len(obj["closed"]) for obj in self.objects
        )
        self.query_count = 0
        self.cache = OrderedDict()
        self.forward_cache = None
        self.material_cache = OrderedDict()
        self.reverse_cache = OrderedDict()
        self.sample_keys = {}
        self.clearance_cache = MovingClearanceCache()
        self._separated_summaries = {}
        self._clearance_enabled = False  # Only production repair enables screening.

    def fresh_state(self):
        """Reuse immutable mesh/BVH construction with independent query state.

        Source geometry, material samples and certified hull memoization are
        invariant for this sequence. Pose buffers, per-transform summaries,
        query counters and trajectory prepasses belong to the new consumer.
        """
        import copy
        result = copy.copy(self)
        result.data = mujoco.MjData(self.model)
        result.cfg = dict(self.cfg)
        result.query_count = 0
        result.cache = OrderedDict()
        result.forward_cache = None
        result.material_cache = OrderedDict()
        result.reverse_cache = OrderedDict()
        result.sample_keys = {}
        result.clearance_cache = MovingClearanceCache()
        result._separated_summaries = {}
        return result

    def prepare_trajectory(self, poses):
        from ._sequence.moving_cache import MovingForwardQueries

        if self.forward_cache is None:
            self.forward_cache = MovingForwardQueries(self)
        self.forward_cache.prepare(poses)

    def describe(self):
        return {
            "representation": "original component triangles and reverse material samples",
            "prescribed_object_motion": True,
            "whole_object_convexification": False,
            "unknown_components": self.unknown_components,
            "closed_components": [len(o["closed"]) for o in self.objects],
            "reverse_samples": [len(o["samples"]) for o in self.objects],
            "locally_observed_open_components": [len(o["observed_open"]) for o in self.objects],
            "nearly_closed_components": [len(o["nearly_closed"]) for o in self.objects],
            "local_evidence_points": [len(o["query"].evidence_points) for o in self.objects],
            "open_forward_scope": "nearest original surface patch within the 3D source-evidence radius",
            "robot_collision_geoms": len(self.geom_ids),
            "reverse_primitive_geometry": "inscribed hull; sampled lower bound",
            "collision_free_certificate": False,
            "query_count": self.query_count,
        }

    def interval_steps(self, first, second, frame, spacing=0.01, max_substeps=64):
        robot_bound = float(
            np.max(self.motion_bounds.displacements(first, second), initial=0)
        )
        object_bound = 0.0
        for obj in self.objects:
            surface = obj["surface"]
            a, b = (
                interpolate_object(surface, frame),
                interpolate_object(surface, frame + 1),
            )
            angle = Rotation.from_matrix(a[1].T @ b[1]).magnitude()
            travel = np.linalg.norm(b[0] - a[0]) + obj["radius"] * (
                max(a[2], b[2]) * angle + abs(b[2] - a[2])
            )
            object_bound = max(object_bound, float(travel))
        steps = max(1, int(np.ceil((robot_bound + object_bound) / spacing)))
        if steps > max_substeps:
            raise MovingSceneBudgetError(
                f"moving object interval {frame}: {steps} substeps exceed {max_substeps}"
            )
        return steps

    def _row(self, point, direction, clearance, gid, oid, sample, kind, dofs, **extra):
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jac(
            self.model, self.data, jac, None, point, int(self.model.geom_bodyid[gid])
        )
        return dict(
            point=np.array(point),
            direction=np.array(direction),
            clearance=float(clearance),
            jacobian=np.asarray(direction) @ jac[:, dofs],
            geom_id=gid,
            object_id=oid,
            sample_id=int(sample),
            kind=kind,
            **extra,
        )

    def _forward_samples(self, obj, translation, rotation, scale, band, shells):
        """Batch identical signed queries across robot geoms per component.

        The Python fast-winding binding rebuilds its mesh acceleration data
        on every call. A separate call for every nearby robot geom repeats
        that work many times on a detailed object. Preserve every original
        material sample and split the batched answers back into their geoms.
        """
        samples = {}
        for shell in shells:
            gid = shell["geom_id"]
            world = (shell["offsets"] @ self.data.geom_xmat[gid].reshape(3, 3).T
                     + self.data.geom_xpos[gid])
            local = (world - translation) @ rotation / scale
            samples[gid] = (world, local, {})
        # A component-local box can exclude a shell even when the shell is
        # near another part of this object. Use the already transformed actual
        # samples, not the robot's visual mesh or an approximate object SDF.
        sample_bounds = {gid: (local.min(0), local.max(0))
                         for gid, (_, local, _) in samples.items()} if len(obj["closed"]) > 1 else {}
        for cid in obj["closed"]:
            bounds = obj["query"].components[cid][0].bounds
            groups, points = [], []
            for gid, (_, local, _) in samples.items():
                if sample_bounds:
                    lower, upper = sample_bounds[gid]
                    box_gap = np.maximum(np.maximum(bounds[0] - upper, lower - bounds[1]), 0.)
                    # Same nonnegative three-coordinate norm as the point test;
                    # a cushion keeps the activation boundary unpruned.
                    distance = np.linalg.norm(box_gap)
                    cushion = 1e-12 * max(1., band / scale, distance)
                    if distance > band / scale + cushion:
                        continue
                gap = np.maximum(np.maximum(bounds[0] - local, local - bounds[1]), 0)
                ids = np.flatnonzero(np.linalg.norm(gap, axis=1) <= band / scale)
                if len(ids):
                    groups.append((gid, ids))
                    points.append(local[ids])
            if not points:
                continue
            reduction = obj.get("repair_reduction")
            if reduction is None:
                signed = obj["query"].query(np.vstack(points), component_ids=[cid])
            else:
                ends = np.r_[0, np.cumsum([len(part) for part in points])]
                signed = reduction.query(np.vstack(points), list(zip(ends[:-1], ends[1:])),
                                         cid, band / scale)
            start = 0
            for gid, ids in groups:
                stop = start + len(ids)
                samples[gid][2][cid] = (
                    ids, {key: value[start:stop] for key, value in signed.items()}
                )
                start = stop
        return samples

    def _material_samples(self, oid, obj, pose, time, translation, rotation, scale, band, shells):
        """Reuse exact unchanged material transforms, never cached Jacobians.

        Fixed-root local corrections often move only one limb. Object-side
        witnesses on unchanged bodies remain identical across those candidates.
        The cache is bounded and includes the exact prescribed object transform.
        """
        samples, missing, signatures = {}, [], {}
        object_pose = (translation.tobytes(), rotation.tobytes(), float(scale))
        for shell in shells:
            gid = shell["geom_id"]
            key = (oid, float(time), gid, self.data.geom_xpos[gid].tobytes(),
                   self.data.geom_xmat[gid].tobytes(), object_pose, float(band))
            signatures[gid] = key
            entry = self.material_cache.get(key)
            if entry is not None:
                samples[gid] = entry
                self.material_cache.move_to_end(key)
            else:
                missing.append(shell)
        if missing:
            prepared = None if self.forward_cache is None else self.forward_cache.get(pose, time, oid)
            fresh = (self._forward_samples(obj, translation, rotation, scale, band, missing)
                     if prepared is None else {shell["geom_id"]: prepared[shell["geom_id"]] for shell in missing})
            self._open_forward_samples(obj, fresh, band / scale)
            for gid, values in fresh.items():
                key = signatures[gid]
                values = summarize(self, oid, obj, gid, values, rotation, scale, band)
                self.material_cache[key] = values
                self.material_cache.move_to_end(key)
                samples[gid] = values
            while len(self.material_cache) > 8192:
                self.material_cache.popitem(last=False)
        return samples

    def _open_forward_samples(self, obj, samples, band):
        """Reliable local exits on the nearest original open patch only.

        A robot point can lie entirely behind an open panel without any panel
        sample inside the robot. Reverse intersections miss that case. Do not
        extend every remote triangle into an infinite collision halfspace;
        query the actual nearest component and require fixed source evidence.
        A nearly closed component encloses its own volume: samples it can
        decide by its winding number (``SceneQuery._decisive_winding``) are
        not left unknown because another surface is nearest.
        """
        if not (obj["observed_open"] or obj["nearly_closed"]) or not samples:
            return
        gids = list(samples)
        local = np.concatenate([samples[g][1] for g in gids])
        surface = obj["surface"]
        # Free-side evidence is local in 3D, not an infinite extrusion of a
        # tangent patch. A distant head behind a lamp fragment is not proof
        # that the head penetrates the lamp. Keep such samples unknown.
        # The search bound only prunes; samples beyond it are out of scope.
        query = obj["query"]
        # Outside the object's hull a sample is known outside at any distance.
        # Beyond the activation band it needs no query: its distance to the
        # object is at least its hull margin, so it has no row either.
        outside = np.zeros(len(local), bool)
        far = np.zeros(len(local), bool)
        if query._hull is not None:
            lower, upper = query._hull_bounds
            box = np.linalg.norm(np.maximum(np.maximum(lower - local, local - upper), 0.), axis=1)
            far = box > band
            outside[far] = True
            near = np.flatnonzero(~far)
            classified = query.hull_classification(local[near], HULL_TOLERANCE, band)
            outside[near], far[near] = classified.T
        # Hull-outside/far samples do not need their evidence scope: the hull
        # already puts them in scope, and far samples are discarded below.
        in_scope = np.zeros(len(local), bool)
        if query.evidence_tree is not None:
            scope = obj.get("scope")
            if scope is None:
                scope = obj["scope"] = EvidenceScope(
                    query.evidence_tree, query.evidence_points, query.evidence_radius)
            unresolved = np.flatnonzero(~outside & ~far)
            if len(unresolved):
                in_scope[unresolved] = scope(local[unresolved])
        in_scope = (in_scope | outside) & ~far
        # Nearest triangles are per point; out-of-scope samples never use one.
        components = np.full(len(local), -1)
        if in_scope.any():
            _, triangles, _ = surface.tree.squared_distance(
                surface.vertices, surface.faces, local[in_scope])
            components[in_scope] = obj["face_components"][triangles]
        begins = np.r_[0, np.cumsum([len(samples[g][1]) for g in gids])]
        for index, gid in enumerate(gids):
            far_indices = np.flatnonzero(far[begins[index]:begins[index + 1]])
            if len(far_indices):
                samples[gid][2]["hull"] = far_indices
        # Same component order as before; components without samples are skipped.
        present = set(np.unique(components[in_scope]).tolist())
        known = np.zeros(len(local), bool)
        for cid in obj["observed_open"]:
            if cid not in present:
                continue
            selected = np.flatnonzero((components == cid) & in_scope)
            known[selected] = self._store(samples, gids, begins, cid, selected,
                                          obj["query"].query(local[selected], component_ids=[cid],
                                                             outside_hull=outside[selected]))
        # A nearly closed component encloses its own volume whichever surface
        # is nearest: a sample still unknown near its bounds (``band``, object
        # units) that its winding number decides inside penetrates it. Being
        # outside it says nothing about the nearer, unknown surface.
        unknown = np.flatnonzero(~known)
        hull_exit = False
        for cid, bounds in zip(obj["nearly_closed"], obj["nearly_bounds"]):
            if not len(unknown):
                break
            candidates = unknown[np.all((local[unknown] >= bounds[0] - band)
                                        & (local[unknown] <= bounds[1] + band), axis=1)]
            gap = np.maximum(np.maximum(bounds[0] - local[candidates], local[candidates] - bounds[1]), 0)
            selected = candidates[(components[candidates] != cid) & (np.linalg.norm(gap, axis=1) <= band)]
            if not len(selected):
                continue
            signed = obj["query"].winding_side(local[selected], cid)
            reliable = ~signed["uncertain"] & (signed["clearance"] < 0)
            selected = selected[reliable]
            known[selected] = self._store(samples, gids, begins, cid, selected,
                                          {key: value[reliable] for key, value in signed.items()})
            unknown = np.flatnonzero(~known)

    @staticmethod
    def _store(samples, gids, begins, cid, selected, signed):
        for index, gid in enumerate(gids):
            keep = (selected >= begins[index]) & (selected < begins[index + 1])
            if np.any(keep):
                samples[gid][2][cid] = (
                    selected[keep] - begins[index],
                    {key: value[keep] for key, value in signed.items()},
                )
        return ~np.asarray(signed["uncertain"], bool)

    def query(self, pose, time, dofs=None):
        dofs = np.arange(self.model.nv) if dofs is None else np.asarray(dofs, dtype=int)
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (self.model.nq,) or not np.isfinite(pose).all():
            raise ValueError("Moving query requires a finite robot qpos")
        self.data.qpos[:] = pose
        mujoco.mj_forward(self.model, self.data)
        self.query_count += 1
        shape = (len(self.objects), self.model.ngeom)
        result = {
            "rows": [],
            "forward_depth": np.zeros(shape),
            "reverse_depth": np.zeros(shape),
            "open_evidence_depth": np.zeros(shape),
            "unknown_open_samples": 0,
            "open_known_samples": set(),
            "open_penetrating_samples": set(),
            "forward_components": [],
            "unknown_components": self.unknown_components,
            "failures": [],
            "reverse_witness_count": 0,
            "available_rows": 0,
            "selected_rows": 0,
            "time": float(time),
        }
        band = self.cfg["activation_distance"]
        for oid, obj in enumerate(self.objects):
            translation, rotation, scale = interpolate_object(obj["surface"], time)
            component_depth = np.zeros((len(obj["query"].components), self.model.ngeom))
            result["forward_components"].append(component_depth)
            # Both surfaces fit inside these spheres. Exclude distant robot
            # geoms before material sampling, winding and reverse KD queries.
            shells = [
                shell for shell in self.shells
                if np.linalg.norm(self.data.geom_xpos[shell["geom_id"]] - translation)
                <= obj["radius"] * scale + shell["radius"] + band
            ]
            if not shells:
                continue
            separated = (self.clearance_cache.separated(
                self, oid, obj, time, translation, rotation, scale, band, shells)
                if self._clearance_enabled else set())
            forward_samples = self._material_samples(
                oid, obj, pose, time, translation, rotation, scale, band,
                [shell for shell in shells if shell["geom_id"] not in separated])
            for shell in shells:
                gid = shell["geom_id"]
                if gid in separated:
                    result["open_known_samples"].update(empty_summary(self, oid, shell)["known"])
                    continue
                center = self.data.geom_xpos[gid]
                robot_rotation = self.data.geom_xmat[gid].reshape(3, 3)
                summary = forward_samples[gid]
                result["open_known_samples"].update(summary["known"])
                result["open_penetrating_samples"].update(summary["penetrating"])
                result["unknown_open_samples"] += summary["unknown"]
                result["failures"].extend(summary["failures"])
                for cid, closed, depth in summary["depths"]:
                    component_depth[cid, gid] = depth
                    if not closed:
                        result["open_evidence_depth"][oid, gid] = max(
                            result["open_evidence_depth"][oid, gid], depth)
                for point, direction, clearance, sample, closed, cid in summary["rows"]:
                    result["rows"].append(self._row(
                        point, direction, clearance, gid, oid, sample,
                        "closed_surface" if closed else "observed_open_surface", dofs,
                        component_id=cid))
                result["forward_depth"][oid, gid] = component_depth[:, gid].max(
                    initial=0
                )
                reverse = cached_reverse(self, oid, obj, shell, time, center,
                                         robot_rotation, translation, rotation, scale, band)
                result["reverse_depth"][oid, gid] = reverse["depth"]
                result["reverse_witness_count"] += reverse["count"]
                result["available_rows"] += reverse["available"]
                for point, direction, phi, sample, facet in reverse["rows"]:
                    result["rows"].append(self._row(point, direction, phi, gid, oid,
                                                   sample, "reverse_surface", dofs,
                                                   facet_id=facet))
                result["selected_rows"] += len(reverse["rows"])
        return result

    def cached(self, pose, time):
        """Bounded cache for repeated immutable-baseline acceptance queries."""
        key = (float(time), np.ascontiguousarray(pose).tobytes())
        if key not in self.cache:
            value = self.query(pose, time, [])
            value = {key: val for key, val in value.items() if key != "rows"}
            self.cache[key] = value
            # One window can contain thousands of interframe poses. Keep its
            # exact scene answers through merit and nonlinear acceptance.
            if len(self.cache) > 8192:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]


def moving_regression_reasons(before, after, tolerance=0.0005, regression=0.0005):
    """Known closed components and reverse evidence are checked separately."""
    reasons = []
    if after["failures"]:
        reasons.append("moving_geometry_query_failed")
    if after["unknown_components"] != before["unknown_components"]:
        reasons.append("moving_geometry_coverage_changed")
    if not before.get("open_penetrating_samples", set()).issubset(
            after.get("open_known_samples", set())):
        reasons.append("moving_local_evidence_lost")
    pairs = list(zip(before["forward_components"], after["forward_components"]))
    pairs.append((before["reverse_depth"], after["reverse_depth"]))
    for old, new in pairs:
        bound = np.where(old <= tolerance, tolerance, old + regression)
        if np.any(new > bound + 1e-8):
            reasons.append("moving_collision_regression")
            break
    return reasons
