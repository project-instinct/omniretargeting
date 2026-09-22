"""Moving original meshes, reliable signed samples and reverse intersections.

Objects have prescribed poses, never optimization coordinates. Open components
remain unknown. Reverse witnesses prove sampled surface/robot intersections;
their absence is not a certificate of collision-free geometry. In particular,
we never replace a chair and its gaps by a single convex hull.
"""

from collections import OrderedDict

import mujoco
import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation

from .collision_motion import CollisionMotionBounds
from .scene_geometry import SceneQuery, collision_geom_ids, geom_surface_offsets
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
            query = SceneQuery(spec["mesh"], full_geometry=False)
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
                    "radius": float(np.max(np.linalg.norm(surface.vertices, axis=1))),
                }
            )
        self.unknown_components = sum(
            len(obj["query"].components) - len(obj["closed"]) for obj in self.objects
        )
        self.query_count = 0
        self.cache = OrderedDict()

    def describe(self):
        return {
            "representation": "original component triangles and reverse material samples",
            "prescribed_object_motion": True,
            "whole_object_convexification": False,
            "unknown_components": self.unknown_components,
            "closed_components": [len(o["closed"]) for o in self.objects],
            "reverse_samples": [len(o["samples"]) for o in self.objects],
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

    def _forward_samples(self, obj, translation, rotation, scale, band):
        """Batch identical signed queries across robot geoms per component.

        The Python fast-winding binding rebuilds its mesh acceleration data
        on every call. A separate call for every nearby robot geom repeats
        that work many times on a detailed object. Preserve every original
        material sample and split the batched answers back into their geoms.
        """
        samples = {}
        for shell in self.shells:
            gid = shell["geom_id"]
            world = (shell["offsets"] @ self.data.geom_xmat[gid].reshape(3, 3).T
                     + self.data.geom_xpos[gid])
            local = (world - translation) @ rotation / scale
            samples[gid] = (world, local, {})
        for cid in obj["closed"]:
            bounds = obj["query"].components[cid][0].bounds
            groups, points = [], []
            for gid, (_, local, _) in samples.items():
                gap = np.maximum(np.maximum(bounds[0] - local, local - bounds[1]), 0)
                ids = np.flatnonzero(np.linalg.norm(gap, axis=1) <= band / scale)
                if len(ids):
                    groups.append((gid, ids))
                    points.append(local[ids])
            if not points:
                continue
            signed = obj["query"].query(np.vstack(points), component_ids=[cid])
            start = 0
            for gid, ids in groups:
                stop = start + len(ids)
                samples[gid][2][cid] = (
                    ids, {key: value[start:stop] for key, value in signed.items()}
                )
                start = stop
        return samples

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
            forward_samples = self._forward_samples(obj, translation, rotation, scale, band)
            for shell in self.shells:
                gid = shell["geom_id"]
                center = self.data.geom_xpos[gid]
                robot_rotation = self.data.geom_xmat[gid].reshape(3, 3)
                world, _, component_samples = forward_samples[gid]
                for cid, (ids, signed) in component_samples.items():
                    if np.any(signed["uncertain"] | ~np.isfinite(signed["clearance"])):
                        result["failures"].append(
                            f"uncertain closed object {oid} component {cid} geom {gid}"
                        )
                        continue
                    k = int(np.argmin(signed["clearance"]))
                    clearance = float(signed["clearance"][k] * scale)
                    component_depth[cid, gid] = max(0.0, -clearance)
                    if clearance < band:
                        direction = rotation @ signed["directions"][k]
                        result["rows"].append(
                            self._row(
                                world[ids[k]],
                                direction,
                                clearance,
                                gid,
                                oid,
                                ids[k],
                                "closed_surface",
                                dofs,
                                component_id=cid,
                            )
                        )
                result["forward_depth"][oid, gid] = component_depth[:, gid].max(
                    initial=0
                )
                # Object-local sphere broad phase contains the entire robot hull.
                indices = np.asarray(
                    obj["tree"].query_ball_point(
                        (center - translation) @ rotation / scale,
                        (shell["radius"] + band) / scale,
                    ),
                    dtype=int,
                )
                if not len(indices):
                    continue
                points = obj["samples"][indices] @ rotation.T * scale + translation
                in_robot = (points - center) @ robot_rotation
                bounds = shell["bounds"]
                keep = np.all(
                    (in_robot >= bounds[0] - band) & (in_robot <= bounds[1] + band),
                    axis=1,
                )
                indices, points, in_robot = indices[keep], points[keep], in_robot[keep]
                if not len(indices):
                    continue
                values = in_robot @ shell["planes"][:, :3].T + shell["planes"][:, 3]
                facets = np.argmax(values, axis=1)
                phi = values[np.arange(len(values)), facets]
                result["reverse_depth"][oid, gid] = max(0.0, -float(phi.min()))
                result["reverse_witness_count"] += int(np.count_nonzero(phi < -1e-9))
                selected = [
                    int(np.flatnonzero(facets == face)[np.argmin(phi[facets == face])])
                    for face in np.unique(facets[phi < band])
                ]
                selected.sort(key=lambda k: phi[k])
                result["available_rows"] += len(selected)
                selected = selected[: int(self.cfg["max_rows_per_geom"])]
                for k in selected:
                    # At a fixed object material point, moving the robot has the
                    # opposite sign of moving that point out of the robot hull.
                    direction = -(robot_rotation @ shell["planes"][facets[k], :3])
                    result["rows"].append(
                        self._row(
                            points[k],
                            direction,
                            phi[k],
                            gid,
                            oid,
                            indices[k],
                            "reverse_surface",
                            dofs,
                            facet_id=int(facets[k]),
                        )
                    )
                result["selected_rows"] += len(selected)
        return result

    def cached(self, pose, time):
        """Bounded cache for repeated immutable-baseline acceptance queries."""
        key = (float(time), np.ascontiguousarray(pose).tobytes())
        if key not in self.cache:
            value = self.query(pose, time, [])
            value = {key: val for key, val in value.items() if key != "rows"}
            self.cache[key] = value
            if len(self.cache) > 256:
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
    pairs = list(zip(before["forward_components"], after["forward_components"]))
    pairs.append((before["reverse_depth"], after["reverse_depth"]))
    for old, new in pairs:
        bound = np.where(old <= tolerance, tolerance, old + regression)
        if np.any(new > bound + 1e-8):
            reasons.append("moving_collision_regression")
            break
    return reasons
