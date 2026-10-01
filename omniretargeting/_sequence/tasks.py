"""Immutable mapped positions, scene graphs and physical end-effector tasks."""

from dataclasses import replace

import mujoco
import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay, QhullError, cKDTree

from .._refinement.math import _lock_contact_offsets, _skew_symmetric
from .._solver.targets import (
    _approach_contact_target,
    _transform_swing_target,
    _world_target_regime,
)
from ..contact import _is_surface_only_contact_mode
from ..scene_geometry import collision_geom_ids, geom_surface_offsets
from ..self_collision import SelfCollisionQuery
from ..utils import calculate_laplacian_matrix
from .orientation import OrientationTargets


def frozen_array(value, shape, label):
    value = np.asarray(value, dtype=float).copy()
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(f"{label} must be finite with shape {shape}")
    value.setflags(write=False)
    return value


def _uniform_delaunay_laplacian(simplices, count):
    """``calculate_laplacian_matrix(uniform_weight=True)`` of the simplex graph."""
    first, second = np.triu_indices(simplices.shape[1], 1)
    adjacency = np.zeros((count, count), dtype=bool)
    adjacency[simplices[:, first], simplices[:, second]] = True
    adjacency |= adjacency.T
    degree = adjacency.sum(1)
    connected = degree > 0
    laplacian = np.zeros((count, count))
    laplacian[adjacency] = -np.repeat(
        np.ones(count)[connected] / degree[connected], degree[connected]
    )
    laplacian[np.flatnonzero(connected), np.flatnonzero(connected)] = 1.0
    return laplacian


def graph_operator(points, robot_count):
    """Use the source-fixed Delaunay graph; handle planar/short skeletons."""
    try:
        laplacian = _uniform_delaunay_laplacian(Delaunay(points).simplices, len(points))
    except QhullError:
        adjacency = {i: set() for i in range(len(points))}
        neighbors = cKDTree(points).query(points, k=min(5, len(points)))[1]
        for i, row in enumerate(np.asarray(neighbors).reshape(len(points), -1)):
            for j in row:
                if i != j:
                    adjacency[i].add(int(j))
                    adjacency[int(j)].add(i)
        laplacian = calculate_laplacian_matrix(points, adjacency, uniform_weight=True)
    operator = np.zeros((3 * len(points), 3 * robot_count))
    for axis in range(3):
        operator[axis::3, axis::3] = laplacian[:, :robot_count]
    return operator


class SequenceTasks:
    def __init__(
        self,
        model,
        coordinates,
        initial,
        source_positions,
        source_names,
        mapping,
        root_positions,
        tasks,
        priority,
        config,
        *,
        environment_points=None,
        scene_query=None,
        contact_offset_choice=None,
        extra_tasks=(),
        orientation_targets=(),
    ):
        self.model, self.coords = model, coordinates
        self.data = mujoco.MjData(model)
        self.config = config
        self.frames = len(initial)
        self.names = list(mapping)
        if not self.names:
            raise ValueError("Sequence retargeting needs a nonempty joint mapping")
        all_positions = frozen_array(
            source_positions, (self.frames, len(source_names), 3), "source_positions"
        )
        indices = [list(source_names).index(name) for name in self.names]
        self.human = all_positions[:, indices].copy()
        self.human.setflags(write=False)
        self.roots = frozen_array(root_positions, (self.frames, 3), "root_positions")
        self.bodyids = [model.body(mapping[name]).id for name in self.names]
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        self.root_body = (
            model.body(priority["source_root_robot_body"]).id
            if priority.get("source_root_robot_body") is not None
            else int(model.jnt_bodyid[free[0]])
            if len(free)
            else None
        )
        if environment_points is None:
            environment_points = np.empty((self.frames, 0, 3))
        env = np.asarray(environment_points, dtype=float)
        if env.ndim == 2:
            env = np.broadcast_to(env, (self.frames, *env.shape))
        if (
            env.ndim != 3
            or env.shape[0] != self.frames
            or env.shape[2] != 3
            or not np.all(np.isfinite(env))
        ):
            raise ValueError(
                "environment_points must be finite (frames, points, 3) or (points, 3)"
            )
        self.operators = [
            graph_operator(np.vstack([self.human[f], env[f]]), len(self.names))
            for f in range(self.frames)
        ]
        self.axes = []
        raw_axes = list(config.get("pose_axes", []))
        for edge in priority.get("pose_direction", {}).get("edges", []):
            if all(name in self.names for name in edge):
                raw_axes.append(
                    {
                        "weights": {edge[0]: -1, edge[1]: 1},
                        "sigma": config["bone_sigma"],
                    }
                )
        for axis in raw_axes:
            weights = np.zeros(len(self.names))
            for name, value in axis["weights"].items():
                if name not in self.names:
                    raise ValueError(f"Unknown source name in pose axis: {name}")
                weights[self.names.index(name)] = float(value)
            if not np.isclose(weights.sum(), 0) or not np.any(weights):
                raise ValueError(
                    "Pose axes must be nonzero translation-invariant linear combinations"
                )
            sigma = float(axis.get("sigma", config["bone_sigma"]))
            if not np.isfinite(sigma) or sigma <= 0:
                raise ValueError("Pose axis sigma must be finite and positive")
            self.axes.append((weights, sigma))
        # If no semantic edges were provided, derive them from mapped robot ancestry.
        if not self.axes:
            mapped = {b: i for i, b in enumerate(self.bodyids)}
            for i, b in enumerate(self.bodyids):
                parent = int(model.body_parentid[b])
                while parent and parent not in mapped:
                    parent = int(model.body_parentid[parent])
                if parent in mapped:
                    weights = np.zeros(len(mapped))
                    weights[i] = 1
                    weights[mapped[parent]] -= 1
                    self.axes.append((weights, config["bone_sigma"]))
        self.effectors = {}
        for name, task in tasks.items():
            task = replace(task, preserve_contact_anchors=True)
            if name not in priority.get("effectors", {}):
                raise ValueError(f"Missing robot correspondence for effector {name}")
            spec = dict(priority["effectors"][name])
            tracking = np.asarray(spec.get("tracking_offset", [0, 0, 0]), dtype=float)
            offsets = np.asarray(
                spec.get("contact_offsets", [tracking]), dtype=float
            ).reshape(-1, 3)
            if (
                len(task.tracking_positions) != self.frames
                or len(task.contact_mask) != self.frames
            ):
                raise ValueError("Contact tasks must follow the original timeline")
            self.effectors[name] = dict(
                spec,
                body_id=model.body(spec["robot_body"]).id,
                task=task,
                tracking_offset=tracking,
                contact_offsets=offsets,
            )
            if (
                config.get("source_side_support", False)
                and scene_query is not None
                and spec.get("preserve_world_target", False)
                and spec.get("track_orientation", False)
            ):
                # A finite terrain slab has an outside below its bottom too.
                # During cold initialization retain the source foot's witnessed
                # side, so the initial IK cannot escape through that underside.
                # Normals come from the actual scene, with no axis classification.
                self.effectors[name]["support_witness"] = scene_query.query(
                    np.asarray(task.tracking_positions, dtype=float)
                )
        self.choices = (
            _lock_contact_offsets(model, initial, self.effectors)
            if contact_offset_choice is None
            else {
                k: np.asarray(v, dtype=int).copy()
                for k, v in contact_offset_choice.items()
            }
        )
        for name, spec in self.effectors.items():
            choice = self.choices[name]
            if (
                choice.shape != (self.frames,)
                or np.any(choice < 0)
                or np.any(choice >= len(spec["contact_offsets"]))
            ):
                raise ValueError(f"Invalid contact offset choices for {name}")
        self.scene = scene_query
        self.shells = (
            [
                (
                    int(g),
                    geom_surface_offsets(model, int(g), config["collision_spacing_m"]),
                )
                for g in collision_geom_ids(model)
            ]
            if scene_query is not None
            else []
        )
        from .scene_cache import SceneTrajectoryQueries

        self.scene_cache = (
            SceneTrajectoryQueries(model, scene_query, self.shells)
            if getattr(scene_query, "closed_sign_method", None) == "winding"
            else None
        )
        self.self_query = (
            SelfCollisionQuery(model, config.get("self_collision"))
            if config["self_collision_enabled"]
            else None
        )
        self.extra_tasks = tuple(task for task in extra_tasks if task is not None)
        from ..contact_phase import phase_policy

        self.contact_phase = phase_policy(*self.extra_tasks, *tasks.values())
        self.orientations = OrientationTargets(model, self.frames, orientation_targets)

    def point(self, body, offset=None):
        p = self.data.xpos[body].copy()
        if offset is not None:
            p += self.data.xmat[body].reshape(3, 3) @ offset
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jac, None, p, body)
        return p, jac[:, self.coords.cols]

    def root_anchor(self, frame):
        if self.root_body is None:
            return self.roots[frame], np.zeros((3, self.coords.width))
        return self.point(self.root_body)

    def clearance_rows(self, stats=None, frame=None):
        """Scene and self clearance deficits (metres) at the forwarded pose."""
        stats = {} if stats is None else stats
        if frame is not None:
            for task in self.extra_tasks:
                if hasattr(task, "clearance_rows"):
                    yield from task.clearance_rows(
                        self.data.qpos, frame, self.coords.cols, stats
                    )
        # A closed scene's winding hierarchy is rebuilt by the libigl Python
        # binding on each call. Query all original shell points together, then
        # keep the same per-geom worst witness. Geometry and coverage are unchanged.
        groups = [
            offsets @ self.data.geom_xmat[gid].reshape(3, 3).T
            + self.data.geom_xpos[gid]
            for gid, offsets in self.shells
        ]
        queried = (
            None
            if self.scene_cache is None
            else self.scene_cache.get(self.data.qpos, frame)
        )
        if queried is None and groups:
            queried = self.scene.query(np.concatenate(groups))
        begin = 0
        for (gid, _), points in zip(self.shells, groups):
            end = begin + len(points)
            measured = {key: value[begin:end] for key, value in queried.items()}
            begin = end
            clearance = measured["clearance"]
            reliable = np.flatnonzero(np.isfinite(clearance))
            if not len(reliable):
                continue
            k = reliable[np.argmin(clearance[reliable])]
            stats["scene_depth_m"] = max(
                stats.get("scene_depth_m", 0.0), -float(clearance[k])
            )
            deficit = self.config["scene_clearance_m"] - clearance[k]
            if deficit > 0:
                jac = np.zeros((3, self.model.nv))
                mujoco.mj_jac(
                    self.model,
                    self.data,
                    jac,
                    None,
                    points[k],
                    int(self.model.geom_bodyid[gid]),
                )
                yield (
                    np.atleast_1d(-deficit),
                    measured["directions"][k] @ jac[:, self.coords.cols],
                )
        if self.self_query is not None:
            for row in self.self_query.query(data=self.data):
                stats["self_depth_m"] = max(
                    stats.get("self_depth_m", 0.0), -row["clearance"]
                )
                deficit = self.config["self_clearance_m"] - row["clearance"]
                if deficit > 0:
                    yield (
                        np.atleast_1d(-deficit),
                        self.self_query.jacobian(row, self.data)[self.coords.cols],
                    )

    def frame_rows(self, pose, frame):
        self.data.qpos[:] = pose
        mujoco.mj_forward(self.model, self.data)
        residuals, matrices = [], []
        stats = dict(
            sticking_max_m=0.0,
            contact_normal_max_m=0.0,
            self_depth_m=0.0,
            scene_depth_m=0.0,
            world_tracking_max_m=0.0,
            free_tracking_max_m=0.0,
        )

        def add(residual, jac, sigma):
            residuals.append(np.atleast_1d(residual) / sigma)
            matrices.append(np.atleast_2d(jac) / sigma)

        points, jac = zip(*(self.point(b) for b in self.bodyids))
        points, jac = np.array(points), np.array(jac)
        operator = self.operators[frame]
        add(
            operator @ (points - self.human[frame]).ravel(),
            operator @ jac.reshape(-1, self.coords.width),
            self.config["interaction_sigma_m"],
        )
        for weights, sigma in self.axes:
            vector, target = weights @ points, weights @ self.human[frame]
            norm, target_norm = np.linalg.norm(vector), np.linalg.norm(target)
            if min(norm, target_norm) < 1e-8:
                continue
            direction = vector / norm
            derivative = np.einsum("i,ijk->jk", weights, jac)
            add(
                direction - target / target_norm,
                (np.eye(3) - np.outer(direction, direction)) @ derivative / norm,
                sigma,
            )
        orientation_rows, orientation_jacs, orientation_error = self.orientations.rows(
            self.data, self.coords.cols, frame
        )
        residuals.extend(orientation_rows)
        matrices.extend(orientation_jacs)
        stats["body_orientation_max_rad"] = orientation_error
        for name, spec in self.effectors.items():
            target_data = spec["task"].frame(frame)
            active = target_data["contact"]
            if active and target_data.get("surface_managed", False):
                continue  # same contact is constrained by its rigid surface block
            offset = (
                spec["contact_offsets"][self.choices[name][frame]]
                if active
                else spec["tracking_offset"]
            )
            if not active:
                share = target_data["approach_weight"]
                # Handing over only the target but switching the robot point
                # at onset leaves a jump equal to the center-to-contact offset.
                # Blend the point in the body's local frame as well; its FK
                # Jacobian then includes rotation of the entire offset.
                offset = (1 - share) * offset + share * spec["contact_offsets"][
                    self.choices[name][frame]
                ]
            point, derivative = self.point(spec["body_id"], offset)
            witness = spec.get("support_witness")
            if (
                witness is not None
                and witness["confidence"][frame] > 0
                and witness["clearance"][frame] >= -1e-4
            ):
                normal = witness["directions"][frame]
                surface = witness["surface_points"][frame]
                for support_offset in spec["contact_offsets"]:
                    support, support_jac = self.point(spec["body_id"], support_offset)
                    gap = float(normal @ (support - surface))
                    if gap < self.config["scene_clearance_m"]:
                        add(
                            gap - self.config["scene_clearance_m"],
                            normal @ support_jac,
                            self.config["collision_sigma_m"],
                        )
            if active:
                residual = point - target_data["contact_position"]
                if _is_surface_only_contact_mode(target_data["contact_mode"]):
                    normal = np.asarray(target_data["surface_normal"], dtype=float)
                    norm = np.linalg.norm(normal)
                    if not np.isfinite(norm) or norm < 1e-8:
                        raise ValueError(
                            "Surface-only contacts require reliable nonzero normals"
                        )
                    normal = normal / norm
                    tangent = np.eye(3) - np.outer(normal, normal)
                    add(
                        normal @ residual,
                        normal @ derivative,
                        self.config["contact_normal_sigma_m"],
                    )
                    add(
                        tangent @ residual,
                        tangent @ derivative,
                        self.config["contact_tangent_sigma_m"],
                    )
                    stats["contact_normal_max_m"] = max(
                        stats["contact_normal_max_m"], abs(float(normal @ residual))
                    )
                else:
                    add(residual, derivative, self.config["contact_sigma_m"])
                    stats["sticking_max_m"] = max(
                        stats["sticking_max_m"], float(np.linalg.norm(residual))
                    )
            else:
                world = _world_target_regime(spec, target_data)
                root_point, root_jac = self.root_anchor(frame)
                target = _transform_swing_target(
                    target_data["tracking_position"],
                    self.roots[frame],
                    root_point[2] - self.roots[frame, 2],
                    spec.get("swing_morphology_scale", 1.0),
                    world,
                    robot_root_position=root_point,
                )
                target, free_share = _approach_contact_target(target, target_data)
                if not world:
                    derivative -= free_share * root_jac
                add(
                    point - target,
                    derivative,
                    self.config["world_tracking_sigma_m"]
                    if world
                    else self.config["free_tracking_sigma_m"],
                )
                key = "world_tracking_max_m" if world else "free_tracking_max_m"
                stats[key] = max(stats[key], float(np.linalg.norm(point - target)))
            if spec.get("track_orientation", False):
                rotation = self.data.xmat[spec["body_id"]].reshape(3, 3)
                angular_jac = np.zeros((3, self.model.nv))
                mujoco.mj_jacBody(
                    self.model, self.data, None, angular_jac, spec["body_id"]
                )
                for local_key, target_key, default in [
                    ("orientation_local_up", "up_direction", [0, 0, 1]),
                    ("orientation_local_forward", "forward_direction", [1, 0, 0]),
                ]:
                    if target_key not in target_data:
                        continue
                    direction = rotation @ np.asarray(
                        spec.get(local_key, default), dtype=float
                    )
                    target = np.asarray(target_data[target_key], dtype=float)
                    if np.linalg.norm(target) < 1e-8:
                        continue
                    target = target / np.linalg.norm(target)
                    add(
                        direction - target,
                        -_skew_symmetric(direction) @ angular_jac[:, self.coords.cols],
                        self.config["orientation_sigma"],
                    )
        for residual, jac in self.clearance_rows(stats, frame):
            add(residual, jac, self.config["collision_sigma_m"])
        for task in self.extra_tasks:
            for weight, residual, jac in task.rows(pose, frame, self.coords.cols):
                add(residual * np.sqrt(weight), jac * np.sqrt(weight), 1.0)
            if hasattr(task, "position_rows") and hasattr(task, "modes"):
                for index, residual, _ in task.position_rows(pose, frame, []):
                    key = (
                        "sticking_max_m"
                        if task.modes[frame, index] == "sticking"
                        else "contact_normal_max_m"
                    )
                    stats[key] = max(stats[key], float(np.linalg.norm(residual)))
        return np.concatenate(residuals), sparse.csr_matrix(np.vstack(matrices)), stats
