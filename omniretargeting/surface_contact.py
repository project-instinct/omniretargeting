"""Whole-body scene contact maps on fixed learned surface correspondences.

UMR III-C: source and robot use the SAME environment point assignment. The
residual is (robot_point - environment_point) - source_contact_vector. Both
the environment points and source vectors remain in unscaled metric world
coordinates. Only source-active samples generate objectives.
"""

from dataclasses import asdict

import numpy as np

from .contact_schedule import _iter_true_runs
from .surface_learning import bind_whole_body
from .surface_observation import observe_contacts
from .surface_tasks import SurfaceKinematics, _face_frames

DEFAULTS = {
    "enabled": True,
    "distance": 0.03,
    "min_contact_frames": 1,
    "sigma": 0.015,
    "weight": 5.0,
    "priority": "balanced",
    "tolerance": 0.01,
    "regression": 0.002,
    "touch_distance": 0.008,
    "normal_opposition": 0.2,
    "normal_speed": 0.2,
    "sticking_speed": 0.04,
    "endpoint_regression": 0.001,
}


def surface_contact_config(config=None):
    cfg = {**DEFAULTS, **(config or {})}
    if cfg["priority"] not in {"balanced", "after_end_effectors"}:
        raise ValueError(
            "surface_contact.priority must be balanced or after_end_effectors"
        )
    for key in (
        "distance",
        "sigma",
        "weight",
        "tolerance",
        "regression",
        "touch_distance",
        "normal_speed",
        "sticking_speed",
        "endpoint_regression",
    ):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"surface_contact.{key} must be finite and positive")
    if cfg["touch_distance"] >= cfg["distance"]:
        raise ValueError("surface_contact.touch_distance must be smaller than distance")
    if not 0 <= cfg["normal_opposition"] < 1:
        raise ValueError("surface_contact.normal_opposition must be in [0, 1)")
    if (
        int(cfg["min_contact_frames"]) != cfg["min_contact_frames"]
        or cfg["min_contact_frames"] < 1
    ):
        raise ValueError(
            "surface_contact.min_contact_frames must be a positive integer"
        )
    return cfg


class SurfaceContactTasks:
    """Immutable contact map shared by all solving and acceptance stages."""

    def __init__(
        self,
        model,
        bindings,
        environment_points,
        source_vectors,
        active,
        normals,
        config=None,
        diagnostics=None,
        *,
        confidence=None,
        modes=None,
        surface_ids=None,
        pose_normal_targets=None,
        pose_normal_valid=None,
    ):
        self.model, self.cfg = model, surface_contact_config(config)
        self.bindings = tuple(bindings)
        self.kinematics = SurfaceKinematics(model, bindings)
        self.environment_points = np.array(environment_points, dtype=float, copy=True)
        self.source_vectors = np.array(source_vectors, dtype=float, copy=True)
        self.active = np.array(active, dtype=bool, copy=True)
        self.normals = np.array(normals, dtype=float, copy=True)
        self.confidence = np.array(
            self.active if confidence is None else confidence, dtype=float, copy=True
        )
        self.modes = np.array(
            np.where(self.active, "contact", "none") if modes is None else modes,
            dtype="U8",
            copy=True,
        )
        self.surface_ids = np.array(
            np.zeros(self.active.shape, int) if surface_ids is None else surface_ids,
            dtype=int,
            copy=True,
        )
        self.targets = self.environment_points + self.source_vectors
        # Environment normals describe contact, whereas transported robot
        # normals describe pose. They must never be silently interchanged.
        self.pose_normal_targets = (
            None
            if pose_normal_targets is None
            else np.array(pose_normal_targets, dtype=float, copy=True)
        )
        self.pose_normal_valid = (
            np.zeros(self.active.shape, dtype=bool)
            if pose_normal_targets is None
            else np.array(
                self.active if pose_normal_valid is None else pose_normal_valid,
                dtype=bool,
                copy=True,
            )
        )
        shape = self.targets.shape
        if (
            len(shape) != 3
            or shape[1:] != (len(bindings), 3)
            or self.environment_points.shape != shape
            or self.source_vectors.shape != shape
            or self.normals.shape != shape
            or self.active.shape != shape[:2]
            or self.confidence.shape != shape[:2]
            or self.modes.shape != shape[:2]
            or self.surface_ids.shape != shape[:2]
            or not np.isfinite(self.confidence).all()
            or np.any((self.confidence < 0) | (self.confidence > 1))
            or not all(
                np.isfinite(a).all()
                for a in (
                    self.targets,
                    self.environment_points,
                    self.source_vectors,
                    self.normals,
                )
            )
        ):
            raise ValueError(
                "Surface contact map requires finite arrays sharing the original timeline"
            )
        if np.any(np.linalg.norm(self.normals[self.active], axis=-1) < 1e-8):
            raise ValueError("Active surface contacts require nonzero scene normals")
        if self.pose_normal_valid.shape != self.active.shape:
            raise ValueError("Pose normal validity must share the contact timeline")
        if self.pose_normal_targets is not None:
            if (
                self.pose_normal_targets.shape != shape
                or not np.isfinite(self.pose_normal_targets).all()
                or np.any(
                    np.linalg.norm(
                        self.pose_normal_targets[self.pose_normal_valid], axis=-1
                    )
                    < 1e-8
                )
            ):
                raise ValueError(
                    "Pose normal targets must be finite nonzero unit directions"
                )
            self.pose_normal_targets /= np.maximum(
                np.linalg.norm(self.pose_normal_targets, axis=-1, keepdims=True), 1e-12
            )
            self.pose_normal_targets.setflags(write=False)
        self.pose_normal_valid.setflags(write=False)
        self.weights = np.zeros(self.active.shape)
        labels = np.array([b.part for b in bindings])
        for label in np.unique(labels):
            group = labels == label
            count = max(int(self.active[:, group].sum(1).max()), 1)
            self.weights[:, group] = (
                self.active[:, group]
                * self.confidence[:, group]
                * (self.cfg["weight"] / count)
            )
        for array in (
            self.environment_points,
            self.source_vectors,
            self.targets,
            self.normals,
            self.active,
            self.weights,
            self.confidence,
            self.modes,
            self.surface_ids,
        ):
            array.setflags(write=False)
        self.diagnostics = diagnostics or {}

    def position_rows(self, pose, frame, dofs):
        for index in np.flatnonzero(self.active[frame]):
            point, _, jacobian, _ = self.kinematics.at(pose, int(index))
            yield int(index), point - self.targets[frame, index], jacobian[:, dofs]

    def rows(self, pose, frame, dofs):
        for index, residual, jacobian in self.position_rows(pose, frame, dofs):
            yield (
                self.weights[frame, index],
                residual / self.cfg["sigma"],
                jacobian / self.cfg["sigma"],
            )

    def normal_rows(self, pose, frame, dofs):
        """Relative reference-pose normal targets, separate from contact rows."""
        if self.pose_normal_targets is None:
            return
        for index in np.flatnonzero(self.active[frame] & self.pose_normal_valid[frame]):
            _, normal, _, jacobian = self.kinematics.at(pose, int(index))
            yield (
                self.weights[frame, index],
                normal - self.pose_normal_targets[frame, index],
                jacobian[:, dofs],
            )

    def errors(self, poses, frame_start=0):
        errors = np.full((len(poses), len(self.bindings)), np.nan)
        for frame, pose in enumerate(poses):
            for index, residual, _ in self.position_rows(pose, frame + frame_start, []):
                errors[frame, index] = np.linalg.norm(residual)
        return errors

    def metrics(self, poses, frame_start=0):
        errors = self.errors(poses, frame_start)
        values = errors[np.isfinite(errors)]
        groups = {}
        labels = np.array([b.part for b in self.bindings])
        for name in np.unique(labels):
            group = errors[:, labels == name]
            finite = group[np.isfinite(group)]
            groups[str(name)] = {
                "active_frame_points": len(finite),
                "mean_m": float(finite.mean()) if len(finite) else None,
                "max_m": float(finite.max()) if len(finite) else None,
            }
        worst = None
        if len(values):
            frame, index = np.unravel_index(np.nanargmax(errors), errors.shape)
            worst = {
                "frame": int(frame + frame_start),
                "sample": int(index),
                "part": self.bindings[index].part,
                "error_m": float(errors[frame, index]),
            }
        return {
            "contact_vector_error_mean_m": float(values.mean())
            if len(values)
            else None,
            "contact_vector_error_max_m": float(values.max()) if len(values) else None,
            "active_frame_points": len(values),
            "groups": groups,
            "worst": worst,
            "penetration_is_measured_separately": True,
        }

    def export(self):
        """Numeric/string NPZ payload, without pickle or source-geometry rewrites."""
        payload = {
            "surface_contact_targets_w": self.targets,
            "surface_contact_environment_w": self.environment_points,
            "surface_contact_source_vectors_w": self.source_vectors,
            "surface_contact_normals_w": self.normals,
            "surface_contact_mask": self.active,
            "surface_contact_confidence": self.confidence,
            "surface_contact_modes": self.modes,
            "surface_contact_surface_ids": self.surface_ids,
            "surface_contact_parts": np.array([b.part for b in self.bindings]),
            "surface_contact_face_ids": np.array([b.face_id for b in self.bindings]),
            "surface_contact_barycentric": np.array(
                [b.barycentric for b in self.bindings]
            ),
            "surface_contact_robot_geoms": np.array(
                [self.model.geom(b.robot_geom).id for b in self.bindings]
            ),
            "surface_contact_robot_local_points": np.array(
                [b.local_point for b in self.bindings]
            ),
            "surface_contact_robot_local_normals": np.array(
                [b.local_normal for b in self.bindings]
            ),
        }
        if self.pose_normal_targets is not None:
            payload["surface_contact_pose_normal_targets_w"] = self.pose_normal_targets
            payload["surface_contact_pose_normal_valid"] = self.pose_normal_valid
        return payload


def build_surface_contact_tasks(
    model,
    source,
    robot_reference_qpos,
    scene_mesh,
    config=None,
    *,
    joint_mapping=None,
    framerate=30.0,
    scene_objects=(),
):
    cfg = surface_contact_config(config)
    report = {
        "status": "disabled",
        "reason": "missing_whole_body_skin_samples",
        "configuration": cfg,
    }
    if not cfg["enabled"]:
        return None, {**report, "reason": "configured_off"}
    if source is None or "full_body" not in source:
        return None, report
    if robot_reference_qpos is None:
        return None, {**report, "reason": "missing_calibrated_robot_reference"}
    if (scene_mesh is None or not len(scene_mesh.faces)) and not scene_objects:
        return None, {**report, "reason": "missing_scene_surface"}
    full = source["full_body"]
    triangles, bary = np.asarray(full["triangles"]), np.asarray(full["barycentric"])
    if (
        triangles.ndim != 4
        or triangles.shape[1:] != (len(bary), 3, 3)
        or bary.shape != (len(bary), 3)
        or not np.isfinite(bary).all()
    ):
        raise ValueError(
            "Whole-body skin samples must have shape (frames, samples, 3, 3)"
        )
    frames, valid = _face_frames(triangles)
    points = np.einsum("tnij,ni->tnj", triangles, bary)
    observation = observe_contacts(
        np.where(valid[..., None], points, 0),
        frames[..., 2],
        valid,
        scene_mesh,
        cfg,
        framerate=framerate,
        scene_objects=scene_objects,
    )
    environment, normals, active = (
        observation[k] for k in ("environment", "normals", "active")
    )
    report["observation"] = observation["diagnostics"]
    source_active = int(active.sum())
    if not source_active:
        return None, {
            **report,
            "reason": "no_source_surface_contact",
            "samples": len(bary),
        }
    bindings, binding_valid, correspondence = bind_whole_body(
        model,
        source,
        robot_reference_qpos,
        cfg.get("correspondence"),
        joint_mapping=joint_mapping,
    )
    active &= binding_valid[None, :]
    retained = np.flatnonzero(active.any(0))
    report.update(
        correspondence=correspondence,
        source_active_frame_points=source_active,
        rejected_frame_points=source_active - int(active.sum()),
    )
    if not len(retained):
        return None, {**report, "reason": "no_valid_contact_correspondences"}
    selected = tuple(bindings[i] for i in retained)
    pose_normals, pose_valid = transported_contact_normals(
        model, source, selected, retained, robot_reference_qpos
    )
    vectors = points[:, retained] - environment[:, retained]
    vectors[~valid[:, retained]] = 0
    report.update(
        status="active",
        reason=None,
        active_frame_points=int(active.sum()),
        contact_samples=len(selected),
        contact_regions=sorted({b.part for b in selected}),
        source_sample_indices=retained.tolist(),
        bindings=[asdict(b) for b in selected],
        runs=[
            {"sample": int(k), "intervals": list(_iter_true_runs(active[:, i]))}
            for k, i in enumerate(retained)
        ],
    )
    return SurfaceContactTasks(
        model,
        selected,
        environment[:, retained],
        vectors,
        active[:, retained],
        normals[:, retained],
        cfg,
        report,
        confidence=observation["confidence"][:, retained],
        modes=observation["modes"][:, retained],
        surface_ids=observation["surface_ids"][:, retained],
        pose_normal_targets=pose_normals,
        pose_normal_valid=pose_valid,
    ), report


def transported_contact_normals(
    model, source, bindings, source_indices, robot_reference_qpos
):
    """UMR reference-face transport without equating human and robot normals.

    The caller supplies the original sample indices; no nearest face search or
    new binding is allowed when replaying a saved contact map.
    """
    full = source["full_body"]
    indices = np.asarray(source_indices, dtype=int)
    reference_triangles = np.asarray(full["reference_vertices"])[
        np.asarray(full["faces"])[np.asarray(full["face_ids"])[indices]]
    ]
    reference, rest_valid = _face_frames(reference_triangles)
    frames, valid = _face_frames(np.asarray(full["triangles"])[:, indices])
    kinematics = SurfaceKinematics(model, bindings)
    normals = np.array(
        [kinematics.at(robot_reference_qpos, k)[1] for k in range(len(bindings))]
    )
    relative = frames @ np.swapaxes(reference, -1, -2)
    valid &= rest_valid[None, :]
    targets = np.einsum("tnij,nj->tni", relative, normals)
    return np.where(valid[..., None], targets, 0.0), valid
