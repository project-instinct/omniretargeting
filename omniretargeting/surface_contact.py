"""Whole-body scene contact maps on fixed learned surface correspondences.

Observed environment points and source contact vectors remain in unscaled
metric world coordinates. Prepared active contacts target observed scene
material points; source penetration is preserved as evidence, not copied into
the robot goal. Low-level callers without explicit goals retain the UMR vector
residual. Only source-active samples generate objectives.
"""

from dataclasses import asdict, replace

import mujoco
import numpy as np

from .contact_schedule import _iter_true_runs
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
    "normal_weight": 0.05,
    "normal_sigma": 0.35,
    "contact_binding_calibration": True,
}


def surface_contact_config(config=None):
    from .config_validation import known_options

    known_options(config, DEFAULTS, 'surface_contact', extra=('correspondence',))
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
        "normal_weight",
        "normal_sigma",
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
    if not isinstance(cfg["contact_binding_calibration"], (bool, np.bool_)):
        raise TypeError("surface_contact.contact_binding_calibration must be boolean")
    return cfg


def calibrate_contact_bindings(model, bindings, sample_indices, active, targets, seed):
    """Resolve a sphere's contact-facing side once from a warm robot motion.

    Learned whole-body correspondence supplies the robot geom identity. On a
    symmetric collision sphere its point can land on the far side, even when
    the source skin touches the near side. The original source contact targets
    and the unchanged warm trajectory choose one persistent shell point per
    active skin sample; they do not move a target or introduce framewise slip.
    """
    poses = np.asarray(seed, dtype=float)
    if poses.shape != (len(active), model.nq) or not np.isfinite(poses).all():
        raise ValueError("Contact binding seed must match the source timeline")
    selected = list(bindings)
    candidates = [
        (i, int(model.geom(binding.robot_geom).id))
        for i, binding in enumerate(selected)
        if (
            active[:, sample_indices[i]].any()
            and model.geom_type[int(model.geom(binding.robot_geom).id)]
            == mujoco.mjtGeom.mjGEOM_SPHERE
        )
    ]
    if not candidates:
        return tuple(selected), {"calibrated_samples": 0, "mean_shell_shift_m": 0.0}
    used_frames = np.flatnonzero(active[:, sample_indices].any(axis=1))
    data = mujoco.MjData(model)
    centers, rotations = {}, {}
    geoms = sorted({geom for _, geom in candidates})
    for frame in used_frames:
        data.qpos[:] = poses[frame]
        mujoco.mj_forward(model, data)
        centers[frame] = {g: data.geom_xpos[g].copy() for g in geoms}
        rotations[frame] = {g: data.geom_xmat[g].reshape(3, 3).copy() for g in geoms}
    displacements = []
    for index, geom in candidates:
        sample = int(sample_indices[index])
        frames = np.flatnonzero(active[:, sample])
        local_targets = np.array(
            [
                rotations[f][geom].T @ (targets[f, sample] - centers[f][geom])
                for f in frames
            ]
        )
        direction = np.mean(local_targets, axis=0)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            continue
        unit = direction / norm
        point = float(model.geom_size[geom, 0]) * unit
        old = np.asarray(selected[index].local_point, dtype=float)
        selected[index] = replace(
            selected[index], local_point=tuple(point), local_normal=tuple(unit)
        )
        displacements.append(float(np.linalg.norm(point - old)))
    return tuple(selected), {
        "calibrated_samples": len(displacements),
        "mean_shell_shift_m": float(np.mean(displacements)) if displacements else 0.0,
    }


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
        contact_targets=None,
        effector_names=None,
    ):
        self.model, self.cfg = model, surface_contact_config(config)
        self.bindings = tuple(bindings)
        self.effector_names = np.array(
            [b.part for b in self.bindings] if effector_names is None else effector_names,
            dtype=str, copy=True,
        )
        if self.effector_names.shape != (len(self.bindings),):
            raise ValueError("Surface effector identities must match the bindings")
        self.effector_names.setflags(write=False)
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
        self.targets = (
            self.environment_points + self.source_vectors
            if contact_targets is None
            else np.array(contact_targets, float, copy=True)
        )
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

    def position_rows(self, pose, frame, dofs, *, project_sliding=True):
        for index in np.flatnonzero(self.active[frame]):
            point, _, jacobian, _ = self.kinematics.at(pose, int(index))
            residual = point - self.targets[frame, index]
            jacobian = jacobian[:, dofs]
            if project_sliding and self.modes[frame, index] in {"sliding", "touch"}:
                normal = self.normals[frame, index]
                projector = np.outer(normal, normal) / (normal @ normal)
                residual, jacobian = projector @ residual, projector @ jacobian
            yield int(index), residual, jacobian

    def hold_position_rows(self, pose, frame, dofs):
        """Attained positions at each timestamp, including authored sliding."""
        yield from self.position_rows(pose, frame, dofs, project_sliding=False)

    def rows(self, pose, frame, dofs, *, include_normals=True):
        for index, residual, jacobian in self.position_rows(pose, frame, dofs, project_sliding=False):
            yield (
                self.weights[frame, index],
                residual / self.cfg["sigma"],
                jacobian / self.cfg["sigma"],
            )
        normal_rows = self.normal_rows(pose, frame, dofs) if include_normals else ()
        for weight, residual, jacobian in normal_rows:
            yield (
                self.cfg["normal_weight"] * weight,
                residual / self.cfg["normal_sigma"],
                jacobian / self.cfg["normal_sigma"],
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

    def metrics(self, poses, frame_start=0, *, errors=None):
        if errors is None:
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
            "surface_contact_effector_names": self.effector_names,
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
        if getattr(self, "contact_phase", None) is not None:
            payload.update(self.contact_phase.export())
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
    surface_motion=None,
    robot_seed_trajectory=None,
    support_evidence=None,
    source_contact_table=None,
    scene_enclosed_free_space=False,
):
    cfg = surface_contact_config(config)
    report = {
        "status": "unresolved",
        "enabled": True,
        "reason": "missing_whole_body_skin_samples",
        "configuration": cfg,
    }
    if not cfg["enabled"]:
        return None, {**report, "status": "disabled", "enabled": False,
                      "reason": "configured_off"}
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
    _, valid = _face_frames(triangles)
    points = np.einsum("tnij,ni->tnj", triangles, bary)
    if source_contact_table is None:
        from .source_contacts import build_source_contacts

        source_contact_table = build_source_contacts(
            full,
            scene_mesh,
            cfg,
            framerate=framerate,
            scene_objects=scene_objects,
            scene_enclosed_free_space=scene_enclosed_free_space,
            min_duration_s=cfg["min_contact_frames"] / framerate,
        )
    if source_contact_table.points.shape != points.shape or not np.allclose(
        source_contact_table.points[valid], points[valid], atol=1e-9, rtol=0
    ):
        raise ValueError(
            "Source contact table must address the unchanged skin observations"
        )
    observation = {
        k: v.copy() if isinstance(v, np.ndarray) else v
        for k, v in source_contact_table.observation.items()
    }
    observation["active"] = source_contact_table.active.copy()
    if "accepted_modes" in observation:
        observation["modes"] = observation["accepted_modes"]
        observation["confidence"] = observation["accepted_confidence"]
    report["source_contacts"] = source_contact_table.diagnostics
    contact_targets = None
    if support_evidence is not None:
        from .support_observation import apply_support_observations

        observation, contact_targets = apply_support_observations(
            observation, points, valid, support_evidence
        )
        report["support_inference"] = observation["support_inference"]
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
    if surface_motion is None:
        from .surface_motion import SurfaceMotion

        surface_motion = SurfaceMotion(
            model,
            source,
            robot_reference_qpos,
            cfg.get("correspondence"),
            mapping=joint_mapping,
        )
    bindings, binding_valid, correspondence = surface_motion.binding_result
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
    if cfg["contact_binding_calibration"] and robot_seed_trajectory is not None:
        selected, calibration = calibrate_contact_bindings(
            model, selected, retained, active, points, robot_seed_trajectory
        )
        report["contact_binding_calibration"] = calibration
    pose_normals, pose_valid = transported_contact_normals(
        model, source, selected, retained, robot_reference_qpos
    )
    from .contact_anchors import SOURCE_ORIENTATION_TOLERANCE_RAD, build_rigid_anchors

    anchor_targets, anchor_report = build_rigid_anchors(
        model,
        selected,
        retained,
        observation,
        surface_motion,
        scene_present=scene_mesh is not None and len(scene_mesh.faces) > 0,
        scene_objects=scene_objects,
        rotation_budget_rad=SOURCE_ORIENTATION_TOLERANCE_RAD,
    )
    contact_targets = points.copy()
    contact_targets[:, retained] = np.where(
        active[:, retained, None], anchor_targets, points[:, retained]
    )
    report["anchors"] = anchor_report
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
    task = SurfaceContactTasks(
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
        contact_targets=(
            None if contact_targets is None else contact_targets[:, retained]
        ),
    )
    task.static_surface_id = (
        0 if scene_mesh is not None and len(scene_mesh.faces) else None
    )
    task.source_contact_table = source_contact_table
    labels = np.asarray(full.get("sample_effectors", [""] * len(bary)))
    if labels.shape != (len(bary),):
        raise ValueError("Sample effector labels must follow surface identities")
    task.effector_names = labels[retained].copy()
    task.effector_names.setflags(write=False)
    outside = observation["clear_side"]
    task.source_clearances = {}
    for name in np.unique(labels):
        if not name:
            continue
        group = labels == name
        # All observed material samples on an effector must be clear. Missing
        # samples are unknown, not evidence that a foot has lifted.
        task.source_clearances[str(name)] = np.where(
            (valid[:, group] & outside[:, group]).all(axis=1),
            observation["distance"][:, group].min(axis=1),
            np.nan,
        )
    return task, report


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
