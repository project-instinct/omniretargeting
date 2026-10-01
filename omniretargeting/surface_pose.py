"""Dense surface position and relative-normal tasks, including free motion.

UMR III-C supplies persistent surface targets and reference-normal transport.
Free positions here use calibrated morphology and a moving robot-root target;
observed free-target regimes and contact anchors retain their own semantics.
"""

import mujoco
import numpy as np

from .surface_motion import SurfaceMotion
from .surface_tasks import SurfaceKinematics

DEFAULTS = {
    "enabled": True,
    "weight": 0.2,
    "position_sigma": 0.05,
    "normal_sigma": 0.35,
    "endpoint_regression": 0.001,
    "morphology_scale": None,
}


def surface_pose_config(config=None):
    from .config_validation import known_options

    known_options(config, DEFAULTS, 'surface_pose', extra=('correspondence',))
    cfg = {**DEFAULTS, **(config or {})}
    for key in ("weight", "position_sigma", "normal_sigma", "endpoint_regression"):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"surface_pose.{key} must be finite and positive")
    if cfg["morphology_scale"] is not None and (
        not np.isfinite(cfg["morphology_scale"]) or cfg["morphology_scale"] <= 0
    ):
        raise ValueError("surface_pose.morphology_scale must be finite and positive")
    return cfg


class SurfacePoseTasks:
    """Every valid learned surface point contributes, even without contacts."""

    protect_primary_endpoints = True

    def __init__(
        self,
        motion,
        root_positions,
        reference_root,
        *,
        config=None,
        scene_tasks=None,
        priority=None,
        surface_contacts=None,
        self_contacts=None,
    ):
        self.model, self.cfg = motion.model, surface_pose_config(config)
        self.bindings = motion.bindings
        self.kinematics = SurfaceKinematics(self.model, self.bindings)
        self.valid = motion.valid
        self.normal_targets = motion.normal_targets
        frames, count = self.valid.shape
        roots = np.array(root_positions, dtype=float, copy=True)
        reference_root = np.asarray(reference_root, dtype=float)
        if (
            roots.shape != (frames, 3)
            or reference_root.shape != (3,)
            or not np.isfinite(roots).all()
            or not np.isfinite(reference_root).all()
        ):
            raise ValueError(
                "Surface pose needs the original finite source root timeline and bind root"
            )
        free = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) > 1:
            raise ValueError("Surface pose supports at most one floating root")
        self.root_body = int(self.model.jnt_bodyid[free[0]]) if len(free) else 0
        self._root_pose, self._root_jacobian = None, None
        self.kinematics.at(motion.reference_qpos, 0)
        root_ref = self.kinematics.data.xpos[self.root_body].copy()
        source_extent = np.ptp(
            motion.reference_points[motion.binding_valid], axis=0
        ).max()
        robot_extent = np.ptp(motion.robot_points[motion.binding_valid], axis=0).max()
        if min(source_extent, robot_extent) <= 1e-8:
            raise ValueError(
                "Surface pose requires non-degenerate corresponding surfaces"
            )
        self.scale = float(self.cfg["morphology_scale"] or robot_extent / source_extent)
        scales = np.full(count, self.scale)
        self.root_relative = np.full((frames, count), bool(len(free)))
        world_tracking = np.zeros((frames, count), bool)
        self.position_active = self.valid.copy()
        self.normal_active = self.valid.copy()
        for name, spec in (priority or {}).get("effectors", {}).items():
            body = spec.get("robot_body")
            if body is None:
                continue
            body_id = self.model.body(body).id
            # Include rigid children such as a palm collision shell.
            group = (
                self.model.body_weldid[motion.body_ids]
                == self.model.body_weldid[body_id]
            )
            task = (scene_tasks or {}).get(name)
            world = np.ones(frames, bool)
            if task is not None and task.swing_world_mask is not None:
                world = task.swing_world_mask
            elif "preserve_world_target" in spec:
                world = np.full(frames, bool(spec["preserve_world_target"]))
            self.root_relative[:, group] &= ~world[:, None]
            world_tracking[:, group] = world[:, None]
            scales[group] = spec.get("swing_morphology_scale", self.scale)
            if name in (scene_tasks or {}):
                mask = np.asarray(scene_tasks[name].contact_mask, bool)
                if mask.shape != (frames,):
                    raise ValueError("Primary contacts must share the surface timeline")
                self.position_active[:, group] &= ~mask[:, None]
        index_by_identity = {
            (b.face_id, b.barycentric): i for i, b in enumerate(self.bindings)
        }
        if surface_contacts is not None:
            for k, binding in enumerate(surface_contacts.bindings):
                i = index_by_identity.get((binding.face_id, binding.barycentric))
                if i is not None:
                    self.position_active[:, i] &= ~surface_contacts.active[:, k]
                    self.normal_active[:, i] &= ~surface_contacts.active[:, k]
        if self_contacts is not None:
            for k in range(len(self_contacts.events)):
                for binding in self_contacts.bindings[2 * k : 2 * k + 2]:
                    i = index_by_identity.get((binding.face_id, binding.barycentric))
                    if i is not None:
                        self.position_active[:, i] &= (
                            self_contacts.confidence[:, k] <= 0
                        )
        relative = motion.frames @ np.swapaxes(motion.reference_frames, -1, -2)
        offset = (
            motion.robot_points
            - root_ref
            - scales[:, None] * (motion.reference_points - reference_root)
        )
        # These are target coordinates, not edits to source motion. The bind
        # offset rotates with the source face, avoiding an unrotated shape bias.
        self.targets = scales[None, :, None] * (
            motion.points - roots[:, None]
        ) + np.einsum("tnij,nj->tni", relative, offset)
        world = ~self.root_relative
        self.targets += np.where(world[..., None], roots[:, None], 0)
        # Unknown clearance retains the immutable world observation.
        self.targets[world_tracking] = motion.points[world_tracking]
        labels = np.array([b.part for b in self.bindings])
        self.weights = np.zeros(count)
        for label in np.unique(labels):
            group = (labels == label) & motion.binding_valid
            self.weights[group] = self.cfg["weight"] / max(int(group.sum()), 1)
        self.diagnostics = {
            "status": "active",
            "reason": None,
            "configuration": dict(self.cfg),
            "samples": count,
            "valid_frame_samples": int(self.valid.sum()),
            "position_frame_samples": int(self.position_active.sum()),
            "morphology_scale": self.scale,
            "correspondence": motion.diagnostics,
            "position_targets": "calibrated_root_relative_with_observed_terminal_regimes",
            "normal_targets": "relative_source_face_rotation_of_robot_reference_normal",
        }
        for value in vars(self).values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)

    def at(self, pose, frame, index):
        point, normal, jac, jnormal = self.kinematics.at(pose, index)
        target = self.targets[frame, index]
        if self.root_relative[frame, index]:
            data = self.kinematics.data
            if self._root_pose is None or not np.array_equal(self._root_pose, pose):
                self._root_pose = np.array(pose, copy=True)
                self._root_jacobian = np.zeros((3, self.model.nv))
                mujoco.mj_jacBody(
                    self.model, data, self._root_jacobian, None, self.root_body
                )
            target = target + data.xpos[self.root_body]
            jac = jac - self._root_jacobian
        return point - target, jac, normal - self.normal_targets[frame, index], jnormal

    def rows(self, pose, frame, dofs):
        """One stacked position block and one normal block, pre-weighted.

        Identical to per-sample rows scaled by sqrt(weight)/sigma; stacking
        avoids thousands of tiny arrays per frame.
        """
        indices = np.flatnonzero(self.valid[frame])
        if not len(indices):
            return
        point, normal, jac, jnormal = self.kinematics.batch(pose, indices)
        target = self.targets[frame, indices]
        relative = self.root_relative[frame, indices]
        if relative.any():
            root = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.kinematics.data, root, None, self.root_body)
            target = target + relative[:, None] * self.kinematics.data.xpos[self.root_body]
            jac = jac - relative[:, None, None] * root
        scale = np.sqrt(self.weights[indices])
        for active, residual, jacobian, sigma in (
            (self.position_active[frame, indices], point - target, jac,
             self.cfg["position_sigma"]),
            (self.normal_active[frame, indices], normal - self.normal_targets[frame, indices],
             jnormal, self.cfg["normal_sigma"]),
        ):
            if active.any():
                factor = scale[active] / sigma
                yield (
                    1.0,
                    (residual[active] * factor[:, None]).ravel(),
                    (jacobian[active][:, :, dofs] * factor[:, None, None]).reshape(
                        -1, len(np.arange(self.model.nv)[dofs])
                    ),
                )

    def metrics(self, poses, frame_start=0):
        positions, angles = [], []
        for f, pose in enumerate(poses, frame_start):
            for i in np.flatnonzero(self.valid[f]):
                residual, _, normal, _ = self.at(pose, f, int(i))
                if self.position_active[f, i]:
                    positions.append(float(np.linalg.norm(residual)))
                if self.normal_active[f, i]:
                    angles.append(
                        float(2 * np.arcsin(np.clip(np.linalg.norm(normal) / 2, 0, 1)))
                    )

        def summary(values):
            return {
                "count": len(values),
                "mean": float(np.mean(values)) if values else None,
                "max": max(values) if values else None,
            }

        return {
            "position_error_m": summary(positions),
            "normal_error_rad": summary(angles),
        }

    def export(self):
        return {
            "surface_pose_targets": self.targets,
            "surface_pose_root_relative": self.root_relative,
            "surface_pose_position_mask": self.position_active,
            "surface_pose_valid": self.valid,
            "surface_pose_normal_mask": self.normal_active,
            "surface_pose_normal_targets_w": self.normal_targets,
            "surface_pose_face_ids": np.array([b.face_id for b in self.bindings]),
            "surface_pose_barycentric": np.array(
                [b.barycentric for b in self.bindings]
            ),
            "surface_pose_robot_geoms": np.array(
                [b.robot_geom for b in self.bindings]
            ),
            "surface_pose_robot_local_points": np.array(
                [b.local_point for b in self.bindings]
            ),
            "surface_pose_robot_local_normals": np.array(
                [b.local_normal for b in self.bindings]
            ),
            "surface_pose_parts": np.array([b.part for b in self.bindings]),
        }


def build_surface_pose_tasks(
    model,
    source,
    reference_qpos,
    root_positions,
    reference_root,
    config=None,
    *,
    mapping=None,
    surface_motion=None,
    **kwargs,
):
    cfg = surface_pose_config(config)
    reason = (
        "configured_off"
        if not cfg["enabled"]
        else "missing_whole_body_skin_samples"
        if source is None or "full_body" not in source
        else "missing_calibrated_robot_reference"
        if reference_qpos is None
        else "missing_source_root_reference"
        if root_positions is None or reference_root is None
        else None
    )
    if reason:
        return None, {"status": "disabled", "reason": reason}
    motion = surface_motion or SurfaceMotion(
        model, source, reference_qpos, cfg.get("correspondence"), mapping
    )
    if not motion.binding_valid.any():
        return None, {"status": "disabled", "reason": "no_valid_correspondences"}
    task = SurfacePoseTasks(
        motion, root_positions, reference_root, config=cfg, **kwargs
    )
    return task, task.diagnostics
