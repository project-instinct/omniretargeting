"""Final trajectory smoothing, safety limits and quality reporting."""

from __future__ import annotations

from typing import Dict, List

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..contact import (
    SceneContactTask,
    _is_surface_only_contact_mode,
    build_world_tracking_tasks,
)


class MotionPostprocessingMixin:
    """Final trajectory smoothing, safety limits and quality reporting.

    Internal methods sharing the owning OmniRetargeter instance; no separate state.
    """

    def _validation_foot_references(self, trajectory, tasks, priority):
        """Conservatively preserve feet when a legacy profile has no source tasks.

        Reuse the existing foot-body and sole-marker resolvers. These references
        describe the incoming *robot* motion and are only used for validation;
        they are never exported as human anchors or assigned contact modes.
        """
        validated = dict(tasks)
        specs = dict(priority.get("effectors", {}) or {})
        fallback = []
        if not getattr(self, "valid_joint_mapping", None):
            return validated, dict(priority), fallback
        body_ids = self._resolve_foot_body_ids(self._get_foot_stabilization_config())
        data = mujoco.MjData(self.robot_model)
        for side, body_id in body_ids.items():
            name = f"{side}_foot"
            if name in tasks or body_id < 0:
                continue
            marker = self._infer_foot_marker_offset(body_id)
            if marker is None:
                continue
            offset = np.asarray(marker[0], dtype=float)
            positions = []
            for q in trajectory:
                data.qpos[:] = q
                mujoco.mj_forward(self.robot_model, data)
                positions.append(data.xpos[body_id] + data.xmat[body_id].reshape(3, 3) @ offset)
            validated.update(build_world_tracking_tasks({name: {"tracking_positions": np.asarray(positions)}}))
            specs[name] = dict(robot_body=self.robot_model.body(body_id).name,
                               tracking_offset=offset.tolist(), preserve_world_target=True)
            fallback.append(name)
        return validated, dict(priority, effectors=specs), fallback


    def _clamp_final_joint_safety_limits(
        self, trajectory: np.ndarray
    ) -> np.ndarray:
        """Guarantee that the saved motion respects task-safe joint ranges."""
        # Keep the public orchestration method usable with lightweight wrappers
        # and tests that supply an already-retargeted trajectory but no MuJoCo
        # model.  A real OmniRetargeter always owns ``robot_model``; without it
        # there are no mechanical limits that can be resolved safely.
        if len(trajectory) == 0 or not hasattr(self, "robot_model"):
            return trajectory
        from .._solver.targets import (
            _apply_configured_joint_limits,
            _apply_joint_limit_safety_fraction,
        )

        qpos_indices = []
        mechanical_lower = []
        mechanical_upper = []
        for joint_id in range(self.robot_model.njnt):
            if not bool(self.robot_model.jnt_limited[joint_id]):
                continue
            if self.robot_model.jnt_type[joint_id] not in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                continue
            qpos_indices.append(int(self.robot_model.jnt_qposadr[joint_id]))
            mechanical_lower.append(float(self.robot_model.jnt_range[joint_id, 0]))
            mechanical_upper.append(float(self.robot_model.jnt_range[joint_id, 1]))
        if not qpos_indices:
            return trajectory

        qpos_indices_array = np.asarray(qpos_indices, dtype=int)
        lower, upper = _apply_joint_limit_safety_fraction(
            self.robot_model,
            qpos_indices_array,
            np.asarray(mechanical_lower),
            np.asarray(mechanical_upper),
            float(
                self.retargeting_config.get(
                    "joint_limit_safety_fraction", 1.0
                )
            ),
        )
        lower, upper = _apply_configured_joint_limits(
            self.robot_model,
            qpos_indices_array,
            lower,
            upper,
            dict(
                self.retargeting_config.get("end_effector_priority", {}) or {}
            ).get("joint_posture_limits"),
        )
        result = np.asarray(trajectory, dtype=float).copy()
        before = result[:, qpos_indices_array].copy()
        result[:, qpos_indices_array] = np.clip(
            before, lower[None, :], upper[None, :]
        )
        max_correction = float(
            np.max(np.abs(result[:, qpos_indices_array] - before))
        )
        if max_correction > 1e-10:
            print(
                "Applied final joint safety clamp: "
                f"max correction={max_correction:.4f} rad"
            )
        return result


    def _apply_trajectory_smoothing(
        self,
        trajectory: np.ndarray,
        framerate: float,
    ) -> np.ndarray:
        """Remove zero-mean framewise IK chatter without moving the root.

        A symmetric Gaussian convolution is zero phase: it suppresses the
        high-frequency redundant-joint solution switching without introducing
        a time delay between the human and robot.  Its positive kernel also
        avoids ringing around contact transitions.  Only scalar robot joints
        are filtered.  Floating-base translation and orientation remain
        exactly as solved, so scene alignment and base position cannot drift.
        """
        cfg = dict(
            getattr(self, "retargeting_config", {}).get(
                "trajectory_smoothing", {}
            )
            or {}
        )
        if not bool(cfg.get("enabled", False)) or len(trajectory) < 8:
            return trajectory
        sigma_frames = float(cfg.get("sigma_frames", 1.0))
        if sigma_frames <= 0.0:
            raise ValueError(
                "trajectory_smoothing.sigma_frames must be positive, got "
                f"{sigma_frames:g}."
            )
        joint_qpos_indices = []
        for joint_id in range(self.robot_model.njnt):
            joint_type = self.robot_model.jnt_type[joint_id]
            if joint_type in (
                mujoco.mjtJoint.mjJNT_HINGE,
                mujoco.mjtJoint.mjJNT_SLIDE,
            ):
                qpos_index = int(self.robot_model.jnt_qposadr[joint_id])
                if qpos_index >= 7:
                    joint_qpos_indices.append(qpos_index)
        if not joint_qpos_indices:
            return trajectory

        result = np.asarray(trajectory, dtype=float).copy()
        result[:, joint_qpos_indices] = gaussian_filter1d(
            result[:, joint_qpos_indices],
            sigma=sigma_frames,
            axis=0,
            mode="nearest",
        )

        # The zero-phase filter can overshoot a saturated joint by a fraction
        # of a degree.  Restore the model's exact scalar joint limits.
        for joint_id in range(self.robot_model.njnt):
            if not self.robot_model.jnt_limited[joint_id]:
                continue
            qpos_index = int(self.robot_model.jnt_qposadr[joint_id])
            if qpos_index not in joint_qpos_indices:
                continue
            lower, upper = self.robot_model.jnt_range[joint_id]
            result[:, qpos_index] = np.clip(
                result[:, qpos_index], lower, upper
            )
        return result


    def _report_final_motion_quality(
        self,
        trajectory: np.ndarray,
        tasks: Dict[str, SceneContactTask],
    ) -> None:
        """Print acceptance metrics for the trajectory that is delivered.

        End-effector world-position errors, contact residuals, and temporal
        jitter are reported separately and after every post-processing stage,
        so the numbers describe the motion actually written to disk (scene
        penetration is reported by the final validator against this exact
        trajectory).
        """
        end_effector_cfg = dict(
            self.retargeting_config.get("end_effector_priority", {}) or {}
        )
        specs = dict(end_effector_cfg.get("effectors", {}) or {})
        if not tasks or not specs or len(trajectory) == 0:
            return
        data = mujoco.MjData(self.robot_model)
        choices = getattr(self, "last_collision_diagnostics", {}).get("local_repair", {}).get("contact_offset_choice", {})
        sticking_errors: Dict[str, List[float]] = {name: [] for name in tasks}
        sliding_normal_errors: Dict[str, List[float]] = {
            name: [] for name in tasks
        }
        sliding_tangent_errors: Dict[str, List[float]] = {
            name: [] for name in tasks
        }
        transition_normal_errors: Dict[str, List[float]] = {
            name: [] for name in tasks
        }
        swing_errors: Dict[str, List[float]] = {name: [] for name in tasks}
        for frame_idx, qpos in enumerate(np.asarray(trajectory, dtype=float)):
            data.qpos[:] = qpos
            mujoco.mj_forward(self.robot_model, data)
            for name, task in tasks.items():
                spec = dict(specs.get(name) or {})
                body_name = spec.get("robot_body")
                if not body_name:
                    continue
                body_id = mujoco.mj_name2id(
                    self.robot_model, mujoco.mjtObj.mjOBJ_BODY, str(body_name)
                )
                if body_id < 0:
                    continue
                rotation = data.xmat[body_id].reshape(3, 3)
                tracking_offset = np.asarray(
                    spec.get("tracking_offset", [0.0, 0.0, 0.0]), dtype=float
                ).reshape(3)
                if bool(task.contact_mask[frame_idx]):
                    contact_offsets = np.asarray(
                        spec.get("contact_offsets", [tracking_offset.tolist()]),
                        dtype=float,
                    )
                    if contact_offsets.ndim == 1:
                        contact_offsets = contact_offsets[None, :]
                    candidates = (
                        data.xpos[body_id][None, :] + contact_offsets @ rotation.T
                    )
                    if name in choices:
                        chosen = int(choices[name][frame_idx])
                        candidates = candidates[chosen:chosen+1]
                    residuals = candidates - task.frame(frame_idx)["contact_position"][None, :]
                    contact_mode = (
                        str(task.contact_modes[frame_idx])
                        if task.contact_modes is not None
                        else "sticking"
                    )
                    if _is_surface_only_contact_mode(contact_mode):
                        normal = np.asarray(
                            task.surface_normals[frame_idx].copy(), dtype=float
                        )
                        normal /= max(float(np.linalg.norm(normal)), 1e-10)
                        signed_normal_errors = residuals @ normal
                        marker_idx = int(
                            np.argmin(np.abs(signed_normal_errors))
                        )
                        normal_error = signed_normal_errors[marker_idx]
                        tangent_error = (
                            residuals[marker_idx] - normal_error * normal
                        )
                        if contact_mode == "sliding":
                            sliding_normal_errors[name].append(
                                float(abs(normal_error))
                            )
                            sliding_tangent_errors[name].append(
                                float(np.linalg.norm(tangent_error))
                            )
                        else:
                            transition_normal_errors[name].append(
                                float(abs(normal_error))
                            )
                    else:
                        sticking_errors[name].append(
                            float(np.min(np.linalg.norm(residuals, axis=1)))
                        )
                else:
                    marker = data.xpos[body_id] + rotation @ tracking_offset
                    swing_errors[name].append(
                        float(
                            np.linalg.norm(
                                marker - task.tracking_positions[frame_idx]
                            )
                        )
                    )
        print("Final motion quality (after all post-processing):")
        for name in tasks:
            parts = []
            for label, errors in (
                ("sticking_xyz", sticking_errors[name]),
                ("sliding_normal", sliding_normal_errors[name]),
                ("sliding_tangent", sliding_tangent_errors[name]),
                ("transition_normal", transition_normal_errors[name]),
                ("swing", swing_errors[name]),
            ):
                if errors:
                    values = np.asarray(errors) * 100.0
                    parts.append(
                        f"{label}[{len(errors)}f] mean={values.mean():.2f} "
                        f"p95={np.percentile(values, 95):.2f} "
                        f"max={values.max():.2f} cm"
                    )
                else:
                    parts.append(f"{label}[0f]")
            print(f"  {name}: " + " | ".join(parts))
        actuated = np.asarray(trajectory, dtype=float)[:, 7:]
        if len(actuated) >= 4 and actuated.shape[1] > 0:
            acceleration = np.diff(actuated, n=2, axis=0)
            jerk = np.diff(actuated, n=3, axis=0)
            base_acceleration = np.diff(
                np.asarray(trajectory, dtype=float)[:, :3], n=2, axis=0
            )
            print(
                "  temporal jitter: joint |acc| "
                f"rms={np.sqrt(np.mean(acceleration**2)):.5f} "
                f"max={np.max(np.abs(acceleration)):.4f} rad/frame^2, "
                f"|jerk| rms={np.sqrt(np.mean(jerk**2)):.5f}, "
                f"base |acc| rms={np.sqrt(np.mean(base_acceleration**2)):.5f} m/frame^2"
            )
