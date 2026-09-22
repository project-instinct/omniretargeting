"""Foot contact stabilization and local surface corrections."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import mujoco
import numpy as np
import trimesh

from ..utils import compute_mesh_height_at_point


class FootStabilizationMixin:
    """Foot contact stabilization and local surface corrections.

    Internal methods sharing the owning OmniRetargeter instance; no separate state.
    """

    def _get_foot_stabilization_config(self) -> Dict[str, Any]:
        """Return merged foot stabilization settings."""
        defaults = {
            "enabled": False,
            "clearance": 0.01,
            "surface_clearance": 0.005,
            "contact_clearance": 0.04,
            "contact_vertical_speed": 0.18,
            "min_contact_frames": 3,
            "anchor_frames": 3,
            "xy_correction_gain": 1.0,
            "xy_smoothing_window": 5,
            "z_smoothing_window": 5,
            "contact_point_height_band": 0.01,
            "max_xy_correction": 0.08,
            "max_surface_correction": 0.08,
            "surface_iterations": 4,
            "wall_red_axis_only": True,
            "wall_normal_z_threshold": 0.35,
            "wall_x_dominance_threshold": 0.5,
            "body_names": {},
            "ground_lock": True,
            "ground_lock_clearance": 0.0,
            "ground_lock_body_tokens": ["ankle", "foot"],
            "ground_lock_geom_groups": [2, 3],
            "preserve_root_z": False,
        }
        cfg = dict(defaults)
        cfg.update(self.retargeting_config.get("foot_stabilization", {}) or {})
        if bool(self.retargeting_config.get("preserve_root_z", False)):
            cfg["preserve_root_z"] = True
        return cfg


    def _apply_foot_stabilization(
        self,
        retargeted_motion: np.ndarray,
        terrain_mesh: trimesh.Trimesh,
        framerate: float | None = None,
    ) -> np.ndarray:
        """Post-process the retargeted motion to reduce foot penetration and stance slip."""
        cfg = self._get_foot_stabilization_config()
        if not cfg.get("enabled", False):
            return retargeted_motion
        if retargeted_motion.size == 0:
            return retargeted_motion

        foot_specs = self._build_foot_stabilization_specs(cfg)
        if not foot_specs:
            print("Foot stabilization skipped: no foot bodies could be resolved.")
            return retargeted_motion

        ground_lock_specs = []
        if bool(cfg.get("ground_lock", True)):
            ground_lock_specs = self._build_foot_ground_lock_specs(cfg)
            if not ground_lock_specs:
                print("Foot ground lock skipped: no foot geoms could be resolved.")

        stabilized = np.array(retargeted_motion, copy=True)
        preserve_root_z = bool(cfg.get("preserve_root_z", False))
        original_root_z = stabilized[:, 2].copy()
        stabilized, wall_contact_mask = self._apply_surface_collision_corrections(
            stabilized,
            terrain_mesh,
            foot_specs,
            cfg,
        )
        if preserve_root_z:
            stabilized[:, 2] = original_root_z

        contact_band = float(cfg["contact_point_height_band"])
        clearance_target = float(cfg.get("ground_lock_clearance", cfg["clearance"])) if ground_lock_specs else float(cfg["clearance"])
        z_window = int(cfg["z_smoothing_window"])
        xy_window = int(cfg["xy_smoothing_window"])
        min_contact_frames = int(cfg["min_contact_frames"])
        anchor_frames = max(int(cfg["anchor_frames"]), 1)
        xy_gain = float(cfg["xy_correction_gain"])
        max_xy_correction = float(cfg["max_xy_correction"])
        vertical_speed_threshold = float(cfg["contact_vertical_speed"])
        contact_clearance = float(cfg["contact_clearance"])

        if ground_lock_specs:
            positions, min_z = self._compute_foot_ground_lock_series(stabilized, ground_lock_specs, contact_band)
            wall_contact_mask = np.zeros_like(min_z, dtype=bool)
        else:
            positions, min_z = self._compute_foot_contact_series(stabilized, foot_specs, contact_band)
        terrain_heights = self._compute_terrain_heights(terrain_mesh, positions[:, :, :2])
        clearances = min_z - terrain_heights

        lift_clearances = np.where(~wall_contact_mask, clearances, np.inf)
        min_clearance = np.min(lift_clearances, axis=1)
        base_lift = np.maximum(clearance_target - min_clearance, 0.0)
        base_lift[~np.isfinite(min_clearance)] = 0.0
        base_lift_smoothed = self._smooth_signal(base_lift, z_window)
        base_lift = np.maximum(base_lift, base_lift_smoothed)
        if preserve_root_z:
            base_lift[:] = 0.0
        else:
            stabilized[:, 2] += base_lift

        if ground_lock_specs:
            positions, min_z = self._compute_foot_ground_lock_series(stabilized, ground_lock_specs, contact_band)
        else:
            positions, min_z = self._compute_foot_contact_series(stabilized, foot_specs, contact_band)
        terrain_heights = self._compute_terrain_heights(terrain_mesh, positions[:, :, :2])
        clearances = min_z - terrain_heights

        dt = 1.0 / framerate if framerate and framerate > 0 else 1.0
        z_vel = np.zeros_like(min_z)
        if len(stabilized) > 1:
            z_vel[1:] = np.abs(np.diff(min_z, axis=0)) / dt

        contact_mask = (clearances <= contact_clearance) & (z_vel <= vertical_speed_threshold) & (~wall_contact_mask)
        for foot_idx in range(contact_mask.shape[1]):
            contact_mask[:, foot_idx] = self._filter_short_contact_runs(
                contact_mask[:, foot_idx],
                min_contact_frames=min_contact_frames,
            )

        corrections = np.zeros((len(stabilized), 2), dtype=float)
        weights = np.zeros(len(stabilized), dtype=float)

        for foot_idx in range(contact_mask.shape[1]):
            for start, end in self._iter_true_runs(contact_mask[:, foot_idx]):
                anchor_end = min(start + anchor_frames, end)
                anchor_xy = np.median(positions[start:anchor_end, foot_idx, :2], axis=0)
                drift = positions[start:end, foot_idx, :2] - anchor_xy
                corrections[start:end] += -xy_gain * drift
                weights[start:end] += 1.0

        active = weights > 0
        if np.any(active):
            corrections[active] /= weights[active, None]
            corrections = self._smooth_signal(corrections, xy_window)
            norms = np.linalg.norm(corrections, axis=1)
            oversized = norms > max_xy_correction
            if np.any(oversized):
                corrections[oversized] *= (max_xy_correction / norms[oversized])[:, None]
            stabilized[:, :2] += corrections

            if ground_lock_specs:
                positions, min_z = self._compute_foot_ground_lock_series(stabilized, ground_lock_specs, contact_band)
            else:
                positions, min_z = self._compute_foot_contact_series(stabilized, foot_specs, contact_band)
            terrain_heights = self._compute_terrain_heights(terrain_mesh, positions[:, :, :2])
            clearances = min_z - terrain_heights
            lift_clearances = np.where(~wall_contact_mask, clearances, np.inf)
            min_clearance = np.min(lift_clearances, axis=1)
            residual_lift = np.maximum(clearance_target - min_clearance, 0.0)
            residual_lift[~np.isfinite(min_clearance)] = 0.0
            if not preserve_root_z:
                stabilized[:, 2] += residual_lift

        stabilized, _ = self._apply_surface_collision_corrections(
            stabilized,
            terrain_mesh,
            foot_specs,
            cfg,
        )
        if preserve_root_z:
            stabilized[:, 2] = original_root_z

        if np.any(base_lift > 0) or np.any(active):
            print(
                "Applied foot stabilization: "
                f"max base lift={base_lift.max():.4f}m, "
                f"contact frames={int(np.sum(contact_mask))}"
            )

        return stabilized


    def _build_foot_stabilization_specs(self, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Resolve foot bodies and pre-sample contact candidates in body coordinates."""
        specs = []
        body_ids = self._resolve_foot_body_ids(cfg)
        for side, body_id in body_ids.items():
            if body_id < 0:
                continue
            sample_points = self._collect_body_contact_points(body_id)
            if sample_points.size == 0:
                sample_points = np.zeros((1, 3), dtype=float)
            specs.append({
                "side": side,
                "body_id": body_id,
                "body_name": mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_id),
                "sample_points": sample_points,
                "collision_points": self._select_collision_probe_points(sample_points),
            })
        return specs


    def _build_foot_ground_lock_specs(self, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Collect visible/collision foot geoms used for ground locking."""
        body_tokens = tuple(str(token).lower() for token in cfg.get("ground_lock_body_tokens", ["ankle", "foot"]))
        geom_groups = cfg.get("ground_lock_geom_groups", [2, 3])
        geom_groups = set(int(group) for group in geom_groups) if geom_groups is not None else None

        specs = []
        for side in ("left", "right"):
            geom_specs = []
            for geom_id in range(self.robot_model.ngeom):
                if geom_groups is not None and int(self.robot_model.geom_group[geom_id]) not in geom_groups:
                    continue
                body_id = int(self.robot_model.geom_bodyid[geom_id])
                body_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
                body_name_lower = body_name.lower()
                if not self._body_name_matches_side(body_name_lower, side):
                    continue
                if not any(token in body_name_lower for token in body_tokens):
                    continue

                sample_points = self._sample_geom_points_in_body_frame(geom_id)
                if sample_points.size == 0:
                    continue
                geom_specs.append({
                    "geom_id": geom_id,
                    "body_id": body_id,
                    "body_name": body_name,
                    "sample_points": sample_points,
                })

            if geom_specs:
                specs.append({"side": side, "geom_specs": geom_specs})
        return specs


    def _compute_foot_contact_series(
        self,
        motion: np.ndarray,
        foot_specs: List[Dict[str, Any]],
        contact_band: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-frame foot support positions and minimum support heights."""
        num_frames = len(motion)
        num_feet = len(foot_specs)
        positions = np.zeros((num_frames, num_feet, 3), dtype=float)
        min_z = np.zeros((num_frames, num_feet), dtype=float)

        for frame_idx, qpos in enumerate(motion):
            self.robot_data.qpos[:] = qpos
            mujoco.mj_forward(self.robot_model, self.robot_data)

            for foot_idx, spec in enumerate(foot_specs):
                body_id = spec["body_id"]
                body_pos = np.asarray(self.robot_data.xpos[body_id], dtype=float)
                body_rot = np.asarray(self.robot_data.xmat[body_id], dtype=float).reshape(3, 3)
                world_points = spec["sample_points"] @ body_rot.T + body_pos
                point_z = world_points[:, 2]
                frame_min_z = float(np.min(point_z))
                support_mask = point_z <= frame_min_z + contact_band
                support_points = world_points[support_mask]
                positions[frame_idx, foot_idx] = np.mean(support_points, axis=0)
                min_z[frame_idx, foot_idx] = frame_min_z

        return positions, min_z


    def _compute_foot_ground_lock_series(
        self,
        motion: np.ndarray,
        ground_lock_specs: List[Dict[str, Any]],
        contact_band: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute support points from all configured visible/collision foot geoms."""
        num_frames = len(motion)
        num_feet = len(ground_lock_specs)
        positions = np.zeros((num_frames, num_feet, 3), dtype=float)
        min_z = np.zeros((num_frames, num_feet), dtype=float)

        for frame_idx, qpos in enumerate(motion):
            self.robot_data.qpos[:] = qpos
            mujoco.mj_forward(self.robot_model, self.robot_data)

            for foot_idx, spec in enumerate(ground_lock_specs):
                world_points = []
                for geom_spec in spec["geom_specs"]:
                    body_id = geom_spec["body_id"]
                    body_pos = np.asarray(self.robot_data.xpos[body_id], dtype=float)
                    body_rot = np.asarray(self.robot_data.xmat[body_id], dtype=float).reshape(3, 3)
                    world_points.append(geom_spec["sample_points"] @ body_rot.T + body_pos)
                world_points = np.vstack(world_points)
                point_z = world_points[:, 2]
                frame_min_z = float(np.min(point_z))
                support_mask = point_z <= frame_min_z + contact_band
                support_points = world_points[support_mask]
                positions[frame_idx, foot_idx] = np.mean(support_points, axis=0)
                min_z[frame_idx, foot_idx] = frame_min_z

        return positions, min_z


    def _compute_terrain_heights(self, terrain_mesh: trimesh.Trimesh, xy_points: np.ndarray) -> np.ndarray:
        """Compute terrain heights for a batch of XY positions."""
        heights = np.zeros(xy_points.shape[:2], dtype=float)
        for frame_idx in range(xy_points.shape[0]):
            for foot_idx in range(xy_points.shape[1]):
                x, y = xy_points[frame_idx, foot_idx]
                heights[frame_idx, foot_idx] = compute_mesh_height_at_point(terrain_mesh, float(x), float(y))
        return heights


    def _apply_surface_collision_corrections(
        self,
        motion: np.ndarray,
        terrain_mesh: trimesh.Trimesh,
        foot_specs: List[Dict[str, Any]],
        cfg: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project penetrating foot probe points out of nearby terrain surfaces and flag wall-contact feet."""
        if len(motion) == 0 or not foot_specs:
            return motion, np.zeros((len(motion), len(foot_specs)), dtype=bool)

        triangles = np.asarray(terrain_mesh.triangles, dtype=float)
        if len(triangles) == 0:
            return motion, np.zeros((len(motion), len(foot_specs)), dtype=bool)
        face_normals = np.asarray(terrain_mesh.face_normals, dtype=float)

        surface_clearance = float(cfg["surface_clearance"])
        max_surface_correction = float(cfg["max_surface_correction"])
        surface_iterations = max(int(cfg["surface_iterations"]), 1)
        preserve_root_z = bool(cfg.get("preserve_root_z", False))

        stabilized = np.array(motion, copy=True)
        total_xy_shift = np.zeros(len(stabilized), dtype=float)
        wall_contact_mask = np.zeros((len(stabilized), len(foot_specs)), dtype=bool)

        for frame_idx, qpos in enumerate(stabilized):
            for _ in range(surface_iterations):
                correction_vectors: List[np.ndarray] = []
                wall_x_corrections: List[float] = []
                self.robot_data.qpos[:] = qpos
                mujoco.mj_forward(self.robot_model, self.robot_data)

                for foot_idx, spec in enumerate(foot_specs):
                    collision_points = spec.get("collision_points")
                    if collision_points is None or collision_points.size == 0:
                        continue

                    body_id = spec["body_id"]
                    body_pos = np.asarray(self.robot_data.xpos[body_id], dtype=float)
                    body_rot = np.asarray(self.robot_data.xmat[body_id], dtype=float).reshape(3, 3)
                    world_points = collision_points @ body_rot.T + body_pos

                    for point in world_points:
                        correction, wall_contact = self._compute_surface_point_correction(
                            point,
                            triangles,
                            face_normals,
                            clearance=surface_clearance,
                            cfg=cfg,
                        )
                        wall_contact_mask[frame_idx, foot_idx] |= wall_contact
                        if correction is not None:
                            if wall_contact:
                                wall_x_corrections.append(float(correction[0]))
                            else:
                                correction_vectors.append(correction)

                if not correction_vectors and not wall_x_corrections:
                    break

                correction = np.zeros(3, dtype=float)
                if correction_vectors:
                    correction = np.mean(correction_vectors, axis=0)
                    correction[2] = max(correction[2], 0.0)

                if wall_x_corrections:
                    pos = max((value for value in wall_x_corrections if value > 0.0), default=0.0)
                    neg = min((value for value in wall_x_corrections if value < 0.0), default=0.0)
                    correction[0] = pos if abs(pos) >= abs(neg) else neg

                if preserve_root_z:
                    correction[2] = 0.0

                norm = np.linalg.norm(correction)
                if norm > max_surface_correction and norm > 1e-12:
                    correction *= max_surface_correction / norm

                qpos[:3] += correction
                total_xy_shift[frame_idx] += np.linalg.norm(correction[:2])

            stabilized[frame_idx] = qpos

        if np.any(total_xy_shift > 0):
            print(
                "Applied surface collision correction: "
                f"max xy shift={total_xy_shift.max():.4f}m"
            )

        return stabilized, wall_contact_mask


    def _compute_surface_point_correction(
        self,
        point: np.ndarray,
        triangles: np.ndarray,
        face_normals: np.ndarray,
        clearance: float,
        cfg: Dict[str, Any],
    ) -> Tuple[np.ndarray | None, bool]:
        """Return a correction vector plus whether the point is in wall-contact mode."""
        repeated_points = np.repeat(point[None, :], len(triangles), axis=0)
        closest_points = trimesh.triangles.closest_point(triangles, repeated_points)
        delta = point[None, :] - closest_points
        dist2 = np.einsum("ij,ij->i", delta, delta)
        face_idx = int(np.argmin(dist2))

        closest = closest_points[face_idx]
        normal = face_normals[face_idx]
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-12:
            return None, False
        normal = normal / normal_norm

        signed_offset = float(np.dot(point - closest, normal))
        penetration = clearance - signed_offset
        if penetration <= 0.0:
            return None, False

        wall_contact = False
        if bool(cfg.get("wall_red_axis_only", True)):
            wall_normal_z_threshold = float(cfg.get("wall_normal_z_threshold", 0.35))
            wall_x_dominance_threshold = float(cfg.get("wall_x_dominance_threshold", 0.5))
            if abs(normal[2]) <= wall_normal_z_threshold and abs(normal[0]) >= wall_x_dominance_threshold:
                correction = np.array([penetration * np.sign(normal[0]), 0.0, 0.0], dtype=float)
                wall_contact = True
            else:
                correction = penetration * normal
        else:
            correction = penetration * normal

        if not wall_contact:
            correction[2] = max(correction[2], 0.0)
        if np.linalg.norm(correction) < 1e-9:
            return None, wall_contact
        return correction, wall_contact


    def _smooth_signal(self, values: np.ndarray, window: int) -> np.ndarray:
        """Apply edge-padded moving average smoothing to 1D or 2D arrays."""
        if window <= 1 or values.shape[0] <= 1:
            return np.array(values, copy=True)

        kernel = np.ones(window, dtype=float) / float(window)
        pad_left = window // 2
        pad_right = window - 1 - pad_left

        if values.ndim == 1:
            padded = np.pad(values, (pad_left, pad_right), mode="edge")
            return np.convolve(padded, kernel, mode="valid")

        padded = np.pad(values, ((pad_left, pad_right), (0, 0)), mode="edge")
        smoothed = np.zeros_like(values, dtype=float)
        for col_idx in range(values.shape[1]):
            smoothed[:, col_idx] = np.convolve(padded[:, col_idx], kernel, mode="valid")
        return smoothed


    def _filter_short_contact_runs(self, mask: np.ndarray, min_contact_frames: int) -> np.ndarray:
        """Drop contact runs shorter than the configured minimum."""
        if min_contact_frames <= 1:
            return np.array(mask, copy=True)

        filtered = np.array(mask, copy=True)
        for start, end in self._iter_true_runs(mask):
            if end - start < min_contact_frames:
                filtered[start:end] = False
        return filtered


    def _iter_true_runs(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Return half-open index ranges for all True runs in a boolean mask."""
        runs: List[Tuple[int, int]] = []
        start = None
        for idx, value in enumerate(mask):
            if value and start is None:
                start = idx
            elif not value and start is not None:
                runs.append((start, idx))
                start = None
        if start is not None:
            runs.append((start, len(mask)))
        return runs
