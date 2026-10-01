"""Persistent body-surface contact maps, complementary to geometric avoidance.

UMR supplies the idea of contact vectors between corresponding surface points.
Whole-body inputs reuse learned correspondence. Legacy part-only adapters keep
their anatomical projection. Identities remain fixed during each run, and both
moving robot surfaces contribute Jacobians.
"""

from dataclasses import asdict

import mujoco
import numpy as np
from scipy.spatial import cKDTree

from .contact_schedule import _iter_true_runs, _remove_short_runs
from .surface_correspondence import bind_sampled_parts
from .surface_tasks import SurfaceKinematics

DEFAULTS = {
    "enabled": True,
    "distance": 0.02,
    "min_contact_frames": 3,
    "sigma": 0.01,
    "normal_sigma": 0.35,
    "weight": 1.0,
    "minimum_gap": 0.002,
    "regression": 0.002,
    "tolerance": 0.01,
    "minimum_geodesic_distance": 0.12,
}


def self_contact_config(config=None):
    from .config_validation import known_options

    known_options(config, DEFAULTS, 'self_contact', extra=('correspondence',))
    result = {**DEFAULTS, **(config or {})}
    for key in (
        "distance",
        "sigma",
        "normal_sigma",
        "weight",
        "regression",
        "tolerance",
        "minimum_geodesic_distance",
    ):
        if not np.isfinite(result[key]) or result[key] <= 0:
            raise ValueError(f"self_contact.{key} must be finite and positive")
    if not 0 <= result["minimum_gap"] <= result["distance"]:
        raise ValueError("self_contact.minimum_gap must be within contact distance")
    if (
        int(result["min_contact_frames"]) != result["min_contact_frames"]
        or result["min_contact_frames"] < 1
    ):
        raise ValueError("self_contact.min_contact_frames must be a positive integer")
    return result


def _skew(vector):
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


class SelfContactTasks:
    def __init__(
        self,
        model,
        bindings,
        targets,
        confidence,
        events,
        config,
        diagnostics,
        normal_targets,
    ):
        self.model, self.cfg = model, self_contact_config(config)
        self.bindings, self.events = tuple(bindings), tuple(events)
        self.kinematics = SurfaceKinematics(model, bindings)
        self.targets = np.array(targets, dtype=float, copy=True)
        self.confidence = np.array(confidence, dtype=float, copy=True)
        self.normal_targets = np.array(normal_targets, dtype=float, copy=True)
        if (
            self.targets.ndim != 3
            or self.targets.shape[1:] != (len(events), 3)
            or self.confidence.shape != self.targets.shape[:2]
            or self.normal_targets.shape != self.targets.shape
            or not np.isfinite(self.normal_targets).all()
            or len(bindings) != 2 * len(events)
            or not np.isfinite(self.targets).all()
            or not np.isfinite(self.confidence).all()
            or np.any(self.confidence < 0)
        ):
            raise ValueError(
                "Self-contact bindings/targets must share a finite timeline"
            )
        self.targets.setflags(write=False)
        self.confidence.setflags(write=False)
        self.normal_targets.setflags(write=False)
        self.diagnostics = diagnostics
        self.protect_primary_endpoints = diagnostics.get("discovery") == "non_adjacent_source_segments_and_robot_welds"

    def at(self, pose, index):
        pa, na, ja, jna = self.kinematics.at(pose, 2 * index)
        pb, _nb, jb, _jnb = self.kinematics.at(pose, 2 * index + 1)
        binding = self.bindings[2 * index + 1]
        gid = self.model.geom(binding.robot_geom).id
        data = self.kinematics.data
        rotation = data.geom_xmat[gid].reshape(3, 3)
        jr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, data, None, jr, int(self.model.geom_bodyid[gid]))
        delta = pa - pb
        # Express the pair in the receiving geom frame: common root motion
        # cancels, while bending the thigh moves its contact anchor correctly.
        return (
            rotation.T @ delta,
            rotation.T @ (ja - jb + _skew(delta) @ jr),
            rotation.T @ na,
            rotation.T @ (jna + _skew(na) @ jr),
        )

    def rows(self, pose, frame, dofs):
        for index in np.flatnonzero(self.confidence[frame] > 0):
            vector, jacobian, normal, jnormal = self.at(pose, index)
            weight = self.cfg["weight"] * self.confidence[frame, index]
            yield (
                weight,
                (vector - self.targets[frame, index]) / self.cfg["sigma"],
                jacobian[:, dofs] / self.cfg["sigma"],
            )
            yield (
                weight,
                (normal - self.normal_targets[frame, index]) / self.cfg["normal_sigma"],
                jnormal[:, dofs] / self.cfg["normal_sigma"],
            )

    def position_rows(self, pose, frame, dofs):
        for index in np.flatnonzero(self.confidence[frame] > 0):
            vector, jacobian, _, _ = self.at(pose, index)
            yield index, vector - self.targets[frame, index], jacobian[:, dofs]

    def errors(self, poses, frame_start=0):
        values = np.full((len(poses), len(self.events)), np.nan)
        for f, pose in enumerate(poses):
            for index, residual, _ in self.position_rows(pose, frame_start + f, []):
                values[f, index] = np.linalg.norm(residual)
        return values

    def metrics(self, poses, frame_start=0, *, errors=None):
        if errors is None:
            errors = self.errors(poses, frame_start)
        groups = {}
        for k, event in enumerate(self.events):
            values = errors[:, k]
            values = values[np.isfinite(values)]
            a = event.get("part_a", event.get("hand"))
            b = event.get("part_b", event.get("body"))
            groups[f"{a}->{b}:{event['start']}"] = {
                "count": len(values),
                "mean_m": float(values.mean()) if len(values) else None,
                "max_m": float(values.max()) if len(values) else None,
            }
        values = errors[np.isfinite(errors)]
        return {
            "active_frame_pairs": len(values),
            "pair_vector_error_mean_m": float(values.mean()) if len(values) else None,
            "pair_vector_error_max_m": float(values.max()) if len(values) else None,
            "groups": groups,
            "penetration_is_measured_separately": True,
        }

    def export(self):
        return {
            "self_contact_targets_local": self.targets,
            "self_contact_normal_targets_local": self.normal_targets,
            "self_contact_confidence": self.confidence,
            "self_contact_face_ids": np.array([b.face_id for b in self.bindings]),
            "self_contact_barycentric": np.array([b.barycentric for b in self.bindings]),
            "self_contact_robot_geoms": np.array([b.robot_geom for b in self.bindings]),
            "self_contact_robot_local_points": np.array([b.local_point for b in self.bindings]),
            "self_contact_robot_local_normals": np.array([b.local_normal for b in self.bindings]),
            "self_contact_parts": np.array([b.part for b in self.bindings]),
        }


def _nearest_pairs(hand, body):
    count = len(hand.points)
    distance = np.full(count, np.inf)
    indices = np.zeros((count, 2), dtype=int)
    for f in range(count):
        hi, bi = np.flatnonzero(hand.valid[f]), np.flatnonzero(body.valid[f])
        if not len(hi) or not len(bi):
            continue
        ds, js = cKDTree(body.points[f, bi]).query(
            hand.points[f, hi], k=min(8, len(bi))
        )
        ds, js = (
            np.asarray(ds).reshape(len(hi), -1),
            np.asarray(js).reshape(len(hi), -1),
        )
        opposite = (
            np.einsum("ik,ijk->ij", hand.normals[f, hi], body.normals[f, bi[js]]) < -0.5
        )
        ds = np.where(opposite, ds, np.inf)
        h, neighbour = np.unravel_index(np.argmin(ds), ds.shape)
        distance[f], indices[f] = ds[h, neighbour], (hi[h], bi[js[h, neighbour]])
    return distance, indices


def build_self_contact_tasks(
    model, source, mapping, robot_reference_qpos, scene_tasks=None, config=None,
    *, surface_motion=None,
):
    cfg = self_contact_config(config)
    report = {"status": "disabled", "reason": "missing_skin_samples", "events": []}
    if not cfg["enabled"]:
        report["reason"] = "configured_off"
        return None, report
    if source is None or robot_reference_qpos is None:
        return None, report
    if "full_body" in source:
        from .full_body_self_contact import build_full_body_self_contact
        # Dense observations keep brief touches; the historical part-only
        # fallback retains its three-frame noise rejection default.
        cfg["min_contact_frames"] = (config or {}).get("min_contact_frames", 1)
        return build_full_body_self_contact(
            model, source, mapping, robot_reference_qpos, scene_tasks, cfg,
            surface_motion=surface_motion,
        )
    parts, reports = bind_sampled_parts(model, source, mapping, robot_reference_qpos)
    report.update(
        parts=reports,
        correspondence="anatomical_rest_region_projection",
        geometry_avoidance="existing_MuJoCo_collision_constraints",
    )
    hands = [p for p in parts.values() if p.effector]
    bodies = [p for p in parts.values() if not p.effector]
    bindings, targets, confidences, events, normal_targets = [], [], [], [], []
    for hand in hands:
        if not bodies:
            continue
        frames = len(hand.points)
        if any(len(b.points) != frames for b in bodies):
            raise ValueError("Body surfaces must share the source timeline")
        nearest = [_nearest_pairs(hand, body) for body in bodies]
        distances = np.array([x[0] for x in nearest])
        choice = np.argmin(distances, axis=0)
        scene_mask = np.zeros(frames, bool)
        if hand.effector in (scene_tasks or {}):
            scene_mask = np.asarray(scene_tasks[hand.effector].contact_mask, bool)
            if scene_mask.shape != (frames,):
                raise ValueError("Scene contacts and body surfaces must share timeline")
        for b, body in enumerate(bodies):
            active = _remove_short_runs(
                (choice == b) & (distances[b] <= cfg["distance"]) & ~scene_mask,
                int(cfg["min_contact_frames"]),
            )
            for start, stop in _iter_true_runs(active):
                # One correspondence per hand/run avoids multiplying its
                # objective weight when a larger surface receives more samples.
                middle = start + (stop - start) // 2
                h, j = nearest[b][1][middle]
                valid = hand.valid[:, h] & body.valid[:, j]
                delta = hand.points[:, h] - body.points[:, j]
                local = np.einsum("tji,tj->ti", body.frames[:, j], delta)
                # Sparse points can differ tangentially even at physical
                # contact. Preserve that vector; never attract through skin.
                local[:, 2] = np.maximum(local[:, 2], cfg["minimum_gap"])
                target = local @ body.robot_frames[j].T
                target[~valid] = 0
                confidence = np.zeros(frames)
                confidence[start:stop] = valid[start:stop]
                if not np.any(confidence):
                    continue
                # Ease task strength at source contact boundaries, without
                # extending the inferred run or changing the source timing.
                ramp = np.minimum(
                    np.arange(stop - start) + 1, np.arange(stop - start, 0, -1)
                )
                confidence[start:stop] *= np.minimum(ramp / 3.0, 1.0)
                bindings.extend([hand.bindings[h], body.bindings[j]])
                targets.append(target)
                relative_normal = np.einsum(
                    "tji,tj->ti", body.frames[:, j], hand.normals[:, h]
                )
                normal_target = relative_normal @ body.robot_frames[j].T
                normal_target[~valid] = 0
                normal_targets.append(normal_target)
                confidences.append(confidence)
                events.append(
                    {
                        "hand": hand.name,
                        "body": body.name,
                        "start": int(start),
                        "stop": int(stop),
                        "hand_binding": asdict(hand.bindings[h]),
                        "body_binding": asdict(body.bindings[j]),
                        "hand_sample": int(h),
                        "body_sample": int(j),
                        "source_nearest_gap_mean_m": float(
                            distances[b, start:stop].mean()
                        ),
                    }
                )
    report["events"] = events
    if not events:
        report["reason"] = "no_supported_source_self_contact"
        return None, report
    report.update(
        status="active",
        reason=None,
        event_count=len(events),
        active_frame_pairs=int(np.count_nonzero(confidences)),
    )
    return SelfContactTasks(
        model,
        bindings,
        np.stack(targets, axis=1),
        np.stack(confidences, axis=1),
        events,
        cfg,
        report,
        np.stack(normal_targets, axis=1),
    ), report
