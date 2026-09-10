"""Sparse source-driven skeletal relations, in calibrated reference-body axes.

This is a coarse layout descriptor, not a skin clearance/contact estimator.
Bindings and source confidence are fixed before solving. Rows use MuJoCo's nv
tangent coordinates; callers convert to their actual optimization coordinates.
"""

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class BodyRelationBinding:
    group: str
    point_body: str
    reference_body: str
    source_point: str
    source_reference: str


def _skew(v):
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


class BodyRelationTasks:
    """One metric vector per semantic arm group; no all-pairs graph.

    The chest-to-shoulder reference offset comes from robot morphology, while
    the source's local shoulder/arm displacement is scaled by bind arm length.
    Both reference poses must already share the existing calibrated world.
    No motion/scene/world contact arrays are edited or used as robot priors.
    """

    def __init__(self, model, bindings, targets, confidence, *, weight, sigma, diagnostics):
        self.model = model
        self.bindings = tuple(bindings)
        self.targets = np.array(targets, dtype=float, copy=True)
        self.confidence = np.array(confidence, dtype=float, copy=True)
        self.sigma = float(sigma)
        counts = {b.group: sum(c.group == b.group for c in bindings) for b in bindings}
        self.weights = np.array([weight / counts[b.group] for b in bindings])
        self.body_ids = [(model.body(b.point_body).id, model.body(b.reference_body).id)
                         for b in bindings]
        self.data = mujoco.MjData(model)
        self._qpos, self._cache = None, {}
        self.diagnostics = diagnostics
        for array in (self.targets, self.confidence, self.weights):
            array.setflags(write=False)

    def at(self, pose, index):
        if self._qpos is None or not np.array_equal(self._qpos, pose):
            self._qpos = np.array(pose, copy=True)
            self.data.qpos[:] = pose
            mujoco.mj_forward(self.model, self.data)
            self._cache.clear()
        if index not in self._cache:
            a, b = self.body_ids[index]
            delta = self.data.xpos[a] - self.data.xpos[b]
            rotation = self.data.xmat[b].reshape(3, 3)
            ja, jb, jw = (np.zeros((3, self.model.nv)) for _ in range(3))
            mujoco.mj_jacBody(self.model, self.data, ja, None, a)
            mujoco.mj_jacBody(self.model, self.data, jb, jw, b)
            # d(R_B^T delta) = R_B^T (dp_A - dp_B + [delta]x omega_B).
            self._cache[index] = (rotation.T @ delta, rotation.T @ (ja - jb + _skew(delta) @ jw))
        return self._cache[index]

    def rows(self, pose, frame, dofs):
        """Yield weight, dimensionless residual and Jacobian using ONE formula."""
        for index, weight in enumerate(self.weights):
            effective = float(weight * self.confidence[frame, index])
            if effective <= 0:
                continue  # Zero-weight wiring must not even add solver rows.
            vector, jacobian = self.at(pose, index)
            yield effective, (vector - self.targets[frame, index]) / self.sigma, jacobian[:, dofs] / self.sigma

    def metrics(self, poses, frame_start=0):
        groups = {}
        for index, binding in enumerate(self.bindings):
            errors, weighted = [], []
            for local_frame, pose in enumerate(poses):
                frame = frame_start + local_frame
                if self.confidence[frame, index] <= 0:
                    continue
                error = float(np.linalg.norm(self.at(pose, index)[0] - self.targets[frame, index]))
                errors.append(error)
                weighted.append(float(self.weights[index] * self.confidence[frame, index] * (error / self.sigma)**2))
            groups[binding.group] = dict(
                valid_frames=len(errors), weight=float(self.weights[index]),
                mean_m=float(np.mean(errors)) if errors else None,
                p95_m=float(np.percentile(errors, 95)) if errors else None,
                max_m=max(errors) if errors else None,
                weighted_cost=float(sum(weighted)),
            )
        return groups


def build_body_relation_tasks(model, config, mapping, priority, source_names,
                              positions, orientations, reference_positions,
                              reference_orientations, robot_reference_qpos):
    """Resolve semantic bindings from the existing profile/adapter contract.

    Missing reference data disables this module, rather than treating the first
    robot output as a desired pose. Invalid motion slots are reported and faded
    based only on the source; they never change identity or normalize a tiny
    direction. Source-specific names/vertices do not belong in this function.
    """
    report = dict(status="disabled", representation="coarse_skeleton_local_vector",
                  surface_clearance=False, bindings=[], disabled=[], valid_relations=0)
    if not config.get("enabled", False):
        report["reason"] = "experiment_disabled"
        return None, report
    weight, sigma = float(config.get("weight", 1.0)), float(config.get("sigma", 0.12))
    if not np.isfinite(weight) or weight < 0 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("body_relations requires finite weight >= 0 and sigma > 0")
    report.update(weight_per_group=weight, sigma_m=sigma,
                  coordinate_source="source bind rotations + existing robot reference-pose calibration")
    if any(x is None for x in (orientations, reference_positions, reference_orientations, robot_reference_qpos)):
        report["reason"] = "missing_explicit_source_bind_or_robot_calibration"
        return None, report
    reference_name = config.get("reference_source")
    if reference_name not in mapping or reference_name not in source_names:
        report["reason"] = "missing_mapped_reference_source"
        return None, report
    names = {name: i for i, name in enumerate(source_names)}
    pos, ori = np.asarray(positions), np.asarray(orientations)
    ref, ref_ori = np.asarray(reference_positions), np.asarray(reference_orientations)
    if (pos.ndim != 3 or pos.shape[1:] != (len(names), 3)
            or ori.shape != (*pos.shape[:2], 4)
            or ref.shape != (len(names), 3) or ref_ori.shape != (len(names), 4)):
        report["reason"] = "incompatible_source_array_shapes"
        return None, report
    b = names[reference_name]
    quat = ref_ori[b]
    if not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-8:
        report["reason"] = "invalid_reference_orientation"
        return None, report
    qref = np.asarray(robot_reference_qpos)
    if qref.shape != (model.nq,) or not np.isfinite(qref).all():
        report["reason"] = "invalid_robot_reference_pose"
        return None, report
    data = mujoco.MjData(model)
    data.qpos[:] = qref
    mujoco.mj_forward(model, data)
    rb = data.xmat[model.body(mapping[reference_name]).id].reshape(3, 3)
    rs = Rotation.from_quat(quat, scalar_first=True).as_matrix()
    calibration = rb.T @ rs
    valid_orientation = np.isfinite(ori[:, b]).all(axis=-1) & (np.linalg.norm(ori[:, b], axis=-1) > 1e-8)
    source_rotation = np.tile(np.eye(3), (len(pos), 1, 1))
    source_rotation[valid_orientation] = Rotation.from_quat(ori[valid_orientation, b], scalar_first=True).as_matrix()
    bindings, targets, confidence = [], [], []
    for group, effector in priority.get("effectors", {}).items():
        arm = effector.get("arm_posture", {})
        chain = [arm.get("source_" + key) for key in ("shoulder", "elbow", "wrist")]
        if not any(chain):
            continue
        if any(name not in mapping or name not in names for name in chain):
            report["disabled"].append(dict(group=group, reason="missing_mapped_arm_chain"))
            continue
        ids = [names[name] for name in chain]
        robot_points = np.array([data.xpos[model.body(mapping[name]).id] for name in chain])
        source_points = ref[ids]
        lengths = np.linalg.norm(np.diff(source_points, axis=0), axis=-1)
        robot_lengths = np.linalg.norm(np.diff(robot_points, axis=0), axis=-1)
        if (not np.isfinite(ref[[b, *ids]]).all() or np.any(lengths < 1e-5)
                or np.any(robot_lengths < 1e-5)):
            report["disabled"].append(dict(group=group, reason="degenerate_reference_chain"))
            continue
        scale = float(robot_lengths.sum() / lengths.sum())
        source_shoulder = rs.T @ (source_points[0] - ref[b])
        robot_shoulder = rb.T @ (robot_points[0] - data.xpos[model.body(mapping[reference_name]).id])
        vector = pos[:, ids[-1]] - pos[:, b]
        length = np.linalg.norm(vector, axis=-1)
        valid = valid_orientation & np.isfinite(pos[:, [b, *ids]]).all(axis=(1, 2)) & (length > 1e-6)
        # Source arm collapse indicates corrupt skeleton data, not a folded arm:
        # a folded arm keeps both bone lengths nonzero.
        motion_lengths = np.linalg.norm(np.diff(pos[:, ids], axis=1), axis=-1)
        valid &= np.all(motion_lengths > 1e-5, axis=-1)
        local = np.einsum("tji,tj->ti", source_rotation, np.where(valid[:, None], vector, 0.0))
        target = robot_shoulder + scale * ((local - source_shoulder) @ calibration.T)
        target[~valid] = 0.0
        conf = np.where(valid, np.minimum(np.nan_to_num(length, nan=0.0) / 0.05, 1.0)**2, 0.0)
        bindings.append(BodyRelationBinding(group, mapping[chain[-1]], mapping[reference_name], chain[-1], reference_name))
        targets.append(target)
        confidence.append(conf)
        report["bindings"].append(dict(
            group=group, source_chain=chain, source_reference=reference_name,
            robot_chain=[mapping[name] for name in chain], robot_reference=mapping[reference_name],
            arm_length_ratio=scale, source_to_robot_local_rotation=calibration.tolist(),
            valid_frames=int(valid.sum()), invalid_frames=np.flatnonzero(~valid).tolist(),
            invalid_reason="nonfinite_or_collapsed_source_geometry_or_invalid_reference_frame",
            confidence=conf.tolist(),
        ))
    if not bindings:
        report["reason"] = "no_reliable_semantic_relations"
        return None, report
    report.update(status="active" if weight > 0 else "wired_zero_weight", valid_relations=len(bindings))
    return BodyRelationTasks(model, bindings, np.stack(targets, axis=1), np.stack(confidence, axis=1),
                             weight=weight, sigma=sigma, diagnostics=report), report
