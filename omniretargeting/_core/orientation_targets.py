"""Transfer the existing bind calibration into generic sequence rotation tasks."""

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

# Distal segments that no other task observes (hands) follow their calibrated
# source orientation this much more tightly than other segments. Landmark
# relations cannot see a leaf segment turning about its own axis, and without a
# sole frame nothing else holds it, so a weak target let contact rows twist it.
UNOBSERVED_LEAF_SIGMA_SCALE = 0.25


def unobserved_leaf_bodies(model, mapped_bodies, effectors):
    """Robot bodies of leaf segments whose orientation no other task observes.

    A leaf has no mapped body below it in the kinematic tree; everything below
    it is its own terminal chain. It is observed when an effector on that chain
    tracks its own orientation (a sole frame, ``track_orientation``). Only the
    tree and the resolved effector specifications decide; no body or effector
    names are used.
    """
    mapped = {int(body) for body in mapped_bodies if int(body) > 0}
    inner = set()
    for body in mapped:
        parent = int(model.body_parentid[body])
        while parent > 0:
            inner.add(parent)
            parent = int(model.body_parentid[parent])
    observed = set()
    for spec in effectors.values():
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(spec.get("robot_body", "")))
        if not spec.get("track_orientation", False):
            continue
        while body > 0:
            observed.add(int(body))
            body = int(model.body_parentid[body])
    return {body for body in range(1, model.nbody) if body not in inner and body not in observed}


def sequence_body_orientation_targets(
    retargeter, source_orientations, base_orientations
):
    if source_orientations is None or not retargeter.body_orientation_enabled:
        return []
    source = np.asarray(source_orientations, dtype=float)
    indices = retargeter._source_orientation_indices
    if not indices:
        return []
    calibrated = {}
    for name, index in indices.items():
        calibrated[name] = (
            Rotation.from_quat(source[:, index], scalar_first=True).as_matrix()
            @ retargeter._source_orientation_references[name].T
            @ retargeter._robot_orientation_references[name]
        )
    result = []
    global_weight = float(
        retargeter.end_effector_config.get("body_orientation_weight", 1.0)
    )
    if not np.isfinite(global_weight) or global_weight < 0:
        raise ValueError("body_orientation_weight must be finite and nonnegative")
    if global_weight == 0:
        return result
    model = retargeter.robot_model
    leaves = unobserved_leaf_bodies(
        model,
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in retargeter.joint_mapping.values()],
        retargeter.end_effector_config.get("effectors", {}) or {},
    )
    for name, task in retargeter.body_orientation_tasks.items():
        target = calibrated[name]
        sigma = float(task["sigma"]) / np.sqrt(global_weight * float(task["weight"]))
        if int(task["body_id"]) in leaves:
            sigma *= UNOBSERVED_LEAF_SIGMA_SCALE
        spec = {"robot_body": task["body_name"], "sigma": sigma}
        if "parent_source" in task:
            parent = calibrated[task["parent_source"]]
            parent_id = task["parent_body_id"]
            if (
                base_orientations is not None
                and retargeter.robot_model.body_parentid[parent_id] == 0
            ):
                parent = Rotation.from_quat(
                    base_orientations, scalar_first=True
                ).as_matrix()
            target = parent.transpose(0, 2, 1) @ target
            spec["parent_body"] = retargeter.robot_model.body(parent_id).name
        result.append(dict(spec, rotations=target))
    return result
