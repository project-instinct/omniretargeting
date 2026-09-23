"""Transfer the existing bind calibration into generic sequence rotation tasks."""

import numpy as np
from scipy.spatial.transform import Rotation


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
    for name, task in retargeter.body_orientation_tasks.items():
        target = calibrated[name]
        spec = {
            "robot_body": task["body_name"],
            "sigma": float(task["sigma"])
            / np.sqrt(global_weight * float(task["weight"])),
        }
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
