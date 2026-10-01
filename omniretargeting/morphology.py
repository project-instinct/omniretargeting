"""Measure terminal-chain scale from immutable source and robot reference poses."""

from copy import deepcopy
from dataclasses import replace

import mujoco
import numpy as np


def calibrated_target_regimes(tasks, calibration):
    """Unknown limb scale keeps world targets even with observed free space."""
    return {
        name: task
        if calibration.get(name, {}).get("status") in {"measured", "explicit_override"}
        else replace(task, swing_world_mask=np.ones(len(task.contact_mask), bool))
        for name, task in tasks.items()
    }


def calibrate_terminal_scales(model, reference_qpos, mapping, source, effectors):
    """Use the longest mapped, unbranched source chain ending at each terminal.

    Prefer the highest mapped joint below a branch. If that leaves no
    measurable segment, continue to the nearest mapped ancestor, including
    across branches. The robot ancestry must independently agree; a source
    branch is not evidence that the available calibration is missing.
    Intervening unmapped joints still count toward arc length. Missing calibration is
    reported and must retain world targets, never a guessed limb ratio.
    """
    result = deepcopy(effectors)
    names = list(source.get("joint_names", ()))
    parents = np.asarray(source.get("joint_parents", ()), int)
    positions = np.asarray(source.get("reference_joint_positions", ()), float)
    report = {}
    if parents.shape != (len(names),) or positions.shape != (len(names), 3):
        return result, {name: {"status": "missing_reference"} for name in result}
    children = np.bincount(parents[parents >= 0], minlength=len(names))
    data = mujoco.MjData(model)
    data.qpos[:] = reference_qpos
    mujoco.mj_forward(model, data)
    for name, spec in result.items():
        terminal = spec.get("source_joint")
        if terminal not in names:
            report[name] = {"status": "unmapped_source"}
            continue
        path = [names.index(terminal)]
        while parents[path[-1]] >= 0 and children[parents[path[-1]]] == 1:
            path.append(int(parents[path[-1]]))
        mapped = [i for i in path if names[i] in mapping]
        fallback = len(mapped) < 2
        if fallback:
            while parents[path[-1]] >= 0:
                path.append(int(parents[path[-1]]))
                if names[path[-1]] in mapping:
                    break
            mapped = [i for i in path if names[i] in mapping]
        if len(mapped) < 2:
            report[name] = {"status": "insufficient_chain"}
            continue
        origin = mapped[-1]
        path = path[: path.index(origin) + 1]
        source_length = np.linalg.norm(np.diff(positions[path], axis=0), axis=1).sum()
        root_body = int(model.body(mapping[names[origin]]).id)
        robot_path = [int(model.body(spec["robot_body"]).id)]
        while robot_path[-1] != root_body and robot_path[-1] != 0:
            robot_path.append(int(model.body_parentid[robot_path[-1]]))
        robot_length = np.linalg.norm(
            np.diff(data.xpos[robot_path], axis=0), axis=1
        ).sum()
        if robot_path[-1] != root_body or min(source_length, robot_length) <= 1e-10:
            report[name] = {"status": "degenerate_or_incompatible_chain"}
            continue
        measured = float(robot_length / source_length)
        explicit = "swing_morphology_scale" in spec and not spec.get(
            "_scale_calibrated", False
        )
        if not explicit:
            spec["swing_morphology_scale"] = measured
            spec["_scale_calibrated"] = True
        report[name] = {
            "status": "explicit_override" if explicit else "measured",
            "source_chain_root": names[origin],
            "chain_selection": "nearest_mapped_ancestor" if fallback else "unbranched",
            "source_length_m": float(source_length),
            "robot_length_m": float(robot_length),
            "measured_ratio": measured,
            "effective_ratio": spec["swing_morphology_scale"],
        }
    return result, report
