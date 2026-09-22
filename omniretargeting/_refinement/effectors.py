"""Resolve refinement end-effector bodies, markers and joint masks."""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from .math import (
    _joint_coordinate_mask,
    _local_semantic_rotation,
)


def build_refinement_effectors(
    *,
    config,
    end_effector_config,
    model,
    qpos_indices,
    source_joint_positions,
    source_name_to_index,
    tasks,
):
    """Resolve refinement end-effector bodies, markers and joint masks."""
    raw_specs = dict(end_effector_config.get("effectors", {}) or {})
    effectors: dict[str, dict[str, Any]] = {}
    for name, task in tasks.items():
        if name not in raw_specs:
            continue
        spec = dict(raw_specs[name] or {})
        body_name = spec.get("robot_body")
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, str(body_name)
        )
        if body_id < 0:
            continue
        reach_reference_body = spec.get("reach_reference_body")
        reach_reference_body_id = -1
        if reach_reference_body:
            reach_reference_body_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                str(reach_reference_body),
            )
            if reach_reference_body_id < 0:
                raise ValueError(
                    f"End-effector '{name}' refers to missing reach reference "
                    f"body '{reach_reference_body}'."
                )
        tracking_offset = np.asarray(
            spec.get("tracking_offset", [0.0, 0.0, 0.0]), dtype=float
        ).reshape(3)
        contact_offsets = np.asarray(
            spec.get("contact_offsets", [tracking_offset]), dtype=float
        )
        if contact_offsets.ndim == 1:
            contact_offsets = contact_offsets[None, :]
        raw_arm_posture = dict(spec.get("arm_posture", {}) or {})
        arm_posture = None
        if bool(raw_arm_posture.get("enabled", False)):
            shoulder_name = str(raw_arm_posture["shoulder_body"])
            elbow_name = str(raw_arm_posture["elbow_body"])
            wrist_name = raw_arm_posture.get("wrist_body")
            shoulder_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, shoulder_name
            )
            elbow_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, elbow_name
            )
            wrist_body_id = (
                -1
                if wrist_name is None
                else mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, str(wrist_name)
                )
            )
            if shoulder_body_id < 0 or elbow_body_id < 0:
                raise ValueError(
                    f"End-effector '{name}' arm_posture refers to a missing "
                    "shoulder or elbow body."
                )
            if wrist_name is not None and wrist_body_id < 0:
                raise ValueError(
                    f"End-effector '{name}' arm_posture refers to missing "
                    f"wrist body '{wrist_name}'."
                )
            source_shoulder_name = str(raw_arm_posture["source_shoulder"])
            source_elbow_name = str(raw_arm_posture["source_elbow"])
            source_wrist_name = raw_arm_posture.get("source_wrist")
            if source_joint_positions is not None and (
                source_shoulder_name not in source_name_to_index
                or source_elbow_name not in source_name_to_index
            ):
                raise ValueError(
                    f"End-effector '{name}' arm_posture source joints are missing."
                )
            if (
                source_joint_positions is not None
                and source_wrist_name is not None
                and str(source_wrist_name) not in source_name_to_index
            ):
                raise ValueError(
                    f"End-effector '{name}' arm_posture source wrist is missing."
                )
            if (wrist_name is None) != (source_wrist_name is None):
                raise ValueError(
                    f"End-effector '{name}' arm_posture must configure both "
                    "wrist_body and source_wrist together."
                )
            arm_posture = {
                "shoulder_body_id": int(shoulder_body_id),
                "elbow_body_id": int(elbow_body_id),
                "wrist_body_id": int(wrist_body_id),
                "source_shoulder_idx": source_name_to_index.get(
                    source_shoulder_name
                ),
                "source_elbow_idx": source_name_to_index.get(source_elbow_name),
                "source_wrist_idx": (
                    None
                    if source_wrist_name is None
                    else source_name_to_index.get(str(source_wrist_name))
                ),
                "upper_arm_length": float(raw_arm_posture["upper_arm_length"]),
                "forearm_length": float(raw_arm_posture["forearm_length"]),
                "upper_direction_sigma": max(
                    float(
                        raw_arm_posture.get(
                            "upper_direction_sigma",
                            raw_arm_posture.get("sigma", 0.08),
                        )
                    ),
                    1e-6,
                ),
                "forearm_direction_sigma": max(
                    float(
                        raw_arm_posture.get(
                            "forearm_direction_sigma",
                            raw_arm_posture.get("sigma", 0.08),
                        )
                    ),
                    1e-6,
                ),
            }
        refine_orientation = bool(
            spec.get("refine_orientation", spec.get("track_orientation", False))
        )
        orientation_local_rotation = _local_semantic_rotation(
            spec.get("orientation_local_forward", [1.0, 0.0, 0.0]),
            spec.get("orientation_local_up", [0.0, 0.0, 1.0]),
        )
        orientation_joint_tokens = tuple(
            str(token)
            for token in spec.get("orientation_joint_tokens", [])
        )
        orientation_coordinate_mask = _joint_coordinate_mask(
            model,
            qpos_indices,
            orientation_joint_tokens,
        )
        if (
            refine_orientation
            and orientation_joint_tokens
            and not np.any(orientation_coordinate_mask)
        ):
            raise ValueError(
                f"End-effector '{name}' orientation_joint_tokens do not "
                "match any scalar robot joint."
            )
        effectors[name] = {
            "task": task,
            "body_id": body_id,
            "tracking_offset": tracking_offset,
            "contact_offsets": contact_offsets,
            "track_swing": bool(spec.get("track_swing", True)),
            "contact_position_sigma": (
                None
                if spec.get("contact_position_sigma") is None
                else max(float(spec["contact_position_sigma"]), 1e-6)
            ),
            "swing_position_sigma": (
                None
                if spec.get("swing_position_sigma") is None
                else max(float(spec["swing_position_sigma"]), 1e-6)
            ),
            "track_sliding_tangent": bool(
                spec.get("track_sliding_tangent", True)
            ),
            "sliding_tangent_sigma": max(
                float(
                    spec.get(
                        "sliding_tangent_sigma",
                        end_effector_config.get("sliding_tangent_sigma", 0.10),
                    )
                ),
                1e-6,
            ),
            "track_orientation": refine_orientation,
            "orientation_sigma": float(
                spec.get(
                    "orientation_sigma",
                    config.get("orientation_sigma", np.deg2rad(7.5)),
                )
            ),
            "orientation_mode": str(
                spec.get("orientation_mode", "rotation")
            ).lower(),
            "forward_orientation_sigma": max(
                float(
                    spec.get(
                        "forward_orientation_sigma",
                        spec.get(
                            "orientation_sigma",
                            config.get("orientation_sigma", np.deg2rad(7.5)),
                        ),
                    )
                ),
                1e-6,
            ),
            "up_orientation_sigma": max(
                float(
                    spec.get(
                        "up_orientation_sigma",
                        spec.get(
                            "orientation_sigma",
                            config.get("orientation_sigma", np.deg2rad(7.5)),
                        ),
                    )
                ),
                1e-6,
            ),
            "contact_orientation_weight": max(
                float(spec.get("contact_orientation_weight", 1.0)), 0.0
            ),
            "swing_orientation_weight": max(
                float(spec.get("swing_orientation_weight", 1.0)), 0.0
            ),
            "orientation_local_rotation": orientation_local_rotation,
            "orientation_coordinate_mask": (
                orientation_coordinate_mask
                if orientation_joint_tokens
                else None
            ),
            "swing_morphology_scale": float(
                spec.get(
                    "swing_morphology_scale",
                    end_effector_config.get("swing_morphology_scale", 1.0),
                )
            ),
            "preserve_world_target": bool(
                spec.get("preserve_world_target", False)
            ),
            "reach_reference_body_id": int(reach_reference_body_id),
            "reach_reference_offset": np.asarray(
                spec.get("reach_reference_offset", [0.0, 0.0, 0.0]),
                dtype=float,
            ).reshape(3),
            "max_swing_reach": (
                None
                if spec.get("max_swing_reach") is None
                else float(spec["max_swing_reach"])
            ),
            "arm_posture": arm_posture,
        }

    return effectors
