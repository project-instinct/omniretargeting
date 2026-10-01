"""Stable defaults for contact-aware humanoid retargeting.

As in common GMR-style retargeters, solver policy belongs to the algorithm.
Robot profiles only describe correspondence, geometry, and genuine
model-specific overrides.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from typing import Any, Dict


CONTACT_RETARGETING_DEFAULTS: Dict[str, Any] = {
    "body_relations": {"enabled": False, "weight": 1.0, "sigma": 0.12},
    "self_contact": {"enabled": True},
    "surface_contact": {"enabled": True},
    "surface_pose": {"enabled": True},
    "contact_phase_priority": {"enabled": True},
    "whole_body_correction": {"enabled": True, "moving_collision_enabled": True},
    "preserve_root_z": False,
    "collision_detection_threshold": 0.1,
    "terrain_sample_points": 100,
    "terrain_sample_seed": 0,
    "penetration_resolver": "hard_constraint",
    "self_collision_enabled": True,
    "joint_limit_safety_fraction": 1.0,
    "end_effector_priority": {
        "enabled": True,
        "preserve_noncontact_baseline": False,
        "local_contact_chain_only": False,
        "hierarchy": True,
        "free_motion_priority": "balanced",
        "weighted_scale": 2.0,
        "contact_sigma": 0.025,
        "swing_sigma": 0.06,
        "interaction_mesh": {
            "weighting": "uniform",
            "kappa": 3.0,  # Experimental exponential decay, 1/metre.
            "laplacian_weight": 10.0,
        },
        "priority_relative_tolerance": 0.01,
        "priority_absolute_tolerance": 0.04,
        "swing_priority_relative_tolerance": 0.01,
        "swing_priority_absolute_tolerance": 0.04,
        "orientation_priority_relative_tolerance": 0.01,
        "orientation_priority_absolute_tolerance": 0.04,
        "pose_priority_relative_tolerance": 0.02,
        "pose_priority_absolute_tolerance": 0.04,
        "temporal_weight": 0.0,
        "tie_break_regularization": 1e-8,
        "contact_normal_tolerance": 0.005,
        "orientation_sigma": 0.1308996939,
        "swing_morphology_scale": 1.0,
        "pose_direction": {
            "enabled": True,
            "velocity_enabled": False,
            "sigma": 0.12,
            "arm_weight": 1.0,
            "leg_weight": 1.0,
            "edge_weight_overrides": {
                "L_Shoulder->L_Elbow": 2.0,
                "R_Shoulder->R_Elbow": 2.0,
                "L_Elbow->L_Wrist": 6.0,
                "R_Elbow->R_Wrist": 6.0,
            },
            "velocity_sigma": 0.06,
            "acceleration_enabled": False,
            "acceleration_sigma": 0.1,
            "edges": [
                ["Pelvis", "Spine3"],
                ["L_Shoulder", "L_Elbow"],
                ["L_Elbow", "L_Wrist"],
                ["R_Shoulder", "R_Elbow"],
                ["R_Elbow", "R_Wrist"],
                ["L_Hip", "L_Knee"],
                ["L_Knee", "L_Ankle"],
                ["R_Hip", "R_Knee"],
                ["R_Knee", "R_Ankle"],
            ],
        },
        "pose_initialization": {"enabled": False, "max_iterations": 20},
        "temporal": {
            "enabled": True,
            "acceleration_sigma": 0.015,
            "jerk_sigma": 0.01,
            "max_base_step": 0.08,
            "max_base_acceleration_step": 0.03,
        },
        "base_position_constraint": {
            # Contact reachability can move the body horizontally. The source
            # path is a soft reference, independent of torso orientation.
            "enabled": True,
            "use_initial_offset": False,
            "lock_initial_xy": False,
            "track_xy": True,
            "lock_xy": False,
            "track_z": False,
            "hard_constraints": False,
            "xy_sigma": 0.25,
            "max_xy_error": 0.03,
            "z_offset": 0.0,
            "max_z_error": 0.08,
            "centering_weight": 1.0,
        },
        "contact_detection": {
            "distance_threshold": 0.06,
            "relative_speed_threshold": 0.2,
            "distance_softness": 0.01,
            "speed_softness": 0.08,
            "min_contact_frames": 3,
            "max_gap_frames": 2,
            "transition_frames": 4,
            "anchor_contacts": True,
            "anchor_contact_orientation": False,
            "anchor_window_frames": 3,
            "normal_alignment_threshold": 0.5235987756,
            "surface_clearance": 0.005,
            "normal_orientation_deadzone": 0.005,
        },
        "effectors": {},
        "base_orientation_weight": 400.0,
        "body_orientation": {
            "position_anchor_weight": 4.0,
        },
    },
    "local_collision_repair": {"enabled": True, "full_geometry": True,
                               "bidirectional_enabled": True,
                               "max_query_points": None, "max_substeps": 256},
    "trajectory_smoothing": {"enabled": False, "sigma_frames": 2.0},
    "trajectory_refinement": {
        "enabled": True,
        "iterations": 16,
        "contact_position_sigma": 0.008,
        "swing_position_sigma": 0.015,
        "orientation_sigma": 0.0698131701,
        "reference_sigma": 0.8,
        "acceleration_enabled": True,
        "acceleration_sigma": 0.02,
        "jerk_enabled": True,
        "jerk_sigma": 0.01,
        "joint_limit_avoidance": {
            "enabled": True,
            "hard_margin": 0.0,
            "hard_margin_ratio": 0.05,
            "soft_margin": 0.0,
            "soft_margin_ratio": 0.15,
            "sigma": 0.05,
            "sigma_ratio": 0.03,
        },
        "trust_region": 0.12,
        "solver_tolerance": 1e-5,
        "solver_iterations": 80,
        "step_tolerance": 0.0001,
        "sigma_reference_framerate": 30.0,
        "penetration": {
            "enabled": True,
            "report": True,
            "target_max_depth": 0.10,
            "clearance": 0.005,
            "margin": 0.05,
            "sigma": 0.008,
            "depth_priority_start": 0.05,
            "max_depth_weight": 9.0,
            "contact_anchor_exclusion_radius": 0.0,
        },
    },

}


def _deep_merge(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> Dict[str, Any]:
    result: Dict[str, Any] = deepcopy(dict(base))
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = deepcopy(value)
    return result


def apply_retargeting_defaults(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply the same algorithm defaults to every robot profile.

    Effector geometry is inferred later from the loaded model. Declarations
    here are optional robot-fact overrides, not an activation gate.
    """
    overrides = dict(config)
    priority = overrides.get("end_effector_priority")
    declared_effectors = dict((priority or {}).get("effectors", {}))
    default_effectors = CONTACT_RETARGETING_DEFAULTS["end_effector_priority"][
        "effectors"
    ]
    merged = _deep_merge(CONTACT_RETARGETING_DEFAULTS, overrides)
    merged["end_effector_priority"]["effectors"] = {
        name: _deep_merge(default_effectors.get(name, {}), spec or {})
        for name, spec in declared_effectors.items()
    }
    return merged
