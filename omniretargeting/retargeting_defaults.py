"""Stable defaults for contact-aware humanoid retargeting.

As in common GMR-style retargeters, solver policy belongs to the algorithm.
Robot profiles only describe correspondence, geometry, and genuine
model-specific overrides.
"""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from typing import Any, Dict


def _hand_defaults(side: str) -> Dict[str, Any]:
    prefix = "L" if side == "left" else "R"
    return {
        "preserve_world_target": False,
        "track_swing": False,
        "track_sliding_tangent": False,
        "contact_sigma": 0.02,
        "contact_position_sigma": 0.012,
        "swing_position_sigma": 0.01,
        "refine_orientation": True,
        "orientation_mode": "axes",
        "forward_orientation_sigma": 0.18,
        "up_orientation_sigma": 0.22,
        "contact_orientation_weight": 1.5,
        "swing_orientation_weight": 1.0,
        "arm_posture": {
            "enabled": True,
            "source_shoulder": f"{prefix}_Shoulder",
            "source_elbow": f"{prefix}_Elbow",
            "source_wrist": f"{prefix}_Wrist",
            "upper_direction_sigma": 0.015,
            "forearm_direction_sigma": 0.015,
        },
        "swing_morphology_scale": 0.9,
    }


def _foot_defaults() -> Dict[str, Any]:
    return {
        "preserve_world_target": True,
        "track_orientation": True,
        "track_swing_heading": True,
        "track_swing_orientation": True,
        "orientation_sigma": 0.0698131701,
        "swing_sigma": 0.02,
        "swing_morphology_scale": 1.0,
        "max_orientation_step": 0.1745329252,
    }


CONTACT_RETARGETING_DEFAULTS: Dict[str, Any] = {
    "body_relations": {"enabled": False, "weight": 1.0, "sigma": 0.12},
    "self_contact": {"enabled": True},
    "surface_contact": {"enabled": True},
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
        "world_tracking_only": False,
        "world_tracking_enabled": True,
        "preserve_noncontact_baseline": True,
        "local_contact_chain_only": True,
        "semantic_contact_effectors": ["left_hand", "right_hand"],
        "world_tracking_effectors": ["left_foot", "right_foot"],
        "hierarchy": True,
        "free_motion_priority": "balanced",
        "weighted_scale": 2.0,
        "contact_sigma": 0.025,
        "swing_sigma": 0.06,
        "interaction_mesh": {
            "weighting": "uniform",
            "kappa": 3.0,  # Experimental exponential decay, 1/metre.
            "laplacian_weight": 10.0,
            "joint_weight_overrides": {
                "L_Elbow": 5.0,
                "R_Elbow": 5.0,
                "L_Knee": 5.0,
                "R_Knee": 5.0,
                "L_Wrist": 2.0,
                "R_Wrist": 2.0,
                "L_Ankle": 2.0,
                "R_Ankle": 2.0,
            },
        },
        "combine_swing_foot_pose": True,
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
            "edge_joint_effectors": {
                "L_Shoulder->L_Elbow": "left_hand",
                "R_Shoulder->R_Elbow": "right_hand",
                "L_Elbow->L_Wrist": "left_hand",
                "R_Elbow->R_Wrist": "right_hand",
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
            # Register the robot to the source once on frame zero, then keep
            # the source root's horizontal displacement exactly.  Height is
            # deliberately left to the leg/contact/terrain solve.
            "enabled": True,
            "use_initial_offset": True,
            "lock_initial_xy": True,
            "track_xy": True,
            "lock_xy": True,
            "track_z": False,
            "hard_constraints": True,
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
            "effectors": {
                "left_hand": {
                    "distance_threshold": 0.03,
                    "relative_speed_threshold": 2.0,
                    "require_normal_alignment": False,
                    "anchor_contacts": True,
                    "sticking_tangential_speed_threshold": 0.15,
                    "sticking_normal_speed_threshold": 0.15,
                    "min_sticking_frames": 3,
                },
                "right_hand": {
                    "distance_threshold": 0.03,
                    "relative_speed_threshold": 2.0,
                    "require_normal_alignment": False,
                    "anchor_contacts": True,
                    "sticking_tangential_speed_threshold": 0.15,
                    "sticking_normal_speed_threshold": 0.15,
                    "min_sticking_frames": 3,
                },
                "left_foot": {
                    "distance_threshold": 0.07,
                    "relative_speed_threshold": 0.3,
                    "anchor_drift_tolerance": 0.05,
                },
                "right_foot": {
                    "distance_threshold": 0.07,
                    "relative_speed_threshold": 0.3,
                    "anchor_drift_tolerance": 0.05,
                },
            },
        },
        "effectors": {
            "left_hand": _hand_defaults("left"),
            "right_hand": _hand_defaults("right"),
            "left_foot": _foot_defaults(),
            "right_foot": _foot_defaults(),
        },
        "base_orientation_weight": 150.0,
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
        "joint_temporal_weights": {"shoulder": 2.0, "elbow": 3.0, "wrist": 6.0},
        "joint_posture_targets": {
            "shoulder_pitch": {"target": 0.0, "sigma": 1.1},
            "shoulder_roll": {"target": 0.0, "sigma": 0.9},
            "shoulder_yaw": {"target": 0.0, "sigma": 0.8},
            "wrist_roll": {"target": 0.0, "sigma": 0.3},
            "wrist_pitch": {"target": 0.0, "sigma": 0.3},
            "wrist_yaw": {"target": 0.0, "sigma": 0.35},
        },
        "joint_limit_avoidance": {
            "enabled": True,
            "hard_margin": 0.0,
            "hard_margin_ratio": 0.05,
            "soft_margin": 0.0,
            "soft_margin_ratio": 0.15,
            "sigma": 0.05,
            "sigma_ratio": 0.03,
        },
        "swing_joint_step_avoidance": {
            "enabled": True,
            "max_step": 0.1047197551,
            "sigma": 0.0087266463,
            "effectors": {
                "left_foot": ["left_hip", "left_knee", "left_ankle"],
                "right_foot": ["right_hip", "right_knee", "right_ankle"],
            },
        },
        "joint_step_avoidance": {
            "enabled": True,
            "groups": {
                "waist": {
                    "joint_tokens": ["waist"],
                    "max_step": 0.1396263402,
                    "sigma": 0.0087266463,
                },
                "shoulder": {
                    "joint_tokens": ["shoulder"],
                    "max_step": 0.1745329252,
                    "sigma": 0.0087266463,
                },
                "elbow": {
                    "joint_tokens": ["elbow"],
                    "max_step": 0.1745329252,
                    "sigma": 0.0087266463,
                },
                "wrist": {
                    "joint_tokens": ["wrist"],
                    "max_step": 0.2094395102,
                    "sigma": 0.0087266463,
                },
            },
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
    "foot_stabilization": {
        "enabled": True,
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
        "max_xy_correction": 0.0,
        "max_surface_correction": 0.0,
        "surface_iterations": 4,
        "wall_red_axis_only": True,
        "wall_normal_z_threshold": 0.35,
        "wall_x_dominance_threshold": 0.5,
        "ground_lock": True,
        "preserve_root_z": False,
        "ground_lock_clearance": 0.0,
        "ground_lock_body_tokens": ["ankle", "foot"],
        "ground_lock_geom_groups": [2, 3],
    },
    "object_alignment": {
        "enabled": True,
        "method": "frame_yaw_xy",
        "smooth_window": 5,
        "apply_yaw_to_orientation": True,
        "reference_joints": [
            "Pelvis",
            "Spine1",
            "Spine3",
            "L_Hip",
            "R_Hip",
            "L_Knee",
            "R_Knee",
            "L_Ankle",
            "R_Ankle",
            "L_Shoulder",
            "R_Shoulder",
            "L_Elbow",
            "R_Elbow",
            "L_Wrist",
            "R_Wrist",
        ],
    },
    "terrain_sample_seed": 0,
}


def _deep_merge(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> Dict[str, Any]:
    result: Dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = deepcopy(value)
    return result


def apply_retargeting_defaults(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Apply interaction defaults when a robot declares contact effectors.

    A plain legacy profile keeps its existing behavior. Declaring physical
    end-effectors is the only opt-in needed; there is no user-facing preset or
    external parameter file.
    """
    overrides = dict(config)
    priority = overrides.get("end_effector_priority")
    has_effectors = isinstance(priority, Mapping) and bool(priority.get("effectors"))
    if not has_effectors:
        return deepcopy(overrides)

    declared_effectors = dict(priority["effectors"])
    default_effectors = CONTACT_RETARGETING_DEFAULTS["end_effector_priority"][
        "effectors"
    ]
    merged = _deep_merge(CONTACT_RETARGETING_DEFAULTS, overrides)
    merged["end_effector_priority"]["effectors"] = {
        name: _deep_merge(default_effectors.get(name, {}), spec or {})
        for name, spec in declared_effectors.items()
    }
    return merged
