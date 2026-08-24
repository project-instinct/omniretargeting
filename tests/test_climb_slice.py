"""Climbing vertical slice for feasibility, root shaping, and handoff policy."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields

import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting.contact import ContactMode, SceneContactTask
from omniretargeting.core import (
    OmniRetargeter,
    _SupportHandoffGuard,
    _build_robot_base_reference,
)
from omniretargeting.feasibility import INFEASIBLE
from omniretargeting.trajectory_refinement import measure_scene_penetration
from tests.robot_fixtures import VALID_JOINT_NAMES, build_robot


@dataclass(frozen=True)
class _ClimbSource:
    scene: trimesh.Trimesh
    trajectory: np.ndarray
    tasks: dict[str, SceneContactTask]
    root_positions: np.ndarray
    contact_tolerance: float
    motion_scale: float
    step_height: float


@dataclass(frozen=True)
class _RobotSlice:
    name: str
    retargeter: OmniRetargeter
    model: mujoco.MjModel
    config: dict
    tasks: dict[str, SceneContactTask]
    penetration: dict
    handoff: dict


def _scene(step_height: float) -> trimesh.Trimesh:
    ground = trimesh.creation.box(extents=[3.0, 3.0, 0.10])
    ground.apply_translation([0.0, 0.0, -0.05])
    # One connected step component.  Its top contains both the source palm and
    # lifted-foot locations, while connected-component extraction keeps the
    # ground and step affordances semantically distinct.
    step = trimesh.creation.box(extents=[0.80, 0.90, step_height])
    step.apply_translation([0.25, 0.80, 0.5 * step_height])
    return trimesh.util.concatenate([ground, step])


def _source_trajectory(root_positions: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Use one medium humanoid only as a source skeleton, never as a scale."""
    xml, config = build_robot("medium")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    frames = []
    markers = {
        name: []
        for name in ("left_hand", "left_foot", "right_foot")
    }
    for root in root_positions:
        data.qpos[:] = model.qpos0
        data.qpos[:3] = root
        mujoco.mj_forward(model, data)
        frames.append(
            np.asarray(
                [
                    data.xpos[model.body(config["joint_mapping"][name]).id].copy()
                    for name in VALID_JOINT_NAMES
                ]
            )
        )
        for name in markers:
            spec = config["end_effector_priority"]["effectors"][name]
            body_id = model.body(spec["robot_body"]).id
            rotation = data.xmat[body_id].reshape(3, 3)
            markers[name].append(
                data.xpos[body_id]
                + rotation @ np.asarray(spec["tracking_offset"], dtype=float)
            )
    return np.asarray(frames), {
        name: np.asarray(values) for name, values in markers.items()
    }


def _task(
    name: str,
    tracking: np.ndarray,
    segments: list[tuple[int, int, int, np.ndarray]],
) -> SceneContactTask:
    num_frames = len(tracking)
    modes = np.full(num_frames, int(ContactMode.NONE), dtype=np.uint8)
    segment_ids = np.full(num_frames, -1, dtype=np.int32)
    contacts = np.asarray(tracking, dtype=float).copy()
    for segment_id, start, end, point in segments:
        modes[start:end] = int(ContactMode.SUPPORT)
        segment_ids[start:end] = int(segment_id)
        contacts[start:end] = np.asarray(point, dtype=float)
    active = modes != int(ContactMode.NONE)
    return SceneContactTask(
        name=name,
        tracking_positions=np.asarray(tracking, dtype=float).copy(),
        contact_positions=contacts,
        surface_points=contacts.copy(),
        surface_normals=np.repeat([[0.0, 0.0, 1.0]], num_frames, axis=0),
        distances=np.zeros(num_frames),
        relative_speeds=np.zeros(num_frames),
        contact_confidence=active.astype(float),
        contact_modes=modes,
        contact_segment_ids=segment_ids,
        transition_weights=active.astype(float),
        forward_directions=np.repeat([[1.0, 0.0, 0.0]], num_frames, axis=0),
        up_directions=np.repeat([[0.0, 0.0, 1.0]], num_frames, axis=0),
    )


def _climb_source() -> _ClimbSource:
    step_height = 0.90
    # Both feet support first.  The hand locks at frame 2, one foot releases at
    # frame 4, the lifted foot reaches the top at frame 9, the root rises on
    # frame 10.  The source requests old-foot release at frame 9; the handoff
    # guard holds it through establishment and releases it at frame 11.
    root_z = np.asarray(
        [0.55, 0.55, 0.55, 0.55, 0.60, 0.68, 0.75, 0.80, 1.00, 1.20, 1.35, 1.50]
    )
    roots = np.column_stack(
        [np.zeros(len(root_z)), np.full(len(root_z), 0.15), root_z]
    )
    trajectory, markers = _source_trajectory(roots)
    hand_top = np.asarray([0.0, 1.08, step_height])
    left_ground = np.asarray([0.12, 0.255, 0.0])
    right_ground = np.asarray([0.12, 0.045, 0.0])
    left_top = np.asarray([0.25, 0.355, step_height])
    tasks = {
        "left_hand": _task(
            "left_hand", markers["left_hand"], [(0, 2, 6, hand_top)]
        ),
        "left_foot": _task(
            "left_foot",
            markers["left_foot"],
            [(0, 0, 4, left_ground), (1, 9, len(roots), left_top)],
        ),
        "right_foot": _task(
            "right_foot", markers["right_foot"], [(0, 0, 11, right_ground)]
        ),
    }
    motion_scale = float(np.linalg.norm(np.ptp(roots, axis=0)))
    return _ClimbSource(
        scene=_scene(step_height),
        trajectory=trajectory,
        tasks=tasks,
        root_positions=roots,
        # Two percent of the source root excursion, rather than a measured
        # solver residual, defines the required interaction tolerance.
        contact_tolerance=0.02 * motion_scale,
        motion_scale=motion_scale,
        step_height=step_height,
    )


def _adapter(size: str, end_effector_config: dict) -> tuple[OmniRetargeter, mujoco.MjModel, dict]:
    xml, robot_config = build_robot(size)
    model = mujoco.MjModel.from_xml_string(xml)
    retargeter = object.__new__(OmniRetargeter)
    retargeter.robot_model = model
    retargeter.robot_data = mujoco.MjData(model)
    retargeter.robot_height = robot_config["robot_height"]
    retargeter.joint_mapping = deepcopy(robot_config["joint_mapping"])
    retargeter.valid_joint_mapping = deepcopy(robot_config["joint_mapping"])
    retargeter.valid_joint_names = list(VALID_JOINT_NAMES)
    retargeter.smplx_joint_names = list(VALID_JOINT_NAMES)
    retargeter.smplx_joint_indices = {
        name: index for index, name in enumerate(VALID_JOINT_NAMES)
    }
    retargeter.mapped_joint_indices = list(range(len(VALID_JOINT_NAMES)))
    retargeter.base_orientation_config = deepcopy(robot_config["base_orientation"])
    retargeter.retargeting_config = {
        "end_effector_priority": end_effector_config
    }
    return retargeter, model, robot_config


def _climb_config(robot_config: dict, source: _ClimbSource) -> dict:
    config = deepcopy(robot_config["end_effector_priority"])
    config.update(
        {
            "contact_bridge_frames": 2,
            "orientation_bridge_frames": 2,
            "morphology_reach_samples": 32,
            "base_position_constraint": {
                "enabled": True,
                "offset": [0.0, 0.0, 0.0],
            },
            "root_warp": {
                "enabled": True,
                "approach_frames": 4,
                "release_frames": 4,
                "max_xyz": [
                    0.16 * source.motion_scale,
                    0.16 * source.motion_scale,
                    0.13 * source.motion_scale,
                ],
                "max_translation_norm": 0.19 * source.motion_scale,
                "yaw_enabled": False,
                "max_yaw": 0.0,
                "max_linear_velocity": 2.0 * source.motion_scale,
                "max_linear_acceleration": 80.0 * source.motion_scale,
                "max_yaw_rate": 2.0,
                "max_yaw_acceleration": 20.0,
            },
            "interaction_feasibility": {
                "enabled": True,
                "infeasible_policy": "raise",
                "required_contacts": {
                    "left_hand": True,
                    "left_foot": {"0": False, "1": True},
                    "right_foot": False,
                },
                "affordance": {
                    "object_id": "fixed_climb_scene",
                    "neighborhood_radius": 0.20 * source.step_height,
                },
                "options": {
                    "max_root_translation": [
                        0.16 * source.motion_scale,
                        0.16 * source.motion_scale,
                        0.13 * source.motion_scale,
                    ],
                    "max_root_translation_norm": 0.19 * source.motion_scale,
                    "max_root_yaw": 0.0,
                    "max_root_velocity": [
                        20.0 * source.motion_scale,
                        20.0 * source.motion_scale,
                        20.0 * source.motion_scale,
                    ],
                    "max_root_acceleration": [
                        1000.0 * source.motion_scale,
                        1000.0 * source.motion_scale,
                        1000.0 * source.motion_scale,
                    ],
                    "contact_tolerance": source.contact_tolerance,
                    "max_tangential_relaxation": 0.01 * source.step_height,
                    "contact_point_weight": 1000.0,
                    "max_sqp_iterations": 12,
                    "sqp_step_fraction": 0.40,
                    "root_velocity_weight": 0.0,
                    "root_acceleration_weight": 0.0,
                    "joint_velocity_weight": 0.005,
                },
            },
        }
    )
    return config


def _joint_limit_saturation(model: mujoco.MjModel, qpos: np.ndarray) -> int:
    saturated = 0
    for joint_id in range(model.njnt):
        if not model.jnt_limited[joint_id]:
            continue
        if model.jnt_type[joint_id] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            continue
        qpos_index = int(model.jnt_qposadr[joint_id])
        lower, upper = model.jnt_range[joint_id]
        values = qpos[:, qpos_index]
        assert np.all(values >= lower - 1e-8)
        assert np.all(values <= upper + 1e-8)
        margin = 0.01 * max(float(upper - lower), 1e-8)
        saturated += int(
            np.count_nonzero(
                (values <= lower + margin) | (values >= upper - margin)
            )
        )
    return saturated


def _run_slice(size: str, source: _ClimbSource) -> _RobotSlice:
    _, base_robot_config = build_robot(size)
    config = _climb_config(base_robot_config, source)
    retargeter, model, robot_config = _adapter(size, config)
    prepared = retargeter._prepare_contact_aware_references(
        source.trajectory,
        source.tasks,
        config,
        base_orientations=np.repeat(
            [[1.0, 0.0, 0.0, 0.0]], len(source.trajectory), axis=0
        ),
        scene_mesh=source.scene,
        framerate=30.0,
    )
    result = retargeter.last_interaction_feasibility_result
    assert result is not None

    # Exercise the Stage 3b support-transfer guard on the same source schedule.
    guard = _SupportHandoffGuard(
        prepared,
        {
            "enabled": True,
            "hold_frames": 2,
            "position_error_threshold": source.contact_tolerance,
            "max_penetration": 0.10 * source.step_height,
            "events": [
                {
                    "type": "support_transfer",
                    "old_supports": ["right_foot"],
                    "new_supports": ["left_foot"],
                    "release_frame": 9,
                }
            ],
        },
    )
    top = result.segment("left_foot", 1)
    top_errors = {
        int(frame): float(np.linalg.norm(error))
        for frame, error in zip(top.frame_indices, top.contact_errors)
    }
    for frame in range(len(source.trajectory)):
        guard.overrides_for_frame(frame)
        guard.observe(
            frame,
            {"left_foot": top_errors.get(frame, np.inf)},
            0.0,
        )
    prepared = guard.materialize(prepared)
    handoff = guard.diagnostics()[0]

    penetration = measure_scene_penetration(
        model, result.configurations, source.scene
    )
    return _RobotSlice(
        size,
        retargeter,
        model,
        robot_config,
        prepared,
        penetration,
        handoff,
    )


@pytest.fixture(scope="module")
def climb_slice() -> tuple[_ClimbSource, dict[str, _RobotSlice]]:
    source = _climb_source()
    source_before = source.trajectory.copy()
    task_before = {
        (name, field.name): (
            None
            if getattr(task, field.name) is None
            else np.asarray(getattr(task, field.name)).copy()
        )
        for name, task in source.tasks.items()
        for field in fields(SceneContactTask)
        if isinstance(getattr(task, field.name), np.ndarray)
    }
    vertices_before = source.scene.vertices.copy()
    faces_before = source.scene.faces.copy()
    runs = {size: _run_slice(size, source) for size in ("short", "medium", "tall")}

    # The source and scene are shared read-only inputs across all morphologies.
    np.testing.assert_array_equal(source.trajectory, source_before)
    np.testing.assert_array_equal(source.scene.vertices, vertices_before)
    np.testing.assert_array_equal(source.scene.faces, faces_before)
    for (name, field_name), expected in task_before.items():
        np.testing.assert_array_equal(
            getattr(source.tasks[name], field_name), expected
        )
    return source, runs


def test_climb_slice_same_affordance_adapts_three_morphologies(
    climb_slice: tuple[_ClimbSource, dict[str, _RobotSlice]],
) -> None:
    source, runs = climb_slice
    hand_segments = {
        name: run.retargeter.last_interaction_feasibility_result.segment(
            "left_hand", 0
        )
        for name, run in runs.items()
    }
    region_ids = {segment.region_id for segment in hand_segments.values()}
    assert len(region_ids) == 1
    for run in runs.values():
        region = run.retargeter.last_interaction_contact_regions[("left_hand", 0)]
        assert region.region_id in region_ids
        assert all(
            region.contains(point, plane_tolerance=1e-7)
            for point in hand_segments[run.name].chosen_contact_points_world
        )

    # Point freedom is semantic-region freedom, not a morphology-scaled scene.
    short_point = np.median(
        hand_segments["short"].chosen_contact_points_world, axis=0
    )
    tall_point = np.median(
        hand_segments["tall"].chosen_contact_points_world, axis=0
    )
    assert np.linalg.norm(short_point - tall_point) > 0.02 * source.step_height

    # Compare the same required ledge-reach interval for every morphology.
    # The later foot placement is a distinct simultaneous-support problem.
    ledge_warps = {}
    for name, run in runs.items():
        frames = hand_segments[name].frame_indices
        correction = (
            run.retargeter.last_robot_base_reference_w[frames]
            - run.retargeter.last_unwarped_robot_base_reference_w[frames]
        )
        ledge_warps[name] = float(np.max(np.linalg.norm(correction, axis=1)))
    assert ledge_warps["short"] > ledge_warps["medium"] > ledge_warps["tall"]
    assert ledge_warps["tall"] < 0.025 * source.motion_scale

    for name, run in runs.items():
        segment = hand_segments[name]
        errors = np.linalg.norm(segment.contact_errors, axis=1)
        assert np.percentile(errors, 95) <= source.contact_tolerance
        assert np.max(errors) <= source.contact_tolerance
        np.testing.assert_array_equal(
            run.tasks["left_hand"].contact_positions[segment.frame_indices],
            segment.chosen_contact_points_world,
        )
        foot_segment = (
            run.retargeter.last_interaction_feasibility_result.segment(
                "left_foot", 1
            )
        )
        assert foot_segment.required
        assert foot_segment.reachability_status != INFEASIBLE
        assert np.max(np.linalg.norm(foot_segment.contact_errors, axis=1)) <= (
            source.contact_tolerance
        )
        np.testing.assert_array_equal(
            run.tasks["left_foot"].contact_positions[
                foot_segment.frame_indices
            ],
            foot_segment.chosen_contact_points_world,
        )
        foot_region = run.retargeter.last_interaction_contact_regions[
            ("left_foot", 1)
        ]
        assert all(
            foot_region.contains(point, plane_tolerance=1e-7)
            for point in foot_segment.chosen_contact_points_world
        )
        assert _joint_limit_saturation(
            run.model,
            run.retargeter.last_interaction_feasibility_result.configurations,
        ) >= 0
        assert np.isfinite(run.penetration["max_depth"])
        assert run.penetration["max_depth"] < 0.11 * source.motion_scale

        diagnostics = run.retargeter.last_root_warp_diagnostics
        root_cfg = _climb_config(run.config, source)["root_warp"]
        assert diagnostics["correction_authority"] == "interaction_feasibility"
        assert diagnostics["max_linear_velocity"] <= (
            root_cfg["max_linear_velocity"] + 1e-9
        )
        assert diagnostics["max_linear_acceleration"] <= (
            root_cfg["max_linear_acceleration"] + 1e-9
        )
        yaw = run.retargeter.last_robot_base_yaw_correction
        assert np.max(np.abs(np.diff(yaw) * 30.0)) <= (
            root_cfg["max_yaw_rate"] + 1e-9
        )

        exported = run.retargeter.last_interaction_feasibility_export_payload
        expected = run.retargeter.last_interaction_feasibility_result.to_npz_dict()
        assert exported.keys() == expected.keys()
        for key in expected:
            np.testing.assert_array_equal(exported[key], expected[key])


def test_climb_contact_order_and_handoff_guard(
    climb_slice: tuple[_ClimbSource, dict[str, _RobotSlice]],
) -> None:
    source, runs = climb_slice
    assert np.all(
        source.tasks["left_foot"].contact_modes[:4]
        == int(ContactMode.SUPPORT)
    )
    assert np.all(
        source.tasks["right_foot"].contact_modes[:4]
        == int(ContactMode.SUPPORT)
    )
    assert source.tasks["left_hand"].contact_segment_ids[2] == 0
    assert source.tasks["left_foot"].contact_segment_ids[4] == -1
    assert source.tasks["left_foot"].contact_segment_ids[9] == 1
    assert source.root_positions[10, 2] > source.root_positions[9, 2]
    assert source.tasks["right_foot"].contact_modes[11] == int(ContactMode.NONE)

    for run in runs.values():
        assert run.handoff["established_frame"] is not None
        assert run.handoff["established_frame"] < run.handoff["actual_release_frame"]
        assert run.handoff["actual_release_frame"] >= 11
        assert run.handoff["held_extra_frames"] >= 2
        # Contact indices retain the source order after materializing any hold.
        hand_frames = np.flatnonzero(
            run.tasks["left_hand"].contact_modes != int(ContactMode.NONE)
        )
        top_frames = np.flatnonzero(
            run.tasks["left_foot"].contact_segment_ids == 1
        )
        old_ground_frames = np.flatnonzero(
            run.tasks["right_foot"].contact_segment_ids == 0
        )
        assert hand_frames[0] < hand_frames[-1] < top_frames[0]
        assert top_frames[0] < old_ground_frames[-1] + 1


def test_feasibility_root_decision_is_shaped_but_never_added_to_legacy_warp() -> None:
    """This fails immediately if reach-excess and feasibility are ever summed."""
    num_frames = 12
    roots = np.zeros((num_frames, 3))
    targets = np.repeat([[0.0, 0.90, 0.0]], num_frames, axis=0)
    task = _task(
        "left_hand",
        targets,
        [(0, 4, 8, np.asarray([0.0, 0.90, 0.0]))],
    )
    profile = {"origin_offset": np.asarray([0.35, 0.0, 0.0]), "max_reach": 0.20}
    config = {
        "enabled": True,
        "approach_frames": 0,
        "release_frames": 0,
        "max_xyz": [0.5, 0.5, 0.5],
        "max_translation_norm": 0.8,
        "yaw_enabled": True,
        "max_yaw": 0.5,
    }
    decided_translation = np.repeat([[0.04, -0.02, 0.01]], 4, axis=0)
    decided_yaw = np.repeat(0.07, 4)
    shaped, unwarped, yaw, diagnostics = _build_robot_base_reference(
        roots,
        {"left_hand": task},
        {"left_hand": profile},
        config,
        0.0,
        base_orientations=np.repeat(
            [[1.0, 0.0, 0.0, 0.0]], num_frames, axis=0
        ),
        return_diagnostics=True,
        feasibility_segment_corrections={
            ("left_hand", 0): {
                "translation": decided_translation,
                "yaw": decided_yaw,
            }
        },
    )
    np.testing.assert_allclose(
        shaped[4:8] - unwarped[4:8], decided_translation, atol=1e-14
    )
    np.testing.assert_allclose(yaw[4:8], decided_yaw, atol=1e-14)
    assert diagnostics["correction_authority"] == "interaction_feasibility"

    legacy, legacy_unwarped = _build_robot_base_reference(
        roots,
        {"left_hand": task},
        {"left_hand": profile},
        config,
        0.0,
    )
    legacy_correction = legacy[4:8] - legacy_unwarped[4:8]
    assert np.max(np.linalg.norm(legacy_correction, axis=1)) > 0.1
    # The external path equals the decision, not decision + the independently
    # nonzero legacy correction demonstrated above.
    assert not np.allclose(
        shaped[4:8] - unwarped[4:8],
        decided_translation + legacy_correction,
    )


def test_disabled_prepass_is_bit_exact_with_the_previous_pipeline() -> None:
    source = _climb_source()
    _, robot_config = build_robot("medium")
    legacy_config = deepcopy(robot_config["end_effector_priority"])
    legacy_config.update(
        {
            "contact_bridge_frames": 2,
            "morphology_reach_samples": 32,
            "base_position_constraint": {"offset": [0.0, 0.0, 0.0]},
            "root_warp": {"enabled": False},
        }
    )
    disabled_config = deepcopy(legacy_config)
    disabled_config["interaction_feasibility"] = {"enabled": False}
    outputs = []
    references = []
    for config in (legacy_config, disabled_config):
        retargeter, _, _ = _adapter("medium", config)
        outputs.append(
            retargeter._prepare_contact_aware_references(
                source.trajectory,
                source.tasks,
                config,
                base_orientations=np.repeat(
                    [[1.0, 0.0, 0.0, 0.0]], len(source.trajectory), axis=0
                ),
                scene_mesh=source.scene,
                framerate=30.0,
            )
        )
        references.append(retargeter.last_robot_base_reference_w.copy())
        assert retargeter.last_interaction_feasibility_result is None
        assert retargeter.last_interaction_feasibility_export_payload == {}
    np.testing.assert_array_equal(references[0], references[1])
    for name in outputs[0]:
        for field in fields(SceneContactTask):
            first = getattr(outputs[0][name], field.name)
            second = getattr(outputs[1][name], field.name)
            if isinstance(first, np.ndarray):
                np.testing.assert_array_equal(first, second)
            else:
                assert first == second


def test_unreachable_required_climb_contact_raises_with_retained_slack() -> None:
    source = _climb_source()
    _, robot_config = build_robot("short")
    config = _climb_config(robot_config, source)
    model = mujoco.MjModel.from_xml_string(build_robot("short")[0])
    locked_limits = {}
    for joint_id in range(model.njnt):
        if model.jnt_type[joint_id] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            continue
        qpos_index = int(model.jnt_qposadr[joint_id])
        value = float(model.qpos0[qpos_index])
        locked_limits[model.joint(joint_id).name] = [value, value]
    config["joint_posture_limits"] = locked_limits
    config["root_warp"].update(
        max_xyz=[0.0, 0.0, 0.0], max_translation_norm=0.0, max_yaw=0.0
    )
    config["interaction_feasibility"]["required_contacts"] = True
    config["interaction_feasibility"]["options"].update(
        max_root_translation=[0.0, 0.0, 0.0],
        max_root_translation_norm=0.0,
        max_root_yaw=0.0,
        max_sqp_iterations=6,
    )
    retargeter, _, _ = _adapter("short", config)
    with pytest.raises(RuntimeError, match="Required interaction feasibility failed"):
        retargeter._prepare_contact_aware_references(
            source.trajectory,
            {"left_hand": source.tasks["left_hand"]},
            config,
            base_orientations=np.repeat(
                [[1.0, 0.0, 0.0, 0.0]], len(source.trajectory), axis=0
            ),
            scene_mesh=source.scene,
            framerate=30.0,
        )
    result = retargeter.last_interaction_feasibility_result
    assert result is not None
    segment = result.segment("left_hand", 0)
    assert segment.reachability_status == INFEASIBLE
    assert segment.max_contact_slack > source.contact_tolerance
    region = retargeter.last_interaction_contact_regions[("left_hand", 0)]
    assert all(region.contains(point) for point in segment.chosen_contact_points_world)
