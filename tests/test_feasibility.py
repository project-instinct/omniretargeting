"""Interaction-feasibility tests on non-uniform inline-MJCF humanoids."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import pytest
import trimesh

from omniretargeting.affordance import ContactRegion, build_contact_region
from omniretargeting.contact import ContactMode, SceneContactTask
from omniretargeting.feasibility import (
    ADAPTED,
    FALLBACK_NONE,
    FALLBACK_REGION,
    FALLBACK_REQUIRED_SLACK,
    INFEASIBLE,
    REACHABLE,
    FeasibilityOptions,
    InteractionFeasibilityResult,
    solve_interaction_feasibility,
)
from omniretargeting.morphology import (
    analyze_robot_morphology,
    resolve_end_effector_chain_specs,
)
from tests.robot_fixtures import build_robot


@dataclass(frozen=True)
class _SharedInteraction:
    scene: trimesh.Trimesh
    region: ContactRegion
    source_root: np.ndarray
    source_contact: np.ndarray
    task: SceneContactTask


@dataclass(frozen=True)
class _RobotSolve:
    model: mujoco.MjModel
    result: InteractionFeasibilityResult


def _shared_interaction() -> _SharedInteraction:
    # The contacted side of this one real box is y=0.90.  Its vertical region
    # is large enough for different arm morphologies to select different
    # heights, but it cannot move toward the robot or leave the object plane.
    scene = trimesh.creation.box(
        extents=[0.60, 0.10, 0.70],
        transform=trimesh.transformations.translation_matrix([0.0, 0.85, 1.50]),
    )
    region_samples = np.asarray(
        [
            [-0.08, 0.90, 1.64],
            [0.00, 0.90, 1.65],
            [0.08, 0.90, 1.66],
        ],
        dtype=float,
    )
    normals = np.repeat([[0.0, 1.0, 0.0]], len(region_samples), axis=0)
    region = build_contact_region(
        scene,
        region_samples,
        normals,
        object_id="shared_box",
        neighborhood_radius=0.38,
    )

    num_frames = 3
    source_root = np.repeat([[0.0, 0.0, 0.86]], num_frames, axis=0)
    source_contact = np.repeat([[0.0, 0.90, 1.65]], num_frames, axis=0)
    task = SceneContactTask(
        name="left_hand",
        tracking_positions=source_contact.copy(),
        contact_positions=source_contact.copy(),
        surface_points=source_contact.copy(),
        surface_normals=np.repeat([[0.0, 1.0, 0.0]], num_frames, axis=0),
        distances=np.zeros(num_frames),
        relative_speeds=np.zeros(num_frames),
        contact_confidence=np.ones(num_frames),
        contact_modes=np.full(
            num_frames, int(ContactMode.SUPPORT), dtype=np.uint8
        ),
        contact_segment_ids=np.zeros(num_frames, dtype=np.int32),
        transition_weights=np.ones(num_frames),
    )
    return _SharedInteraction(scene, region, source_root, source_contact, task)


def _solve_robot(
    size: str,
    shared: _SharedInteraction,
    *,
    options: FeasibilityOptions,
    configured_joint_limits: dict[str, list[float]] | None = None,
) -> _RobotSolve:
    xml, config = build_robot(size)
    model = mujoco.MjModel.from_xml_string(xml)
    chains = resolve_end_effector_chain_specs(
        model, config["end_effector_priority"]
    )
    morphology = analyze_robot_morphology(
        model,
        {"root_body": "pelvis", "chains": chains},
        configured_joint_limits=configured_joint_limits,
        reach_samples=64,
    )
    result = solve_interaction_feasibility(
        model,
        morphology,
        {"left_hand": shared.task},
        {("left_hand", 0): shared.region},
        shared.source_root,
        configured_joint_limits=configured_joint_limits,
        options=options,
    )
    return _RobotSolve(model, result)


@pytest.fixture(scope="module")
def feasibility_solves() -> tuple[
    _SharedInteraction, dict[str, _RobotSolve], _RobotSolve
]:
    shared = _shared_interaction()
    scene_vertices = shared.scene.vertices.copy()
    scene_faces = shared.scene.faces.copy()
    source_root = shared.source_root.copy()
    source_contact = shared.source_contact.copy()
    task_arrays = {
        name: np.asarray(getattr(shared.task, name)).copy()
        for name in (
            "tracking_positions",
            "contact_positions",
            "surface_points",
            "surface_normals",
            "contact_modes",
            "contact_segment_ids",
        )
    }

    options = FeasibilityOptions(
        max_root_translation=(0.18, 0.18, 0.14),
        max_root_translation_norm=0.20,
        contact_tolerance=0.004,
        root_velocity_weight=0.1,
        root_acceleration_weight=0.01,
    )
    solves = {
        size: _solve_robot(size, shared, options=options)
        for size in ("short", "medium", "tall")
    }

    # Lock every scalar joint and the root for a genuinely impossible variant
    # of the exact same source contact and scene.  The only correct fallback is
    # an explicit required-contact residual against the unchanged box surface.
    short_model = solves["short"].model
    locked_limits: dict[str, list[float]] = {}
    for joint_id in range(short_model.njnt):
        if short_model.jnt_type[joint_id] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            continue
        name = short_model.joint(joint_id).name
        qpos_index = int(short_model.jnt_qposadr[joint_id])
        value = float(short_model.qpos0[qpos_index])
        locked_limits[name] = [value, value]
    impossible_options = FeasibilityOptions(
        max_root_translation=(0.0, 0.0, 0.0),
        max_root_translation_norm=0.0,
        max_root_yaw=0.0,
        contact_tolerance=0.004,
        max_sqp_iterations=16,
    )
    impossible = _solve_robot(
        "short",
        shared,
        options=impossible_options,
        configured_joint_limits=locked_limits,
    )

    np.testing.assert_array_equal(shared.scene.vertices, scene_vertices)
    np.testing.assert_array_equal(shared.scene.faces, scene_faces)
    np.testing.assert_array_equal(shared.source_root, source_root)
    np.testing.assert_array_equal(shared.source_contact, source_contact)
    for name, expected in task_arrays.items():
        np.testing.assert_array_equal(getattr(shared.task, name), expected)
    return shared, solves, impossible


def _assert_mechanical_joint_limits(model: mujoco.MjModel, qpos: np.ndarray) -> None:
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
        assert np.all(qpos[:, qpos_index] >= lower - 1e-8)
        assert np.all(qpos[:, qpos_index] <= upper + 1e-8)


def test_morphology_changes_minimal_root_and_same_region_point(
    feasibility_solves: tuple[_SharedInteraction, dict[str, _RobotSolve], _RobotSolve],
) -> None:
    shared, solves, _ = feasibility_solves
    short = solves["short"].result.segment("left_hand", 0)
    medium = solves["medium"].result.segment("left_hand", 0)
    tall = solves["tall"].result.segment("left_hand", 0)

    assert tall.reachability_status == REACHABLE
    assert tall.fallback_level == FALLBACK_NONE
    assert tall.max_root_warp < 1e-6

    assert short.reachability_status == ADAPTED
    assert short.fallback_level == FALLBACK_REGION
    assert 0.05 < short.max_root_warp <= 0.20 + 1e-6
    assert medium.reachability_status == ADAPTED
    assert 0.005 < medium.max_root_warp <= 0.20 + 1e-6

    # Root adaptation is attempted before region freedom.  The medium robot
    # needs root-only adaptation and retains the source point, whereas the
    # short robot exhausts that bounded warp and legitimately chooses a lower
    # point on the same box face.
    np.testing.assert_allclose(
        medium.chosen_contact_points_world,
        shared.source_contact,
        atol=2e-5,
    )
    assert np.max(
        np.linalg.norm(
            short.chosen_contact_points_world - tall.chosen_contact_points_world,
            axis=1,
        )
    ) > 0.05
    for solve in solves.values():
        segment = solve.result.segment("left_hand", 0)
        assert segment.region_id == shared.region.region_id
        assert all(
            shared.region.contains(point, plane_tolerance=1e-7)
            for point in segment.chosen_contact_points_world
        )
        # This exact coordinate is the actual connected box face.  A radial
        # reach-shell projection would fail this invariant.
        np.testing.assert_allclose(
            segment.chosen_contact_points_world[:, 1], 0.90, atol=1e-9
        )
        assert segment.max_contact_error <= 0.0041
        _assert_mechanical_joint_limits(solve.model, solve.result.configurations)


def test_impossible_required_contact_reports_slack_without_leaving_surface(
    feasibility_solves: tuple[_SharedInteraction, dict[str, _RobotSolve], _RobotSolve],
) -> None:
    shared, _, impossible_solve = feasibility_solves
    result = impossible_solve.result
    segment = result.segment("left_hand", 0)

    assert not result.feasible
    assert result.fallback_level_reached == FALLBACK_REQUIRED_SLACK
    assert segment.fallback_level == FALLBACK_REQUIRED_SLACK
    assert segment.reachability_status == INFEASIBLE
    assert segment.max_contact_slack > 0.05
    assert segment.max_contact_error > 0.05
    assert "required_contact_residual_exceeds_tolerance" in segment.infeasible_reason
    assert "joint-limit-aware SQP" in segment.infeasible_reason
    assert all(
        shared.region.contains(point, plane_tolerance=1e-7)
        for point in segment.chosen_contact_points_world
    )
    np.testing.assert_allclose(
        segment.chosen_contact_points_world[:, 1], 0.90, atol=1e-9
    )
    _assert_mechanical_joint_limits(impossible_solve.model, result.configurations)
    np.testing.assert_allclose(
        result.configurations[:, 7:],
        np.repeat(
            impossible_solve.model.qpos0[None, 7:],
            len(result.configurations),
            axis=0,
        ),
        atol=1e-10,
    )

    exported = result.to_npz_dict()
    assert exported["interaction_feasibility_segment_status"][0] == INFEASIBLE
    assert exported["interaction_feasibility_contact_point_world"].shape == (3, 3)
    assert all(value.dtype != object for value in exported.values())
