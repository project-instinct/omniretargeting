"""Regression tests for preserving primary foot endpoints around surface tasks."""

import mujoco
import numpy as np
import trimesh

from omniretargeting.retargeting import GenericInteractionRetargeter
from omniretargeting.surface_contact import SurfaceContactTasks
from tests._contact_fixtures import _task
from tests.test_surface_contact import contact_case


def _make_solver(model):
    return GenericInteractionRetargeter(
        model,
        mujoco.MjData(model),
        trimesh.creation.box(extents=[2.0, 2.0, 0.1]),
        {"root": "root", "forearm": "forearm"},
        1.0,
        terrain_sample_points=4,
        terrain_sample_seed=0,
        hard_penetration_constraint=False,
        self_collision_enabled=False,
        step_size=0.2,
        end_effector_config={
            "enabled": True,
            "hierarchy": True,
            "free_motion_priority": "lexicographic",
            "pose_direction": {"enabled": False},
            "effectors": {
                "left_foot": {
                    "robot_body": "forearm",
                    "tracking_offset": [0.0, 0.0, -0.1],
                    "contact_offsets": [[0.0, 0.0, -0.1]],
                    "track_swing": True,
                }
            },
        },
    )


def _foot_position(model, pose):
    data = mujoco.MjData(model)
    data.qpos[:] = pose
    mujoco.mj_forward(model, data)
    body = model.body("forearm").id
    return data.xpos[body] + data.xmat[body].reshape(3, 3) @ [0.0, 0.0, -0.1]


def _mapped_points(model, pose):
    data = mujoco.MjData(model)
    data.qpos[:] = pose
    mujoco.mj_forward(model, data)
    return np.array(
        [data.xpos[model.body(name).id] for name in ("root", "forearm")]
    )


def _shifted_surface_tasks(model, source_tasks, offset):
    return SurfaceContactTasks(
        model,
        source_tasks.bindings,
        source_tasks.environment_points,
        source_tasks.source_vectors + np.asarray(offset),
        source_tasks.active,
        source_tasks.normals,
        config={**source_tasks.cfg, "endpoint_regression": 0.001},
    )


def test_conflicting_whole_body_surface_cannot_purchase_feet_endpoint_error():
    model, poses, source_tasks, _ = contact_case(frames=2)
    solver = _make_solver(model)
    solver.surface_contact_tasks = _shifted_surface_tasks(
        model, source_tasks, [0.12, 0.0, 0.0]
    )
    endpoint = _foot_position(model, poses[0])
    result = solver.retarget_frame(
        _mapped_points(model, poses[0]),
        poses[0],
        max_iter=4,
        relation_frame=0,
        end_effector_targets={
            "left_foot": _task(
                np.tile(endpoint, (2, 1)), np.ones(2, dtype=bool)
            ).frame(0)
        },
    )

    result_endpoint = _foot_position(model, result)
    surface_target = solver.surface_contact_tasks.targets[0, 0]
    assert np.isfinite(result).all()
    assert np.linalg.norm(result_endpoint - endpoint) <= 0.002
    # The shifted whole-body target remains visibly conflicting; it was not
    # allowed to hide a foot regression inside an aggregate objective.
    assert np.linalg.norm(result_endpoint - surface_target) >= 0.1


def test_unreachable_source_endpoint_returns_finite_bounded_solution():
    model, poses, source_tasks, _ = contact_case(frames=2)
    solver = _make_solver(model)
    solver.surface_contact_tasks = _shifted_surface_tasks(
        model, source_tasks, [-0.2, 0.0, 0.0]
    )
    current_endpoint = _foot_position(model, poses[0])
    unreachable_target = current_endpoint + [5.0, 0.0, 0.0]
    result = solver.retarget_frame(
        _mapped_points(model, poses[0]),
        poses[0],
        max_iter=1,
        relation_frame=0,
        end_effector_targets={
            "left_foot": _task(
                np.tile(unreachable_target, (2, 1)), np.ones(2, dtype=bool)
            ).frame(0)
        },
    )

    result_endpoint = _foot_position(model, result)
    assert np.isfinite(result).all()
    # The source target is deliberately outside one bounded SQP step.  A
    # feasible-but-inexact result is preferable to an infeasible zero-error
    # demand or a NaN/failed frame.
    assert np.linalg.norm(result_endpoint - unreachable_target) > 4.5

