import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting.local_collision_repair import repair_local_collisions
from omniretargeting.surface_tasks import (
    SurfaceBinding,
    SurfaceKinematics,
    SurfaceTasks,
)
from tests.test_local_candidates import directional_case
from tests.test_local_collision_repair import setup_motion


def binding():
    return SurfaceBinding(
        "forearm", 0, (0.2, 0.3, 0.5), "toe", (0.01, 0, 0), (1.0, 0, 0)
    )


def test_reference_normal_transport_is_identity_and_known_rotation_is_correct():
    model, q, _ = setup_motion(2)
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    motion = np.array([vertices, vertices @ rotation.T + [4, 5, 6]])
    original = motion.copy()
    task = SurfaceTasks(
        model,
        [binding()],
        vertices,
        [[0, 1, 2]],
        motion,
        q[0],
        corresponding_reference_poses=True,
    )
    np.testing.assert_allclose(task.targets[:, 0], [[1, 0, 0], [0, 1, 0]], atol=1e-15)
    # Human's reference normal is z; blindly matching it would already be wrong.
    assert task.score(q[0], 0) == 0
    point0 = np.array([0.3, 0.5, 0])
    np.testing.assert_allclose(
        task.source_points[:, 0], [point0, rotation @ point0 + [4, 5, 6]]
    )
    np.testing.assert_array_equal(motion, original)
    assert task.diagnostics["position_weight"] == 0


def test_point_and_normal_jacobian_use_nv_and_invalidate_after_qpos_update():
    model, q, _ = setup_motion(1)
    cache = SurfaceKinematics(model, [binding()])
    p, _n, jp, jn = cache.at(q[0], 0)
    assert jp.shape == jn.shape == (3, model.nv)
    for dof in range(model.nv):
        values = []
        for sign in [-1, 1]:
            pose = q[0].copy()
            dq = np.zeros(model.nv)
            dq[dof] = sign * 1e-6
            mujoco.mj_integratePos(model, pose, dq, 1)
            values.append(cache.at(pose, 0)[:2])
        np.testing.assert_allclose(
            jp[:, dof], (values[1][0] - values[0][0]) / 2e-6, atol=1e-7
        )
        np.testing.assert_allclose(
            jn[:, dof], (values[1][1] - values[0][1]) / 2e-6, atol=1e-7
        )
    # Rotation about the normal is intentionally unobservable.
    assert np.linalg.norm(jn[:, 3]) < 1e-8
    changed = q[0].copy()
    changed[7] += 0.1
    np.testing.assert_allclose(cache.at(changed, 0)[0] - p, [0.1, 0, 0])


def test_reference_confirmation_and_degenerate_faces_are_explicit():
    model, q, _ = setup_motion(2)
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    with pytest.raises(ValueError, match="reference poses"):
        SurfaceTasks(model, [binding()], v, [[0, 1, 2]], [v, v], q[0])
    task = SurfaceTasks(
        model,
        [binding()],
        v,
        [[0, 1, 2]],
        [v, v * 0],
        q[0],
        corresponding_reference_poses=True,
    )
    assert task.diagnostics["invalid_frame_slots"] == [[1, 0]]
    assert not list(task.rows(q[0], 1, np.arange(model.nv)))
    with pytest.raises(ValueError, match="Degenerate reference"):
        SurfaceTasks(
            model,
            [binding()],
            v * 0,
            [[0, 1, 2]],
            [v, v],
            q[0],
            corresponding_reference_poses=True,
        )


def test_sample_count_does_not_increase_total_weight_and_existing_axes_take_priority():
    model, q, _ = setup_motion(1)
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    for count in (1, 4):
        task = SurfaceTasks(
            model,
            [binding()] * count,
            v,
            [[0, 1, 2]],
            [v],
            q[0],
            corresponding_reference_poses=True,
        )
        assert task.weights.sum() == pytest.approx(0.05)
        assert not list(
            task.rows(q[0], 0, np.arange(model.nv), [model.body("foot").id])
        )


def test_optional_surface_path_runs_through_same_local_acceptance_and_is_disabled_without_bindings():
    model, q, scene = directional_case()
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    task = SurfaceTasks(
        model,
        [binding()],
        v,
        [[0, 1, 2]],
        np.tile(v, (len(q), 1, 1)),
        q[0],
        corresponding_reference_poses=True,
    )
    result, report = repair_local_collisions(
        model, q, scene, {}, {}, {}, framerate=15, surface_tasks=task
    )
    assert report["fidelity"]["metrics"]["surface_normal_cost"] == 0
    assert report["invariants"]["fixed_coordinates_unchanged"]
    assert report["after"]["interframe_max_depth"] <= 0.001
    plain, disabled = repair_local_collisions(model, q, scene, {}, {}, {}, framerate=15)
    np.testing.assert_allclose(result, plain, atol=1e-12)
    assert disabled["surface_enhancement"]["status"] == "disabled"
