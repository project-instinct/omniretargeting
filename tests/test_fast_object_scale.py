"""Moving SDFs must match recorded metric geometry, including frame scale."""
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from omniretargeting._fast import sdf as sdf_module
from omniretargeting._fast.objective import _Accumulator
from omniretargeting._fast.pipeline import _object_grids
from tests.test_fast_backend import _objective


class PlaneField:
    spacing = 0.01

    def query(self, points, gradient=False):
        return np.asarray(points)[..., 2] - 1.5

    def gradient(self, points):
        normal = np.zeros_like(points)
        normal[..., 2] = 1.0
        return normal


def scaled_objective():
    objective, poses, _ = _objective()
    frames = len(poses)
    rotations = Rotation.from_euler('y', np.linspace(0.15, 0.35, frames)[:, None]).as_matrix()
    scales = np.linspace(0.4, 1.4, frames)
    translation = np.tile([0.2, -0.1, 0.3], (frames, 1))
    objective.objects = [{'sdf': PlaneField(), 'rotations': rotations,
                              'translations': translation, 'scales': scales}]
    return objective, poses


def test_scaled_fields_match_world_plane_and_all_shell_samples():
    objective, poses = scaled_objective()
    objective.kin.evaluate(poses, with_jacobian=True)
    _, field, rotation, translation, scale = objective._clearance_fields()[0]
    frames = np.repeat(np.arange(len(poses)), len(objective.shell_geoms))
    geoms = np.tile(np.arange(len(objective.shell_geoms)), len(poses))
    points, valid = objective._shell_points(frames, geoms)
    normal = rotation[frames, :, 2]
    expected = np.einsum('nsi,ni->ns', points - translation[frames, None], normal)
    expected -= 1.5 * scale[frames, None]
    np.testing.assert_allclose(objective._inside(0, frames, points), expected, atol=1e-14)
    full_min = np.where(valid, expected, np.inf).min(1).reshape(len(poses), -1)
    measured = objective._deepest_samples(field, rotation, translation, scale)[0]
    active = full_min < objective.cfg['scene_clearance_m']
    np.testing.assert_allclose(measured[active], full_min[active], atol=1e-14)
    # Verify world-distance gradients: uniform scale cancels from the chain rule.
    acc = _Accumulator(len(poses), objective.width, True)
    objective._clearance_terms(acc, {})
    for col in range(3):
        direction = np.zeros((len(poses), objective.width))
        direction[:, col] = 1.0
        costs = []
        for sign in (-1, 1):
            q = objective.coords.integrate(poses, direction, sign * 1e-6)
            objective.kin.evaluate(q, with_jacobian=False)
            value = _Accumulator(len(poses), objective.width, False)
            objective._clearance_terms(value, {})
            costs.append(value.cost)
        derivative = (costs[1] - costs[0]) / 2e-6
        assert derivative == pytest.approx(2 * acc.gradient[:, col].sum(), rel=1e-6, abs=1e-5)


def test_scale_timeline_survives_parallel_frame_slicing():
    objective, poses = scaled_objective()
    objective.kin.evaluate(poses, with_jacobian=True)
    whole = _Accumulator(len(poses), objective.width, True)
    objective._clearance_terms(whole, {})
    gradients = []
    hessians = []
    costs = []
    for start, stop in ((0, 13), (13, len(poses))):
        chunk = objective.for_frames(slice(start, stop))
        chunk.kin.evaluate(poses[start:stop], with_jacobian=True)
        acc = _Accumulator(stop-start, objective.width, True)
        chunk._clearance_terms(acc, {})
        gradients.append(acc.gradient)
        hessians.append(acc.hessian)
        costs.append(acc.cost)
    np.testing.assert_allclose(np.concatenate(gradients), whole.gradient, atol=1e-12)
    np.testing.assert_allclose(np.concatenate(hessians), whole.hessian, atol=1e-12)
    assert sum(costs) == pytest.approx(whole.cost)


def test_object_grid_is_metric_and_does_not_mutate_authored_mesh(tmp_path, monkeypatch):
    monkeypatch.setattr(sdf_module, 'CACHE_DIR', tmp_path)
    mesh = trimesh.creation.box(extents=[20, 16, 12])
    original = mesh.vertices.copy()
    raw_scale = np.array([0.025, 0.05])
    rotations = np.tile(np.eye(3), (2, 1, 1))
    owner = SimpleNamespace(_scene_objects=[{'mesh': mesh, 'scales': raw_scale,
                            'rotations': rotations, 'translations': np.zeros((2, 3))}])
    field = _object_grids(owner)[0]
    assert np.prod(field['sdf'].shape) < 2_000_000
    np.testing.assert_array_equal(mesh.vertices, original)
    np.testing.assert_array_equal(raw_scale, [0.025, 0.05])
    point = np.array([[0.0, 0.0, 0.1]])
    for frame in range(2):
        scale = field['scales'][frame]
        distance = field['sdf'].query(point / scale, gradient=False) * scale
        # Closest surface is the box top at z = 6 * recorded scale.
        np.testing.assert_allclose(distance, 0.1 - 6 * raw_scale[frame], atol=1e-7)


@pytest.mark.parametrize('scales', ([0.5], [0.5, 0.0], [np.nan, 1.0], [-0.1, 1.0]))
def test_invalid_object_scale_is_not_silently_used(scales):
    owner = SimpleNamespace(_scene_objects=[{'mesh': None, 'scales': scales,
                                                'translations': np.zeros((2, 3))}])
    with pytest.raises(ValueError, match='scales'):
        _object_grids(owner)


def test_other_surface_task_interfaces_are_not_misread_as_contacts():
    from omniretargeting._fast.pipeline import _surface_tasks
    from omniretargeting.self_contact import SelfContactTasks
    from omniretargeting.surface_pose import SurfacePoseTasks
    objective, _, (_, surface, _) = _objective(extra=True)
    others = [object.__new__(SelfContactTasks), object.__new__(SurfacePoseTasks)]
    for task in others:
        task.kinematics = objective.kin
        task.targets = np.zeros((1, 1, 3))
    assert _surface_tasks([None, *others, surface]) == [surface]
