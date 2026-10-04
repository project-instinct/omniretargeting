"""Float64 objective parity, unknown grids, support ties and moving shells."""

import os

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting._fast.sdf import GridSDF
from tests.test_fast_backend import _objective


@pytest.fixture
def device():
    torch = pytest.importorskip('torch')
    pytest.importorskip('triton')
    if not torch.cuda.is_available():
        pytest.skip('CUDA is optional')
    return 'cuda'


def _grid():
    rng = np.random.default_rng(90)
    values = rng.normal(scale=.05, size=(17, 13, 19)).astype(np.float32)
    values[3, 4, 7] = np.nan
    return GridSDF([-1.1, -.7, -.2], .13, values, np.zeros((*values.shape, 3), np.float32))


def test_grid_unknown_masks_and_boundary_values_are_bitwise_equal(device):
    from omniretargeting._fast.gpu_grid import GPUGrid
    reference = _grid()
    rng = np.random.default_rng(500)
    points = rng.uniform([-1.4, -.9, -.4], [1.3, 1.1, 2.4], size=(7000, 3))
    nodes = np.array(np.meshgrid(*(np.arange(n) for n in reference.shape), indexing='ij')).reshape(3, -1).T
    points = np.vstack((points, reference.lower + reference.spacing * nodes,
                        [[np.nan, 0, 0], [np.inf, 0, 0], [-np.inf, 0, 0]]))
    actual = GPUGrid(reference).query(points, gradient=False)
    expected = reference.query(points, gradient=False)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))


@pytest.mark.parametrize('moving', [False, True])
def test_shell_first_minima_unknowns_and_selected_points_are_exact(device, moving):
    from omniretargeting._fast.gpu_objective import GPUFrameEvaluator
    obj, poses, _ = _objective(extra=True)
    obj.kin.evaluate(poses, False)
    reference = _grid()
    count = obj.frames
    rotation = Rotation.from_rotvec(np.random.default_rng(7).normal(scale=.1, size=(count, 3))).as_matrix() if moving else None
    translation = np.tile([.02, -.01, .03], (count, 1)) if moving else None
    scales = np.linspace(.9, 1.1, count) if moving else None
    evaluator = GPUFrameEvaluator(obj)
    evaluator.shell.prepare(obj.kin)
    for values in (reference, GridSDF(reference.lower, reference.spacing, np.zeros_like(reference.clearance), reference.directions)):
        expected = obj._deepest_samples(values, rotation, translation, scales)
        actual = evaluator.shell.deepest(evaluator.grid(values), rotation, translation, scales)
        for left, right in zip(actual, expected):
            np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize('jacobian', [False, True])
def test_all_families_and_temporal_rows_equal_reference(device, jacobian):
    from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
    obj, poses, _ = _objective(temporal=True, extra=True)
    obj.scene_sdf = _grid()
    obj.hinge_guards = (AcquisitionNormalGuard(obj, poses),)
    obj.surfaces[0]['hold'][:] = .015
    tangent = np.random.default_rng(99).normal(scale=.02, size=(obj.frames, obj.width))
    changed = obj.coords.integrate(poses, tangent, 1.)
    obj.evaluation_backend = 'legacy'
    old = obj.evaluate(changed, jacobian)
    obj.evaluation_backend = 'gpu'
    new = obj.evaluate(changed, jacobian)
    assert new[0].cost == pytest.approx(old[0].cost, rel=2e-15)
    np.testing.assert_allclose(new[0].frame_cost, old[0].frame_cost, rtol=3e-15, atol=1e-9)
    if jacobian:
        np.testing.assert_allclose(new[0].hessian, old[0].hessian, rtol=2e-14, atol=1e-9)
        np.testing.assert_allclose(new[0].gradient, old[0].gradient, rtol=2e-14, atol=1e-9)
        np.testing.assert_array_equal(new[2].toarray(), old[2].toarray())
    np.testing.assert_array_equal(new[1], old[1])
    assert new[3].keys() == old[3].keys()
    for key in old[3]:
        assert new[3][key] == pytest.approx(old[3][key], rel=2e-15, abs=1e-15)
    chunk = obj.for_frames(slice(0, 8))
    assert '_gpu_frames' not in chunk.__dict__


def test_mesh_support_ties_use_original_first_index(device):
    from omniretargeting._fast.gpu_objective import GPUFrameEvaluator
    from omniretargeting._sequence.proximity import support_points
    from tests.test_convex_mesh_distance import mesh_pair
    from types import SimpleNamespace
    model = mesh_pair()
    evaluator = GPUFrameEvaluator(SimpleNamespace(model=model))
    directions = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1e-17, 0], [.1, .2, .3]])
    actual = evaluator.support_unique(0, directions)
    np.testing.assert_array_equal(actual, support_points(model, 0, directions))
    assert evaluator.diagnostics['support_tie_fallbacks'] >= 3


def test_owning_process_guard_keeps_inherited_cuda_on_cpu(monkeypatch):
    obj, _, _ = _objective(temporal=True)
    obj.evaluation_backend = 'gpu'
    obj._gpu_pid = os.getpid() + 1
    assert obj._gpu_evaluator() is None


def test_unavailable_native_kernel_retains_reference_pool(device, monkeypatch):
    from omniretargeting._fast import batched_self_collision

    obj, poses, _ = _objective(temporal=True, extra=True)
    obj.evaluation_backend = 'legacy'
    expected = obj.evaluate(poses)

    def unavailable():
        raise batched_self_collision.NativeUnavailable('fixture unavailable compiler')

    monkeypatch.setattr(batched_self_collision, 'native_library', unavailable)
    obj.evaluation_backend = 'gpu'
    assert obj._gpu_evaluator() is None
    obj.start_workers()
    try:
        assert obj.pool is not None
        actual = obj.evaluate(poses)
    finally:
        obj.stop_workers()
    assert actual[0].cost == expected[0].cost
    np.testing.assert_array_equal(actual[0].hessian, expected[0].hessian)
    np.testing.assert_array_equal(actual[0].gradient, expected[0].gradient)
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(actual[2].toarray(), expected[2].toarray())
    assert actual[3] == expected[3]


def test_threaded_frames_are_exact_with_reweighted_hinges_and_temporal_rows(device):
    from omniretargeting._fast.objective import FastObjective
    from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
    from tests.test_fast_backend import _problem, _surface_tasks
    model, cfg, coords, poses, tasks, rotations = _problem(frames=72)
    surface, prox = _surface_tasks(model, len(poses))
    obj = FastObjective(tasks, cfg, reference=poses, temporal_reference=poses[:, coords.qi],
                        root_rotations=rotations, framerate=30., surface_tasks=[surface], proximity_tasks=[prox])
    obj.scene_sdf = _grid()
    obj.hinge_guards = (AcquisitionNormalGuard(obj, poses),)
    changed = coords.integrate(poses, np.random.default_rng(12).normal(scale=.02, size=(len(poses), obj.width)), 1.)
    obj.surfaces[0]['hold'][:] = .007
    obj.envelope_weight[::3] = 4
    obj.temporal_weight[0][::5] = 3
    for with_jacobian in (False, True):
        obj.evaluation_backend = 'legacy'
        expected = obj.evaluate(changed, with_jacobian)
        obj.evaluation_backend = 'gpu'
        actual = obj.evaluate(changed, with_jacobian)
        assert actual[0].cost == expected[0].cost
        np.testing.assert_array_equal(actual[0].frame_cost, expected[0].frame_cost)
        assert actual[3] == expected[3]
        np.testing.assert_array_equal(actual[1], expected[1])
        if with_jacobian:
            np.testing.assert_array_equal(actual[0].hessian, expected[0].hessian)
            np.testing.assert_array_equal(actual[0].gradient, expected[0].gradient)
            np.testing.assert_array_equal(actual[2].toarray(), expected[2].toarray())
