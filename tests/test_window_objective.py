"""A window is the exact column/row restriction of the original objective."""

import numpy as np
import pytest

from omniretargeting._fast.frame_kinematics import FrameKinematicsCache
from omniretargeting._fast.window_objective import temporal_row_indices
from tests.test_fast_backend import _objective


@pytest.mark.parametrize('frames', [range(0, 1), range(0, 32), range(4, 36), range(12, 20), range(35, 36)])
@pytest.mark.parametrize('jacobian', [False, True])
def test_window_rows_equal_full_restriction(frames, jacobian):
    obj, poses, _ = _objective(temporal=True, extra=True)
    rng = np.random.default_rng(141)
    moved = obj.coords.integrate(poses, rng.normal(scale=.012, size=(len(poses), obj.width)), 1.)
    # Quaternion signs, inactive temporal rows and nonuniform/replaced weights.
    moved[::3, 3:7] *= -1
    for i in range(3):
        obj.temporal_weight[i] = rng.uniform(.3, 2., len(obj.temporal_weight[i]))
        obj.joint_temporal_weight[i][::3] = 0
        obj.root_translation_weight[i] *= .7
        obj.root_rotation_weight[i] *= 1.2
    full = obj.evaluate(moved, jacobian)
    local = obj.evaluate(moved, jacobian, frames=frames)
    rows = temporal_row_indices(obj, frames)
    span = slice(frames.start, frames.stop)
    np.testing.assert_array_equal(local[1], full[1][rows])
    np.testing.assert_allclose(local[0].frame_cost, full[0].frame_cost[span], rtol=1e-12, atol=1e-12)
    assert local[0].cost == pytest.approx(full[0].frame_cost[span].sum() + full[1][rows] @ full[1][rows], rel=1e-12)
    if jacobian:
        np.testing.assert_array_equal(local[2].toarray(), full[2][rows, frames.start * obj.width:frames.stop * obj.width].toarray())
        np.testing.assert_allclose(local[0].gradient, full[0].gradient[span], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(local[0].hessian, full[0].hessian[span], rtol=1e-12, atol=1e-12)


def test_fixed_neighbour_changes_affect_boundary_rows_and_fk_cache():
    obj, poses, _ = _objective(temporal=True, extra=True)
    frames = range(10, 20)
    old = obj.evaluate(poses, frames=frames)
    cache = obj._window_frame_cache
    assert cache.computed_frames <= len(frames) + 6
    obj.evaluate(poses, frames=frames)
    assert cache.computed_frames <= len(frames) + 6
    modified = poses.copy()
    modified[9, 7] += .01
    modified[9, 2] += .001
    before = cache.computed_frames
    new = obj.evaluate(modified, frames=frames)
    assert cache.computed_frames == before + 1
    np.testing.assert_array_equal(old[0].frame_cost, new[0].frame_cost)
    assert np.max(np.abs(old[1] - new[1])) > 0
    full = obj.evaluate(modified)
    np.testing.assert_array_equal(new[1], full[1][temporal_row_indices(obj, frames)])
    # The retained old arrays cannot change during a subsequent evaluation.
    assert not np.shares_memory(old[0].gradient, new[0].gradient)


def test_per_frame_cache_upgrades_jacobians_without_recomputing_fixed_frames():
    obj, poses, _ = _objective()
    cache = FrameKinematicsCache(obj.model, obj.coords.cols)
    first = cache.evaluate(poses, range(0, 8), False)
    cache.evaluate(poses, range(0, 8), True)
    assert cache.computed_frames == 16
    retained = first.xpos.copy()
    cache.evaluate(poses, range(4, 12), True)
    assert cache.computed_frames == 20
    np.testing.assert_array_equal(first.xpos, retained)
    assert not first.xpos.flags.writeable


def test_window_weights_are_refreshed_after_in_place_and_replacement_edits():
    obj, poses, _ = _objective(temporal=True, extra=True)
    frames = range(4, 16)
    obj.evaluate(poses, frames=frames)
    obj.envelope_weight = np.linspace(1., 2., obj.frames)
    obj.temporal_weight[1] *= 1.7
    obj.temporal_weight[2] = obj.temporal_weight[2] * .6
    obj.correction_temporal_scale = 2.3
    obj.continuity_joints = poses[:, obj.coords.qi] * .99
    full, window = obj.evaluate(poses), obj.evaluate(poses, frames=frames)
    np.testing.assert_array_equal(window[1], full[1][temporal_row_indices(obj, frames)])
    np.testing.assert_array_equal(window[0].frame_cost, full[0].frame_cost[4:16])


@pytest.mark.parametrize('frames', [range(0, 36, 2), range(-1, 10), range(4, 4), slice(0, 8)])
def test_invalid_windows_are_rejected(frames):
    obj, poses, _ = _objective()
    with pytest.raises(ValueError):
        obj.evaluate(poses, frames=frames)


def test_forked_window_creates_no_threads_pools_or_cuda_and_is_deterministic():
    import json
    import os
    import select
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    from omniretargeting.self_collision import SelfCollisionQuery
    obj, poses, _ = _objective(temporal=True, extra=True)
    obj.self_collision = SelfCollisionQuery(obj.model)
    obj.evaluation_backend = 'cpu'
    obj.prepare_window_evaluation()
    frames = range(4, 36)
    changed = obj.coords.integrate(poses, np.random.default_rng(48).normal(scale=.002, size=(len(poses), obj.width)), 1.)
    expected = obj.evaluate(changed, frames=frames)
    reader, writer = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(reader)
        try:
            def forbidden(*args, **kwargs):
                raise AssertionError('Window child tried to create a pool/thread/CUDA evaluator')
            threading.Thread.start = forbidden
            ThreadPoolExecutor.__init__ = forbidden
            obj._gpu_evaluator = forbidden
            before = len(list(Path('/proc/self/task').iterdir()))
            # Distinct candidates force collision and FK cache misses.
            another = changed.copy()
            another[frames.start:frames.stop, 7] += 1e-5
            obj.evaluate(another, frames=frames)
            actual = obj.evaluate(changed, frames=frames)
            for name in ('frame_cost', 'gradient', 'hessian'):
                np.testing.assert_array_equal(getattr(actual[0], name), getattr(expected[0], name))
            np.testing.assert_array_equal(actual[1], expected[1])
            np.testing.assert_array_equal(actual[2].toarray(), expected[2].toarray())
            np.testing.assert_array_equal(actual[3], expected[3])
            after = len(list(Path('/proc/self/task').iterdir()))
            assert before == after == 1
            os.write(writer, json.dumps({'threads': after, 'equal': True}).encode())
            os._exit(0)
        except BaseException as error:
            os.write(writer, repr(error).encode())
            os._exit(1)
    os.close(writer)
    ready, _, _ = select.select([reader], [], [], 20)
    if not ready:
        import signal
        os.kill(pid, signal.SIGKILL)
    output = os.read(reader, 4096) if ready else b'timed out'
    os.close(reader)
    _, status = os.waitpid(pid, 0)
    assert status == 0, output.decode()
    assert json.loads(output)['threads'] == 1
