"""CPU window children consume exact shell results from a separate CUDA owner."""

import multiprocessing

import numpy as np
import pytest

from omniretargeting._fast.window_shell_service import WindowShellService
from omniretargeting._fast.window_objective import _views
from tests.test_fast_legacy_window import problem
from tests.test_gpu_objective import _grid


def _same(actual, expected):
    for a, b in zip(actual, expected):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert a.tobytes() == b.tobytes()


def _child(connection, objective, poses, frames):
    try:
        obj = objective
        obj.evaluate(poses, frames=frames)
        local, _, _ = _views(obj, frames)
        results = [local._deepest_samples(*field[1:]) for field in local._clearance_fields()]
        connection.send(('ok', results))
    except BaseException as error:
        connection.send(('error', repr(error)))
    finally:
        connection.close()


def test_parallel_cpu_children_have_exact_private_service_channels():
    torch = pytest.importorskip('torch')
    pytest.importorskip('triton')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    obj, poses = problem(extra=True)
    obj.scene_sdf = _grid()
    obj.prepare_window_evaluation()
    windows = [range(0, 24), range(12, 36)]
    expected = []
    for frames in windows:
        obj.evaluate(poses, frames=frames)
        local, _, _ = _views(obj, frames)
        expected.append([local._deepest_samples(*field[1:]) for field in local._clearance_fields()])
    service = WindowShellService(obj)
    jobs = []
    try:
        obj._window_shell_client = service.client()
        context = multiprocessing.get_context('fork')
        for frames in windows:
            reader, writer = context.Pipe()
            child = context.Process(target=_child, args=(writer, obj, poses, frames))
            child.start()
            writer.close()
            jobs.append((child, reader))
        for (child, reader), cpu in zip(jobs, expected):
            assert reader.poll(30)
            status, actual = reader.recv()
            assert status == 'ok', actual
            for a, b in zip(actual, cpu):
                _same(a, b)
            child.join(timeout=10)
            assert child.exitcode == 0
    finally:
        for child, reader in jobs:
            if child.is_alive():
                child.terminate()
                child.join()
            reader.close()
        obj.__dict__.pop('_window_shell_client').close()
        service.close()
    assert service.process.poll() == 0
