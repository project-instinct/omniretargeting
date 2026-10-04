"""GPU service byte repeatability across batching order and child processes."""
import multiprocessing

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from omniretargeting._fast.window_eval_service import evaluation_payload, WindowEvalService
from tests.test_fast_backend import _problem, _surface_tasks
from tests.test_gpu_objective import _grid


def fixture():
    from omniretargeting._fast.objective import FastObjective
    from omniretargeting._fast.acquisition_guard import AcquisitionNormalGuard
    model, cfg, coords, poses, tasks, rotations = _problem(frames=96)
    surface, prox = _surface_tasks(model, len(poses))
    obj = FastObjective(tasks, cfg, reference=poses, temporal_reference=poses[:, coords.qi],
                        root_rotations=rotations, framerate=30., surface_tasks=[surface],
                        proximity_tasks=[prox])
    obj.scene_sdf = _grid()
    obj.objects = [dict(sdf=_grid(), rotations=Rotation.from_rotvec(
        np.random.default_rng(54).normal(scale=.02, size=(len(poses), 3))).as_matrix(),
        translations=np.tile([.02, .01, .03], (len(poses), 1)),
        scales=np.linspace(.95, 1.05, len(poses)))]
    obj.objects.append(dict(obj.objects[0], translations=np.tile([-.04, .03, -.02], (len(poses), 1))))
    obj.hinge_guards = (AcquisitionNormalGuard(obj, poses),)
    poses = coords.integrate(poses, np.random.default_rng(55).normal(
        scale=.002, size=(len(poses), obj.width)), 1.)
    return obj, poses


def same(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], np.ndarray):
            assert a[key].shape == b[key].shape and a[key].dtype == b[key].dtype
            assert a[key].tobytes() == b[key].tobytes(), key
        else:
            assert a[key] == b[key], key


def cuda():
    torch = pytest.importorskip('torch')
    pytest.importorskip('triton')
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')


def test_gpu_requests_are_bitwise_independent_of_batch_membership_and_order():
    cuda()
    from omniretargeting._fast.window_eval_batch import WindowEvaluationBatch
    obj, poses = fixture()
    engine = WindowEvaluationBatch(evaluation_payload(obj))
    requests = [(3, 75, poses[3:75], True, obj.envelope_weight[3:75]),
                (12, 92, poses[12:92], False, obj.envelope_weight[12:92])]
    try:
        alone = [engine.run([(i, r)])[0] for i, r in enumerate(requests)]
        together = engine.run(list(enumerate(requests)))
        reverse = engine.run(list(reversed(list(enumerate(requests)))))
        for i in range(2):
            same(alone[i], together[i])
            same(alone[i], reverse[1 - i])
        from omniretargeting._fast.gpu_objective import GPUFrameEvaluator
        direct = GPUFrameEvaluator(evaluation_payload(obj).for_frames(slice(3, 75)))
        direct.device_moments = True
        reference = direct.evaluate(requests[0][2], True, requests[0][4])
        for key in ('root_point', 'root_jacobian', 'effector_point', 'effector_jacobian'):
            reference.pop(key)
        same(alone[0], reference)
    finally:
        engine.close()


def _child(writer, client, poses, frames, weight):
    try:
        a = client.evaluate(poses, frames, True, weight)
        b = client.evaluate(poses, frames, True, weight)
        same(a, b)
        writer.send(('ok', a))
    except BaseException as error:
        writer.send(('error', repr(error)))
    finally:
        client.close()
        writer.close()


def test_private_fork_channels_and_global_temporal_rows_survive_service():
    cuda()
    obj, poses = fixture()
    obj.evaluation_backend = 'cpu'
    obj.prepare_window_evaluation()
    frames = range(3, 75)
    cpu = obj.evaluate(poses, True, frames=frames)
    service = WindowEvalService(obj)
    jobs = []
    client = service.client()
    try:
        obj._window_eval_client = client
        first = obj.evaluate(poses, True, frames=frames)
        second = obj.evaluate(poses, True, frames=frames)
        assert first[0].frame_cost.tobytes() == second[0].frame_cost.tobytes()
        assert first[0].hessian.tobytes() == second[0].hessian.tobytes()
        assert first[1].tobytes() == cpu[1].tobytes()
        assert first[2].data.tobytes() == cpu[2].data.tobytes()
        assert first[2].indices.tobytes() == cpu[2].indices.tobytes()
        assert first[2].indptr.tobytes() == cpu[2].indptr.tobytes()
        for span in (range(3, 75), range(12, 92)):
            reader, writer = multiprocessing.get_context('fork').Pipe()
            child = multiprocessing.get_context('fork').Process(target=_child, args=(
                writer, client, poses[span.start:span.stop], span,
                obj.envelope_weight[span.start:span.stop]))
            child.start()
            writer.close()
            jobs.append((child, reader))
        for child, reader in jobs:
            assert reader.poll(60)
            status, result = reader.recv()
            assert status == 'ok', result
            child.join(timeout=10)
            assert child.exitcode == 0
    finally:
        for child, reader in jobs:
            if child.is_alive():
                child.terminate()
                child.join()
            reader.close()
        obj.__dict__.pop('_window_eval_client', None)
        client.close()
        service.close()
    assert service.process.poll() == 0
    assert service.diagnostics['requests'] == 6


def test_slow_window_does_not_delay_another_response(monkeypatch):
    import threading
    cuda()
    from omniretargeting._fast.window_eval_batch import WindowEvaluationBatch
    obj, poses = fixture()
    engine = WindowEvaluationBatch(evaluation_payload(obj))
    released = threading.Event()
    original = engine.evaluate
    def delayed(item):
        if len(item[1][2]) == 72:
            assert released.wait(30)
        return original(item)
    monkeypatch.setattr(engine, 'evaluate', delayed)
    try:
        futures = engine.submit([(0, (3, 75, poses[3:75], True, obj.envelope_weight[3:75])),
                                 (1, (12, 92, poses[12:92], True, obj.envelope_weight[12:92]))])
        assert futures[1].result(timeout=30)['hessian'].shape[0] == 80
        assert not futures[0].done()
        released.set()
        assert futures[0].result(timeout=30)['hessian'].shape[0] == 72
    finally:
        released.set()
        engine.close()
