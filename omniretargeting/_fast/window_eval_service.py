"""Default-off GPU frame-row service for forked acquisition windows.

The service owns CUDA in a fresh executable. Workers retain CPU temporal rows
and all nonlinear guards. Device normal equations change numerical paths.
"""
import copy
import multiprocessing
import os
from pathlib import Path
import pickle
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

WINDOW_GPU_EVALUATION = False


def evaluation_payload(objective):
    """Detach execution resources; retain immutable per-frame row definitions."""
    from .acquisition_guard import AcquisitionNormalGuard
    result = objective.for_frames(slice(0, objective.frames))
    for name in list(result.__dict__):
        if name.startswith(('_gpu', '_native', '_window', '_temporal')) or name == '_self_batch':
            result.__dict__.pop(name, None)
    if getattr(result, 'original_clearance', None) is not None:
        raise ValueError('Window GPU service does not adapt original clearance')
    result.pool, result.worker_output = None, None
    result.source_shape, result.candidate_guards = None, []
    result.hinge_guards = []
    for guard in getattr(objective, 'hinge_guards', ()):
        if type(guard) is not AcquisitionNormalGuard:
            raise ValueError(f'Unsupported GPU service hinge guard: {type(guard).__name__}')
        private = copy.copy(guard)
        private.objective = SimpleNamespace(model=objective.model)
        for name in list(private.__dict__):
            if name == 'kin' or name.startswith(('_native', '_window')):
                private.__dict__.pop(name)
        result.hinge_guards.append(private)
    result.surfaces = [dict(surface, task=SimpleNamespace(cfg=surface['task'].cfg))
                       for surface in result.surfaces]
    result.evaluation_backend = 'cpu'
    return result


class WindowEvalClient:
    def __init__(self, address, token):
        self.address, self.token = address, token
        self.connection, self.pid = None, None

    def evaluate(self, poses, frames, with_jacobian, envelope_weight):
        from multiprocessing.connection import Client
        if self.pid != os.getpid():
            self.close()
            self.connection = Client(self.address, family='AF_UNIX', authkey=self.token)
            self.pid = os.getpid()
        started = time.monotonic()
        self.connection.send((frames.start, frames.stop, poses, bool(with_jacobian), envelope_weight))
        status, result = self.connection.recv()
        if status != 'ok':
            raise RuntimeError(f'Window GPU evaluation failed: {result}')
        result.pop('service_diagnostics', None)
        self.last_seconds = time.monotonic() - started
        return result

    def close(self):
        if self.connection is not None:
            self.connection.close()
        self.connection, self.pid = None, None


class WindowEvalService:
    def __init__(self, objective):
        self.directory = tempfile.mkdtemp(prefix='omniretargeting_window_eval_')
        self.address = str(Path(self.directory) / 'service.sock')
        self.token = secrets.token_bytes(32)
        self.control, child = multiprocessing.Pipe()
        self.process, self.diagnostics = None, {}
        try:
            payload = Path(self.directory) / 'objective.pkl'
            with payload.open('wb') as handle:
                pickle.dump(evaluation_payload(objective), handle, protocol=5)
            self.process = subprocess.Popen(
                [sys.executable, '-m', 'omniretargeting._fast.window_eval_worker',
                 str(child.fileno()), self.address, self.token.hex(), str(payload)],
                pass_fds=(child.fileno(),))
            child.close()
            if not self.control.poll(120):
                raise RuntimeError('Window GPU service startup timed out')
            status, result = self.control.recv()
            if status != 'ready':
                raise RuntimeError(f'Cannot start GPU evaluation service: {result}')
            payload.unlink()
        except BaseException:
            child.close()
            self.close()
            raise

    def client(self):
        return WindowEvalClient(self.address, self.token)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            try:
                self.control.send(None)
                if self.control.poll(20):
                    self.diagnostics = self.control.recv()
                self.process.wait(timeout=20)
            except (BrokenPipeError, EOFError, OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=20)
        self.control.close()
        shutil.rmtree(self.directory, ignore_errors=True)
