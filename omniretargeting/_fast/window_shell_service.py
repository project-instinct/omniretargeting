"""Optional single CUDA owner for exact shell queries from CPU window children."""

import multiprocessing
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

# Exactness/performance experiment: leave disabled until full-clip validation.
WINDOW_GPU_SHELL = False


class WindowShellClient:
    """One private synchronous channel; inherited clients reconnect after fork."""

    def __init__(self, address, token, fields):
        self.address, self.token, self.fields = address, token, fields
        self.connection, self.pid = None, None

    def deepest(self, objective, sdf, rotation, translation, scales):
        from multiprocessing.connection import Client
        if self.pid != os.getpid():
            if self.connection is not None:
                self.connection.close()
            self.connection = Client(self.address, family='AF_UNIX', authkey=self.token)
            self.pid = os.getpid()
        self.connection.send((self.fields[id(sdf)], objective.kin.geom_xpos,
                              objective.kin.geom_xmat, rotation, translation, scales))
        status, result = self.connection.recv()
        if status != 'ok':
            raise RuntimeError(f'Window shell service failed: {result}')
        return result

    def close(self):
        if self.connection is not None:
            self.connection.close()
        self.connection, self.pid = None, None


class WindowShellService:
    """Exec a new process so its CUDA state is never inherited from the parent."""

    def __init__(self, objective):
        from .sdf import GridSDF
        fields = [sdf for _, sdf, *_ in objective._clearance_fields() if isinstance(sdf, GridSDF)]
        unique = {id(sdf): sdf for sdf in fields}
        self.fields = {key: index for index, key in enumerate(unique)}
        self.directory = tempfile.mkdtemp(prefix='omniretargeting_window_shell_')
        self.address = str(Path(self.directory) / 'service.sock')
        self.token = secrets.token_bytes(32)
        self.process = None
        self.control, child = multiprocessing.Pipe()
        payload = SimpleNamespace(
            model=SimpleNamespace(ngeom=objective.model.ngeom), cfg=objective.cfg,
            shell_geoms=objective.shell_geoms, shell_index=objective.shell_index,
            shell_local=objective.shell_local, shell_radius=objective.shell_radius)
        try:
            self.process = subprocess.Popen(
                [sys.executable, '-m', 'omniretargeting._fast.window_shell_worker',
                 str(child.fileno()), self.address, self.token.hex()], pass_fds=(child.fileno(),))
            child.close()
            self.control.send((payload, list(unique.values())))
            status, result = self.control.recv()
            if status != 'ready':
                raise RuntimeError(f'Cannot start window shell service: {result}')
        except BaseException:
            child.close()
            self.close()
            raise

    def client(self):
        return WindowShellClient(self.address, self.token, self.fields)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            try:
                self.control.send(None)
                self.process.wait(timeout=20)
            except (BrokenPipeError, EOFError, OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=20)
        self.control.close()
        shutil.rmtree(self.directory, ignore_errors=True)
