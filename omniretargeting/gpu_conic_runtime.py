"""Optional cuDSS loader setup before any CUDA context is created."""
from __future__ import annotations

import multiprocessing
import os
from functools import lru_cache
from pathlib import Path
import sys
import sysconfig

_MARKER = 'OMNIRETARGETING_GPU_CONIC_REEXEC'
_OWNER_PID = os.getpid()
_ERROR = None


def requested():
    enabled = os.environ.get('OMNIRETARGETING_FAST_ENABLE', '').split(',')
    disabled = os.environ.get('OMNIRETARGETING_FAST_DISABLE', '').split(',')
    return 'gpu_conic' in enabled and 'gpu_conic' not in disabled


def main_process():
    # The PID check also covers raw os.fork and joblib's fork workers.
    return os.getpid() == _OWNER_PID and multiprocessing.current_process().name == 'MainProcess'


def prepare():
    """Make the loader search path available via at most one startup restart.

    sys.orig_argv preserves -m, wrapper scripts and interpreter switches even
    when a wrapper later replaces sys.argv. Never restart a fork/spawn worker.
    Missing dependencies quietly leave the CPU path available.
    """
    global _ERROR
    if not requested() or not main_process():
        return
    try:
        site = Path(sysconfig.get_paths()['purelib'])
        cuda = site / 'nvidia/cu13/lib'
        cudss = cuda / 'libcudss.so.0'
        if not cudss.is_file():
            _ERROR = 'missing_cudss'
            return
        cache = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache')))
        links = cache / 'omniretargeting/qoco_cuda/lib'
        links.mkdir(parents=True, exist_ok=True)
        for name in ('libcudss.so', 'libcudss.so.1'):
            link = links / name
            if link.is_symlink() and link.resolve() == cudss.resolve():
                continue
            # Atomic replacement permits concurrent startup without unlink races.
            temporary = links / f'{name}.{os.getpid()}.tmp'
            temporary.symlink_to(cudss)
            temporary.replace(link)
        required = [str(links), str(cuda)]
        toolkit = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda')) / 'targets/x86_64-linux/lib'
        if toolkit.is_dir():
            required.append(str(toolkit))
        current = os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep)
        if all(directory in current for directory in required):
            return
        if os.environ.get(_MARKER):
            _ERROR = 'loader_restart_exhausted'
            return
        argv = getattr(sys, 'orig_argv', None)
        if not argv or '-c' in argv or '-' in argv:
            _ERROR = 'non_restartable_interpreter'
            return
        env = dict(os.environ, **{_MARKER: '1', 'LD_LIBRARY_PATH': os.pathsep.join(
            required + [directory for directory in current if directory])})
        os.execve(sys.executable, [sys.executable, *argv[1:]], env)
    except OSError as error:
        _ERROR = f'loader_{type(error).__name__}'


def error():
    return _ERROR


@lru_cache(maxsize=1)
def device_error():
    """Check the driver without creating a context or entering native QOCO.

    A CUDA wheel can import even on a host with no usable device. Checking
    before QOCO setup also handles explicit GPU masking quietly.
    """
    if os.environ.get('CUDA_VISIBLE_DEVICES') in ('', '-1'):
        return 'cuda_devices_masked'
    import ctypes
    try:
        driver = ctypes.CDLL('libcuda.so.1')
        code = driver.cuInit(0)
        if code:
            return f'cuda_driver_error_{code}'
        count = ctypes.c_int()
        code = driver.cuDeviceGetCount(ctypes.byref(count))
        if code or not count.value:
            return f'cuda_no_devices_{code}'
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')[0]
        if visible.isdigit() and int(visible) >= count.value:
            return 'cuda_invalid_visible_device'
    except (OSError, AttributeError) as error:
        return f'cuda_driver_{type(error).__name__}'
    return None
