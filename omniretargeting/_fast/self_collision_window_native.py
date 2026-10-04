"""Parent-prepared, thread-free screening for exact self-pair window queries."""

import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import subprocess

from .batched_self_collision import NativeUnavailable


@lru_cache(maxsize=1)
def screen_library():
    source = Path(__file__).with_suffix('.cpp')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:24]
    cache = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache'))
    cache = cache / 'omniretargeting' / 'native'
    target = cache / f'self_window_{digest}.so'
    try:
        cache.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_suffix(f'.{os.getpid()}.so')
            try:
                subprocess.run(['g++', '-std=c++17', '-O3', '-ffp-contract=off',
                                '-shared', '-fPIC', str(source), '-o', str(temporary)],
                               check=True, capture_output=True, timeout=60)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        library = ctypes.CDLL(str(target))
        library.ws3_window_screen.argtypes = ([ctypes.c_int64] * 3
            + [ctypes.c_void_p] * 4 + [ctypes.c_double] + [ctypes.c_void_p] * 11)
        library.ws3_window_screen.restype = None
        return library
    except (OSError, AttributeError, subprocess.SubprocessError) as error:
        raise NativeUnavailable(f'Self-window screening unavailable: {error}') from error
