"""Consume speculative conic proposals in LM order without an eager barrier.

The existing banded barrier still determines exactly the same conic attempts
and damping numbers. Those independent conic children start at the same
point, but the parent receives only the proposal it is about to use. Unused
results and guard counter changes are discarded when the iteration ends.
"""

import multiprocessing
import os
import weakref

from .proposals import Attempts, _serve


def _close_jobs(jobs, owner):
    """Cancel unconsumed work; no numerical state or guard counters are copied."""
    if os.getpid() != owner:
        return
    running = list(jobs.values())
    jobs.clear()
    # Terminate all children before joining any: no unused child needs to
    # finish a solve while the parent joins another one.
    for process, _ in running:
        if process.is_alive():
            process.terminate()
    for process, connection in running:
        process.join()
        connection.close()


class LazyAttempts(Attempts):
    """Same proposals, exception ordering and consumed counters as Attempts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending = {}
        self._owner = os.getpid()
        # No callback reference to self: releasing an iteration cancels its
        # unused children immediately, including when the solver raises.
        self._finalizer = weakref.finalize(self, _close_jobs, self.pending, self._owner)

    def _fork(self, kind, attempts, function):
        if kind != 'conic':
            return super()._fork(kind, attempts, function)
        context = multiprocessing.get_context('fork')
        try:
            for attempt in attempts:
                parent, child = context.Pipe(duplex=False)
                process = context.Process(target=_serve, args=(function, attempt, child), daemon=True)
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                self.pending[kind, attempt] = process, parent
        except BaseException:
            self.close()
            raise

    def _take(self, kind, attempt):
        job = self.pending.pop((kind, attempt), None)
        if job is not None:
            process, connection = job
            try:
                self.cache[kind, attempt] = connection.recv()
            finally:
                connection.close()
                # A failed receive must not leave a blocked send/solve behind.
                if (kind, attempt) not in self.cache and process.is_alive():
                    process.terminate()
                process.join()
        return super()._take(kind, attempt)

    def close(self):
        _close_jobs(self.pending, self._owner)


def attempts_type(reference=Attempts):
    """Keep the eager reference scheduler available for paired validation."""
    from .runtime_options import OPTIONS
    if not getattr(OPTIONS, 'lazy_attempts', False):
        return reference
    return LazyAttempts
