"""Bounded, ordered mechanics maps shared by the two support passes."""

import multiprocessing
import os

from .cpu_budget import available_cpus


class SupportPool:
    """Lazily fork at most eight workers; preserve serial map ordering.

    One preparation owns this context. Small jobs and platforms without fork
    use the same functions serially. Failed jobs propagate their exceptions;
    the context terminates the pool rather than leaving workers behind.
    """

    def __init__(self):
        self.workers = min(8, available_cpus())
        self.pool = None

    def __enter__(self):
        return self

    def map(self, function, items):
        items = list(items)
        if not items:
            return []
        if self.workers <= 1 or not hasattr(os, "fork") or len(items) < 32:
            return list(map(function, items))
        if self.pool is None:
            self.pool = multiprocessing.get_context("fork").Pool(self.workers)
        return self.pool.map(function, items, chunksize=max(1, len(items) // (4 * self.workers)))

    def __exit__(self, exc_type, exc_value, traceback):
        if self.pool is not None:
            if exc_type is None:
                self.pool.close()
            else:
                self.pool.terminate()
            self.pool.join()
            self.pool = None
