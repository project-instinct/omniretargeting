"""One fork-shared CPU budget for windows, fractions and native helpers.

Only resource availability chooses thread counts. Windows, fractions and
their numerical consumption order never depend on this diagnostic state.
"""
from contextlib import contextmanager
import multiprocessing
import os


def native_workers():
    # The four-thread tail preserves whole-clip qpos bytes; 1 disables borrowing.
    return max(1, min(4, int(os.environ.get('OMNIRETARGETING_WINDOW_NATIVE_THREADS', '4'))))


class WindowCpuBudget:
    def __init__(self, capacity):
        context = multiprocessing.get_context('fork')
        self.capacity = capacity
        self.permits = context.BoundedSemaphore(capacity)
        self.windows = context.Value('i', 0)
        self.calls = context.Array('q', 5)
        self.calls_by_kind = {key: context.Array('q', 5) for key in ('self', 'deepest')}
        self.window_owner_pid = None

    @contextmanager
    def window(self):
        # Count queued workers before acquiring: helpers never block waiting
        # windows by reserving their own quota after this count becomes busy.
        with self.windows.get_lock():
            self.windows.value += 1
        previous_owner = self.window_owner_pid
        try:
            with self.permits:
                self.window_owner_pid = os.getpid()
                yield
        finally:
            self.window_owner_pid = previous_owner
            with self.windows.get_lock():
                self.windows.value -= 1

    @contextmanager
    def borrow(self, requested, *, kind='self'):
        # A disposable fraction may be killed at any instruction. It must
        # never touch inherited shared locks, even for diagnostic counters.
        if self.window_owner_pid != os.getpid():
            yield 1
            return
        with self.windows.get_lock():
            active = self.windows.value
        ceiling = min(requested, self.capacity // max(1, active), 4) if active <= 3 else 1
        leased = 0
        try:
            for _ in range(max(0, ceiling - 1)):
                if not self.permits.acquire(block=False):
                    break
                leased += 1
            count = 1 + leased
            with self.calls.get_lock():
                self.calls[count] += 1
            counters = self.calls_by_kind[kind]
            with counters.get_lock():
                counters[count] += 1
            yield count
        finally:
            for _ in range(leased):
                self.permits.release()

    def report(self):
        with self.calls.get_lock():
            calls = list(self.calls)
        return dict(capacity=self.capacity, maximum_native_threads=max(
                    (i for i, count in enumerate(calls) if count), default=1),
                    native_calls_by_threads={str(i): calls[i] for i in range(1, 5)},
                    leases_by_kind={kind: {str(i): counters[i] for i in range(1, 5)}
                                    for kind, counters in self.calls_by_kind.items()})
