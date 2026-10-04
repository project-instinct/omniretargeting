"""Use spare window-worker capacity for serially consumed restorations.

Only independent fractions are prefetched. The serial loop still chooses
which result to consume and retains every cost/guard test and restoration
limit. Discarded results, errors and guard counters never affect the caller.
"""
from .shortening import FractionRestorations
from .speculation import GuardedCall

# QOCO + budget64 paired 957 control: acquisition 145.538 -> 143.991 s;
# final/reference/repair-entry bytes and every per-item metric are identical.
# The 26 s static-repair improvement in these runs belongs to its native query.
WINDOW_PARALLEL_RESTORATIONS = True


class LeasedCall:
    def __init__(self, call, permits):
        self.call, self.permits, self.released = call, permits, False

    def __getattr__(self, name):
        return getattr(self.call, name)

    def release(self):
        if not self.released:
            self.released = True
            self.permits.release()

    def receive(self):
        try:
            return self.call.receive()
        finally:
            self.release()

    def close(self):
        try:
            self.call.close()
        finally:
            self.release()


class WindowFractionRestorations(FractionRestorations):
    def __init__(self, *args, enabled):
        super().__init__(*args, enabled=True)
        self.permits = self.objective._window_restoration_permits

    def __enter__(self):
        if not self.enabled:
            return self
        try:
            for fraction in self.fractions[:self.limit]:
                # Never wait for extra capacity: the unchanged serial path
                # remains available when all window workers are occupied.
                if not self.permits.acquire(block=False):
                    self.stats['capacity_unavailable'] = self.stats.get('capacity_unavailable', 0) + 1
                    break
                try:
                    call = GuardedCall(self.restore_function, (
                        self.objective, self.origin, self.step * (self.alpha * fraction)), self.guards)
                except BaseException:
                    self.permits.release()
                    raise
                self.jobs[fraction] = LeasedCall(call, self.permits)
                self.stats['launched'] += 1
        except BaseException:
            self.close()
            raise
        return self


def fraction_class(objective, original):
    if (not WINDOW_PARALLEL_RESTORATIONS
            or getattr(objective, '_window_restoration_permits', None) is None):
        return original
    from .window_driver import fork_status
    return WindowFractionRestorations if fork_status()[0] else original
