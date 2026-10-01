"""Speculative restoration of conic fractions, consumed in serial order.

Only restoration runs in children: it reads immutable bounds and local FK,
never the objective's inherited frame pool. The parent retains all original
cost/guard checks and the same limit on consumed restorations. Unused results,
exceptions and rejection counts are discarded when the serial loop returns.
"""
from .speculation import GuardedCall, can_fork


class FractionRestorations:
    """At most ``limit`` independent children; consume only requested fractions."""

    def __init__(self, restore, objective, origin, step, alpha, fractions, limit, guards, *, enabled):
        self.restore_function, self.objective = restore, objective
        self.origin, self.step, self.alpha = origin, step, alpha
        self.fractions, self.limit, self.guards = fractions, limit, guards
        self.enabled = enabled and can_fork()
        self.jobs = {}
        self.stats = None
        if self.enabled:
            self.stats = getattr(objective, 'parallel_shortening_stats', None)
            if self.stats is None:
                self.stats = dict(launched=0, consumed=0, discarded=0, serial=0, wait_seconds=0.)
                objective.parallel_shortening_stats = self.stats

    def __enter__(self):
        if not self.enabled:
            return self
        try:
            for fraction in self.fractions[:self.limit]:
                self.jobs[fraction] = GuardedCall(self.restore_function, (
                    self.objective, self.origin, self.step * (self.alpha * fraction)), self.guards)
                self.stats['launched'] += 1
        except BaseException:
            self.close()
            raise
        return self

    def restore(self, fraction):
        job = self.jobs.pop(fraction, None)
        if job is None:
            if self.stats is not None:
                self.stats['serial'] += 1
            return self.restore_function(self.objective, self.origin,
                                         self.step * (self.alpha * fraction))
        self.stats['consumed'] += 1
        try:
            return job.receive()
        finally:
            self.stats['wait_seconds'] += job.wait_seconds

    def close(self):
        for job in self.jobs.values():
            job.close()
            self.stats['discarded'] += 1
        self.jobs.clear()

    def __exit__(self, *_):
        self.close()
