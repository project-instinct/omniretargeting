"""Retry deferred collision groups only after a relevant change or new witness."""

import numpy as np


class RepairSchedule:
    def __init__(self, groups, limit, *, defer_enabled=True, total_limit=None):
        self.defer_enabled = defer_enabled
        self.order = list(groups)
        self.limit = int(limit)
        self.ready = list(groups)
        self.total_limit = self.limit * len(self.order) if total_limit is None else int(total_limit)
        self.total = 0
        self.attempts = dict.fromkeys(groups, 0)
        self.deferred = {}
        self.improving_deep = set()
        self.wakeups = set()
        self.was_wakeup = False

    def _enqueue(self, group, *, wake=False):
        if self.attempts[group] < self.limit and group not in self.ready:
            self.ready.append(group)
            if wake:
                self.wakeups.add(group)
            self.ready.sort(key=self.order.index)

    def next(self):
        if not self.ready or self.total >= self.total_limit:
            return None
        group = self.ready.pop(0)
        self.was_wakeup = group in self.wakeups
        self.wakeups.discard(group)
        self.attempts[group] += 1
        self.total += 1
        return group, self.attempts[group] - 1

    def peek_first(self, count):
        """Look ahead without consuming attempts or predicting group retries."""
        available = max(0, self.total_limit - self.total)
        return [group for group in self.ready[:available] if self.attempts[group] == 0][:count]

    def retry(self, group):
        self.deferred.pop(group, None)
        self._enqueue(group)

    def witness_retry(self, group):
        if not self.defer_enabled or group not in self.improving_deep:
            return False
        self.deferred.pop(group, None)
        self._enqueue(group, wake=True)
        return True

    def note_progress(self, group, before, after):
        # Only a real accepted decrease qualifies a deep group for wakeups.
        # A newly observed witness alone is not evidence of useful progress.
        if after <= .01:
            self.improving_deep.discard(group)
        elif before - after > 1e-6:
            self.improving_deep.add(group)

    def defer(self, group, poses, columns, reason):
        if not self.defer_enabled or group not in self.improving_deep:
            return
        columns = np.array(sorted(columns), dtype=int)
        if self.attempts[group] < self.limit:
            self.deferred[group] = (columns, poses[:, columns].copy(), reason)

    def changed(self, poses):
        for group, (columns, previous, _) in list(self.deferred.items()):
            # Suppress roundoff-only wakeups. The threshold is below every
            # geometric/continuity acceptance slack; it grants no new budget.
            if np.max(np.abs(poses[:, columns] - previous), initial=0.) > 1e-8:
                del self.deferred[group]
                self._enqueue(group, wake=True)
